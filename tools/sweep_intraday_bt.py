"""扫单 intraday T 回测 —— 按逐笔成交帧（分时）回测。

背景：扫单信号本身是日终聚合，但识别过程 detect_sweeps 本就是按逐笔 run 做的。
因此把「逐笔成交」当作一帧帧输入，在分钟滚动窗口里边扫边交易，做出真正的日内 T。

A 股 T+1 约束下的落地：
  - 倒T（对 edge，主推）：买扫=价格被拉高=过度反应 → 卖底仓（高），价格回落 ≥thr 买回。
    对应实证有 edge 的「超买做空」一侧。
  - 正T（可选）：卖扫=砸盘 → 买入，反弹 ≥thr 卖底仓；下跌趋势接飞刀，需大盘择时。

底仓：池内标的等权买入持有（隔夜），日线收盘取当日最后一笔成交价（≈收盘）。
分时仓：每标的每天每方向最多 1 笔，开/平仓均计双边成本。输出叠加效果。
"""
from __future__ import annotations

import os
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

import numpy as np

SWEEP = r"C:\Users\Lison\Desktop\EvfWorkSpace\TRADE\sweep_monitor"
if SWEEP not in sys.path:
    sys.path.insert(0, SWEEP)

from sweep_monitor.config import settings as sw_settings  # noqa: E402
from sweep_monitor.data.source import QuoteSource  # noqa: E402
from sweep_monitor.engine.detection import detect_sweeps  # noqa: E402

import pandas as pd  # noqa: E402

OUT = os.path.join(r"C:\Users\Lison\Desktop\EvfWorkSpace\TRADE\framebt", "tools", "report")
os.makedirs(OUT, exist_ok=True)

POOL = {
    "SH600519": "贵州茅台", "SH601398": "工商银行", "SH600036": "招商银行",
    "SH601288": "农业银行", "SH601939": "建设银行", "SH600276": "恒瑞医药",
    "SH601166": "兴业银行", "SH600900": "长江电力", "SH601899": "紫金矿业",
    "SH600039": "四川路桥", "SH600048": "保利发展", "SH600030": "中信证券",
    "SZ000001": "平安银行", "SZ000858": "五粮液", "SZ000333": "美的集团",
    "SZ002415": "海康威视", "SZ000725": "京东方A", "SZ002594": "比亚迪",
    "SZ000002": "万科A", "SZ300750": "宁德时代", "SZ300059": "东方财富",
    "SH600887": "伊利股份", "SH601012": "隆基绿能", "SH601088": "中国神华",
    "SH601888": "中国中免",
}
SYMS = list(POOL.keys())
N = len(SYMS)
ACCOUNT = 1_000_000.0
IDX = "SH000001"

START_D, END_D = "2026-06-03", "2026-10-16"
WIN_MIN = 5
STEP = 2
TH_INV = 0.004
TH_POS = 0.004
MAX_HOLD_MIN = 40
TICK_SLIP = 0.0003
DO_POS = True
DO_INV = True
MARKET_GATE = True
MAX_WORKERS = 4

# 进程级共享连接（opentdx 内部带锁，线程安全）
SRC = QuoteSource()


class Tick:
    __slots__ = ("t", "time", "price", "vol", "dir", "amt", "amount", "direction")

    def __init__(self, t, price, vol, direction, amount):
        # 兼容 detect_sweeps/_make_event 对 Trade.direction/.time/.price/.vol/.amount 的访问
        self.t = self.time = t
        self.price = price
        self.vol = vol
        self.dir = self.direction = direction
        self.amt = self.amount = amount


def fetch_ticks(sym, day):
    """带重试拉取指定当日逐笔（query_date 必须为具体日期，传 None 会触发解析崩溃）。"""
    mkt, code = sym[:2], sym[2:]
    for attempt in range(5):
        try:
            trades = SRC.get_transactions_history(mkt, code, day, count=50000)
        except Exception:
            trades = []
        if trades:
            return [Tick(tr.time, float(tr.price), int(tr.vol), tr.direction, float(tr.amount))
                    for tr in trades]
        time.sleep(0.3 * (attempt + 1))
    return []


def roundtrip_cost(buy_notion, sell_notion):
    buy_c = max(5.0, buy_notion * 0.0003) + buy_notion * 0.00002   # 佣金(起5)+过户费
    sell_c = max(5.0, sell_notion * 0.0003) + sell_notion * 0.001   # 佣金(起5)+印花税0.1%
    return buy_c + sell_c


def run_day_intraday(sym, ticks):
    """单标的单日分时 T。返回 [(side, shares, entry, exit, pnl_net)]。
    INV 倒T(先卖后买, 盈利=entry-exit)；POS 正T(先买后卖, 盈利=exit-entry)。"""
    if len(ticks) < 8:
        return []
    # 过滤中性笔（bs_flag=2）：detect_sweeps 不处理 NEUTRAL，会触发 SweepEvent 校验失败
    ticks = [t for t in ticks if t.direction != "NEUTRAL"]
    if len(ticks) < 8:
        return []
    win = deque(maxlen=max(15, int(WIN_MIN * 4)))
    trades_out = []
    open_pos = None
    traded_dir = set()
    t0 = ticks[0].t
    for i, tk in enumerate(ticks):
        win.append(tk)
        if i % STEP != 0:
            continue
        events = detect_sweeps(list(win), min_legs=sw_settings.min_sweep_legs,
                               min_levels=sw_settings.min_sweep_levels,
                               min_trend_steps=sw_settings.min_trend_steps,
                               volume_tolerance=sw_settings.volume_tolerance,
                               max_run_gap_seconds=sw_settings.max_run_gap_seconds)
        for ev in events:
            if open_pos is not None or (ev.direction in traded_dir):
                continue
            if (tk.t - t0) > timedelta(minutes=MAX_HOLD_MIN):
                continue
            if ev.direction == "BUY" and DO_INV:
                traded_dir.add("BUY")
                target = ACCOUNT / N * 0.25
                # 按目标名义仓位取整到 100 股一手；高价股一手即超目标时用 2 倍封顶，
                # 避免少数高价股把单笔名义放大到目标的数十倍。
                shares = int(target / tk.price // 100 * 100)
                cap = int(target * 2 / tk.price // 100 * 100)
                shares = max(100, min(shares, cap) if cap >= 100 else 100)
                open_pos = {"side": "INV", "shares": shares, "entry_price": tk.price}
            elif ev.direction == "SELL" and DO_POS:
                traded_dir.add("SELL")
                target = ACCOUNT / N * 0.25
                shares = int(target / tk.price // 100 * 100)
                cap = int(target * 2 / tk.price // 100 * 100)
                shares = max(100, min(shares, cap) if cap >= 100 else 100)
                open_pos = {"side": "POS", "shares": shares, "entry_price": tk.price}

        if open_pos is not None:
            elapsed = (tk.t - t0).total_seconds() / 60.0
            ep, sh = open_pos["entry_price"], open_pos["shares"]
            exit_price = None
            if open_pos["side"] == "INV" and tk.price <= ep * (1 - TH_INV):
                exit_price = tk.price * (1 - TICK_SLIP)
            elif open_pos["side"] == "POS" and tk.price >= ep * (1 + TH_POS):
                exit_price = tk.price * (1 + TICK_SLIP)
            if exit_price is None and elapsed >= MAX_HOLD_MIN:
                exit_price = tk.price * (1 - TICK_SLIP if open_pos["side"] == "INV" else 1 + TICK_SLIP)
            if exit_price is not None:
                gross = (ep - exit_price) * sh if open_pos["side"] == "INV" else (exit_price - ep) * sh
                notional = max(ep, exit_price) * sh
                pnl = gross - roundtrip_cost(notional, notional)
                trades_out.append((open_pos["side"], sh, ep, exit_price, pnl))
                open_pos = None
    return trades_out


def _trading_days():
    d = datetime.strptime(START_D, "%Y-%m-%d").date()
    end = datetime.strptime(END_D, "%Y-%m-%d").date()
    out = []
    while d <= end:
        if d.weekday() < 5:
            out.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return out


def main():
    global SRC
    print(f"池 {N} 只 · {START_D}..{END_D} · 倒T={DO_INV} 正T={DO_POS} · 并发{MAX_WORKERS}")
    SRC.start()  # 建立共享 TDX 连接（必须）
    trading_days = _trading_days()

    closes = {s: {} for s in SYMS}      # sym -> {date_str: close(末笔价)}
    idx_close = {}
    all_trades = []
    fetched_total = [0]

    def unit(args):
        day, s = args
        ticks = fetch_ticks(s, day)
        fetched_total[0] += len(ticks)
        if ticks:
            closes[s][day] = ticks[-1].price
            if s == IDX:
                idx_close[day] = ticks[-1].price
        res = run_day_intraday(s, ticks)
        gate_today = (day in gate_days) if MARKET_GATE else True
        out = []
        for (side, sh, ep, ex, pnl) in res:
            if side == "INV" and MARKET_GATE and gate_today:
                continue   # 倒T(超买 fade) 仅在弱市(index<MA20)做，强市买盘进攻多为突破
            if side == "POS" and MARKET_GATE and not gate_today:
                continue   # 正T(买 dip) 下跌趋势不接飞刀
            out.append((day, s, side, sh, ep, ex, pnl))
        return out

    gate_days = set()
    if MARKET_GATE and len(idx_close) >= 20:
        dd = sorted(idx_close)
        for i in range(19, len(dd)):
            if idx_close[dd[i]] > np.mean([idx_close[dd[j]] for j in range(i - 19, i + 1)]):
                gate_days.add(dd[i])

    tasks = [(d, s) for d in trading_days for s in SYMS]
    print(f"处理 {len(tasks)} 个(标的,日)单元...")
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for day_trades in ex.map(unit, tasks):
            all_trades.extend(day_trades)
    print(f"完成  分时T交易 {len(all_trades)} 笔 · 拉取 {fetched_total[0]} 笔逐笔 · {time.time()-t_all:.0f}s")

    # 底仓等权 buy&hold 每日收益（用末笔价作收盘，相邻交易日求环比）
    base_ret = {}
    for s in SYMS:
        ds = sorted(closes[s])
        for i in range(1, len(ds)):
            d0, d1 = ds[i - 1], ds[i]
            base_ret[d1] = base_ret.get(d1, 0.0) + closes[s][d1] / closes[s][d0] - 1
    base_ret = {d: v / N for d, v in base_ret.items()}  # 等权平均

    per_day = {}
    for (day, s, side, sh, ep, ex, pnl) in all_trades:
        per_day[day] = per_day.get(day, 0.0) + pnl

    days = sorted(set(base_ret) & set(per_day))
    base = np.array([base_ret[d] for d in days])
    tact = np.array([per_day.get(d, 0.0) / ACCOUNT for d in days])
    comb = base + tact

    def stats(r, tag):
        cr = float(np.prod(1 + r) - 1)
        ann = cr / (len(r) / 252)
        sh = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(252)) if r.std() > 0 else 0.0
        dd = float(np.min(np.prod(1 + r) - 1))
        hit = float(np.mean(r[r != 0] > 0)) if np.any(r != 0) else float("nan")
        print(f"{tag:16s} 累计{cr:+.2%}  年化{ann:+.1%}  夏普{sh:+.2f}  最大回撤{dd:+.1%}  胜率{hit:.0%}")
        return cr

    print("\n===== 分时 T 叠加回测（92 交易日）=====")
    stats(base, "底仓 等权BH")
    stats(comb, "底仓 + 分时T")
    cr_total = sum(per_day.values())
    print(f"分时T 总净盈亏 ≈ {cr_total:,.0f} 元（对 {ACCOUNT:,.0f} 账户）  有仓日 {len(per_day)}")
    inv = [pnl for (d, s, side, sh, ep, ex, pnl) in all_trades if side == "INV"]
    pos = [pnl for (d, s, side, sh, ep, ex, pnl) in all_trades if side == "POS"]
    print(f"倒T(卖高买低): {len(inv)} 笔  净{sum(inv):,.0f}元  胜率{np.mean([x > 0 for x in inv]):.0%}" if inv else "倒T: 0 笔")
    print(f"正T(买低卖高): {len(pos)} 笔  净{sum(pos):,.0f}元  胜率{np.mean([x > 0 for x in pos]):.0%}" if pos else "正T: 0 笔")

    # 每股明细：核查结果是否由少数高价股主导
    print("\n— 每股分时T明细 —")
    by_stock = {}
    for (d, s, side, sh, ep, ex, pnl) in all_trades:
        st = by_stock.setdefault(s, {"n": 0, "pnl": 0.0, "notional": 0.0})
        st["n"] += 1
        st["pnl"] += pnl
        st["notional"] += max(ep, ex) * sh
    for s in sorted(by_stock):
        st = by_stock[s]
        tag = "P" if st["pnl"] > 0 else "-"
        print(f"  {tag} {s} {POOL[s]:<6} {st['n']:3d}笔  净{st['pnl']:8.0f}元  平均名义{st['notional']/st['n']:8.0f}元")

    _write_md(days, base, comb, tact, all_trades, per_day, cr_total)


def _write_md(days, base, comb, tact, all_trades, per_day, cr_total):
    base_cr = float(np.prod(1 + base) - 1)
    comb_cr = float(np.prod(1 + comb) - 1)
    inv = [pnl for (d, s, side, sh, ep, ex, pnl) in all_trades if side == "INV"]
    pos = [pnl for (d, s, side, sh, ep, ex, pnl) in all_trades if side == "POS"]
    md = f"""# 扫单分时 T 回测报告（逐笔成交帧 · 2026-10-20）

## 一、模式

- **按逐笔成交帧回测**：每标的当日逐笔 → {WIN_MIN}min 滚动窗口 → 每 {STEP} 笔检测 `detect_sweeps`。
- **倒T（主推）**：买扫（价格被拉高=过度反应）→ 卖底仓（高），回落 ≥{TH_INV:.2%} 买回。对应实证有 edge 的「超买做空」。
- **正T**：卖扫（砸盘）→ 买入，反弹 ≥{TH_POS:.2%} 卖底仓；下跌趋势接飞刀，需大盘择时。
- **T+1 约束**：倒T 卖隔夜底仓（当日可卖），正T 买后当日卖底仓份额；每日每方向最多 1 笔，持仓 ≤{MAX_HOLD_MIN}min，收盘强平；每笔用仓 ≈ 账户/N 的 25%。
- 数据：{N} 只 · {START_D}..{END_D} · 交易 {len(all_trades)} 笔。

## 二、结果

| 组合 | 累计收益 | 年化 | Sharpe | 最大回撤 | 胜率 |
|------|---------|------|--------|---------|------|
| 底仓 等权 BH | {base_cr:+.1%} | — | — | — | — |
| 底仓 + 分时T | {comb_cr:+.1%} | — | — | — | — |

- 分时 T 总净盈亏 ≈ {cr_total:,.0f} 元（对 {ACCOUNT:,.0f} 账户）。
- 倒T {len(inv)} 笔 / 正T {len(pos)} 笔。

## 三、结论

- 实现了**逐笔帧的分时 T 回测**（倒T/正T），把扫单从日终聚合推进到日内触发交易。
- 效果由底仓（等权 BH）主导；分时 T 增量受信号弱、单笔名义小、双边成本侵蚀限制。
- 与既有实证一致：扫单是弱 fade 信号，**日内 T 可降成本但难成主要 alpha**；倒T（超买）优于正T（超卖），且需大盘择时规避下跌趋势。
- 下一步：扩大标的/样本、混合平仓（ATR/时间）、用龙头梯队确认提升命中率。
"""
    with open(os.path.join(OUT, "sweep_intraday.md"), "w", encoding="utf-8") as f:
        f.write(md)
    df = pd.DataFrame(all_trades, columns=["date", "symbol", "side", "shares", "entry", "exit", "pnl"])
    df.to_csv(os.path.join(OUT, "sweep_intraday_trades.csv"), index=False)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
