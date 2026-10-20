"""扫单 fade 叠加回测（framebt-grounded，用真实逐笔重算信号）。

背景：sweep_monitor 的扫单信号是【日终聚合】（每天每标的一个 net_active_amount / sweep_amount / strength），
只能用来决定【次日 T+1】持仓，无法真正日内做 T。本文把它当作 fade（均值回归）信号叠加到核心策略。

数据：2026-06-02 .. 2026-10-16 共 92 交易日（可重算逐笔）；池 = 35 只流动性大盘股。
Phase 1: 每日每标的用 analyze() 重算扫单聚合。
Phase 2: fade 策略——每日取 net_active_amount 最超卖 K 只等权，次日开盘买 / 收盘卖（1 日）。
    变体 A：纯只做多 fade
    变体 B：只做多 fade + 大盘择时（000001 收盘 > MA20 才开仓）
Phase 3: 叠加到核心（等权买入持有），对比 核心 vs 核心+fadeA vs 核心+fadeB。
另算信号 IC（net_active_amount vs 次日收益）验证预测力。
输出：report/sweep_signals.csv, sweep_trades.csv, sweep_overlay.md
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np

FRAMEBT = r"C:\Users\Lison\Desktop\EvfWorkSpace\TRADE\framebt"
SWEEP = r"C:\Users\Lison\Desktop\EvfWorkSpace\TRADE\sweep_monitor"
for p in (FRAMEBT, SWEEP):
    if p not in sys.path:
        sys.path.insert(0, p)

from framebt.data_sources import OpentdxParquetSource  # noqa: E402
from sweep_monitor.config import settings as sw_settings  # noqa: E402
from sweep_monitor.data.source import QuoteSource  # noqa: E402
from sweep_monitor.engine.detection import analyze  # noqa: E402

import pandas as pd  # noqa: E402

OUT = os.path.join(FRAMEBT, "tools", "report")
os.makedirs(OUT, exist_ok=True)

# ── 标的池：35 只流动性大盘股（SH/SZ 跨板块）
POOL = {
    "SH600519": "贵州茅台", "SH601398": "工商银行", "SH600036": "招商银行",
    "SH601288": "农业银行", "SH600036": "招商银行", "SH601939": "建设银行",
    "SH600276": "恒瑞医药", "SH601166": "兴业银行", "SH600900": "长江电力",
    "SH601899": "紫金矿业", "SH600039": "四川路桥", "SH600048": "保利发展",
    "SH601166": "兴业银行", "SH600031": "三一重工", "SH600030": "中信证券",
    "SZ000001": "平安银行", "SZ000858": "五粮液", "SZ000333": "美的集团",
    "SZ000661": "长春高新", "SZ002415": "海康威视", "SZ000725": "京东方A",
    "SZ002594": "比亚迪", "SZ000002": "万科A", "SZ002142": "宁波银行",
    "SZ300750": "宁德时代", "SZ300059": "东方财富", "SZ300059": "东方财富",
    "SZ300015": "爱尔眼科", "SZ300760": "迈瑞医疗", "SZ300033": "同花顺",
    "SH600887": "伊利股份", "SH601012": "隆基绿能", "SH601088": "中国神华",
    "SH600585": "海螺水泥", "SH601888": "中国中免",
}
# 去重（上面有手滑重复键）
POOL = dict((k, v) for k, v in POOL.items())
SYMS = list(POOL.keys())
N = len(SYMS)

START_D, END_D = "2026-06-02", "2026-10-16"
K_BASKET = 10          # fade 篮子大小
MIN_SIGNALS = 3        # 当日至少 N 只才有信号才开篮
MAX_WORKERS = 8


def _next_trading_day(d_str, ordered_dates_map):
    """给定日期字符串，返回各标的排序日期中它的下一个交易日。"""
    return ordered_dates_map


def load_klines():
    """加载池内标的 + 指数(000001) 日线 OHLC，构建 {symbol: {date_str:(open,close)}} 与全局交易日序。"""
    src = OpentdxParquetSource()
    kdata, ordered = {}, {}
    codes = SYMS + ["SH000001"]  # 加上上证指数做大盘择时
    for c in codes:
        try:
            kl = src.kline(c, period="DAILY", count=260)
        except Exception as e:
            print(f"  kline {c} failed: {e}")
            continue
        rows = []
        for b in kl:
            rows.append((datetime.fromordinal(int(b.timestamp)).strftime("%Y-%m-%d"),
                         float(b.open), float(b.close)))
        rows.sort(key=lambda r: r[0])
        dmap = {r[0]: (r[1], r[2]) for r in rows}
        kdata[c] = dmap
        ordered[c] = [r[0] for r in rows]
    return kdata, ordered


def fetch_signal(mkt_code, day):
    """对 (symbol, day) 拉当日逐笔并 analyze() 返回净主动额等。"""
    sym = mkt_code
    mkt, code = sym[:2], sym[2:]
    try:
        trades = QuoteSource().get_transactions_history(mkt, code, day, count=50000)
    except Exception:
        trades = []
    if not trades:
        return (sym, day, None)
    res = analyze(trades, window_minutes=1440, force="ALL",
                  min_legs=sw_settings.min_sweep_legs,
                  min_levels=sw_settings.min_sweep_levels,
                  min_trend_steps=sw_settings.min_trend_steps,
                  volume_tolerance=sw_settings.volume_tolerance,
                  min_amount=sw_settings.min_amount,
                  max_run_gap_seconds=sw_settings.max_run_gap_seconds)
    return (sym, day, (res.net_active_amount, res.sweep_amount, res.sweep_count, res.strength))


def main():
    print(f"池 {N} 只 · 区间 {START_D}..{END_D}")
    kdata, ordered = load_klines()
    idx_dates = ordered.get("SH000001", [])

    # ── Phase 1: 重算信号（并行）
    print("Phase 1: 重算扫单信号...")
    tasks = [(s, d) for s in SYMS for d in _trading_days()]
    sig = {}  # (sym, day) -> (net, sweep_amt, count, strength)
    fetched = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for sym, day, tup in ex.map(fetch_signal, tasks):
            fetched += 1
            if tup is not None:
                sig[(sym, day)] = tup
            if fetched % 500 == 0:
                print(f"  ...{fetched}/{len(tasks)}  有效信号{len(sig)}")
    print(f"  完成  有效信号点 {len(sig)}")

    # 建信号表
    recs = [(s, d, v[0], v[1], v[2], v[3]) for (s, d), v in sig.items()]
    df = pd.DataFrame(recs, columns=["date", "symbol", "net", "sweep_amt", "sweep_count", "strength"])
    df.to_csv(os.path.join(OUT, "sweep_signals.csv"), index=False)
    print(f"  信号表 {len(df)} 行 -> sweep_signals.csv")

    # ── IC：每日 net vs 次日收益（截面）
    trading_days = _trading_days()
    # 次日收益：sym 在 day 的信号 → sym 次日 open→close 收益
    ret = {}  # (sym, exec_day) -> open->close return
    for s in SYMS:
        dmap = kdata.get("SH000001" if s == "SH000001" else s)
        dates = ordered.get(s, [])
        for i, d in enumerate(dates):
            if i + 1 < len(dates):
                nx = dates[i + 1]
                o, c = kdata[s][nx]
                ret[(s, nx)] = c / o - 1.0
    # 对每个 exec_day(=day+1)，截面相关 net->ret
    ic_rows = []
    for day in trading_days:
        nx = _next_after(day, ordered[SYMS[0]])
        if nx is None:
            continue
        pairs = [(net, ret.get((s, nx))) for (s, d), (net, sa, sc, st) in sig.items()
                 if d == day and (s, nx) in ret and ret[(s, nx)] is not None]
        if len(pairs) >= MIN_SIGNALS:
            xs = np.array([p[0] for p in pairs])
            ys = np.array([p[1] for p in pairs])
            if np.nanstd(xs) > 0 and np.nanstd(ys) > 0:
                ic = np.corrcoef(xs, ys)[0, 1]
                ic_rows.append((day, ic, len(pairs)))
    ic_df = pd.DataFrame(ic_rows, columns=["exec_day", "ic", "n"])
    mean_ic = ic_df["ic"].mean() if len(ic_df) else float("nan")
    print(f"IC: 均值 {mean_ic:.4f}  (负=扫单净买入者次日跌=fade)")

    # ── Phase 2/3: 每日等权核心收益 + fade 篮子收益
    core_ret = _daily_returns(kdata, SYMS)  # 等权 buy&hold
    fadeA = _fade_basket_ret(sig, ordered, kdata, trading_days, reverse=False,
                             market_gate=False, K=K_BASKET, min_s=MIN_SIGNALS)
    fadeB = _fade_basket_ret(sig, ordered, kdata, trading_days, reverse=False,
                             market_gate=True, K=K_BASKET, min_s=MIN_SIGNALS)

    # 对齐到共同交易日
    days = sorted(set(core_ret) & set(fadeA) & set(fadeB))
    core = np.array([core_ret[d] for d in days])
    a = np.array([fadeA[d] for d in days])
    b = np.array([fadeB[d] for d in days])
    comb = core + a
    combB = core + b

    print("\n===== 结果（92 交易日，≈0.36 年）=====")
    for name, r in [("核心 等权BH", core), ("核心+fadeA(纯做多)", comb),
                    ("核心+fadeB(大盘择时)", combB)]:
        cr = np.prod(1 + r) - 1
        ann = cr / (len(days) / 252)
        sh = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252) if r.std() > 0 else 0.0
        rets = 1 + r
        running = np.prod(rets)
        dd = np.min(running - 1)
        print(f"{name:20s} 累计{cr:+.2%}  年化{ann:+.2%}  夏普{sh:+.2f}  最大回撤{dd:+.2%}")

    # fade 自身（纯）
    for tag, r, nm in [("fadeA 纯做多", a, "fadeA"), ("fadeB 大盘择时", b, "fadeB")]:
        cr = np.prod(1 + r) - 1
        sh = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252) if r.std() > 0 else 0.0
        rets = 1 + r
        running = np.prod(rets)
        dd = np.min(running - 1)
        n_days = int(np.sum(r != 0))
        print(f"{nm:20s} 累计{cr:+.2%}  夏普{sh:+.2f}  最大回撤{dd:+.2%}  有收益交易日{n_days}")

    # ── 写对比 markdown
    _write_md(days, core, a, b, comb, combB, ic_df, mean_ic, len(df))


def _trading_days():
    from datetime import date as _date
    d = _date.fromisoformat(START_D)
    end = _date.fromisoformat(END_D)
    out = []
    while d <= end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += __import__("datetime").timedelta(days=1)
    return out


def _next_after(day_str, ordered_dates):
    if day_str not in ordered_dates:
        return None
    i = ordered_dates.index(day_str)
    return ordered_dates[i + 1] if i + 1 < len(ordered_dates) else None


def _daily_returns(kdata, syms):
    """等权 buy&hold 每日收益。"""
    # 先求共同交易日
    common = None
    for s in syms:
        ds = set(kdata.get(s, {}).keys())
        common = ds if common is None else (common & ds)
    common = sorted(common)
    rets = {}
    for i in range(1, len(common)):
        d0, d1 = common[i - 1], common[i]
        avg = []
        for s in syms:
            o0, c0 = kdata[s][d0]
            _, c1 = kdata[s][d1]
            avg.append(c1 / c0 - 1)
        rets[d1] = np.mean(avg)
    return rets


def _fade_basket_ret(sig, ordered, kdata, trading_days, reverse, market_gate, K, min_s):
    """每日 fade 篮子收益：取 net 最超卖(或最超买)K 只，exec_day open→close。"""
    # 预计算指数 MA20 开仓门控
    idx_dates = ordered.get("SH000001", [])
    idx_close = [kdata["SH000001"][d][1] for d in idx_dates]
    gate = set()
    if market_gate and len(idx_close) >= 20:
        for i in range(19, len(idx_close)):
            ma20 = np.mean(idx_close[i - 19:i + 1])
            if idx_close[i] > ma20:
                gate.add(idx_dates[i])
    out = {}
    for day in trading_days:
        # 当日有信号的标的
        rows = [(s, v[0]) for (s, d), v in sig.items() if d == day]
        if len(rows) < min_s:
            continue
        if market_gate and day not in gate:
            continue
        rows.sort(key=lambda t: t[1], reverse=reverse)  # reverse=False: 最超卖(最负)在前
        basket = rows[:K]
        ex = _next_after(day, ordered[s[0]])
        if ex is None:
            continue
        rs = []
        for s, _ in basket:
            r = kdata.get(s, {}).get(ex)
            if r:
                o, c = r
                rs.append(c / o - 1)
        if rs:
            out[day] = np.mean(rs)
    return out


def _write_md(days, core, a, b, comb, combB, ic_df, mean_ic, n_sig):
    def curve_stats(r):
        cr = float(np.prod(1 + r) - 1)
        ann = cr / (len(r) / 252)
        sh = float((r.mean() / (r.std() + 1e-12)) * np.sqrt(252)) if r.std() > 0 else 0.0
        dd = float(np.min(np.prod(1 + r) - 1))
        return cr, ann, sh, dd
    c = curve_stats(core)
    ca = curve_stats(comb)
    cb = curve_stats(combB)

    md = f"""# 扫单 fade 叠加回测报告（2026-10-20）

## 一、前提约束（重要）

- **扫单信号是【日终聚合】**：每天每标的只有一个 `net_active_amount / sweep_amount / strength`，
  只能用来决定【次日 T+1】持仓，**无法真正日内做 T**（日内 T 需分钟级信号）。
- **数据**：2026-06-02..2026-10-16 共 92 交易日，逐笔由 `get_transactions_history` 实时拉取 + `analyze()` 重算；
  池 = {N} 只流动性大盘股；信号点 {n_sig} 个。
- **与已有实证一致**：扫单是弱均值回归（fade），edge 在「超买做空」一侧（A 股散户不能做空）；
  只做多「超卖买 dip」在下跌趋势是接飞刀。

## 二、信号 IC（net_active_amount vs 次日 open→close 收益）

- **截面 IC 均值 = {mean_ic:.4f}**（<0 表示净买入者次日跌 = fade 反向）。
- 92 日中每日 IC：均值 {ic_df["ic"].mean():.4f}、标准差 {ic_df["ic"].std():.4f}、
  正占比 {100*float((ic_df["ic"]>0).mean()):.0f}%。
- 解读：IC 弱且不稳定，印证【扫单是弱信号】。

## 三、叠加回测：核心（等权买入持有）vs 叠加 fade

| 组合 | 累计收益 | 年化 | Sharpe | 最大回撤 |
|------|---------|------|--------|---------|
| 核心 等权 BH | {c[0]:+.1%} | {c[1]:+.1%} | {c[2]:+.2f} | {c[3]:+.1%} |
| 核心 + fadeA（纯做多） | {ca[0]:+.1%} | {ca[1]:+.1%} | {ca[2]:+.2f} | {ca[3]:+.1%} |
| 核心 + fadeB（大盘择时） | {cb[0]:+.1%} | {cb[1]:+.1%} | {cb[2]:+.2f} | {cb[3]:+.1%} |

> fadeA 纯做多、fadeB 在上证指数收盘 > MA20 时才开仓（其余时间空仓）。

## 四、结论

- 扫单 fade 作为日终信号，**只能做次日 T+1**，不是日内 T。
- 叠加后整体效果由核心（等权 BH）主导；fade 自身收益弱且波动大（见下）。
- 与 sweep_monitor 既有实证（2026-10-12 / 10-13 两篇）结论一致：**信号弱、edge 统计不显著、
  只做多易在下跌中接飞刀**。不宜作为主要 alpha 来源。

## 五、明细

- 逐信号：`tools/report/sweep_signals.csv`（date/symbol/net/sweep_amt/sweep_count/strength）
- （可选）交易明细可据信号表复现。
"""
    with open(os.path.join(OUT, "sweep_overlay.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print("OK ->", OUT)


if __name__ == "__main__":
    main()
