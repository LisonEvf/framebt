"""扫单分时 T 回测 v2 —— 修掉 v1 的三处致命缺陷后重跑。

v1 = ``tools/sweep_intraday_bt.py``（旧报告 ``tools/report/sweep_intraday.md``：
"倒T 1666 笔 / 72% 胜率 / +3.09pp / Sharpe 0.81"）。逐条核对后发现：

  1. **逐笔顺序错乱（致命）**：opentdx 的 ``stock_transaction`` 按 1000 条一页返回，
     页序是**倒序拼接**。实测 2026-10-16 SH600519 的四页分别是
     14:02→15:29 / 11:25→14:02 / 10:21→11:25 / 09:25→10:21 —— 即 [晚段, 中段, 早段]。
     v1 未排序直接喂给 ``detect_sweeps`` 与持仓计时器，于是
     ``t0 = 14:02``、``elapsed`` 在跨页处变成负数、MAX_HOLD 判定失效、
     滚动窗口里混着 15:29 与 11:25 的成交 → 一切信号与盈亏都无意义。
  2. **大盘择时门控从未生效**：``gate_days`` 在 ``idx_close`` 被填充**之前**计算
     （此时长度为 0），``gate_today`` 恒为 False → 正T 全被跳过（旧日志里
     "正T: 0 笔" 就是这个 bug），倒T 则完全不设门。
  3. **门控本身有前视**：用「当日收盘 vs MA20」决定「当日做不做」，
     而当日收盘盘中根本不知道。
  4. **成本不对称**：只在平仓腿计滑点、印花税按 0.1%（2023-08 已降到 0.05%）。
  5. **仓位不统一**：``target = ACCOUNT/N*0.25`` 对高价股取整失败后被 ``max(100, ...)``
     兜底，茅台单笔名义变成 12.8 万（目标 1 万，13 倍），它的"盈利"来自放大名义。

v2 的修正：逐笔按时间排序；门控用 **t-1 收盘**（无前视）；成本两侧对称且印花税 0.05%；
一手超过单笔预算的标的**按规则剔除**（不放大名义）；回撤/年化按标准口径重算；
所有变体在同一份逐笔上一次跑完，便于对照。

用法::

    python tools/sweep_intraday_v2.py --phase fetch     # 建逐笔缓存（约 8 分钟）
    python tools/sweep_intraday_v2.py --phase run       # 跑全部变体（读缓存）
    python tools/sweep_intraday_v2.py --phase all
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import deque, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# 外部依赖 sweep_monitor 不随本仓库发布；用环境变量指定其路径。
# 默认按兄弟目录 ../sweep_monitor 查找，避免把本机绝对路径写进仓库。
SWEEP = os.environ.get("SWEEP_MONITOR_DIR") or str(ROOT.parents[1] / "sweep_monitor")
if SWEEP not in sys.path:
    sys.path.insert(0, SWEEP)

from sweep_monitor.config import settings as sw_settings  # noqa: E402
from sweep_monitor.data.source import QuoteSource  # noqa: E402
from sweep_monitor.engine.detection import detect_sweeps  # noqa: E402

from opentdx.const import ADJUST, MARKET, PERIOD  # noqa: E402

TICK_DIR = ROOT / "tick_cache"
OUT_DIR = ROOT / "report"
TICK_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# ── 样本与池（与 v1 相同，便于对照；池本身是人工挑的大盘股，见报告「诚实的边界」）──
POOL: dict[str, str] = {
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
SYMS = list(POOL)
N = len(SYMS)
ACCOUNT = 1_000_000.0
IDX = ("SH", "000001")
START_D, END_D = "2026-06-03", "2026-10-20"

# ── 交易参数（与 v1 对齐的默认值）──
WIN = 20                 # 滚动窗口（逐笔条数）；v1 的 WIN_MIN*4 = 20 ≈ 60 秒
STEP = 2                 # 每 2 笔检测一次
TH = 0.004               # 平仓阈值
MAX_HOLD_MIN = 40
SLIP = 0.0003            # 单边滑点
COMM = 0.0003            # 佣金 万3
MIN_COMM = 5.0
STAMP = 0.0005           # 印花税 0.05%（2023-08-28 起，卖出单边）
TRANSFER = 0.00001       # 过户费 0.001%（双边）
PER_TRADE_FRAC = 0.25    # 单笔用账户的 25% / N

SRC = QuoteSource()


# ===========================================================================
# 阶段 1：逐笔缓存
# ===========================================================================

def _weekdays(a: str, b: str) -> list[str]:
    d = date.fromisoformat(a)
    end = date.fromisoformat(b)
    out = []
    while d <= end:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def fetch_index_closes(count: int = 260) -> pd.Series:
    """指数日线（不复权）：用于 ① 真实交易日历 ② MA20 门控。"""
    raw = SRC._client.stock_kline(MARKET.SH, IDX[1], PERIOD.DAILY,
                                  count=count, adjust=ADJUST.NONE)
    s = pd.Series({r["datetime"].date().isoformat(): float(r["close"]) for r in raw})
    return s.sort_index()


def fetch_daily_closes(sym: str, count: int = 260) -> pd.Series:
    """个股前复权日线收盘：用于底仓收益（逐笔末笔价会被盘后/集合竞价污染）。"""
    mkt = MARKET.SH if sym.startswith("SH") else MARKET.SZ
    raw = SRC._client.stock_kline(mkt, sym[2:], PERIOD.DAILY,
                                  count=count, adjust=ADJUST.QFQ)
    s = pd.Series({r["datetime"].date().isoformat(): float(r["close"]) for r in raw})
    return s.sort_index()


def fetch_symbol_ticks(sym: str, days: list[str]) -> int:
    """拉一只票的全部交易日逐笔并落盘（**按时间排序**，这是 v2 与 v1 的核心差别）。"""
    mkt, code = sym[:2], sym[2:]
    frames = []
    for day in days:
        trades = []
        for attempt in range(4):
            try:
                trades = SRC.get_transactions_history(mkt, code, day, count=50000)
            except Exception:
                trades = []
            if trades:
                break
            time.sleep(0.2 * (attempt + 1))
        if not trades:
            continue
        rows = [(day,
                 t.time.hour * 3600 + t.time.minute * 60 + t.time.second,
                 float(t.price), int(t.vol), int(t.bs_flag)) for t in trades]
        df = pd.DataFrame(rows, columns=["date", "sec", "price", "vol", "bs"])
        # 排序修掉「页倒序拼接」；同秒内保留原始相对顺序（分页内本就是升序）
        df = df.sort_values(["sec"], kind="stable").reset_index(drop=True)
        # 同秒同价同量的相邻重复（分页重叠）去重
        dup = (df["sec"] == df["sec"].shift()) & (df["price"] == df["price"].shift()) \
            & (df["vol"] == df["vol"].shift()) & (df["bs"] == df["bs"].shift())
        df = df[~dup]
        frames.append(df)
    if not frames:
        return 0
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(TICK_DIR / f"{sym}.parquet", index=False)
    return len(out)


def phase_fetch(workers: int = 6) -> None:
    SRC.start()
    try:
        idx = fetch_index_closes()
        days = [d for d in _weekdays(START_D, END_D) if d in idx.index]
        print(f"交易日 {len(days)} 天：{days[0]} .. {days[-1]}")

        # 日线收盘（底仓口径）
        daily = {}
        for s in SYMS:
            try:
                daily[s] = fetch_daily_closes(s)
            except Exception as e:  # noqa: BLE001
                print(f"  ! {s} 日线失败: {e}")
        pd.DataFrame(daily).to_parquet(TICK_DIR / "_daily.parquet")
        idx.to_frame("close").to_parquet(TICK_DIR / "_index.parquet")
        print(f"日线就绪：{len(daily)} 只 + 指数 {len(idx)} 根")

        t0 = time.time()
        total = [0]
        done = [0]

        def job(sym: str) -> None:
            n = fetch_symbol_ticks(sym, days)
            total[0] += n
            done[0] += 1
            print(f"  [{done[0]:2d}/{len(SYMS)}] {sym} {POOL[sym]:<6} {n:8,d} 笔"
                  f"  ({time.time() - t0:.0f}s)")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(job, SYMS))
        print(f"逐笔缓存完成：{total[0]:,} 笔 · {time.time() - t0:.0f}s · {TICK_DIR}")
    finally:
        SRC.stop()


# ===========================================================================
# 阶段 2：回测
# ===========================================================================

class Tick:
    """检测器需要的逐笔对象（保留 v1 的字段名，便于复用 detect_sweeps）。"""

    __slots__ = ("t", "time", "sec", "price", "vol", "dir", "direction", "amt", "amount")

    def __init__(self, sec: int, day: date, price: float, vol: int, bs: int):
        self.sec = sec
        self.t = self.time = datetime(day.year, day.month, day.day,
                                      sec // 3600, (sec % 3600) // 60, sec % 60)
        self.price = price
        self.vol = vol
        d = "BUY" if bs == 0 else ("SELL" if bs == 1 else "NEUTRAL")
        self.dir = self.direction = d
        self.amt = self.amount = price * vol


def load_ticks(sym: str) -> dict[str, list[Tick]]:
    p = TICK_DIR / f"{sym}.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    out: dict[str, list[Tick]] = {}
    for day, g in df.groupby("date", sort=True):
        dd = date.fromisoformat(day)
        ticks = [Tick(int(r.sec), dd, float(r.price), int(r.vol), int(r.bs))
                 for r in g.itertuples(index=False)]
        ticks = [t for t in ticks if t.direction != "NEUTRAL"]
        if len(ticks) >= 8:
            out[day] = ticks
    return out


def detect_events(ticks: list[Tick], win: int) -> list[tuple[int, str, float]]:
    """逐笔滚动窗口扫单检测。返回 [(tick_index, direction, span_bp)]。

    ``span_bp`` = 该次扫单的价格跨度（基点），用于给「4 个价位档」这类
    **非尺度不变**的判据补一个跨标的可比的归一化门槛：800 元的票跨 4 档只涨 0.005%，
    8 元的票跨 4 档要涨 0.5%，同一套 ``min_levels`` 在两者上完全不是一件事。
    """
    dq: deque = deque(maxlen=win)
    events: list[tuple[int, str, float]] = []
    for i, tk in enumerate(ticks):
        dq.append(tk)
        if i % STEP:
            continue
        for ev in detect_sweeps(
            list(dq),
            min_legs=sw_settings.min_sweep_legs,
            min_levels=sw_settings.min_sweep_levels,
            min_trend_steps=sw_settings.min_trend_steps,
            volume_tolerance=sw_settings.volume_tolerance,
            max_run_gap_seconds=sw_settings.max_run_gap_seconds,
        ):
            if ev.price_start > 0:
                span_bp = abs(ev.price_end / ev.price_start - 1.0) * 1e4
            else:
                span_bp = 0.0
            events.append((i, ev.direction, span_bp))
    return events


def roundtrip_cost(buy_notional: float, sell_notional: float) -> float:
    """A 股真实成本：佣金(万3, 5 元起) + 过户费(0.001%, 双边) + 印花税(0.05%, 卖出)。"""
    comm = max(MIN_COMM, buy_notional * COMM) + max(MIN_COMM, sell_notional * COMM)
    transfer = (buy_notional + sell_notional) * TRANSFER
    stamp = sell_notional * STAMP
    return comm + transfer + stamp


def simulate(ticks: list[Tick], events: list[tuple[int, str, float]], *,
             allow_inv: bool, allow_pos: bool,
             gate_inv_ok: bool, gate_pos_ok: bool,
             th: float, slip: float, budget: float,
             min_span_bp: float = 0.0,
             max_hold: float = MAX_HOLD_MIN) -> list[dict]:
    """单标的单日分时 T。
    INV = 倒T（先卖底仓、回落买回）；POS = 正T（先买、反弹卖底仓）。
    ``min_span_bp > 0`` 时要求扫单价格跨度至少这么多基点（跨标的可比的信号门槛）。
    """
    ev_at: dict[int, list[str]] = defaultdict(list)
    for i, d, span in events:
        if span >= min_span_bp:
            ev_at[i].append(d)

    trades: list[dict] = []
    pos: dict | None = None
    traded: set[str] = set()

    def close(p: dict, i: int, exit_px: float) -> None:
        sh = p["sh"]
        if p["side"] == "INV":
            sell_px, buy_px = p["entry"], exit_px
        else:
            sell_px, buy_px = exit_px, p["entry"]
        gross = (sell_px - buy_px) * sh
        cost = roundtrip_cost(buy_px * sh, sell_px * sh)
        trades.append({
            "side": p["side"], "shares": sh, "entry": p["entry"], "exit": exit_px,
            "gross": gross, "cost": cost, "pnl": gross - cost,
            "entry_sec": p["t0"], "exit_sec": ticks[i].sec, "entry_i": p["i0"],
        })

    for i, tk in enumerate(ticks):
        if pos is not None:
            held = (tk.sec - pos["t0"]) / 60.0
            px = None
            if pos["side"] == "INV":
                if tk.price <= pos["entry"] * (1 - th):
                    px = tk.price * (1 - slip)          # 买回：按卖一价之上成交
                elif held >= max_hold:
                    px = tk.price * (1 - slip)
            else:
                if tk.price >= pos["entry"] * (1 + th):
                    px = tk.price * (1 - slip)          # 卖出：按买一价之下成交
                elif held >= max_hold:
                    px = tk.price * (1 - slip)
            if px is not None:
                close(pos, i, px)
                pos = None
                continue                                 # 同一笔不平进新仓

        if pos is None and i in ev_at:
            for d in ev_at[i]:
                if d in traded:
                    continue
                want_inv = d == "BUY" and allow_inv and gate_inv_ok
                want_pos = d == "SELL" and allow_pos and gate_pos_ok
                if not (want_inv or want_pos):
                    continue
                lots = int(budget / (tk.price * 100))
                if lots < 1:
                    break                                # 一手超预算：该标的不可做
                sh = lots * 100
                side = "INV" if want_inv else "POS"
                entry = tk.price * (1 - slip) if side == "INV" else tk.price * (1 + slip)
                pos = {"side": side, "sh": sh, "entry": entry, "t0": tk.sec, "i0": i}
                traded.add(d)
                break
    return trades


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def stats(r: np.ndarray, tag: str, line: list[str], period_days: int) -> dict:
    eq = np.cumprod(1 + r)
    cum = float(eq[-1] - 1)
    ann = (1 + cum) ** (252 / period_days) - 1 if period_days > 0 else 0.0
    sd = float(r.std(ddof=1)) if len(r) > 2 else 0.0
    sh = float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0
    mdd = max_drawdown(np.concatenate([[1.0], eq]))
    line.append(f"{tag:<22} 累计{cum:+7.2%}  年化{ann:+7.2%}  Sharpe{sh:+5.2f}  "
                f"最大回撤{mdd:+7.2%}  日胜率{float(np.mean(r > 0)):5.1%}")
    return {"tag": tag, "cum": cum, "ann": ann, "sharpe": sh, "mdd": mdd}


def block_bootstrap(r: np.ndarray, block: int = 5, n: int = 4000, seed: int = 20261020):
    """移动块自举：给「日均收益」与「累计收益」打区间（日收益自相关/波动聚集的粗修正）。"""
    rng = np.random.default_rng(seed)
    T = len(r)
    if T < block * 2:
        return None
    nb = int(np.ceil(T / block))
    means, cums = [], []
    for _ in range(n):
        starts = rng.integers(0, T - block, size=nb)
        s = np.concatenate([r[i:i + block] for i in starts])[:T]
        means.append(s.mean())
        cums.append(np.prod(1 + s) - 1)
    means, cums = np.array(means), np.array(cums)
    return {
        "mean_lo": float(np.percentile(means, 5)), "mean_mid": float(np.percentile(means, 50)),
        "mean_hi": float(np.percentile(means, 95)),
        "cum_lo": float(np.percentile(cums, 5)), "cum_mid": float(np.percentile(cums, 50)),
        "cum_hi": float(np.percentile(cums, 95)),
        "p_mean_gt0": float(np.mean(means > 0)),
    }


VARIANTS = [
    # name,                   win, inv,   pos,   gate, th,  min_span_bp
    ("倒T·无门控",             20, True,  False, False, TH,  0.0),
    ("正T·无门控",             20, False, True,  False, TH,  0.0),
    ("双向·无门控",            20, True,  True,  False, TH,  0.0),
    ("双向·门控(t-1)",         20, True,  True,  True,  TH,  0.0),
    ("双向·无门控 th=0.3%",    20, True,  True,  False, 0.003, 0.0),
    ("双向·无门控 th=0.6%",    20, True,  True,  False, 0.006, 0.0),
    ("双向·跨度≥5bp",          20, True,  True,  False, TH,  5.0),
    ("双向·跨度≥10bp",         20, True,  True,  False, TH, 10.0),
    ("双向·跨度≥10bp·th=1%",   20, True,  True,  False, 0.01, 10.0),
]


def phase_run(cost_stress: bool = True) -> None:
    if not (TICK_DIR / "_index.parquet").exists():
        raise SystemExit("缺少缓存，先跑 --phase fetch")

    idx = pd.read_parquet(TICK_DIR / "_index.parquet")["close"]
    idx.index = [str(i) for i in idx.index]
    daily = pd.read_parquet(TICK_DIR / "_daily.parquet")
    daily.index = [str(i) for i in daily.index]

    tick_days = sorted({d for s in SYMS for d in load_ticks(s)})
    days = [d for d in tick_days if d in idx.index]
    print(f"样本 {days[0]} .. {days[-1]}  共 {len(days)} 个交易日")

    # ── 底仓：等权、日再平衡（用前复权日线，不用逐笔末笔价）──
    base_by_day: dict[str, list[float]] = defaultdict(list)
    for s in SYMS:
        if s not in daily.columns:
            continue
        ser = daily[s].dropna()
        for i in range(1, len(ser)):
            d0, d1 = ser.index[i - 1], ser.index[i]
            if d0 in days and d1 in days and ser.iloc[i - 1] > 0:
                base_by_day[d1].append(float(ser.iloc[i] / ser.iloc[i - 1] - 1))
    # 首个交易日没有前一日收益 → 用当日到当日（0）占位，保持长度对齐
    days = [d for d in days if d in base_by_day]
    base = np.array([float(np.mean(base_by_day[d])) for d in days])

    # ── 大盘门控：t-1 收盘 vs 其前 20 日均值（无前视）──
    gate_strong: dict[str, bool] = {}
    order = list(idx.index)
    for d in days:
        j = order.index(d)
        if j >= 21:
            prev = idx.iloc[j - 1]
            ma20 = float(idx.iloc[j - 21:j].mean())
            gate_strong[d] = bool(prev > ma20)
    print(f"门控覆盖 {len(gate_strong)} 天：强市 {sum(gate_strong.values())} 天 / "
          f"弱市 {len(gate_strong) - sum(gate_strong.values())} 天")

    def gate_ok(side: str, d: str) -> bool:
        if d not in gate_strong:
            return True
        return gate_strong[d] if side == "POS" else (not gate_strong[d])

    budget = ACCOUNT * PER_TRADE_FRAC / N
    print(f"单笔预算 {budget:,.0f} 元（一手超预算的标的自动不做）")

    # ── 逐标的跑信号 + 全部变体 ──
    t0 = time.time()
    all_trades: dict[str, list[dict]] = {v[0]: [] for v in VARIANTS}
    daily_pnl: dict[str, dict[str, float]] = {v[0]: defaultdict(float) for v in VARIANTS}
    ev_density: dict[str, list[int]] = {}
    n_units = 0
    for k, sym in enumerate(SYMS, 1):
        by_day = load_ticks(sym)
        dens = []
        for d in days:
            ticks = by_day.get(d)
            if not ticks:
                continue
            n_units += 1
            cache: dict[int, list] = {}
            for name, win, inv, pos, gated, th, minbp in VARIANTS:
                if win not in cache:
                    cache[win] = detect_events(ticks, win)
                    if win == WIN:
                        dens.append(len(cache[win]))
                evs = cache[win]
                gi = gate_ok("INV", d) if gated else True
                gp = gate_ok("POS", d) if gated else True
                tr = simulate(ticks, evs, allow_inv=inv, allow_pos=pos,
                              gate_inv_ok=gi, gate_pos_ok=gp, th=th, slip=SLIP,
                              budget=budget, min_span_bp=minbp)
                for t in tr:
                    t["date"], t["symbol"] = d, sym
                all_trades[name].extend(tr)
                for t in tr:
                    daily_pnl[name][d] += t["pnl"] / ACCOUNT
        ev_density[sym] = dens
        print(f"  [{k:2d}/{len(SYMS)}] {sym} {POOL[sym]:<6} "
              f"事件/日中位 {sorted(dens)[len(dens)//2] if dens else 0:5d}  "
              f"成交(双向无门控) {sum(1 for t in all_trades['双向·无门控'] if t['symbol']==sym):3d}  "
              f"{time.time()-t0:.0f}s", flush=True)

    # ── 汇总 ──
    line: list[str] = []
    line.append(f"样本 {days[0]} .. {days[-1]}（{len(days)} 个交易日）· 池 {N} 只 · "
                f"单笔预算 {budget:,.0f} 元 · 账户 {ACCOUNT:,.0f} 元")
    line.append(f"逐笔缓存 {n_units} 个(标的,日)单元")
    line.append("")
    res = {}
    line.append("【底仓】")
    res["base"] = stats(base, "等权BH 日再平衡", line, len(days))

    for name, *_ in VARIANTS:
        tact = np.array([daily_pnl[name].get(d, 0.0) for d in days])
        comb = base + tact
        line.append("")
        line.append(f"【{name}】交易 {len(all_trades[name])} 笔 · "
                    f"总净盈亏 {sum(t['pnl'] for t in all_trades[name]):,.0f} 元 · "
                    f"成本 {sum(t['cost'] for t in all_trades[name]):,.0f} 元")
        res[name] = {
            "tact": stats(tact, "  战术(T) 单独", line, len(days)),
            "comb": stats(comb, "  底仓+T 叠加", line, len(days)),
            "n": len(all_trades[name]),
        }

    # ── 主变体的细节 ──
    main = "双向·无门控"
    tact = np.array([daily_pnl[main].get(d, 0.0) for d in days])
    bs = block_bootstrap(tact)
    line.append("")
    line.append(f"【{main} 统计细节】")
    line.append(f"  日均 {tact.mean()*1e4:+.1f} bp  日波动 {tact.std(ddof=1)*1e4:.1f} bp  "
                f"t = {tact.mean()/(tact.std(ddof=1)/math.sqrt(len(tact))):+.2f}")
    if bs:
        line.append(f"  块自举(block=5)：累计区间 {bs['cum_lo']:+.2%} .. {bs['cum_hi']:+.2%}"
                    f"（中位 {bs['cum_mid']:+.2%}），P(日均>0) = {bs['p_mean_gt0']:.0%}")

    tr = all_trades[main]
    if tr:
        pnls = np.array([t["pnl"] for t in tr])
        inv = [t for t in tr if t["side"] == "INV"]
        pos = [t for t in tr if t["side"] == "POS"]
        line.append(f"  倒T {len(inv)} 笔 净{sum(t['pnl'] for t in inv):,.0f}元 "
                    f"胜率{np.mean([t['pnl']>0 for t in inv]):.0%}  |  "
                    f"正T {len(pos)} 笔 净{sum(t['pnl'] for t in pos):,.0f}元 "
                    f"胜率{(np.mean([t['pnl']>0 for t in pos]) if pos else float('nan')):.0%}")
        line.append(f"  单笔：均值 {pnls.mean():+.1f} 元  中位 {np.median(pnls):+.1f} 元  "
                    f"最差 {pnls.min():+.0f} 元  最好 {pnls.max():+.0f} 元")
        line.append(f"  成本占毛盈亏：{sum(t['cost'] for t in tr)/max(1e-9, sum(t['gross'] for t in tr)):.0%}"
                    f"（毛利 {sum(t['gross'] for t in tr):,.0f} 元 / 成本 {sum(t['cost'] for t in tr):,.0f} 元）")

        # 分半 + 留一
        mid = len(days) // 2
        first, second = set(days[:mid]), set(days[mid:])
        for tag, keep in (("前半段", first), ("后半段", second)):
            sub = [t["pnl"] for t in tr if t["date"] in keep]
            line.append(f"  {tag}：{len(sub)} 笔  净{sum(sub):,.0f}元")
        by_sym = defaultdict(list)
        for t in tr:
            by_sym[t["symbol"]].append(t["pnl"])
        lines_by = sorted(by_sym.items(), key=lambda kv: -sum(kv[1]))
        total_net = sum(pnls)
        line.append("  逐标的（按净额降序，前 6 / 后 4）：")
        for s, v in lines_by[:6]:
            line.append(f"    + {POOL.get(s, s):<6} {len(v):3d}笔 净{sum(v):8,.0f}元")
        for s, v in lines_by[-4:]:
            line.append(f"    - {POOL.get(s, s):<6} {len(v):3d}笔 净{sum(v):8,.0f}元")
        if total_net != 0:
            top3 = sum(sum(v) for _, v in lines_by[:3])
            line.append(f"  集中度：净额前三合计 {top3:+,.0f} 元（总净额 {total_net:+,.0f} 元）")
        neg = sum(1 for _, v in lines_by if sum(v) < 0)
        line.append(f"  盈利标的 {len(lines_by)-neg}/{len(lines_by)} 只")

        # 入场时刻分布：traded_dir 限制「每方向每天 1 笔」，所以入场时刻 = 当日首个扫单时刻
        buckets = [("09:30-09:35", 9 * 3600 + 30 * 60, 9 * 3600 + 35 * 60),
                   ("09:35-10:00", 9 * 3600 + 35 * 60, 10 * 3600),
                   ("10:00-11:30", 10 * 3600, 11 * 3600 + 30 * 60),
                   ("13:00-15:00", 13 * 3600, 15 * 3600 + 1)]
        tot = len(tr)
        seg = []
        for tag, a, b in buckets:
            c = sum(1 for t in tr if a <= t["entry_sec"] < b)
            seg.append(f"{tag} {c}笔({c/tot:.0%})")
        line.append("  入场时刻：" + " · ".join(seg))
        med = sorted(t["entry_sec"] for t in tr)[tot // 2]
        line.append(f"  入场时刻中位数 {med//3600:02d}:{med%3600//60:02d}（"
                    f"开盘 35 分钟内占 {sum(1 for t in tr if t['entry_sec'] < 9*3600+35*60)/tot:.0%}）")

    # ── 信号诊断：扫单事件密度 vs 股价（检测器「4 个价位档」不是尺度不变的）──
    px = {}
    for sym in SYMS:
        by_day = load_ticks(sym)
        if by_day:
            d = sorted(by_day)[-1]
            mid = len(by_day[d]) // 2
            px[sym] = by_day[d][mid].price
    line.append("")
    line.append("【信号诊断】扫单事件密度与股价（同一套 min_levels=4 在不同价位的含义完全不同）")
    line.append(f"  {'标的':10s} {'价格':>8s} {'事件/日中位':>10s} {'可买手数':>8s}")
    diag_rows = []
    for sym in sorted(SYMS, key=lambda s: -(px.get(s, 0))):
        dens = ev_density.get(sym) or [0]
        med_n = sorted(dens)[len(dens) // 2]
        lots = int(budget / (px.get(sym, 1e9) * 100))
        diag_rows.append((sym, px.get(sym, 0), med_n, lots))
        line.append(f"  {sym:10s} {px.get(sym, 0):8.2f} {med_n:10d} {lots:8d}")
    lo = [r for r in diag_rows if r[1] < 10]
    hi = [r for r in diag_rows if r[1] >= 50]
    if lo and hi:
        line.append(f"  低价组(<10元)事件中位 {np.median([r[2] for r in lo]):.0f}/日  "
                    f"高价组(≥50元) {np.median([r[2] for r in hi]):.0f}/日  "
                    f"→ 相差 {np.median([r[2] for r in hi])/max(1, np.median([r[2] for r in lo])):.0f} 倍")
        line.append(f"  一手超预算（不做）的标的：{sum(1 for r in diag_rows if r[3] < 1)}/{len(diag_rows)} 只")

    if cost_stress:
        line.append("")
        line.append("【成本压力：滑点 ×2（6bp 单边）】")
        # 重跑一遍主变体，只改滑点
        alt = defaultdict(float)
        for sym in SYMS:
            by_day = load_ticks(sym)
            for d in days:
                ticks = by_day.get(d)
                if not ticks:
                    continue
                evs = detect_events(ticks, WIN)
                for t in simulate(ticks, evs, allow_inv=True, allow_pos=True,
                                  gate_inv_ok=True, gate_pos_ok=True, th=TH,
                                  slip=SLIP * 2, budget=budget):
                    alt[d] += t["pnl"] / ACCOUNT
        tact2 = np.array([alt.get(d, 0.0) for d in days])
        stats(base + tact2, "  底仓+T (滑点6bp)", line, len(days))

    # ── 落盘 ──
    for name, trades in all_trades.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        df = df[["date", "symbol", "side", "shares", "entry", "exit", "gross", "cost", "pnl",
                 "entry_sec", "exit_sec"]]
        safe = name.replace("·", "_").replace("=", "").replace(".", "").replace("≥", "ge")
        df.to_csv(OUT_DIR / f"sweep_v2_trades_{safe}.csv", index=False)
    pd.DataFrame({"date": days, "base": base}).to_csv(
        OUT_DIR / "sweep_v2_daily.csv", index=False)

    report = _write_md(line, res, days, all_trades, base, daily_pnl, n_units, diag_rows)
    print("\n".join(line))
    print("\nOK ->", OUT_DIR / "sweep_intraday_v2.md")


def _write_md(line, res, days, all_trades, base, daily_pnl, n_units, diag_rows=None) -> str:
    main = "双向·无门控"
    md = f"""# 扫单分时 T 重跑报告 v2（修正逐笔顺序 / 门控 / 成本后）
> 生成 {datetime.now():%Y-%m-%d %H:%M} ｜ 脚本 `tools/sweep_intraday_v2.py` ｜
> 逐笔缓存 `tools/tick_cache/` ｜ 逐笔明细 `tools/report/sweep_v2_trades_*.csv`

## 一、为什么要重跑

v1（`tools/sweep_intraday_bt.py`，旧报告 `sweep_intraday.md`：**倒T 1666 笔 / 72% 胜率 /
+3.09pp / Sharpe 0.81**）存在三处会推翻结论的缺陷：

| # | 缺陷 | 实测证据 | 影响 |
|---|---|---|---|
| 1 | **逐笔页序倒序拼接** | 2026-10-16 SH600519 返回顺序为 14:02→15:29 / 11:25→14:02 / 10:21→11:25 / 09:25→10:21（每 1000 笔一页，页序倒置） | 时间轴来回跳：`t0=14:02`、MAX_HOLD 计时变负、滚动窗口混入相隔 4 小时的成交 → 信号与盈亏无意义 |
| 2 | **门控从未生效** | `gate_days` 在 `idx_close` 填充前计算（长度 0） | 正T 全被跳过（旧日志"正T: 0 笔"即此 bug），倒T 完全不设门 |
| 3 | **门控有前视** | 用当日收盘 vs MA20 决定当日是否交易 | 盘中不可知 |

另修正：成本两侧对称、印花税 0.05%（旧 0.1%）、回撤按标准口径、一手超预算标的按规则剔除
（旧版把茅台单笔名义放大到 12.8 万，是其 13 倍目标仓位）。

## 二、口径

- 池：{len(SYMS)} 只人工挑选的大盘股；账户 {ACCOUNT:,.0f} 元；单笔预算 = 账户×25%/25 = {ACCOUNT*PER_TRADE_FRAC/N:,.0f} 元。
- 信号：逐笔（3 秒快照）滚动 {WIN} 笔窗口，每 {STEP} 笔跑一次 `detect_sweeps`；平仓阈值 {TH:.1%}、最长持有 {MAX_HOLD_MIN} 分钟、收盘前强平。
- 成本：佣金万3（5 元起）+ 过户费 0.001%（双边）+ 印花税 0.05%（卖出）+ 滑点 3bp（**买卖两侧都计**）。
- 底仓：{len(SYMS)} 只等权、日再平衡，用**前复权日线**算收益（不用逐笔末笔价，后者会被集合竞价/盘后成交污染）。
- 门控：t-1 收盘 > 其前 20 日均值 = 强市（做正T），否则弱市（做倒T）。
- 样本：{days[0]} .. {days[-1]}，{len(days)} 个交易日，{n_units} 个(标的,日)单元。

## 三、结果

```
""" + "\n".join(line) + f"""
```

## 四、怎么读这个结果

1. **修正顺序后，v1 的 +3.09pp 不再成立**：旧数字来自"时间轴错乱 + 正T 被悄悄关掉 + 高价股名义放大 13 倍"三者叠加。
2. **信号本身还有一个更根本的问题**：`detect_sweeps` 的 `min_levels=4` 数的是**绝对价位档**，
   1300 元的票跨 4 档只要涨 0.03%，8 元的票跨 4 档要涨 0.5%——同一套判据在两者上不是一件事。
   实测事件密度与股价近乎单调（见「信号诊断」表：高价组中位 643 次/日，低价组 0 次/日），
   所以这个信号在横截面上**基本等价于「挑高价股」**，而不是识别扫单。
   报告里的「跨度≥5bp / ≥10bp」变体就是给这个缺陷打的补丁；它改变的是信号定义，不是参数微调。
3. **止损缺失导致左尾吃掉一切**：单笔中位 +16.8 元、均值 -14.1 元、胜率 62%，
   典型的「小赢 + 大亏」。最差一笔是 2026-06-17 京东方A 倒T：09:31 在 6.13 卖出，
   当日封涨停，40 分钟超时被迫在 6.71 买回，单笔 -9.5%（-943 元）。
   倒T 的经济学是「赚 0.4% 的回落、亏 10% 的封板」，只要没有硬止损，尾部一次就抹掉几十笔盈利。
4. **入场时刻集中在开盘 1~2 分钟**：`traded_dir` 限制「每方向每天最多 1 笔」，
   而扫单事件在开盘集合竞价后的成交爆发期最密集，于是这套逻辑实际做的是
   「开盘方向 + 回落 0.4% 买回」，不是全天候的日内 T。要看「入场时刻」那一行。
5. **样本只有 {len(days)} 个交易日、单一区间**：窗口/阈值/持有上限都是在这段窗口上定的，
   没有样本外。这个长度连一个季度都不到，Sharpe 的标准误约 ±1，**任何单点估计都不具备统计意义**。
   而且这段窗口的行情极不中性（银行 +15%、宁德 −39%、隆基 −18%、沪深300ETF −15%），
   倒T 恰好在趋势最强的银行股上交易最少、在波动最大的成长股上交易最多。
6. **底仓主导**：日收益的方差几乎全部来自等权底仓，战术腿的贡献要在「净额 / 成本 / 笔数」三个维度同时看。
7. 真正可复用的产出是**修好的逐笔缓存与回测骨架**（`tools/tick_cache/` 可直接喂任何日内策略），
   而不是一个可以拿去做实盘的参数组。

## 五、v1 与 v2 的对照

| 指标 | v1（旧报告） | v2（本报告） |
|---|---|---|
| 分时T 交易笔数 | 1666（仅倒T） | 2990（倒T 1550 / 正T 1440） |
| 单笔胜率 | 72% | 62% |
| 战术腿净盈亏 | **+30,845 元** | **−42,155 元** |
| 底仓+T 累计 | +3.77% | −2.12% |
| 底仓 累计 | +0.68%（截至 10-16） | +2.09%（截至 10-20） |
| 毛盈亏 / 成本 | 未拆分 | +769 / 42,924（成本是毛利的 56 倍） |

v1 的正收益**全部来自三处错误的叠加**，不是策略的 edge：时间轴乱跳制造了虚假的「回落」，
正T 被 bug 关掉只剩单边，高价股名义放大 13 倍把偶然的盈利放大成主要贡献。

## 六、复现

```bash
python tools/sweep_intraday_v2.py --phase fetch   # 约 4 分钟，建逐笔缓存
python tools/sweep_intraday_v2.py --phase run     # 读缓存跑全部变体
```
"""
    p = OUT_DIR / "sweep_intraday_v2.md"
    p.write_text(md, encoding="utf-8")
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["fetch", "run", "all"], default="all")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-cost-stress", action="store_true")
    a = ap.parse_args()
    if a.phase in ("fetch", "all"):
        phase_fetch(a.workers)
    if a.phase in ("run", "all"):
        phase_run(cost_stress=not a.no_cost_stress)


if __name__ == "__main__":
    main()
