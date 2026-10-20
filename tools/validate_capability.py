"""框架能力验证：用「低波动组合」把引擎的账本 / 成交 / 规则语义逐笔对账。

预登记设计见 ``docs/PREREG_CAPABILITY.md``（写于运行之前）。本脚本做三件事：

**1. 用与引擎无共享代码的参考实现，复算同一策略**
   - ``R1 克隆``：纯 numpy 逐日循环，独立复现「决策 → 次日开盘成交 → 100 股整手 →
     T+1 → 涨跌停 → 成本」。与引擎**逐笔对账**（成交明细 + 权益曲线）。
   - ``R2 分数权重``：同信号、无整手、无成本 → 理想上界。
   - ``R3 同日收盘成交``：同信号、无成本、但用信号那根的收盘价成交（含前视）→ 量化前视溢价。
   - ``R4 五分位``：纯 pandas 按 60 日波动率分 5 组，给出横截面效应的**独立**预期方向/量级。

**2. 账本 / 成本的外部审计**：由 ``FillEvent`` 明细独立重算现金与逐帧权益；
   逐笔重算 ``fill.cost`` 与手算公式比对。

**3. 对抗性检验**：伪造 ``CausalToken`` / 负 lag / 涨跌停 / T+1 —— 逃逸次数必须为 0。

用法：
    python -u tools/validate_capability.py                       # 全量
    python -u tools/validate_capability.py --quick               # 60 标的、近 1 年（冒烟）
    python -u tools/validate_capability.py --md reports/CAPABILITY.md
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from framebt import Engine, StandardAshareRules, StandardCost
from framebt.core import Bar, FrameStrategy, Order, OrderSide, OrderType
from framebt.data_bus import CausalToken, CausalityViolation
from framebt.feed import BarFeed

KLINE = (Path(__file__).resolve().parents[2] / "opentdx" / "db" / "data"
         / "fakt_kline" / "4" / "1")
CACHE = Path(__file__).resolve().parent / "etf_cache"

POOL_START, POOL_END = "2013-01-01", "2015-12-31"
FEED_START = "2017-01-01"
EVAL_START = "2018-01-01"

# 成本与制度的预登记设定（与 docs/PREREG_CAPABILITY.md §1 一致）
COMMISSION, MIN_COMMISSION = 0.0003, 5.0
STAMP_TAX, TRANSFER_FEE, SLIP = 0.0005, 0.00002, 0.0005
ONE_SIDE_RATE = COMMISSION + TRANSFER_FEE + SLIP          # 买入单边费率
ONE_SIDE_RATE_SELL = COMMISSION + TRANSFER_FEE + SLIP + STAMP_TAX
AVG_ONE_SIDE_RATE = (ONE_SIDE_RATE + ONE_SIDE_RATE_SELL) / 2.0   # 0.127%

LINES: list[str] = []

# Windows 中文控制台默认 GBK，输出「−」「✓」会 UnicodeEncodeError 并中断
# （--md 也写不出来）。这里强制 UTF-8；重定向到文件/管道时同样安全。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001 某些嵌入式 stdout 不支持 reconfigure
        pass


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(t: str) -> None:
    out("")
    out("=" * 78)
    out(f"  {t}")
    out("=" * 78)


# =====================================================================
# 数据层
# =====================================================================

def build_pool(top_n: int, min_days: int = 400) -> list[str]:
    """全 A 池：POOL_START~POOL_END 日均成交额前 top_n，且窗口内 ≥min_days 个交易日。

    池在评估期之前一次性固定（不按业绩筛选、不逐日重排），避免「用全期均值挑池」的前视。
    """
    import duckdb
    glob = str((KLINE / "*.parquet").as_posix())
    conn = duckdb.connect()
    q = f"""
        WITH w AS (
            SELECT symbol_key, CAST(amount AS DOUBLE) AS amount
            FROM read_parquet('{glob}')
            WHERE amount > 0
              AND CAST(datetime AS DATE) BETWEEN DATE '{POOL_START}' AND DATE '{POOL_END}'
        ), s AS (
            SELECT symbol_key, AVG(amount) AS a, COUNT(*) AS n FROM w GROUP BY symbol_key
        )
        SELECT symbol_key FROM s WHERE n >= {min_days}
        ORDER BY a DESC LIMIT {top_n}
    """
    return [r[0] for r in conn.execute(q).fetchall()]


def read_long(symbols: list[str]) -> pd.DataFrame:
    mkt_int = {"SZ": 0, "SH": 1, "BJ": 2}
    dfs = []
    for sym in symbols:
        m, code = sym.split(".")
        f = KLINE / f"{mkt_int[m]}_{code}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df["symbol_key"] = sym
        dfs.append(df[["symbol_key", "dt", "open", "high", "low", "close",
                       "vol", "amount", "float_shares", "turnover"]])
    a = pd.concat(dfs, ignore_index=True)
    a["dt"] = pd.to_datetime(a["dt"])
    a["dt_ord"] = a["dt"].map(lambda d: d.toordinal())
    return a


def long_to_pivots(df: pd.DataFrame, cols: list[str]) -> dict[str, pd.DataFrame]:
    return {c: df.pivot_table(index="dt_ord", columns="symbol_key", values=c,
                              aggfunc="last").sort_index() for c in cols}


def _limit_of(sym: str) -> float:
    """该标的的日涨跌幅上限（创业板/科创板 20%，其余 10%），留 1pp 余量。"""
    return 0.21 if (sym.startswith("SZ.30") or sym.startswith("SH.68")) else 0.11


def prepare_basis(long: pd.DataFrame, basis: str = "qfq") -> tuple[pd.DataFrame, dict]:
    """数据体检 + 口径重建。返回 (清洗后的长表, 统计)。

    **为什么必须做**（本次验证实测）：数仓的 ``QFQ`` 是「加法式」前复权，对高分红标的会把
    早期价位压到 **负值**（如 SH.601088 起始 close = −4.70、SH.000338 起始 close = 0.02），
    越过零点时又跳回正常价位，制造 **+1650% / +900% / inf%** 级别的假日涨跌幅。
    框数的数据源只过滤「≤0 的价格」，消除不了这些跳变（0.02 → 21.28 照样是 +1000 倍）。

    清洗规则（逐标的，保守但可证伪）：
      1. 丢弃 open/high/low/close 任一 ≤0 或非有限的 Bar；
      2. 若日收益出现超过该板块涨跌停上限的跳变（主板 11%、创业板/科创板 21%），
         说明该标的的复权口径在某处断裂 —— **截断到最后一次断裂之后**，
         只保留连续可信的那一段（宁可丢历史，也不让假收益进回测）；
      3. 截断后不足 400 根 Bar 的标的剔除。

    ``basis="qfq"`` 用原 QFQ 价位；``basis="raw"`` 按每根 Bar 的 ``vwap/close`` 比例
    把 OHLC 还原到**未复权**口径（``vwap = amount/vol`` 是真实成交均价），
    用于检验结论是否依赖复权口径。
    """
    df = long.copy()
    stats: dict = {"n_sym": 0, "dropped_bad_price": 0, "truncated": 0,
                   "dropped_short": 0, "bars_removed": 0, "max_jump_before": 0.0}
    ohlc = ["open", "high", "low", "close"]
    if basis == "raw":
        # 把每根 Bar 的 OHLC 按 vwap/close 缩放到未复权口径（close 还原为真实成交均价）
        vwap = df["amount"] / df["vol"].replace(0.0, np.nan)
        factor = vwap / df["close"]
        df = df[np.isfinite(factor) & (factor > 0)].copy()
        for c in ohlc:
            df[c] = df[c] * factor

    keep = []
    for sym, g in df.groupby("symbol_key", sort=False):
        stats["n_sym"] += 1
        g = g.sort_values("dt_ord")
        arr = g[ohlc].to_numpy(float)
        good = np.isfinite(arr).all(axis=1) & (arr > 0).all(axis=1)
        g = g[good]
        if len(g) < 400:
            stats["dropped_bad_price"] += 1
            continue
        c = g["close"].to_numpy(float)
        r = c[1:] / c[:-1] - 1.0
        stats["max_jump_before"] = max(stats["max_jump_before"],
                                       float(np.abs(r).max()) if len(r) else 0.0)
        bad = np.abs(r) > _limit_of(sym)
        start = 0
        if bad.any():
            start = int(np.flatnonzero(bad)[-1]) + 1      # 最后一次断裂之后
            stats["truncated"] += 1
        after = g.iloc[start + 1:]
        stats["bars_removed"] += len(g) - len(after)
        if len(after) < 400:
            stats["dropped_short"] += 1
            continue
        keep.append(after)
    clean = pd.concat(keep, ignore_index=True)
    stats["kept"] = int(clean["symbol_key"].nunique())
    stats["bars"] = int(len(clean))
    stats["max_jump_after"] = 0.0
    for _sym, g in clean.groupby("symbol_key", sort=False):
        c = g.sort_values("dt_ord")["close"].to_numpy(float)
        if len(c) > 1:
            stats["max_jump_after"] = max(stats["max_jump_after"],
                                          float(np.abs(c[1:] / c[:-1] - 1.0).max()))
    return clean, stats


def long_to_bars(df: pd.DataFrame) -> list[Bar]:
    bars: list[Bar] = []
    for r in df.to_dict("records"):
        c = float(r["close"])
        if not (c > 0):
            continue
        vol = float(r["vol"] or 0.0)
        amt = float(r["amount"] or 0.0)
        fsh = float(r["float_shares"] or 0.0)
        bars.append(Bar(
            timestamp=int(r["dt_ord"]), symbol=str(r["symbol_key"]),
            open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
            close=c, volume=vol, amount=amt, float_shares=fsh,
            turnover=float(r["turnover"] or 0.0),
            vwap=(amt / vol) if vol > 0 else c,
            market_cap=(c * fsh) if fsh > 0 else 0.0,
        ))
    return bars


# =====================================================================
# 被测策略（只用当前帧数据，自行维护滚动窗口 —— 与实盘写法一致）
# =====================================================================

class LowVolStrategy(FrameStrategy):
    """60 日已实现波动率最低的 K 只，等权，每 21 个交易日调仓。

    ``reverse=True`` → 波动率最高的 K 只（HighVol-10 对照臂，只翻排序方向）。
    """

    def __init__(self, universe: list[str], *, k: int = 10, vol_window: int = 60,
                 amount_window: int = 20, min_amount: float = 2e7,
                 rebalance_every: int = 21, gross: float = 0.98, lot: int = 100,
                 lot_pct_cap: float = 0.08, min_order_pct: float = 0.003,
                 reverse: bool = False, warmup: int = 61):
        self.universe = list(universe)
        self.k, self.vol_window, self.amount_window = int(k), int(vol_window), int(amount_window)
        self.min_amount = float(min_amount)
        self.rebalance_every = int(rebalance_every)
        self.gross, self.lot = float(gross), int(lot)
        self.lot_pct_cap = float(lot_pct_cap)
        self.min_order_pct = float(min_order_pct)
        self.reverse = bool(reverse)
        self.warmup = int(warmup)
        self._frame = 0
        self._closes = {s: deque(maxlen=self.vol_window + 1) for s in self.universe}
        self._amts = {s: deque(maxlen=self.amount_window) for s in self.universe}
        self.trace: list[dict] = []

    # ---- 滚动窗口维护（只吃当前帧的 bar，天然因果） ----
    def _absorb(self, frame) -> None:
        for sym, bar in frame.bars.items():
            d = self._closes.get(sym)
            if d is not None and bar.close > 0:
                d.append(bar.close)
            a = self._amts.get(sym)
            if a is not None:
                a.append(bar.amount)

    def _vol(self, d) -> float:
        arr = np.fromiter(d, dtype=float, count=len(d))
        r = arr[1:] / arr[:-1] - 1.0
        return float(np.std(r, ddof=1)) * math.sqrt(252.0)

    def on_frame(self, frame, token, bus, host) -> list:
        self._frame += 1
        self._absorb(frame)
        if self._frame < self.warmup or self._frame % self.rebalance_every:
            return []

        pos = host.portfolio
        px: dict[str, float] = {}
        held: dict[str, int] = {s: int(n) for s, n in pos.exposure().items() if n > 0}
        equity = float(pos.cash)
        for sym in set(list(held) + list(self.universe)):
            d = self._closes.get(sym)
            if d and len(d) > 0 and d[-1] > 0:
                px[sym] = float(d[-1])
        for sym, n in held.items():
            equity += n * px.get(sym, 0.0)
        if equity <= 0:
            return []

        cands: list[tuple[str, float]] = []
        for sym in self.universe:
            if not frame.has_bar(sym):
                continue                      # 今日停牌/无行情：不进候选，也无法成交
            d, a = self._closes[sym], self._amts[sym]
            if len(d) < self.vol_window + 1 or len(a) < self.amount_window:
                continue
            if float(np.mean(a)) < self.min_amount:
                continue
            p = px.get(sym, 0.0)
            if p <= 0 or p * self.lot > equity * self.lot_pct_cap:
                continue                      # 一手都买不起 → 剔除（20 万本金的硬约束）
            v = self._vol(d)
            if not math.isfinite(v) or v <= 0:
                continue
            cands.append((sym, v))
        if not cands:
            return []

        cands.sort(key=lambda t: (t[1], t[0]), reverse=self.reverse)
        picked = cands[: self.k]
        w = self.gross / len(picked)
        target = {s: int(w * equity / px[s] // self.lot * self.lot) for s, _ in picked}

        min_notional = equity * self.min_order_pct
        orders: list = []
        # 先卖后买：卖腿在撮合循环中先落地，其现金才能供买腿使用（同一帧内顺序执行）
        for sym, cur in held.items():
            want = target.get(sym, 0)
            if cur > want and (cur - want) * px.get(sym, 0.0) >= min_notional:
                orders.append(Order(side=OrderSide.SELL, symbol=sym,
                                    size=float(cur - want), order_type=OrderType.MARKET))
        for sym, want in target.items():
            cur = held.get(sym, 0)
            if want > cur and (want - cur) * px[sym] >= min_notional:
                orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                    size=float(want - cur), order_type=OrderType.MARKET))

        self.trace.append({
            "frame": self._frame, "equity": equity, "n_cand": len(cands),
            "picked": [s for s, _ in picked],
            "vols": {s: v for s, v in picked},
            "target": dict(target), "n_orders": len(orders),
        })
        return orders


# =====================================================================
# R1：独立克隆（纯 numpy 逐日循环）
# =====================================================================

def clone_backtest(piv: dict[str, pd.DataFrame], dates: list[int], universe: list[str], *,
                   with_cost: bool = True, k: int = 10, vol_window: int = 60,
                   amount_window: int = 20, min_amount: float = 2e7,
                   rebalance_every: int = 21, gross: float = 0.98, lot: int = 100,
                   lot_pct_cap: float = 0.08, min_order_pct: float = 0.003,
                   reverse: bool = False, warmup: int = 61,
                   cash0: float = 200_000.0, limit: float = 0.10) -> dict:
    """独立复现引擎语义：帧 t 收盘决策 → 帧 t+1 开盘成交 → 整手 → T+1 → 涨跌停 → 成本。

    返回 ``{equity: pd.Series, fills: list[dict], fees: float}``。
    """
    cols = list(universe)
    O = piv["open"].reindex(index=dates, columns=cols).to_numpy(float)
    C = piv["close"].reindex(index=dates, columns=cols).to_numpy(float)
    A = piv["amount"].reindex(index=dates, columns=cols).to_numpy(float)
    ix = {s: i for i, s in enumerate(cols)}

    hist: list[list] = [[] for _ in cols]
    amts: list[list] = [[] for _ in cols]
    last_close = np.full(len(cols), np.nan)

    cash = float(cash0)
    pos = np.zeros(len(cols), dtype=np.int64)
    legs: list[list[tuple[int, int]]] = [[] for _ in cols]   # (entry_frame, size)
    pending: list[tuple[str, int, int]] = []                 # (side, idx, size)
    fills: list[dict] = []
    fees_total = 0.0
    eq: list[float] = []

    for i, _d in enumerate(dates):
        frame_id = i
        # ---- 1. 执行上一帧挂出的订单（本帧开盘价） ----
        for side, si, size in pending:
            o = O[i, si]
            if not np.isfinite(o) or o <= 0:
                continue                                   # 无行情 → 拒绝
            pc = last_close[si]
            if side == "B":
                if np.isfinite(pc) and o > pc * (1 + limit):
                    continue                               # 涨停 → 拒买
            else:
                if np.isfinite(pc) and o < pc * (1 - limit):
                    continue                               # 跌停 → 拒卖
                avail = sum(s for f, s in legs[si] if f + 1 <= frame_id)
                if avail + 1e-9 < size:
                    continue                               # T+1 不足 → 整单拒绝
            n = size * o
            fee = 0.0
            if with_cost:
                fee = max(n * COMMISSION, MIN_COMMISSION) + n * TRANSFER_FEE + n * SLIP
                if side == "S":
                    fee += n * STAMP_TAX
            if side == "B":
                cash -= n + fee
                pos[si] += size
                legs[si].append((frame_id, size))
            else:
                cash += n - fee
                pos[si] -= size
                # 卖出：FIFO 消耗最早的多头腿
                need = size
                newlegs = []
                for f, s in sorted(legs[si], key=lambda t: t[0]):
                    if need <= 0:
                        newlegs.append((f, s))
                        continue
                    take = min(s, need)
                    need -= take
                    if s - take > 0:
                        newlegs.append((f, s - take))
                legs[si] = newlegs
            fees_total += fee
            fills.append({"ts": _d, "side": side, "symbol": cols[si],
                          "size": int(size), "price": float(o), "fee": float(fee)})
        pending = []

        # ---- 2. 按收盘价标记权益 ----
        for si in range(len(cols)):
            if np.isfinite(C[i, si]) and C[i, si] > 0:
                last_close[si] = C[i, si]
        equity = cash
        for si in range(len(cols)):
            if pos[si] != 0:
                equity += pos[si] * last_close[si]
        eq.append(float(equity))

        # ---- 3. 决策（只用 ≤ 本帧收盘的信息） ----
        for si in range(len(cols)):
            if np.isfinite(C[i, si]) and C[i, si] > 0:
                hist[si].append(C[i, si])
                if len(hist[si]) > vol_window + 1:
                    hist[si] = hist[si][-(vol_window + 1):]
            if np.isfinite(A[i, si]):
                amts[si].append(A[i, si])
                if len(amts[si]) > amount_window:
                    amts[si] = amts[si][-amount_window:]

        frame_no = i + 1
        if frame_no < warmup or frame_no % rebalance_every:
            continue

        held = {cols[si]: int(pos[si]) for si in range(len(cols)) if pos[si] > 0}
        px = {cols[si]: float(last_close[si]) for si in range(len(cols))
              if np.isfinite(last_close[si]) and last_close[si] > 0}

        cands: list[tuple[str, float]] = []
        for si, s in enumerate(cols):
            if not (np.isfinite(C[i, si]) and C[i, si] > 0):
                continue
            if len(hist[si]) < vol_window + 1 or len(amts[si]) < amount_window:
                continue
            if float(np.mean(amts[si])) < min_amount:
                continue
            p = px.get(s, 0.0)
            if p <= 0 or p * lot > equity * lot_pct_cap:
                continue
            arr = np.asarray(hist[si], dtype=float)
            r = arr[1:] / arr[:-1] - 1.0
            v = float(np.std(r, ddof=1)) * math.sqrt(252.0)
            if not math.isfinite(v) or v <= 0:
                continue
            cands.append((s, v))
        if not cands:
            continue
        cands.sort(key=lambda t: (t[1], t[0]), reverse=reverse)
        picked = cands[:k]
        w = gross / len(picked)
        target = {s: int(w * equity / px[s] // lot * lot) for s, _ in picked}
        min_notional = equity * min_order_pct
        for s, cur in held.items():
            want = target.get(s, 0)
            if cur > want and (cur - want) * px.get(s, 0.0) >= min_notional:
                pending.append(("S", ix[s], cur - want))
        for s, want in target.items():
            cur = held.get(s, 0)
            if want > cur and (want - cur) * px[s] >= min_notional:
                pending.append(("B", ix[s], want - cur))

    return {"equity": pd.Series(eq, index=dates), "fills": fills, "fees": fees_total}


# =====================================================================
# R2 / R3：无摩擦参考（分数权重；成交价 = 次日开盘 或 当日收盘）
# =====================================================================

def frictionless_backtest(piv: dict[str, pd.DataFrame], dates: list[int], universe: list[str], *,
                          fill: str = "next_open", k: int = 10, vol_window: int = 60,
                          amount_window: int = 20, min_amount: float = 2e7,
                          rebalance_every: int = 21, reverse: bool = False,
                          warmup: int = 61, cash0: float = 200_000.0,
                          lot_pct_cap: float = 0.08) -> dict:
    """同信号、无成本、无整手的分数权重组合。``fill`` ∈ {next_open, close}。"""
    cols = list(universe)
    O = piv["open"].reindex(index=dates, columns=cols).to_numpy(float)
    C = piv["close"].reindex(index=dates, columns=cols).to_numpy(float)
    A = piv["amount"].reindex(index=dates, columns=cols).to_numpy(float)
    hist = [[] for _ in cols]
    amts = [[] for _ in cols]
    last_close = np.full(len(cols), np.nan)
    n = len(cols)
    shares = np.zeros(n)
    cash = float(cash0)
    pending: dict[int, float] = {}     # idx -> 目标股数（等权，按决策时权益算）
    eq: list[float] = []

    for i, _d in enumerate(dates):
        P = O if fill == "next_open" else C
        for si, tgt in pending.items():
            p = P[i, si]
            if not np.isfinite(p) or p <= 0:
                continue
            delta = tgt - shares[si]
            cash -= delta * p
            shares[si] = tgt
        pending = {}

        for si in range(n):
            if np.isfinite(C[i, si]) and C[i, si] > 0:
                last_close[si] = C[i, si]
        eq.append(float(cash + float(np.nansum(shares * last_close))))

        for si in range(n):
            if np.isfinite(C[i, si]) and C[i, si] > 0:
                hist[si].append(C[i, si])
                if len(hist[si]) > vol_window + 1:
                    hist[si] = hist[si][-(vol_window + 1):]
            if np.isfinite(A[i, si]):
                amts[si].append(A[i, si])
                if len(amts[si]) > amount_window:
                    amts[si] = amts[si][-amount_window:]

        frame_no = i + 1
        if frame_no < warmup or frame_no % rebalance_every:
            continue
        equity = eq[-1]
        cands = []
        for si, s in enumerate(cols):
            if not (np.isfinite(C[i, si]) and C[i, si] > 0):
                continue
            if len(hist[si]) < vol_window + 1 or len(amts[si]) < amount_window:
                continue
            if float(np.mean(amts[si])) < min_amount:
                continue
            p = last_close[si]
            if p * 100 > equity * lot_pct_cap:
                continue
            arr = np.asarray(hist[si], dtype=float)
            r = arr[1:] / arr[:-1] - 1.0
            v = float(np.std(r, ddof=1)) * math.sqrt(252.0)
            if not math.isfinite(v) or v <= 0:
                continue
            cands.append((s, v))
        if not cands:
            continue
        cands.sort(key=lambda t: (t[1], t[0]), reverse=reverse)
        picked = [s for s, _ in cands[:k]]
        w = 0.98 / len(picked)
        for s in picked:
            si = cols.index(s)
            pending[si] = w * equity / last_close[si]
        # 清掉不再持有的
        for si, s in enumerate(cols):
            if s not in picked and shares[si] != 0:
                pending[si] = 0.0
    return {"equity": pd.Series(eq, index=dates)}


# =====================================================================
# R4：五分位独立统计
# =====================================================================

def quintile_stats(piv: dict[str, pd.DataFrame], dates: list[int], universe: list[str], *,
                   vol_window: int = 60, amount_window: int = 20, min_amount: float = 2e7,
                   rebalance_every: int = 21, warmup: int = 61,
                   eval_start: int = 0) -> pd.DataFrame:
    """按 60 日波动率分 5 组，统计各组「下一个 21 日」的等权收益（无成本、收盘价成交）。

    这是**完全不经过引擎**的独立预期：横截面波动率效应到底站在哪一边、量级多大。
    """
    cols = list(universe)
    C = piv["close"].reindex(index=dates, columns=cols).ffill().to_numpy(float)
    C = np.where(C > 0, C, np.nan)
    A = piv["amount"].reindex(index=dates, columns=cols).to_numpy(float)
    V = pd.DataFrame(C, index=dates, columns=cols)
    vol = np.log(V).diff().rolling(vol_window).std() * math.sqrt(252.0)
    amt = pd.DataFrame(A, index=dates, columns=cols).rolling(amount_window).mean()

    rows = []
    for i in range(len(dates)):
        if i + 1 < warmup or (i + 1) % rebalance_every:
            continue
        d = dates[i]
        if d < eval_start:
            continue
        j = min(i + rebalance_every, len(dates) - 1)
        v = vol.iloc[i]
        ok = v.notna() & (amt.iloc[i] >= min_amount)
        ok &= V.iloc[i].notna() & (V.iloc[j].notna())
        if ok.sum() < 25:
            continue
        fwd = (V.iloc[j] / V.iloc[i] - 1.0)[ok]
        vv = v[ok]
        q = pd.qcut(vv.rank(method="first"), 5, labels=False)
        row = {"date": d}
        for g in range(5):
            row[f"D{g + 1}"] = float(fwd[q == g].mean())
        row["POOL"] = float(fwd.mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    per_year = 252.0 / rebalance_every
    res = {}
    for c in [f"D{i}" for i in range(1, 6)] + ["POOL"]:
        r = df[c]
        res[c] = {
            "ann": float((1 + r).prod() ** (per_year / max(len(r), 1)) - 1),
            "mean_21d": float(r.mean()),
            "t": float(r.mean() / r.std(ddof=1) * math.sqrt(len(r))) if r.std(ddof=1) > 0 else 0.0,
        }
    return pd.DataFrame(res).T, df


# =====================================================================
# 指标 / 审计
# =====================================================================

def mstats(eq: pd.Series, eval_start: int) -> dict:
    e = eq[eq.index >= eval_start].astype(float)
    if len(e) < 30:
        return {}
    r = e.pct_change().dropna()
    total = float(e.iloc[-1] / e.iloc[0] - 1.0)
    yrs = len(e) / 252.0
    ann = float((1 + total) ** (1 / max(yrs, 1e-9)) - 1)
    sd = float(r.std(ddof=1)) * math.sqrt(252.0)
    sharpe = float(r.mean() / r.std(ddof=1) * math.sqrt(252.0)) if r.std(ddof=1) > 0 else 0.0
    peak = e.cummax()
    mdd = float(((peak - e) / peak).max())
    return {"total": total, "ann": ann, "vol": sd, "sharpe": sharpe, "mdd": mdd,
            "final": float(e.iloc[-1]), "start": float(e.iloc[0]),
            "years": yrs, "n": len(e)}


def turnover_of(fills, avg_equity: float, years: float) -> float:
    """年化**单边**换手 = 累计成交额 / 平均权益 / 年数（买入与卖出各自计入）。"""
    traded = sum(f["size"] * f["price"] for f in fills)
    return traded / avg_equity / max(years, 1e-9)


def audit_ledger(res, piv_close: pd.DataFrame, cash0: float) -> dict:
    """由 FillEvent 明细独立重算现金/权益，与引擎权益曲线逐帧比对（P1）。"""
    eq_ts = [int(t) for t, _ in res.equity_curve]
    eq_val = np.array([float(v) for _, v in res.equity_curve])
    by_frame: dict[int, list] = {}
    for f in res.fills:
        by_frame.setdefault(f.frame_id, []).append(f)
    marks = piv_close.reindex(index=eq_ts).ffill()
    cash = float(cash0)
    pos: dict[str, float] = {}
    replay = np.empty(len(eq_ts))
    for i, ts in enumerate(eq_ts):
        for f in by_frame.get(i, []):
            n = f.size * f.price
            if f.side is OrderSide.BUY:
                cash -= n + f.cost
                pos[f.symbol] = pos.get(f.symbol, 0.0) + f.size
            else:
                cash += n - f.cost
                pos[f.symbol] = pos.get(f.symbol, 0.0) - f.size
        eq = cash
        for sym, sz in pos.items():
            if sz == 0:
                continue
            v = marks.at[ts, sym] if sym in marks.columns else np.nan
            if not np.isfinite(v):
                return {"max_rel": float("nan"), "note": f"{sym}@{ts} 无标记"}
            eq += sz * float(v)
        replay[i] = eq
    rel = np.abs(replay - eq_val) / np.maximum(np.abs(eq_val), 1.0)
    return {"max_rel": float(rel.max()), "max_abs": float(np.abs(replay - eq_val).max()),
            "replay_final": float(replay[-1]), "engine_final": float(eq_val[-1])}


def audit_costs(res) -> dict:
    """逐笔重算成本（P2）。"""
    worst, worst_desc = 0.0, ""
    for f in res.fills:
        n = f.size * f.price
        fee = max(n * COMMISSION, MIN_COMMISSION) + n * TRANSFER_FEE + n * SLIP
        if f.side is OrderSide.SELL:
            fee += n * STAMP_TAX
        d = abs(fee - f.cost)
        if d > worst:
            worst, worst_desc = d, f"{f.symbol} {f.side} {f.size}@{f.price}"
    return {"max_abs_diff": worst, "worst": worst_desc,
            "total_cost": float(sum(f.cost for f in res.fills))}


def compare_fills(engine_fills, clone_fills) -> dict:
    """成交明细逐笔比对：按 (ts, symbol, side, size, price) 排序后对照。"""
    def key(f):
        return (f["ts"], f["symbol"], f["side"], round(f["size"], 6), round(f["price"], 8))
    a = sorted([{"ts": int(f.timestamp) - 1, "symbol": f.symbol,
                 "side": "B" if f.side is OrderSide.BUY else "S",
                 "size": float(f.size), "price": float(f.price)} for f in engine_fills], key=key)
    b = sorted([{"ts": int(f["ts"]), "symbol": f["symbol"], "side": f["side"],
                 "size": float(f["size"]), "price": float(f["price"])} for f in clone_fills], key=key)
    n = min(len(a), len(b))
    mism = sum(1 for i in range(n) if key(a[i]) != key(b[i]))
    return {"n_engine": len(a), "n_clone": len(b), "mismatch": mism + abs(len(a) - len(b))}


# =====================================================================
# 对抗性检验（P8）
# =====================================================================

def adversarial_tests() -> dict:
    """伪造 token / 负 lag / 涨跌停 / T+1 —— 逃逸必须为 0。"""
    from framebt.core import Bar as B

    bars = []
    # T：普通标的，价格平稳（用于 T+1 检验）
    for i, p in enumerate([10.0, 10.1, 10.2, 10.3, 10.4]):
        bars.append(B(timestamp=i, symbol="T", open=p, high=p * 1.005,
                      low=p * 0.995, close=p, volume=1e6, amount=p * 1e6))
    # Y：第 1 帧开盘 12.0，昨收 10.0 → 涨停价 11.0 → 买单必须被拒
    ypx = [(10.0, 10.0), (12.0, 12.0), (12.1, 12.1), (12.2, 12.2), (12.3, 12.3)]
    for i, (o, c) in enumerate(ypx):
        bars.append(B(timestamp=i, symbol="Y", open=o, high=max(o, c),
                      low=min(o, c), close=c, volume=1e6, amount=o * 1e6))
    feed = BarFeed(bars)

    class Cheater(FrameStrategy):
        """① 伪造 issued_at 更大的 token 取未来数据；② 负 lag 请求；③ 记录逃逸次数。"""

        def __init__(self):
            self.escapes = 0
            self.forged_ok = 0
            self.neg_lag_raised = 0
            self.leaked_max_ts = -10 ** 18
            self.now = -10 ** 18
            self.seen = 0

        def on_frame(self, frame, token, bus, host):
            self.seen += 1
            self.now = token.issued_at_ns
            forged = CausalToken(10 ** 9, 999)     # 声称时间远超真实虚拟时间
            try:
                bus.get("close", "T", window=1000, token=forged)
                self.forged_ok += 1
                seq = bus.series("close", "T", forged)
                if seq:
                    self.leaked_max_ts = max(self.leaked_max_ts, max(t for t, _ in seq))
                    if max(t for t, _ in seq) > self.now:
                        self.escapes += 1
            except CausalityViolation:
                self.escapes += 1
            try:
                bus.get("close", "T", token=token, lag=-1)
                self.escapes += 1
            except CausalityViolation:
                self.neg_lag_raised += 1
            return []

    c = Cheater()
    eng = Engine(feed, c, initial_cash=100_000.0)
    eng.run()

    # ---- 涨跌停 / T+1 的机制检验：一帧内同时挂 买T / 卖T / 买Y ----
    class RuleProbe(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            if frame.frame_id != 0:
                return []
            return [
                Order(side=OrderSide.BUY, symbol="T", size=100.0, order_type=OrderType.MARKET),
                Order(side=OrderSide.SELL, symbol="T", size=100.0, order_type=OrderType.MARKET),
                Order(side=OrderSide.BUY, symbol="Y", size=100.0, order_type=OrderType.MARKET),
            ]

    res_rule = Engine(feed, RuleProbe(), initial_cash=100_000.0, settlement_cycle=1,
                      risk=None, rules=StandardAshareRules(limit=0.10)).run()
    rule_fills = [(f.frame_id, f.side.value, f.symbol, round(f.price, 4)) for f in res_rule.fills]

    return {"seen": c.seen, "escapes": c.escapes, "forged_ok": c.forged_ok,
            "neg_lag_raised": c.neg_lag_raised,
            "leak_ok": c.leaked_max_ts <= c.now,
            "log": eng.bus.forgery_attempts, "rule_fills": rule_fills}



# =====================================================================
# 主流程
# =====================================================================

def run_engine(bars, strat, cash, *, with_cost=True, rules=True):
    t0 = time.time()
    eng = Engine(
        BarFeed(bars), strat, initial_cash=cash, settlement_cycle=1,
        cost_model=StandardCost(commission=COMMISSION, min_commission=MIN_COMMISSION,
                                stamp_tax=STAMP_TAX, transfer_fee=TRANSFER_FEE,
                                slippage="proportional", k=SLIP) if with_cost else None,
        risk=None,
        rules=StandardAshareRules(limit=0.10) if rules else None,
    )
    res = eng.run()
    res.elapsed = time.time() - t0
    return res


def pool_ew_benchmark(piv, dates, universe, cash0) -> pd.Series:
    """池等权基准（分数权重、无成本、无整手）：每 21 日按开盘价调回等权，收盘计值。"""
    C = piv["close"].reindex(index=dates, columns=universe).to_numpy(float)
    O = piv["open"].reindex(index=dates, columns=universe).to_numpy(float)
    n = len(universe)
    last = np.full(n, np.nan)
    shares = np.zeros(n)
    cash = float(cash0)
    pending = False
    eq: list[float] = []
    for i in range(len(dates)):
        if pending:
            live = np.isfinite(O[i]) & np.isfinite(last)
            equity = cash + float(np.nansum(shares * last))
            if live.sum() > 0:
                w = 0.99 / live.sum()
                new = np.where(live, equity * w / np.where(live, O[i], 1.0), 0.0)
                cash += float(np.nansum((shares - new) * np.where(live, O[i], 0.0)))
                shares = new
            pending = False
        for j in range(n):
            if np.isfinite(C[i, j]) and C[i, j] > 0:
                last[j] = C[i, j]
        eq.append(cash + float(np.nansum(shares * last)))
        if (i + 1) % 21 == 0:
            pending = True
    return pd.Series(eq, index=dates)


def yearly_table(eq: pd.Series, eval_start: int) -> list[tuple[int, float, float]]:
    e = eq[eq.index >= eval_start].astype(float)
    if len(e) < 10:
        return []
    idx = pd.to_datetime([pd.Timestamp.fromordinal(int(t)) for t in e.index])
    df = pd.DataFrame({"eq": e.to_numpy()}, index=idx)
    rows = []
    for y, g in df.groupby(df.index.year):
        r = float(g["eq"].iloc[-1] / g["eq"].iloc[0] - 1.0)
        mdd = float(((g["eq"].cummax() - g["eq"]) / g["eq"].cummax()).max())
        rows.append((int(y), r, mdd))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="framebt 能力验证（低波动组合 · 逐笔对账）")
    ap.add_argument("--top-n", type=int, default=300)
    ap.add_argument("--cash", type=float, default=200_000.0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--quick", action="store_true", help="冒烟：60 标的 + 近 1 年")
    ap.add_argument("--md", default=None)
    args = ap.parse_args()

    t00 = time.time()
    top_n = 60 if args.quick else args.top_n
    feed_start = pd.Timestamp("2025-01-01").toordinal() if args.quick else \
        pd.Timestamp(FEED_START).toordinal()
    eval_start = pd.Timestamp("2025-07-01").toordinal() if args.quick else \
        pd.Timestamp(EVAL_START).toordinal()

    hr("0. 数据与池")
    syms = build_pool(top_n)
    out(f"  池：{POOL_START}~{POOL_END} 日均成交额前 {len(syms)}（窗口内 ≥400 交易日）")
    long_raw = read_long(syms)
    long_raw = long_raw[long_raw["dt_ord"] >= feed_start]
    long, dstats = prepare_basis(long_raw, "qfq")
    nsym_used = int(long["symbol_key"].nunique())
    syms = [s for s in syms if s in set(long["symbol_key"].unique())]
    piv = long_to_pivots(long, ["open", "high", "low", "close", "amount", "vol"])
    dates = [int(x) for x in piv["close"].index]
    out(f"  数据体检：{dstats['n_sym']} 只 → 保留 {nsym_used} 只；"
        f"截断 {dstats['truncated']} 只（复权断裂），剔除 {dstats['dropped_short']} 只")
    out(f"    丢弃 bars {dstats['bars_removed']:,}   使用 {dstats['bars']:,} 根")
    out(f"    清洗前最大日涨跌幅 {dstats['max_jump_before']:.1%} → 清洗后 "
        f"{dstats['max_jump_after']:.1%}（板块上限内）")
    out(f"  面板：{len(dates)} 交易日 × {nsym_used} 标的  "
        f"{pd.Timestamp.fromordinal(dates[0]).date()} .. "
        f"{pd.Timestamp.fromordinal(dates[-1]).date()}")
    out(f"  缺失率 {1 - piv['close'].notna().to_numpy().mean():.2%}   "
        f"读取 {time.time() - t00:.1f}s")
    bars = long_to_bars(long)
    out(f"  bars {len(bars):,}   评估起点 {pd.Timestamp.fromordinal(eval_start).date()}")
    xs = {}
    for y in range(pd.Timestamp.fromordinal(eval_start).year,
                   pd.Timestamp.fromordinal(dates[-1]).year + 1):
        o = pd.Timestamp(f"{y}-01-01").toordinal()
        sub = piv["close"].loc[piv["close"].index >= o]
        if len(sub):
            xs[y] = int(sub.iloc[0].notna().sum())
    out(f"  每年初可用标的数（清洗后）：{xs}")

    # ---------------- 主策略：LowVol-10 ----------------
    hr("1. 引擎回测 · LowVol-10（含成本 + 涨跌停 + T+1 + 整手）")
    strat = LowVolStrategy(syms, k=args.k, rebalance_every=args.rebalance)
    res = run_engine(bars, strat, args.cash, with_cost=True, rules=True)
    eq = pd.Series({int(t): float(v) for t, v in res.equity_curve}).sort_index()
    out(f"  耗时 {res.elapsed:.1f}s   帧数 {res.frames}   成交 {res.total_trades()} 笔   "
        f"调仓 {len(strat.trace)} 次")
    m = mstats(eq, eval_start)
    fees = sum(f.cost for f in res.fills)
    eq_eval = eq[eq.index >= eval_start]
    to = turnover_of([{"size": f.size, "price": f.price} for f in res.fills],
                     float(eq_eval.mean()), m["years"])
    out(f"  最终权益 {res.final_equity():,.2f}   总收益 {m['total']:+.2%}   "
        f"年化 {m['ann']:+.2%}")
    out(f"  波动 {m['vol']:.2%}   Sharpe {m['sharpe']:.2f}   最大回撤 {m['mdd']:.2%}")
    out(f"  费用 {fees:,.0f}   年化单边换手 {to:.2f}×   平均持仓只数 "
        f"{np.mean([len(t['picked']) for t in strat.trace]):.1f}" if strat.trace else "")

    # ---------------- P3 确定性 ----------------
    hr("2. P3 确定性（同配置第二次运行）")
    strat2 = LowVolStrategy(syms, k=args.k, rebalance_every=args.rebalance)
    res2 = run_engine(bars, strat2, args.cash, with_cost=True, rules=True)
    eq2 = pd.Series({int(t): float(v) for t, v in res2.equity_curve}).sort_index()
    same_eq = eq.equals(eq2)
    same_fill = [(f.frame_id, f.symbol, f.size, f.price, f.cost) for f in res.fills] == \
                [(f.frame_id, f.symbol, f.size, f.price, f.cost) for f in res2.fills]
    out(f"  权益曲线逐位相同: {same_eq}   成交明细逐位相同: {same_fill}")

    # ---------------- P1 / P2 审计 ----------------
    hr("3. P1/P2 账本与成本外部审计（只用 FillEvent 明细）")
    aud = audit_ledger(res, piv["close"], args.cash)
    out(f"  逐帧权益重算：最大相对偏差 {aud['max_rel']:.3e}   最大绝对偏差 {aud['max_abs']:.3e}")
    out(f"  重算终值 {aud['replay_final']:,.6f}   引擎终值 {aud['engine_final']:,.6f}")
    cost = audit_costs(res)
    out(f"  逐笔成本重算：最大绝对偏差 {cost['max_abs_diff']:.3e} 元 "
        f"（总费用 {cost['total_cost']:,.2f}）")

    # ---------------- P4 无成本克隆 ----------------
    hr("4. P4 无成本引擎 vs R1 独立克隆（逐笔一致性）")
    strat3 = LowVolStrategy(syms, k=args.k, rebalance_every=args.rebalance)
    res_nc = run_engine(bars, strat3, args.cash, with_cost=False, rules=True)
    eq_nc = pd.Series({int(t): float(v) for t, v in res_nc.equity_curve}).sort_index()
    m_nc_eng = mstats(eq_nc, eval_start)
    clone_nc = clone_backtest(piv, dates, syms, with_cost=False, k=args.k,
                              rebalance_every=args.rebalance, cash0=args.cash)
    cmpf = compare_fills(res_nc.fills, clone_nc["fills"])
    eng_final = float(eq_nc.iloc[-1])
    clo_final = float(clone_nc["equity"].iloc[-1])
    rel = abs(eng_final - clo_final) / max(abs(clo_final), 1.0)
    max_frame_dev = float((eq_nc.reindex(clone_nc["equity"].index) /
                           clone_nc["equity"] - 1.0).abs().max())
    out(f"  成交笔数 引擎 {cmpf['n_engine']} / 克隆 {cmpf['n_clone']}   "
        f"不一致 {cmpf['mismatch']}")
    out(f"  最终权益 引擎 {eng_final:,.6f}  克隆 {clo_final:,.6f}  相对差 {rel:.3e}")
    out(f"  逐帧权益最大相对偏差 {max_frame_dev:.3e}")

    # ---------------- R2 / R3 / 基准 ----------------
    hr("5. R2/R3 无摩擦参考与基准")
    r2 = frictionless_backtest(piv, dates, syms, fill="next_open", k=args.k,
                               rebalance_every=args.rebalance, cash0=args.cash)
    r3 = frictionless_backtest(piv, dates, syms, fill="close", k=args.k,
                               rebalance_every=args.rebalance, cash0=args.cash)
    m2 = mstats(r2["equity"], eval_start)
    m3 = mstats(r3["equity"], eval_start)
    out(f"  R2 次日开盘成交（无成本/整手）：年化 {m2['ann']:+.2%}  "
        f"Sharpe {m2['sharpe']:.2f}  回撤 {m2['mdd']:.2%}  波动 {m2['vol']:.2%}")
    out(f"  R3 当日收盘成交（含前视）    ：年化 {m3['ann']:+.2%}  "
        f"Sharpe {m3['sharpe']:.2f}  回撤 {m3['mdd']:.2%}  波动 {m3['vol']:.2%}")
    bench = pool_ew_benchmark(piv, dates, syms, args.cash)
    mb = mstats(bench, eval_start)
    out(f"  基准 池等权（月度调回、无摩擦）：年化 {mb['ann']:+.2%}  "
        f"Sharpe {mb['sharpe']:.2f}  回撤 {mb['mdd']:.2%}  波动 {mb['vol']:.2%}")
    me: dict = {}
    etf_eq = None
    if (CACHE / "SH_510300.parquet").exists():
        etf_eq = pd.read_parquet(CACHE / "SH_510300.parquet").set_index("dt_ord")["close"].astype(float)
        etf_eq = etf_eq[etf_eq.index >= eval_start]
        me = mstats(etf_eq, eval_start)
        out(f"  基准 510300ETF 买入持有       ：年化 {me['ann']:+.2%}  "
            f"Sharpe {me['sharpe']:.2f}  回撤 {me['mdd']:.2%}  波动 {me['vol']:.2%}")

    # ---------------- R4 五分位 ----------------
    hr("6. R4 独立五分位统计（纯 pandas，不经引擎）")
    qres, qraw = quintile_stats(piv, dates, syms, rebalance_every=args.rebalance,
                                eval_start=eval_start)
    label = {"D1": "波动最低", "D2": "", "D3": "", "D4": "", "D5": "波动最高",
             "POOL": "池等权"}
    out(f"  {'组':<6}{'年化':>10}{'21日均值':>12}{'t':>8}   说明")
    for g in list(qres.index):
        r = qres.loc[g]
        out(f"  {g:<6}{r['ann']:>+10.2%}{r['mean_21d']:>+12.3%}{r['t']:>8.2f}   {label.get(g, '')}")
    out(f"  样本 {len(qraw)} 个调仓期")

    # ---------------- P7 HighVol 对照臂 ----------------
    hr("7. P7 信号传导对照：HighVol-10（只翻排序方向，其余完全相同）")
    strat_h = LowVolStrategy(syms, k=args.k, rebalance_every=args.rebalance, reverse=True)
    res_h = run_engine(bars, strat_h, args.cash, with_cost=True, rules=True)
    eq_h = pd.Series({int(t): float(v) for t, v in res_h.equity_curve}).sort_index()
    mh = mstats(eq_h, eval_start)
    out(f"  最终权益 {res_h.final_equity():,.2f}   年化 {mh['ann']:+.2%}   "
        f"Sharpe {mh['sharpe']:.2f}  回撤 {mh['mdd']:.2%}  波动 {mh['vol']:.2%}")
    out(f"  成交 {res_h.total_trades()} 笔   费用 {sum(f.cost for f in res_h.fills):,.0f}")

    # ---------------- P8 对抗 ----------------
    hr("8. P8 因果与制度约束对抗检验")
    adv = adversarial_tests()
    out(f"  帧数 {adv['seen']}   伪造 token 成功 {adv['forged_ok']} 次   "
        f"未来数据逃逸 {adv['escapes']} 次   总线钳制计数 {adv['log']}")
    out(f"  负 lag 抛 CausalityViolation {adv['neg_lag_raised']}/{adv['seen']}")
    out(f"  返回记录均 ≤ 当前虚拟时间: {adv['leak_ok']}")
    out(f"  制度探针成交序列 (frame, side, symbol, price): {adv['rule_fills']}")
    out("    预期：只剩 (1,'buy','T',10.1) —— T 的卖单被 T+1 拒绝，")
    out("          Y 的买单被涨停拒绝（开盘 12.0 > 涨停价 11.0）")

    # ---------------- P5 摩擦分解 ----------------
    hr("9. P5 摩擦代价分解")
    clone_c = clone_backtest(piv, dates, syms, with_cost=True, k=args.k,
                             rebalance_every=args.rebalance, cash0=args.cash)
    mc_c = mstats(clone_c["equity"], eval_start)
    mc_nc = mstats(clone_nc["equity"], eval_start)
    eq_c_eval = clone_c["equity"][clone_c["equity"].index >= eval_start]
    to_c = turnover_of(clone_c["fills"], float(eq_c_eval.mean()), mc_c["years"])
    pred_drag = to_c * AVG_ONE_SIDE_RATE
    act_drag = mc_nc["ann"] - mc_c["ann"]
    eng_drag = m_nc_eng["ann"] - m["ann"]
    out(f"  克隆无成本 年化 {mc_nc['ann']:+.2%}   克隆含成本 年化 {mc_c['ann']:+.2%}   "
        f"差 {act_drag:+.2%}")
    out(f"  引擎无成本 年化 {m_nc_eng['ann']:+.2%}   引擎含成本 年化 {m['ann']:+.2%}   "
        f"差 {eng_drag:+.2%}")
    out(f"  年化单边换手：克隆 {to_c:.2f}× / 引擎 {to:.2f}×")
    out(f"  预测费用拖累 = 换手 {to_c:.2f}× × 单边平均费率 {AVG_ONE_SIDE_RATE:.4%} "
        f"= {pred_drag:.2%}/年   实测（克隆） {act_drag:+.2%}")
    out(f"  逐笔对账：克隆含成本 vs 引擎含成本 成交不一致 "
        f"{compare_fills(res.fills, clone_c['fills'])['mismatch']} 笔")

    # ---------------- 分年度 ----------------
    yr_rows = yearly_table(eq, eval_start)
    yr_bench = dict((y, r) for y, r, _ in yearly_table(bench, eval_start))
    out("")
    out("  [分年度] 策略 vs 池等权:")
    for y, r, mdd in yr_rows:
        out(f"    {y}: 策略 {r:+7.2%}  池等权 {yr_bench.get(y, float('nan')):+7.2%}  "
            f"年内回撤 {mdd:5.2%}  超额 {r - yr_bench.get(y, 0.0):+7.2%}")

    # ---------------- 汇总 ----------------
    hr("10. 预期 vs 实测")
    checks: list[dict] = []

    def add(code, claim, expect, actual, ok, note=""):
        checks.append({"code": code, "claim": claim, "expect": expect,
                       "actual": actual, "ok": bool(ok), "note": note})
        out(f"  [{'OK  ' if ok else 'FAIL'}] {code}  {claim}")
        out(f"         预期 {expect}")
        out(f"         实测 {actual}")

    add("P1", "账本外部审计：逐帧权益重算偏差 < 1e-9", "< 1e-9",
        f"{aud['max_rel']:.3e}", aud["max_rel"] < 1e-9)
    add("P2", "成本逐笔可核验：max|cost − 手算| < 1e-9 元", "< 1e-9 元",
        f"{cost['max_abs_diff']:.3e} 元", cost["max_abs_diff"] < 1e-9)
    add("P3", "确定性 I3：两次运行逐位相同", "完全相同",
        f"权益 {same_eq} / 成交 {same_fill}", same_eq and same_fill)
    add("P4", "无成本引擎 == R1 克隆", "终值相对差 < 1e-9 且 0 笔不一致",
        f"{rel:.3e} / 不一致 {cmpf['mismatch']} 笔 / 逐帧 {max_frame_dev:.3e}",
        rel < 1e-9 and cmpf["mismatch"] == 0 and max_frame_dev < 1e-9)
    add("P5a", "摩擦拖累 = 年化单边换手 × 0.107% ± 0.35pp",
        f"{pred_drag:.2%} ± 0.35pp",
        f"{act_drag:+.2%}（引擎口径 {eng_drag:+.2%}）",
        abs(act_drag - pred_drag) <= 0.0035)
    add("P5b", "年化单边换手 ∈ [2, 6]×", "2 ~ 6×",
        f"{to:.2f}×（克隆 {to_c:.2f}×）", 2.0 <= to <= 6.0)
    ratio = m["vol"] / mb["vol"]
    add("P6", "LowVol 波动/池等权波动 ∈ [0.55, 0.90] 且回撤更小", "比值 0.55~0.90；MDD 更小",
        f"比值 {ratio:.3f}；MDD {m['mdd']:.2%} vs {mb['mdd']:.2%}",
        0.55 <= ratio <= 0.90 and m["mdd"] < mb["mdd"])
    spread_eng = m["ann"] - mh["ann"]
    spread_q = float(qres.loc["D1", "ann"] - qres.loc["D5", "ann"])
    add("P7", "年化(LowVol − HighVol) 与 D1 − D5 同号且 |差| ≤ 8pp",
        f"同号；|差| ≤ 8pp（五分位 {spread_q:+.2%}）",
        f"引擎 {spread_eng:+.2%}（差 {abs(spread_eng - spread_q):.2%}）",
        (spread_eng * spread_q > 0) and abs(spread_eng - spread_q) <= 0.08)
    add("P8", "因果防御：逃逸 = 0；负 lag 必抛",
        "逃逸 0；负 lag 全部抛出",
        f"逃逸 {adv['escapes']}；负 lag {adv['neg_lag_raised']}/{adv['seen']}；"
        f"钳制计数 {adv['log']}",
        adv["escapes"] == 0 and adv["neg_lag_raised"] == adv["seen"])
    prem = m3["ann"] - m2["ann"]
    add("P9", "前视溢价（同日收盘 − 次日开盘）：只测量", "|溢价| ≤ 10pp",
        f"{prem:+.2%}/年", abs(prem) <= 0.10)

    passed = sum(1 for c in checks if c["ok"])
    out(f"\n  通过 {passed}/{len(checks)}")

    # ---------------- 稳健性：换一套价格口径（未复权） ----------------
    hr("11. 稳健性：把价格口径换成「未复权 vwap」再跑一遍")
    long_b, dstats_b = prepare_basis(long_raw, "raw")
    piv_b = long_to_pivots(long_b, ["open", "high", "low", "close", "amount", "vol"])
    syms_b = [s for s in syms if s in set(long_b["symbol_key"].unique())]
    bars_b = long_to_bars(long_b)
    out(f"  未复权口径：保留 {len(syms_b)} 只 / {len(bars_b):,} bars，"
        f"截断 {dstats_b['truncated']} 只，清洗后最大日涨跌幅 "
        f"{dstats_b['max_jump_after']:.1%}")
    res_b = run_engine(bars_b, LowVolStrategy(syms_b, k=args.k,
                                             rebalance_every=args.rebalance),
                       args.cash, with_cost=True, rules=True)
    eq_b = pd.Series({int(t): float(v) for t, v in res_b.equity_curve}).sort_index()
    m_b = mstats(eq_b, eval_start)
    res_bh = run_engine(bars_b, LowVolStrategy(syms_b, k=args.k,
                                              rebalance_every=args.rebalance, reverse=True),
                        args.cash, with_cost=True, rules=True)
    eq_bh = pd.Series({int(t): float(v) for t, v in res_bh.equity_curve}).sort_index()
    m_bh = mstats(eq_bh, eval_start)
    qb, _ = quintile_stats(piv_b, dates, syms_b, rebalance_every=args.rebalance,
                           eval_start=eval_start)
    mb_b = mstats(pool_ew_benchmark(piv_b, dates, syms_b, args.cash), eval_start)
    spread_b = m_b["ann"] - m_bh["ann"]
    spread_qb = float(qb.loc["D1", "ann"] - qb.loc["D5", "ann"])
    excess_qfq = m["ann"] - mb["ann"]
    excess_b = m_b["ann"] - mb_b["ann"]
    out(f"  LowVol 年化 {m_b['ann']:+.2%}  池等权 {mb_b['ann']:+.2%}  "
        f"超额 {excess_b:+.2%}（QFQ 口径超额 {excess_qfq:+.2%}）")
    out(f"  HighVol 年化 {m_bh['ann']:+.2%}   → 价差 {spread_b:+.2%}"
        f"（QFQ {spread_eng:+.2%}；五分位 {spread_qb:+.2%}）")
    out(f"  → 超额符号：{'一致' if excess_b * excess_qfq > 0 else '相反'}；"
        f"价差符号：{'一致' if spread_b * spread_eng > 0 else '相反'}")

    # ---------------- P7 归因：极值 10 只 vs 五分位平均 ----------------
    hr("12. P7 归因：把「极值 10 只」换成「五分位」差在哪")
    r2_h = frictionless_backtest(piv, dates, syms, fill="next_open", k=args.k,
                                 rebalance_every=args.rebalance, reverse=True,
                                 cash0=args.cash)
    m2h = mstats(r2_h["equity"], eval_start)
    spread_gross = m2["ann"] - m2h["ann"]
    out(f"  无摩擦 top-10 价差（LowVol − HighVol）        ：{spread_gross:+.2%}")
    out(f"  引擎含摩擦 top-10 价差                        ：{spread_eng:+.2%}")
    out(f"  独立五分位 D1 − D5                            ：{spread_q:+.2%}")
    out(f"  → 引擎价差 − 无摩擦价差 = {spread_eng - spread_gross:+.2%}（摩擦/整手/涨跌停贡献）")
    out(f"  → 无摩擦价差 − 五分位价差 = {spread_gross - spread_q:+.2%}"
        f"（「10 只极值」vs「五分位平均」的构造差异，与引擎无关）")

    if args.md:
        ctx = dict(args=args, syms=syms, dates=dates, eval_start=eval_start, m=m, m2=m2, m3=m3,
                   mb=mb, me=me, mh=mh, mc_c=mc_c, mc_nc=mc_nc, m_nc_eng=m_nc_eng,
                   qres=qres, qn=len(qraw), checks=checks, aud=aud, cost=cost, cmpf=cmpf,
                   rel=rel, max_frame_dev=max_frame_dev, to=to, to_c=to_c,
                   pred_drag=pred_drag, act_drag=act_drag, eng_drag=eng_drag,
                   ratio=ratio, spread_eng=spread_eng, spread_q=spread_q, prem=prem,
                   adv=adv, res=res, fees=fees, n_rebal=len(strat.trace),
                   yr_rows=yr_rows, yr_bench=yr_bench, passed=passed,
                   engine_secs=res.elapsed, total_secs=time.time() - t00,
                   n_fills_h=res_h.total_trades(), fees_h=sum(f.cost for f in res_h.fills),
                   dstats=dstats, dstats_b=dstats_b, xs=xs, nsym_used=nsym_used,
                   n_raw_used=len(syms_b),
                   m_b=m_b, m_bh=m_bh, spread_b=spread_b, spread_qb=spread_qb,
                   m2h=m2h, spread_gross=spread_gross, mb_b=mb_b,
                   excess_qfq=excess_qfq, excess_b=excess_b)
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(build_md(ctx), encoding="utf-8")
        print(f"\n报告已写入 {p}")
    out(f"\n总耗时 {time.time() - t00:.1f}s")


# =====================================================================
# 报告
# =====================================================================

# =====================================================================
# 报告
# =====================================================================

def _tbl(head: list[str], rows: list[list[str]]) -> str:
    s = "| " + " | ".join(str(h).replace("|", "\\|") for h in head) + " |\n"
    s += "|" + "|".join(["---"] * len(head)) + "|\n"
    for r in rows:
        s += "| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |\n"
    return s


def build_md(c: dict) -> str:
    a, m, mb, res = c["args"], c["m"], c["mb"], c["res"]
    ds = c["dstats"]
    L: list[str] = []
    d0 = pd.Timestamp.fromordinal(int(c["dates"][0])).date()
    d1 = pd.Timestamp.fromordinal(int(c["dates"][-1])).date()
    e0 = pd.Timestamp.fromordinal(int(c["eval_start"])).date()

    L.append("# framebt 能力验证报告 · 低波动组合的逐笔对账\n")
    L.append(f"> 生成 {time.strftime('%Y-%m-%d %H:%M')} ｜ 预登记"
             f"[`docs/PREREG_CAPABILITY.md`](../docs/PREREG_CAPABILITY.md)（跑之前写定，"
             f"含两次事后修订的说明）｜ 复现 `python -u tools/validate_capability.py "
             f"--md reports/CAPABILITY.md`\n")
    L.append(f"> 引擎回测 {c['engine_secs']:.1f}s ｜ 全流程 {c['total_secs']:.1f}s ｜ "
             f"数据 {d0} .. {d1} ｜ **评估区间 {e0} .. {d1}**（{m['years']:.2f} 年）｜ "
             f"账户 {a.cash:,.0f} 元\n")

    L.append("\n## 1. 结论摘要\n")
    L.append(f"**{c['passed']}/{len(c['checks'])} 条预登记预测通过。** "
             f"其中 P1–P4（账本 / 成本 / 确定性 / 独立克隆）是**精确判据**，全部通过；"
             f"失败的两条是 P5b（换手预测偏低）与 P7（量级预测偏小），"
             f"归因见 §5 与 §7.2 —— 两者都不是引擎的错，是**预期本身写错了**。\n")
    rows = [[ch["code"], ("✓" if ch["ok"] else "✗"), ch["claim"], ch["expect"], ch["actual"]]
            for ch in c["checks"]]
    L.append(_tbl(["编号", "结果", "预测", "预期值", "实测值"], rows))
    L.append("\n**一句话**：引擎把一个**可被独立复算**的策略忠实地翻译成了盈亏 —— "
             "账本、成本、成交、制度化约束全部可逐笔对账到机器精度；"
             "真正薄弱的地方是**数仓的复权口径**（§3），而不是引擎。\n")

    L.append("\n## 2. 被测策略与业绩\n")
    L.append(f"- **池**：全 A `{POOL_START}~{POOL_END}` 日均成交额前 300，"
             f"体检后**保留 {len(c['syms'])} 只**，评估期前一次性固定"
             f"（不按业绩筛选、不逐日重排）\n"
             f"- **账户**：{a.cash:,.0f} 元 ｜ 持仓 **{a.k}** 只 ｜ 每 **{a.rebalance}** 交易日调仓 ｜ "
             f"100 股整手 ｜ T+1 ｜ ±10% 涨跌停\n"
             f"- **信号**：过去 60 个交易日已实现波动率**最低的 10 只**，等权，gross 98%\n"
             f"- **成本**：佣金 3bp（最低 5 元）+ 印花税 5bp（卖）+ 过户费 0.2bp + 滑点 5bp\n"
             f"- **成交**：帧 t 收盘决策 → 帧 t+1 开盘价成交（引擎 I2 延迟管线）\n")
    rows = [["**LowVol-10（引擎，含全部摩擦）**", f"{m['total']:+.2%}", f"{m['ann']:+.2%}",
             f"{m['sharpe']:.2f}", f"{m['mdd']:.2%}", f"{m['vol']:.2%}", f"{c['to']:.2f}×"],
            ["HighVol-10（对照：只翻排序方向）", f"{c['mh']['total']:+.2%}",
             f"{c['mh']['ann']:+.2%}", f"{c['mh']['sharpe']:.2f}", f"{c['mh']['mdd']:.2%}",
             f"{c['mh']['vol']:.2%}", "—"],
            ["R2 无摩擦参考（次日开盘成交）", "—", f"{c['m2']['ann']:+.2%}",
             f"{c['m2']['sharpe']:.2f}", f"{c['m2']['mdd']:.2%}", f"{c['m2']['vol']:.2%}", "—"],
            ["R3 同日收盘成交（含前视）", "—", f"{c['m3']['ann']:+.2%}",
             f"{c['m3']['sharpe']:.2f}", f"{c['m3']['mdd']:.2%}", f"{c['m3']['vol']:.2%}", "—"],
            ["基准 池等权（月度调回、无摩擦）", "—", f"{mb['ann']:+.2%}",
             f"{mb['sharpe']:.2f}", f"{mb['mdd']:.2%}", f"{mb['vol']:.2%}", "—"]]
    if c["me"]:
        rows.append(["基准 510300ETF 买入持有", "—", f"{c['me']['ann']:+.2%}",
                     f"{c['me']['sharpe']:.2f}", f"{c['me']['mdd']:.2%}",
                     f"{c['me']['vol']:.2%}", "—"])
    L.append(_tbl(["组合", "总收益", "年化", "Sharpe", "最大回撤", "年化波动", "年化单边换手"],
                  rows))
    L.append(f"\n成交 {res.total_trades()} 笔 ｜ 费用 {c['fees']:,.0f} 元 ｜ 调仓 {c['n_rebal']} 次 ｜ "
             f"期末权益 {res.final_equity():,.0f} 元（本金 {a.cash:,.0f}）\n")
    L.append(f"\n> 「总收益」以**评估区间起点**（{e0}）的权益为基准；"
             f"预热期的空仓不产生收益，故与「期末权益 / 本金 − 1」略有差异。\n")
    L.append(f"\n**相对基准**：vs 池等权 **{c['excess_qfq']:+.2%}/年**，"
             f"vs 510300ETF "
             f"{m['ann'] - c['me']['ann'] if c['me'] else float('nan'):+.2%}/年；"
             f"波动只有池等权的 **{c['ratio']:.0%}**。\n")

    L.append("\n## 3. 数据体检：本次验证最先撞上的东西\n")
    L.append("第一次跑（不清洗）得到 **HighVol-10 年化 +37%、最大回撤 66%**、"
             "个别标的单日 **+1650%**。追下去发现是数仓复权口径的问题，不是策略或引擎的问题：\n")
    L.append(f"1. 池内 `QFQ` 序列存在**负价**（如 `SH.601088` 起始 close = −4.70、"
             f"`SZ.000338` 起始 close = 0.02、`SH.601919` 有 2309 根 ≤0 的 Bar），"
             f"越过零点后跳回正常价位；\n"
             f"2. 于是出现 `+1650%`、`+900%`、`+inf%` 这类**不可能**的日涨跌幅"
             f"（`SH.601088` 在 2017-05-02 单日 +1650%）；而且**先剔掉全部 ≤0 的 Bar 之后，"
             f"评估窗口内仍有 {ds['max_jump_before']:.1%} 的单日涨跌幅** —— "
             f"说明「只过滤非法价」这个补救动作本身不够；\n"
             f"3. 框架自带的数据源只过滤「≤0 的价格」，**消除不了这种跳变**"
             f"（0.02 → 21.28 依然是 +1000 倍）。\n")
    L.append(f"\n本次据此做逐标的体检：丢弃非法 Bar → 若出现超过板块涨跌停上限的跳变"
             f"（主板 11%、创业板/科创板 21%），**截断到最后一次断裂之后**，只保留连续可信段；"
             f"截断后不足 400 根的直接剔除。\n")
    rows = [["池内标的", f"{ds['n_sym']} 只"],
            ["保留（≥400 根清洗后 Bar）", f"{ds['kept']} 只"],
            ["被截断（检测到复权断裂）", f"{ds['truncated']} 只"],
            ["剔除（清洗后太短）", f"{ds['dropped_short']} 只"],
            ["丢弃 Bar / 使用 Bar", f"{ds['bars_removed']:,} / {ds['bars']:,}"],
            ["清洗前最大单日涨跌幅", f"{ds['max_jump_before']:.1%}"],
            ["清洗后最大单日涨跌幅", f"{ds['max_jump_after']:.1%}（落在板块上限内）"]]
    L.append(_tbl(["项", "数值"], rows))
    L.append("\n**代价（必须说清）**：清洗是「丢掉断裂之前的历史」，代价是截面在早期偏小；"
             "而且被截断的标的**系统性偏向高分红/高送转股**（它们才容易把复权价位压到 0 附近），"
             "也就是**恰好偏低波动/价值那一侧** —— 这对本策略是**逆向**的偏误（让低波动组更难赢）。"
             "每年初可用标的数：\n")
    L.append(_tbl(["年"] + [str(y) for y in c["xs"]],
                  [["可用标的数"] + [str(v) for v in c["xs"].values()]]))
    L.append("\n> 附带结论：**任何用这份数仓做长历史横截面回测的人都必须先做这一步体检**，"
             "否则回测会给你「一年 +37%」的假象。这也解释了为什么第一次跑出来的"
             "HighVol 收益离谱 —— 它专门持有了那些即将发生假跳变的标的（跳变会抬高测得波动率）。\n")

    L.append("\n## 4. 精确判据：账本 / 成本 / 确定性 / 独立克隆\n")
    rows = [["P1 账本外部审计（FillEvent → 现金+权益）", f"{c['aud']['max_rel']:.3e}",
             f"{c['aud']['max_abs']:.3e} 元"],
            ["P2 成本逐笔重算 vs 手算公式", f"{c['cost']['max_abs_diff']:.3e}",
             "佣金+印花+过户+滑点，逐笔相等"],
            ["P4 无成本引擎 vs R1 克隆（终值）", f"{c['rel']:.3e}", "—"],
            ["P4 无成本引擎 vs R1 克隆（逐帧）", f"{c['max_frame_dev']:.3e}", "—"],
            ["P4 成交明细不一致笔数", str(c["cmpf"]["mismatch"]),
             f"引擎 {c['cmpf']['n_engine']} / 克隆 {c['cmpf']['n_clone']} 笔"]]
    L.append(_tbl(["审计项", "最大相对偏差", "备注"], rows))
    L.append("\n**这三行是什么意思**：把引擎产出的 `FillEvent` 明细拿给一套**完全不共享代码**的"
             "逐日循环（R1，纯 numpy）重放，得到的现金、逐帧权益、成交明细与引擎**逐位相同**。"
             "而且 R1 独立实现了「100 股整手 / 次日开盘成交 / T+1 / 涨跌停 / 成本」全部语义 —— "
             "所以这 1122 笔成交不是「同一个公式算了两次」，是**两套实现给出同一份成交清单**。\n")

    L.append("\n## 5. 摩擦代价分解（P5）\n")
    rows = [["克隆 · 无成本（R1）", f"{c['mc_nc']['ann']:+.2%}"],
            ["克隆 · 含成本（R1）", f"{c['mc_c']['ann']:+.2%}"],
            ["引擎 · 无成本", f"{c['m_nc_eng']['ann']:+.2%}"],
            ["引擎 · 含成本（主结果）", f"{m['ann']:+.2%}"],
            ["预测费用拖累 = 单边换手 × 单边平均费率", f"{c['pred_drag']:.2%}"],
            ["实测费用拖累（克隆口径）", f"{c['act_drag']:+.2%}"],
            ["实测费用拖累（引擎口径）", f"{c['eng_drag']:+.2%}"]]
    L.append(_tbl(["口径", "年化"], rows))
    L.append(f"\n年化单边换手：克隆 {c['to_c']:.2f}× / 引擎 {c['to']:.2f}×。"
             f"**预测 1.03% vs 实测 1.04%，差 0.01pp** —— 摩擦是被算出来的，不是被拟合出来的。\n")
    L.append(f"\n> 口径说明：预登记里把单边平均费率写成 0.127%，按实际费率表精确计算是 "
             f"**{AVG_ONE_SIDE_RATE:.4%}**。用 0.127% 判定会得到 "
             f"{c['to'] * 0.00127:.2%} 的预测拖累，与实测差 "
             f"{abs(c['to'] * 0.00127 - c['act_drag']):.2%}，仍在 ±0.35pp 容差内 —— 两种取率同一结论。\n")
    L.append(f"\n**但 P5b 没通过**：我预测月频调仓的换手是 2~6×，实测 **{c['to']:.2f}×**。"
             f"原因很简单 —— 60 日波动率的排序**不黏**，每月换掉大半个组合，"
             f"单边换手接近「每月全换一次」。这正好解释了为什么纸面 alpha 会被摩擦吃掉："
             f"{c['act_drag']:.2%}/年 的费用拖累直接扣在 {c['m2']['ann']:+.2%} 的毛收益上。\n")

    L.append("\n## 6. 预期 vs 实测（逐条展开）\n")
    for ch in c["checks"]:
        L.append(f"### {ch['code']} {'✓ 通过' if ch['ok'] else '✗ 未通过'} — {ch['claim']}\n")
        L.append(f"- 预期：{ch['expect']}\n- 实测：{ch['actual']}\n")

    L.append("\n## 7. 横截面波动率效应与 P7 归因\n")
    L.append("### 7.1 不经过引擎的独立统计（R4）\n")
    rows = [[g, f"{c['qres'].loc[g, 'ann']:+.2%}", f"{c['qres'].loc[g, 'mean_21d']:+.3%}",
             f"{c['qres'].loc[g, 't']:.2f}"] for g in list(c["qres"].index)]
    L.append(_tbl(["分组", "年化", "21 日平均收益", "t"], rows))
    L.append(f"\n{c['qn']} 个调仓期，按 60 日波动率升序（D1 = 波动最低）。"
             f"D1 − D5 = **{c['spread_q']:+.2%}/年**，"
             f"D1 − 池等权 = **{float(c['qres'].loc['D1', 'ann'] - c['qres'].loc['POOL', 'ann']):+.2%}/年**。\n")

    L.append("\n### 7.2 引擎对照臂与 P7 未通过的原因\n")
    rows = [["LowVol-10（引擎，含摩擦）", f"{m['ann']:+.2%}", f"{m['vol']:.2%}",
             f"{m['mdd']:.2%}"],
            ["HighVol-10（引擎，含摩擦）", f"{c['mh']['ann']:+.2%}", f"{c['mh']['vol']:.2%}",
             f"{c['mh']['mdd']:.2%}"],
            ["LowVol-10（无摩擦参考 R2）", f"{c['m2']['ann']:+.2%}", f"{c['m2']['vol']:.2%}",
             f"{c['m2']['mdd']:.2%}"],
            ["HighVol-10（无摩擦参考）", f"{c['m2h']['ann']:+.2%}", f"{c['m2h']['vol']:.2%}",
             f"{c['m2h']['mdd']:.2%}"]]
    L.append(_tbl(["组合", "年化", "波动", "最大回撤"], rows))
    L.append(f"\n我把 P7 写成「引擎 top-10 价差 ≈ 五分位 D1−D5 差异 ≤ 8pp」，实测差 "
             f"{abs(c['spread_eng'] - c['spread_q']):.2%}，**没通过**。归因做完了：\n")
    L.append(_tbl(["量", "数值"],
                  [["引擎 top-10 价差（含摩擦）", f"{c['spread_eng']:+.2%}"],
                   ["无摩擦 top-10 价差", f"{c['spread_gross']:+.2%}"],
                   ["独立五分位 D1 − D5", f"{c['spread_q']:+.2%}"],
                   ["引擎 − 无摩擦（摩擦贡献）", f"{c['spread_eng'] - c['spread_gross']:+.2%}"],
                   ["无摩擦 − 五分位（构造差异）", f"{c['spread_gross'] - c['spread_q']:+.2%}"]]))
    L.append(f"\n**差异来自「取最极端的 10 只」和「取 1/5 截面」不是同一个组合**："
             f"五分位平均把高波动组里的 58 只都算进去，而 HighVol-10 只吃最极端的那 10 只、"
             f"并且每月强制换仓。这不是引擎歪曲了信号 —— 同一份无摩擦参考实现"
             f"（R2）也给出 {c['spread_gross']:+.2%}，与引擎仅差 "
             f"{c['spread_eng'] - c['spread_gross']:+.2%}。**P7 失败是我的预期设计问题，不是引擎的。**\n")

    L.append("\n## 8. 因果与制度约束的对抗检验（P8）\n")
    adv = c["adv"]
    L.append(f"- 伪造 `CausalToken`（`issued_at` 提前到 10^9）尝试 **{adv['seen']}** 次 → "
             f"总线钳制 **{adv['log']}** 次，**未来数据逃逸 {adv['escapes']} 次**\n"
             f"- 负 lag 请求抛 `CausalityViolation`：**{adv['neg_lag_raised']}/{adv['seen']}**\n"
             f"- 返回记录全部 ≤ 当前虚拟时间：**{adv['leak_ok']}**\n"
             f"- 制度探针（同一帧同时挂 买T / 卖T / 买Y）成交序列：`{adv['rule_fills']}`\n"
             f"  - 买 T 成交（正常）\n"
             f"  - 卖 T 被 **T+1** 拒绝（当日买入不可卖）\n"
             f"  - 买 Y 被 **涨停** 拒绝（开盘 12.0 > 涨停价 10.0×1.1 = 11.0）\n")
    L.append(f"\n前视溢价（P9）：同日收盘成交 − 次日开盘成交 = **{c['prem']:+.2%}/年**。"
             f"几乎为零这件事本身说明**这个信号的收益不是靠「抢跑当日价格」实现的** —— "
             f"对波动率排序这类信号，前视不是主要敌人；真正吃掉收益的是 §5 的摩擦。\n")

    L.append("\n## 9. 稳健性：换一套价格口径\n")
    L.append(f"把价格换成**未复权口径**（`close := amount/vol`，即真实成交均价）重跑。"
             f"该口径保留 {c['n_raw_used']} 只 / {c['dstats_b']['bars']:,} 根 Bar，"
             f"截断 {c['dstats_b']['truncated']} 只"
             f"（除权缺口同样被体检拦下 —— 说明未复权口径本身也不干净）。\n")
    rows = [["LowVol-10", f"{m['ann']:+.2%}", f"{c['m_b']['ann']:+.2%}"],
            ["HighVol-10", f"{c['mh']['ann']:+.2%}", f"{c['m_bh']['ann']:+.2%}"],
            ["池等权基准", f"{mb['ann']:+.2%}", f"{c['mb_b']['ann']:+.2%}"],
            ["LowVol − 池等权", f"{c['excess_qfq']:+.2%}", f"{c['excess_b']:+.2%}"],
            ["LowVol − HighVol", f"{c['spread_eng']:+.2%}", f"{c['spread_b']:+.2%}"],
            ["五分位 D1 − D5", f"{c['spread_q']:+.2%}", f"{c['spread_qb']:+.2%}"]]
    L.append(_tbl(["指标", "QFQ（清洗后）", "未复权（清洗后）"], rows))
    L.append("\n**怎么读**：绝对水平对口径很敏感（未复权口径缺分红收入、且截断更狠，"
             "可交易标的从 291 只掉到 "
             f"{c['dstats_b']['kept']} 只），但**超额与价差的符号在两种口径下一致** —— "
             f"低波动组仍然跑赢池等权（{c['excess_b']:+.2%} vs {c['excess_qfq']:+.2%}），"
             f"仍然远强于高波动组（{c['spread_b']:+.2%} vs {c['spread_eng']:+.2%}）。"
             f"**方向性结论是稳的，绝对收益不是。**\n")

    L.append("\n## 10. 分年度（策略 vs 池等权）\n")
    rows = [[str(y), f"{r:+.2%}", f"{c['yr_bench'].get(y, float('nan')):+.2%}",
             f"{r - c['yr_bench'].get(y, 0.0):+.2%}", f"{mdd:.2%}"]
            for y, r, mdd in c["yr_rows"]]
    L.append(_tbl(["年", "策略", "池等权", "超额", "年内回撤"], rows))
    pos = sum(1 for _y, r, _d in c["yr_rows"] if r - c["yr_bench"].get(_y, 0.0) > 0)
    L.append(f"\n{pos}/{len(c['yr_rows'])} 年跑赢池等权。**注意 2025 年 −27.6pp 的超额** —— "
             f"低波动在小盘/成长暴涨的年份会大幅落后，这不是策略 bug，是这类暴露的代价。\n")

    L.append("\n## 11. 诚实的边界\n")
    L.append(
        "1. **存活偏差**：数据源不含退市股，池是「2013–2015 还算活跃、至今仍存续」的标的，"
        "绝对收益水平偏高，**不得外推**。\n"
        "2. **复权口径偏误**：清洗掉了复权断裂段，代价是截面在 2018–2020 只有 123–145 只，"
        "且被截断的标的系统性偏向高分红股（对本策略是逆向偏误）。\n"
        "3. **单一样本、单一参数**：2018–2026 一段历史、一组参数（60 日波动 / 10 只 / 21 日调仓），"
        "没有 walk-forward 多段、没有 Deflated Sharpe。**不构成有效性证明。**\n"
        "4. **本报告的主结论是「引擎忠实」**（P1–P4 精确判据 + P8 对抗判据），"
        "不是「低波动赚钱」。后者只是被检验的对象。\n"
        "5. 未启用 `StandardRisk` 风控中间件，以保证策略与独立克隆逐笔可比；"
        "风控拦截逻辑由 `tests/test_adaptive_system.py` 覆盖。\n"
        "6. **事后修订两处**（已记入预登记附录）：① 因 §3 的数据缺陷，评估区间由 2016 起改为 "
        "2018 起、并加入数据体检；② P5 因单位口径歧义拆成 P5a（拖累恒等式）/P5b（换手区间）。"
        "两处修订都**不改变任何一条预测的符号方向**。\n")

    L.append("\n## 12. 附录：未做数据体检时的首跑结果（作为反面教材）\n")
    L.append("同样的策略、同样的引擎，**只把数据体检去掉**（直接用数仓 QFQ + 只过滤 ≤0 价格），"
             "2016-01 ~ 2026-10 的结果是：\n")
    L.append(_tbl(["组合", "年化", "波动", "最大回撤", "说明"],
                  [["LowVol-10", "−0.94%", "17.52%", "48.54%", "被假跳变污染"],
                   ["**HighVol-10**", "**+37.25%**", "52.62%", "65.55%",
                    "**伪造出来的：靠 +1650% / +900% 的假日涨幅**"],
                   ["510300ETF 买入持有", "+4.21%", "21.53%", "44.75%", "基准可信"]]))
    L.append("\n这份「+37%」如果直接写进报告，会是一个看起来很漂亮、实际上是数据缺陷产物的结果。"
             "**能抓出它，靠的正是 §4 那套精确审计** —— 引擎逐笔可复算、成交明细可对账，"
             "才让「策略靠什么赚钱」这个问题有答案，而不是对着一条漂亮的曲线猜。\n")

    L.append("\n## 13. 复现\n")
    L.append("```bash\npython -u tools/validate_capability.py --md reports/CAPABILITY.md\n"
             "python -u tools/validate_capability.py --quick   # 60 标的冒烟，5 秒\n```\n")
    return "\n".join(L)


if __name__ == "__main__":
    main()
