"""多因子 sweep 策略的数据准备与因子计算（研究侧，向量化、可缓存）。

职责：
  1. 加载稳定标的池（全 A 成交额前 N，按历史日均成交额排序）；
  2. 由 BarFeed 构造 ``Panel``，取**近 3 年**为回测窗口；
  3. 计算选定的 WorldQuant 101 Alpha 因子子集，做
     去极值(MAD) → 标准化(z-score) → 市值中性化(log mcap)；
  4. 派生策略所需的辅助面板：
       - ``eligibility``   ：当日是否流动性合格（动态流动性筛选）
       - ``sweep_signal``  ：sweep 突破放量信号（做 T 入场）
       - ``sweep_exit``    ：sweep 跌破触发位信号（做 T 出场）
       - ``regime``        ：市场指数 > 其 N 日均线 → 风险偏好
  5. ``compute_composite``：按权重合成截面打分（权重可变，供权重优化复用）；
  6. 缓存到 ``cache/``，避免多次回测重复算因子（因子计算是最耗时步骤）。

依赖：pandas + numpy。
"""

from __future__ import annotations

import os
import pickle

import numpy as np
import pandas as pd

from framebt.feed import BarFeed
from framebt.data_sources import OpentdxParquetSource
from framebt.factors.library import FACTOR_LIBRARY, compute_many
from framebt.factors.preprocess import mad_winsorize, zscore, neutralize, log_market_cap
from framebt.factors.ops import ts_max, ts_mean, returns as _returns

# ---------------------------------------------------------------- 配置
# 回测窗口：近 3 个自然年。1095 天 ≈ 3 年（约 750 交易日）。
WINDOW_DAYS = 1095
# 因子计算所需的历史预热（需覆盖最长因子窗口 250 + 窗口预热）。
COMPUTE_SPAN_DAYS = 1200
# 稳定标的池：全 A 成交额前 N（按历史日均成交额）。
POOL_SIZE = 300
MIN_BARS = 260

# 选定的多因子子集：动量/反转 + 放量突破 + 波动率 + vwap 偏离 + 日内动量，跨风格。
FACTOR_NAMES = [
    "wq002", "wq005", "wq007", "wq017", "wq018", "wq019",
    "wq024", "wq034", "wq041", "wq043", "wq051", "wq057",
    "wq096", "wq101",
]

# sweep 做 T 参数（与策略对齐）
SWEEP_LOOKBACK = 15
SWEEP_VOL_MULT = 1.5
SWEEP_HOLD_MAX = 5

# 流动性筛选：尾部平均成交额 >= 当日标的池内最高值的比例
ELIG_REL_MIN = 0.3
ELIG_MIN_HISTORY = 120

# regime 阈值
REGIME_MA = 60

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_NAME = "mf_sweep_cache.pkl"


def _prepare_pool(src, max_symbols=POOL_SIZE, min_bars=MIN_BARS):
    """稳定池：按历史日均成交额取前 N（口径与 tools/ab_test_strategies 一致）。"""
    universe = src.available_symbols()
    feed = src.universe(universe)
    counts, amounts = {}, {}
    for b in feed.bars:
        counts[b.symbol] = counts.get(b.symbol, 0) + 1
        amounts[b.symbol] = amounts.get(b.symbol, 0.0) + b.amount
    avg = {s: a / max(1, counts[s]) for s, a in amounts.items()}
    elig = [s for s in universe if counts.get(s, 0) >= min_bars]
    elig.sort(key=lambda s: avg.get(s, 0), reverse=True)
    return elig[:max_symbols], feed


def _slice_window(feed, span_days):
    """取最近 span_days 根 K 线的合并 feed。"""
    bars = list(feed.bars)
    if not bars:
        return BarFeed([])
    max_ts = max(b.timestamp for b in bars)
    min_ts = max_ts - span_days
    return BarFeed([b for b in bars if b.timestamp >= min_ts])


def compute_factor_panels(panel, names):
    """计算并预处理因子面板（去极值 → 标准化 → 市值中性化）。"""
    raw = compute_many(names, panel)
    mcap = log_market_cap(panel)
    out = {}
    for name in names:
        f = raw[name]
        f = mad_winsorize(f, 3.0)
        f = zscore(f)
        f = neutralize(f, [mcap], add_const=True)
        f = zscore(f)  # 中性化后重新标准化
        out[name] = f
    return out


def compute_eligibility(panel, symbols=None):
    """当日流动性合格：近 120 日平均成交额 >= ELIG_REL_MIN × 当日标的池内最高 120 日均额。

    阈值相对“标的池（300 只）”而非全 5575 只计算，否则 0.5×单只最活跃股会排除掉
    几乎所有标的。
    """
    amt = panel.amount
    if symbols is not None:
        amt = amt[[c for c in symbols if c in amt.columns]]
    trailing = ts_mean(amt, 120)
    pool_max = trailing.max(axis=1)
    rel = trailing.div(pool_max.where(pool_max > 0), axis=0)
    has_hist = trailing.notna().apply(lambda s: s.dropna().count() >= ELIG_MIN_HISTORY, axis=1)
    # has_hist 是按日（行）的布尔 Series，需按行广播到 DataFrame，不能用 & 直接乘 Series
    elig = (rel >= ELIG_REL_MIN) & has_hist.values.reshape(-1, 1)
    return elig.fillna(False)


def compute_sweep_signals(panel, lookback=SWEEP_LOOKBACK, vol_mult=SWEEP_VOL_MULT):
    """sweep 突破放量信号（入场）与跌破触发位（出场）。

    入场：收盘价突破近 lookback 日高点（不含当日）且放量（> vol_mult × 均量）且当日阳线。
    出场：收盘价跌回突破位下方（跌破近 lookback 日高点，不含当日）。
    """
    close = panel.close
    open_ = panel.open
    vol = panel.volume
    prev_high = ts_max(close, lookback).shift(1)
    adv = ts_mean(vol, lookback)

    breakout = close > prev_high
    volume_ok = vol > vol_mult * adv
    bullish = close > open_
    signal = (breakout & volume_ok & bullish).fillna(False)

    exit_sig = close < prev_high
    exit_sig = exit_sig.fillna(False)
    return signal, exit_sig


def compute_regime(panel, ma=REGIME_MA):
    """价值加权市场指数：指数运行在其 N 日均线之上 = 风险偏好。"""
    close = panel.close
    float_sh = panel.float_shares.where(panel.float_shares > 0, np.nan)
    ret = _returns(close)
    cap = close * float_sh
    w = cap.div(cap.sum(axis=1), axis=0)
    w_lag = w.shift(1)
    mret = (w_lag * ret).sum(axis=1)
    mret = mret.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    level = (1 + mret).cumprod()
    ma_series = level.rolling(ma, min_periods=max(20, ma // 3)).mean()
    regime = level > ma_series
    return regime.fillna(True)


def compute_composite(factor_panels, weights):
    """按权重合成截面打分 ``score = Σ wᵢ·fᵢ``（按因子取齐，缺失跳过）。

    与回测侧 ``MultiFactorSweepStrategy._composite`` 完全同构。
    """
    total = None
    wsum = None
    for name, w in weights.items():
        f = factor_panels.get(name)
        if f is None or abs(w) < 1e-12:
            continue
        contrib = f * w
        total = contrib if total is None else total.add(contrib, fill_value=0.0)
        aw = f.notna().astype(float) * abs(w)
        wsum = aw if wsum is None else wsum.add(aw, fill_value=0.0)
    if total is None:
        return None
    return total.div(wsum.where(wsum > 1e-12))


class PrepData:
    """封装数据准备结果，供优化器/运行器复用。"""

    def __init__(self, data: dict):
        self.universe = data["universe"]
        self.close = data["close"]
        self.volume = data["volume"]
        self.amount = data["amount"]
        self.float_shares = data["float_shares"]
        self.factor_panels = data["factor_panels"]       # 已预处理
        self.sweep_signal = data["sweep_signal"]
        self.sweep_exit = data["sweep_exit"]
        self.eligibility = data["eligibility"]
        self.regime = data["regime"]
        self.window_start = data["window_start"]
        self.window_end = data["window_end"]
        self.n_days = data["n_days"]
        self.feed = data["feed"]

    def composite(self, weights):
        return compute_composite(self.factor_panels, weights)

    def to_dict(self):
        return {k: getattr(self, k) for k in dir(self)
                if k not in ("composite", "to_dict") and not k.startswith("_")}


def prepare(force=False):
    """（重）准备数据并缓存，返回 PrepData。"""
    cache_path = os.path.join(CACHE_DIR, CACHE_NAME)
    if not force and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return PrepData(pickle.load(f))

    src = OpentdxParquetSource()
    pool, feed = _prepare_pool(src)
    print(f"[data] 稳定池 {len(pool)} 只")

    # 因子计算：全池（5575 只）；引擎撮合用：标的池（300 只）的窗口 feed
    window_feed = _slice_window(feed, COMPUTE_SPAN_DAYS)
    pool_feed = _slice_window(src.universe(pool), COMPUTE_SPAN_DAYS)
    panel = _panel_from_feed(window_feed)

    # 窗口切取（近 3 年）。注意：时间戳是日历 ordinal（date.toordinal），
    # 不能当成 pd.Timestamp(int) 的纳秒解释，否则显示会错。
    max_ts = int(panel.close.index.max())
    min_ts = max_ts - WINDOW_DAYS
    sel = panel.close.index >= min_ts
    close = panel.close.loc[sel]
    window = int(min_ts)
    end = int(max_ts)

    factor_panels = compute_factor_panels(panel, FACTOR_NAMES)
    # 只保留窗口内的因子面板
    factor_panels = {n: f.loc[sel] for n, f in factor_panels.items()}

    eligibility = compute_eligibility(panel, symbols=pool).loc[sel]
    sig, exit_sig = compute_sweep_signals(panel)
    sweep_signal = sig.loc[sel]
    sweep_exit = exit_sig.loc[sel]
    regime = compute_regime(panel).loc[sel]

    data = {
        "universe": pool,
        "close": close,
        "volume": panel.volume.loc[sel],
        "amount": panel.amount.loc[sel],
        "float_shares": panel.float_shares.loc[sel],
        "factor_panels": factor_panels,
        "sweep_signal": sweep_signal,
        "sweep_exit": sweep_exit,
        "eligibility": eligibility,
        "regime": regime,
        "window_start": window,
        "window_end": end,
        "n_days": int(sel.sum()),
        "feed": pool_feed,
    }
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    print(f"[data] 窗口 {pd.Timestamp.fromordinal(window).date()} .. "
          f"{pd.Timestamp.fromordinal(end).date()}，{data['n_days']} 交易日，"
          f"因子 {len(FACTOR_NAMES)} 个，已缓存")
    return PrepData(data)


def _panel_from_feed(feed: BarFeed):
    """构造 Panel（惰性 import pandas，避免无数据依赖时报错）。"""
    from framebt.factors.panel import Panel
    return Panel.from_feed(feed)


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = prepare(force=True)
    print(f"[data] 完成，耗时 {time.time()-t0:.1f}s")
    print(f"[data] 窗口 {d.window_start}..{d.window_end}，{d.n_days} 交易日，池 {len(d.universe)}")
