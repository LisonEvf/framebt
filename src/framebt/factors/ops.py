"""因子算子库（WorldQuant 101 Alpha 的算子集合）。

所有算子都作用在 **面板（Panel）** 上：``pandas.DataFrame``，``index=时间戳``、
``columns=标的``。这样同一份算子既能用于研究（整段面板向量化），也能用于回测
（每帧的面板切片），从而满足设计文档「因子定义不因运行模式而分叉」的核心原则。

算子分三类：
  - **截面类**：``rank`` / ``zscore`` / ``scale``（沿 axis=1，同一时刻跨标的）
  - **时序类**：``delta`` / ``delay`` / ``ts_sum`` / ``ts_mean`` / ``ts_std`` /
    ``ts_min`` / ``ts_max`` / ``ts_argmax`` / ``ts_argmin`` / ``ts_rank`` /
    ``decay_linear`` / ``correlation`` / ``covariance``（沿 axis=0，逐标的滚动）
  - **逐元素类**：``sign`` / ``log`` / ``sqrt`` / ``signed_power`` / ``where``

约定：所有除法都用 :func:`safe_div` 保护（分母接近 0 → NaN）；所有算子保持
NaN 传播，不做静默填充 —— 缺失由预处理流水线显式处理。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


# ===========================================================================
# 基础设施
# ===========================================================================

def safe_div(a, b):
    """受保护除法：|分母| < EPS 或 NaN → NaN（不产生 inf）。

    支持标量与面板混合（标量自动广播）—— 否则 ``safe_div(1.0, close)`` 会被
    当成 1×1 面板与 (T,N) 面板相除，静默产生全 NaN。
    """
    a_scalar, b_scalar = np.isscalar(a), np.isscalar(b)
    if a_scalar and b_scalar:
        return (float(a) / float(b)) if abs(float(b)) > EPS else np.nan
    if a_scalar:
        d = _as_df(b)
        return float(a) / d.where(d.abs() > EPS)
    if b_scalar:
        d = _as_df(a)
        return d / float(b) if abs(float(b)) > EPS else d * np.nan
    a, b = _as_df(a), _as_df(b)
    denom = b.where(b.abs() > EPS)
    return a / denom


def _mp(n: int) -> int:
    """滚动窗口的最小有效观测数。

    必须 ``<= n``，否则窗口比最小观测还短会**永远**产出 NaN
    （例如 correlation 窗口 2 却要求 3 个观测）。
    """
    return max(1, min(n, max(2, n // 2)))


def _as_df(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    if isinstance(x, pd.Series):
        return x.to_frame()
    raise TypeError(
        f"期望面板（DataFrame/Series），得到标量 {x!r}；"
        "标量请用 safe_div / where 等支持广播的算子"
    )


def adv(volume: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    """adv{n}：n 日平均成交量（沿用 WQ101 参考实现的「成交量」口径）。"""
    return volume.rolling(n, min_periods=_mp(n)).mean()


# ===========================================================================
# 截面算子（axis=1：同一时刻跨标的）
# ===========================================================================

def rank(df) -> pd.DataFrame:
    """截面百分位排名（0,1]，平均秩处理并列；NaN 不参与也不占位。"""
    return _as_df(df).rank(axis=1, pct=True, na_option="keep")


def zscore(df) -> pd.DataFrame:
    """截面 z-score。"""
    d = _as_df(df)
    mu = d.mean(axis=1)
    sd = d.std(axis=1)
    return d.sub(mu, axis=0).div(sd.where(sd.abs() > EPS), axis=0)


def scale(df, a: float = 1.0) -> pd.DataFrame:
    """截面缩放：使 Σ|x| = a（WQ101 的 scale）。"""
    d = _as_df(df)
    s = d.abs().sum(axis=1)
    return d.div(s.where(s.abs() > EPS), axis=0) * a


# ===========================================================================
# 时序算子（axis=0：逐标的滚动）
# ===========================================================================

def delay(df, n: int) -> pd.DataFrame:
    return _as_df(df).shift(n)


def delta(df, n: int) -> pd.DataFrame:
    d = _as_df(df)
    return d - d.shift(n)


def ts_sum(df, n: int) -> pd.DataFrame:
    return _as_df(df).rolling(n, min_periods=_mp(n)).sum()


def ts_mean(df, n: int) -> pd.DataFrame:
    return _as_df(df).rolling(n, min_periods=_mp(n)).mean()


def ts_std(df, n: int) -> pd.DataFrame:
    return _as_df(df).rolling(n, min_periods=_mp(n)).std()


def ts_min(df, n: int) -> pd.DataFrame:
    return _as_df(df).rolling(n, min_periods=_mp(n)).min()


def ts_max(df, n: int) -> pd.DataFrame:
    return _as_df(df).rolling(n, min_periods=_mp(n)).max()


def ts_argmax(df, n: int) -> pd.DataFrame:
    """窗口内最大值的位置（1-based，1 = 窗口最早）。"""
    return _as_df(df).rolling(n, min_periods=_mp(n)).apply(
        lambda x: float(np.argmax(x) + 1) if np.isfinite(x).any() else np.nan, raw=True)


def ts_argmin(df, n: int) -> pd.DataFrame:
    return _as_df(df).rolling(n, min_periods=_mp(n)).apply(
        lambda x: float(np.argmin(x) + 1) if np.isfinite(x).any() else np.nan, raw=True)


def ts_rank(df, n: int) -> pd.DataFrame:
    """当前值在最近 n 期窗口内的百分位排名（0,1]。"""
    return _as_df(df).rolling(n, min_periods=_mp(n)).rank(pct=True)


def decay_linear(df, n: int) -> pd.DataFrame:
    """线性衰减加权平均：权重 n, n-1, ..., 1（越近权重越大）。"""
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return _as_df(df).rolling(n, min_periods=_mp(n)).apply(
        lambda x: float(np.dot(x, w)) if np.isfinite(x).all() else np.nan, raw=True)


def correlation(x, y, n: int) -> pd.DataFrame:
    """逐标的滚动 Pearson 相关。"""
    return _as_df(x).rolling(n, min_periods=_mp(n)).corr(_as_df(y))


def covariance(x, y, n: int) -> pd.DataFrame:
    """逐标的滚动协方差。"""
    return _as_df(x).rolling(n, min_periods=_mp(n)).cov(_as_df(y))


# ===========================================================================
# 逐元素算子
# ===========================================================================

def sign(df) -> pd.DataFrame:
    return np.sign(_as_df(df))


def log(df) -> pd.DataFrame:
    d = _as_df(df)
    return np.log(d.where(d > EPS))


def sqrt(df) -> pd.DataFrame:
    return np.sqrt(_as_df(df).clip(lower=0))


def abs_(df) -> pd.DataFrame:
    return _as_df(df).abs()


def signed_power(df, e: float) -> pd.DataFrame:
    d = _as_df(df)
    return np.sign(d) * (d.abs() ** e)


def where(cond, a, b) -> pd.DataFrame:
    """向量化三元：``cond`` 为真取 ``a``，否则取 ``b``。

    ``a`` / ``b`` 可以是面板，也可以是标量；``cond`` 必须是面板。
    """
    c = _as_df(cond).astype(bool)
    a_scalar, b_scalar = np.isscalar(a), np.isscalar(b)
    if a_scalar and b_scalar:
        base = pd.DataFrame(float(b), index=c.index, columns=c.columns)
        return base.where(~c, float(a))
    if b_scalar:
        return _as_df(a).where(c, float(b))
    if a_scalar:
        return _as_df(b).where(~c, float(a))
    return _as_df(a).where(c, _as_df(b))


def returns(close: pd.DataFrame) -> pd.DataFrame:
    """日简单收益率。"""
    c = _as_df(close)
    return c / c.shift(1) - 1.0


def corrwith_rows(a, b, min_valid: int = 5) -> pd.Series:
    """**逐行**（截面）Pearson 相关，NaN/零方差安全。

    注意：``DataFrame.corrwith(axis=1)`` 在**零方差行**上返回 ``inf`` 而非 ``NaN``
    （0/0 在 pandas 的向量化实现里落到 inf 分支）。零方差截面在真实数据中并不罕见
    —— 停牌日全体价格不变、某因子在某日对所有标的取值相同，都会触发。若不处理，
    ``inf`` 会随 ``.mean()`` 传播为 NaN 并抛 RuntimeWarning。
    因此这里统一把 ``±inf`` 归一为 ``NaN``，并丢弃有效标的本数不足的行。
    """
    a, b = _as_df(a), _as_df(b)
    common = a.notna() & b.notna()
    n_valid = common.sum(axis=1)
    corr = a.where(common).corrwith(b.where(common), axis=1)
    corr = corr.replace([np.inf, -np.inf], np.nan)
    return corr.where(n_valid >= min_valid)
