"""统一的绩效指标口径（单一事实来源）。

为什么单独一个模块
------------------
在此之前，指标散在三处且口径不一致：

- ``core/result.py``：``sharpe = 年化收益 / 波动``（CAGR/vol，不是标准 Sharpe），
  年化按 ``frames / 244``，``total_return`` 以**第一帧权益**为基准（漏掉首帧盈亏）；
- ``report.py``：直接复用 ``Performance``，于是 HTML 卡片继承了上述口径；
- 各 tools 脚本：又各写一套（有的用 ``√252``、有的用算术均值）。

本模块把定义收敛到一处，所有入口（``Performance`` / 报告 / 工具）都调用它。
**口径一旦改，只改这里。**

约定
----
- ``periods_per_year``：一年多少个「帧/交易日」。A 股默认 **244**；美股/外部记录可传 252。
- 收益率：简单收益 ``e_t / e_{t-1} - 1``（不用对数收益）。
- 年化收益（CAGR）：``(final / initial) ** (1 / years) - 1``，``years = n_periods / ppy``。
- 波动：样本标准差（``ddof=1``）× ``sqrt(ppy)``。
- Sharpe：``(mean(r) - rf/ppy) / std(r) * sqrt(ppy)``（标准定义，均值/标准差，
  不是 CAGR/vol；旧口径保留为 ``sharpe_cagr``）。
- Sortino：``(mean(r) - rf/ppy) / downside_dev * sqrt(ppy)``，
  ``downside_dev = sqrt(mean(min(r - rf/ppy, 0)^2))``。
- 最大回撤：正数（0.25 = 回撤 25%）。
- Calmar：``CAGR / max_drawdown``。

与旧报告的关系
--------------
2026-10-29 之前生成的报告，``sharpe`` 是 ``CAGR / 波动``（旧口径），且波动用 √252、
年化用 244（混用）。从本模块起统一为 ``periods_per_year`` 一个参数；旧口径可用
``Performance.sharpe_cagr`` 取到。**新旧 Sharpe 不可直接比较**。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "DEFAULT_PERIODS_PER_YEAR",
    "MetricConfig",
    "returns_from_equity",
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calmar_ratio",
    "summarize",
]

DEFAULT_PERIODS_PER_YEAR = 244


@dataclass(frozen=True)
class MetricConfig:
    """指标计算参数。"""

    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR
    risk_free: float = 0.0
    ddof: int = 1

    def __post_init__(self) -> None:
        if self.periods_per_year <= 0:
            raise ValueError("periods_per_year 必须为正")
        if self.ddof < 0:
            raise ValueError("ddof 不能为负")


def _clean(equity: Sequence[float]) -> list[float]:
    return [float(x) for x in equity if x is not None and math.isfinite(float(x))]


def returns_from_equity(equity: Sequence[float]) -> list[float]:
    """相邻权益的简单收益；起点为 0 的段跳过（无法定义收益率）。"""
    eq = _clean(equity)
    return [(b - a) / a for a, b in zip(eq, eq[1:]) if a != 0]


def total_return(equity: Sequence[float], initial: float | None = None) -> float:
    """总收益。``initial`` 缺省用权益曲线首点（旧行为；建议传初始资金）。"""
    eq = _clean(equity)
    if not eq:
        return 0.0
    base = float(initial) if initial is not None and initial > 0 else eq[0]
    if base <= 0:
        return 0.0
    return eq[-1] / base - 1.0


def cagr(
    equity: Sequence[float],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    n_periods: int | None = None,
    initial: float | None = None,
) -> float:
    """年化收益（复合）。``n_periods`` 缺省为 ``len(equity) - 1``。"""
    eq = _clean(equity)
    if len(eq) < 2:
        return 0.0
    base = float(initial) if initial is not None and initial > 0 else eq[0]
    n = int(n_periods) if n_periods is not None else len(eq) - 1
    if base <= 0 or n <= 0 or eq[-1] <= 0:
        return 0.0
    years = n / float(periods_per_year)
    if years <= 0:
        return 0.0
    return (eq[-1] / base) ** (1.0 / years) - 1.0


def annualized_volatility(
    equity: Sequence[float],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    ddof: int = 1,
) -> float:
    """年化波动：简单收益的样本标准差 × √periods_per_year。"""
    r = returns_from_equity(equity)
    if len(r) < 2:
        return 0.0
    mean = sum(r) / len(r)
    var = sum((x - mean) ** 2 for x in r) / (len(r) - ddof) if len(r) > ddof else 0.0
    return math.sqrt(var) * math.sqrt(periods_per_year) if var > 0 else 0.0


def sharpe_ratio(
    equity: Sequence[float],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free: float = 0.0,
    ddof: int = 1,
) -> float:
    """标准 Sharpe：``(mean(r) - rf/ppy) / std(r) * sqrt(ppy)``。"""
    r = returns_from_equity(equity)
    if len(r) < 2:
        return 0.0
    mean = sum(r) / len(r)
    excess = mean - risk_free / periods_per_year
    var = sum((x - mean) ** 2 for x in r) / (len(r) - ddof) if len(r) > ddof else 0.0
    if var <= 0:
        return 0.0
    return excess / math.sqrt(var) * math.sqrt(periods_per_year)


def sortino_ratio(
    equity: Sequence[float],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    risk_free: float = 0.0,
) -> float:
    """Sortino：分母只计下行偏差（相对目标收益 ``rf/ppy``）。"""
    r = returns_from_equity(equity)
    if len(r) < 2:
        return 0.0
    target = risk_free / periods_per_year
    mean = sum(r) / len(r)
    downside = [min(x - target, 0.0) ** 2 for x in r]
    dd = math.sqrt(sum(downside) / len(r))
    if dd <= 0:
        return 0.0
    return (mean - target) / dd * math.sqrt(periods_per_year)


def max_drawdown(equity: Sequence[float]) -> float:
    """最大回撤（正数）。"""
    eq = _clean(equity)
    peak = -math.inf
    mdd = 0.0
    for e in eq:
        if e > peak:
            peak = e
        if peak > 0:
            dd = (peak - e) / peak
            if dd > mdd:
                mdd = dd
    return mdd


def calmar_ratio(
    equity: Sequence[float],
    *,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    n_periods: int | None = None,
    initial: float | None = None,
) -> float:
    """Calmar：CAGR / 最大回撤。"""
    mdd = max_drawdown(equity)
    if mdd <= 0:
        return 0.0
    return cagr(equity, periods_per_year=periods_per_year,
                n_periods=n_periods, initial=initial) / mdd


def summarize(
    equity: Sequence[float],
    *,
    initial: float | None = None,
    periods_per_year: int = DEFAULT_PERIODS_PER_YEAR,
    n_periods: int | None = None,
    risk_free: float = 0.0,
    ddof: int = 1,
) -> dict[str, float]:
    """一次算出全部标准指标（键名与 ``Performance.detailed()`` 一致）。"""
    eq = _clean(equity)
    if not eq:
        return {
            "total_return": 0.0, "annualized_return": 0.0, "volatility": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "calmar": 0.0,
            "periods": 0.0, "years": 0.0,
        }
    n = int(n_periods) if n_periods is not None else max(0, len(eq) - 1)
    return {
        "total_return": total_return(eq, initial),
        "annualized_return": cagr(eq, periods_per_year=periods_per_year,
                                  n_periods=n, initial=initial),
        "volatility": annualized_volatility(eq, periods_per_year=periods_per_year, ddof=ddof),
        "sharpe": sharpe_ratio(eq, periods_per_year=periods_per_year,
                               risk_free=risk_free, ddof=ddof),
        "sortino": sortino_ratio(eq, periods_per_year=periods_per_year,
                                 risk_free=risk_free),
        "max_drawdown": max_drawdown(eq),
        "calmar": calmar_ratio(eq, periods_per_year=periods_per_year,
                               n_periods=n, initial=initial),
        "periods": float(n),
        "years": n / float(periods_per_year),
    }
