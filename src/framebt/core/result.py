"""回测结果与绩效指标（result + perf）。

``BacktestResult`` 收集成交、权益曲线、风险指标；``perf`` 提供可复用的绩效度量。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .events import FillEvent


@dataclass
class BacktestResult:
    fills: list[FillEvent] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)  # (frame_id, equity)
    frames: int = 0
    start_ns: int = 0
    end_ns: int = 0

    # ---- 简单聚合 ----

    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0

    def total_trades(self) -> int:
        return len(self.fills)

    @property
    def frames_span(self) -> int:
        return self.end_ns - self.start_ns

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"BacktestResult(frames={self.frames}, trades={self.total_trades()}, "
            f"final_equity={self.final_equity():.4f})"
        )


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: Sequence[float], mean: float | None = None) -> float:
    if len(xs) < 2:
        return 0.0
    m = mean if mean is not None else _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


class Performance:
    """从权益曲线与成交计算绩效指标（年化基于每帧步长 period_frames）。"""

    def __init__(self, result: BacktestResult, *, period_frames: int = 244, risk_free: float = 0.0):
        self.result = result
        self.period_frames = max(1, int(period_frames))  # 一年多少帧，用于年化
        self.risk_free = risk_free
        self._equities = [e for _, e in result.equity_curve]

    @property
    def total_return(self) -> float:
        if not self._equities or self._equities[0] == 0:
            return 0.0
        return self._equities[-1] / self._equities[0] - 1.0

    @property
    def annualized_return(self) -> float:
        if not self._equities or self._equities[0] == 0 or self.result.frames <= 1:
            return 0.0
        n_years = self.result.frames / self.period_frames
        total = self._equities[-1] / self._equities[0]
        if total <= 0:
            return 0.0
        return total ** (1 / n_years) - 1

    @property
    def volatility(self) -> float:
        if len(self._equities) < 2:
            return 0.0
        r = [
            (b - a) / a
            for a, b in zip(self._equities, self._equities[1:])
            if a != 0
        ]
        return _std(r) * math.sqrt(self.period_frames)

    @property
    def max_drawdown(self) -> float:
        peak = -math.inf
        mdd = 0.0
        for e in self._equities:
            if e > peak:
                peak = e
            if peak > 0:
                dd = (peak - e) / peak
                if dd > mdd:
                    mdd = dd
        return mdd

    @property
    def sharpe(self) -> float:
        if self.volatility == 0:
            return 0.0
        return (self.annualized_return - self.risk_free) / self.volatility

    def win_rate(self) -> float:
        """按「每标的一个完整回合」计算胜率：买入开仓 → 卖出平仓的已实现盈亏。

        遍历成交，按标的累积净敞口；当某标的净敞口归零（一个回合闭合）时，
        用该回合内所有买入/卖出的金额差算出已实现盈亏，统计正盈亏占比。
        """
        from .events import OrderSide

        round_trips: list[float] = []
        open_notional: dict[str, float] = {}  # 当前未平腿的成交金额累计
        open_side: dict[str, OrderSide] = {}
        for f in self.result.fills:
            sym = f.symbol
            if f.side is OrderSide.BUY:
                open_notional[sym] = open_notional.get(sym, 0.0) + f.size * f.price
                open_side[sym] = OrderSide.BUY
            else:
                close_amount = f.size * f.price
                if open_side.get(sym) is OrderSide.BUY:
                    pnl = close_amount - open_notional.get(sym, 0.0)
                    round_trips.append(pnl)
                    open_notional.pop(sym, None)
                    open_side.pop(sym, None)
        if not round_trips:
            return 0.0
        return sum(1 for p in round_trips if p > 0) / len(round_trips)

    def detailed(self) -> dict[str, float]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "trades": self.result.total_trades(),
            "frames": self.result.frames,
        }
