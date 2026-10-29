"""回测结果与绩效指标（result + perf）。

指标定义统一在 :mod:`framebt.metrics`；本文件只做「结果对象 → 指标」的适配，
不再各写一套公式。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import metrics
from .events import FillEvent


@dataclass
class BacktestResult:
    fills: list[FillEvent] = field(default_factory=list)
    equity_curve: list[tuple[int, float]] = field(default_factory=list)  # (frame_id, equity)
    frames: int = 0
    start_ns: int = 0
    end_ns: int = 0
    #: 初始资金；由 Engine 写入。用于把「首帧盈亏」计入总收益（旧实现以首帧权益为基准）。
    initial_cash: float | None = None

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


class Performance:
    """从权益曲线与成交计算绩效指标。

    所有指标委托给 :mod:`framebt.metrics`，口径见该模块文档。``period_frames``
    是一年多少帧（A 股日线默认 244；美股可用 252），CAGR / 波动 / Sharpe 共用它。

    与旧版差异（2026-10-29 起）：
      - ``sharpe`` 改为标准定义 ``mean(r)/std(r)*√ppy``（旧版是 ``CAGR/vol``，
        保留为 :attr:`sharpe_cagr`）；
      - ``total_return`` 优先以 ``result.initial_cash`` 为基准（含首帧盈亏）；
      - 新增 ``sortino`` / ``calmar``。
    """

    def __init__(self, result: BacktestResult, *, period_frames: int = 244,
                 risk_free: float = 0.0):
        self.result = result
        self.period_frames = max(1, int(period_frames))
        self.risk_free = risk_free
        self._equities = [e for _, e in result.equity_curve]
        init = result.initial_cash
        self._initial = float(init) if init is not None and init > 0 else None

    # ---- 基础指标 ----

    @property
    def total_return(self) -> float:
        return metrics.total_return(self._equities, self._initial)

    @property
    def annualized_return(self) -> float:
        return metrics.cagr(
            self._equities,
            periods_per_year=self.period_frames,
            n_periods=self.result.frames,
            initial=self._initial,
        )

    @property
    def volatility(self) -> float:
        return metrics.annualized_volatility(
            self._equities, periods_per_year=self.period_frames)

    @property
    def sharpe(self) -> float:
        """标准 Sharpe：均值/标准差 × √ppy（risk_free 按年化给定）。"""
        return metrics.sharpe_ratio(
            self._equities,
            periods_per_year=self.period_frames,
            risk_free=self.risk_free,
        )

    @property
    def sharpe_cagr(self) -> float:
        """旧口径 ``CAGR / 波动``（保留用于和 2026-10-29 之前的报告对账）。"""
        vol = self.volatility
        if vol == 0:
            return 0.0
        return (self.annualized_return - self.risk_free) / vol

    @property
    def sortino(self) -> float:
        return metrics.sortino_ratio(
            self._equities,
            periods_per_year=self.period_frames,
            risk_free=self.risk_free,
        )

    @property
    def max_drawdown(self) -> float:
        return metrics.max_drawdown(self._equities)

    @property
    def calmar(self) -> float:
        return metrics.calmar_ratio(
            self._equities,
            periods_per_year=self.period_frames,
            n_periods=self.result.frames,
            initial=self._initial,
        )

    # ---- 逐笔胜率 ----

    def win_rate(self) -> float:
        """按「每标的一个完整回合」计算胜率：买入开仓 → 卖出平仓的已实现盈亏。

        遍历成交，按标的累积净敞口（``open_notional`` / ``open_size``）；
        - 卖出为**部分平仓**时，按比例摊销成本、不闭合回合（市价单受成交量约束会
          部分成交，留下残持仓，不能当成一个完整回合）；
        - 当某标的净敞口归零（一个回合闭合）时，用该回合内所有买入/卖出的金额差
          算出已实现盈亏，统计正盈亏占比。
        """
        from .events import OrderSide

        round_trips: list[float] = []
        open_notional: dict[str, float] = {}   # 当前未平腿的成交金额累计
        open_size: dict[str, float] = {}       # 当前未平腿的数量累计
        open_side: dict[str, OrderSide] = {}
        for f in self.result.fills:
            sym = f.symbol
            if f.side is OrderSide.BUY:
                open_notional[sym] = open_notional.get(sym, 0.0) + f.size * f.price
                open_size[sym] = open_size.get(sym, 0.0) + f.size
                open_side[sym] = OrderSide.BUY
            else:
                osz = open_size.get(sym, 0.0)
                if osz <= 0 or open_side.get(sym) is not OrderSide.BUY:
                    continue
                close_amount = f.size * f.price
                remaining = osz - f.size
                if remaining <= 1e-9:
                    # 完全平仓：闭合一个回合（用整笔成本，避免残差）
                    pnl = close_amount - open_notional.get(sym, 0.0)
                    round_trips.append(pnl)
                    open_notional.pop(sym, None)
                    open_size.pop(sym, None)
                    open_side.pop(sym, None)
                else:
                    # 部分平仓：按比例回收成本，不闭合回合
                    avg = open_notional.get(sym, 0.0) / osz
                    open_notional[sym] = open_notional.get(sym, 0.0) - avg * f.size
                    open_size[sym] = remaining
        if not round_trips:
            return 0.0
        return sum(1 for p in round_trips if p > 0) / len(round_trips)

    def detailed(self) -> dict[str, float]:
        """全部指标（键名稳定；report/tools 可直接复用）。"""
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe": self.sharpe,
            "sharpe_cagr": self.sharpe_cagr,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "calmar": self.calmar,
            "trades": float(self.result.total_trades()),
            "frames": float(self.result.frames),
            "periods_per_year": float(self.period_frames),
        }
