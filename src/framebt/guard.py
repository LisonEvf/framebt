"""Guard：风控中间件（§9）。

- 盘前检查（pre_trade_check）：订单到达撮合引擎前拦截（仓位限制 / 资金充足 / 集中度）。
- 增量风险指标：VaR、最大回撤、暴露度逐帧更新。

风控是独立中间件，不混入策略代码，也不混入撮合引擎（由 Engine 在 Phase 3 注入）。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional, Sequence

from .core.events import Order, OrderSide
from .core.market_frame import MarketFrame
from .ledger import Ledger


class RiskMiddleware(ABC):
    @abstractmethod
    def pre_trade_check(self, order: Order, portfolio: Ledger, frame: MarketFrame) -> Optional[str]:
        """返回拒绝原因字符串，或 None 表示放行。"""
        ...


class NoOpRisk(RiskMiddleware):
    def pre_trade_check(self, order, portfolio, frame) -> Optional[str]:
        return None


class IncrementalRisk:
    """增量风险指标（§9.2）：逐帧更新，只更新变化部分。"""

    def __init__(self, equity_curve: Sequence[tuple[int, float]]):
        self._equities = [e for _f, e in equity_curve]

    def update(self, frame_id: int, equity: float) -> None:
        self._equities.append(equity)

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

    def var(self, confidence: float = 0.95) -> float:
        """历史模拟法 VaR（单帧，返回正数表示亏损幅度）。"""
        if len(self._equities) < 2:
            return 0.0
        returns = [
            (b - a) / a
            for a, b in zip(self._equities, self._equities[1:])
            if a != 0
        ]
        if not returns:
            return 0.0
        sorted_ret = sorted(returns)
        idx = max(0, int(math.ceil((1 - confidence) * len(sorted_ret))) - 1)
        return -sorted_ret[idx]

    def exposure(self) -> float:
        """总持仓市值暴露的近似。"""
        return abs(self._equities[-1]) if self._equities else 0.0


class StandardRisk(RiskMiddleware):
    """标准盘前风控。

    参数：
      max_position_pct：单标的最大仓位（占权益比例，默认 1.0 = 全仓）。
      max_order_pct：单笔订单最大占权益比例（默认 1.0）。
      min_cash：最小保留现金（默认 0）。
      reserve_pending：是否把**在途买单**占用的资金计入预留（默认 True）。

    ``reserve_pending`` 很关键：订单有强制延迟（I2），同一帧内可以合法地提交多笔
    买单，每笔单独看都「资金充足」，但合计会透支账户。风控必须按
    「现金 − 在途买单已占用」来判断可用资金，否则会放行超额下单。
    """

    def __init__(
        self,
        max_position_pct: float = 1.0,
        max_order_pct: float = 1.0,
        min_cash: float = 0.0,
        reserve_pending: bool = True,
    ):
        self.max_position_pct = max_position_pct
        self.max_order_pct = max_order_pct
        self.min_cash = min_cash
        self.reserve_pending = reserve_pending
        self.matcher = None   # 由 Engine 注入，用于查询在途订单

    # ---- 权益口径 ----

    @staticmethod
    def _equity(portfolio: Ledger, frame: MarketFrame) -> float:
        """**真实权益** = 现金 + 持仓按当帧价格计值。

        不能直接用 ``Ledger.equity()`` —— 它返回的是**现金**（见其 docstring：
        「未标记时的现金」）。若拿现金当权益，满仓后现金趋近 0，任何订单都会被
        判成「超最大订单比例」而拒绝，表现为整场回测只有零星成交。
        """
        eq = portfolio.cash
        for sym, size in portfolio.exposure().items():
            if frame.has_bar(sym):
                eq += size * frame.last(sym)
        return eq

    def reserved_cash(self, frame: MarketFrame, exclude: Order | None = None) -> float:
        """在途买单已占用的资金（``exclude`` 用于排除正在被校验的那笔订单）。

        **必须排除当前订单**：否则「预留」里含它自己，可用资金被自身吃掉，
        会把每一笔合法买单都判成资金不足 —— 表现为整场回测零成交。
        """
        if not self.reserve_pending or self.matcher is None:
            return 0.0
        total = 0.0
        for o in self.matcher.pending_orders():
            if o.side is OrderSide.BUY and (exclude is None or o.id != exclude.id):
                total += self._order_notional(o, frame)
        return total

    def pre_trade_check(self, order: Order, portfolio: Ledger, frame: MarketFrame) -> Optional[str]:
        notional = self._order_notional(order, frame)
        equity = self._equity(portfolio, frame)
        if equity <= 0 or notional <= 0:
            return None  # 无法评估时放行（如预算单无明确金额）

        # 单笔订单占比（相对**真实权益**）
        if notional > equity * self.max_order_pct:
            return (
                f"单笔订单 {notional:.0f} 超最大订单比例 "
                f"({self.max_order_pct:.0%} × 权益 {equity:.0f})"
            )

        if order.side is OrderSide.BUY:
            # 资金充足（扣除**其他**在途买单的预留）
            reserved = self.reserved_cash(frame, exclude=order)
            available = portfolio.cash - reserved
            if notional > available + 1e-6:
                return (
                    f"资金不足：需 {notional:.0f}，可用 {available:.0f}"
                    f"（现金 {portfolio.cash:.0f} − 在途预留 {reserved:.0f}）"
                )
            # 买入后单标的仓位不超限
            post_size = portfolio.position(order.symbol).net_size() + order.size
            post_notional = post_size * self._price(order, frame)
            if post_notional > equity * self.max_position_pct:
                return (
                    f"买入后 {order.symbol} 仓位 {post_notional:.0f} 超最大仓位 "
                    f"({self.max_position_pct:.0%} × 权益 {equity:.0f})"
                )
            return None

        else:  # SELL
            proceeds = notional
            if portfolio.cash + proceeds < self.min_cash:
                return f"卖出后现金 {portfolio.cash + proceeds:.0f} 低于最小保留 {self.min_cash:.0f}"
            return None

    def _order_notional(self, order: Order, frame: MarketFrame) -> float:
        price = self._price(order, frame)
        if price > 0 and order.size > 0:
            return order.size * price
        return order.budget

    def _price(self, order: Order, frame: MarketFrame) -> float:
        if frame.has_bar(order.symbol):
            return frame.last(order.symbol)
        return 0.0
