"""Rules：市场规则引擎（§7）。

A 股规则（本默认实现）：
  - 涨跌停：默认 ±10%（相对昨收），可配置；触发价越出涨跌停带拒绝
  - T+1：当日买入不可当日卖出（``settlement_cycle`` 控制）
  - 停牌：可配置停牌标的集合
  - 交易日历：可配置交易日历（缺省每个逻辑日视为可交易）
  - 可交易时段：``is_trading``（MarketFrame.is_trading）

接口（§7.1）：``can_buy/can_sell/price_limit/is_suspended/is_tradable/settlement_cycle``。
``check(order, ledger, frame)`` 为组合检查，返回拒绝原因或 None（供引擎调用）。

``NoOpRules``：不做任何限制（Phase 0 / 非 A 股场景）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from .core.events import Order, OrderSide
from .core.market_frame import MarketFrame
from .ledger import Ledger


class MarketRules(ABC):
    @abstractmethod
    def can_buy(self, asset: str, frame: MarketFrame) -> tuple[bool, str]:
        ...

    @abstractmethod
    def can_sell(self, asset: str, frame: MarketFrame) -> tuple[bool, str]:
        ...

    @abstractmethod
    def price_limit(self, asset: str, frame: MarketFrame) -> tuple[Optional[float], Optional[float]]:
        ...

    @abstractmethod
    def is_suspended(self, asset: str, frame: MarketFrame) -> bool:
        ...

    @abstractmethod
    def is_tradable(self, asset: str, frame: MarketFrame) -> bool:
        ...

    @abstractmethod
    def settlement_cycle(self, asset: str) -> int:
        ...

    @abstractmethod
    def available_to_borrow(self, asset: str, frame: MarketFrame) -> float:
        """可融券借出的最大数量（用于 SELL_SHORT 约束）。"""
        ...

    def observe(self, frame: MarketFrame) -> None:  # 逐帧钩子（更新昨收等状态）
        pass

    def check(self, order: Order, ledger: Ledger, frame: MarketFrame) -> Optional[str]:
        """组合检查：返回拒绝原因字符串，或 None 表示放行。

        **必须有「本帧无该标的行情」的早退分支**：多标的回测中，各标的的上市日/
        停牌日不同，某些帧天然缺某些标的。若不先判断就直接 ``frame.open(symbol)``，
        会抛 KeyError 并把整场回测打断（真实数据下必然触发）。
        无法定价的订单一律拒绝，与撮合层「无标的价格数据」的处理保持一致。
        """
        symbol = order.symbol

        if not frame.has_bar(symbol):
            return f"{symbol} 本帧无行情数据（停牌/未上市/数据缺失）"
        if self.is_suspended(symbol, frame):
            return f"{symbol} 停牌"
        if not frame.is_trading:
            return "非交易时段"
        if not self.is_tradable(symbol, frame):
            return f"{symbol} 不可交易"

        upper, lower = self.price_limit(symbol, frame)

        if order.side is OrderSide.BUY:
            ok, reason = self.can_buy(symbol, frame)
            if not ok:
                return reason
            if upper is not None and frame.open(symbol) > upper:
                return f"{symbol} 涨停（开盘价 {frame.open(symbol):.2f} > 涨停价 {upper:.2f}）"
            return None

        if order.side is OrderSide.SELL_SHORT:
            # ---- 融券做空（A 股约束）----
            ok, reason = self.can_sell(symbol, frame)
            if not ok:
                return reason
            # 涨停不可开空（开盘价触及涨停价）
            if upper is not None and frame.open(symbol) > upper:
                return (
                    f"{symbol} 涨停（开盘价 {frame.open(symbol):.2f} > 涨停价 {upper:.2f}）不可开空"
                )
            # 可借数量约束：开空量不得超过可融券数量
            if order.size > 0:
                borrowable = self.available_to_borrow(symbol, frame)
                if borrowable < float("inf") and order.size > borrowable + 1e-9:
                    return (
                        f"{symbol} 可借 {borrowable:.0f} 股 < 开空 {order.size:.0f} 股"
                    )
            return None

        else:  # SELL
            ok, reason = self.can_sell(symbol, frame)
            if not ok:
                return reason
            if lower is not None and frame.open(symbol) < lower:
                return f"{symbol} 跌停（开盘价 {frame.open(symbol):.2f} < 跌停价 {lower:.2f}）"
            # T+1：可卖出的已结算持仓 >= 卖出数量（预算单无法预知数量，跳过）
            if order.size > 0:
                pos = ledger.position(symbol)
                available = pos.available_to_sell(frame.frame_id)
                if available + 1e-9 < order.size:
                    return (
                        f"{symbol} T+1：可卖出 {available:.0f} 股 < 下单 {order.size:.0f} 股"
                    )
            return None


class NoOpRules(MarketRules):
    """无规则（默认）。"""

    def __init__(self, settlement_cycle: int = 1):
        self._settlement_cycle = settlement_cycle

    def can_buy(self, asset, frame) -> tuple[bool, str]:
        return True, ""

    def can_sell(self, asset, frame) -> tuple[bool, str]:
        return True, ""

    def price_limit(self, asset, frame) -> tuple[None, None]:
        return None, None

    def is_suspended(self, asset, frame) -> bool:
        return False

    def is_tradable(self, asset, frame) -> bool:
        return True

    def settlement_cycle(self, asset) -> int:
        return self._settlement_cycle

    def available_to_borrow(self, asset, frame) -> float:
        return float("inf")


class StandardAshareRules(MarketRules):
    """A 股标准规则。

    参数：
      limit：涨跌停幅度（默认 0.10 = ±10%）。
      settlement_cycle：结算周期（T+1 → 1）。
      suspended：停牌标的集合（或 callable(symbol, frame)->bool）。
      trading_days：交易日历（逻辑时间戳集合；缺省全部可交易）。
    """

    def __init__(
        self,
        limit: float = 0.10,
        settlement_cycle: int = 1,
        suspended=None,
        trading_days=None,
    ):
        self.limit = limit
        self.settlement_cycle_val = settlement_cycle
        self._suspended = suspended
        self._trading_days = set(trading_days) if trading_days else None
        self._prev_close: dict[str, float] = {}

    def observe(self, frame: MarketFrame) -> None:
        """逐帧更新昨收。"""
        for sym, bar in frame.bars.items():
            self._prev_close[sym] = bar.close

    def can_buy(self, asset, frame) -> tuple[bool, str]:
        return True, ""

    def can_sell(self, asset, frame) -> tuple[bool, str]:
        return True, ""

    def price_limit(self, asset, frame) -> tuple[Optional[float], Optional[float]]:
        prev = self._prev_close.get(asset)
        if prev is None or prev <= 0:
            return None, None
        return prev * (1 + self.limit), prev * (1 - self.limit)

    def is_suspended(self, asset, frame) -> bool:
        if callable(self._suspended):
            return bool(self._suspended(asset, frame))
        if self._suspended is not None and asset in self._suspended:
            return True
        return False

    def is_tradable(self, asset, frame) -> bool:
        """交易日历检查。

        ``trading_days`` 传逻辑时间戳集合。引擎的 ``frame.end_ns`` 是
        ``bar.timestamp + delta_t``（日线下即「bar 时间戳 + 1」），因此同时匹配
        ``end_ns`` 与 ``end_ns - 1``，两种传法都能用。未提供 ``trading_days``
        时不做限制（与旧行为一致）。
        """
        if self._trading_days is None:
            return True
        return (frame.end_ns in self._trading_days
                or (frame.end_ns - 1) in self._trading_days)

    def settlement_cycle(self, asset) -> int:
        return self.settlement_cycle_val

    def available_to_borrow(self, asset, frame) -> float:
        """可融券借出量：以行情流通股本 ``float_shares`` 为上限代理。

        数据源在 Bar 上提供 ``float_shares``（流通股本）时返回之；缺失时返回
        ``inf``（不约束，避免无数据时误拒开空）。
        """
        if frame.has_bar(asset):
            fs = frame.bars[asset].float_shares
            if fs and fs > 0:
                return fs
        return float("inf")
