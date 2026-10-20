"""示例策略：均线趋势 + 跟踪止损条件单（§5.2）。

买入后挂出「跌破止损价自动卖出」的条件单：条件单由引擎在帧间监控触发，
而非策略逐帧主动判断（§5.2）。

因果性要点：
  - 买入订单经延迟管线于下一帧成交（I2）。
  - 成交回报触发 on_fill，在 on_fill 中挂出止损条件单。
  - 止损触发价基于成交后的最新价，由引擎在后续帧监控。
"""

from __future__ import annotations

from ..core import FrameStrategy, Order, OrderSide, OrderType, Condition, Trigger


class TrendWithStop(FrameStrategy):
    def __init__(self, symbol: str, ma_period: int = 20, stop_pct: float = 0.05):
        self.symbol = symbol
        self.ma_period = ma_period
        self.stop_pct = stop_pct
        self._entry_price: float | None = None

    def on_frame(self, frame, token, bus, host) -> list:
        closes = bus.get("close", self.symbol, window=self.ma_period, token=token)
        if len(closes) < self.ma_period:
            return []

        ma = sum(closes[-self.ma_period:]) / self.ma_period
        last = closes[-1]
        pos = host.portfolio.position(self.symbol)

        # 关键：订单有强制延迟（I2），本帧下的单不会立刻反映到持仓上。
        # 若只看持仓判断，会在成交落地前**每帧重复全仓下单**，导致仓位指数级爆炸。
        # 因此先检查是否已有未落地订单。
        if host.has_pending(self.symbol):
            return []

        orders: list = []
        # 趋势确认：价格在均线上方且尚未持仓 → 全仓买入（现金不足时不下单）
        if last > ma and pos.net_size() == 0 and host.portfolio.cash > 0:
            orders.append(
                Order(
                    side=OrderSide.BUY,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    budget=host.portfolio.cash,
                )
            )
        # 趋势破坏：价格跌破均线 → 市价平仓
        elif last < ma and pos.net_size() > 0:
            orders.append(
                Order(
                    side=OrderSide.SELL,
                    symbol=self.symbol,
                    size=pos.net_size(),
                    order_type=OrderType.MARKET,
                )
            )
        return orders

    def on_fill(self, fill) -> list:
        # 成交后挂出跟踪止损条件单：以最新价跌破 (1 - stop_pct) 触发卖出。
        # 返回订单供引擎登记入在途队列（I2），由引擎在帧间监控触发（§5.2）。
        if fill.side is OrderSide.BUY:
            self._entry_price = fill.price
            stop_level = fill.price * (1 - self.stop_pct)
            cond = Condition(trigger=Trigger.LAST_LE, level=stop_level,
                             description=f"止损 @{stop_level:.2f}")
            sell = Order(
                side=OrderSide.SELL,
                symbol=fill.symbol,
                size=fill.size,
                order_type=OrderType.MARKET,
                condition=cond,
            )
            return [sell]
        return []
