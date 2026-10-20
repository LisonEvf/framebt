"""示例策略：Bracket（入场 + 止盈 + 止损，§5.1）。

Bracket 由既有原语组合而成，无需专门的订单类型：
  1. 入场：均线突破时市价买入；
  2. 成交回报（``on_fill``）后提交一个 **OCO 组**：止盈限价卖单 + 止损条件单；
  3. 任一腿成交 → 引擎自动撤销另一腿（§5.1 OCO 联动）。

因果性：入场订单经延迟管线于下一帧成交（I2）；成交后才挂出止盈/止损，
触发价基于成交价，由引擎在后续帧监控。
"""

from __future__ import annotations

from ..core import Condition, FrameStrategy, Order, OrderSide, OrderType, Trigger
from ..execution import OrderGroup


class BracketStrategy(FrameStrategy):
    def __init__(
        self,
        symbol: str,
        ma_period: int = 5,
        take_profit_pct: float = 0.05,
        stop_loss_pct: float = 0.03,
    ):
        self.symbol = symbol
        self.ma_period = ma_period
        self.take_profit_pct = take_profit_pct
        self.stop_loss_pct = stop_loss_pct
        self._armed = False  # 已有未平仓的 bracket

    def on_frame(self, frame, token, bus, host) -> list:
        closes = bus.get("close", self.symbol, window=self.ma_period, token=token)
        if len(closes) < self.ma_period:
            return []
        ma = sum(closes[-self.ma_period:]) / self.ma_period
        pos = host.portfolio.position(self.symbol)

        # 趋势向上且空仓且未挂 bracket → 入场（现金不足时不下单）
        if closes[-1] > ma and pos.net_size() == 0 and not self._armed and host.portfolio.cash > 0:
            self._armed = True
            return [
                Order(
                    side=OrderSide.BUY,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    budget=host.portfolio.cash,
                )
            ]
        if pos.net_size() == 0:
            self._armed = False
        return []

    def on_fill(self, fill) -> list:
        # 入场成交 → 挂出 OCO 组（止盈 或 止损，一腿成交另一腿自动撤销）
        if fill.side is not OrderSide.BUY:
            return []
        tp = Order(
            side=OrderSide.SELL,
            symbol=fill.symbol,
            size=fill.size,
            order_type=OrderType.LIMIT,
            price=fill.price * (1 + self.take_profit_pct),
        )
        sl = Order(
            side=OrderSide.SELL,
            symbol=fill.symbol,
            size=fill.size,
            order_type=OrderType.MARKET,
            condition=Condition(
                trigger=Trigger.LAST_LE,
                level=fill.price * (1 - self.stop_loss_pct),
                description="止损",
            ),
        )
        return [OrderGroup.oco([tp, sl])]
