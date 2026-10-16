"""示例策略：双均线交叉（§Phase 0 / 验收用例）。

因果性要点（严格遵循 I1/I2）：
  - 交叉判断用「本帧 MAs」与「上一帧 MAs」。上一帧 MAs 来自 bus 返回序列的
    前缀（已过去帧），绝不使用当前帧之后的数据（无前视）。
  - 订单在触发帧生成，经延迟管线于下一帧开盘价成交（I2）。
  - 全额下单用 budget=host.portfolio.cash，数量在成交价处反算。
"""

from __future__ import annotations

from ..core import FrameStrategy, Order, OrderSide, OrderType


class DualMAStrategy(FrameStrategy):
    def __init__(self, symbol: str, short: int = 5, long: int = 20, warmup: int | None = None):
        if short >= long:
            raise ValueError("short 必须小于 long")
        self.symbol = symbol
        self.short = short
        self.long = long
        self.warmup = warmup if warmup is not None else long
        self._prev_short: float | None = None
        self._prev_long: float | None = None

    def on_frame(self, frame, token, bus, host) -> list:
        closes = bus.get("close", self.symbol, window=self.long, token=token)
        if len(closes) < self.warmup:
            return []  # 预热不足

        short_ma = sum(closes[-self.short:]) / self.short
        long_ma = sum(closes[-self.long:]) / self.long

        buy_signal = (
            self._prev_short is not None
            and self._prev_short <= self._prev_long
            and short_ma > long_ma
        )
        sell_signal = (
            self._prev_short is not None
            and self._prev_short >= self._prev_long
            and short_ma < long_ma
        )
        self._prev_short = short_ma
        self._prev_long = long_ma

        pos = host.portfolio.position(self.symbol)
        orders: list = []
        if buy_signal and pos.net_size() == 0:
            orders.append(
                Order(
                    side=OrderSide.BUY,
                    symbol=self.symbol,
                    order_type=OrderType.MARKET,
                    budget=host.portfolio.cash,
                )
            )
        elif sell_signal and pos.net_size() > 0:
            orders.append(
                Order(
                    side=OrderSide.SELL,
                    symbol=self.symbol,
                    size=pos.net_size(),
                    order_type=OrderType.MARKET,
                )
            )
        return orders
