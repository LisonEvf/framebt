"""策略基类与唤醒类型（§10.1）。

策略继承 ``FrameStrategy``，覆写关心的回调。返回 ``list[Order]`` 交给引擎登记。
读取历史用 ``bus.get(key, symbol, window=..., token=token)``；查账户用
``host.portfolio``（Ledger）。

只有被注册的唤醒类型才会被引擎分发（见 ``StrategyHost.register``）。
"""

from __future__ import annotations

from .events import CancelEvent, FillEvent, Order
from .host import WakeupType


class FrameStrategy:
    """策略基类。

    覆写 ``on_frame`` 实现逻辑；返回 ``list[Order]`` 交给引擎登记。
    其余回调可选覆写，默认空实现。
    """

    #: 性能提示（Phase 1）：声明关注的因子/参数，**非白名单**。
    #: 未声明的请求实时计算并记录请求模式（§4.2）。
    prefetch_hints: dict = {}

    #: 默认注册的唤醒类型；引擎据此决定调度哪些回调。
    #: FILL 默认开启：``on_fill`` 是挂出止损/止盈等**被动单**的标准时机
    #: （成交后才知道实际成交价与数量），关闭它会让这类策略静默失效。
    default_wakeups: list[WakeupType] = [WakeupType.FRAME, WakeupType.FILL]

    def on_frame(
        self, frame, token, bus, host
    ) -> list[Order]:
        raise NotImplementedError

    # ---- 可选钩子（未注册时引擎不调度） ----

    def on_sub_step(self, frame, step: int, total: int, token, bus) -> list[Order]:
        return []

    def on_event(self, event, token, bus) -> None:  # pragma: no cover
        pass

    def on_condition(self, cond, token, bus) -> None:  # pragma: no cover
        pass

    def on_schedule(self, schedule, token, bus) -> None:  # pragma: no cover
        pass

    def on_fill(self, fill: FillEvent) -> None:  # pragma: no cover
        pass

    def on_cancel(self, event: CancelEvent) -> None:  # pragma: no cover
        pass

    def on_lifecycle(self, event, token, bus) -> None:  # pragma: no cover
        pass
