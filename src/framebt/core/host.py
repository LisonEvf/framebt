"""StrategyHost：唤醒分发 + 确定性 RNG + 账户访问（§10）。

引擎每帧调用 ``dispatch_frame`` 触发策略 ``on_frame``；策略返回的订单交由引擎登记。
RNG 由 host 持有并注入回调，保证 I3（策略不得使用外部随机源）。

唤醒类型在引擎推进时由 StrategyHost 分发（§10.1）。策略只注册它关心的类型，
未注册的回调即便被引擎调用也直接跳过——这解决了「所有策略被迫适配每帧主动检查」
的范式单一问题。
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Optional

from .determinism import DetermRNG
from .events import CancelEvent, FillEvent, Order
from .market_frame import MarketFrame


class WakeupType(str, Enum):
    FRAME = "on_frame"
    SUB_STEP = "on_sub_step"
    EVENT = "on_event"
    CONDITION = "on_condition"
    SCHEDULE = "on_schedule"
    FILL = "on_fill"
    LIFECYCLE = "on_lifecycle"


class StrategyHost:
    def __init__(self, seed: int = 0):
        self.strategy = None
        self.portfolio = None          # 由 Engine 注入（Ledger）
        self.matcher = None            # 由 Engine 注入（MatchingEngine），用于查询在途订单
        self.rng = DetermRNG(seed=seed)
        self._wakeups: set[WakeupType] = {WakeupType.FRAME}
        self._fill_callbacks: list[Callable[[FillEvent], None]] = []
        self._cancel_callbacks: list[Callable[[CancelEvent], None]] = []

    # ---- 在途订单查询 ----
    #
    # 订单有强制延迟（I2），因此「本帧下的单」不会立刻反映到持仓/现金上。
    # 策略若只看持仓做判断，就会在成交落地前**每帧重复下单**（典型后果：仓位爆炸）。
    # 因此宿主提供在途订单查询，让「避免重复下单」这件事可以被正确表达。

    def pending_orders(self, symbol: Optional[str] = None) -> list[Order]:
        """返回当前在途/挂单中的订单（可按标的过滤）。"""
        if self.matcher is None:
            return []
        return [o for o in self.matcher.pending_orders() if symbol is None or o.symbol == symbol]

    def has_pending(self, symbol: Optional[str] = None) -> bool:
        """是否有未落地的订单（策略用它避免重复下单）。"""
        return bool(self.pending_orders(symbol))

    def register(self, strategy, wakeups: Optional[list[WakeupType]] = None) -> None:
        """注册策略及其关心的唤醒类型（§10.1）。

        未显式给出 ``wakeups`` 时，采用策略自身声明的 ``default_wakeups``
        （而非硬编码只开 FRAME）——否则 ``on_fill`` 等钩子会被静默跳过。
        """
        self.strategy = strategy
        if wakeups:
            self._wakeups = set(wakeups)
        else:
            self._wakeups = set(getattr(strategy, "default_wakeups", None)
                                or [WakeupType.FRAME])

    def on_fill(self, cb: Callable[[FillEvent], None]) -> None:
        self._fill_callbacks.append(cb)

    def on_cancel(self, cb: Callable[[CancelEvent], None]) -> None:
        self._cancel_callbacks.append(cb)

    # ---- 分发入口 ----

    def _active(self, w: WakeupType) -> bool:
        return w in self._wakeups

    def dispatch_frame(
        self, frame: MarketFrame, token, bus
    ) -> list[Order]:
        """分发 ``on_frame``：调用策略，收集返回订单。"""
        if self.strategy is None:
            raise RuntimeError("未注册策略")
        if not self._active(WakeupType.FRAME):
            return []
        orders = self.strategy.on_frame(frame, token, bus, self)
        return orders or []

    def dispatch_sub_step(self, frame, step: int, total: int, token, bus) -> list[Order]:
        if not self._active(WakeupType.SUB_STEP):
            return []
        orders = self.strategy.on_sub_step(frame, step, total, token, bus)
        return orders or []

    def dispatch_event(self, event, token, bus) -> None:
        if self.strategy is None or not self._active(WakeupType.EVENT):
            return
        self.strategy.on_event(event, token, bus)

    def dispatch_condition(self, cond, token, bus) -> None:
        if self.strategy is None or not self._active(WakeupType.CONDITION):
            return
        self.strategy.on_condition(cond, token, bus)

    def dispatch_schedule(self, schedule, token, bus) -> None:
        if self.strategy is None or not self._active(WakeupType.SCHEDULE):
            return
        self.strategy.on_schedule(schedule, token, bus)

    def dispatch_fill(self, fill: FillEvent) -> list[Order]:
        for cb in self._fill_callbacks:
            cb(fill)
        orders: list[Order] = []
        if self.strategy is not None and self._active(WakeupType.FILL):
            orders = self.strategy.on_fill(fill) or []
        return orders

    def dispatch_cancel(self, event: CancelEvent) -> None:
        for cb in self._cancel_callbacks:
            cb(event)
        if self.strategy is not None and self._active(WakeupType.FILL):
            self.strategy.on_cancel(event)

    def dispatch_lifecycle(self, event, token, bus) -> None:
        if self.strategy is None or not self._active(WakeupType.LIFECYCLE):
            return
        self.strategy.on_lifecycle(event, token, bus)
