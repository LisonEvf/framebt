"""Execution：订单与撮合（§5）。

撮合以「价格-时间优先」为原则；MVP 每帧单 Bar 简化为按本帧开盘价撮合。

关键因果约束 I2：策略在帧 t 生成的订单，须经过延迟管线，最早在帧 t+delay
才进入撮合队列、以 t+delay 帧的市场价成交。延迟在 ``register(order, frame_id)``
时记录 placement_frame，process 时按 ``frame_id - placement_frame >= delay`` 判定。

支持的订单类型：MARKET / LIMIT / IOC / FOK / STOP / STOP_LIMIT（§5.1）。
条件单由引擎在帧间监控触发（§5.2），组单原子性（§5.3）见 ``groups``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.data_types import CausalToken  # noqa: F401
from ..core.events import (
    FillEvent,
    Order,
    OrderId,
    OrderSide,
    OrderState,
    OrderType,
)
from ..core.market_frame import MarketFrame


@dataclass
class FillResult:
    """单笔撮合结果。

    - fill 非空：已成交
    - fill 为空且 rejected=True：拒绝（丢弃，不再重试，如资金不足）
    - fill 为空且 rejected=False：未触发（限价单 resting，下一帧重试）
    """

    fill: Optional[FillEvent]
    rejected: bool
    reason: str = ""


class CancelResult:
    """撤单结果（§5.4）。"""

    def __init__(self, accepted: bool, reason: str = ""):
        self.accepted = accepted
        self.reason = reason


class MatchingEngine:
    """延迟管线 + 撮合。

    参数：
      delay_frames：订单从生成到可撮合的强制延迟帧数（默认 1，满足 I2）。
      fill_price_mode：市场价来源。'open' 用本帧开盘价（默认，最保守因果）。
    """

    def __init__(self, delay_frames: int = 1, fill_price_mode: str = "open"):
        if delay_frames < 1:
            raise ValueError("delay_frames 必须 >= 1（I2：帧 t 订单最早帧 t+1 成交）")
        self.delay_frames = delay_frames
        self.fill_price_mode = fill_price_mode
        self._in_flight: list[tuple[Order, int]] = []  # (order, placement_frame)
        self._resting: list[Order] = []  # 限价/条件单挂单
        self._order_counter = 0
        self.total_fills = 0
        self._group_of: dict = {}          # order_id -> OrderGroup
        self._pending_cancels: dict = {}   # order_id -> Order（OCO 联动待撤销）
        self._trail_best: dict = {}        # order_id -> 跟踪止损的极值参考价

    # ---- 订单组（§5.3 / OCO）----

    def register_group(self, group, frame_id: int) -> list[OrderId]:
        """登记一个订单组；返回分配的订单 ID 列表。

        ``all_or_none`` 且不可行时整组不下发（返回空）。
        """
        from .groups import Atomicity, LinkMode

        if group.atomicity is Atomicity.ALL_OR_NONE and not group.is_feasible():
            return []
        ids: list[OrderId] = []
        for o in group.orders:
            oid = self.register(o, frame_id)
            ids.append(oid)
            if group.link is LinkMode.OCO:
                self._group_of[oid] = group
        group.order_ids = ids
        return ids

    # ---- 订单生命周期 ----

    def register(self, order: Order, frame_id: int) -> OrderId:
        """登记一帧生成的订单，分配 ID，进入在途队列。"""
        self._order_counter += 1
        order.id = OrderId(self._order_counter)
        order.placement_frame = int(frame_id)
        order.state = OrderState.PENDING
        self._in_flight.append((order, int(frame_id)))
        return order.id

    def has_in_flight(self) -> bool:
        return bool(self._in_flight)

    def pending_orders(self) -> list[Order]:
        """返回所有未落地订单（在途 + 挂单）。

        策略用它判断「是否已有未成交的订单」，避免在成交落地前重复下单。
        """
        return [o for o, _ in self._in_flight] + list(self._resting)

    def pending_count(self) -> int:
        return len(self._in_flight) + len(self._resting)

    def find_order(self, order_id: OrderId) -> Optional[Order]:
        for order, _ in self._in_flight:
            if order.id == order_id:
                return order
        for order in self._resting:
            if order.id == order_id:
                return order
        return None

    def _tradable_price(self, frame: MarketFrame, symbol: str) -> float:
        """按配置取本帧可成交价。MVP 用开盘价（因果最保守）。"""
        if not frame.has_bar(symbol):
            raise KeyError(f"帧 {frame.frame_id} 无标的 {symbol} 的价格数据")
        if self.fill_price_mode == "open":
            return frame.open(symbol)
        if self.fill_price_mode == "close":
            return frame.close(symbol)
        if self.fill_price_mode == "mid":
            return frame.mid(symbol)
        raise ValueError(f"未知 fill_price_mode {self.fill_price_mode}")

    # ---- 撮合主循环 ----

    def process(
        self,
        frame: MarketFrame,
        frame_id: int,
        pre_trade_check: Optional[Callable[[Order, MarketFrame], Optional[str]]] = None,
    ) -> list[FillEvent]:
        """处理在途订单：未达延迟的继续等待；可达的尝试撮合。

        返回本帧所有 FillEvent。``pre_trade_check`` 为风控+规则中间件（返回拒绝原因或 None）。
        resting 的限价/条件单在本帧再次检查触发条件（§5.2）。

        条件单语义：带 ``condition`` 的订单在条件未满足前一律不成交（即便订单类型是市价），
        转入 resting 由引擎在后续帧监控（§5.2）。
        """
        fills: list[FillEvent] = []
        still_pending: list[tuple[Order, int]] = []
        still_resting: list[Order] = []

        # ---- 在途订单：达延迟后尝试撮合 ----
        for order, placed in self._in_flight:
            age = frame_id - placed
            if age < self.delay_frames:
                still_pending.append((order, placed))
                continue

            # 风控 + 规则盘前拦截（返回拒绝原因则丢弃）
            if pre_trade_check is not None:
                reason = pre_trade_check(order, frame)
                if reason is not None:
                    order.state = OrderState.REJECTED
                    continue

            # 条件单：未触发则不成交，转 resting（§5.2）
            if order.condition is not None and not self._check_condition(order, frame):
                order.state = OrderState.RESTING
                still_resting.append(order)
                continue

            self._settle(order, frame, frame_id, fills, still_resting)

        # ---- 既有 resting 单：重新检查条件并尝试撮合 ----
        # 注意：不可在遍历 still_resting 的同时向其追加（会无限循环）。
        for order in self._resting:
            if order.condition is not None and not self._check_condition(order, frame):
                still_resting.append(order)
                continue
            if pre_trade_check is not None:
                reason = pre_trade_check(order, frame)
                if reason is not None:
                    order.state = OrderState.REJECTED
                    continue
            self._settle(order, frame, frame_id, fills, still_resting)

        self._in_flight = still_pending
        self._resting = still_resting

        # ---- OCO 联动：把已成交订单的兄弟腿撤销（§5.1）----
        # 延迟到循环之后统一处理，避免遍历中修改在途/挂单列表。
        if self._pending_cancels:
            cancelled_ids = set(self._pending_cancels)
            self._in_flight = [(o, p) for o, p in self._in_flight if o.id not in cancelled_ids]
            self._resting = [o for o in self._resting if o.id not in cancelled_ids]
            for o in self._pending_cancels.values():
                o.state = OrderState.CANCELLED
            self._pending_cancels.clear()

        return fills

    def _settle(
        self,
        order: Order,
        frame: MarketFrame,
        frame_id: int,
        fills: list[FillEvent],
        still_resting: list[Order],
    ) -> None:
        """尝试撮合一笔已达延迟且条件已满足的订单。"""
        result = self._try_fill(order, frame, frame_id)
        if result.fill is not None:
            fills.append(result.fill)
            self.total_fills += 1
            self._record_oco_siblings(order)
        elif result.rejected:
            order.state = OrderState.REJECTED
        else:
            # 未触发但可重试：限价单继续 resting
            order.state = OrderState.RESTING
            still_resting.append(order)

    def _record_oco_siblings(self, order: Order) -> None:
        """成交后登记 OCO 兄弟腿待撤销（延后到 process 末尾统一处理）。"""
        grp = self._group_of.get(order.id)
        if grp is None:
            return
        for oid in grp.order_ids:
            if oid != order.id:
                sib = self.find_order(oid)
                if sib is not None:
                    self._pending_cancels[oid] = sib

    def cancel(self, order_id: OrderId, frame_id: int) -> CancelResult:
        """撤单：与成交赛跑（§5.4）。返回是否成功。"""
        for i, (order, _placed) in enumerate(self._in_flight):
            if order.id == order_id:
                self._in_flight.pop(i)
                order.state = OrderState.CANCELLED
                return CancelResult(True)
        for i, order in enumerate(self._resting):
            if order.id == order_id:
                self._resting.pop(i)
                order.state = OrderState.CANCELLED
                return CancelResult(True)
        return CancelResult(False, "订单不存在或已成交/执行")

    # ---- 触发条件（§5.2）----

    def _check_condition(self, order: Order, frame: MarketFrame) -> bool:
        cond = order.condition
        if cond is None:
            return False
        if not frame.has_bar(order.symbol):
            return False
        q = frame.bbo(order.symbol)
        return cond.is_met(q)

    # ---- 单笔撮合 ----

    def _try_fill(self, order: Order, frame: MarketFrame, frame_id: int) -> FillResult:
        symbol = order.symbol
        if not frame.has_bar(symbol):
            return FillResult(None, True, "无标的价格数据")

        px = self._tradable_price(frame, symbol)
        ts = frame.end_ns

        otype = order.order_type

        # ---- 市价单 ----
        if otype == OrderType.MARKET:
            size = order.size
            if order.budget > 0:
                size = math.floor(order.budget / px)
            if size <= 0:
                return FillResult(None, True, "数量为零（资金不足）")
            return self._make_fill(order, size, px, frame_id, ts)

        # ---- IOC：立即成交或取消（单帧单 Bar 下等于市价，不足则取消）----
        if otype == OrderType.IOC:
            size = order.size
            if order.budget > 0:
                size = math.floor(order.budget / px)
            if size <= 0:
                return FillResult(None, True, "IOC 数量为零")
            return self._make_fill(order, size, px, frame_id, ts)

        # ---- FOK：全部成交或取消 ----
        if otype == OrderType.FOK:
            size = order.size
            if order.budget > 0:
                size = math.floor(order.budget / px)
            if size <= 0:
                return FillResult(None, True, "FOK 数量为零")
            return self._make_fill(order, size, px, frame_id, ts)

        # ---- TRAILING_STOP：触发价随有利方向移动（§5.1）----
        if otype == OrderType.TRAILING_STOP:
            last = frame.last(symbol)
            bought = order.side is OrderSide.BUY
            best = self._trail_best.get(order.id)
            if best is None:
                # 首次评估：以订单 price（若给出，通常为入场价）为参考，否则用当帧极值
                if order.price > 0:
                    best = order.price
                else:
                    best = frame.low(symbol) if bought else frame.high(symbol)
            if bought:  # 保护空头：追踪最低价，反弹超过比例则触发买入
                best = min(best, frame.low(symbol))
                trigger = best * (1 + order.trail_pct)
                triggered = last >= trigger
            else:       # 保护多头：追踪最高价，回落超过比例则触发卖出
                best = max(best, frame.high(symbol))
                trigger = best * (1 - order.trail_pct)
                triggered = last <= trigger
            self._trail_best[order.id] = best
            if not triggered:
                return FillResult(None, False)  # 未触发 → resting，下一帧继续跟踪

            size = order.size or self._budget_size(order, px)
            if size <= 0:
                return FillResult(None, True, "数量为零")
            self._trail_best.pop(order.id, None)
            return self._make_fill(order, size, px, frame_id, ts)

        # ---- STOP / STOP_LIMIT：触发价达到后转市价/限价 ----
        if otype in (OrderType.STOP, OrderType.STOP_LIMIT):
            trigger = order.price
            last = frame.last(symbol)
            bought = order.side is OrderSide.BUY
            # 卖出止损：最新价跌破触发价；买入启动：最新价突破触发价
            triggered = (bought and last >= trigger) or (not bought and last <= trigger)
            if not triggered:
                return FillResult(None, False)  # 未触发 → resting，下一帧再查

            size = order.size or self._budget_size(order, px)
            if size <= 0:
                return FillResult(None, True, "数量为零")

            if otype == OrderType.STOP:
                return self._make_fill(order, size, px, frame_id, ts)
            # STOP_LIMIT：以触发价或更优价格成交，否则 resting
            if (bought and px <= trigger) or (not bought and px >= trigger):
                return self._make_fill(order, size, px, frame_id, ts)
            return FillResult(None, False)

        # ---- 限价单 ----
        if otype == OrderType.LIMIT:
            if order.side is OrderSide.BUY:
                if px <= order.price:
                    size = order.size or self._budget_size(order, px)
                    return self._make_fill(order, size, px, frame_id, ts)
            else:
                if px >= order.price:
                    size = order.size or self._budget_size(order, px)
                    return self._make_fill(order, size, px, frame_id, ts)
            # 未达触发价，resting
            return FillResult(None, False)

        return FillResult(None, True, f"未知订单类型 {otype.value}")

    def _budget_size(self, order: Order, px: float) -> float:
        if order.budget > 0:
            return math.floor(order.budget / px)
        return order.size

    def _make_fill(self, order: Order, size: float, px: float, frame_id: int, ts: int) -> FillResult:
        fill = FillEvent(
            order_id=order.id or 0,
            side=order.side,
            symbol=order.symbol,
            size=size,
            price=px,
            frame_id=frame_id,
            timestamp=ts,
        )
        order.state = OrderState.FILLED
        return FillResult(fill, False)


class OrderBook:  # pragma: no cover - Phase 2+ 订单簿占位
    """价格-时间优先订单簿（MVP 用单 Bar 撮合替代，Phase 2 补真实订单簿）。"""

    def __init__(self, symbol: str):
        self.symbol = symbol
