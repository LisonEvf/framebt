"""实盘复用地基（Phase 6 · ①）。

「回测 / 实盘同一套策略与引擎调度」的落地载体在本模块：

  - ``LiveFeed``      —— ``MarketFeed`` 的实盘实现，把「外部实时流」缓存成有序 Bar。
  - ``LiveConnector`` —— ``ExecutionBackend`` 的实盘实现，封装券商 / CTP 订单流。
  - ``Transport``      —— 券商 / venue 的抽象接口（报单 / 撤单 / 收成交 / 查状态）。
  - ``SimulatedTransport`` —— 确定性的「参考 venue」，用于离线模拟与测试，
                            其成交语义与 ``MatchingEngine`` 对齐。

边界契约（与回测完全一致，切换零改动）：

  - ``MarketFeed.bars``      —— 按 ``(timestamp, sequence_id)`` 升序的有序 Bar。
  - ``ExecutionBackend``      —— ``submit / cancel / process / has_in_flight /
                                  pending_orders / register_group``。

引擎只依赖这两个抽象，故同一策略 + 同一 Engine 调度逻辑在回测（``BarFeed`` +
``MatchingEngine``）与实盘（``LiveFeed`` + ``LiveConnector``）之间零改动切换。
"""

from __future__ import annotations

from typing import Callable, Optional

from .core.events import (
    Bar,
    FillEvent,
    Order,
    OrderId,
    OrderSide,
    OrderState,
    OrderType,
)
from .core.market_frame import MarketFrame
from .execution.backend import ExecutionBackend
from .execution.engine import CancelResult
from .feed import MarketFeed


# ===========================================================================
# LiveFeed —— MarketFeed 的实盘实现
# ===========================================================================

class LiveFeed(MarketFeed):
    """把「外部实时流」缓存成有序 Bar 的实盘数据源（``MarketFeed`` 实盘实现）。

    与回测的 ``BarFeed`` 不同，``LiveFeed`` 支持**增量喂入**：实时流逐根 Bar 到达时
    调用 ``push``，内部按 ``(timestamp, sequence_id)`` 维护升序；对外仍暴露与
    ``BarFeed`` 完全相同的 ``bars`` 语义，故引擎（只读有序 Bar）无需任何改动。

    设计取舍：
      - 引擎 ``run()`` 读 ``feed.bars`` 全量列表 —— 故本类把已到达的 Bar 缓存为有序列表。
        真正的「流式拉取、边跑边喂」引擎集成是后续阶段（⑤ 实盘 connector 的流式驱动）；
        本类先保证「实盘数据源遵守 MarketFeed 契约」这一边界正确。
      - ``push`` 自动分配 ``sequence_id``（与 ``BarFeed`` 同口径，保 I4）。
    """

    def __init__(self, bars: Optional[list[Bar]] = None):
        self._bars: list[Bar] = []
        self._seq = 0
        if bars:
            for b in bars:
                self.push(b)

    def push(self, bar: Bar, *, ensure_sorted: bool = True) -> None:
        """增量喂入一根 Bar（实盘实时流到达时调用）。

        未显式指定 ``sequence_id`` 的 Bar 按到达顺序分配（保 I4 全序）；
        已显式指定则原样保留（尊重上游消息序号）。对外仍按 ``(timestamp, sequence_id)``
        升序暴露，与 ``BarFeed`` 同口径。
        """
        if bar.sequence_id == 0:
            bar = Bar(**{**bar.__dict__, "sequence_id": self._seq})
        self._bars.append(bar)
        if ensure_sorted:
            self._bars.sort(key=lambda b: (b.timestamp, b.sequence_id))
        self._seq = max(self._seq, max((b.sequence_id for b in self._bars), default=-1) + 1)

    @property
    def bars(self) -> list[Bar]:
        return self._bars

    def reset(self) -> None:
        """清空缓存（新交易日 / 重跑时调用）。"""
        self._bars.clear()
        self._seq = 0


# ===========================================================================
# Transport —— 券商 / venue 抽象接口
# ===========================================================================

class Transport:
    """券商 / venue 的抽象接口（实盘订单流的入口）。

    真券商 / CTP 实现 ``Transport`` 即可切到实盘；测试用 ``SimulatedTransport``。
    ``LiveConnector`` 只依赖本接口，故不依赖具体券商。
    """

    def submit(self, order: Order) -> OrderId:
        """向 venue 报单，返回 venue 侧订单 ID。"""
        raise NotImplementedError

    def cancel(self, venue_id: OrderId) -> bool:
        """向 venue 发撤单，返回是否受理。"""
        raise NotImplementedError

    def get_fills(self, frame: MarketFrame, frame_id: int) -> list[tuple[OrderId, float, float]]:
        """返回自上次调用以来 venue 的新成交 ``(venue_id, size, price)``。

        ``frame_id`` 供 venue 执行「延迟管线」语义（订单达延迟后才可撮合，与
        ``MatchingEngine`` 的 ``delay_frames`` 同构）。
        """
        raise NotImplementedError

    def get_state(self, venue_id: OrderId) -> OrderState:
        """查 venue 订单状态（FILLED / PENDING / RESTING / CANCELLED / REJECTED）。"""
        raise NotImplementedError

    def available_to_borrow(self, symbol: str) -> float:
        """该标的可融券数量（A 股做空用）；默认 0（不可做空）。"""
        return 0.0


# ===========================================================================
# SimulatedTransport —— 确定性的参考 venue
# ===========================================================================

class SimulatedTransport(Transport):
    """确定性的「参考 venue」：成交语义与 ``MatchingEngine`` 对齐（离线模拟 / 测试）。

    它复用了 MatchingEngine 的核心语义（延迟 1 帧、市价单按开盘价、限价单帧内
    价路径 crossing），但与 ``MatchingEngine`` 是**独立实现** —— 用它证明
    ``LiveConnector`` 与回测后端在模拟场景下逐笔一致。
    """

    def __init__(self, delay: int = 1):
        self.delay = delay
        self._venue_counter = 0
        # (order, placement_frame, venue_id)
        self._in_flight: list[tuple[Order, int, OrderId]] = []
        self._states: dict[OrderId, OrderState] = {}

    def submit(self, order: Order) -> OrderId:
        self._venue_counter += 1
        venue_id = OrderId(self._venue_counter)
        order.id = venue_id
        order.state = OrderState.PENDING
        self._states[venue_id] = OrderState.PENDING
        # placement_frame 由 LiveConnector.submit 填写；直调时回退 0
        placement = int(order.placement_frame or 0)
        self._in_flight.append((order, placement, venue_id))
        return venue_id

    def cancel(self, venue_id: OrderId) -> bool:
        for i, (_, _, vid) in enumerate(self._in_flight):
            if vid == venue_id:
                del self._in_flight[i]
                self._states[venue_id] = OrderState.CANCELLED
                return True
        return False

    def get_fills(self, frame: MarketFrame, frame_id: int) -> list[tuple[OrderId, float, float]]:
        # 延迟管线：未达延迟的订单跳过（保留在 in_flight，帧 2 再撮合）
        fills: list[tuple[OrderId, float, float]] = []
        still: list[tuple[Order, int, OrderId]] = []
        for order, placed, venue_id in self._in_flight:
            if frame_id - placed < self.delay:
                still.append((order, placed, venue_id))  # 未达延迟，保留待后续帧
                continue
            if order.state in (OrderState.FILLED, OrderState.CANCELLED, OrderState.REJECTED):
                continue  # 已落地，由 LiveConnector 从 in_flight 移除
            # 达延迟后按开盘价 / 帧内 crossing 撮合（与 MatchingEngine 同语义）
            fill = self._simulate_fill(order, frame)
            if fill is not None:
                size, px = fill
                fills.append((venue_id, size, px))
                self._states[venue_id] = OrderState.FILLED
                order.state = OrderState.FILLED
            else:
                still.append((order, placed, venue_id))
        self._in_flight = still
        return fills

    def _simulate_fill(self, order: Order, frame: MarketFrame):
        if not frame.has_bar(order.symbol):
            return None
        px = frame.open(order.symbol)
        if order.order_type is OrderType.MARKET:
            return order.size, px
        # 限价单：帧内价路径 crossing（与 MatchingEngine._try_fill LIMIT 分支同）
        if order.side is OrderSide.BUY:
            if frame.low(order.symbol) <= order.price:
                fill_px = frame.open(order.symbol) if frame.open(order.symbol) <= order.price else frame.low(order.symbol)
                return order.size, fill_px
        else:
            if frame.high(order.symbol) >= order.price:
                fill_px = frame.open(order.symbol) if frame.open(order.symbol) >= order.price else frame.high(order.symbol)
                return order.size, fill_px
        return None

    def get_state(self, venue_id: OrderId) -> OrderState:
        return self._states.get(venue_id, OrderState.REJECTED)

    def available_to_borrow(self, symbol: str) -> float:
        return 0.0


# ===========================================================================
# LiveConnector —— ExecutionBackend 的实盘实现
# ===========================================================================

class LiveConnector(ExecutionBackend):
    """封装券商 / venue 订单流的实盘执行后端（``ExecutionBackend`` 实盘实现）。

    与回测的 ``MatchingEngine`` 遵守同一契约：
      - ``submit`` 经 ``transport`` 报单，返回引擎侧 ``OrderId``（= venue_id）。
      - ``cancel`` 经 ``transport`` 撤单。
      - ``process`` 轮询 ``transport.get_fills``，把 venue 成交转成 ``FillEvent``。
      - ``has_in_flight / pending_orders / find_order`` 查 ``transport.get_state``。
      - ``register_group`` 委托 venue 或本地模拟 OCO。

    切换回测 / 实盘 = 换 ``Transport`` 实现（真券商 / ``SimulatedTransport``），
    策略与 Engine 调度逻辑零改动。
    """

    def __init__(
        self,
        transport: Transport,
        *,
        delay: int = 1,
        pre_trade_check: Optional[Callable[[Order, MarketFrame], Optional[str]]] = None,
    ):
        self.transport = transport
        self.delay = delay
        self.pre_trade_check = pre_trade_check
        self._in_flight: list[tuple[Order, int]] = []  # (order, placement_frame)
        self._counter = 0
        self.total_fills = 0
        self._group_of: dict = {}

    # ---- 订单生命周期 ----

    def submit(self, order: Order, frame_id: int) -> OrderId:
        self._counter += 1
        order.placement_frame = int(frame_id)
        order.state = OrderState.PENDING
        venue_id = self.transport.submit(order)
        order.id = venue_id
        self._in_flight.append((order, int(frame_id)))
        return venue_id

    def register(self, order: Order, frame_id: int) -> OrderId:
        """``submit`` 的兼容别名。"""
        return self.submit(order, frame_id)

    def cancel(self, order_id: OrderId, frame_id: int) -> CancelResult:
        if self.transport.cancel(order_id):
            for i, (order, _) in enumerate(self._in_flight):
                if order.id == order_id:
                    del self._in_flight[i]
                    order.state = OrderState.CANCELLED
                    return CancelResult(True)
            return CancelResult(True)  # venue 已受理，本地队列可能已无（已成交）
        return CancelResult(False, "venue 未受理或订单已成交")

    def has_in_flight(self) -> bool:
        return bool(self._in_flight)

    def pending_orders(self) -> list[Order]:
        return [o for o, _ in self._in_flight]

    def register_group(self, group, frame_id: int) -> list[OrderId]:
        from .execution.groups import Atomicity, LinkMode

        # all_or_none 原子性：回测侧判定可行性，避免整组下发不可行的组单
        if group.atomicity is Atomicity.ALL_OR_NONE and not group.is_feasible():
            return []
        ids = []
        for o in group.orders:
            oid = self.submit(o, frame_id)
            ids.append(oid)
            if group.link is LinkMode.OCO:
                self._group_of[oid] = group
        group.order_ids = ids
        return ids

    # ---- 撮合主循环 ----

    def process(
        self,
        frame: MarketFrame,
        frame_id: int,
        pre_trade_check: Optional[Callable[[Order, MarketFrame], Optional[str]]] = None,
    ) -> list[FillEvent]:
        checks = self.pre_trade_check if pre_trade_check is None else pre_trade_check
        venue_fills = self.transport.get_fills(frame, frame_id)
        fills: list[FillEvent] = []
        done_ids: set = set()
        for venue_id, size, px in venue_fills:
            order = self._find_in_flight(venue_id)
            if order is None:
                continue  # 已处理或未知
            if checks is not None:
                reason = checks(order, frame)
                if reason is not None:
                    order.state = OrderState.REJECTED
                    self.transport.cancel(venue_id)
                    done_ids.add(venue_id)
                    continue
            fills.append(
                FillEvent(
                    order_id=venue_id,
                    side=order.side,
                    symbol=order.symbol,
                    size=size,
                    price=px,
                    frame_id=frame_id,
                    timestamp=frame.end_ns,
                )
            )
            order.state = OrderState.FILLED
            self.total_fills += 1
            done_ids.add(venue_id)

        # 移除已落地订单；未成交的继续保留在 in_flight（RESTING）
        self._in_flight = [(o, p) for o, p in self._in_flight if o.id not in done_ids]

        # OCO 联动：成交订单的兄弟腿撤销（仅撤销尚未成交的兄弟）
        if self._group_of:
            cancelled: set = set()
            for oid in list(self._group_of):
                if oid in done_ids:
                    for sib_oid in self._group_of[oid].order_ids:
                        if sib_oid != oid and sib_oid not in done_ids:
                            cancelled.add(sib_oid)
            # 先标记状态，再从 in_flight 移除（否则标记时已不在队列）
            for o, _ in list(self._in_flight):
                if o.id in cancelled:
                    o.state = OrderState.CANCELLED
            self._in_flight = [(o, p) for o, p in self._in_flight if o.id not in cancelled]
            self._group_of.clear()

        return fills

    def _find_in_flight(self, venue_id: OrderId) -> Optional[Order]:
        for order, _ in self._in_flight:
            if order.id == venue_id:
                return order
        return None

    # ---- 跨后端通用默认实现（pending_count / find_order）由 ExecutionBackend 给出 ----
