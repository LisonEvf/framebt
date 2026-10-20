"""Execution：订单与撮合（§5）。

撮合以「价格-时间优先」为原则。MVP 每帧单 Bar 简化为按本帧开盘价撮合；
Phase 2 升级为「帧内子步进执行引擎」：

- **价格-时间优先订单簿**（``OrderBook``）：每个标的维护买 / 卖两侧，每侧按
  价位分层，同一价位内 FIFO。resting 的限价单入队，成交 / 撤单出队；
  ``queue_position`` / ``depth`` 给出「队列位置 / 市场深度」—— 队列越靠后，
  帧内成交量分摊时越晚被喂到，成交概率越低。
- **帧内价路径 crossing**：限价单在本帧价路径触及限价时即触发（买限价单
  ``low <= 限价``、卖限价单 ``high >= 限价``），而非仅看开盘价。
  成交价取「限价或更优」：开盘已更优则按开盘价，否则按限价成交
  （不按当日极值 low/high —— 那等于事后知道最低/最高点）。
- **成交量分配模型（FIFO 成交量分摊）**：本帧已触发价的限价单按价格-时间优先
  分摊该标的当日成交量；成交量受限时部分成交，剩余量记入 ``_order_remaining``，
  下帧以剩余量继续参与分配（逐位可复现、无需 RNG）。

关键因果约束 I2：策略在帧 t 生成的订单，须经过延迟管线，最早在帧 t+delay
才进入撮合队列。延迟在 ``register(order, frame_id)`` 时记录 placement_frame，
process 时按 ``frame_id - placement_frame >= delay`` 判定。

支持的订单类型：MARKET / LIMIT / IOC / FOK / STOP / STOP_LIMIT（§5.1）。
条件单由引擎在帧间监控触发（§5.2），组单原子性（§5.3）见 ``groups``。
"""

from __future__ import annotations

import math
from collections import defaultdict
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
from .backend import ExecutionBackend  # noqa: E402
from .book import OrderBook  # noqa: E402


@dataclass
class FillResult:
    """单笔撮合结果。

    - fill 非空：已成交
    - fill 为空且 rejected=True：拒绝（丢弃，不再重试，如资金不足）
    - fill 为空且 rejected=False：未触发（限价单 resting，下一帧重试）
    - ``reached=True``：限价单已触发价格（帧内价路径触及限价），进入成交量分配
      （``fill_px`` / ``fill_size`` 给出成交价与名义数量），由引擎批量分摊本帧成交量。
    """

    fill: Optional[FillEvent]
    rejected: bool
    reason: str = ""
    reached: bool = False
    fill_px: Optional[float] = None
    fill_size: Optional[float] = None
    cancelled: bool = False  # IOC 余量 / FOK 不足 → 撤单，而不是拒绝或 resting


class CancelResult:
    """撤单结果（§5.4）。"""

    def __init__(self, accepted: bool, reason: str = ""):
        self.accepted = accepted
        self.reason = reason


class MatchingEngine(ExecutionBackend):
    """延迟管线 + 撮合（ExecutionBackend 的回测实现）。

    参数：
      delay_frames：订单从生成到可撮合的强制延迟帧数（默认 1，满足 I2）。
      fill_price_mode：市场价来源。'open' 用本帧开盘价（默认，最保守因果）。

    ``MatchingEngine`` 是 ``ExecutionBackend`` 的回测后端；实盘用 ``LiveConnector``
    （Phase 6 接入券商时实现），遵守同一契约，使策略与引擎在回测 / 实盘间零改动切换。
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
        self._order_remaining: dict[OrderId, float] = {}  # 部分成交：挂单剩余量（Phase 2）
        self._book: dict[str, OrderBook] = {}  # 标的 → 价格-时间优先订单簿（Phase 2）
        self._group_of: dict = {}          # order_id -> OrderGroup
        self._pending_cancels: dict = {}   # order_id -> Order（OCO 联动待撤销）
        self._trail_best: dict = {}        # order_id -> 跟踪止损的极值参考价
        self._last_frame: Optional[MarketFrame] = None  # 最近一帧（供 estimated_fill_fraction）

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

    def submit(self, order: Order, frame_id: int) -> OrderId:
        """提交一帧生成的订单，分配 ID，进入在途队列（``ExecutionBackend.submit``）。

        这是后端抽象的规范入口；``register`` 是其兼容别名（保留以不破坏既有调用）。
        """
        self._order_counter += 1
        order.id = OrderId(self._order_counter)
        order.placement_frame = int(frame_id)
        order.state = OrderState.PENDING
        self._in_flight.append((order, int(frame_id)))
        return order.id

    def register(self, order: Order, frame_id: int) -> OrderId:
        """登记一帧生成的订单（``submit`` 的兼容别名）。"""
        return self.submit(order, frame_id)

    def has_in_flight(self) -> bool:
        return bool(self._in_flight)

    def pending_orders(self) -> list[Order]:
        """返回所有未落地订单（在途 + 挂单）。

        策略用它判断「是否已有未成交的订单」，避免在成交落地前重复下单。
        """
        return [o for o, _ in self._in_flight] + list(self._resting)

    # ``pending_count`` / ``find_order`` 由 ``ExecutionBackend`` 给出跨后端通用默认
    # 实现（仅依赖 ``pending_orders``），此处无需重复编写。

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
        reached: list[tuple[Order, float, float]] = []  # (order, fill_px, fill_size)

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

            # 条件单：未触发 → resting；刚触发 → 记下 armed 帧并 resting，
            # 下一帧开盘才成交（条件用收盘判定，不能回头用当帧开盘价成交）。
            if order.condition is not None and order.condition_armed is None:
                if not self._check_condition(order, frame):
                    order.state = OrderState.RESTING
                    still_resting.append(order)
                    continue
                order.condition_armed = frame_id
                order.state = OrderState.RESTING
                still_resting.append(order)
                continue

            self._apply_result(order, self._try_fill(order, frame, frame_id),
                               fills, still_resting, reached)

        # ---- 既有 resting 单：重新检查条件并尝试撮合 ----
        # 注意：不可在遍历 still_resting 的同时向其追加（会无限循环）。
        for order in self._resting:
            if order.condition is not None and order.condition_armed is None:
                if not self._check_condition(order, frame):
                    still_resting.append(order)
                    continue
                order.condition_armed = frame_id
                still_resting.append(order)
                continue
            if pre_trade_check is not None:
                reason = pre_trade_check(order, frame)
                if reason is not None:
                    order.state = OrderState.REJECTED
                    continue
            self._apply_result(order, self._try_fill(order, frame, frame_id),
                               fills, still_resting, reached)

        self._in_flight = still_pending
        # 先落地未触发 resting 单：使 _allocate_volume 内 _record_oco_siblings 的
        # find_order 能在 _resting 中查到待撤销的 OCO 兄弟腿。
        self._resting = still_resting

        # ---- Phase 2：限价单成交量分配（价格-时间优先，FIFO 成交量分摊）----
        # 本帧已触发价（reached）的限价单按价格-时间优先分摊本帧成交量：
        #   买单高价先、卖价低价先；同价按到达顺序（FIFO）；成交量耗尽则继续 resting。
        # 部分成交：剩余量记入 _order_remaining，下帧以剩余量继续参与分配。
        unfilled = self._allocate_volume(reached, frame, frame_id, fills)
        for o in unfilled:
            o.state = OrderState.RESTING
        self._resting = self._resting + unfilled

        # 同步价格-时间优先订单簿（供队列位置 / 深度查询）
        self._sync_book()
        # 记录最近一帧（供 estimated_fill_fraction 取本帧成交量）
        self._last_frame = frame

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

    def _apply_result(
        self,
        order: Order,
        result: FillResult,
        fills: list[FillEvent],
        still_resting: list[Order],
        reached: list[tuple[Order, float, float]],
    ) -> None:
        """把单笔撮合结果落到对应队列。

        - 成交：入 fills + 登记 OCO 兄弟腿。
        - reached：限价单已触发价，入 reached 列表交成交量分配（本帧不直接成交）。
        - rejected：丢弃（不再重试）。
        - 未触发：resting（下帧重评估）。
        """
        if result.fill is not None:
            fills.append(result.fill)
            self.total_fills += 1
            self._record_oco_siblings(order)
        elif result.reached:
            assert result.fill_px is not None and result.fill_size is not None
            reached.append((order, result.fill_px, result.fill_size))
        elif result.rejected:
            order.state = OrderState.REJECTED
        elif result.cancelled:
            order.state = OrderState.CANCELLED
        else:
            order.state = OrderState.RESTING
            still_resting.append(order)

    # ---- Phase 2：成交量分配（价格-时间优先，FIFO 成交量分摊）----

    def _allocate_volume(
        self,
        reached: list[tuple[Order, float, float]],
        frame: MarketFrame,
        frame_id: int,
        fills: list[FillEvent],
    ) -> list[Order]:
        """对本帧已触发价的限价单按价格-时间优先分摊本帧成交量。

        每个标的当日成交量按「价格-时间优先」喂给该标的的挂单队列：
          - 买单：价高者优先；卖单：价低者优先；同价按到达顺序（FIFO，见 order.id）。
          - 每笔取 min(剩余量, 剩余成交量)；成交量耗尽则剩余挂单继续 resting。
          - 部分成交：剩余量记入 ``_order_remaining``，下帧以剩余量继续参与分配。

        返回本帧未全额成交、需继续 resting 的订单。
        """
        by_symbol: dict[str, list[tuple[Order, float, float]]] = defaultdict(list)
        for order, px, size in reached:
            by_symbol[order.symbol].append((order, px, size))

        unfilled: list[Order] = []
        for symbol, legs in by_symbol.items():
            if not frame.has_bar(symbol):
                continue
            vol = frame.volume(symbol) or 0
            # 价格-时间优先：买单高价先、卖价低价先；同价按到达顺序（id 递增 = 到达先后）。
            legs.sort(key=lambda it: (
                -it[0].price if it[0].side is OrderSide.BUY else it[0].price,
                it[0].id,
            ))
            remaining = vol
            for order, px, size in legs:
                if remaining <= 0:
                    # 成交量耗尽：本单及之后（价格-时间优先更靠后）的单继续 resting
                    unfilled.append(order)
                    continue
                # 部分成交剩余量：优先用已登记的剩余；否则按名义数量
                rem = self._order_remaining.get(order.id, size)
                if rem <= 0:
                    unfilled.append(order)
                    continue
                fill_i = int(min(rem, remaining))
                if fill_i <= 0:
                    unfilled.append(order)
                    continue
                res = self._make_fill(order, fill_i, px, frame_id, frame.end_ns)
                if res is None or res.fill is None:
                    continue
                fills.append(res.fill)
                self.total_fills += 1
                self._record_oco_siblings(order)
                remaining -= fill_i
                new_rem = rem - fill_i
                if new_rem <= 0:
                    order.state = OrderState.FILLED
                    self._order_remaining.pop(order.id, None)
                else:
                    # 部分成交：剩余量继续 resting（下帧以剩余量参与分配）
                    self._order_remaining[order.id] = new_rem
                    unfilled.append(order)
        return unfilled

    def _sync_book(self) -> None:
        """把当前 resting 订单同步进价格-时间优先订单簿（标的 → OrderBook）。

        订单簿只容纳限价挂单（``order.price > 0``）；市价 / 止损等即时成交类
        不在簿中排队，跳过。
        """
        self._book = {}
        for order in self._resting:
            if order.order_type is not OrderType.LIMIT:
                continue
            book = self._book.get(order.symbol)
            if book is None:
                book = OrderBook(order.symbol)
                self._book[order.symbol] = book
            book.add(order)

    def estimated_fill_fraction(self, order: Order) -> float:
        """估计某 resting 限价单在「最近一帧」内被成交的比例（确定性，无 RNG，保 I3）。

        帧内成交量按价格-时间优先 FIFO 喂给该标的的 resting 挂单。模型综合两个维度：
          - 队列位置 ``queue_position``：本帧成交量先喂队列前面的单，位置越靠后越先被
            「成交量耗尽」卡住（FIFO 加权）。
          - 价位深度 ``depth``：同价位挂单越多，本单分到的成交量份额越小。

        ``fill_fraction = min(1.0, 本帧成交量 / 该价位挂单总量) × 1/(queue_position + 1)``

        归一到 ``[0, 1]``：``1`` = 预计本帧全额成交。resting 数据缺失（无订单簿 /
        非 resting / 市价 / 条件单）→ ``0``。

        注：这是给策略在 ``on_sub_step`` 里做「是否挂单 / 挂多大」的**启发式估计**，
        非精确成交预测；实际成交仍走确定性 FIFO 成交量分摊（``_allocate_volume``）。
        实盘 ``LiveConnector`` 未接入 venue 成交模型时由 ``ExecutionBackend`` 默认返回 0。
        """
        book = self._book.get(order.symbol)
        if book is None:
            return 0.0
        qp = book.queue_position(order)
        if qp < 0:
            return 0.0  # 不在簿内（未 resting / 已成交 / 市价 / 条件单）
        depth = book.depth(order)
        if depth <= 0:
            return 0.0
        frame = self._last_frame
        if frame is None or not frame.has_bar(order.symbol):
            return 0.0
        vol = frame.volume(order.symbol) or 0
        # 该价位挂单总量（shares）：depth 笔 × 该单量近似（启发式）
        level_shares = depth * (order.size or 1)
        coverage = min(1.0, vol / level_shares) if level_shares else 0.0
        fifo_weight = 1.0 / (qp + 1)
        return min(1.0, coverage * fifo_weight)

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
                self._order_remaining.pop(order.id, None)
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
            # 成交量约束：市价单受本帧成交量限制，超额部分不成交。
            # 否则大单压过低流动性标的时滑点 notional·k·(size/vol) 爆炸、费用超现金，
            # 使现金被超额扣成负数（全仓买入亦会因「未用部分 < 费用」而透支）。
            if frame.has_bar(symbol):
                vol = frame.volume(symbol)
                if vol and size > vol:
                    size = vol
            return self._make_fill(order, size, px, frame_id, ts)

        # ---- IOC：立即成交或取消（受当帧可成交量约束，余量取消）----
        if otype == OrderType.IOC:
            size = order.size
            if order.budget > 0:
                size = math.floor(order.budget / px)
            if size <= 0:
                return FillResult(None, True, "IOC 数量为零")
            vol = frame.volume(symbol) if frame.has_bar(symbol) else 0.0
            filled = min(size, vol) if vol > 0 else size
            if filled <= 0:
                return FillResult(None, False, "IOC 无可成交量", cancelled=True)
            res = self._make_fill(order, filled, px, frame_id, ts)
            if filled < size:
                order.state = OrderState.CANCELLED  # 余量立即取消
            return res

        # ---- FOK：全部成交或取消（可成交量不足则整单取消）----
        if otype == OrderType.FOK:
            size = order.size
            if order.budget > 0:
                size = math.floor(order.budget / px)
            if size <= 0:
                return FillResult(None, True, "FOK 数量为零")
            vol = frame.volume(symbol) if frame.has_bar(symbol) else 0.0
            if vol > 0 and size > vol:
                return FillResult(None, False, "FOK 可成交量不足，整单取消", cancelled=True)
            return self._make_fill(order, size, px, frame_id, ts)

        # ---- TRAILING_STOP：触发价随有利方向移动（§5.1）----
        # 用帧内 high/low 更新极值并检测反向穿越，而不是用收盘价触发、开盘价成交：
        # 后者会让「收盘才知道跌破」的止损回到开盘卖出（帧内前视）。
        # 路径假设：先更新有利极值、再检查不利极值（对多头是保守的：更早触发）。
        if otype == OrderType.TRAILING_STOP:
            open_px = frame.open(symbol)
            high_px = frame.high(symbol)
            low_px = frame.low(symbol)
            bought = order.side is OrderSide.BUY
            best = self._trail_best.get(order.id)
            if best is None:
                # 首次评估：以订单 price（若给出，通常为入场价）为参考，否则用当帧极值
                if order.price > 0:
                    best = order.price
                else:
                    best = low_px if bought else high_px
            # 先用「上一帧结束时」的触发价判断开盘跳空；开盘价发生在当帧极值之前，
            # 不能拿当帧刚更新的触发价回头套在开盘价上（否则会虚假跳空成交）。
            prev_trigger = (best * (1 + order.trail_pct) if bought
                            else best * (1 - order.trail_pct))
            if bought:
                if open_px >= prev_trigger:
                    fill_px = open_px          # 跳空高开：按开盘价
                else:
                    best = min(best, low_px)
                    trigger = best * (1 + order.trail_pct)
                    if high_px >= trigger:
                        fill_px = trigger      # 帧内反弹触及：按触发价
                    else:
                        self._trail_best[order.id] = best
                        return FillResult(None, False)
            else:
                if open_px <= prev_trigger:
                    fill_px = open_px          # 跳空低开：按开盘价
                else:
                    best = max(best, high_px)
                    trigger = best * (1 - order.trail_pct)
                    if low_px <= trigger:
                        fill_px = trigger      # 帧内回落触及：按触发价
                    else:
                        self._trail_best[order.id] = best
                        return FillResult(None, False)

            self._trail_best[order.id] = best
            size = order.size or self._budget_size(order, fill_px)
            if size <= 0:
                return FillResult(None, True, "数量为零")
            self._trail_best.pop(order.id, None)
            return self._make_fill(order, size, fill_px, frame_id, ts)

        # ---- STOP / STOP_LIMIT：触发价达到后转市价/限价 ----
        # 同样按帧内价路径判断：开盘已越过触发价 → 按开盘价（跳空）；否则
        # high/low 触及触发价 → 按触发价成交。绝不用收盘触发、开盘成交。
        if otype in (OrderType.STOP, OrderType.STOP_LIMIT):
            trigger = order.price
            open_px = frame.open(symbol)
            high_px = frame.high(symbol)
            low_px = frame.low(symbol)
            bought = order.side is OrderSide.BUY
            if bought:
                if open_px >= trigger:
                    fill_px = open_px
                elif high_px >= trigger:
                    fill_px = trigger
                else:
                    return FillResult(None, False)  # 未触发 → resting
            else:
                if open_px <= trigger:
                    fill_px = open_px
                elif low_px <= trigger:
                    fill_px = trigger
                else:
                    return FillResult(None, False)

            if otype == OrderType.STOP:
                size = order.size or self._budget_size(order, fill_px)
                if size <= 0:
                    return FillResult(None, True, "数量为零")
                return self._make_fill(order, size, fill_px, frame_id, ts)

            # STOP_LIMIT：Order 只有一个 price 字段，当前语义为「限价 = 触发价」。
            # 开盘已越过限价（买：open > trigger / 卖：open < trigger）则不成交，继续 resting。
            if bought and open_px > trigger:
                return FillResult(None, False)
            if (not bought) and open_px < trigger:
                return FillResult(None, False)
            size = order.size or self._budget_size(order, fill_px)
            if size <= 0:
                return FillResult(None, True, "数量为零")
            return self._make_fill(order, size, fill_px, frame_id, ts)

        # ---- 限价单（Phase 2：帧内价路径 crossing + 成交量分配）----
        if otype == OrderType.LIMIT:
            if not frame.has_bar(symbol):
                return FillResult(None, True, "无标的价格数据")
            bought = order.side is OrderSide.BUY
            limit_px = order.price
            open_px = frame.open(symbol)
            low_px = frame.low(symbol)
            high_px = frame.high(symbol)
            # 帧内价路径 crossing：买限价单 low≤限价、卖限价单 high≥限价 即触发。
            # 成交价取「限价或更优」：开盘已更优则按开盘价，否则按限价成交。
            # 不按当日极值（low/high）成交——那等于事后知道最低/最高点，偏乐观。
            if bought:
                reached = low_px <= limit_px
                fill_px = open_px if open_px <= limit_px else limit_px
            else:
                reached = high_px >= limit_px
                fill_px = open_px if open_px >= limit_px else limit_px
            if not reached:
                return FillResult(None, False)  # 未触发 → resting，下一帧重试
            size = order.size or self._budget_size(order, fill_px)
            if size <= 0:
                return FillResult(None, True, "数量为零（资金不足）")
            return FillResult(None, False, reached=True, fill_px=fill_px, fill_size=size)

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
