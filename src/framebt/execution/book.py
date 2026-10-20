"""订单簿：价格-时间优先（Phase 2 · 帧内子步进执行引擎）。

``OrderBook`` 是「市场自己认定的撮合结构」的显式建模：每个标的维护买 / 卖两侧，
每侧按价位分层，同一价位内按到达先后 FIFO（价格-时间优先的核心）。

引擎用它：
  - 把 resting 的限价单入队（``add``）；成交 / 撤单后出队（``remove``）。
  - ``queue_position`` 给出某挂单在其价位的 FIFO 位置（0 = 队首，最优先）——
    这是「队列位置」维度：队列越靠后，帧内成交量分摊时越晚被喂到，成交概率越低。
  - ``depth`` 给出某价位的挂单总量（市场深度）。

价格-时间优先（price-time priority）：
  - 买单：价格高者优先；同价按 FIFO。
  - 卖单：价格低者优先；同价按 FIFO。
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Optional

from ..core.events import Order, OrderId, OrderSide


class OrderBook:
    """单标的价格-时间优先订单簿。

    每个价位是一个 FIFO 队列（``deque``）；``_bids`` 买价升序排列、``_asks``
    卖价降序排列，便于引擎取最优价（BBO）。本类只维护「价格-时间」结构，
    不关心成交代价 / 数量分摊 —— 那由引擎的成交量分配模型（``MatchingEngine._allocate_volume``）
    基于本簿的队列顺序完成。
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._bids: dict[float, deque[Order]] = defaultdict(deque)
        self._asks: dict[float, deque[Order]] = defaultdict(deque)

    # ---- 入队 / 出队 ----

    def add(self, order: Order) -> None:
        """把挂单入队到其价位的 FIFO 队列尾部。价格必须为正。"""
        if order.price <= 0:
            raise ValueError("订单簿只接受正价格挂单")
        if order.side is OrderSide.BUY:
            self._bids[order.price].append(order)
        else:
            self._asks[order.price].append(order)

    def remove(self, order_id: OrderId) -> bool:
        """按 ID 出队（成交 / 撤单后调用）。返回是否命中。"""
        for levels in (self._bids, self._asks):
            for price, q in list(levels.items()):
                for i, o in enumerate(q):
                    if o.id == order_id:
                        del q[i]
                        if not q:
                            del levels[price]
                        return True
        return False

    # ---- 队列位置 / 深度（Phase 2 · 成交概率的「队列位置」维度）----

    def queue_position(self, order: Order) -> int:
        """该挂单在其价位的 FIFO 位置（0 = 队首，最优先；不在簿内返回 -1）。

        队列位置决定帧内成交量分摊的优先级：位置越靠后，越晚被成交量喂到，
        在成交量受限时成交概率越低。
        """
        levels = self._bids if order.side is OrderSide.BUY else self._asks
        q = levels.get(order.price)
        if q is None:
            return -1
        for i, o in enumerate(q):
            if o.id == order.id:
                return i
        return -1

    def depth(self, order: Order) -> int:
        """该挂单所在价位的挂单总量（含自身）。"""
        levels = self._bids if order.side is OrderSide.BUY else self._asks
        q = levels.get(order.price)
        return len(q) if q else 0

    # ---- 最优价（BBO 近似）----

    def best_bid(self) -> Optional[float]:
        """最高买价（买一）。"""
        return max(self._bids) if self._bids else None

    def best_ask(self) -> Optional[float]:
        """最低卖价（卖一）。"""
        return min(self._asks) if self._asks else None

    def is_crossed(self, order: Order) -> bool:
        """该挂单是否已越过对手方（市价化）：买单最优价 ≥ 其价，或卖价 ≤ 其价。"""
        if order.side is OrderSide.BUY:
            ask = self.best_ask()
            return ask is not None and ask <= order.price
        bid = self.best_bid()
        return bid is not None and bid >= order.price

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"OrderBook({self.symbol}: bids={sorted(self._bids)}, "
            f"asks={sorted(self._asks)})"
        )
