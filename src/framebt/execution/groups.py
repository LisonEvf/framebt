"""订单组、原子性与联动（§5.3 / §5.1 OCO）。

- ``OrderGroup``：把多腿订单绑为一个单元，原子性等级
  ``all_or_none`` / ``best_effort`` / ``sequential``（§5.3）。
- ``LinkMode``：腿间联动。``oco`` = 一单成交其余自动撤销（§5.1 OCO）。

**Bracket（入场 + 止盈 + 止损）** 由既有原语组合而成：
策略在 ``on_fill`` 中返回一个 ``OrderGroup.oco([止盈, 止损])``，
引擎在成交回报后登记该 OCO 组；任一腿成交即自动撤销另一腿。
见 ``framebt.strategies.bracket_orders``。

用法：
    grp = OrderGroup([
        Order(BUY, "A", size=100),
        Order(SELL, "B", size=100),
    ], atomicity="all_or_none")

    oco = OrderGroup.oco([take_profit_order, stop_loss_order])
"""

from __future__ import annotations

from enum import Enum

from ..core.events import Order, OrderId


class Atomicity(str, Enum):
    ALL_OR_NONE = "all_or_none"     # 全部成交或全部不成交（需外部撮合保证原子）
    BEST_EFFORT = "best_effort"     # 尽量多腿成交，部分成交也可接受
    SEQUENTIAL = "sequential"       # 按顺序，前腿未成交则后续取消


class LinkMode(str, Enum):
    NONE = "none"
    OCO = "oco"                     # one-cancels-other：任一腿成交，其余腿自动撤销


class OrderGroup:
    def __init__(
        self,
        orders: list[Order],
        atomicity: str = "all_or_none",
        link: str = "none",
    ):
        self.orders = list(orders)
        self.atomicity = Atomicity(atomicity)
        self.link = LinkMode(link)
        self.order_ids: list[OrderId] = []  # 由撮合引擎在登记时填写

    @classmethod
    def oco(cls, orders: list[Order]) -> "OrderGroup":
        """构造 OCO 组：任一腿成交，其余腿自动撤销（§5.1）。"""
        return cls(orders, atomicity="best_effort", link="oco")

    def is_feasible(self) -> bool:
        """所有腿都指定了 size 或 budget 且价格合法。"""
        for o in self.orders:
            if o.size <= 0 and o.budget <= 0:
                return False
            if o.order_type.value in ("limit", "stop", "stop_limit") and o.price <= 0:
                return False
        return True

    def resolve(self) -> list[Order]:
        """返回应下发的订单列表。

        ``all_or_none``：所有订单都满足内部约束才全部返回，否则返回空（一条都不触发）。
        """
        if self.atomicity is Atomicity.ALL_OR_NONE and not self.is_feasible():
            return []
        return list(self.orders)

    # 兼容旧名
    def _all_feasible(self) -> bool:  # pragma: no cover - 向后兼容
        return self.is_feasible()
