"""Execution backend 抽象（Phase 6 · 实盘复用地基）。

「回测 / 实盘同一套策略代码」的关键在边界解耦：

  - 策略只依赖 ``FrameStrategy`` / ``bus`` / ``host``，从不直接触碰数据源或撮合器。
  - 引擎只依赖两个抽象接口：
      * ``MarketFeed``        —— 按 ``(timestamp, sequence_id)`` 提供有序 Bar 的数据源
                                （回测 = ``BarFeed``；实盘 = 把实时流切成帧的 LiveFeed）。
      * ``ExecutionBackend``  —— 订单生命周期（提交 / 撤单 / 撮合 / 在途查询）的后端
                                （回测 = ``MatchingEngine``；实盘 = 封装券商 / CTP 订单
                                流的 LiveConnector）。

  切换回测 / 实盘 = 换这两个接口的实现，策略与 Engine 调度逻辑零改动。

``MatchingEngine`` 是本后端的一个实现（回测）。实盘实现 ``LiveConnector`` 将在接入券商
时编写，它必须遵守同样的语义契约：

  - ``submit(order, frame_id) -> OrderId``        登记订单，返回引擎侧订单 ID。
  - ``cancel(order_id, frame_id) -> CancelResult`` 撤单：与成交赛跑。
  - ``process(frame, frame_id, pre_trade_check=None) -> list[FillEvent]``
        轮询「上次 process 之后」venue 的新成交 / 撤单，按引擎侧语义批量返回。
  - ``has_in_flight / pending_orders / find_order`` 供策略「避免重复下单」
        （实盘查 venue 订单状态，而非本地缓存）。
  - ``register_group`` 组单 / OCO：回测在本地联动，实盘可委托 venue 或本地模拟。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..core.events import FillEvent, Order, OrderId
from ..core.market_frame import MarketFrame


class ExecutionBackend(ABC):
    """订单与撮合后端（Phase 6）。

    回测实现 = ``MatchingEngine``；实盘实现 = 封装券商 / CTP 订单流的 ``LiveConnector``。
    两者遵守同一语义契约，使策略与 Engine 在回测 / 实盘之间零改动切换。

    抽象方法必须由子类实现；``pending_count`` / ``find_order`` 已给出跨后端通用的默认
    实现（仅依赖 ``pending_orders``），实盘 / 回测无需重复编写。
    """

    # ---- 必须由子类实现的抽象方法 ----

    @abstractmethod
    def submit(self, order: Order, frame_id: int) -> OrderId:
        """提交一笔订单，分配引擎侧 ``OrderId``，返回之。

        对应回测的「延迟管线登记」：订单进入在途队列，最早在 ``frame_id + delay`` 才
        可撮合（``MatchingEngine`` 实现）；实盘则向 venue 报单并拿回本地句柄。
        """

    @abstractmethod
    def cancel(self, order_id: OrderId, frame_id: int) -> "CancelResult":
        """撤单：与成交赛跑（§5.4）。返回是否受理。"""

    @abstractmethod
    def process(
        self,
        frame: MarketFrame,
        frame_id: int,
        pre_trade_check: Optional[Callable[[Order, MarketFrame], Optional[str]]] = None,
    ) -> list[FillEvent]:
        """处理自上次 ``process`` 以来的新成交 / 撤单。

        返回本批 ``FillEvent``。``pre_trade_check`` 为风控 + 规则中间件（返回拒绝原因
        或 None），回测用它做盘前拦截，实盘对应 venue 的订单校验（如资金 / 持仓 / 价格
        有效）。
        """

    @abstractmethod
    def has_in_flight(self) -> bool:
        """是否有未落地的订单（在途 + 挂单）。供引擎判断空闲区间是否需补静态帧。"""

    @abstractmethod
    def pending_orders(self) -> list[Order]:
        """返回所有未落地订单（在途 + 挂单）。

        策略用它判断「是否已有未成交的订单」，避免在成交落地前重复下单。
        """

    @abstractmethod
    def register_group(self, group: object, frame_id: int) -> list[OrderId]:
        """登记一个订单组（组单原子性 / OCO）；返回分配的订单 ID 列表。

        回测在本地实现 all_or_none 可行性与 OCO 联动；实盘可委托 venue 或本地模拟。
        """

    # ---- 跨后端通用的默认实现（仅依赖上面两个抽象方法）----

    def pending_count(self) -> int:
        """未落地订单数量（默认实现，无需子类重写）。"""
        return len(self.pending_orders())

    def find_order(self, order_id: OrderId) -> Optional[Order]:
        """按 ID 查找在途 / 挂单订单（默认实现，遍历 ``pending_orders``）。"""
        for order in self.pending_orders():
            if order.id == order_id:
                return order
        return None

    def estimated_fill_fraction(self, order: Order) -> float:
        """本帧内成交比例估计（跨后端默认）。

        回测 ``MatchingEngine`` 覆写为确定性 FIFO 成交量分摊启发式；实盘 ``LiveConnector``
        在接入真实 venue 成交模型后实现。默认 ``0`` = 「未知 / 不可估计」。
        """
        return 0.0
