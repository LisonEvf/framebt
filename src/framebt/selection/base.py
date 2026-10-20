"""选股抽象层（§17）：把「交易哪些标的」从具体策略里解耦成可复用的选择器。

动机
----
框架既有的示例策略大多是「钉死一个标的」的：构造时传入单个 ``symbol``，
``on_frame`` 全程只服务这一只。想要轮动多个标的，就得把策略逻辑整份复制一遍——
这就是「在一个标的上死磕」。

本模块把「选股」提升为框架的一等公民能力，分两部分：

- ``SymbolSelector``：每帧从选股池里决定「这一帧交易哪些标的」。它是一个纯粹的
  **感知**组件——只读 ``bus`` / ``frame``，不产生任何订单。具体选择器见
  :mod:`framebt.selection.selectors`。
- ``SymbolBudgetPolicy``：决定「投入多少权益」——默认实现用凯利公式 + 即时胜率
  （见 :mod:`framebt.selection.kelly`）。

二者由 :class:`framebt.strategies.RotationStrategy` 组合：先 ``selector`` 选标，
再对每个活跃标的委托一个「单标的策略工厂」交易，最后用 ``budget_policy`` 归一化资金。

因果性
------
选择器与交易策略跑在同一因果时钟下，所有 ``bus`` 请求都附带引擎签发的
``CausalToken`` 并带 ``lag``，杜绝同帧泄漏（§4.3）。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from ..core.events import Order


class SymbolSelector(ABC):
    """选股器：从选股池决定本帧交易的标的子集。

    子类覆写 :meth:`select`。选择器是纯感知组件——**不产生订单**，只返回要交易的
    标的列表（顺序即为权重分配的先后，通常按因子降序）。
    """

    #: 选股池（可选）：``None`` 表示以 ``RotationStrategy`` 传入的 universe 为准。
    universe: Optional[list[str]] = None

    @abstractmethod
    def select(
        self,
        frame,
        token,
        bus,
        host,
        universe: list[str],
    ) -> list[str]:
        """返回本帧要交易的标的（因子降序）。

        参数：
          frame     ：当前市场帧（可取 BBO / 极差等）
          token     ：引擎签发的因果令牌（所有 bus 请求必须附带）
          bus       ：因果数据总线
          host      ：策略宿主（可查持仓 / 在途订单）
          universe  ：当前可用的完整选股池

        应返回 ``universe`` 的子集；返回空表表示本帧不交易。
        """
        raise NotImplementedError


class SymbolBudgetPolicy(ABC):
    """资金策略：决定对「本帧各活跃标的」各自投入多少预算。

    ``budget(orders, active, host, frame)`` 应**原地修改**传入订单的 ``budget``
    （仅买入腿），并丢弃预算被压到 ``<=0`` 的买入单。卖出腿（按数量）不动。

    默认实现 :class:`~framebt.selection.kelly.KellyBudgetPolicy` 用凯利公式决定总
    投入比例，再在活跃标的间等权分配。
    """

    @abstractmethod
    def budget(
        self,
        orders: list[Order],
        active: list[str],
        host,
        frame,
    ) -> list[Order]:
        """原地归一化买入预算，返回（可能更短的）订单列表。"""
        raise NotImplementedError
