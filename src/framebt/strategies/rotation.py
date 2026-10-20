"""RotationStrategy：把「单标的策略」改造成「先选股、再对选中标轮动」的多标的策略（§17.3）。

组合三个部件：
  1. ``selector``        ：每帧从选股池决定「交易哪些标的」（见 :mod:`framebt.selection`）
  2. ``make``            ：「单标的策略工厂」 ``Callable[[str], FrameStrategy]``——给定标的
                           返回一个**带独立状态**的单标的策略实例（传类或 lambda 均可）
  3. ``budget_policy``   ：资金策略，默认 :class:`KellyBudgetPolicy`（凯利 + 即时胜率）

每帧流程：
  1. ``selector.select(...)`` 选出活跃标的；
  2. 对每个活跃标的，懒创建/复用其单标的实例并委托 ``on_frame``，汇总订单；
  3. **在途去重**：跳过已有在途同向订单的标的——否则像 ``DualMAStrategy`` 这类不查在途
     的策略会在成交落地前每帧重复全仓下单，导致仓位指数级爆炸（回归保护）；
  4. ``budget_policy`` 把各买入腿预算归一化到凯利总额，并在活跃标的间等权分配。

成交转发
--------
``on_fill`` 同时做两件事：喂给 ``budget_policy`` 维护即时胜率，并转发给对应标的的单标的
实例（``TrendWithStop`` / ``BracketStrategy`` 在成交后挂条件单，依赖此转发）。每个标的的
实例持有独立状态（均线历史 / ``_armed`` / ``_entry_price``），故转发到**同一**实例保证
状态不串。

唤醒类型
--------
``default_wakeups`` 取「所有单标的实例声明的并集」（用模板实例探测），并恒含 ``FRAME``。
因此当组合了依赖 ``FILL`` 唤醒的策略时，引擎会正确分发 ``on_fill``。
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from ..core import FrameStrategy, Order, OrderSide, OrderType
from ..core.host import WakeupType
from ..core.events import FillEvent
from ..selection import SymbolSelector, KellyBudgetPolicy

# 构造模板实例用的占位标的（仅用于探测 default_wakeups，实例不用）
_TEMPLATE_SYMBOL = "\x00rotation-template"


class RotationStrategy(FrameStrategy):
    """选股 + 轮动包装器。

    参数：
      universe          ：选股池（若 ``selector.universe`` 非空则以其为准）
      selector          ：选股器（决定每帧交易哪些标的）
      make              ：「单标的策略工厂」 ``Callable[[str], FrameStrategy]``，可传
                          策略类（如 ``DualMAStrategy``）或 lambda（如
                          ``lambda s: DualMAStrategy(s, short=5, long=20)``）
      budget_policy     ：资金策略，缺省 :class:`KellyBudgetPolicy`（半凯利）
      dedup_in_flight   ：是否在途去重（默认 True；回归保护，建议保持开启）
    """

    def __init__(
        self,
        universe: list[str],
        selector: SymbolSelector,
        make: Callable[[str], FrameStrategy] | type[FrameStrategy],
        *,
        budget_policy: Optional[ KellyBudgetPolicy] = None,
        dedup_in_flight: bool = True,
        rotate: bool = False,
    ):
        self.universe = list(selector.universe) if selector.universe else list(universe)
        if not self.universe:
            raise ValueError("选股池不能为空")
        self.selector = selector
        self._make = make
        self.budget_policy = budget_policy if budget_policy is not None else KellyBudgetPolicy()
        self.dedup_in_flight = dedup_in_flight
        self.rotate = rotate

        # 探测各单标的实例声明的唤醒类型并集（用模板实例，构造开销极小）
        _template = self._make(_TEMPLATE_SYMBOL)
        base = list(getattr(_template, "default_wakeups", [WakeupType.FRAME]))
        if WakeupType.FRAME not in base:
            base.insert(0, WakeupType.FRAME)
        self.default_wakeups = base

        self._instances: dict[str, FrameStrategy] = {}
        self._traded: list[str] = []  # 历史交易过的标的（便于复盘）

    # ---- 工厂 / 缓存 ----

    def _instance(self, symbol: str) -> FrameStrategy:
        inst = self._instances.get(symbol)
        if inst is None:
            inst = self._make(symbol)
            self._instances[symbol] = inst
        return inst

    # ---- 主循环 ----

    def on_frame(self, frame, token, bus, host) -> list[Order]:
        active = self.selector.select(frame, token, bus, host, self.universe)
        if not active:
            # 没有活跃标的：开启轮动则清仓，否则空仓
            if self.rotate:
                return self._clear_positions(host)

        orders: list[Order] = []
        # 轮动离场：持有但已跌出活跃池的标的 → 市价卖出（不受凯利约束，走在途去重/T+1）
        if self.rotate:
            orders.extend(self._rotation_exits(active, host))
        for sym in active:
            if not frame.has_bar(sym):
                continue  # 本帧无数据：不触发（也不计入已交易）
            inst = self._instance(sym)
            sym_orders = inst.on_frame(frame, token, bus, host) or []
            orders.extend(sym_orders)
            if sym not in self._traded:
                self._traded.append(sym)

        if self.dedup_in_flight:
            orders = self._dedup_in_flight(orders, host)
        if orders:
            orders = self.budget_policy.budget(orders, active, host, frame)
        return orders

    def _dedup_in_flight(self, orders: list[Order], host) -> list[Order]:
        """跳过「已有在途同向订单」的标的，避免重复下单。"""
        pending_buys: set[str] = set()
        pending_sells: set[str] = set()
        for o in host.pending_orders():
            if o.side is OrderSide.BUY:
                pending_buys.add(o.symbol)
            elif o.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
                pending_sells.add(o.symbol)
        kept: list[Order] = []
        for o in orders:
            if o.side is OrderSide.BUY and o.symbol in pending_buys:
                continue
            if o.side in (OrderSide.SELL, OrderSide.SELL_SHORT) and o.symbol in pending_sells:
                continue
            kept.append(o)
        return kept

    def _rotation_exits(self, active, host) -> list[Order]:
        """持有但已跌出活跃池的标的 → 市价卖出（轮动离场）。在途去重由外层统一处理。"""
        active_set = set(active)
        sells: list[Order] = []
        for sym, size in host.portfolio.exposure().items():
            if size > 0 and sym not in active_set:
                sells.append(
                    Order(side=OrderSide.SELL, symbol=sym, size=size,
                          order_type=OrderType.MARKET)
                )
        return sells

    def _clear_positions(self, host) -> list[Order]:
        """清仓全部多头（活跃池为空且开启轮动时）。"""
        sells = [
            Order(side=OrderSide.SELL, symbol=sym, size=size, order_type=OrderType.MARKET)
            for sym, size in host.portfolio.exposure().items() if size > 0
        ]
        return sells

    # ---- 成交转发 ----

    def on_fill(self, fill: FillEvent) -> list[Order]:
        self.budget_policy.record(fill)
        inst = self._instances.get(fill.symbol)
        if inst is None:
            return []
        cb = getattr(inst, "on_fill", None)
        if cb is None:
            return []
        return cb(fill) or []

    # ---- 复盘辅助 ----

    def held_symbols(self) -> list[str]:
        """回测期间交易过的全部标的（按首次触发排序）。"""
        return list(self._traded)
