"""Ledger：账户、持仓与生命周期（§6）。

- 现金 + 多腿持仓（金字塔加仓 / 分批止盈跟踪每笔成本与结算日）
- T+1 结算：买入腿在 ``entry_frame + settlement_cycle`` 之前不可卖出
- 未实现/已实现盈亏、MTM 权益、暴露度
- 公司行动处理（§6.3）：拆股 / 分红 / 退市现金结算

多腿持仓（PositionLeg）：
    PositionLeg(entry_price, size, entry_frame, tag)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .core.events import CorporateAction, FillEvent, OrderSide, LifecycleEvent


@dataclass
class PositionLeg:
    """单笔子仓位：记录成本、数量、入场帧（用于 T+1 结算）与标签。

    ``size`` 带符号：多头为正、空头为负；``short=True`` 显式标记空头腿（融券）。
    """

    entry_price: float
    size: float
    entry_frame: int
    tag: str = ""
    settle_frame: int = 0  # 可卖出帧 = entry_frame + settlement_cycle，由 apply_fill 填写
    short: bool = False  # 空头腿（融券借入）

    def available(self, current_frame: int) -> float:
        """当前可卖出的数量（已结算的多头腿；空头腿不可「卖出」，返回 0）。"""
        if self.short or self.size <= 0:
            return 0.0
        return self.size if self.settle_frame <= current_frame else 0.0


class Position:
    """多腿持仓。"""

    def __init__(self, symbol: str, settlement_cycle: int = 1):
        self.symbol = symbol
        self.legs: list[PositionLeg] = []
        self.settlement_cycle = settlement_cycle
        self.realized_pnl: float = 0.0
        self.dividend_pnl: float = 0.0

    def net_size(self) -> float:
        return sum(leg.size for leg in self.legs)

    def avg_price(self) -> float:
        net = self.net_size()
        if net == 0:
            return 0.0
        # 多头返回正均价；空头（净持仓为负）返回正的均价（融券入场价）
        return sum(leg.entry_price * leg.size for leg in self.legs) / net

    def unrealized_pnl(self, mark: float) -> float:
        # 带符号公式对多空通用：多头 (mark-entry)*size，空头 (entry-mark)*|size|
        return sum((mark - leg.entry_price) * leg.size for leg in self.legs)

    def available_to_sell(self, current_frame: int) -> float:
        """T+1：当前可卖出的已结算多头（空头腿无「卖出」，返回 0）。"""
        return sum(leg.available(current_frame) for leg in self.legs)

    def apply_fill(self, fill: FillEvent) -> None:
        """按成交更新多腿持仓（FIFO 已实现盈亏，多空分别结算）。

        - BUY：先 FIFO 平掉现有空头（实现空头盈亏），剩余建多。
        - SELL_SHORT：开空，新增空头腿（``short=True``、``size`` 为负）。
        - SELL：FIFO 消耗已结算多头腿，实现多头盈亏。
        """
        if fill.side is OrderSide.SELL_SHORT:
            self.legs.append(
                PositionLeg(
                    entry_price=fill.price,
                    size=-fill.size,
                    entry_frame=fill.frame_id,
                    tag="short",
                    settle_frame=fill.frame_id + self.settlement_cycle,
                    short=True,
                )
            )
        elif fill.side is OrderSide.BUY:
            short_net = -sum(leg.size for leg in self.legs if leg.size < -1e-12)
            if short_net > 0:
                # 先平空（FIFO），已实现盈亏 = (入场价 - 平仓价) × 数量
                remaining = fill.size
                shorts = sorted((leg for leg in self.legs if leg.size < -1e-12),
                                key=lambda l: l.entry_frame)
                for leg in shorts:
                    if remaining <= 0:
                        break
                    take = min(-leg.size, remaining)
                    self.realized_pnl += (leg.entry_price - fill.price) * take
                    leg.size += take  # 负腿向 0 靠拢
                    remaining -= take
                self.legs = [leg for leg in self.legs if leg.size > 1e-12 or leg.size < -1e-12]
                remaining_after = fill.size - short_net
                if remaining_after > 1e-12:
                    self.legs.append(
                        PositionLeg(
                            entry_price=fill.price,
                            size=remaining_after,
                            entry_frame=fill.frame_id,
                            tag="auto",
                            settle_frame=fill.frame_id + self.settlement_cycle,
                        )
                    )
            else:
                self.legs.append(
                    PositionLeg(
                        entry_price=fill.price,
                        size=fill.size,
                        entry_frame=fill.frame_id,
                        tag="auto",
                        settle_frame=fill.frame_id + self.settlement_cycle,
                    )
                )
        else:  # SELL：FIFO 消耗已结算多头腿，计算已实现盈亏
            remaining = fill.size
            live = [leg for leg in self.legs if leg.size > 1e-12]
            for leg in sorted(live, key=lambda l: l.entry_frame):
                if remaining <= 0:
                    break
                take = min(leg.size, remaining)
                self.realized_pnl += (fill.price - leg.entry_price) * take
                leg.size -= take
                remaining -= take
            self.legs = [leg for leg in self.legs if leg.size > 1e-12]

    def apply_corporate_action(self, action: CorporateAction) -> None:
        """公司行动：拆股（调整腿数量与成本）、现金分红（记录分红收益）。"""
        if action.kind == "split" and action.ratio > 0:
            for leg in self.legs:
                leg.size *= action.ratio
                leg.entry_price /= action.ratio
        elif action.kind == "dividend" and action.cash_per_share > 0:
            self.dividend_pnl += self.net_size() * action.cash_per_share
        elif action.kind == "delist":
            # 退市：按公告价现金结算
            for leg in self.legs:
                self.realized_pnl += (action.ratio - leg.entry_price) * leg.size
            self.legs.clear()


class Ledger:
    """账本：现金 + 各标的多腿持仓。"""

    def __init__(self, cash: float = 0.0, settlement_cycle: int = 1, margin_ratio: float = 0.5):
        self.cash = float(cash)
        self.settlement_cycle = settlement_cycle
        self.margin_ratio = margin_ratio  # 空头保证金比例（空头名义市值 × 该比例）
        self._positions: Dict[str, Position] = {}

    def _pos(self, symbol: str) -> Position:
        if symbol not in self._positions:
            self._positions[symbol] = Position(symbol, self.settlement_cycle)
        return self._positions[symbol]

    def apply_fill(self, fill: FillEvent) -> None:
        """成交入账：更新现金与多腿持仓。"""
        pos = self._pos(fill.symbol)
        pos.apply_fill(fill)

        notional = fill.size * fill.price
        if fill.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            # 卖出 / 开空：收到现金（卖空所得），扣除成本
            self.cash += notional - fill.cost
        else:
            # 买入 / 平空：付出现金
            self.cash -= notional + fill.cost

    def position(self, symbol: str) -> Position:
        return self._positions.get(symbol, Position(symbol, self.settlement_cycle))

    def equity(self) -> float:
        """未标记时的现金。"""
        return self.cash

    def mark_to_market(self, mark_price: Dict[str, float]) -> float:
        """按给定报价计算 MTM 权益：权益 = 现金 + Σ(持仓 × 市价)。

        现金已扣除买入成本 / 已计入卖出收入，故持仓部分只按市价计值；
        未实现盈亏 = 市价计值 - 成本，已隐含其中，不可重复累加。
        """
        eq = self.cash
        for pos in self._positions.values():
            if pos.net_size() != 0:
                if pos.symbol not in mark_price:
                    raise KeyError(f"标记缺失：{pos.symbol}")
                eq += pos.net_size() * mark_price[pos.symbol]
        return eq

    def exposure(self) -> Dict[str, float]:
        """各标的净持仓暴露。"""
        return {s: p.net_size() for s, p in self._positions.items() if p.net_size() != 0}

    def margin_requirement(self, mark_price: "Dict[str, float] | None" = None) -> float:
        """保证金要求（§3）：多头为 0；空头 = Σ 空头名义市值 × ``margin_ratio``。

        空头名义市值 = Σ |空头腿| × 价格（优先用 ``mark_price``，缺省用入场价）。
        空头保证金是「被冻结」的购买力，由风控在盘前检查中约束不得超权益比例；
        它不是成本，不计入 ``equity()``（权益为净清算价值，见 ``mark_to_market``）。
        """
        total = 0.0
        for pos in self._positions.values():
            short_notional = 0.0
            for leg in pos.legs:
                if leg.size < -1e-12:
                    price = None
                    if mark_price:
                        price = mark_price.get(pos.symbol)
                    if price is None:
                        price = leg.entry_price
                    short_notional += (-leg.size) * price
            total += short_notional * self.margin_ratio
        return total

    def apply_corporate_action(self, action: CorporateAction) -> None:
        pos = self._positions.get(action.symbol)
        # 先记下事件前的净持仓：delist 会清空腿，之后 net_size() 变成 0。
        qty_before = pos.net_size() if pos is not None else 0.0
        if pos is not None:
            pos.apply_corporate_action(action)
        # 现金分红入账
        if action.kind == "dividend" and action.cash_per_share > 0:
            self.cash += action.cash_per_share * (pos.net_size() if pos else 0.0)
        # 退市现金结算：按 action.ratio（每股现金对价）把持仓换成现金。
        # 空头净持仓（qty_before < 0）则为现金流出，符号自洽。
        elif action.kind == "delist" and qty_before != 0.0:
            self.cash += action.ratio * qty_before

    def apply_lifecycle(self, event: LifecycleEvent) -> None:  # pragma: no cover - Phase 6
        self.cash += event.payoff
