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
    """单笔子仓位：记录成本、数量、入场帧（用于 T+1 结算）与标签。"""

    entry_price: float
    size: float
    entry_frame: int
    tag: str = ""
    settle_frame: int = 0  # 可卖出帧 = entry_frame + settlement_cycle，由 apply_fill 填写

    def available(self, current_frame: int) -> float:
        """当前可卖出的数量（已结算的腿）。"""
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
        total = sum(leg.size for leg in self.legs)
        if total <= 0:
            return 0.0
        return sum(leg.entry_price * leg.size for leg in self.legs) / total

    def unrealized_pnl(self, mark: float) -> float:
        return sum((mark - leg.entry_price) * leg.size for leg in self.legs)

    def available_to_sell(self, current_frame: int) -> float:
        """T+1：当前可卖出的已结算持仓。"""
        return sum(leg.available(current_frame) for leg in self.legs)

    def apply_fill(self, fill: FillEvent) -> None:
        """按成交更新多腿持仓（FIFO 已实现盈亏）。"""
        if fill.side is OrderSide.BUY:
            self.legs.append(
                PositionLeg(
                    entry_price=fill.price,
                    size=fill.size,
                    entry_frame=fill.frame_id,
                    tag="auto",
                    settle_frame=fill.frame_id + self.settlement_cycle,
                )
            )
        else:  # SELL：FIFO 消耗已结算腿，计算已实现盈亏
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

    def __init__(self, cash: float = 0.0, settlement_cycle: int = 1):
        self.cash = float(cash)
        self.settlement_cycle = settlement_cycle
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
        if fill.side is OrderSide.SELL:
            self.cash += notional - fill.cost
        else:
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

    def margin_requirement(self) -> float:  # pragma: no cover - Phase 6
        """保证金要求（多头为 0；做空/衍生品在 Phase 6 实现）。"""
        return 0.0

    def apply_corporate_action(self, action: CorporateAction) -> None:
        pos = self._positions.get(action.symbol)
        if pos is not None:
            pos.apply_corporate_action(action)
        # 现金分红入账
        if action.kind == "dividend" and action.cash_per_share > 0:
            self.cash += action.cash_per_share * (pos.net_size() if pos else 0.0)

    def apply_lifecycle(self, event: LifecycleEvent) -> None:  # pragma: no cover - Phase 6
        self.cash += event.payoff
