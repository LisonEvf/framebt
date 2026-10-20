"""framebt.selection：可复用的「选股 + 资金」抽象（§17）。

- ``SymbolSelector``        ：每帧从选股池决定「交易哪些标的」的纯感知组件
- ``SymbolBudgetPolicy``    ：决定「投入多少权益」的资金策略
- 具体选择器：
    ``FixedUniverseSelector``   ：整个选股池
    ``CrossSectionRankSelector``：按注册因子横截面排名（TopK / Top分数）
    ``TrendPresenceSelector``   ：只在趋势（价 > 均线）向上的标的里
    ``LiquiditySelector``       ：只在流动性（成交额）达标的标的里
    ``CompositeSelector``       ：and/or 组合多个选择器
- ``KellyBudgetPolicy``       ：凯利公式 + 即时胜率

与 :class:`framebt.strategies.RotationStrategy` 组合，即可把任意「单标的策略」
改造成「先选股、再对选中标轮动交易」的多标的策略。
"""

from __future__ import annotations

from .base import SymbolBudgetPolicy, SymbolSelector
from .selectors import (
    CompositeSelector,
    CrossSectionRankSelector,
    FixedUniverseSelector,
    LiquiditySelector,
    TrendPresenceSelector,
)
from .kelly import KellyBudgetPolicy

__all__ = [
    "SymbolSelector",
    "SymbolBudgetPolicy",
    "FixedUniverseSelector",
    "CrossSectionRankSelector",
    "TrendPresenceSelector",
    "LiquiditySelector",
    "CompositeSelector",
    "KellyBudgetPolicy",
]
