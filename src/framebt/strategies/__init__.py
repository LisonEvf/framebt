"""framebt.strategies：示例策略库。

- dual_ma          ：双均线交叉（Phase 0 验收）
- multi_factor     ：横截面动量选股（§4.5 cross_section）
- conditional_orders：趋势 + 跟踪止损条件单（§5.2）
- bracket_orders   ：入场 + OCO 止盈止损（§5.1 Bracket）
"""

from __future__ import annotations

from .dual_ma import DualMAStrategy
from .multi_factor import MomentumCrossSection
from .conditional_orders import TrendWithStop
from .bracket_orders import BracketStrategy

__all__ = [
    "DualMAStrategy",
    "MomentumCrossSection",
    "TrendWithStop",
    "BracketStrategy",
]
