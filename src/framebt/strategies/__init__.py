"""framebt.strategies：示例策略库。

- dual_ma          ：双均线交叉（Phase 0 验收）
- multi_factor     ：横截面动量选股（§4.5 cross_section）
- conditional_orders：趋势 + 跟踪止损条件单（§5.2）
- bracket_orders   ：入场 + OCO 止盈止损（§5.1 Bracket）
- rotation         ：选股 + 轮动包装器（§17）——把任意单标的策略改造成多标的轮动
- multifactor_sweep：主动交易（多因子基础 + sweep 做 T + 主动选股）
"""

from __future__ import annotations

from .dual_ma import DualMAStrategy
from .multi_factor import MomentumCrossSection
from .conditional_orders import TrendWithStop
from .bracket_orders import BracketStrategy
from .rotation import RotationStrategy
from .multifactor_sweep import MultiFactorSweepStrategy

__all__ = [
    "DualMAStrategy",
    "MomentumCrossSection",
    "TrendWithStop",
    "BracketStrategy",
    "RotationStrategy",
    "MultiFactorSweepStrategy",
]
