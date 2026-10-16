"""framebt.portfolio：权重优化与组合构建（设计文档 §3）。

- ``weights``：复合损失（−ICIR + λ₁·Turnover + λ₂·‖w‖₁）+ 在线梯度下降（折扣率 β）
- ``gating`` ：市场状态门控（波动率分位 / 涨跌家数比 / 离散度 → 因子门控系数）
- ``combine``：因子合成打分 → 目标持仓权重 → 目标股数
"""

from __future__ import annotations

from .weights import OnlineWeightOptimizer, WeightState
from .gating import MarketStateGating
from .combine import composite_score, select_portfolio, turnover, weights_to_shares

__all__ = [
    "OnlineWeightOptimizer",
    "WeightState",
    "MarketStateGating",
    "composite_score",
    "select_portfolio",
    "turnover",
    "weights_to_shares",
]
