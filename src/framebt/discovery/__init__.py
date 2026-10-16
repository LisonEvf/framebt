"""Discovery：自主因子发现与入栈（设计文档 §6）。

- ``symbolic``：**符号回归**（遗传规划）—— 在价格面板上演化**可读表达式**，
  适应度为横截面 Spearman IC（带简约惩罚）（§6.2）。
- ``neural``  ：**神经网络隐因子挖掘** —— 可微网络端到端生成隐因子，
  损失含 IC 与「与现有因子池正交」惩罚（§6.3）。需要 PyTorch。
- ``pipeline``：**新因子正交化入栈流水线** —— 正交化 / VIF 共线性 / 增量 ICIR /
  影子运行 / 观察期权重上限（§6.4 / §6.5 / §6.6）。

两条发现路径互补：符号回归产出**人可审查**的表达式，神经网络能表达算子集合之外
的非线性结构但不可读。两者产出的候选因子都**必须**走 ``pipeline`` 入栈，
不允许绕过门槛直接进入因子池。

用法::

    from framebt.discovery import SymbolicFactorDiscovery, FactorStackingPipeline

    engine = SymbolicFactorDiscovery(population_size=300, generations=8, seed=0)
    results = engine.fit(panel, target, top_k=10)

    pipe = FactorStackingPipeline(registry=registry)
    decision = pipe.evaluate("sym_01", candidate_panel, pool, forward_ret)
    print(decision.accepted, decision.reasons)
"""

from __future__ import annotations

from .symbolic import GPResult, SymbolicFactorDiscovery, evaluate_expression, spearman_ic
from .neural import NeuralFactorGenerator
from .pipeline import (
    FactorStackingPipeline,
    ShadowFactorRunner,
    ShadowReport,
    StackingConfig,
    StackingDecision,
    incremental_icir,
    orthogonalize,
    variance_inflation_factors,
)

__all__ = [
    # §6.2 符号回归
    "GPResult",
    "SymbolicFactorDiscovery",
    "evaluate_expression",
    "spearman_ic",
    # §6.3 神经网络
    "NeuralFactorGenerator",
    # §6.4–§6.6 入栈流水线
    "FactorStackingPipeline",
    "StackingConfig",
    "StackingDecision",
    "ShadowFactorRunner",
    "ShadowReport",
    "orthogonalize",
    "variance_inflation_factors",
    "incremental_icir",
]
