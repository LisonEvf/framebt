"""framebt — 帧刷新量化回测框架（Frame-refresh causal backtesting engine）。

以「固定逻辑步长 Δt 主动推进虚拟时间」为骨架的确定性、因果性回测引擎。

三层解耦（详见设计文档 §1 / §16）：
  1. 时间推进与事件到达解耦 —— 固定 Δt 主动推进，事件按真实时间戳注入
  2. 数据可见性与因果性解耦 —— 策略自由请求，引擎在访问层过滤未来
  3. 感知节奏与帧节拍解耦 —— 多唤醒机制让策略自主选择何时被唤醒

核心不变式（§2.3）：
  I1  任何数据请求返回的记录 timestamp <= current_virtual_time
  I2  帧 t 生成的订单，最早可成交时间为帧 t+delay（延迟管线）
  I3  相同配置逐位可复现
  I4  帧内多 Bar 按 (timestamp, sequence_id) 严格排序
  I5  策略对已过去帧的状态修改请求被拒绝

快速开始：
    from framebt import Engine, run_backtest
    from framebt.data_sources import OpentdxParquetSource
    from framebt.strategies import DualMAStrategy

    feed = OpentdxParquetSource().load("SH.600519")
    result = Engine(feed, DualMAStrategy(short=5, long=20)).run()
    print(result.equity_curve)
"""

from __future__ import annotations

__version__ = "1.0.0"

# ---- 核心骨架（§2 / §3）----
from .core import (
    Bar,
    Tick,
    Order,
    OrderId,
    OrderSide,
    OrderType,
    OrderState,
    Trigger,
    Condition,
    CancelEvent,
    FillEvent,
    CorporateAction,
    LifecycleEvent,
    DetermRNG,
    MarketFrame,
    Quote,
    FormingBar,
    StaticFrameBuilder,
    BacktestResult,
    Performance,
    StrategyHost,
    WakeupType,
    FrameStrategy,
    CausalToken,
    CausalityViolation,
)
from .engine import Engine, run_backtest

# ---- 因果数据总线（§4）----
from .data_bus import DataBus, FactorRow

# ---- 数据源接入（§12）----
from .feed import BarFeed
from .data_sources.base import DataSource
from .data_sources.opentdx import OpentdxParquetSource, default_data_dir

# ---- 订单与撮合（§5）----
from .execution import MatchingEngine, FillResult, CancelResult, OrderBook, OrderGroup, Atomicity, LinkMode

# ---- 账本（§6）----
from .ledger import Ledger, Position, PositionLeg

# ---- 市场规则（§7）----
from .rules import MarketRules, NoOpRules, StandardAshareRules

# ---- 成本模型（§8）----
from .cost import CostModel, NoOpCost, StandardCost

# ---- 风控中间件（§9）----
from .guard import RiskMiddleware, NoOpRisk, StandardRisk, IncrementalRisk

__all__ = [
    # core
    "Bar", "Tick", "Order", "OrderId", "OrderSide", "OrderType", "OrderState",
    "Trigger", "Condition", "CancelEvent", "FillEvent", "CorporateAction",
    "LifecycleEvent", "DetermRNG", "MarketFrame", "Quote", "FormingBar",
    "StaticFrameBuilder", "BacktestResult", "Performance", "StrategyHost",
    "WakeupType", "FrameStrategy", "CausalToken", "CausalityViolation", "FactorRow",
    "Engine", "run_backtest",
    # data
    "DataBus", "BarFeed", "DataSource", "OpentdxParquetSource", "default_data_dir",
    # execution
    "MatchingEngine", "FillResult", "CancelResult", "OrderBook", "OrderGroup",
    "Atomicity", "LinkMode",
    # ledger
    "Ledger", "Position", "PositionLeg",
    # rules
    "MarketRules", "NoOpRules", "StandardAshareRules",
    # cost
    "CostModel", "NoOpCost", "StandardCost",
    # guard
    "RiskMiddleware", "NoOpRisk", "StandardRisk", "IncrementalRisk",
]
