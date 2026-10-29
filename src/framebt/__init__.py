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
    result = Engine(feed, DualMAStrategy("SH.600519", short=5, long=20)).run()
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

# ---- 回测报告（图表化展示，依赖 matplotlib）----
# 惰性导入：`import framebt` 不应因为未安装 matplotlib 而失败。报告 API 仍可
# `from framebt import generate_report`（PEP 562 模块级 __getattr__），只有真正
# 调用时才 import report/matplotlib。核心引擎因此保持「零第三方依赖」。
_REPORT_EXPORTS = frozenset({
    "generate_report", "save_report", "render_equity_chart",
    "render_trade_chart", "ReportConfig",
})


def __getattr__(name):
    if name in _REPORT_EXPORTS:
        from . import report as _report

        return getattr(_report, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# ---- 因果数据总线（§4）----
from .data_bus import DataBus, FactorRow

# ---- 数据源接入（§12）----
from .feed import BarFeed, MarketFeed
from .data_sources.base import DataSource
from .data_sources.opentdx import OpentdxParquetSource, default_data_dir

# ---- 订单与撮合（§5 / Phase 6）----
from .execution import (
    MatchingEngine,
    ExecutionBackend,
    FillResult,
    CancelResult,
    OrderBook,
    OrderGroup,
    Atomicity,
    LinkMode,
)

# ---- 实盘复用地基（Phase 6 · ①）----
from .live import (
    LiveFeed,
    LiveConnector,
    Transport,
    SimulatedTransport,
)

# ---- 账本（§6）----
from .ledger import Ledger, Position, PositionLeg

# ---- 市场规则（§7）----
from .rules import MarketRules, NoOpRules, StandardAshareRules

# ---- 成本模型（§8）----
from .cost import CostModel, NoOpCost, StandardCost

# ---- 风控中间件（§9）----
from .guard import RiskMiddleware, NoOpRisk, StandardRisk, IncrementalRisk

# ---- 绩效指标（统一口径，见 framebt.metrics）----
from . import metrics
from .metrics import (
    DEFAULT_PERIODS_PER_YEAR,
    MetricConfig,
    annualized_volatility,
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
)

# ---- 目标权重调仓原语 ----
from .rebalance import RebalanceConfig, rebalance_orders, target_shares, weights_from_positions

# ---- 选股 + 资金抽象（§17）----
from .selection import (
    SymbolSelector,
    SymbolBudgetPolicy,
    FixedUniverseSelector,
    CrossSectionRankSelector,
    TrendPresenceSelector,
    LiquiditySelector,
    CompositeSelector,
    KellyBudgetPolicy,
)
from .strategies import RotationStrategy

__all__ = [
    # core
    "Bar", "Tick", "Order", "OrderId", "OrderSide", "OrderType", "OrderState",
    "Trigger", "Condition", "CancelEvent", "FillEvent", "CorporateAction",
    "LifecycleEvent", "DetermRNG", "MarketFrame", "Quote", "FormingBar",
    "StaticFrameBuilder", "BacktestResult", "Performance", "StrategyHost",
    "WakeupType", "FrameStrategy", "CausalToken", "CausalityViolation", "FactorRow",
    "Engine", "run_backtest",
    # report
    "generate_report", "save_report", "render_equity_chart",
    "render_trade_chart", "ReportConfig",
    # data
    "DataBus", "BarFeed", "MarketFeed", "DataSource", "OpentdxParquetSource", "default_data_dir",
    # execution
    "ExecutionBackend", "MatchingEngine", "FillResult", "CancelResult", "OrderBook",
    "OrderGroup", "Atomicity", "LinkMode",
    # live (Phase 6 · ① 实盘复用地基)
    "LiveFeed", "LiveConnector", "Transport", "SimulatedTransport",
    # ledger
    "Ledger", "Position", "PositionLeg",
    # rules
    "MarketRules", "NoOpRules", "StandardAshareRules",
    # cost
    "CostModel", "NoOpCost", "StandardCost",
    # guard
    "RiskMiddleware", "NoOpRisk", "StandardRisk", "IncrementalRisk",
    # metrics（统一口径）
    "metrics", "MetricConfig", "DEFAULT_PERIODS_PER_YEAR",
    "total_return", "cagr", "annualized_volatility", "sharpe_ratio",
    "sortino_ratio", "max_drawdown", "calmar_ratio", "summarize",
    # rebalance（目标权重调仓原语）
    "RebalanceConfig", "rebalance_orders", "target_shares", "weights_from_positions",
    # selection (§17) + rotation
    "SymbolSelector", "SymbolBudgetPolicy",
    "FixedUniverseSelector", "CrossSectionRankSelector", "TrendPresenceSelector",
    "LiquiditySelector", "CompositeSelector", "KellyBudgetPolicy",
    "RotationStrategy",
]
