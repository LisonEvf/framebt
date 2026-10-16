"""framebt.core：帧刷新回测引擎的核心骨架（§2 / §3）。

- events        ：Bar / Tick / Order / FillEvent / 条件单 / 公司行动数据结构
- determinism   ：确定性 RNG（I3）
- market_frame  ：MarketFrame / Quote / 静态帧
- result        ：BacktestResult + 绩效指标
- host          ：StrategyHost（唤醒分发 + RNG）
- strategy      ：FrameStrategy 基类
- engine        ：Engine（时间模型 + 四帧生命周期编排）
"""

from __future__ import annotations

from .determinism import DetermRNG
from .events import (
    Bar,
    CorporateAction,
    CancelEvent,
    FillEvent,
    LifecycleEvent,
    Order,
    OrderId,
    OrderSide,
    OrderState,
    OrderType,
    Tick,
    Trigger,
    Condition,
)
from .host import StrategyHost, WakeupType
from .market_frame import FormingBar, MarketFrame, Quote, StaticFrameBuilder
from .result import BacktestResult, Performance
from .strategy import FrameStrategy
from .data_types import CausalToken, CausalityViolation

__all__ = [
    "Bar",
    "Tick",
    "Order",
    "OrderId",
    "OrderSide",
    "OrderType",
    "OrderState",
    "Trigger",
    "Condition",
    "CancelEvent",
    "FillEvent",
    "CorporateAction",
    "LifecycleEvent",
    "DetermRNG",
    "MarketFrame",
    "Quote",
    "FormingBar",
    "StaticFrameBuilder",
    "BacktestResult",
    "Performance",
    "StrategyHost",
    "WakeupType",
    "FrameStrategy",
    "CausalToken",
    "CausalityViolation",
]
