"""原始事件与订单/成交数据结构（core）。

时间戳约定：统一使用整数（如交易日序号 / 毫秒），引擎不解释其物理含义，
只比较大小、做差、按 Δt 切帧。这保证 I3（确定性）不依赖任何时钟语义。

``sequence_id`` 用于 I4：同一时刻多事件按 ``(timestamp, sequence_id)`` 严格全序，
在数据加载阶段预分配，回测期间不再变化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"  # A 股融券做空（区别于 SELL 平多）


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"
    IOC = "ioc"
    FOK = "fok"


class OrderState(str, Enum):
    PENDING = "pending"        # 在途（延迟管线中）
    RESTING = "resting"        # 限价/条件单挂单中
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderId(int):
    """新式订单 ID（底层是 int，便于哈希与比较）。"""


# 条件单触发方式（§5.2）
class Trigger(str, Enum):
    """条件单的触发比较方式。"""

    BID_LE = "bid_lte"     # 买价 <= 触发价
    BID_GE = "bid_gte"
    ASK_LE = "ask_lte"
    ASK_GE = "ask_gte"
    LAST_LE = "last_lte"
    LAST_GE = "last_gte"
    MARK_LE = "mark_lte"
    MARK_GE = "mark_gte"


@dataclass(frozen=True)
class Bar:
    """单根 K 线（原始事件的最小聚合单元）。

    ``float_shares`` / ``turnover`` 为可选的基本面/流动性伴随字段；
    ``vwap`` / ``market_cap`` 为**口径已对齐**的派生量（见下）。

    复权口径一致性（重要）
    ----------------------
    行情接口的 ``open/high/low/close`` 是**复权后**价格，而 ``amount``（成交额）
    与 ``vol``（成交量）是**未复权**原始值。因此 ``amount/vol`` 得到的是**原始**
    vwap，与复权后的 close 不同口径 —— 直接混用会污染所有 vwap/市值类因子。

    数据源在同时具备复权(adjust=1)与不复权(adjust=0)数据时，会解出复权因子
    ``f(t) = raw_close(t) / adj_close(t)``，据此提供：
      - ``vwap``       = (amount/vol) / f(t)   —— 与复权价同口径
      - ``market_cap`` = raw_close(t) × float_shares —— 真实市值（不受复权影响）
    两者缺省为 0，表示数据源无法提供（此时引擎退回近似值）。
    """

    timestamp: int
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    amount: float = 0.0
    float_shares: float = 0.0
    turnover: float = 0.0
    vwap: float = 0.0
    market_cap: float = 0.0
    sequence_id: int = 0


@dataclass(frozen=True)
class Tick:
    """逐笔成交（高频模式 / 帧内事件队列使用，Phase 6）。"""

    timestamp: int
    symbol: str
    price: float
    volume: float
    side: OrderSide | None = None
    sequence_id: int = 0


@dataclass
class Order:
    """策略生成的订单。``id`` 由撮合引擎在登记时分配。

    ``size`` 与 ``budget`` 二选一：
      - 指定 size：固定成交数量（限价/市价）。
      - 指定 budget>0：「全额下单」，撮合时按 fill_price 反算数量。

    全额下单（``budget``）与数量（``size``）二选一，``size`` 优先。
    """

    side: OrderSide
    symbol: str
    size: float = 0.0
    budget: float = 0.0
    order_type: OrderType = OrderType.MARKET
    price: float = 0.0  # 限价单触发价/限价；条件单为触发价
    trail_pct: float = 0.0  # 跟踪止损回撤比例（仅 TRAILING_STOP）
    id: OrderId = field(default=None)
    placement_frame: int | None = None  # 由引擎填写
    state: OrderState = OrderState.PENDING
    condition: "Condition | None" = None  # 条件单关联的触发条件
    condition_armed: int | None = None  # 条件已满足的帧；下一帧开盘才允许成交（防帧内前视）
    short: bool = False  # 融券做空标记（= side is SELL_SHORT，由 __post_init__ 统一定义）

    def __post_init__(self):
        self.short = self.side is OrderSide.SELL_SHORT
        if self.order_type in (OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT) and self.price <= 0:
            raise ValueError(f"订单 {self.order_type.value} 必须提供正价格 price")
        if self.order_type is OrderType.TRAILING_STOP and self.trail_pct <= 0:
            raise ValueError("跟踪止损单必须提供正比例 trail_pct")
        if self.size <= 0 and self.budget <= 0:
            raise ValueError("订单必须指定 size 或 budget（>0）")


@dataclass
class FillEvent:
    """成交回报。``cost`` 由 Cost 模型在成交时计入（Phase 4）。"""

    order_id: OrderId
    side: OrderSide
    symbol: str
    size: float
    price: float
    frame_id: int
    timestamp: int
    cost: float = 0.0
    reason: str = ""

    @property
    def notional(self) -> float:
        return self.size * self.price


@dataclass
class CancelEvent:
    """撤单回报。"""

    order_id: OrderId
    frame_id: int
    accepted: bool
    reason: str = ""


# 条件单触发条件（§5.2）：在帧间由引擎监控，而非策略逐帧主动判断。
@dataclass
class Condition:
    """条件单的触发谓词。

    ``trigger`` 描述比较方式（见 ``Trigger``），``level`` 为触发价。
    引擎在 Phase 3 用当前帧 BBO/last/mark 与 ``level`` 比较判定触发。
    """

    trigger: Trigger
    level: float
    description: str = ""

    def is_met(self, quote) -> bool:  # pragma: no cover - 简单比较
        if self.trigger is Trigger.BID_LE:
            return quote.bid <= self.level
        if self.trigger is Trigger.BID_GE:
            return quote.bid >= self.level
        if self.trigger is Trigger.ASK_LE:
            return quote.ask <= self.level
        if self.trigger is Trigger.ASK_GE:
            return quote.ask >= self.level
        if self.trigger is Trigger.LAST_LE:
            return quote.last <= self.level
        if self.trigger is Trigger.LAST_GE:
            return quote.last >= self.level
        if self.trigger is Trigger.MARK_LE:
            return quote.mid <= self.level
        if self.trigger is Trigger.MARK_GE:
            return quote.mid >= self.level
        raise ValueError(f"未知触发方式 {self.trigger}")


@dataclass
class CorporateAction:
    """公司行动（Phase 3 Ledger）：拆股/分红/配股等。"""

    symbol: str
    kind: str  # 'split' / 'dividend' / 'right' / 'delist'
    date: int
    ratio: float = 1.0        # 拆股比例（如 10 拆 3 → 3.0）
    cash_per_share: float = 0.0  # 税前每股现金分红
    ex_date: int = 0


@dataclass
class LifecycleEvent:
    """衍生品生命周期：到期/行权/现金结算（Phase 6）。"""

    symbol: str
    kind: str  # 'expire' / 'exercise' / 'cash_settle'
    frame: int
    payoff: float
