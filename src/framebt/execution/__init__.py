"""framebt.execution：订单与撮合（§5）。

- engine  ：MatchingEngine（延迟管线 + 价格-时间优先 + 订单类型 + 条件单 + OCO）
- groups  ：OrderGroup / Atomicity / LinkMode（组单原子性 §5.3、OCO §5.1）
"""

from __future__ import annotations

from .engine import MatchingEngine, FillResult, CancelResult, OrderBook
from .groups import OrderGroup, Atomicity, LinkMode

__all__ = [
    "MatchingEngine",
    "FillResult",
    "CancelResult",
    "OrderBook",
    "OrderGroup",
    "Atomicity",
    "LinkMode",
]
