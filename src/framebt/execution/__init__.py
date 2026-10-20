"""framebt.execution：订单与撮合（§5 / Phase 6）。

- backend  ：ExecutionBackend（订单 / 撮合后端抽象 —— 回测 MatchingEngine，实盘 LiveConnector）
- engine   ：MatchingEngine（延迟管线 + 价格-时间优先 + 订单类型 + 条件单 + OCO）
- book     ：OrderBook（价格-时间优先订单簿 —— Phase 2 帧内子步进执行引擎的排队结构）
- groups   ：OrderGroup / Atomicity / LinkMode（组单原子性 §5.3、OCO §5.1）
"""

from __future__ import annotations

from .backend import ExecutionBackend
from .engine import MatchingEngine, FillResult, CancelResult, OrderBook
from .groups import OrderGroup, Atomicity, LinkMode

__all__ = [
    "ExecutionBackend",
    "MatchingEngine",
    "FillResult",
    "CancelResult",
    "OrderBook",
    "OrderGroup",
    "Atomicity",
    "LinkMode",
]
