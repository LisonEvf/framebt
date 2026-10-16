"""核心数据类型：因果令牌与越界异常（§4.1）。

``CausalToken`` 由引擎每帧签发，携带当前虚拟时间；策略不可伪造。
所有数据访问必须附带它，引擎在返回前强制过滤未来数据（I1）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


class CausalityViolation(Exception):
    """策略试图访问尚未发生（issued_at 之后）的数据时抛出。

    这是引擎的硬约束：违反 I1 的请求要么被静默过滤（拿不到未来值），
    要么显式请求越界时抛出本异常。
    """


@dataclass
class CausalToken:
    """因果令牌：引擎每帧签发的数据访问凭证，携带当前虚拟时间。

    策略不可伪造；所有 ``bus.get(...)`` / ``bus.cross_section(...)`` 必须附带本对象。
    """

    __slots__ = ("issued_at_ns", "frame_id", "sub_step")

    def __init__(self, issued_at, frame_id: int, sub_step: Optional[int] = None):
        # issued_at / issued_at_ns 同义（文档 §4.1 用 issued_at）
        self.issued_at_ns = int(issued_at)
        self.frame_id = int(frame_id)
        self.sub_step = sub_step

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"CausalToken(issued_at={self.issued_at_ns}, "
            f"frame={self.frame_id}, sub_step={self.sub_step})"
        )
