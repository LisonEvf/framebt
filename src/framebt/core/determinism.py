"""确定性随机源（I3 保障之一）。

引擎管理 RNG，策略禁止使用内置 ``random`` / ``numpy.random``，否则破坏逐位复现。
这里实现 mulberry32（快速、确定性、周期 2^32），并补一个 Box-Muller 正态采样。

该 RNG 不追求密码学强度，只追求：给定 seed 后，任意机器上同一调用序列
产出逐位相同的 float/int 序列。
"""

from __future__ import annotations

import math


class DetermRNG:
    """mulberry32 + Box-Muller 确定性随机数生成器。

    用法与 numpy 风格接近，但完全可复现：
        rng = DetermRNG(seed=1234)
        x = rng.random()          # [0, 1)
        n = rng.normal(0, 1)      # 标准正态
        i = rng.int(0, 100)       # 整数 [lo, hi)
    """

    MASK = 0xFFFFFFFF

    def __init__(self, seed: int = 0):
        # mulberry32 初始状态必须为非零
        self._state = (int(seed) ^ 0x1234567) & self.MASK or 0x04D5A4C6

    def _next_u32(self) -> int:
        self._state = (self._state + self.MASK) & self.MASK
        z = (self._state ^ (self._state >> 15)) * (1 | self._state) & self.MASK
        z = (z ^ (z >> 11)) & self.MASK
        return z

    def random(self) -> float:
        """返回 [0, 1) 内的浮点。"""
        return (self._next_u32() >> 5) / float(1 << 27)

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        """标准正态采样（Box-Muller，确定性近似）。"""
        u1 = self.random()
        u2 = self.random()
        if u1 < 1e-300:
            u1 = 1e-300
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return mu + sigma * z

    def int(self, lo: int, hi: int) -> int:
        """返回 [lo, hi) 内的整数。"""
        if hi <= lo:
            raise ValueError("int(lo, hi) requires hi > lo")
        span = hi - lo
        return lo + (self._next_u32() % span)

    def shuffle(self, seq: list) -> list:
        """Fisher–Yates 确定性打乱（返回新列表，不修改入参）。"""
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.int(0, i + 1)
            out[i], out[j] = out[j], out[i]
        return out

    def getstate(self) -> int:
        return self._state

    def setstate(self, state: int) -> None:
        self._state = int(state) & self.MASK or 0x04D5A4C6
