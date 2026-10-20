"""Data Bus：因果数据总线（§4）。

核心：**所有数据访问必须附带引擎签发的 CausalToken**。引擎在返回前强制过滤
``timestamp <= token.issued_at`` 的记录；显式请求越界数据抛出 ``CausalityViolation``。

写侧（I5）：引擎启动后 ``enable_write_guard(key)``，``record / set_now / reset`` 必须携带
同一个私有 key；策略只拿到 bus，调用会抛 ``CausalityViolation``，无法改写历史或权威时间。
未启用 guard 的离线用法（单测/研究脚本）行为不变。

实现（按设计文档阶段递进）：
  - 单序列 ``get()``：按因子名索引的列存序列，支持 window/period 与 lag。
  - lag 原语（§4.3）：强制排除当前帧，窗口变为 ``[t-window, t-lag]``。
  - 多时间框架对齐（§4.4）：``closed_only`` 语义。
  - 横截面请求（§4.5）：``cross_section()`` 返回 ``{asset: FactorRow}``。
  - 声明层（§4.2）：``register_factor()`` 注册命名因子（性能/语义提示，非白名单）。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

from .core.data_types import CausalToken, CausalityViolation


@dataclass(frozen=True)
class FactorRow:
    """横截面矩阵中的一行（§4.5）。"""

    value: float
    rank: int          # 1 = 最高（对长因子）
    zscore: float
    is_valid: bool


FactorFn = Callable[[list[tuple[int, float]], "CausalToken"], float]


class DataBus:
    """因果数据总线。

    内部按 ``(key, symbol)`` 存列存序列：``list[(timestamp, value)]``，按注入时间单调增。
    策略声明的 ``prefetch_hints`` 属于性能提示（§4.2），非白名单；未声明的请求
    实时计算并记录请求模式。
    """

    def __init__(self):
        self._series: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        self._factors: dict[str, FactorFn] = {}
        self.request_log: list[str] = []  # 记录未注册因子的请求模式（§4.2）
        self._seq_counter = 0
        # 引擎授权的权威「当前虚拟时间」。None = 无引擎驱动（直接使用总线），
        # 此时 token 自身权威（便于单测/离线分析）。有引擎时用于钳制被伪造的 token。
        self._now_ns: Optional[int] = None
        self.forgery_attempts: int = 0
        # I5 写保护：引擎启动时用 enable_write_guard(key) 设置一个私有 key，
        # 之后 record / set_now / reset 必须携带同一个 key（identity 比较）。
        # 策略只拿到 bus，拿不到 key，因此无法改写历史或篡改权威时间。
        # 未启用 guard 时（离线分析 / 单元测试）行为不变。
        self._guard_key: Optional[object] = None

    def enable_write_guard(self, key: object) -> None:
        """引擎调用：启用写保护。``key`` 由引擎私有持有，不传给策略。

        已启用后不允许用新 key 替换（否则策略可先 ``enable_write_guard(new_key)``
        再写历史）。重复用同一个 key 调用是幂等的。
        """
        if key is None:
            raise ValueError("write guard key 不能为 None")
        if self._guard_key is not None and key is not self._guard_key:
            raise CausalityViolation("DataBus 写保护已启用，不能替换 guard key（I5）")
        self._guard_key = key

    def _check_write(self, key: Optional[object]) -> None:
        if self._guard_key is not None and key is not self._guard_key:
            raise CausalityViolation("DataBus 写操作仅限引擎持有者（I5：不可改写历史/权威时间）")

    def set_now(self, ts: int, *, _key: Optional[object] = None) -> None:
        """引擎每帧调用：登记权威当前虚拟时间（§4.1 token 不可伪造的强制点）。

        启用写保护后必须携带引擎私有 key；策略调用会抛 ``CausalityViolation``。
        """
        self._check_write(_key)
        self._now_ns = int(ts)

    def _effective_ts(self, token: CausalToken) -> int:
        """取生效时间上界。

        设计文档 §4.1 要求 token「由引擎签发，策略不可伪造」。但 ``CausalToken``
        是普通对象，策略完全可以自造一个 ``issued_at`` 更大的实例。因此这里不用
        token 自称的时间，而是与引擎的权威时间取 **较小者** —— 伪造 token 拿不到
        任何额外数据，I1 由总线自身而非「恰好没有预加载未来数据」来保证。
        """
        if self._now_ns is None:
            return token.issued_at_ns
        if token.issued_at_ns > self._now_ns:
            self.forgery_attempts += 1
            return self._now_ns
        return token.issued_at_ns

    def record(self, key: str, symbol: str, timestamp: int, value: float, *,
               _key: Optional[object] = None) -> None:
        """引擎在 Phase 1 注入 Bar 时调用：把 OHLCV 写进列存。

        启用写保护后必须携带引擎私有 key；策略调用会抛 ``CausalityViolation``。
        """
        self._check_write(_key)
        self._series[(key, symbol)].append((int(timestamp), float(value)))

    def reset(self, *, _key: Optional[object] = None) -> None:
        self._check_write(_key)
        self._series.clear()
        self.request_log.clear()
        self._now_ns = None
        self.forgery_attempts = 0

    def register_factor(self, name: str, fn: FactorFn) -> None:
        """注册一个命名因子（§4.2 声明层）。"""
        self._factors[name] = fn

    # ---- 单序列请求（§4.1 / §4.3 / §4.4）----

    def get(
        self,
        key: str,
        symbol: Optional[str] = None,
        *,
        token: Optional[CausalToken] = None,
        window: Optional[int] = None,
        period: Optional[int] = None,
        lag: int = 0,
        closed_only: bool = True,
        all_history: bool = False,
    ) -> list[float]:
        """返回因子序列（按时间升序）。

        因果语义：
          - 必须携带 token；无 token 直接抛 CausalityViolation。
          - 返回记录的 timestamp 必须 <= token.issued_at - lag。
          - window/period 取最近 N 个有效值。
          - lag>0 强制排除当前帧（§4.3），窗口变为 ``[t-window, t-lag]``。
          - closed_only=True：排除「正在形成」的当前帧（§4.4，默认）。
        """
        if token is None:
            raise CausalityViolation("bus.get 必须附带引擎签发的 CausalToken")
        if lag < 0:
            raise CausalityViolation(f"lag={lag} 为负，等于显式请求未来数据（违反 I1）")

        # 生效时间：钳制到引擎权威时间，伪造 token 无效（§4.1）
        now = self._effective_ts(token)
        allowed_ts = now - int(lag)
        if allowed_ts < 0:
            raise CausalityViolation(f"lag={lag} 超出当前虚拟时间 {now}")

        if symbol is None:
            series: list[tuple[int, float]] = []
            for (k, _s), seq in self._series.items():
                if k == key:
                    series.extend(seq)
        else:
            series = list(self._series.get((key, symbol), []))

        # 因果过滤：只返回 timestamp <= allowed_ts 的记录（这就是 I1 的强制点）。
        # 注意：总线内部可能持有 issued_at 之后的记录（引擎跨帧增量注入是常态，
        # 预加载式数据源亦然），这本身不违反 I1 —— 策略无法通过任何接口取到它们。
        filtered = [(ts, v) for ts, v in series if ts <= allowed_ts]

        values = [v for _ts, v in filtered]

        if not all_history and (window is not None or period is not None):
            n = window if window is not None else period
            if n is not None:
                # window=0 不得退化成「全量」：Python 的 values[-0:] 就是 values[0:]。
                # 负数窗口没有定义，显式报错而不是返回一段无意义切片。
                if n < 0:
                    raise ValueError(f"window/period 不能为负：{n}")
                values = values[-n:] if n > 0 else []

        return values

    def series(
        self,
        key: str,
        symbol: str,
        token: CausalToken,
        lag: int = 0,
    ) -> list[tuple[int, float]]:
        """返回 ``(timestamp, value)`` 原始序列（用于自定义因子计算）。

        ``lag`` 的因果语义与 :meth:`get` 完全一致：负 lag 等于显式请求未来数据，
        必须与 ``get`` 一样抛 ``CausalityViolation``（否则 ``series`` 会成为
        I1 的旁路；``cross_section`` 也走这里）。
        """
        if lag < 0:
            raise CausalityViolation(f"lag={lag} 为负，等于显式请求未来数据（违反 I1）")
        allowed_ts = self._effective_ts(token) - int(lag)
        seq = self._series.get((key, symbol), [])
        return [(ts, v) for ts, v in seq if ts <= allowed_ts]

    # ---- 多时间框架对齐（§4.4）----

    def get_multi_tf(
        self,
        key: str,
        symbol: str,
        token: CausalToken,
        *,
        bucket: int = 5,
    ) -> tuple[list[tuple[int, float]], Optional[tuple[int, float]]]:
        """把基础序列按每 ``bucket`` 根聚合为高周期 bar（§4.4 多时间框架对齐）。

        返回 ``(closed_bars, forming_bar)``：
          - ``closed_bars``：已收盘的高周期 bar ``[(ts, value), ...]``，可安全用于信号；
          - ``forming_bar``：当前正在形成的那根 ``(ts, value)``，未满 ``bucket`` 根；
            恰好整除时为 ``None``。

        语义（§4.4）：**默认禁止在 forming bar 上触发信号**（因其 high/low 还会随帧变化），
        策略如确需使用必须显式取 ``forming_bar`` 并自担同帧泄漏责任。
        聚合规则：每桶取最后一根基础 bar 的时间戳与值（等价于「收盘价」聚合）。
        """
        if bucket <= 0:
            raise ValueError("bucket 必须为正")
        seq = self.series(key, symbol, token)
        n_closed = len(seq) - len(seq) % bucket
        closed = [(seq[i + bucket - 1][0], seq[i + bucket - 1][1])
                  for i in range(0, n_closed, bucket)]
        forming = (seq[-1][0], seq[-1][1]) if len(seq) % bucket else None
        return closed, forming

    # ---- 横截面请求（§4.5）----

    def cross_section(
        self,
        factor: str,
        universe: list[str],
        token: CausalToken,
        lag: int = 0,
        key: str = "close",
    ) -> dict[str, FactorRow]:
        """计算横截面因子矩阵（§4.5）。

        返回 ``dict[asset -> FactorRow]``：
          value：因子值（未通过注册因子的，回退为 ``key`` 的最近值）
          rank ：1 = 最高值（长因子约定）
          zscore：标准化
          is_valid：数据是否充分

        约束：所有标的的数据时间戳必须一致（日线天然满足）。
        停牌/数据不足的标的 ``is_valid=False``，value 为 NaN。
        """
        universe = list(universe)
        raw: dict[str, float] = {}

        for sym in universe:
            if factor in self._factors:
                seq = self.series(key, sym, token, lag=lag)
                try:
                    raw[sym] = self._factors[factor](seq, token)
                except Exception:  # noqa: BLE001 - 因子计算失败视为无效
                    raw[sym] = float("nan")
            else:
                # 未注册因子：回退为 key 的最近有效值
                vals = self.get(key, sym, token=token, lag=lag)
                raw[sym] = vals[-1] if vals else float("nan")

        # 有效子集
        valid = {s: v for s, v in raw.items() if v == v}  # 排除 NaN
        if valid:
            vals = list(valid.values())
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / len(vals)
            std = var ** 0.5
            ordered = sorted(valid, key=lambda s: valid[s], reverse=True)  # 高→低
            ranks = {s: i + 1 for i, s in enumerate(ordered)}
        else:
            mean = std = 0.0
            ranks = {}

        out: dict[str, FactorRow] = {}
        for sym in universe:
            v = raw[sym]
            is_valid = v == v and sym in ranks
            out[sym] = FactorRow(
                value=v,
                rank=ranks.get(sym, 0),
                zscore=((v - mean) / std) if std > 0 else 0.0,
                is_valid=is_valid,
            )
        return out

    # ---- 高级接口占位（后续 Phase）----

    def query(self, *args, **kwargs):  # pragma: no cover - Phase 1+
        raise NotImplementedError("复杂查询在 Phase 1+ 实现")

    def subscribe(self, *args, **kwargs):  # pragma: no cover - Phase 5
        raise NotImplementedError("事件订阅在 Phase 5 实现")
