"""具体选股器（§17.1）。

所有选择器都复用引擎的横截面接口 ``bus.cross_section``：它按「因子值」对选股池做
降序排名、算 z-score，并标记数据不足的标的为无效。因子通过 ``bus.register_factor``
声明（声明层 §4.2，性能/语义提示，非白名单）。

选择器在首次使用时向总线注册一个**实例唯一**的命名因子（``id(self)`` 入名），避免
多个选择器实例相互覆盖。
"""

from __future__ import annotations

from math import nan

from .base import SymbolSelector


def _last(values: list[float]) -> float:
    return values[-1] if values else nan


class FixedUniverseSelector(SymbolSelector):
    """不做过滤：整个选股池都要交易。

    用作基类或与其他选择器组合（``CompositeSelector`` 的「全集」 operand）。
    """

    def select(self, frame, token, bus, host, universe: list[str]) -> list[str]:
        return list(universe)


class CrossSectionRankSelector(SymbolSelector):
    """按注册因子的横截面排名选 TopK（或 Top 分数）。

    例：用「N 日动量」因子，每帧买入排名最高的 K 只——这正是 ``MomentumCrossSection``
    的选股部分，现被抽象成可复用的选择器，可与任意交易逻辑组合。

    参数：
      factor_name   ：注册因子名（实例唯一，默认 ``ret``）
      key           ：基础数据键（默认 ``close``）
      period        ：动量/收益率回溯交易日数
      top_k         ：持仓只数（与 ``top_fraction`` 二选一，缺省取全部有效标的）
      top_fraction  ：取有效标的的前分数（0..1]）
      lag           ：因子 lag（默认 1，排除当前帧防泄漏）
    """

    def __init__(
        self,
        factor_name: str = "ret",
        key: str = "close",
        period: int = 60,
        *,
        top_k: int | None = None,
        top_fraction: float | None = None,
        lag: int = 1,
    ):
        if top_k is not None and (top_fraction is None or 0 < top_fraction <= 1):
            if top_fraction is not None and top_k is not None:
                raise ValueError("top_k 与 top_fraction 二选一，不要同时指定")
        if top_fraction is not None and not (0 < top_fraction <= 1):
            raise ValueError("top_fraction 必须在 (0, 1]")
        if period <= 0:
            raise ValueError("period 必须为正")
        self.factor_name = f"_sel_{id(self)}_{factor_name}"
        self.key = key
        self.period = int(period)
        self.top_k = int(top_k) if top_k is not None else None
        self.top_fraction = float(top_fraction) if top_fraction is not None else None
        self.lag = int(lag)
        self._registered = False

    def _ensure_registered(self, bus) -> None:
        if self._registered:
            return
        period, key, fname = self.period, self.key, self.factor_name

        def _factor(seq, token):
            vals = [v for _ts, v in seq]
            if len(vals) < period:
                return nan
            # key 的 period 期收益率（可被用户注册的同名因子覆盖）
            return vals[-1] / vals[-period] - 1.0

        bus.register_factor(fname, _factor)
        self._registered = True

    def select(self, frame, token, bus, host, universe: list[str]) -> list[str]:
        self._ensure_registered(bus)
        rows = bus.cross_section(
            factor=self.factor_name,
            universe=universe,
            token=token,
            lag=self.lag,
            key=self.key,
        )
        valid = [s for s, r in rows.items() if r.is_valid]
        valid.sort(key=lambda s: rows[s].value, reverse=True)

        if self.top_k is not None:
            return valid[: self.top_k]
        if self.top_fraction is not None:
            k = max(1, int(len(valid) * self.top_fraction))
            return valid[:k]
        return valid


class TrendPresenceSelector(SymbolSelector):
    """只在「趋势向上」的标的里交易：收盘价位于均线之上。

    用「收盘价 / 均线」比值作因子（>1 即趋势向上），天然规避在下跌趋势里接刀。
    这是「挑选合适的标的」的典型手法：先筛掉趋势向下的标的，再在其上跑交易逻辑。

    参数：
      period        ：均线交易日数
      min_ratio     ：触发比值下限（默认 1.0；>1 要求「站上均线一定幅度」）
      lag           ：因子 lag（默认 1）
    """

    def __init__(self, period: int = 60, min_ratio: float = 1.0, lag: int = 1):
        if period <= 0:
            raise ValueError("period 必须为正")
        self.factor_name = f"_sel_{id(self)}_trend"
        self.period = int(period)
        self.min_ratio = float(min_ratio)
        self.lag = int(lag)
        self._registered = False

    def _ensure_registered(self, bus) -> None:
        if self._registered:
            return
        period = self.period

        def _factor(seq, token):
            vals = [v for _ts, v in seq]
            if len(vals) < period:
                return nan
            ma = sum(vals[-period:]) / period
            return vals[-1] / ma if ma > 0 else nan

        bus.register_factor(self.factor_name, _factor)
        self._registered = True

    def select(self, frame, token, bus, host, universe: list[str]) -> list[str]:
        self._ensure_registered(bus)
        rows = bus.cross_section(
            factor=self.factor_name,
            universe=universe,
            token=token,
            lag=self.lag,
            key="close",
        )
        return [
            s for s in universe
            if rows[s].is_valid and rows[s].value >= self.min_ratio
        ]


class LiquiditySelector(SymbolSelector):
    """只在流动性达标的标的里交易（按成交额 ``amount``）。

    避免陷入小盘/停牌标的。阈值 ``min_amount`` 为绝对成交额。
    """

    def __init__(self, min_amount: float, key: str = "amount", lag: int = 1):
        if min_amount <= 0:
            raise ValueError("min_amount 必须为正")
        self.factor_name = f"_sel_{id(self)}_liquidity"
        self.min_amount = float(min_amount)
        self.key = key
        self.lag = int(lag)
        self._registered = False

    def _ensure_registered(self, bus) -> None:
        if self._registered:
            return
        fname = self.factor_name

        def _factor(seq, token):
            return _last([v for _ts, v in seq])

        bus.register_factor(fname, _factor)
        self._registered = True

    def select(self, frame, token, bus, host, universe: list[str]) -> list[str]:
        self._ensure_registered(bus)
        rows = bus.cross_section(
            factor=self.factor_name,
            universe=universe,
            token=token,
            lag=self.lag,
            key=self.key,
        )
        return [s for s in universe if rows[s].is_valid and rows[s].value >= self.min_amount]


class CompositeSelector(SymbolSelector):
    """组合多个选择器：``and`` 取交集、``or`` 取并集。

    例：``TrendPresenceSelector(60) & LiquiditySelector(1e8)`` 只在「趋势向上且成交
    额充足」的标的里交易。保持各子选择器的降序语义：``and`` 以第一个选择器的顺序为
    主，``or`` 按首次出现的标的排序。
    """

    def __init__(self, selectors: list[SymbolSelector], combine: str = "and"):
        if combine not in ("and", "or"):
            raise ValueError("combine 必须是 'and' 或 'or'")
        if len(selectors) < 2:
            raise ValueError("至少需要两个选择器组合")
        self.selectors = list(selectors)
        self.combine = combine

    def select(self, frame, token, bus, host, universe: list[str]) -> list[str]:
        groups = [set(s.select(frame, token, bus, host, universe)) for s in self.selectors]
        if self.combine == "and":
            chosen = groups[0].intersection(*groups[1:])
        else:
            chosen = groups[0].union(*groups[1:])
        # 保持第一个选择器的降序
        order = self.selectors[0].select(frame, token, bus, host, universe)
        return [s for s in order if s in chosen]
