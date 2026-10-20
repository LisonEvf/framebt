"""资金策略：凯利公式 + 即时胜率（§17.2）。

传统凯利分数（二元重复下注）
------------------------------
    f* = p − q / b ，其中 p = 胜率、q = 1 − p、b = 盈亏比（平均盈利 / 平均亏损）

``f*`` 是使长期几何增长最大化的「总权益下注比例」。实盘里全凯利波动极大且估计误差
敏感，故默认用**半凯利**（``fraction=0.5``），并可上下界裁剪（``min/max_fraction``）。

分配
----
凯利分数决定「总共投入多少权益」，再由
``available = cash − equity × (1 − f)`` 换算成「本帧可新增投入」——它自动把已建仓的
市值扣回，避免「已 40% 仓位 + 再投全额凯利」式的超配。新增资金在活跃标的间**等权**
分配（各标的用同一策略工厂，边际等权是合理默认）。

样本不足（round-trip 数 < ``min_samples``）时退化为 ``warmup_fraction``（默认全额），
保证回测开局就能建仓。

因果性
------
只用**已落地成交**（``FillEvent``）反推已实现盈亏，不触碰任何未来信息；每帧的权益
用于换算「可新增投入」，而权益本身来自已成交与当前帧标记，均不越界。
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

from .base import SymbolBudgetPolicy


class KellyBudgetPolicy(SymbolBudgetPolicy):
    """凯利公式资金策略。

    参数：
      fraction        ：凯利分数缩放（0.5 = 半凯利；1.0 = 全凯利）
      min_samples     ：触发凯利的最少完整往返笔数（默认 5）
      warmup_fraction ：样本不足时的降级投入比例（默认 1.0 = 全额，等权分配）
      min_fraction    ：下限裁剪（默认 0，即胜率过低时可不下注）
      max_fraction    ：上限裁剪（默认 1.0；建议按风险偏好调低）
      lookback        ：参与统计的最大往返笔数（默认 200）
    """

    def __init__(
        self,
        fraction: float = 0.5,
        min_samples: int = 5,
        warmup_fraction: float = 1.0,
        min_fraction: float = 0.0,
        max_fraction: float = 1.0,
        lookback: int = 200,
    ):
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction 必须在 (0, 1]")
        if min_samples < 1:
            raise ValueError("min_samples 必须 >= 1")
        self.fraction = float(fraction)
        self.min_samples = int(min_samples)
        self.warmup_fraction = float(warmup_fraction)
        self.min_fraction = float(min_fraction)
        self.max_fraction = float(max_fraction)
        self.lookback = int(lookback)

        # symbol -> 最近若干完整往返的已实现盈亏（金额）
        self._trips: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.lookback))
        # symbol -> 当前未平的买入名义市值
        self._open_buy: dict[str, float] = {}

    # ---- 观察成交 ----

    def record(self, fill) -> None:
        """喂入一笔成交，维护各标的的往返盈亏。"""
        sym = fill.symbol
        notional = fill.size * fill.price
        if fill.side.value == "buy":
            self._open_buy[sym] = notional
        elif fill.side.value == "sell" and self._open_buy.get(sym) is not None:
            cost = self._open_buy.pop(sym)
            self._trips[sym].append(notional - cost)

    def note_fills(self, fills: Sequence) -> None:
        for f in fills:
            self.record(f)

    # ---- 凯利分数 ----

    def _win_rate_and_payoff(self) -> tuple[int, float]:
        trips: list[float] = []
        for q in self._trips.values():
            trips.extend(q)
        return len(trips)

    def kelly_fraction(self) -> float:
        """由近期往返盈亏计算（缩放后）的凯利总投入比例。

        样本不足或无法评估时返回 ``warmup_fraction``。
        """
        all_trips: list[float] = []
        for q in self._trips.values():
            all_trips.extend(q)
        n = len(all_trips)
        if n < self.min_samples:
            return self.warmup_fraction

        wins = [t for t in all_trips if t > 0]
        losses = [-t for t in all_trips if t < 0]
        p = len(wins) / n
        q = 1.0 - p
        avg_win = sum(wins) / len(wins) if wins else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        b = avg_win / avg_loss if avg_loss > 0 else 0.0
        if b <= 0:
            return self.warmup_fraction

        f = p - q / b                 # 全凯利
        f = self.fraction * f          # 缩放
        return max(self.min_fraction, min(self.max_fraction, f))

    # ---- SymbolBudgetPolicy 契约 ----

    def budget(self, orders, active, host, frame) -> list:
        """把各买入腿的预算归一化到凯利总额，并在活跃标的间等权。"""
        from ..core.events import OrderSide

        cash = host.portfolio.cash
        equity = self._equity(host, frame)
        if equity <= 0:
            return orders

        f = self.kelly_fraction()
        # 可新增投入 = 总现金 − 为保留 (1-f) 而留存的缓冲（已建仓自动扣回）
        available = cash - equity * (1.0 - f)
        if available <= 0:
            # 无新增资金：丢弃所有买入腿（保留卖出腿）
            return [o for o in orders if o.side is not OrderSide.BUY]

        n = max(1, len(active))
        per = available / n

        kept: list = []
        for o in orders:
            if o.side is OrderSide.BUY and o.budget > 0:
                new_budget = min(o.budget, per)
                if new_budget <= 0:
                    continue
                o.budget = new_budget
            kept.append(o)
        return kept

    def _equity(self, host, frame) -> float:
        """现金 + 持仓按当前帧标记（长仓口径：权益 − 现金 = 持仓市值）。"""
        cash = host.portfolio.cash
        value = 0.0
        for sym, size in host.portfolio.exposure().items():
            if size <= 0:
                continue
            if frame.has_bar(sym):
                value += size * frame.last(sym)
        return cash + value
