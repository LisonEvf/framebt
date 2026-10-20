"""策略 A/B 测试：同一 300 标的池，对比 4 种多标的策略是否有 edge。

  1. Buy&Hold        ：等权买入持有全池（被动基准）
  2. MomentumDaily   ：每日动量轮动 Top30（60 日动量）
  3. Momentum60D     ：每 60 交易日动量轮动 Top30
  4. Reversal60D     ：每 60 交易日反转轮动 Top30（买近 20 日 loser）

结论：哪种设计在清洗后的数据上真正盈利、且跑赢基准。
"""
from __future__ import annotations

from framebt import Engine, Performance, StandardAshareRules, StandardCost
from framebt.core import FrameStrategy, Order, OrderSide, OrderType


# ---------------------------------------------------------------- 公共部件
def _prepare_universe(src, max_symbols=300, min_bars=260):
    universe = src.available_symbols()
    feed = src.universe(universe)
    counts, amounts = {}, {}
    for b in feed.bars:
        counts[b.symbol] = counts.get(b.symbol, 0) + 1
        amounts[b.symbol] = amounts.get(b.symbol, 0.0) + b.amount
    avg = {s: a / max(1, counts[s]) for s, a in amounts.items()}
    elig = [s for s in universe if counts.get(s, 0) >= min_bars]
    elig.sort(key=lambda s: avg.get(s, 0), reverse=True)
    return elig[:max_symbols]


def _cross_return(bus, sym, period, token, key="close"):
    seq = bus.series(key, sym, token, lag=1)
    vals = [v for _t, v in seq]
    if len(vals) < period:
        return None
    return vals[-1] / vals[-period] - 1.0


def _mom(seq, token):
    v = [x for _t, x in seq]
    return v[-1] / v[-60] - 1 if len(v) >= 60 else float("nan")


def _rev(seq, token):
    v = [x for _t, x in seq]
    return v[-1] / v[-20] - 1 if len(v) >= 20 else float("nan")


class _Rebalance(FrameStrategy):
    """通用横截面轮动：每 ``rebalance_days`` 帧把持仓重平衡到 ``selector`` 选出的 TopK。

    ``selector(token, bus, universe, frame) -> (factor_name, reverse)`` 决定买谁：
      动量：rank 降序（买强者）；反转：rank 升序（买弱者）。
    等权分配：目标名义市值 = 权益 / 目标标的数；超买/超卖对齐。
    """

    def __init__(self, universe, factor_fn, top_k, rebalance_days,
                 lookback=60, reverse=False):
        self.universe = list(universe)
        self.factor_fn = factor_fn
        self.top_k = top_k
        self.rebalance_days = rebalance_days
        self.lookback = lookback
        self.reverse = reverse
        self._since = 0
        self._last_ts = None
        self._reg = False

    def _ensure(self, bus):
        if self._reg:
            return
        bus.register_factor("abf", self.factor_fn)
        self._reg = True

    def on_frame(self, frame, token, bus, host) -> list:
        self._ensure(bus)
        ts = frame.frame_id
        if self._last_ts is not None and ts - self._last_ts >= self.rebalance_days:
            self._since = 0
            self._last_ts = ts
        self._since += 1
        if self._since < self.rebalance_days:
            return []

        rows = bus.cross_section(factor="abf", universe=self.universe,
                                 token=token, lag=1, key="close")
        valid = [(s, r.value) for s, r in rows.items() if r.is_valid and r.value is not None]
        valid.sort(key=lambda t: t[1], reverse=not self.reverse)
        target = [s for s, _ in valid[: self.top_k]]
        if not target:
            return []

        pos = host.portfolio
        held = {s: n for s, n in pos.exposure().items() if n > 0}
        equity = pos.cash + sum(n * frame.last(s) for s, n in held.items()
                                if frame.has_bar(s))
        per = equity / len(target)

        orders = []
        # 跌出目标 → 清
        for s in held:
            if s not in target:
                orders.append(Order(OrderSide.SELL, s, size=held[s],
                                    order_type=OrderType.MARKET))
        cash = pos.cash
        # 对齐目标权重
        for s in target:
            cur = held.get(s, 0)
            delta = per - cur * (frame.last(s) if frame.has_bar(s) else 0)
            if delta > 0:  # 加仓
                if cash > 0:
                    orders.append(Order(OrderSide.BUY, s, budget=min(delta, cash),
                                        order_type=OrderType.MARKET))
            elif delta < 0 and cur > 0:  # 减仓
                orders.append(Order(OrderSide.SELL, s, size=-delta / frame.last(s),
                                    order_type=OrderType.MARKET))
        return orders


class _BuyAndHold(FrameStrategy):
    """等权买入持有全池（被动基准）：每标的首次有数据时买入均额并持有。"""

    def __init__(self, universe):
        self.universe = list(universe)
        self._done = set()

    def on_frame(self, frame, token, bus, host) -> list:
        pos = host.portfolio
        held = [s for s in self.universe if s not in self._done and pos.position(s).net_size() == 0]
        if not held:
            return []
        # 每帧最多开 10 仓，避免单帧巨量订单
        batch = held[:10]
        per = pos.cash / len(self.universe)
        orders = []
        for s in batch:
            orders.append(Order(OrderSide.BUY, s, budget=per, order_type=OrderType.MARKET))
            self._done.add(s)
        return orders


def _run(name, strategy, universe, cost):
    eng = Engine(src.universe(universe), strategy, initial_cash=1_000_000.0,
                 settlement_cycle=1, cost_model=cost, rules=StandardAshareRules(limit=0.10))
    res = eng.run()
    p = Performance(res)
    fills = res.fills
    pnls, wins, losses = [], [], []
    open_n, open_s = {}, {}
    for f in fills:
        if f.side is OrderSide.BUY:
            open_n[f.symbol] = open_n.get(f.symbol, 0.0) + f.size * f.price
            open_s[f.symbol] = "buy"
        elif open_s.get(f.symbol) == "buy":
            pnl = f.size * f.price - open_n.pop(f.symbol, 0.0)
            pnls.append(pnl)
            (wins if pnl > 0 else losses).append(pnl)
    pf = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    print(f"\n=== {name} ===")
    print(f"  总收益 {p.total_return:+.2%}  年化 {p.annualized_return:+.2%}  "
          f"Sharpe {p.sharpe:.2f}  最大回撤 {p.max_drawdown:.2%}  "
          f"胜率 {p.win_rate():.0%}  盈亏比/盈利因子 {((sum(wins)/len(wins))/(abs(sum(losses)/len(losses))) if wins and losses else 0):.2f}/{pf:.2f}")
    print(f"  成交 {res.total_trades():,} 笔 {len(pnls)} 回合 {len({f.symbol for f in fills})} 标的  费用 {sum(f.cost for f in fills):,.0f}")
    return p, dict(trades=res.total_trades(), rt=p.total_return, sharpe=p.sharpe,
                   dd=p.max_drawdown, wr=p.win_rate(), names=len({f.symbol for f in fills}))


def _build(name, universe):
    cost = StandardCost(commission=0.0003, min_commission=5.0, stamp_tax=0.001,
                        transfer_fee=0.00002, slippage="volume_based", k=0.001)
    if name == "Buy&Hold":
        # 每日等权轮动全池 = 被动基准（用同一 _Rebalance，top_k=全池，1 日再平衡）
        return _Rebalance(universe, _mom, len(universe), 1, reverse=False), cost
    if name == "MomentumDaily":
        return _Rebalance(universe, _mom, 30, 1, reverse=False), cost
    if name == "Momentum60D":
        return _Rebalance(universe, _mom, 30, 60, reverse=False), cost
    if name == "Reversal60D":
        return _Rebalance(universe, _rev, 30, 60, reverse=True), cost
    raise ValueError(name)


if __name__ == "__main__":
    import sys
    import pandas as pd
    from framebt.data_sources import OpentdxParquetSource

    src = OpentdxParquetSource()
    universe = _prepare_universe(src)
    feed = src.universe(universe)
    lo, hi = feed.span()
    from datetime import datetime
    print(f"universe={len(universe)}  span={datetime.fromordinal(int(lo)).date()}..{datetime.fromordinal(int(hi)).date()}")

    names = sys.argv[1:] or ["Buy&Hold", "MomentumDaily", "Momentum60D", "Reversal60D"]
    for name in names:
        strat, cost = _build(name, universe)
        _run(name, strat, universe, cost)
