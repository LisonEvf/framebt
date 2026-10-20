"""§17 RotationStrategy + 选股/资金抽象的回归测试。

覆盖：
  - 多标的轮动：包装器确实交易多个标的（而非只磕一只）
  - 在途去重：挂单未落地时不重复下单（否则仓位爆炸）
  - on_fill 转发：条件单策略经包装器后仍能挂触发单
  - 选股器过滤：Trend/Composite 只选符合条件的标的
  - 凯利资金策略：无负现金、无爆炸；凯利分数计算正确
  - 资金均分：权益在多个活跃标的间分配，而非单点压满
"""

from __future__ import annotations

import math

import pytest

from framebt import (
    BarFeed, Engine, NoOpCost, StandardCost, StandardRisk,
    FrameStrategy, Order, OrderSide, OrderType, FillEvent,
)
from framebt.core.host import WakeupType
from framebt.strategies import RotationStrategy, DualMAStrategy, TrendWithStop
from framebt.selection import (
    FixedUniverseSelector, CrossSectionRankSelector, TrendPresenceSelector,
    CompositeSelector, LiquiditySelector, KellyBudgetPolicy,
)


def _multi_symbol_bars(symbols: list[str], n: int = 120, base: float = 10.0) -> list[dict]:
    """多标的合成日线：各标的动量相位不同，使横截面排名 / 趋势判断会随时间变化。"""
    bars = []
    for k, sym in enumerate(symbols):
        price = base + k
        for i in range(n):
            price *= 1.0 + 0.03 * math.sin(i / 4.0 + k * 1.7)
            bars.append({"timestamp": i, "symbol": sym, "open": price * 0.995,
                         "high": price * 1.01, "low": price * 0.99,
                         "close": price, "volume": 1e6})
    return bars


def _trend_bars() -> tuple[list[dict], list[str]]:
    """UP 单调上涨（始终 > 均线）、DOWN 单调下跌（始终 < 均线），成交额均达标。"""
    bars = []
    syms = ["UP", "DOWN"]
    for sym, start, step in [("UP", 10.0, 0.05), ("DOWN", 20.0, -0.05)]:
        p = start
        for i in range(80):
            p += step
            bars.append({"timestamp": i, "symbol": sym, "open": p, "high": p,
                         "low": p, "close": p, "volume": 1e6, "amount": p * 1e6})
    return bars, syms


def _resting_count(eng) -> int:
    return len(eng.host.matcher._resting)


# ---------------------------------------------------------------------------
# 多标的轮动
# ---------------------------------------------------------------------------
def test_rotation_trades_multiple_symbols():
    """FixedUniverse + DualMA：应交易多个标的，而非只持有一只。"""
    syms = ["S0", "S1", "S2", "S3", "S4"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")
    rot = RotationStrategy(syms, FixedUniverseSelector(),
                           make=lambda s: DualMAStrategy(s, short=5, long=20))
    res = Engine(feed, rot, initial_cash=1_000_000.0, cost_model=NoOpCost()).run()

    traded = {f.symbol for f in res.fills}
    assert len(traded) > 1, f"应交易多个标的，实际 {traded}"
    assert traded <= set(syms)
    assert res.final_equity() > 0


def test_rotation_budget_spreads_across_symbols():
    """资金应在多个活跃标的间分配：结束时存在多只非零持仓。"""
    syms = ["S0", "S1", "S2", "S3"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")
    rot = RotationStrategy(
        syms, FixedUniverseSelector(),
        make=lambda s: DualMAStrategy(s, short=5, long=20),
        budget_policy=KellyBudgetPolicy(fraction=0.5, min_samples=1),
    )
    eng = Engine(feed, rot, initial_cash=1_000_000.0, cost_model=NoOpCost())
    eng.run()

    held = {s: n for s, n in eng.ledger.exposure().items() if n > 0}
    assert len(held) >= 2, f"应有至少 2 只持仓，实际 {held}"


# ---------------------------------------------------------------------------
# 在途去重（回归：未去重 → 重复挂单 → 仓位爆炸）
# ---------------------------------------------------------------------------
class _LimitBuyProbe(FrameStrategy):
    """每帧无条件发一笔远低于市价的限价买单（永远不成交、始终挂单中）。"""

    def __init__(self, symbol: str):
        self.symbol = symbol

    def on_frame(self, frame, token, bus, host) -> list:
        return [Order(side=OrderSide.BUY, symbol=self.symbol, size=10,
                      order_type=OrderType.LIMIT, price=1.0)]


def test_rotation_dedups_in_flight_orders():
    """去重开启：每个标的只挂 1 笔限价单（其余被跳过）；关闭则逐帧堆积。"""
    syms = ["S0", "S1", "S2", "S3"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")

    eng_on = Engine(feed, RotationStrategy(
        syms, FixedUniverseSelector(), make=_LimitBuyProbe, dedup_in_flight=True),
        initial_cash=1_000_000.0)
    eng_on.run()
    assert _resting_count(eng_on) == len(syms), \
        f"去重开启应只剩 {len(syms)} 笔挂单，实际 {_resting_count(eng_on)}"

    eng_off = Engine(feed, RotationStrategy(
        syms, FixedUniverseSelector(), make=_LimitBuyProbe, dedup_in_flight=False),
        initial_cash=1_000_000.0)
    eng_off.run()
    assert _resting_count(eng_off) > len(syms) * 5, \
        f"去重关闭应大量堆积，实际 {_resting_count(eng_off)}"


# ---------------------------------------------------------------------------
# on_fill 转发（条件单策略经包装器后仍能挂触发单）
# ---------------------------------------------------------------------------
def test_rotation_forwards_on_fill_to_place_stops():
    """TrendWithStop 经 RotationStrategy：买入后挂的止损单应真正触发并成交。"""
    # 单标的序列：先涨（触发买入）→ 后急跌（触发止损）
    closes = [10, 10, 11, 12, 13, 14, 8, 8, 8, 8]
    bars = [{"timestamp": i, "symbol": "S0", "open": c, "high": c * 1.01,
             "low": c * 0.99, "close": c, "volume": 1e6}
            for i, c in enumerate(closes)]
    feed = BarFeed.from_dicts(bars, symbol="S0")

    rot = RotationStrategy(["S0"], FixedUniverseSelector(),
                           make=lambda s: TrendWithStop(s, ma_period=3, stop_pct=0.05))
    eng = Engine(feed, rot, initial_cash=1_000_000.0)
    res = eng.run()

    sides = [f.side.value for f in res.fills]
    assert "buy" in sides and "sell" in sides, f"应有一买一卖（止损触发），实际 {sides}"
    assert res.fills[-1].side is OrderSide.SELL


# ---------------------------------------------------------------------------
# 选股器过滤
# ---------------------------------------------------------------------------
class _Recorder(FrameStrategy):
    """不交易，仅让包装器把「被选中的标的」记入 held_symbols()。"""

    def __init__(self, symbol: str):
        self.symbol = symbol

    def on_frame(self, frame, token, bus, host) -> list:
        return []


def _held_symbols(feed, syms, selector):
    rot = RotationStrategy(syms, selector, make=lambda s: _Recorder(s))
    Engine(feed, rot, initial_cash=1_000_000.0).run()
    return set(rot.held_symbols())


def test_trend_selector_only_picks_uptrend_names():
    """TrendPresenceSelector 只选收盘价在均线之上的标的。"""
    bars, syms = _trend_bars()
    feed = BarFeed.from_dicts(bars, symbol="UP")

    held = _held_symbols(feed, syms, TrendPresenceSelector(period=20, min_ratio=1.0))
    assert held == {"UP"}, f"应只交易趋势向上的 UP，实际 {held}"


def test_composite_selector_and_union():
    """CompositeSelector：and 取交集、or 取并集。"""
    bars, syms = _trend_bars()
    feed = BarFeed.from_dicts(bars, symbol="UP")
    uptrend = TrendPresenceSelector(period=20, min_ratio=1.0)
    liq = LiquiditySelector(min_amount=1e5)  # 成交额远超阈值 → 两器都放行两者

    and_result = _held_symbols(feed, syms, CompositeSelector([uptrend, liq], "and"))
    or_result = _held_symbols(feed, syms, CompositeSelector([uptrend, liq], "or"))

    assert and_result == {"UP"}, f"and 应只保留同时达标的 UP，实际 {and_result}"
    assert or_result == {"UP", "DOWN"}, f"or 应保留两者，实际 {or_result}"


# ---------------------------------------------------------------------------
# 凯利资金策略
# ---------------------------------------------------------------------------
def _fill(sym, side: OrderSide, size: float, price: float, frame: int = 0) -> FillEvent:
    return FillEvent(order_id=frame, side=side, symbol=sym, size=size,
                     price=price, frame_id=frame, timestamp=frame)


def test_kelly_fraction_computes_from_win_rate_and_payoff():
    pol = KellyBudgetPolicy(fraction=0.5, min_samples=5)
    # 3 笔盈利各 +200，2 笔亏损各 -100（同一标的依次往返）
    for _ in range(3):
        pol.record(_fill("S0", OrderSide.BUY, 100, 10.0))
        pol.record(_fill("S0", OrderSide.SELL, 100, 12.0))
    for _ in range(2):
        pol.record(_fill("S0", OrderSide.BUY, 100, 10.0))
        pol.record(_fill("S0", OrderSide.SELL, 100, 9.0))

    # p=0.6, b=200/100=2 → f* = 0.6 − 0.4/2 = 0.4；半凯利 → 0.2
    assert abs(pol.kelly_fraction() - 0.2) < 1e-9, pol.kelly_fraction()


def test_kelly_warmup_when_insufficient_samples():
    pol = KellyBudgetPolicy(fraction=0.5, min_samples=5, warmup_fraction=1.0)
    assert pol.kelly_fraction() == 1.0  # 无样本 → 退化为 warmup_fraction


def test_kelly_budget_never_negative_and_no_explode():
    """凯利资金策略：现金永不为负、仓位不爆炸、多标的轮动。"""
    syms = ["S0", "S1", "S2", "S3"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")
    rot = RotationStrategy(
        syms, FixedUniverseSelector(),
        make=lambda s: DualMAStrategy(s, short=5, long=20),
        budget_policy=KellyBudgetPolicy(fraction=0.5, min_samples=3),
    )
    eng = Engine(feed, rot, initial_cash=1_000_000.0,
                 settlement_cycle=1, cost_model=StandardCost(), risk=StandardRisk())
    res = eng.run()

    assert eng.ledger.cash >= -1.0, f"现金不应为负：{eng.ledger.cash:,.2f}"
    assert all(eq >= 0 for _k, eq in res.equity_curve), "权益不得为负"
    max_size = max((f.size for f in res.fills), default=0)
    assert max_size <= 1_000_000, f"单笔成交量 {max_size:,} 不合理（疑似重复下单）"
    assert len({f.symbol for f in res.fills}) > 1


def test_kelly_reduces_sizing_when_win_rate_falls():
    """记录到更多亏损样本后，凯利分数下降（降低下注比例）——符合「即时胜率」语义。"""
    pol = KellyBudgetPolicy(fraction=0.5, min_samples=5)
    for _ in range(3):
        pol.record(_fill("S0", OrderSide.BUY, 100, 10.0))
        pol.record(_fill("S0", OrderSide.SELL, 100, 12.0))  # 3 盈
    good = pol.kelly_fraction()
    for _ in range(4):
        pol.record(_fill("S0", OrderSide.BUY, 100, 10.0))
        pol.record(_fill("S0", OrderSide.SELL, 100, 8.0))  # 加 4 亏
    bad = pol.kelly_fraction()
    assert bad < good, f"胜率下降后凯利分数应降低：{bad:.3f} vs {good:.3f}"


# ---------------------------------------------------------------------------
# 边界 / 契约
# ---------------------------------------------------------------------------
def test_rotation_rejects_empty_universe():
    with pytest.raises(ValueError):
        RotationStrategy([], FixedUniverseSelector(), make=DualMAStrategy)


def test_rotation_wakeup_union_includes_fill_when_needed():
    """组合依赖 FILL 唤醒的策略时，default_wakeups 必须含 FILL。"""
    rot = RotationStrategy(["S0"], FixedUniverseSelector(), make=TrendWithStop)
    assert WakeupType.FRAME in rot.default_wakeups
    assert WakeupType.FILL in rot.default_wakeups
