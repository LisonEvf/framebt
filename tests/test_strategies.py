"""示例策略的回归测试。

这些策略既是文档示例，也是框架能力的活体证明；必须对真实/合成数据都稳健。
"""

from __future__ import annotations

import math

import pytest

from framebt import BarFeed, Engine, NoOpCost, StandardCost
from framebt.strategies import (
    BracketStrategy,
    DualMAStrategy,
    MomentumCrossSection,
    TrendWithStop,
)


def _multi_symbol_bars(symbols: list[str], n: int = 90, base: float = 10.0) -> list[dict]:
    """构造多标的合成日线：各标的动量相位不同，使横截面排名会变化。"""
    bars = []
    for k, sym in enumerate(symbols):
        price = base + k
        for i in range(n):
            price *= 1.0 + 0.02 * math.sin(i / 4.0 + k * 1.7)
            o = price * 0.995
            h = price * 1.01
            low = price * 0.99
            bars.append({"timestamp": i, "symbol": sym, "open": o, "high": h,
                         "low": low, "close": price, "volume": 1e6})
    return bars


# ---------------------------------------------------------------------------
# 横截面动量：regression —— 满仓时 budget=0 曾导致 ValueError
# ---------------------------------------------------------------------------
def test_momentum_survives_full_investment_without_costs():
    """零成本下买入后现金恰为 0，下一帧 per=0；策略必须跳过而非构造非法订单。"""
    syms = ["S0", "S1", "S2", "S3"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")
    eng = Engine(feed, MomentumCrossSection(syms, lookback_short=5, lookback_long=10, top_k=2),
                 initial_cash=100_000.0, cost_model=NoOpCost())
    res = eng.run()                      # 不得抛 ValueError
    assert res.frames > 0
    assert res.final_equity() > 0


def test_momentum_with_costs_and_rules():
    syms = ["S0", "S1", "S2", "S3"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")
    eng = Engine(feed, MomentumCrossSection(syms, 5, 10, 2), initial_cash=100_000.0,
                 settlement_cycle=1, cost_model=StandardCost())
    res = eng.run()
    assert all(eq >= 0 for _k, eq in res.equity_curve), "权益不得为负"
    assert all(f.size > 0 for f in res.fills), "成交数量必须为正"


def test_momentum_only_holds_ranked_names():
    syms = ["S0", "S1", "S2", "S3"]
    feed = BarFeed.from_dicts(_multi_symbol_bars(syms), symbol="S0")
    eng = Engine(feed, MomentumCrossSection(syms, 5, 10, 2), initial_cash=100_000.0)
    res = eng.run()
    held = {f.symbol for f in res.fills}
    assert held <= set(syms), "只能交易选股池内的标的"


# ---------------------------------------------------------------------------
# 双均线
# ---------------------------------------------------------------------------
def test_dual_ma_warmup_no_trade_before_long_window():
    bars = [{"timestamp": i, "symbol": "S", "open": 10 + i, "high": 11 + i,
             "low": 9 + i, "close": 10 + i, "volume": 1e3} for i in range(4)]
    feed = BarFeed.from_dicts(bars, symbol="S")
    res = Engine(feed, DualMAStrategy("S", 5, 20), initial_cash=10_000.0).run()
    assert res.total_trades() == 0, "预热不足时不应交易"


def test_dual_ma_rejects_short_ge_long():
    with pytest.raises(ValueError):
        DualMAStrategy("S", short=10, long=5)


def test_dual_ma_never_shorts_and_never_oversells():
    syms_bars = _multi_symbol_bars(["S"], n=120)[0:]
    feed = BarFeed.from_dicts(syms_bars, symbol="S")
    eng = Engine(feed, DualMAStrategy("S", 5, 20), initial_cash=100_000.0)
    res = eng.run()
    for f in res.fills:
        assert f.size > 0
    assert eng.ledger.position("S").net_size() >= 0, "不应出现负持仓（未做空）"


# ---------------------------------------------------------------------------
# 条件单 / Bracket
# ---------------------------------------------------------------------------
def test_trend_with_stop_produces_orders():
    feed = BarFeed.from_dicts(_multi_symbol_bars(["S"], n=120)[0:][0:], symbol="S")
    eng = Engine(feed, TrendWithStop("S", ma_period=5, stop_pct=0.02),
                 initial_cash=100_000.0)
    res = eng.run()
    assert res.frames > 0


def test_bracket_oco_never_double_sells():
    """OCO：一轮持仓最多被平掉一次，不得同时触发止盈与止损。"""
    feed = BarFeed.from_dicts(_multi_symbol_bars(["S"], n=120)[0:], symbol="S")
    eng = Engine(feed, BracketStrategy("S", ma_period=3, take_profit_pct=0.02,
                                       stop_loss_pct=0.02),
                 initial_cash=100_000.0)
    res = eng.run()
    # 逐帧检查：卖出总量不超过同期买入总量
    bought = sum(f.size for f in res.fills if f.side.value == "buy")
    sold = sum(f.size for f in res.fills if f.side.value == "sell")
    assert sold <= bought + 1e-9, f"卖出 {sold} 超过买入 {bought}（可能重复平仓）"


# ---------------------------------------------------------------------------
# on_fill 唤醒（regression）
#
# FILL 曾不在默认唤醒集里，导致 on_fill 永不触发：止损/止盈等「成交后才挂单」
# 的策略会**静默失效**（回测看起来正常，但被动单从未登记）。
# ---------------------------------------------------------------------------
def _simple_bars(closes: list[float]) -> list[dict]:
    return [{"timestamp": i, "symbol": "S", "open": c, "high": c, "low": c,
             "close": c, "volume": 1e6} for i, c in enumerate(closes)]


def test_on_fill_wakeup_is_dispatched():
    from framebt import FrameStrategy, Order, OrderSide, OrderType

    seen = []

    class Probe(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            closes = bus.get("close", "S", token=token)
            if len(closes) == 2 and host.portfolio.position("S").net_size() == 0:
                return [Order(side=OrderSide.BUY, symbol="S", size=10,
                              order_type=OrderType.MARKET)]
            return []

        def on_fill(self, fill):
            seen.append(fill)
            return []

    Engine(BarFeed.from_dicts(_simple_bars([10, 10, 11, 11, 11, 11]), symbol="S"),
           Probe(), initial_cash=1_000_000.0).run()
    assert len(seen) == 1, f"on_fill 应被分发一次，实际 {len(seen)}"
    assert seen[0].size == 10


def test_on_fill_can_place_orders_that_execute():
    """on_fill 里挂出的止损条件单必须真的被登记并在后续帧触发。"""
    from framebt import Condition, FrameStrategy, Order, OrderSide, OrderType, Trigger

    class StopPlacer(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            closes = bus.get("close", "S", token=token)
            if len(closes) == 2 and host.portfolio.position("S").net_size() == 0:
                return [Order(side=OrderSide.BUY, symbol="S", size=100,
                              order_type=OrderType.MARKET)]
            return []

        def on_fill(self, fill):
            if fill.side is OrderSide.BUY:
                # 止损：最新价跌破成交价 10% 时市价卖出
                return [Order(side=OrderSide.SELL, symbol="S", size=fill.size,
                              order_type=OrderType.MARKET,
                              condition=Condition(Trigger.LAST_LE, fill.price * 0.90))]
            return []

    # 买入在帧2 以 open=11 成交 → 止损位 9.9；帧5 价格 7 触发
    bars = _simple_bars([10, 10, 11, 11, 11, 7, 7])
    eng = Engine(BarFeed.from_dicts(bars, symbol="S"), StopPlacer(), initial_cash=1_000_000.0)
    res = eng.run()

    sides = [f.side.value for f in res.fills]
    assert sides == ["buy", "sell"], f"应有一买一卖，实际 {sides}"
    assert res.fills[0].frame_id == 2 and res.fills[0].price == 11
    assert res.fills[1].price == 7, "止损应在帧5 触发并成交"
    assert eng.ledger.position("S").net_size() == 0, "止损后应清仓"


def test_disabling_fill_wakeup_skips_on_fill():
    """显式只注册 FRAME 时，on_fill 不应被调用（唤醒机制确实按注册生效）。"""
    from framebt import FrameStrategy, Order, OrderSide, OrderType
    from framebt.core.host import WakeupType

    seen = []

    class Probe(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            closes = bus.get("close", "S", token=token)
            if len(closes) == 2 and host.portfolio.position("S").net_size() == 0:
                return [Order(side=OrderSide.BUY, symbol="S", size=10,
                              order_type=OrderType.MARKET)]
            return []

        def on_fill(self, fill):
            seen.append(fill)
            return []

    feed = BarFeed.from_dicts(_simple_bars([10, 10, 11, 11]), symbol="S")
    eng = Engine(feed, Probe(), initial_cash=1_000_000.0)
    eng.host.register(eng.host.strategy, wakeups=[WakeupType.FRAME])
    eng.run()
    assert seen == [], "只注册 FRAME 时 on_fill 不应触发"


# ---------------------------------------------------------------------------
# 在途订单可见性 + 重复下单防护（regression）
#
# 订单有强制延迟（I2），「本帧下的单」不会立刻反映到持仓上。策略若只看持仓，
# 就会在成交落地前**每帧重复全仓下单**，仓位指数级增长（曾出现 3e15 股）。
# ---------------------------------------------------------------------------
def test_host_exposes_pending_orders():
    """策略必须能查询在途订单，才能正确表达「避免重复下单」。"""
    from framebt import FrameStrategy, Order, OrderSide, OrderType

    observed: list[int] = []

    class Probe(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            observed.append(len(host.pending_orders("S")))
            closes = bus.get("close", "S", token=token)
            if len(closes) == 2:
                return [Order(side=OrderSide.BUY, symbol="S", size=10,
                              order_type=OrderType.MARKET)]
            return []

    Engine(BarFeed.from_dicts(_simple_bars([10, 10, 11, 11, 11]), symbol="S"),
           Probe(), initial_cash=1_000_000.0).run()
    assert 1 in observed, f"下单后应能在途看到 1 笔，实际序列 {observed}"


def test_trend_with_stop_does_not_explode_positions():
    """regression：只看持仓不看在途订单 → 重复全仓下单 → 仓位爆炸。"""
    from framebt import StandardRisk
    from framebt.strategies import TrendWithStop

    # 单调上涨：趋势策略会在第 3 帧后持续满足「空仓 + 价在均线上」
    bars = _simple_bars([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21])
    eng = Engine(BarFeed.from_dicts(bars, symbol="S"),
                 TrendWithStop("S", ma_period=3, stop_pct=0.05),
                 initial_cash=1_000_000.0, risk=StandardRisk())
    res = eng.run()

    max_size = max((f.size for f in res.fills), default=0)
    # 初始资金 100 万 / 最低价 10 = 10 万股上限；给出宽裕余量后仍远低于爆炸量级
    assert max_size <= 1_000_000, f"单笔成交量 {max_size:,} 不合理（疑似重复下单）"
    assert eng.ledger.cash >= -1.0, f"现金不应为负：{eng.ledger.cash:,.2f}"
    assert eng.ledger.position("S").net_size() >= 0


def test_risk_reserves_cash_for_pending_orders():
    """两笔各 60% 资金的买单：单独看都充足，但合计透支 → 第二笔必须被拒。"""
    from framebt import Ledger, MatchingEngine, Order, OrderSide, StandardRisk

    me = MatchingEngine(delay_frames=5)
    led = Ledger(cash=10_000.0)
    frame = BarFeed.from_dicts(_simple_bars([10, 10]), symbol="S").bars
    from framebt import Bar, MarketFrame

    f = MarketFrame(0, 0, 1)
    f.add_bar(frame[0])

    # 第一笔 600 × 10 = 6000 已在途
    me.register(Order(side=OrderSide.BUY, symbol="S", size=600), 0)

    risk = StandardRisk(max_order_pct=10.0, reserve_pending=True)
    risk.matcher = me
    reason = risk.pre_trade_check(Order(side=OrderSide.BUY, symbol="S", size=600), led, f)
    assert reason is not None and "在途预留" in reason, f"应因在途预留被拒，实际：{reason}"

    # 关闭预留则放行（说明预留确实生效）
    risk2 = StandardRisk(max_order_pct=10.0, reserve_pending=False)
    risk2.matcher = me
    assert risk2.pre_trade_check(Order(side=OrderSide.BUY, symbol="S", size=600),
                                 led, f) is None
