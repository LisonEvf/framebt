"""Ledger / Rules / Cost / Guard 测试（§6 / §7 / §8 / §9）。"""

from __future__ import annotations

import math

import pytest

from framebt import (
    Bar,
    CorporateAction,
    FillEvent,
    Ledger,
    MarketFrame,
    Order,
    OrderSide,
    StandardAshareRules,
    StandardCost,
    StandardRisk,
)


def mkframe(fid: int, o: float, h: float, l: float, c: float, *, symbol: str = "S", vol: float = 1000.0):
    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c, volume=vol))
    return f


def fill(side, size, price, frame_id=0, symbol="S", cost=0.0, oid=1):
    return FillEvent(order_id=oid, side=side, symbol=symbol, size=size, price=price,
                     frame_id=frame_id, timestamp=frame_id, cost=cost)


# ===========================================================================
# Ledger（§6）
# ===========================================================================

def test_ledger_buy_updates_cash_and_position():
    led = Ledger(cash=10000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0))
    assert math.isclose(led.cash, 9000.0)
    assert led.position("S").net_size() == 100
    assert math.isclose(led.position("S").avg_price(), 10.0)


def test_ledger_multi_leg_weighted_average():
    """金字塔加仓：多腿持仓的加权均价。"""
    led = Ledger(cash=100000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=0))
    led.apply_fill(fill(OrderSide.BUY, 100, 12.0, frame_id=1, oid=2))
    pos = led.position("S")
    assert len(pos.legs) == 2
    assert pos.net_size() == 200
    assert math.isclose(pos.avg_price(), 11.0)


def test_ledger_realized_pnl_fifo():
    led = Ledger(cash=100000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=0))
    led.apply_fill(fill(OrderSide.BUY, 100, 12.0, frame_id=1, oid=2))
    led.apply_fill(fill(OrderSide.SELL, 100, 15.0, frame_id=2, oid=3))
    pos = led.position("S")
    # FIFO：先平掉 10.0 的腿 → (15-10)*100 = 500
    assert math.isclose(pos.realized_pnl, 500.0)
    assert pos.net_size() == 100


def test_ledger_mark_to_market():
    led = Ledger(cash=10000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0))
    # 现金 9000 + 100 × 11 = 10100
    assert math.isclose(led.mark_to_market({"S": 11.0}), 10100.0)


def test_ledger_mark_to_market_no_double_count():
    """regression：MTM 不得同时累加未实现盈亏与市价计值。"""
    led = Ledger(cash=10000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0))
    eq = led.mark_to_market({"S": 10.0})
    assert math.isclose(eq, 10000.0), "平价时权益应等于初始权益"


def test_ledger_exposure_and_empty():
    led = Ledger(cash=1000.0)
    assert led.exposure() == {}
    led.apply_fill(fill(OrderSide.BUY, 50, 10.0))
    assert led.exposure() == {"S": 50}
    # 清仓后暴露为空
    led.apply_fill(fill(OrderSide.SELL, 50, 11.0, frame_id=1, oid=2))
    assert led.exposure() == {}


def test_ledger_t_plus_1_settlement():
    led = Ledger(cash=10000.0, settlement_cycle=1)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=5))
    pos = led.position("S")
    assert pos.available_to_sell(5) == 0     # 买入当日不可卖
    assert pos.available_to_sell(6) == 100   # T+1 可卖


def test_ledger_t_plus_0_settlement():
    led = Ledger(cash=10000.0, settlement_cycle=0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=5))
    assert led.position("S").available_to_sell(5) == 100


def test_ledger_corporate_action_split():
    led = Ledger(cash=10000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0))
    # 1 拆 2
    led.apply_corporate_action(CorporateAction(symbol="S", kind="split", date=1, ratio=2.0))
    pos = led.position("S")
    assert pos.net_size() == 200
    assert math.isclose(pos.avg_price(), 5.0)


def test_ledger_corporate_action_dividend():
    led = Ledger(cash=1000.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0))  # cash 0
    led.apply_corporate_action(
        CorporateAction(symbol="S", kind="dividend", date=1, cash_per_share=0.5)
    )
    assert math.isclose(led.cash, 50.0)  # 100 股 × 0.5


def test_ledger_delist_cash_settlement():
    led = Ledger(cash=0.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0))
    led.apply_corporate_action(CorporateAction(symbol="S", kind="delist", date=1, ratio=8.0))
    pos = led.position("S")
    assert pos.net_size() == 0
    assert math.isclose(pos.realized_pnl, -200.0)  # (8-10)*100


# ===========================================================================
# Rules（§7）
# ===========================================================================

def test_rules_price_limit_up_blocks_buy():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))       # 昨收 10 → 涨停 11
    f = mkframe(1, 11.5, 12, 11.4, 11.5)              # 开盘 11.5 > 11
    reason = rules.check(Order(side=OrderSide.BUY, symbol="S", size=100), Ledger(cash=1e6), f)
    assert reason is not None and "涨停" in reason


def test_rules_price_limit_down_blocks_sell():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))       # 跌停 9
    led = Ledger(cash=0.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=0))
    f = mkframe(1, 8.5, 8.6, 8.4, 8.5)                # 开盘 8.5 < 9
    reason = rules.check(Order(side=OrderSide.SELL, symbol="S", size=100), led, f)
    assert reason is not None and "跌停" in reason


def test_rules_t_plus_1_blocks_same_day_sell():
    rules = StandardAshareRules(settlement_cycle=1)
    led = Ledger(cash=100000.0, settlement_cycle=1)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=3))
    f = mkframe(3, 10, 11, 9, 10.0)                   # 同一帧
    reason = rules.check(Order(side=OrderSide.SELL, symbol="S", size=100), led, f)
    assert reason is not None and "T+1" in reason


def test_rules_t_plus_1_allows_next_day_sell():
    rules = StandardAshareRules(settlement_cycle=1)
    led = Ledger(cash=100000.0, settlement_cycle=1)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=3))
    f = mkframe(4, 10, 11, 9, 10.0)                   # 次日
    assert rules.check(Order(side=OrderSide.SELL, symbol="S", size=100), led, f) is None


def test_rules_suspension():
    rules = StandardAshareRules(suspended={"S"})
    reason = rules.check(Order(side=OrderSide.BUY, symbol="S", size=1),
                         Ledger(cash=1e6), mkframe(0, 10, 11, 9, 10))
    assert reason is not None and "停牌" in reason


def test_rules_normal_order_passes():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))
    f = mkframe(1, 10.2, 10.5, 10.0, 10.2)
    assert rules.check(Order(side=OrderSide.BUY, symbol="S", size=100),
                       Ledger(cash=1e6), f) is None


def test_rules_missing_symbol_in_frame_rejects_not_crashes():
    """regression：多标的回测中各标的上市日不同，某些帧天然缺标的。

    早期实现直接 ``frame.open(symbol)``，会抛 KeyError 打断整场回测。
    无法定价的订单必须**返回拒绝原因**而非异常。
    """
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0, symbol="S"))
    # 本帧只有 OTHER，没有 S
    f = mkframe(1, 10, 11, 9, 10.0, symbol="OTHER")
    reason = rules.check(Order(side=OrderSide.BUY, symbol="S", size=100),
                         Ledger(cash=1e6), f)
    assert reason is not None and "无行情数据" in reason

    # 卖出方向同样不得抛异常
    led = Ledger(cash=0.0)
    led.apply_fill(fill(OrderSide.BUY, 100, 10.0, frame_id=0))
    reason2 = rules.check(Order(side=OrderSide.SELL, symbol="S", size=100), led, f)
    assert reason2 is not None and "无行情数据" in reason2


def test_engine_survives_incomplete_frames():
    """端到端：多标的各标的起始日不同（帧内标的不全）时引擎不得崩溃。"""
    from framebt import BarFeed, Engine, FrameStrategy, OrderSide as OS

    class AlwaysBuy(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            return [Order(side=OS.BUY, symbol=s, size=100)
                    for s in ("A", "B") if frame.has_bar(s)]

    bars = [
        # A 从 t=0 开始；B 从 t=2 才开始 → 前两帧只有 A
        {"timestamp": 0, "symbol": "A", "open": 10, "high": 10, "low": 10,
         "close": 10, "volume": 1e6},
        {"timestamp": 1, "symbol": "A", "open": 10, "high": 10, "low": 10,
         "close": 10, "volume": 1e6},
        {"timestamp": 2, "symbol": "A", "open": 10, "high": 10, "low": 10,
         "close": 10, "volume": 1e6},
        {"timestamp": 2, "symbol": "B", "open": 20, "high": 20, "low": 20,
         "close": 20, "volume": 1e6},
    ]
    res = Engine(BarFeed.from_dicts(bars, symbol="A"), AlwaysBuy(),
                 initial_cash=1_000_000.0,
                 rules=StandardAshareRules(limit=0.10)).run()
    assert res.frames >= 3
    assert {f.symbol for f in res.fills} == {"A", "B"}


# ===========================================================================
# Cost（§8）
# ===========================================================================

def test_cost_commission_min_applies():
    cost = StandardCost(commission=0.0003, min_commission=5.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="proportional", k=0.0)
    f = fill(OrderSide.BUY, 1000, 10.0)  # notional 10000 → 佣金 max(3,5)=5
    assert math.isclose(cost.cost_for(f, mkframe(0, 10, 11, 9, 10)), 5.0)


def test_cost_stamp_tax_only_on_sell():
    cost = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.001,
                        transfer_fee=0.0, slippage="proportional", k=0.0)
    frame = mkframe(0, 10, 11, 9, 10)
    buy = fill(OrderSide.BUY, 1000, 10.0)
    sell = fill(OrderSide.SELL, 1000, 10.0)
    assert math.isclose(cost.cost_for(buy, frame), 0.0)
    assert math.isclose(cost.cost_for(sell, frame), 10.0)  # 10000 × 0.001


def test_cost_proportional_slippage():
    cost = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="proportional", k=0.001)
    f = fill(OrderSide.BUY, 1000, 10.0)  # notional 10000
    assert math.isclose(cost.cost_for(f, mkframe(0, 10, 11, 9, 10)), 10.0)


def test_cost_sqrt_impact():
    cost = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="sqrt", impact=0.1, adv=1000)
    f = fill(OrderSide.BUY, 100, 10.0)  # notional 1000, size/adv = 0.1
    expected = 1000 * 0.1 * math.sqrt(0.1)
    assert math.isclose(cost.cost_for(f, mkframe(0, 10, 11, 9, 10)), expected)


def test_cost_volume_based_slippage():
    cost = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="volume_based", k=0.001)
    f = fill(OrderSide.BUY, 100, 10.0)  # notional 1000, size/vol = 100/1000
    expected = 1000 * 0.001 * 0.1
    assert math.isclose(cost.cost_for(f, mkframe(0, 10, 11, 9, 10, vol=1000)), expected)


def test_cost_unknown_slippage_raises():
    cost = StandardCost(slippage="nonsense")
    with pytest.raises(ValueError):
        cost.cost_for(fill(OrderSide.BUY, 1, 1.0), mkframe(0, 1, 1, 1, 1))


# ===========================================================================
# Guard（§9）
# ===========================================================================

def test_risk_rejects_oversized_order():
    risk = StandardRisk(max_order_pct=0.5)
    led = Ledger(cash=10000.0)
    f = mkframe(0, 10, 11, 9, 10)
    # 6000 > 0.5 × 10000 = 5000
    reason = risk.pre_trade_check(Order(side=OrderSide.BUY, symbol="S", size=600), led, f)
    assert reason is not None and "单笔订单" in reason


def test_risk_rejects_insufficient_cash():
    risk = StandardRisk(max_order_pct=10.0)
    led = Ledger(cash=100.0)
    f = mkframe(0, 10, 11, 9, 10)
    reason = risk.pre_trade_check(Order(side=OrderSide.BUY, symbol="S", size=100), led, f)
    assert reason is not None and "资金不足" in reason


def test_risk_allows_within_limits():
    risk = StandardRisk(max_order_pct=1.0, max_position_pct=1.0)
    led = Ledger(cash=10000.0)
    f = mkframe(0, 10, 11, 9, 10)
    assert risk.pre_trade_check(Order(side=OrderSide.BUY, symbol="S", size=100), led, f) is None


def test_incremental_risk_metrics():
    from framebt import IncrementalRisk

    risk = IncrementalRisk([(0, 100.0), (1, 110.0), (2, 99.0), (3, 105.0)])
    assert math.isclose(risk.max_drawdown, (110 - 99) / 110)
    assert risk.var(0.95) >= 0.0
    risk.update(4, 120.0)
    assert risk.exposure() == 120.0
