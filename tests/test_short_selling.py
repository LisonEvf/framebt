"""Stage ③：A 股融券做空（SELL_SHORT）测试。

覆盖：
  - OrderSide.SELL_SHORT + Order.short 标记
  - Ledger 空头腿 / 空头未实现·已实现盈亏 / 现金 / 保证金
  - Rules：涨停不可开空、可借数量
  - Cost：borrow_fee（融券费）
  - Guard：空头保证金约束
  - 端到端：做空下跌标的获利、做多反向（做空上涨标的亏损）
"""

from __future__ import annotations

import math

import pytest

from framebt import (
    Bar,
    BarFeed,
    Engine,
    FrameStrategy,
    FillEvent,
    Ledger,
    MarketFrame,
    NoOpRisk,
    NoOpRules,
    Order,
    OrderSide,
    StandardAshareRules,
    StandardCost,
    StandardRisk,
)


def mkframe(fid, o, h, l, c, *, symbol="S", vol=1000.0, float_shares=0.0):
    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c,
                  volume=vol, float_shares=float_shares))
    return f


def fill(side, size, price, frame_id=0, symbol="S", cost=0.0, oid=1):
    return FillEvent(order_id=oid, side=side, symbol=symbol, size=size, price=price,
                     frame_id=frame_id, timestamp=frame_id, cost=cost)


# ===========================================================================
# OrderSide.SELL_SHORT + Order.short 标记
# ===========================================================================

def test_order_side_sell_short_exists():
    assert OrderSide.SELL_SHORT.value == "sell_short"


def test_order_short_marker_reflects_side():
    assert Order(side=OrderSide.SELL_SHORT, symbol="S", size=100).short is True
    assert Order(side=OrderSide.BUY, symbol="S", size=100).short is False
    assert Order(side=OrderSide.SELL, symbol="S", size=100).short is False


# ===========================================================================
# Ledger：空头腿 / 盈亏 / 现金 / 保证金
# ===========================================================================

def test_ledger_sell_short_opens_short_leg():
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0))
    pos = led.position("S")
    assert pos.net_size() == -100
    leg = pos.legs[0]
    assert leg.size == -100 and leg.short is True
    assert math.isclose(leg.entry_price, 10.0)
    # 开空增加现金（卖空所得）
    assert math.isclose(led.cash, 10000.0 + 1000.0)


def test_ledger_short_unrealized_pnl_sign():
    """空头：涨价亏损、跌价盈利。"""
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0))
    # 平价 → 权益 = 初始
    assert math.isclose(led.mark_to_market({"S": 10.0}), 10000.0)
    # 涨到 11 → 亏损 100
    assert math.isclose(led.mark_to_market({"S": 11.0}), 9900.0)
    # 跌到 9 → 盈利 100
    assert math.isclose(led.mark_to_market({"S": 9.0}), 10100.0)
    # Position.unrealized_pnl 直接验证
    pos = led.position("S")
    assert math.isclose(pos.unrealized_pnl(11.0), -100.0)
    assert math.isclose(pos.unrealized_pnl(9.0), 100.0)


def test_ledger_short_realized_pnl_on_cover():
    """平空（BUY 回补）按 FIFO 实现空头盈亏 = (入场 - 回补) × 数量。"""
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0, oid=1))
    led.apply_fill(fill(OrderSide.BUY, 100, 8.0, frame_id=1, oid=2))  # 回补
    pos = led.position("S")
    assert pos.net_size() == 0
    assert len(pos.legs) == 0
    # (10 - 8) × 100 = 200
    assert math.isclose(pos.realized_pnl, 200.0)
    # 现金：开空 +1000，回补 -800
    assert math.isclose(led.cash, 10000.0 + 1000.0 - 800.0)


def test_ledger_short_cover_partial_then_reopen():
    """部分回补后剩余空头可继续持有。"""
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 200, 10.0, frame_id=0, oid=1))
    led.apply_fill(fill(OrderSide.BUY, 100, 9.0, frame_id=1, oid=2))  # 部分回补
    pos = led.position("S")
    assert pos.net_size() == -100
    assert math.isclose(pos.realized_pnl, (10.0 - 9.0) * 100)  # 100
    leg = pos.legs[0]
    assert leg.short is True and leg.size == -100


def test_ledger_avg_price_positive_for_short():
    led = Ledger(cash=10000.0)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0, oid=1))
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 12.0, frame_id=1, oid=2))
    pos = led.position("S")
    assert pos.net_size() == -200
    # 空头均价为正：(10*100 + 12*100) / 200 = 11
    assert math.isclose(pos.avg_price(), 11.0)


def test_ledger_margin_requirement_counts_short_notional():
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    assert math.isclose(led.margin_requirement(), 0.0)  # 无空头
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0))
    # 空头名义市值 1000 × 0.5 = 500
    assert math.isclose(led.margin_requirement(), 500.0)
    # 用 mark 计值：11 × 100 × 0.5 = 550
    assert math.isclose(led.margin_requirement({"S": 11.0}), 550.0)


def test_ledger_margin_ratio_configurable():
    led = Ledger(cash=10000.0, margin_ratio=0.3)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0))
    assert math.isclose(led.margin_requirement(), 1000.0 * 0.3)


# ===========================================================================
# Rules：涨停不可开空 / 可借数量
# ===========================================================================

def test_rules_sell_short_blocked_at_limit_up():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))       # 昨收 10 → 涨停 11
    f = mkframe(1, 11.2, 11.5, 11.0, 11.2)            # 开盘 11.2 > 11（涨停）
    reason = rules.check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100),
                         Ledger(cash=1e6), f)
    assert reason is not None and "涨停" in reason and "开空" in reason


def test_rules_sell_short_allowed_below_limit_up():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))
    f = mkframe(1, 10.2, 10.5, 10.0, 10.2)            # 未涨停
    assert rules.check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100),
                       Ledger(cash=1e6), f) is None


def test_rules_sell_short_not_blocked_at_limit_down():
    """跌停可开空（低价做空更有利），不应被拒。"""
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))
    f = mkframe(1, 8.8, 9.0, 8.5, 8.8)                # 跌停附近
    assert rules.check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100),
                       Ledger(cash=1e6), f) is None


def test_rules_sell_short_rejected_exceeds_borrowable():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))
    f = mkframe(1, 10.2, 10.5, 10.0, 10.2, float_shares=500)
    # 流通股本仅 500，开空 600 → 不可借
    reason = rules.check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=600),
                         Ledger(cash=1e6), f)
    assert reason is not None and "可借" in reason


def test_rules_sell_short_within_borrowable_passes():
    rules = StandardAshareRules(limit=0.10)
    rules.observe(mkframe(0, 10, 11, 9, 10.0))
    f = mkframe(1, 10.2, 10.5, 10.0, 10.2, float_shares=500)
    assert rules.check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=400),
                       Ledger(cash=1e6), f) is None


def test_rules_noop_allows_borrow_unlimited():
    from framebt import NoOpRules
    rules = NoOpRules()
    f = mkframe(0, 10, 11, 9, 10.0)
    assert rules.available_to_borrow("S", f) == float("inf")
    assert rules.check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=10 ** 9),
                       Ledger(cash=1e6), f) is None


# ===========================================================================
# Cost：borrow_fee（融券费）
# ===========================================================================

def test_cost_borrow_fee_only_on_sell_short():
    cost = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="proportional", k=0.0,
                        borrow_fee=0.0005)
    frame = mkframe(0, 10, 11, 9, 10)
    sell_short = fill(OrderSide.SELL_SHORT, 1000, 10.0)  # 名义 10000
    buy = fill(OrderSide.BUY, 1000, 10.0)
    sell = fill(OrderSide.SELL, 1000, 10.0)
    assert math.isclose(cost.cost_for(sell_short, frame), 5.0)  # 10000 × 0.0005
    assert math.isclose(cost.cost_for(buy, frame), 0.0)
    assert math.isclose(cost.cost_for(sell, frame), 0.0)  # 融券费仅开空


def test_cost_no_borrow_fee_when_zero():
    cost = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="proportional", k=0.0, borrow_fee=0.0)
    assert cost.cost_for(fill(OrderSide.SELL_SHORT, 100, 10.0),
                         mkframe(0, 10, 11, 9, 10)) == 0.0


# ===========================================================================
# Guard：空头保证金约束
# ===========================================================================

def test_risk_sell_short_allows_within_margin():
    risk = StandardRisk(max_short_pct=1.0, max_order_pct=100.0)
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    f = mkframe(0, 10, 11, 9, 10)
    assert risk.pre_trade_check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100),
                                led, f) is None


def test_risk_sell_short_rejects_exceeding_margin():
    risk = StandardRisk(max_short_pct=0.01, max_order_pct=100.0)  # 仅 1% 空头额度
    led = Ledger(cash=10000.0, margin_ratio=0.5)
    led.apply_fill(fill(OrderSide.SELL_SHORT, 100, 10.0, frame_id=0, oid=1))  # 已有空头
    f = mkframe(1, 10, 11, 9, 10)
    # 再开空 → 保证金占用 1000 > 1% × 权益(≈10000=100)
    reason = risk.pre_trade_check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100),
                                  led, f)
    assert reason is not None and "保证金" in reason


def test_risk_sell_short_increases_cash():
    """开空所得现金计入可用资金（与买入的「扣现金」相反）。"""
    risk = StandardRisk(max_order_pct=100.0, max_short_pct=10.0)
    led = Ledger(cash=100.0, margin_ratio=0.5)
    f = mkframe(0, 10, 11, 9, 10)
    # 现金仅 100，但开空 100@10 → 收到 1000，可用充足（不是被现金 100 卡住）
    assert risk.pre_trade_check(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100),
                                led, f) is None


# ===========================================================================
# 端到端：做空下跌标的获利 / 做空上涨标的亏损
# ===========================================================================

def _short_then_cover_strategy(short_fid: int, cover_fid: int):
    class ShortThenCover(FrameStrategy):
        def __init__(self):
            self.shorted = False

        def on_frame(self, frame, token, bus, host):
            orders = []
            if not self.shorted and frame.frame_id == short_fid:
                orders.append(Order(side=OrderSide.SELL_SHORT, symbol="S", size=100))
                self.shorted = True
            pos = host.portfolio.position("S")
            if pos.net_size() < 0 and frame.frame_id == cover_fid:
                orders.append(Order(side=OrderSide.BUY, symbol="S", size=100))
            return orders

    return ShortThenCover()


def test_end_to_end_short_declining_stock_profits():
    """下跌序列：开空 9（帧 1 撮合）、回补 4（帧 6 撮合）→ 盈利 (9-4)×100=500。"""
    opens = [10, 9, 8, 7, 6, 5, 4]
    bars = [
        {"timestamp": i, "symbol": "S", "open": o, "high": o, "low": o,
         "close": o, "volume": 1e6}
        for i, o in enumerate(opens)
    ]
    eng = Engine(BarFeed.from_dicts(bars, symbol="S"),
                 _short_then_cover_strategy(short_fid=0, cover_fid=5),
                 initial_cash=100000.0,
                 cost_model=StandardCost(borrow_fee=0.0),
                 rules=NoOpRules(),
                 risk=NoOpRisk())
    res = eng.run()
    assert len(res.fills) == 2
    pos = eng.ledger.position("S")
    # 平仓后净持仓为 0
    assert math.isclose(pos.net_size(), 0.0)
    assert math.isclose(pos.realized_pnl, 500.0)  # (9 - 4) × 100


def test_end_to_end_short_rising_stock_losses():
    """上涨序列：开空 11（帧 1 撮合）、回补 16（帧 6 撮合）→ 亏损 (11-16)×100=-500。"""
    opens = [10, 11, 12, 13, 14, 15, 16]
    bars = [
        {"timestamp": i, "symbol": "S", "open": o, "high": o, "low": o,
         "close": o, "volume": 1e6}
        for i, o in enumerate(opens)
    ]
    eng = Engine(BarFeed.from_dicts(bars, symbol="S"),
                 _short_then_cover_strategy(short_fid=0, cover_fid=5),
                 initial_cash=100000.0,
                 cost_model=StandardCost(borrow_fee=0.0),
                 rules=NoOpRules(),
                 risk=NoOpRisk())
    res = eng.run()
    assert len(res.fills) == 2
    pos = eng.ledger.position("S")
    assert math.isclose(pos.realized_pnl, -500.0)  # (11 - 16) × 100，做空上涨标的亏损
