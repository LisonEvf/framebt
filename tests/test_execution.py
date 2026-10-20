"""Execution 测试（§5）：延迟管线 / 订单类型 / 条件单 / 撤单 / 组单。"""

from __future__ import annotations

import math

import pytest

from framebt import (
    Atomicity,
    Bar,
    Condition,
    MarketFrame,
    MatchingEngine,
    Order,
    OrderGroup,
    OrderSide,
    OrderState,
    OrderType,
    Trigger,
)


def mkframe(fid: int, o: float, h: float, l: float, c: float, *, symbol: str = "S",
            volume: float = 1000.0):
    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c, volume=volume))
    return f


# ---- §5.5 延迟管线（I2）----

def test_market_order_waits_for_delay():
    me = MatchingEngine(delay_frames=1)
    me.register(Order(side=OrderSide.BUY, symbol="S", size=100), 0)
    assert me.process(mkframe(0, 10, 11, 9, 10.5), 0) == []       # 帧 0：延迟未到
    fills = me.process(mkframe(1, 10.2, 11, 9, 10.5), 1)          # 帧 1：成交
    assert len(fills) == 1
    assert fills[0].price == 10.2 and fills[0].size == 100
    assert fills[0].frame_id == 1


def test_delay_frames_two():
    me = MatchingEngine(delay_frames=2)
    me.register(Order(side=OrderSide.BUY, symbol="S", size=10), 0)
    assert me.process(mkframe(1, 10, 11, 9, 10.5), 1) == []
    assert len(me.process(mkframe(2, 10, 11, 9, 10.5), 2)) == 1


def test_delay_must_be_positive():
    with pytest.raises(ValueError):
        MatchingEngine(delay_frames=0)


# ---- §5.1 订单类型 ----

def test_market_budget_sizes_by_price():
    me = MatchingEngine()
    me.register(Order(side=OrderSide.BUY, symbol="S", budget=1000.0), 0)
    fills = me.process(mkframe(1, 33.0, 34, 32, 33.0), 1)
    assert fills[0].size == 30  # floor(1000/33)


def test_limit_order_rests_until_price_reached():
    me = MatchingEngine()
    me.register(Order(side=OrderSide.BUY, symbol="S", size=100,
                      order_type=OrderType.LIMIT, price=9.5), 0)
    # 帧 1：开盘 10 > 9.5 → 不成交，resting
    assert me.process(mkframe(1, 10, 11, 9.6, 10.5), 1) == []
    # 帧 2：low=8.5 <= 9.5 -> 帧内价触及限价（开盘 9 亦 <= 9.5），成交于 9
    fills = me.process(mkframe(2, 9, 10, 8.5, 9.5), 2)
    assert len(fills) == 1 and fills[0].price == 9


def test_limit_order_fills_when_low_touches_but_open_does_not():
    """Phase 2 帧内 crossing：买限价单开盘未突破，但帧内 low 触及限价即成交。

    成交价必须是限价（9.5）或更优开盘价，不能取当日 low（9.2）——那等于
    事后知道最低点，属于帧内前视。
    """
    me = MatchingEngine()
    me.register(Order(side=OrderSide.BUY, symbol="S", size=100,
                       order_type=OrderType.LIMIT, price=9.5), 0)
    # 帧 1：open=10 > 9.5（开盘不突破），low=9.2 <= 9.5 -> 触及限价，成交于 9.5
    fills = me.process(mkframe(1, 10, 11, 9.2, 10.5), 1)
    assert len(fills) == 1 and fills[0].price == 9.5


def test_limit_sell_rests_until_price_reached():
    me = MatchingEngine()
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100,
                      order_type=OrderType.LIMIT, price=11.0), 0)
    assert me.process(mkframe(1, 10, 10.5, 9.5, 10), 1) == []
    fills = me.process(mkframe(2, 11, 11.5, 10.5, 11.2), 2)
    assert len(fills) == 1 and fills[0].price == 11


def test_limit_requires_price():
    with pytest.raises(ValueError):
        Order(side=OrderSide.BUY, symbol="S", size=10, order_type=OrderType.LIMIT)


def test_stop_order_triggers_on_breakdown():
    me = MatchingEngine()
    # 卖出止损：帧内 low 跌破 9.5 触发，成交于触发价（不是收盘价/当帧开盘价）
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100,
                      order_type=OrderType.STOP, price=9.5), 0)
    assert me.process(mkframe(1, 10, 10.5, 9.8, 10.0), 1) == []   # 未跌破
    fills = me.process(mkframe(2, 9.9, 10.0, 9.2, 9.3), 2)        # low=9.2 <= 9.5 触发
    assert len(fills) == 1 and fills[0].price == 9.5


def test_stop_gap_fills_at_open_not_trigger():
    """跳空越过止损价：只能按更差的开盘价成交，不能按触发价成交。"""
    me = MatchingEngine()
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100,
                      order_type=OrderType.STOP, price=9.5), 0)
    fills = me.process(mkframe(1, 8.8, 9.0, 8.5, 8.9), 1)
    assert len(fills) == 1 and fills[0].price == 8.8


def test_fok_and_ioc_fill_in_single_bar():
    for otype in (OrderType.FOK, OrderType.IOC):
        me = MatchingEngine()
        me.register(Order(side=OrderSide.BUY, symbol="S", size=50, order_type=otype), 0)
        fills = me.process(mkframe(1, 10, 11, 9, 10.5), 1)
        assert len(fills) == 1 and fills[0].size == 50


def test_ioc_partial_fills_then_cancels_remainder():
    """IOC：只能成交当帧可成交量，余量立即取消，不进入 resting。"""
    me = MatchingEngine()
    me.register(Order(side=OrderSide.BUY, symbol="S", size=500, order_type=OrderType.IOC), 0)
    fills = me.process(mkframe(1, 10, 11, 9, 10.5, volume=120.0), 1)
    assert len(fills) == 1 and fills[0].size == 120
    assert me.pending_count() == 0


def test_fok_cancels_when_volume_insufficient():
    """FOK：可成交量不足时整单取消，一分不成交。"""
    me = MatchingEngine()
    me.register(Order(side=OrderSide.BUY, symbol="S", size=500, order_type=OrderType.FOK), 0)
    fills = me.process(mkframe(1, 10, 11, 9, 10.5, volume=120.0), 1)
    assert fills == []
    assert me.pending_count() == 0


def test_trailing_stop_trails_the_high():
    me = MatchingEngine()
    # 保护多头：从入场价 10 起，自最高点回落 5% 触发卖出
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100,
                      order_type=OrderType.TRAILING_STOP, trail_pct=0.05,
                      price=10.0), 0)
    # 帧 1：涨到 12 → trigger = 12 × 0.95 = 11.4；low=11.8 未回落触发
    assert me.process(mkframe(1, 11, 12, 11.8, 12), 1) == []
    # 帧 2：最高 12.1 → trigger 11.495；low=11.2 触及 → 按触发价 11.495 成交
    fills = me.process(mkframe(2, 12, 12.1, 11.2, 11.3), 2)
    assert len(fills) == 1 and math.isclose(fills[0].price, 12.1 * 0.95)


def test_trailing_stop_requires_pct():
    with pytest.raises(ValueError):
        Order(side=OrderSide.SELL, symbol="S", size=1, order_type=OrderType.TRAILING_STOP)


# ---- §5.2 条件单 ----

def test_conditional_order_does_not_fill_before_trigger():
    """条件未满足时，即便订单类型是市价也不得成交。"""
    me = MatchingEngine()
    cond = Condition(trigger=Trigger.LAST_LE, level=9.5)
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100, condition=cond), 0)
    # 帧 1：close=10 > 9.5 → 条件未满足 → resting，不成交
    assert me.process(mkframe(1, 10, 11, 9, 10), 1) == []
    assert me.pending_count() == 1


def test_conditional_order_fills_after_trigger():
    """条件用收盘判定 → 只能在下一帧开盘成交，不能回头用触发帧开盘价。"""
    me = MatchingEngine()
    cond = Condition(trigger=Trigger.LAST_LE, level=9.5)
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100, condition=cond), 0)
    me.process(mkframe(1, 10, 11, 9, 10), 1)               # 未触发 → resting
    assert me.process(mkframe(2, 9.6, 10, 8.5, 9.0), 2) == []  # close 9.0 触发，但只 arm
    fills = me.process(mkframe(3, 9.4, 9.8, 9.2, 9.5), 3)  # 下一帧开盘成交
    assert len(fills) == 1
    assert fills[0].price == 9.4


def test_conditional_resting_does_not_infinite_loop():
    """regression：resting 重检不得因边遍历边追加而死循环。"""
    me = MatchingEngine()
    cond = Condition(trigger=Trigger.LAST_LE, level=1.0)  # 永不触发
    me.register(Order(side=OrderSide.SELL, symbol="S", size=100, condition=cond), 0)
    for fid in range(1, 50):
        fills = me.process(mkframe(fid, 10, 11, 9, 10), fid)
        assert fills == []
    assert me.pending_count() == 1


# ---- §5.4 撤单 ----

def test_cancel_in_flight_order():
    me = MatchingEngine(delay_frames=5)
    oid = me.register(Order(side=OrderSide.BUY, symbol="S", size=100), 0)
    assert me.has_in_flight()
    res = me.cancel(oid, 0)
    assert res.accepted
    assert not me.has_in_flight()


def test_cancel_unknown_order_rejected():
    me = MatchingEngine()
    res = me.cancel(999, 0)
    assert not res.accepted


def test_cancel_resting_limit():
    me = MatchingEngine()
    oid = me.register(Order(side=OrderSide.BUY, symbol="S", size=10,
                            order_type=OrderType.LIMIT, price=1.0), 0)
    me.process(mkframe(1, 10, 11, 9, 10), 1)  # 转 resting
    assert me.pending_count() == 1
    assert me.cancel(oid, 1).accepted
    assert me.pending_count() == 0


# ---- §5.3 组单原子性 ----

def test_order_group_all_or_none_resolves():
    grp = OrderGroup([
        Order(side=OrderSide.BUY, symbol="A", size=100),
        Order(side=OrderSide.SELL, symbol="B", size=100),
    ], atomicity="all_or_none")
    assert len(grp.resolve()) == 2


def test_order_group_best_effort():
    grp = OrderGroup([Order(side=OrderSide.BUY, symbol="A", size=1)],
                     atomicity="best_effort")
    assert grp.atomicity is Atomicity.BEST_EFFORT
    assert len(grp.resolve()) == 1


def test_group_all_or_none_registers_all_legs():
    me = MatchingEngine()
    grp = OrderGroup([
        Order(side=OrderSide.BUY, symbol="A", size=100),
        Order(side=OrderSide.SELL, symbol="B", size=100),
    ], atomicity="all_or_none")
    ids = me.register_group(grp, 0)
    assert len(ids) == 2 and grp.order_ids == ids
    assert me.pending_count() == 2


# ---- §5.1 OCO：一单成交其余自动撤销 ----

def test_oco_take_profit_fill_cancels_stop_loss():
    me = MatchingEngine()
    tp = Order(side=OrderSide.SELL, symbol="S", size=100,
               order_type=OrderType.LIMIT, price=11.0)
    sl = Order(side=OrderSide.SELL, symbol="S", size=100,
               order_type=OrderType.STOP, price=9.0)
    me.register_group(OrderGroup.oco([tp, sl]), 0)

    # 开盘 11.2 >= 止盈 11.0 → 止盈成交，止损应被自动撤销
    fills = me.process(mkframe(1, 11.2, 11.5, 11.0, 11.3), 1)
    assert len(fills) == 1
    assert fills[0].price == 11.2
    assert me.pending_count() == 0
    assert sl.state is OrderState.CANCELLED


def test_oco_stop_loss_trigger_cancels_take_profit():
    me = MatchingEngine()
    tp = Order(side=OrderSide.SELL, symbol="S", size=100,
               order_type=OrderType.LIMIT, price=11.0)
    sl = Order(side=OrderSide.SELL, symbol="S", size=100,
               order_type=OrderType.STOP, price=9.0)
    me.register_group(OrderGroup.oco([tp, sl]), 0)

    # 收盘 8.8 <= 止损 9.0 → 止损触发成交，止盈应被自动撤销
    fills = me.process(mkframe(1, 10.0, 10.2, 8.8, 8.8), 1)
    assert len(fills) == 1
    assert me.pending_count() == 0
    assert tp.state is OrderState.CANCELLED


def test_oco_neither_triggers_both_rest():
    me = MatchingEngine()
    tp = Order(side=OrderSide.SELL, symbol="S", size=100,
               order_type=OrderType.LIMIT, price=11.0)
    sl = Order(side=OrderSide.SELL, symbol="S", size=100,
               order_type=OrderType.STOP, price=9.0)
    me.register_group(OrderGroup.oco([tp, sl]), 0)
    # 价格在区间内：两者都不触发
    fills = me.process(mkframe(1, 10.0, 10.5, 9.5, 10.0), 1)
    assert fills == []
    assert me.pending_count() == 2


# ---- 订单受理校验 ----

def test_order_requires_size_or_budget():
    with pytest.raises(ValueError):
        Order(side=OrderSide.BUY, symbol="S")


def test_order_id_assigned_on_register():
    me = MatchingEngine()
    o = Order(side=OrderSide.BUY, symbol="S", size=1)
    oid = me.register(o, 0)
    assert o.id == oid and o.placement_frame == 0 and o.state is OrderState.PENDING


def test_missing_symbol_price_rejects():
    me = MatchingEngine()
    me.register(Order(side=OrderSide.BUY, symbol="MISSING", size=1), 0)
    empty = MarketFrame(1, 1, 2)
    assert me.process(empty, 1) == []  # 无价格数据 → 丢弃，不抛异常
