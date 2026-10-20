"""Phase 2 · 帧内子步进执行引擎测试。

覆盖：
  - OrderBook：价格-时间优先入队 / 出队 / 队列位置 / 深度 / 最优价 / 交叉判定。
  - 成交量分配模型（价格-时间优先，FIFO 成交量分摊）：
      * 同价 FIFO：成交量按到达顺序喂给队列。
      * 价格优先：买单高价先、卖价低价先。
      * 部分成交：成交量不足时部分成交，剩余量下帧以剩余量继续参与分配。
      * 队列位置：队列越靠后，成交量受限时成交越少。
  - 帧内价路径 crossing：买限价单 low 触及、卖限价单 high 触及即触发。
"""

from __future__ import annotations

import pytest

from framebt import (
    Bar,
    MarketFrame,
    MatchingEngine,
    Order,
    OrderId,
    OrderSide,
    OrderState,
    OrderType,
)
from framebt.execution.book import OrderBook


def mkframe(fid, o, h, l, c, *, symbol="S", volume=1000.0):
    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c,
                  volume=volume))
    return f


# ---------------------------------------------------------------------------
# OrderBook
# ---------------------------------------------------------------------------

def test_orderbook_add_remove_fifo():
    book = OrderBook("S")
    a = Order(OrderSide.BUY, "S", size=1, price=9.0, id=1)
    b = Order(OrderSide.BUY, "S", size=1, price=9.0, id=2)
    c = Order(OrderSide.BUY, "S", size=1, price=9.1, id=3)
    book.add(a)
    book.add(b)
    book.add(c)
    assert book.queue_position(a) == 0  # 队首最优先
    assert book.queue_position(b) == 1
    assert book.queue_position(c) == 0  # 不同价位独立队列
    assert book.depth(a) == 2
    assert book.remove(a.id)
    assert book.queue_position(b) == 0  # a 出队后 b 成为队首
    assert not book.remove(OrderId(99999))


def test_orderbook_best_bid_ask_and_cross():
    book = OrderBook("S")
    assert book.best_bid() is None and book.best_ask() is None
    book.add(Order(OrderSide.BUY, "S", size=1, price=9.0))
    book.add(Order(OrderSide.SELL, "S", size=1, price=9.5))
    book.add(Order(OrderSide.BUY, "S", size=1, price=9.2))
    book.add(Order(OrderSide.SELL, "S", size=1, price=9.4))
    assert book.best_bid() == 9.2  # 买价最高
    assert book.best_ask() == 9.4  # 卖价最低
    # 买单 9.2 未越对手（ask 9.4）；卖单 9.4 未越对手（bid 9.2）
    assert not book.is_crossed(Order(OrderSide.BUY, "S", size=1, price=9.2))
    assert not book.is_crossed(Order(OrderSide.SELL, "S", size=1, price=9.4))
    # 买单 9.5 高于卖一 9.4 → 交叉（市价化）
    assert book.is_crossed(Order(OrderSide.BUY, "S", size=1, price=9.5))


def test_orderbook_rejects_zero_price():
    book = OrderBook("S")
    with pytest.raises(ValueError):
        book.add(Order(OrderSide.BUY, "S", size=1, price=0.0))


# ---------------------------------------------------------------------------
# 帧内价路径 crossing（Phase 2）
# ---------------------------------------------------------------------------

def test_buy_limit_fills_when_low_touches():
    me = MatchingEngine()
    me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                     price=9.5), 0)
    # open=10 > 9.5（开盘不突破），但 low=9.2 <= 9.5 → 触及限价，成交于限价 9.5
    fills = me.process(mkframe(1, 10, 11, 9.2, 10.5), 1)
    assert len(fills) == 1 and fills[0].price == 9.5


def test_sell_limit_fills_when_high_touches():
    me = MatchingEngine()
    me.register(Order(OrderSide.SELL, "S", size=100, order_type=OrderType.LIMIT,
                     price=11.0), 0)
    # open=10.8 < 11.0（开盘不突破），但 high=11.2 >= 11.0 → 触及限价，成交于限价 11.0
    fills = me.process(mkframe(1, 10.8, 11.2, 10.5, 10.9), 1)
    assert len(fills) == 1 and fills[0].price == 11.0


def test_limit_rests_when_path_does_not_touch():
    me = MatchingEngine()
    me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                     price=9.5), 0)
    # low=9.6 > 9.5 → 帧内价未触及限价，resting
    assert me.process(mkframe(1, 10, 11, 9.6, 10.5), 1) == []
    assert me.pending_count() == 1


# ---------------------------------------------------------------------------
# 成交量分配模型：价格-时间优先 + FIFO + 部分成交
# ---------------------------------------------------------------------------

def test_fifo_volume_split_same_price():
    """同价两买单：成交量按到达顺序（FIFO）喂给队列。"""
    me = MatchingEngine()
    oid1 = me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                             price=9.0), 0)
    oid2 = me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                             price=9.0), 0)
    # 两单都触及（low=8.5 <= 9.0），成交量=150
    fills = me.process(mkframe(1, 9.5, 10, 8.5, 9.2, volume=150.0), 1)
    by_order = {f.order_id: f for f in fills}
    assert by_order[oid1].size == 100   # 队首全额
    assert by_order[oid2].size == 50    # 队列后部分


def test_price_priority_buy_higher_first():
    """买单价格优先：高价买单先被成交量喂到。"""
    me = MatchingEngine()
    oid_hi = me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                               price=9.5), 0)
    oid_lo = me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                               price=9.0), 0)
    # 成交量=100，只够填高价单
    fills = me.process(mkframe(1, 9.2, 10, 8.5, 9.1, volume=100.0), 1)
    by_order = {f.order_id: f for f in fills}
    assert by_order[oid_hi].size == 100
    assert oid_lo not in by_order          # 无成交量剩余
    assert me.pending_count() == 1         # 低价单继续 resting


def test_price_priority_sell_lower_first():
    """卖价价格优先：低价卖单先被成交量喂到。"""
    me = MatchingEngine()
    oid_lo = me.register(Order(OrderSide.SELL, "S", size=100, order_type=OrderType.LIMIT,
                               price=10.5), 0)
    oid_hi = me.register(Order(OrderSide.SELL, "S", size=100, order_type=OrderType.LIMIT,
                               price=11.0), 0)
    fills = me.process(mkframe(1, 10.8, 11.2, 10.2, 10.6, volume=100.0), 1)
    by_order = {f.order_id: f for f in fills}
    assert by_order[oid_lo].size == 100
    assert oid_hi not in by_order


def test_partial_fill_remainder_rests_next_bar():
    """部分成交：成交量不足时部分成交，剩余量下帧以剩余量继续参与分配。"""
    me = MatchingEngine()
    oid = me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                           price=9.0), 0)
    # 帧 1：成交量=60 -> 成交 60，剩余 40 resting
    fills = me.process(mkframe(1, 9.5, 10, 8.5, 9.2, volume=60.0), 1)
    assert len(fills) == 1 and fills[0].size == 60
    assert me.pending_count() == 1
    # 帧 2：成交量=30 -> 成交 30（剩余量），剩余 10 resting
    fills2 = me.process(mkframe(2, 9.0, 10, 8.0, 8.5, volume=30.0), 2)
    assert len(fills2) == 1 and fills2[0].size == 30
    assert me.pending_count() == 1
    # 帧 3：成交量=100 -> 成交剩余 10，全部落地
    fills3 = me.process(mkframe(3, 8.0, 9, 7.5, 7.8, volume=100.0), 3)
    assert len(fills3) == 1 and fills3[0].size == 10
    assert me.pending_count() == 0


def test_queue_position_deeper_fills_less():
    """队列位置：队列越靠后，成交量受限时成交越少（FIFO 成交量分摊）。"""
    me = MatchingEngine()
    ids = [me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                             price=9.0), 0) for _ in range(3)]
    # 成交量=250，三单各 100：第一单 100、第二单 100、第三单 50（剩余 50 继续 resting）
    fills = me.process(mkframe(1, 9.0, 10, 8.5, 8.8, volume=250.0), 1)
    by_order = {f.order_id: f for f in fills}
    assert by_order[ids[0]].size == 100
    assert by_order[ids[1]].size == 100
    assert by_order[ids[2]].size == 50
    assert me.pending_count() == 1  # 第三单部分成交，剩余 50 继续 resting


def test_zero_volume_never_fills():
    """成交量为零：限价单不成交，持续 resting。"""
    me = MatchingEngine()
    me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                     price=9.0), 0)
    for fid in range(1, 4):
        fills = me.process(mkframe(fid, 9.0, 10, 8.0, 8.5, volume=0.0), fid)
        assert fills == []
    assert me.pending_count() == 1


def test_reached_limit_fills_in_later_frame():
    """触发但本帧成交量被更早（价格-时间优先靠前）的单全额耗尽：目标单得 0，
    下帧以 nominal 规模重新参与分配并全额成交。"""
    me = MatchingEngine()
    # 先挂一单（价格-时间优先靠前），再挂目标单
    me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                     price=9.0), 0)
    oid = me.register(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                           price=9.0), 0)
    # 帧 1：成交量=100 恰好被前单一单全额消耗，目标单无量（resting）
    fills1 = me.process(mkframe(1, 9.0, 10, 8.0, 8.5, volume=100.0), 1)
    assert len(fills1) == 1 and fills1[0].size == 100
    assert me.pending_count() == 1   # 前单一单已全额成交，仅目标单 resting
    # 帧 2：成交量=100，目标单以 nominal 规模 100 全额成交
    fills2 = me.process(mkframe(2, 9.0, 10, 8.0, 8.5, volume=100.0), 2)
    by_order = {f.order_id: f for f in fills2}
    assert by_order[oid].size == 100
    assert me.pending_count() == 0


def test_budget_limit_fills_via_reached():
    """budget 限价单（size=0）经帧内 crossing 触发，按 fill_px 反算数量成交。"""
    me = MatchingEngine()
    oid = me.register(Order(OrderSide.BUY, "S", budget=900.0,
                           order_type=OrderType.LIMIT, price=9.5), 0)
    # open=9.2 <= 9.5 -> 触及限价，fill_px=9.2 -> size=floor(900/9.2)=97
    fills = me.process(mkframe(1, 9.2, 10, 9.0, 9.1, volume=100.0), 1)
    assert len(fills) == 1
    assert fills[0].size == 97
    assert abs(fills[0].price - 9.2) < 1e-9
    assert me.pending_count() == 0
