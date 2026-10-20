"""Phase 6 · ① 实盘复用地基测试（LiveFeed / LiveConnector / Transport）。

核心命题：``LiveFeed`` 是 ``MarketFeed`` 的实盘实现，``LiveConnector`` 是
``ExecutionBackend`` 的实盘实现；二者遵守与回测完全相同的契约，故策略与 Engine
调度逻辑在回测 / 实盘之间零改动切换。

本文件验证：
  - LiveFeed 是 MarketFeed；增量 push 维持 (timestamp, sequence_id) 升序。
  - LiveConnector 是 ExecutionBackend；submit/cancel/process 生命周期。
  - LiveConnector + SimulatedTransport 的成交与 MatchingEngine 逐笔一致。
  - Engine 可注入 LiveConnector（模拟实盘），与 MatchingEngine 对同一场景逐笔一致。
  - LiveConnector OCO 联动撤销。
"""

from __future__ import annotations

from framebt import (
    Bar,
    ExecutionBackend,
    Engine,
    FrameStrategy,
    LiveConnector,
    LiveFeed,
    MarketFeed,
    MatchingEngine,
    Order,
    OrderId,
    OrderSide,
    OrderState,
    OrderType,
    SimulatedTransport,
    Transport,
)
from framebt.core.events import FillEvent
from framebt.execution.engine import CancelResult


# ---------------------------------------------------------------------------
# LiveFeed —— MarketFeed 的实盘实现
# ---------------------------------------------------------------------------

def test_livefeed_is_market_feed():
    assert issubclass(LiveFeed, MarketFeed)
    assert isinstance(LiveFeed(), MarketFeed)


def test_livefeed_incremental_push_keeps_sorted():
    feed = LiveFeed()
    # 乱序喂入（模拟实时流乱序到达）
    feed.push(Bar(timestamp=2, symbol="S", open=10, high=11, low=9, close=10, volume=1.0))
    feed.push(Bar(timestamp=0, symbol="S", open=10, high=11, low=9, close=10, volume=1.0))
    feed.push(Bar(timestamp=1, symbol="S", open=10, high=11, low=9, close=10, volume=1.0))
    # 按 (timestamp, sequence_id) 升序暴露（I4 全序）
    pairs = [(b.timestamp, b.sequence_id) for b in feed.bars]
    assert pairs == sorted(pairs)
    assert [p[0] for p in pairs] == [0, 1, 2]


def test_livefeed_push_preserves_explicit_sequence_id():
    feed = LiveFeed()
    feed.push(Bar(timestamp=0, symbol="S", open=1, high=2, low=1, close=1,
                  volume=1.0, sequence_id=5))
    assert feed.bars[0].sequence_id == 5


# ---------------------------------------------------------------------------
# LiveConnector —— ExecutionBackend 的实盘实现
# ---------------------------------------------------------------------------

def test_liveconnector_is_execution_backend():
    assert issubclass(LiveConnector, ExecutionBackend)
    assert isinstance(LiveConnector(SimulatedTransport()), ExecutionBackend)


def _probe():
    class _Probe(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            return []
    return _Probe()


def _bars(n: int, symbol: str = "S", price: float = 10.0):
    return [
        {"timestamp": i, "symbol": symbol, "open": price, "high": price + 1,
         "low": price - 1, "close": price, "volume": 1e6}
        for i in range(n)
    ]


def _mkframe(fid: int, o: float, h: float, l: float, c: float, *, symbol: str = "S"):
    from framebt import MarketFrame

    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c, volume=1000.0))
    return f


def test_liveconnector_submit_cancel_process():
    transport = SimulatedTransport(delay=1)
    conn = LiveConnector(transport, delay=1)
    oid = conn.submit(Order(OrderSide.BUY, "S", size=100), 1)  # 帧 1 下单
    assert oid is not None
    assert conn.has_in_flight()
    # 延迟 1 帧：帧 1 不可撮合（1-1=0 < 1）
    assert conn.process(_mkframe(1, 10, 11, 9, 10), 1) == []
    assert conn.has_in_flight()
    # 帧 2 撮合（市价按开盘 10）
    fills = conn.process(_mkframe(2, 10, 11, 9, 10), 2)
    assert len(fills) == 1 and fills[0].size == 100 and fills[0].price == 10.0
    assert not conn.has_in_flight()
    # 撤单：未成交的挂单可撤
    oid2 = conn.submit(Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT,
                             price=8.0), 3)  # 低于市价，resting
    assert conn.cancel(oid2, 4).accepted is True
    assert not conn.has_in_flight()


def test_liveconnector_find_order_and_pending_from_backend():
    # pending_count / find_order 由 ExecutionBackend 给出跨后端通用默认实现
    assert LiveConnector.find_order is ExecutionBackend.find_order
    assert LiveConnector.pending_count is ExecutionBackend.pending_count
    conn = LiveConnector(SimulatedTransport())
    conn.submit(Order(OrderSide.BUY, "S", size=1), 0)
    assert conn.pending_count() == 1


def test_liveconnector_oco_group_cancels_sibling():
    from framebt import OrderGroup, LinkMode

    conn = LiveConnector(SimulatedTransport(), delay=1)
    # 买限价 9.0（低于市价， resting）、卖限价 11.0（高于市价， resting）
    buy = Order(OrderSide.BUY, "S", size=100, order_type=OrderType.LIMIT, price=9.0)
    sell = Order(OrderSide.SELL, "S", size=100, order_type=OrderType.LIMIT, price=11.0)
    grp = OrderGroup([buy, sell], link=LinkMode.OCO)
    conn.register_group(grp, 0)  # 帧 0 下单 → 帧 1 可撮合
    assert len(grp.order_ids) == 2
    # 帧 1：low=8.5 触及买限价 9.0（买成交于 8.5）；high=10.5 未触及卖限价 11.0
    fills = conn.process(_mkframe(1, 10, 10.5, 8.5, 10), 1)
    assert len(fills) == 1 and fills[0].price == 8.5
    assert buy.state is OrderState.FILLED
    # OCO：买成交 → 卖单（兄弟）被撤销
    assert sell.state is OrderState.CANCELLED
    assert not conn.has_in_flight()


# ---------------------------------------------------------------------------
# LiveConnector 与 MatchingEngine 对模拟场景逐笔一致
# ---------------------------------------------------------------------------

def _run_with_backend(make_backend, bars, symbol="S"):
    from framebt import BarFeed

    feed = BarFeed.from_dicts(bars, symbol=symbol)
    eng = Engine(feed, _probe(), executor=make_backend(), initial_cash=1_000_000.0)
    # 触发一次撮合：帧 1 挂买单，帧 2 成交
    class _BuyOnFrame1(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            if frame.frame_id == 1:
                return [Order(OrderSide.BUY, symbol, size=100)]
            return []
    eng.host.register(_BuyOnFrame1())
    return eng.run()


def test_liveconnector_matches_matching_engine_market():
    bars = _bars(10)
    r_me = _run_with_backend(lambda: MatchingEngine(delay_frames=1), bars)
    r_live = _run_with_backend(lambda: LiveConnector(SimulatedTransport(), delay=1), bars)
    fills_me = [(f.size, f.price, f.frame_id) for f in r_me.fills]
    fills_live = [(f.size, f.price, f.frame_id) for f in r_live.fills]
    # 帧1下单 → 延迟1帧 → 帧2按开盘价10成交
    assert fills_me == fills_live == [(100.0, 10.0, 2)]


def test_liveconnector_matches_matching_engine_limit():
    bars = _bars(10)
    def strategy(make_backend):
        class _S(FrameStrategy):
            def on_frame(self, frame, token, bus, host):
                if frame.frame_id == 1:
                    return [Order(OrderSide.BUY, "S", size=100,
                                  order_type=OrderType.LIMIT, price=9.0)]
                return []
        return _S()

    from framebt import Engine as _Engine, BarFeed

    def run(make_backend):
        feed = BarFeed.from_dicts(bars, symbol="S")
        eng = _Engine(feed, strategy(make_backend), executor=make_backend(),
                      initial_cash=1_000_000.0)
        return eng.run()

    r_me = run(lambda: MatchingEngine(delay_frames=1))
    r_live = run(lambda: LiveConnector(SimulatedTransport(), delay=1))
    fills_me = [(f.size, f.price, f.frame_id) for f in r_me.fills]
    fills_live = [(f.size, f.price, f.frame_id) for f in r_live.fills]
    # 限价买单 price=9.0：帧内 low=9 触及 → 成交于 9.0（帧 2）
    assert fills_me == fills_live == [(100.0, 9.0, 2)]


def test_delay_validation_skipped_for_live_connector():
    # 自定义 executor（实盘）自管延迟语义，delay_frames 校验跳过
    from framebt import BarFeed

    conn = LiveConnector(SimulatedTransport(), delay=1)
    feed = BarFeed.from_dicts(_bars(10))
    eng = Engine(feed, _probe(), executor=conn)
    assert eng.executor is conn
