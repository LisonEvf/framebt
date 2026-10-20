"""Phase 6 · 实盘复用地基测试（ExecutionBackend / MarketFeed 边界）。

核心命题：策略与 Engine 只依赖两个抽象接口 ——
  - ``MarketFeed``（数据源：BarFeed 回测 / LiveFeed 实盘）
  - ``ExecutionBackend``（撮合后端：MatchingEngine 回测 / LiveConnector 实盘）
换这两个接口的实现，策略与引擎调度逻辑零改动。

本文件验证：
  - MatchingEngine 是 ExecutionBackend 的子类；BarFeed 是 MarketFeed 的子类。
  - submit 为规范入口，register 是其兼容别名。
  - find_order / pending_count 由 ABC 给出跨后端通用默认实现（非子类重复）。
  - Engine 可注入自定义 executor（模拟实盘 LiveConnector），且与 MatchingEngine
    对同一场景产出逐笔一致的成交 —— 即引擎与策略不依赖具体后端。
"""

from __future__ import annotations

from framebt import (
    Bar,
    BarFeed,
    ExecutionBackend,
    Engine,
    FrameStrategy,
    MarketFeed,
    MatchingEngine,
    Order,
    OrderId,
    OrderSide,
    OrderState,
    OrderType,
)
from framebt.core.events import FillEvent
from framebt.execution.engine import CancelResult


class _Probe(FrameStrategy):
    """无操作策略：每帧不下单，仅用于驱动引擎。"""

    def on_frame(self, frame, token, bus, host):
        return []


def mkframe(fid: int, o: float, h: float, l: float, c: float, *, symbol: str = "S"):
    from framebt import MarketFrame

    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c, volume=1000.0))
    return f


def _bars(n: int, symbol: str = "S", price: float = 10.0):
    return [
        {"timestamp": i, "symbol": symbol, "open": price, "high": price + 1,
         "low": price - 1, "close": price, "volume": 1e6}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 边界：子类关系
# ---------------------------------------------------------------------------

def test_matching_engine_is_execution_backend():
    assert issubclass(MatchingEngine, ExecutionBackend)
    assert isinstance(MatchingEngine(), ExecutionBackend)


def test_barfeed_is_market_feed():
    assert issubclass(BarFeed, MarketFeed)
    assert isinstance(BarFeed.from_dicts(_bars(3)), MarketFeed)


# ---------------------------------------------------------------------------
# submit 为规范入口，register 为兼容别名
# ---------------------------------------------------------------------------

def test_submit_and_register_are_aliases():
    me = MatchingEngine(delay_frames=1)
    o1 = Order(side=OrderSide.BUY, symbol="S", size=1)
    o2 = Order(side=OrderSide.BUY, symbol="S", size=1)
    id1 = me.register(o1, 0)
    id2 = me.submit(o2, 0)
    # 共用同一计数器 → 各自独立 ID；placement/state 由 submit 统一填写
    assert id1 != id2
    assert o1.id == id1 and o2.id == id2
    assert o1.placement_frame == 0 and o2.placement_frame == 0
    assert o1.state is OrderState.PENDING and o2.state is OrderState.PENDING


# ---------------------------------------------------------------------------
# find_order / pending_count 由 ABC 给出通用默认实现（非子类重复）
# ---------------------------------------------------------------------------

def test_find_order_and_pending_count_from_backend():
    assert MatchingEngine.find_order is ExecutionBackend.find_order
    assert MatchingEngine.pending_count is ExecutionBackend.pending_count
    # 默认实现仅依赖 pending_orders：在途 + 挂单
    me = MatchingEngine()
    me.submit(Order(side=OrderSide.BUY, symbol="S", size=1), 0)
    assert me.pending_count() == 1
    oid = me.pending_orders()[0].id
    assert me.find_order(oid) is me.pending_orders()[0]
    assert me.find_order(OrderId(99999)) is None


# ---------------------------------------------------------------------------
# 自定义 executor（模拟实盘 LiveConnector）驱动引擎
# ---------------------------------------------------------------------------

class _StubLiveExecutor(ExecutionBackend):
    """测试用执行后端：市价单在达延迟后按开盘价成交（模拟实盘 LiveConnector）。

    它复用了 MatchingEngine 的核心语义（延迟 1 帧、市价单按开盘价成交），但完全是
    一个独立实现 —— 用它证明引擎与策略不依赖具体后端。
    """

    def __init__(self, delay: int = 1):
        self.delay = delay
        self._in_flight: list[tuple[Order, int]] = []
        self._counter = 0
        self.submitted: list[tuple[Order, int]] = []

    def submit(self, order: Order, frame_id: int) -> OrderId:
        self._counter += 1
        order.id = OrderId(self._counter)
        order.placement_frame = int(frame_id)
        order.state = OrderState.PENDING
        self.submitted.append((order, int(frame_id)))
        self._in_flight.append((order, int(frame_id)))
        return order.id

    def cancel(self, order_id: OrderId, frame_id: int) -> CancelResult:
        for i, (o, _) in enumerate(self._in_flight):
            if o.id == order_id:
                del self._in_flight[i]
                o.state = OrderState.CANCELLED
                return CancelResult(True)
        return CancelResult(False)

    def process(self, frame, frame_id, pre_trade_check=None):
        fills: list[FillEvent] = []
        still: list[tuple[Order, int]] = []
        for order, placed in self._in_flight:
            if frame_id - placed < self.delay:
                still.append((order, placed))
                continue
            if pre_trade_check is not None:
                reason = pre_trade_check(order, frame)
                if reason is not None:
                    order.state = OrderState.REJECTED
                    continue
            if order.order_type is OrderType.MARKET and frame.has_bar(order.symbol):
                size = order.size
                if size <= 0:
                    order.state = OrderState.REJECTED
                    still.append((order, placed))
                    continue
                fills.append(
                    FillEvent(
                        order_id=order.id, side=order.side, symbol=order.symbol,
                        size=size, price=frame.open(order.symbol), frame_id=frame_id,
                        timestamp=frame_id,
                    )
                )
                order.state = OrderState.FILLED
            else:
                still.append((order, placed))
        self._in_flight = still
        return fills

    def has_in_flight(self) -> bool:
        return bool(self._in_flight)

    def pending_orders(self) -> list[Order]:
        return [o for o, _ in self._in_flight]

    def register_group(self, group, frame_id):
        return [self.submit(o, frame_id) for o in group.orders]


def test_engine_accepts_custom_executor():
    """引擎可注入自定义 executor；self.matcher == self.executor。"""
    from framebt import Engine as _Engine

    stub = _StubLiveExecutor(delay=1)
    feed = BarFeed.from_dicts(_bars(10), symbol="S")
    eng = _Engine(feed, _Probe(), initial_cash=1_000_000.0, executor=stub)
    assert eng.executor is stub
    assert eng.matcher is stub
    assert isinstance(eng.matcher, ExecutionBackend)


def test_custom_executor_matches_matching_engine():
    """同一场景：自定义 executor 与 MatchingEngine 产出逐笔一致的成交。"""
    from framebt import Engine as _Engine

    class BuyOnce(FrameStrategy):
        def __init__(self):
            self.done = False

        def on_frame(self, frame, token, bus, host):
            if not self.done and frame.has_bar("S") and frame.frame_id >= 2:
                self.done = True
                return [Order(side=OrderSide.BUY, symbol="S", size=100)]
            return []

    bars = _bars(10, price=10.0)

    # 匹配引擎（回测后端）
    r_me = _Engine(BarFeed.from_dicts(bars, symbol="S"), BuyOnce(),
                   initial_cash=1_000_000.0).run()
    fills_me = [(f.order_id, f.price, f.size, f.frame_id) for f in r_me.fills]

    # 自定义 executor（模拟实盘）
    stub = _StubLiveExecutor(delay=1)
    r_stub = _Engine(BarFeed.from_dicts(bars, symbol="S"), BuyOnce(),
                     initial_cash=1_000_000.0, executor=stub).run()
    fills_stub = [(f.price, f.size, f.frame_id) for f in r_stub.fills]

    assert len(fills_me) == len(fills_stub) == 1
    # 帧2下单 → 延迟1帧 → 帧3按开盘价10成交；两后端逐笔一致
    assert fills_me[0][1] == 10.0 and fills_me[0][2] == 100 and fills_me[0][3] == 3
    assert fills_me[0][1:] == fills_stub[0]
    # 自定义 executor 确实提交了订单
    assert len(stub.submitted) == 1 and stub.submitted[0][1] == 2


def test_delay_zero_validation_skipped_for_custom_executor():
    """delay_frames<1 仅对默认 MatchingEngine 校验；自定义 executor（如实盘）自管延迟。

    因此传入自定义 executor 时即便 delay_frames 为默认 1 也正常构造（不抛），
    且自定义 executor 可用任意 delay 语义。
    """
    from framebt import Engine as _Engine

    stub = _StubLiveExecutor(delay=1)
    feed = BarFeed.from_dicts(_bars(10), symbol="S")
    eng = _Engine(feed, _Probe(), initial_cash=1_000_000.0, executor=stub)
    assert eng.executor is stub
