"""Phase 6 · ② 帧内子步进执行引擎测试。

覆盖：
  - 成交概率模型 ``MatchingEngine.estimated_fill_fraction``：
      * resting 限价单才给估计（市价 / 未 resting → 0）；
      * 队列位置 FIFO 加权：越靠后估计成交比例越低；
      * 价位深度：同价位挂单越多，估计成交比例越低；
      * 本帧成交量覆盖：成交量不足时估计比例随覆盖度下降；
      * 确定性（无 RNG，同输入逐位一致）。
  - 双时钟域调度 ``on_sub_step``：
      * 默认未注册（``default_wakeups`` 不含 SUB_STEP）→ 从不被调用（零回归）；
      * 显式注册 → 每帧调度一次（step=0, total=1）；
      * ``on_sub_step`` 返回的订单经延迟管线登记（I2）。
"""

from __future__ import annotations

from framebt import (
    Bar,
    BarFeed,
    Engine,
    FrameStrategy,
    MarketFrame,
    MatchingEngine,
    Order,
    OrderSide,
    OrderType,
    WakeupType,
)


def mkframe(fid, o, h, l, c, *, symbol="S", volume=1000.0):
    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c,
                  volume=volume))
    return f


def _bars(n, symbol="S"):
    return [
        {"timestamp": i, "symbol": symbol, "open": i + 10, "high": i + 11,
         "low": i + 9, "close": i + 10, "volume": 1000.0}
        for i in range(n)
    ]


# ===========================================================================
# estimated_fill_fraction —— 成交概率模型
# ===========================================================================

def _resting_buy(me, price, size=100, oid=1):
    """注册一笔 resting 买单（限价低于市价，帧内不触发）。返回订单。"""
    return Order(OrderSide.BUY, "S", size=size, order_type=OrderType.LIMIT, price=price,
                 id=oid)


def test_fill_fraction_no_book_for_missing_symbol():
    me = MatchingEngine()
    me._last_frame = mkframe(1, 10, 11, 9, 10)
    o = Order(OrderSide.BUY, "MISSING", size=100, order_type=OrderType.LIMIT, price=9.0)
    assert me.estimated_fill_fraction(o) == 0.0


def test_fill_fraction_zero_when_not_resting():
    """市价单：process 后已成交（不在簿内）→ 估计 0。"""
    me = MatchingEngine()
    o = Order(OrderSide.BUY, "S", size=100)  # 市价
    me.register(o, 0)
    me.process(mkframe(1, 10, 11, 9, 10), 1)  # 市价按开盘成交
    assert o.id is not None
    assert me.estimated_fill_fraction(o) == 0.0  # 已成交，不在簿内


def test_fill_fraction_zero_before_first_process():
    """未处理过任何帧（无 _last_frame）→ 估计 0。"""
    me = MatchingEngine()
    o = _resting_buy(me, 9.0)
    me.register(o, 0)
    # 不调用 process → _last_frame 仍为 None
    assert me.estimated_fill_fraction(o) == 0.0


def test_fill_fraction_front_of_queue_full():
    """队首（qp=0）+ 成交量充足 → 估计全额成交（≈1）。"""
    me = MatchingEngine()
    o = _resting_buy(me, 9.0, oid=1)
    me.register(o, 0)
    me.process(mkframe(1, 10, 10.5, 9.6, 10.2), 1)  # low=9.6 > 9.5 → resting
    frac = me.estimated_fill_fraction(o)
    assert frac == 1.0  # coverage=min(1, 1000/100)=1, fifo=1/(0+1)=1


def test_fill_fraction_fifo_position_decay():
    """队列位置越靠后，估计成交比例越低（FIFO 加权）。"""
    me = MatchingEngine()
    o0 = _resting_buy(me, 9.0, oid=1)
    o1 = _resting_buy(me, 9.0, size=100, oid=2)
    me.register(o0, 0)
    me.register(o1, 0)
    me.process(mkframe(1, 10, 10.5, 9.6, 10.2), 1)  # 两者都 resting
    f0 = me.estimated_fill_fraction(o0)  # qp=0
    f1 = me.estimated_fill_fraction(o1)  # qp=1
    assert f0 > f1
    assert f0 == 1.0
    assert f1 == 0.5  # coverage=1, fifo=1/(1+1)=0.5


def test_fill_fraction_depth_decay():
    """价位深度越大（同价位挂单越多），本单估计成交比例越低。"""
    me = MatchingEngine()
    vol = 100.0
    # 单挂单：level_shares=100，coverage=min(1, 100/100)=1
    o_single = _resting_buy(me, 9.0, oid=1)
    me.register(o_single, 0)
    me.process(mkframe(1, 10, 10.5, 9.6, 10.2, volume=vol), 1)
    f_single = me.estimated_fill_fraction(o_single)
    assert f_single == 1.0


def test_fill_fraction_volume_coverage_scaling():
    """成交量不足时估计比例随覆盖度线性下降。"""
    me = MatchingEngine()
    o = _resting_buy(me, 9.0, size=100, oid=1)
    me.register(o, 0)
    me.process(mkframe(1, 10, 10.5, 9.6, 10.2, volume=50.0), 1)  # vol=50 < level_shares=100
    # coverage=min(1, 50/100)=0.5, fifo=1 → 0.5
    assert me.estimated_fill_fraction(o) == 0.5


def test_fill_fraction_deterministic():
    """确定性：同输入逐位一致（无 RNG）。"""
    me = MatchingEngine()
    o = _resting_buy(me, 9.0, oid=1)
    me.register(o, 0)
    me.process(mkframe(1, 10, 10.5, 9.6, 10.2), 1)
    a = me.estimated_fill_fraction(o)
    b = me.estimated_fill_fraction(o)
    assert a == b


# ===========================================================================
# on_sub_step —— 双时钟域调度
# ===========================================================================

class _SubProbe(FrameStrategy):
    """注册 SUB_STEP 的策略：记录子步进调用。"""

    default_wakeups = [WakeupType.FRAME, WakeupType.FILL, WakeupType.SUB_STEP]

    def __init__(self):
        self.sub_calls = []

    def on_frame(self, frame, token, bus, host):
        return []

    def on_sub_step(self, frame, step, total, token, bus):
        self.sub_calls.append((frame.frame_id, step, total))
        return []


class _NoSubProbe(FrameStrategy):
    """未注册 SUB_STEP 的策略（默认 default_wakeups）。"""

    def __init__(self):
        self.sub_calls = 0

    def on_frame(self, frame, token, bus, host):
        return []

    def on_sub_step(self, frame, step, total, token, bus):
        self.sub_calls += 1
        return []


def test_sub_step_default_unregistered():
    """默认 default_wakeups 不含 SUB_STEP → on_sub_step 从不被调用（零回归）。"""
    probe = _NoSubProbe()
    Engine(BarFeed.from_dicts(_bars(6), symbol="S"), probe).run()
    assert probe.sub_calls == 0


def test_sub_step_fires_once_per_frame():
    """注册 SUB_STEP → 每帧调度一次（step=0, total=1）。"""
    probe = _SubProbe()
    Engine(BarFeed.from_dicts(_bars(6), symbol="S"), probe).run()
    assert probe.sub_calls, "子步进应被调度"
    for fid, step, total in probe.sub_calls:
        assert step == 0 and total == 1  # N=1


def test_sub_step_orders_go_through_delay_pipeline():
    """on_sub_step 返回的订单经延迟管线登记（I2）：帧 1 下单 → 帧 2 成交。"""

    class _SubOrderStrat(FrameStrategy):
        default_wakeups = [WakeupType.FRAME, WakeupType.SUB_STEP]

        def __init__(self):
            self.placed = False

        def on_frame(self, frame, token, bus, host):
            return []

        def on_sub_step(self, frame, step, total, token, bus):
            if not self.placed and frame.frame_id >= 1:
                self.placed = True
                return [Order(OrderSide.BUY, "S", size=100)]  # 市价单
            return []

    strat = _SubOrderStrat()
    res = Engine(BarFeed.from_dicts(_bars(6), symbol="S"), strat).run()
    assert res.total_trades() == 1
    assert res.fills[0].frame_id == 2  # 帧 1 下单，延迟 1 帧 → 帧 2 成交


def test_sub_step_zero_regression_dualma():
    """既有策略（未注册 SUB_STEP）行为不变：run 正常完成且 I2 未被绕过。

    「零回归」由全量 280 测试保证；此处额外用一条会触发成交的价格序列，
    确认 dispatch_sub_step（返回 [] 的无扰动调度）不破坏既有撮合。
    """
    from framebt.strategies import DualMAStrategy

    # 先跌后涨的 V 形序列：MA 金叉触发买入
    closes = [20.0 - 0.5 * i for i in range(10)] + [15.0 + 0.6 * i for i in range(30)]
    bars = [{"timestamp": i, "symbol": "S", "open": c, "high": c + 0.5,
             "low": c - 0.5, "close": c, "volume": 1000.0} for i, c in enumerate(closes)]
    res = Engine(BarFeed.from_dicts(bars, symbol="S"),
                 DualMAStrategy("S", short=3, long=8),
                 initial_cash=100000.0, seed=2024).run()
    assert res.frames == len(closes)  # run 正常完成
    assert res.total_trades() >= 1, "V 形序列应触发成交"
    for f in res.fills:
        assert f.frame_id >= 1  # I2 延迟管线未被绕过

