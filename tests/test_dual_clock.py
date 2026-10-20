"""Phase 6 · ④ 双时钟域调度测试。

核心命题：策略跑在「日帧时钟」（``on_frame``，慢时钟），执行跑在「子步进时钟」
（``on_sub_step`` × ``sub_steps``，快时钟）。父调度（``Engine.run`` 循环）在每帧内
按 ``step=0..sub_steps-1`` 顺序推进子步进时钟，把两把时钟同步到同一虚拟时刻。

覆盖：
  - 父调度按 N 子步进调度 ``on_sub_step``（``step`` 0..N-1，``total`` = N）；
  - 默认 N=1（零回归）：每帧调度一次（``step=0``，``total=1``）；
  - 子步进顺序调度、共享状态：子步进 i 登记的订单被子步进 i+1 的
    ``host.pending_orders`` 可见——父调度据此把两把时钟串成顺序链；
  - ``on_sub_step`` 返回的订单经延迟管线登记（I2：帧 t 订单最早帧 t+1 成交）；
  - 双时钟域时序：每帧先 ``on_frame``（日帧时钟），再 ``on_sub_step`` × N（执行时钟）。
"""

from __future__ import annotations

from framebt import (
    Bar,
    BarFeed,
    Engine,
    FrameStrategy,
    MarketFrame,
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


def _bars(n: int, symbol: str = "S"):
    return [
        {"timestamp": i, "symbol": symbol, "open": i + 10, "high": i + 11,
         "low": i + 9, "close": i + 10, "volume": 1000.0}
        for i in range(n)
    ]


# ===========================================================================
# 父调度按 N 子步进调度（step / total）
# ===========================================================================

def test_sub_step_n_fires_n_times():
    """N=3：每帧调度 3 次（step=0,1,2；total=3）。"""
    seen: list[tuple[int, int]] = []

    class _S(FrameStrategy):
        default_wakeups = [WakeupType.FRAME, WakeupType.SUB_STEP]

        def on_frame(self, frame, token, bus, host):
            return []

        def on_sub_step(self, frame, step, total, token, bus):
            seen.append((step, total))
            return []

    Engine(BarFeed.from_dicts(_bars(5)), _S(), sub_steps=3).run()
    # 5 帧 × 3 子步进 = 15 次；每帧内 step 0,1,2 且 total=3
    assert seen == [(s, 3) for _frame in range(5) for s in range(3)]


def test_sub_step_zero_regression_default_n_is_one():
    """默认 N=1（零回归）：每帧调度一次（step=0，total=1）。"""
    seen: list[tuple[int, int]] = []

    class _S(FrameStrategy):
        default_wakeups = [WakeupType.FRAME, WakeupType.SUB_STEP]

        def on_frame(self, frame, token, bus, host):
            return []

        def on_sub_step(self, frame, step, total, token, bus):
            seen.append((step, total))
            return []

    Engine(BarFeed.from_dicts(_bars(5)), _S()).run()
    # 5 根 bar → 5 帧；每帧 on_sub_step 调度一次（step=0, total=1）
    assert len(seen) == 5
    assert all(s == 0 and t == 1 for s, t in seen)


def test_sub_step_invalid_value_rejected():
    """sub_steps < 1 抛 ValueError（双时钟域：每帧至少 1 个子步进）。"""
    import pytest

    with pytest.raises(ValueError):
        Engine(BarFeed.from_dicts(_bars(5)), FrameStrategy(), sub_steps=0)


# ===========================================================================
# 子步进共享状态（父调度顺序同步两把时钟）
# ===========================================================================

def test_sub_step_sequential_state_sharing():
    """子步进 i 登记的订单被子步进 i+1 的 ``host.pending_orders`` 可见。

    父调度在每帧内按 step 顺序调度：子步进 0 登记订单后立刻调用
    ``_register_items``（入 matcher），故同帧子步进 1 的 ``pending_orders`` 就能看到它——
    这正是「执行子步进时钟」在单帧内顺序推进、与「日帧时钟」同步的体现。

    只在首帧发一笔市价买单（次帧 process 即成交），避免数据耗尽后始终有在途订单
    （永不成交的挂单会让引擎在静态帧上自旋）。
    """
    log: list = []

    class _S(FrameStrategy):
        default_wakeups = [WakeupType.FRAME, WakeupType.SUB_STEP]

        def on_frame(self, frame, token, bus, host):
            self.host = host  # on_sub_step 无 host 形参，经 on_frame 持有 host
            return []

        def on_sub_step(self, frame, step, total, token, bus):
            if frame.frame_id == 0 and step == 0:
                # 首帧子步进 0 发一笔市价买单（次帧 process 按开盘成交）
                return [Order(OrderSide.BUY, "S", size=10)]
            log.append(len(self.host.pending_orders()))
            return []

    Engine(BarFeed.from_dicts(_bars(3)), _S(), sub_steps=2).run()
    # 帧 0 子步进 1：能看到帧 0 子步进 0 登记的在途订单（成交前）
    assert log[0] == 1
    # 帧 1 起：市价买单已在 process 成交，在途归零
    assert log[1] == 0


# ===========================================================================
# I2 延迟管线：on_sub_step 返回的订单也受延迟约束
# ===========================================================================

def test_sub_step_orders_go_through_delay_pipeline():
    """``on_sub_step`` 返回的市价买单经延迟管线登记（I2：帧 t 订单最早帧 t+1 成交）。

    只在首帧子步进发一笔市价买单（避免每帧都发 → 数据耗尽后始终有在途订单）。
    """
    feed = BarFeed.from_dicts(_bars(6))
    eng = Engine(feed, _SubStepBuyer(), sub_steps=1)
    res = eng.run()

    assert len(res.fills) >= 1, "子步进市价买单应成交"
    for f in res.fills:
        assert f.frame_id >= 1  # I2：子步进订单最早帧 t+1 成交，未绕过延迟管线


class _SubStepBuyer(FrameStrategy):
    default_wakeups = [WakeupType.FRAME, WakeupType.SUB_STEP]

    def __init__(self):
        self.placed = False

    def on_frame(self, frame, token, bus, host):
        return []

    def on_sub_step(self, frame, step, total, token, bus):
        if self.placed:
            return []
        self.placed = True
        # 市价买单，开盘即成交（开盘价 i+10 始终为正）
        return [Order(OrderSide.BUY, "S", size=100)]


# ===========================================================================
# 双时钟域时序：先 on_frame（日帧时钟），再 on_sub_step × N（执行时钟）
# ===========================================================================

def test_dual_clock_frame_then_sub_step_order():
    """父调度每帧先 ``on_frame``（日帧时钟），再 ``on_sub_step`` × N（执行时钟）。"""
    order: list = []

    class _S(FrameStrategy):
        default_wakeups = [WakeupType.FRAME, WakeupType.SUB_STEP]

        def on_frame(self, frame, token, bus, host):
            order.append(("frame", frame.frame_id))
            return []

        def on_sub_step(self, frame, step, total, token, bus):
            order.append(("sub_step", frame.frame_id, step))
            return []

    Engine(BarFeed.from_dicts(_bars(3)), _S(), sub_steps=2).run()
    assert order == [
        ("frame", 0), ("sub_step", 0, 0), ("sub_step", 0, 1),
        ("frame", 1), ("sub_step", 1, 0), ("sub_step", 1, 1),
        ("frame", 2), ("sub_step", 2, 0), ("sub_step", 2, 1),
    ]


def test_sub_step_does_not_break_existing_strategy():
    """既有策略（未注册 SUB_STEP）行为不变：run 正常且 I2 未被绕过。

    「零回归」由全量测试保证；此处用一条会触发成交的价格序列，确认父调度的
    子步进循环（返回 [] 的无扰动调度）不破坏既有撮合。
    """
    from framebt.strategies import DualMAStrategy

    closes = [20.0 - 0.5 * i for i in range(10)] + [15.0 + 0.6 * i for i in range(30)]
    bars = [{"timestamp": i, "symbol": "S", "open": c, "high": c + 0.5,
             "low": c - 0.5, "close": c, "volume": 1000.0}
            for i, c in enumerate(closes)]
    res = Engine(BarFeed.from_dicts(bars, symbol="S"),
                 DualMAStrategy("S", short=3, long=8),
                 initial_cash=100000.0, seed=2024).run()
    assert res.frames == len(closes)
    assert res.total_trades() >= 1, "V 形序列应触发成交"
    for f in res.fills:
        assert f.frame_id >= 1  # I2 延迟管线未被绕过
