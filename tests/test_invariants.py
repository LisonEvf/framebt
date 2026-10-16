"""不变式测试（§2.3）—— 引擎必须始终满足的硬约束。

  I1  任何数据请求返回的记录 timestamp <= current_virtual_time
  I2  帧 t 生成的订单，最早可成交时间为帧 t+delay
  I3  相同配置逐位可复现
  I4  帧内多 Bar 按 (timestamp, sequence_id) 严格排序
  I5  策略对已过去帧的状态修改请求被拒绝
"""

from __future__ import annotations

import pytest

from framebt import (
    BarFeed,
    CausalToken,
    CausalityViolation,
    Engine,
    FrameStrategy,
    run_backtest,
)
from framebt.strategies import DualMAStrategy


def _bars(n: int, symbol: str = "S"):
    return [
        {"timestamp": i, "symbol": symbol, "open": i + 10, "high": i + 11,
         "low": i + 9, "close": i + 10, "volume": 1000.0}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# I1：因果性 —— 任何数据请求返回 timestamp <= issued_at
# ---------------------------------------------------------------------------
class _Probe(FrameStrategy):
    def __init__(self):
        self.observed = []

    def on_frame(self, frame, token, bus, host):
        seq = bus.series("close", "S", token)
        self.observed.append((frame.frame_id, token.issued_at_ns, [ts for ts, _ in seq]))
        return []


def test_i1_no_data_after_issued_at():
    probe = _Probe()
    Engine(BarFeed.from_dicts(_bars(8), symbol="S"), probe).run()
    assert probe.observed, "策略应至少被唤醒一次"
    for fid, issued_at, timestamps in probe.observed:
        assert all(ts <= issued_at for ts in timestamps), (
            f"帧 {fid} 看到了 issued_at={issued_at} 之后的数据：{timestamps}"
        )


def test_i1_bar_of_current_frame_is_visible():
    """帧 k 应能看到第 k 根 Bar（当日收盘在决策前已知），但不能看到第 k+1 根。"""
    probe = _Probe()
    Engine(BarFeed.from_dicts(_bars(6), symbol="S"), probe).run()
    for fid, _issued, ts in probe.observed:
        assert ts == list(range(fid + 1)), f"帧 {fid} 可见 bar 时间戳应为 0..{fid}，实为 {ts}"


def test_i1_missing_token_raises():
    from framebt import DataBus

    bus = DataBus()
    bus.record("close", "S", 0, 1.0)
    with pytest.raises(CausalityViolation):
        bus.get("close", "S")  # 无 token


def test_i1_future_request_via_token_window():
    """token 只能看到 issued_at 之前的数据，即便数据源已注入全部历史。"""
    from framebt import DataBus

    bus = DataBus()
    for i in range(10):
        bus.record("close", "S", i, float(i))
    tok = CausalToken(4, 3)
    assert bus.get("close", "S", token=tok) == [0.0, 1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------------------
# I2：延迟管线 —— 帧 t 订单最早帧 t+delay 成交
# ---------------------------------------------------------------------------
def test_i2_delay_pipeline_one_frame():
    res = run_backtest(_bars(20), DualMAStrategy("S", short=1, long=3),
                       symbol="S", initial_cash=100000.0, delay_frames=1)
    # 成交发生在触发帧之后（延迟 >= 1）
    for f in res.fills:
        assert f.frame_id >= 1


def test_i2_order_never_fills_on_placement_frame():
    """构造必然在帧 k 触发的信号，验证成交帧 >= k+1。"""
    # 收盘价下跌后急涨，MA1/MA3 在帧 3 金叉
    closes = [10, 9, 8, 10, 12, 14]
    bars = [{"timestamp": i, "symbol": "S", "open": c, "high": c,
             "low": c, "close": c, "volume": 1000.0} for i, c in enumerate(closes)]
    res = run_backtest(bars, DualMAStrategy("S", short=1, long=3),
                       symbol="S", initial_cash=10000.0, delay_frames=1)
    assert res.total_trades() >= 1
    assert res.fills[0].frame_id == 4, "帧3触发，延迟1帧 → 帧4成交"


def test_i2_larger_delay_pushes_fill_later():
    closes = [10, 9, 8, 10, 12, 14, 16, 18]
    bars = [{"timestamp": i, "symbol": "S", "open": c, "high": c,
             "low": c, "close": c, "volume": 1000.0} for i, c in enumerate(closes)]
    res = run_backtest(bars, DualMAStrategy("S", short=1, long=3),
                       symbol="S", initial_cash=10000.0, delay_frames=3)
    assert res.fills[0].frame_id >= 3 + 3


# ---------------------------------------------------------------------------
# I3：确定性 —— 逐位可复现
# ---------------------------------------------------------------------------
def test_i3_identical_output():
    r1 = run_backtest(_bars(40), DualMAStrategy("S", short=5, long=20),
                      symbol="S", initial_cash=100000.0, seed=2024)
    r2 = run_backtest(_bars(40), DualMAStrategy("S", short=5, long=20),
                      symbol="S", initial_cash=100000.0, seed=2024)
    assert r1.equity_curve == r2.equity_curve
    assert [(f.order_id, f.price, f.size, f.frame_id) for f in r1.fills] == \
           [(f.order_id, f.price, f.size, f.frame_id) for f in r2.fills]


def test_i3_determ_rng_reproducible():
    from framebt import DetermRNG

    a = DetermRNG(seed=42)
    b = DetermRNG(seed=42)
    assert [a.random() for _ in range(10)] == [b.random() for _ in range(10)]
    assert DetermRNG(seed=1).random() != DetermRNG(seed=2).random()


# ---------------------------------------------------------------------------
# I4：帧内多 Bar 按 (timestamp, sequence_id) 严格排序
# ---------------------------------------------------------------------------
def test_i4_feed_sorts_by_timestamp_then_sequence():
    # 乱序输入 + 同时间戳多事件
    raw = [
        {"timestamp": 2, "symbol": "A", "open": 1, "close": 1, "sequence_id": 5},
        {"timestamp": 0, "symbol": "A", "open": 1, "close": 1, "sequence_id": 1},
        {"timestamp": 2, "symbol": "B", "open": 1, "close": 1, "sequence_id": 0},
        {"timestamp": 1, "symbol": "A", "open": 1, "close": 1, "sequence_id": 3},
    ]
    from framebt import Bar

    bars = [Bar(timestamp=r["timestamp"], symbol=r["symbol"], open=1.0, high=1.0,
                low=1.0, close=1.0, sequence_id=r["sequence_id"]) for r in raw]
    feed = BarFeed(bars)
    keys = [(b.timestamp, b.sequence_id) for b in feed.bars]
    assert keys == sorted(keys)
    assert keys == [(0, 1), (1, 3), (2, 0), (2, 5)]


# ---------------------------------------------------------------------------
# I5：过去帧不可变 —— 策略在帧 k 无法改写/读取未来
# ---------------------------------------------------------------------------
def test_i5_visible_history_grows_monotonically():
    probe = _Probe()
    Engine(BarFeed.from_dicts(_bars(7), symbol="S"), probe).run()
    counts = [len(ts) for _f, _i, ts in probe.observed]
    assert counts == sorted(counts), "可见历史长度必须单调不减"
    # 且严格每次只多一根
    assert counts == list(range(1, len(counts) + 1))


def test_i5_engine_terminates_on_exhausted_data():
    res = run_backtest(_bars(5), DualMAStrategy("S", short=1, long=2), symbol="S",
                       initial_cash=10000.0)
    assert res.frames >= 5
    assert res.end_ns > 0
