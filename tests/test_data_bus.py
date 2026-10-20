"""Data Bus 测试（§4）：因果令牌 / window / lag / 横截面 / 多时间框架。"""

from __future__ import annotations

import math

import pytest

from framebt import CausalToken, CausalityViolation, DataBus


def _bus_with_series(n: int = 10):
    bus = DataBus()
    for i in range(n):
        bus.record("close", "A", i, 100.0 + i)
        bus.record("close", "B", i, 50.0 + i * 2)
    return bus


# ---- §4.1 因果令牌 ----

def test_get_requires_token():
    bus = _bus_with_series()
    with pytest.raises(CausalityViolation):
        bus.get("close", "A")


def test_get_filters_future_by_token():
    bus = _bus_with_series()
    assert bus.get("close", "A", token=CausalToken(3, 3)) == [100.0, 101.0, 102.0, 103.0]


def test_window_and_period():
    bus = _bus_with_series()
    tok = CausalToken(9, 9)
    assert bus.get("close", "A", token=tok, window=3) == [107.0, 108.0, 109.0]
    assert bus.get("close", "A", token=tok, period=2) == [108.0, 109.0]


def test_window_larger_than_history_returns_all():
    bus = _bus_with_series(3)
    assert bus.get("close", "A", token=CausalToken(2, 2), window=100) == [100.0, 101.0, 102.0]


def test_window_zero_returns_empty_not_all_history():
    """window=0 曾因 ``values[-0:]`` 退化成全量历史；必须为空。"""
    bus = _bus_with_series(5)
    assert bus.get("close", "A", token=CausalToken(4, 4), window=0) == []


def test_negative_window_is_rejected():
    bus = _bus_with_series(5)
    with pytest.raises(ValueError):
        bus.get("close", "A", token=CausalToken(4, 4), window=-1)


def test_series_negative_lag_raises():
    """``series`` 是 get 之外的第二个读接口；负 lag 同样不得成为 I1 旁路。"""
    bus = _bus_with_series(10)
    bus.set_now(4)
    with pytest.raises(CausalityViolation):
        bus.series("close", "A", CausalToken(4, 4), lag=-2)


# ---- §4.3 lag 原语（排除当前帧，防同帧泄漏）----

def test_lag_excludes_current_frame():
    bus = _bus_with_series()
    tok = CausalToken(5, 5)
    # lag=0 含当前帧（ts=5）
    assert bus.get("close", "A", token=tok, window=2, lag=0) == [104.0, 105.0]
    # lag=1 排除当前帧 → 最近两根为 ts=3,4
    assert bus.get("close", "A", token=tok, window=2, lag=1) == [103.0, 104.0]


def test_lag_beyond_time_raises():
    bus = _bus_with_series()
    with pytest.raises(CausalityViolation):
        bus.get("close", "A", token=CausalToken(2, 2), lag=5)


# ---- §4.5 横截面 ----

def test_cross_section_registered_factor():
    bus = _bus_with_series()
    bus.register_factor("level", lambda seq, tok: seq[-1][1] if seq else float("nan"))
    rows = bus.cross_section("level", ["A", "B"], CausalToken(9, 9))
    assert set(rows) == {"A", "B"}
    # B 的 level = 50 + 9*2 = 68 < A 的 109 → A rank 1
    assert rows["A"].rank == 1
    assert rows["B"].rank == 2
    assert rows["A"].is_valid and rows["B"].is_valid
    assert math.isclose(rows["A"].value, 109.0)


def test_cross_section_rank_and_zscore():
    bus = DataBus()
    for i, (sym, val) in enumerate([("X", 1.0), ("Y", 2.0), ("Z", 3.0)]):
        bus.record("close", sym, 0, val)
    rows = bus.cross_section("close", ["X", "Y", "Z"], CausalToken(0, 0))
    assert rows["Z"].rank == 1 and rows["Y"].rank == 2 and rows["X"].rank == 3
    # zscore 均值 0
    assert math.isclose(sum(r.zscore for r in rows.values()), 0.0, abs_tol=1e-9)


def test_cross_section_insufficient_data_invalid():
    bus = DataBus()
    bus.record("close", "X", 0, 1.0)
    # Y 无数据
    rows = bus.cross_section("close", ["X", "Y"], CausalToken(0, 0))
    assert rows["X"].is_valid is True
    assert rows["Y"].is_valid is False
    assert math.isnan(rows["Y"].value)


def test_cross_section_factor_failure_is_nan_not_crash():
    """因子抛异常应被吞掉并标为无效，而非中断回测。"""
    bus = _bus_with_series()

    def boom(seq, tok):
        raise RuntimeError("factor error")

    bus.register_factor("boom", boom)
    rows = bus.cross_section("boom", ["A", "B"], CausalToken(9, 9))
    assert all(not r.is_valid for r in rows.values())
    assert all(math.isnan(r.value) for r in rows.values())


def test_cross_section_lag_excludes_current():
    bus = _bus_with_series()
    bus.register_factor("last", lambda seq, tok: seq[-1][1] if seq else float("nan"))
    rows = bus.cross_section("last", ["A"], CausalToken(5, 5), lag=1)
    assert math.isclose(rows["A"].value, 104.0)  # 排除 ts=5


# ---- 多标的隔离 ----

def test_series_are_per_symbol_isolated():
    bus = _bus_with_series()
    tok = CausalToken(9, 9)
    assert bus.get("close", "A", token=tok, window=1) == [109.0]
    assert bus.get("close", "B", token=tok, window=1) == [68.0]


# ---- §4.4 多时间框架对齐（is_forming）----

def test_multi_tf_closed_bars_exact_multiple():
    bus = _bus_with_series(10)
    closed, forming = bus.get_multi_tf("close", "A", CausalToken(9, 9), bucket=5)
    assert closed == [(4, 104.0), (9, 109.0)]
    assert forming is None  # 10 % 5 == 0


def test_multi_tf_forming_bar_present():
    bus = _bus_with_series(7)
    closed, forming = bus.get_multi_tf("close", "A", CausalToken(6, 6), bucket=5)
    assert closed == [(4, 104.0)]
    assert forming == (6, 106.0)  # 7 % 5 = 2 → 正在形成


def test_multi_tf_respects_causality():
    """forming bar 与 closed bars 都不得越过 token.issued_at。"""
    bus = _bus_with_series(10)
    closed, forming = bus.get_multi_tf("close", "A", CausalToken(6, 6), bucket=5)
    assert closed == [(4, 104.0)]
    assert forming == (6, 106.0)
    assert all(ts <= 6 for ts, _ in closed)
    assert forming[0] <= 6


def test_multi_tf_bucket_must_be_positive():
    bus = _bus_with_series()
    with pytest.raises(ValueError):
        bus.get_multi_tf("close", "A", CausalToken(5, 5), bucket=0)


# ---- §4.1 令牌不可伪造（引擎时钟钳制）----

def test_forged_future_token_is_clamped_by_engine_clock():
    """伪造 issued_at 更大的 token 不得拿到任何额外数据。

    CausalToken 是普通对象，策略可自造。总线以**引擎权威时间**（set_now）钳制，
    因此伪造无效 —— I1 由总线自身保证，而非依赖「恰好没预加载未来数据」。
    """
    bus = _bus_with_series(10)          # close = 100..109，ts 0..9
    bus.set_now(4)                      # 引擎当前虚拟时间 = 4

    legit = bus.get("close", "A", token=CausalToken(4, 4))
    assert legit == [100.0, 101.0, 102.0, 103.0, 104.0]

    forged = bus.get("close", "A", token=CausalToken(9, 999))
    assert forged == legit, "伪造 token 不得多拿数据"
    assert bus.forgery_attempts == 1


def test_forged_token_series_also_clamped():
    bus = _bus_with_series(10)
    bus.set_now(3)
    assert len(bus.series("close", "A", CausalToken(9, 999))) == 4  # ts 0..3


def test_no_engine_clock_token_is_authoritative():
    """无引擎驱动时（离线分析/单测），token 自身权威，保持原有语义。"""
    bus = _bus_with_series(10)
    assert len(bus.get("close", "A", token=CausalToken(4, 4))) == 5
    assert bus.forgery_attempts == 0


def test_engine_clock_bounds_bus_during_backtest():
    """引擎回测中，总线权威时间应始终等于当前帧的结束时间。"""
    from framebt import BarFeed, Engine, FrameStrategy

    seen = []

    class ClockProbe(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            seen.append((token.issued_at_ns, bus._now_ns))
            return []

    bars = [{"timestamp": i, "symbol": "S", "open": 1, "high": 1, "low": 1,
             "close": 1, "volume": 1.0} for i in range(5)]
    Engine(BarFeed.from_dicts(bars, symbol="S"), ClockProbe()).run()
    assert seen, "策略应被唤醒"
    assert all(issued == now for issued, now in seen), "权威时间应等于 token 时间"
