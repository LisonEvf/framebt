"""统一指标口径（``framebt.metrics``）的回归测试。"""

from __future__ import annotations

import math

import pytest

from framebt import Performance, metrics
from framebt.core.result import BacktestResult


EQ = [100.0, 110.0, 99.0, 120.0]


def test_known_series_all_metrics():
    s = metrics.summarize(EQ, initial=100.0, periods_per_year=3, n_periods=3)
    assert s["total_return"] == pytest.approx(0.20, abs=1e-12)
    assert s["annualized_return"] == pytest.approx(0.20, abs=1e-12)
    assert s["volatility"] == pytest.approx(0.27385289597112583, rel=1e-12)
    assert s["sharpe"] == pytest.approx(0.7745808616300975, rel=1e-12)
    assert s["max_drawdown"] == pytest.approx(0.10, abs=1e-12)
    assert s["calmar"] == pytest.approx(2.0, abs=1e-12)
    assert s["years"] == pytest.approx(1.0, abs=1e-12)


def test_sharpe_is_mean_over_std_not_cagr_over_vol():
    r = [(b - a) / a for a, b in zip(EQ, EQ[1:])]
    mean = sum(r) / len(r)
    vol = metrics.annualized_volatility(EQ, periods_per_year=3)
    expected = mean / (vol / math.sqrt(3)) * math.sqrt(3)
    assert metrics.sharpe_ratio(EQ, periods_per_year=3) == pytest.approx(expected, rel=1e-12)
    # 旧口径（CAGR/vol）明显不同
    old = metrics.cagr(EQ, periods_per_year=3, n_periods=3) / vol
    assert old == pytest.approx(0.7303190981083776, rel=1e-9)
    assert old != pytest.approx(expected, rel=1e-6)


def test_initial_capital_includes_first_period_pnl():
    # 首帧从 100 涨到 110：以首帧权益为基准会算成 +10%，以初始资金为基准是 +21%
    eq = [110.0, 121.0]
    assert metrics.total_return(eq) == pytest.approx(0.10, abs=1e-12)
    assert metrics.total_return(eq, initial=100.0) == pytest.approx(0.21, abs=1e-12)
    assert metrics.cagr(eq, periods_per_year=1, n_periods=1, initial=100.0) == pytest.approx(0.21, abs=1e-12)


def test_sortino_only_penalizes_downside():
    up_only = [100.0, 101.0, 102.0, 103.0]
    assert metrics.sortino_ratio(up_only, periods_per_year=244) == 0.0  # 无下行 → 0（不伪造无穷）
    mixed = [100.0, 110.0, 99.0, 120.0]
    assert metrics.sortino_ratio(mixed, periods_per_year=3) > metrics.sharpe_ratio(mixed, periods_per_year=3)


def test_max_drawdown_positive_fraction():
    assert metrics.max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25, abs=1e-12)
    assert metrics.max_drawdown([100.0, 101.0]) == 0.0


def test_periods_per_year_scales_volatility():
    v244 = metrics.annualized_volatility(EQ, periods_per_year=244)
    v252 = metrics.annualized_volatility(EQ, periods_per_year=252)
    assert v252 / v244 == pytest.approx(math.sqrt(252 / 244), rel=1e-12)


def test_performance_delegates_and_uses_initial_cash():
    res = BacktestResult(
        equity_curve=[(0, 110.0), (1, 121.0)],
        frames=2,
        initial_cash=100.0,
    )
    p = Performance(res, period_frames=1)
    s = metrics.summarize([110.0, 121.0], initial=100.0, periods_per_year=1, n_periods=2)
    assert p.total_return == pytest.approx(s["total_return"])
    assert p.annualized_return == pytest.approx(s["annualized_return"])
    assert p.volatility == pytest.approx(s["volatility"])
    assert p.sharpe == pytest.approx(s["sharpe"])
    assert p.max_drawdown == pytest.approx(s["max_drawdown"])
    assert p.calmar == pytest.approx(s["calmar"])
    assert p.detailed()["periods_per_year"] == 1.0


def test_performance_without_initial_cash_falls_back_to_first_equity():
    res = BacktestResult(equity_curve=[(0, 100.0), (1, 120.0)], frames=1)
    p = Performance(res, period_frames=1)
    assert p.total_return == pytest.approx(0.20, abs=1e-12)  # 无 initial_cash → 首点


def test_empty_and_loss_series_are_finite():
    assert metrics.summarize([])["total_return"] == 0.0
    loss = metrics.summarize([100.0, 80.0], initial=100.0, periods_per_year=1, n_periods=1)
    assert loss["total_return"] == pytest.approx(-0.20)
    assert loss["annualized_return"] == pytest.approx(-0.20)
    assert math.isfinite(loss["sharpe"])


def test_engine_records_initial_cash():
    from framebt import FrameStrategy, run_backtest

    class Noop(FrameStrategy):
        def on_frame(self, frame, token, bus, host):
            return []

    bars = [{"timestamp": i, "symbol": "S", "open": 10, "high": 10,
             "low": 10, "close": 10, "volume": 1e6} for i in range(3)]
    res = run_backtest(bars, Noop(), initial_cash=10_000.0)
    assert res.initial_cash == pytest.approx(10_000.0)


def test_total_return_uses_initial_cash_when_equity_starts_below_it():
    # 手工构造 / 带期初持仓的场景：首帧权益 9000，初始资金 10000 → 总收益 -10%
    res = BacktestResult(equity_curve=[(0, 9000.0), (1, 9000.0)],
                         frames=2, initial_cash=10_000.0)
    p = Performance(res, period_frames=1)
    assert p.total_return == pytest.approx(-0.10, abs=1e-12)
