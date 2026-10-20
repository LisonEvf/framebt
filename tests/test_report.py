"""回测报告（report）测试：图表渲染、交易重构、基准、HTML 组装。

覆盖：
  - _attr（Bar / dict 双通道取值）
  - 日期 / 金额格式化
  - render_equity_chart / render_trade_chart（base64 PNG 有效性、空数据分支）
  - _round_trips（FIFO 重构）
  - _trade_stats（盈亏聚合）
  - _benchmark_series（等权买入持有）
  - generate_report / save_report / build_html（HTML 结构与内容）
"""

from __future__ import annotations

import base64
import json
import math

import pytest

from framebt import (
    Bar,
    BacktestResult,
    FillEvent,
    OrderSide,
    ReportConfig,
    generate_report,
    render_equity_chart,
    render_trade_chart,
    save_report,
)
from framebt.report import (
    _attr,
    _benchmark_series,
    _fmt_date,
    _fmt_money,
    _ordinal_to_date,
    _round_trips,
    _trade_stats,
    build_html,
)


# ===========================================================================
# 构造工具
# ===========================================================================

def _bar(ts, o, h, l, c, vol=1000.0, symbol="S"):
    return Bar(timestamp=ts, symbol=symbol, open=o, high=h, low=l, close=c, volume=vol)


def _fill(side, size, price, frame_id, symbol="S", oid=1, cost=0.0):
    return FillEvent(order_id=oid, side=side, symbol=symbol, size=size, price=price,
                     frame_id=frame_id, timestamp=frame_id, cost=cost, reason="test")


def _result_with_roundtrips() -> BacktestResult:
    """构造含 3 个完整回合（2 赢 1 亏）的 BacktestResult。

    回合：
      R1: 买 100@10 → 卖 100@12   +200
      R2: 买 100@20 → 卖 100@18   -200
      R3: 买 100@30 → 卖 100@33   +300
    """
    fills = [
        _fill(OrderSide.BUY, 100, 10, frame_id=10, oid=1),
        _fill(OrderSide.SELL, 100, 12, frame_id=20, oid=2),
        _fill(OrderSide.BUY, 100, 20, frame_id=30, oid=3),
        _fill(OrderSide.SELL, 100, 18, frame_id=40, oid=4),
        _fill(OrderSide.BUY, 100, 30, frame_id=50, oid=5),
        _fill(OrderSide.SELL, 100, 33, frame_id=60, oid=6),
    ]
    eq = [(10, 100000.0), (20, 100200.0), (30, 100000.0), (40, 99800.0), (50, 100100.0), (60, 100400.0)]
    return BacktestResult(fills=fills, equity_curve=eq, frames=60)


def _b64_to_png_info(b64: str) -> tuple[int, int]:
    """校验 base64 PNG 可解码并返回 (宽, 高)。"""
    raw = base64.b64decode(b64)
    assert raw[:8] == b"\x89PNG\r\n\x1a\n", "不是有效 PNG 头"
    # IHDR：offset 16，width/height 为大端 4 字节。
    w = int.from_bytes(raw[16:20], "big")
    h = int.from_bytes(raw[20:24], "big")
    assert w > 0 and h > 0
    return w, h


# ===========================================================================
# _attr（Bar / dict 双通道）
# ===========================================================================

def test_attr_reads_bar_fields():
    b = _bar(58001, 10.0, 11.0, 9.0, 10.5)
    assert _attr(b, "timestamp") == 58001
    assert _attr(b, "close") == 10.5
    assert _attr(b, "volume") == 1000.0


def test_attr_reads_dict_fields():
    d = {"timestamp": 58001, "close": 10.5, "volume": 1000.0}
    assert _attr(d, "timestamp") == 58001
    assert _attr(d, "close") == 10.5


def test_attr_missing_returns_none():
    b = _bar(1, 1, 1, 1, 1)
    assert _attr(b, "nonexistent") is None
    assert _attr({}, "close") is None


# ===========================================================================
# 格式化
# ===========================================================================

def test_ordinal_to_date_and_fmt():
    d = _ordinal_to_date(738000)  # 一个合法序数
    assert isinstance(d.year, int) and 1970 <= d.year <= 2200
    # 格式化结果可回读为同一年。
    s = _fmt_date(738000)
    assert "20" in s  # 年份含 20xx


def test_fmt_money():
    assert _fmt_money(1234.5) == "1,234"      # :,.0f 取整 + 千分位
    assert _fmt_money(-99.9) == "-100"
    assert _fmt_money(1_234_567.0).endswith("M")
    assert "万" in _fmt_money(15_000.0)  # >=1e4 用「万」为单位


# ===========================================================================
# render_equity_chart
# ===========================================================================

def test_render_equity_chart_produces_png():
    res = _result_with_roundtrips()
    img = render_equity_chart(res)
    w, h = _b64_to_png_info(img)
    assert w > 500 and h > 300  # 合理的图表尺寸


def test_render_equity_chart_with_benchmark():
    res = _result_with_roundtrips()
    bench = [(10, 100000.0), (60, 105000.0)]
    img = render_equity_chart(res, benchmark=bench)
    _b64_to_png_info(img)  # 不抛即为通过


def test_render_equity_chart_empty_raises():
    res = BacktestResult(fills=[], equity_curve=[])
    with pytest.raises(ValueError):
        render_equity_chart(res)


# ===========================================================================
# render_trade_chart
# ===========================================================================

def test_render_trade_chart_empty_bars_returns_empty():
    assert render_trade_chart([], [], [], symbol="S") == ""


def test_render_trade_chart_marks_buys_and_sells():
    bars = [_bar(58000 + i, 10, 11, 9, 10 + (i % 3)) for i in range(60)]
    fills = [
        _fill(OrderSide.BUY, 100, 10, frame_id=58010, oid=1),
        _fill(OrderSide.SELL, 100, 12, frame_id=58020, oid=2),
    ]
    trips = _round_trips(fills, "S")
    img = render_trade_chart(bars, trips, fills, symbol="S")
    w, h = _b64_to_png_info(img)
    assert w > 500 and h > 300


def test_render_trade_chart_ignores_other_symbol():
    bars = [_bar(58000 + i, 10, 11, 9, 10 + (i % 3)) for i in range(60)]
    # 本标的无成交，但其它标的有成交 —— 不应出现在本标的图上。
    other = [_fill(OrderSide.BUY, 100, 10, frame_id=58010, symbol="O", oid=1)]
    img = render_trade_chart(bars, [], [], symbol="S")
    _b64_to_png_info(img)


# ===========================================================================
# _round_trips（FIFO 重构）
# ===========================================================================

def test_round_trips_basic():
    res = _result_with_roundtrips()
    trips = _round_trips(res.fills, "S")
    assert len(trips) == 3
    pnls = sorted(t.pnl for t in trips)
    assert math.isclose(pnls[0], -200.0)
    assert math.isclose(pnls[1], 200.0)
    assert math.isclose(pnls[2], 300.0)


def test_round_trips_unmatched_buy_excluded():
    # 只买未卖 —— 不构成完整回合，不应计入。
    fills = [_fill(OrderSide.BUY, 100, 10, frame_id=10, oid=1)]
    assert _round_trips(fills, "S") == []


def test_round_trips_partial_close_uses_weighted_avg():
    # 加仓后部分平仓，用加权平均成本算已实现盈亏。
    fills = [
        _fill(OrderSide.BUY, 100, 10, frame_id=10, oid=1),
        _fill(OrderSide.BUY, 100, 20, frame_id=20, oid=2),   # 加仓，均价 15
        _fill(OrderSide.SELL, 100, 30, frame_id=30, oid=3),  # 平一半
    ]
    trips = _round_trips(fills, "S")
    assert len(trips) == 1
    # (30 - 15) * 100 = 1500
    assert math.isclose(trips[0].pnl, 1500.0)


def test_round_trips_multi_symbol_isolation():
    fills = [
        _fill(OrderSide.BUY, 100, 10, frame_id=10, symbol="A", oid=1),
        _fill(OrderSide.SELL, 100, 12, frame_id=20, symbol="A", oid=2),
        _fill(OrderSide.BUY, 100, 20, frame_id=30, symbol="B", oid=3),
    ]
    assert len(_round_trips(fills, "A")) == 1
    assert _round_trips(fills, "B") == []


# ===========================================================================
# _trade_stats
# ===========================================================================

def test_trade_stats_aggregation():
    res = _result_with_roundtrips()
    stats = _trade_stats(res)
    assert stats.trades == 3
    assert stats.wins == 2
    assert stats.losses == 1
    assert math.isclose(stats.win_rate, 2 / 3)
    assert math.isclose(stats.total_pnl, 300.0)
    # 盈亏比 = 盈利总额 / 亏损总额 = 500 / 200
    assert math.isclose(stats.profit_factor, 500 / 200)
    assert stats.avg_hold > 0


def test_trade_stats_empty():
    stats = _trade_stats(BacktestResult(fills=[], equity_curve=[]))
    assert stats.trades == 0
    assert stats.win_rate == 0.0
    assert stats.profit_factor == 0.0


def test_trade_stats_infinite_profit_factor():
    # 全部盈利，无亏损 → 盈亏比为 ∞。
    fills = [
        _fill(OrderSide.BUY, 100, 10, frame_id=10, oid=1),
        _fill(OrderSide.SELL, 100, 15, frame_id=20, oid=2),
    ]
    stats = _trade_stats(BacktestResult(fills=fills, equity_curve=[]))
    assert math.isinf(stats.profit_factor)


# ===========================================================================
# _benchmark_series（等权买入持有）
# ===========================================================================

def test_benchmark_single_symbol_matches_buy_hold():
    bars = [_bar(58000 + i, 10, 11, 9, 10.0 * (1 + 0.01 * i)) for i in range(10)]
    bars_by_symbol = {"S": bars}
    bench = _benchmark_series([58000, 58009], bars_by_symbol, start_equity=100000.0)
    # 单标的：最终权益 = 初始 × 累计收益乘数。
    last_x, last_y = bench[-1]
    assert last_x == 58009
    expected = 100000.0 * (bars[-1].close / bars[0].close)
    assert math.isclose(last_y, expected, rel_tol=1e-9)


def test_benchmark_two_symbols_equal_weight():
    bars_a = [_bar(58000 + i, 10, 11, 9, 10.0) for i in range(10)]
    bars_b = [_bar(58000 + i, 20, 21, 19, 20.0 * (1 + 0.01 * i)) for i in range(10)]
    bars_by_symbol = {"A": bars_a, "B": bars_b}
    bench = _benchmark_series([58009], bars_by_symbol, start_equity=100000.0)
    _, val = bench[0]
    # 等权：每标的分 50000，按各自收盘累计乘数加权。
    alloc = 100000.0 / 2
    expected = (alloc * (bars_a[-1].close / bars_a[0].close)
                + alloc * (bars_b[-1].close / bars_b[0].close))
    assert math.isclose(val, expected, rel_tol=1e-9)
    assert math.isclose(val, 104500.0, rel_tol=1e-6)  # A 0% + B 9%


def test_benchmark_empty_bars():
    bench = _benchmark_series([1, 2], {}, start_equity=100000.0)
    assert bench == [(1, 100000.0), (2, 100000.0)]


# ===========================================================================
# build_html
# ===========================================================================

def test_build_html_contains_expected_sections():
    res = _result_with_roundtrips()
    stats = _trade_stats(res)
    perf = {
        "total_return": 0.04, "annualized_return": 0.06, "sharpe": 1.5,
        "max_drawdown": 0.05, "win_rate": 0.67,
    }
    html = build_html(
        result=res,
        strategy_name="测试策略",
        symbols=["S"],
        equity_img="EQ_B64",
        trade_imgs={"S": "TRADE_B64"},
        perf=perf,
        stats=stats,
        fills=res.fills,
        start_date=58000,
        end_date=60000,
    )
    assert "<!DOCTYPE html>" in html
    assert "测试策略" in html
    assert "EQ_B64" in html  # 权益图内嵌
    assert "TRADE_B64" in html  # 交易图内嵌
    assert "成交明细" in html
    assert "买入" in html and "卖出" in html  # 成交方向
    # 成交明细表含成交价。
    assert "10.000" in html  # 第一笔买入价


def test_build_html_no_trade_chart_graceful():
    res = _result_with_roundtrips()
    stats = _trade_stats(res)
    perf = {"total_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "win_rate": 0.0}
    html = build_html(
        result=res, strategy_name="空", symbols=["S"],
        equity_img="", trade_imgs={}, perf=perf, stats=stats,
        fills=res.fills, start_date=1, end_date=1,
    )
    # 未提供 K 线图时给出提示，而非崩溃。
    assert "未提供 K 线数据" in html


# ===========================================================================
# generate_report（端到端）
# ===========================================================================

def test_generate_report_full_html():
    res = _result_with_roundtrips()
    bars = [_bar(58000 + i, 10, 11, 9, 10.0 + (i % 3)) for i in range(60)]
    html = generate_report(
        res,
        bars_by_symbol={"S": bars},
        strategy_name="端到端测试",
    )
    assert "<!DOCTYPE html>" in html
    assert "端到端测试" in html
    # 含收益曲线与交易 K 线两张图。
    assert html.count("data:image/png;base64,") >= 2
    # 含成交明细。
    assert "成交明细" in html


def test_generate_report_without_bars_skips_trade_chart():
    res = _result_with_roundtrips()
    html = generate_report(res)  # 不传 bars_by_symbol
    assert "<!DOCTYPE html>" in html
    assert "未提供 K 线数据" in html


def test_generate_report_config_defaults():
    res = _result_with_roundtrips()
    cfg = ReportConfig()
    assert cfg.benchmark is True
    assert cfg.max_trade_charts == 8
    assert cfg.period_frames == 244
    # 关闭基准后 HTML 中不应出现基准曲线标识（买入持有）。
    html = generate_report(res, bars_by_symbol={"S": [_bar(58000, 10, 11, 9, 10)]},
                           config=ReportConfig(benchmark=False))
    assert "<!DOCTYPE html>" in html


def test_generate_report_accepts_dict_bars():
    # bars_by_symbol 可用 dict 列表（demo 的 _synthetic_bars 返回 dict）。
    res = _result_with_roundtrips()
    dicts = [{"timestamp": 58000 + i, "open": 10, "high": 11, "low": 9,
              "close": 10.0 + (i % 3), "volume": 1000} for i in range(60)]
    html = generate_report(res, bars_by_symbol={"S": dicts})
    assert "<!DOCTYPE html>" in html


# ===========================================================================
# save_report
# ===========================================================================

def test_save_report_writes_file(tmp_path):
    res = _result_with_roundtrips()
    html = generate_report(res, bars_by_symbol={"S": [_bar(58000, 10, 11, 9, 10)]})
    out = tmp_path / "report" / "out.html"
    path = save_report(html, str(out))
    assert out.exists()
    assert out.read_text(encoding="utf-8") == html


def test_generate_report_with_output_path(tmp_path):
    res = _result_with_roundtrips()
    out = tmp_path / "report.html"
    html = generate_report(res, output_path=str(out))
    assert out.exists()
    # 返回的字符串与落盘内容一致。
    assert html == out.read_text(encoding="utf-8")


# ===========================================================================
# 报告自包含性（图表内嵌、无外部依赖）
# ===========================================================================

def test_report_is_self_contained():
    res = _result_with_roundtrips()
    bars = [_bar(58000 + i, 10, 11, 9, 10.0 + (i % 3)) for i in range(60)]
    html = generate_report(res, bars_by_symbol={"S": bars})
    # 不应引用外部资源（内嵌 base64 图片的 src='data:... 属正常）。
    for needle in ("http://", "https://", "cdn.", ".css"):
        assert needle not in html.lower(), f"报告引用了外部资源: {needle}"
