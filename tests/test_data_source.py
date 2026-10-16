"""数据源测试（§12）：opentdx 数仓 Parquet 读取。

包含两部分：
  1. 合成 Parquet 夹具 —— 不依赖真实数仓，验证 schema → BarFeed 映射；
  2. 真实数仓冒烟 —— 数仓有数据时才运行。
"""

from __future__ import annotations

import math

import pytest

from framebt import BarFeed, OpentdxParquetSource, Performance, run_backtest
from framebt.strategies import DualMAStrategy


def _write_kline(data_dir, symbol_key: str, rows, *, period: int = 4, adjust: int = 1):
    """按 opentdx 数仓 schema 写一个 K 线 Parquet 文件。

    默认写 DAILY(4) + QFQ(1) —— 与 ``OpentdxParquetSource`` 的默认口径一致。
    """
    pd = pytest.importorskip("pandas")
    market, code = symbol_key.split(".")
    market_code = {"SZ": 0, "SH": 1, "BJ": 2}.get(market, 0)
    path = data_dir / "fakt_kline" / str(period) / str(adjust)
    path.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "datetime": pd.to_datetime([r[0] for r in rows]),
            "open": [r[1] for r in rows],
            "high": [r[2] for r in rows],
            "low": [r[3] for r in rows],
            "close": [r[4] for r in rows],
            "vol": [r[5] for r in rows],
            "amount": [r[1] * r[5] for r in rows],
            "float_shares": [1e9] * len(rows),
            "turnover": [0.1] * len(rows),
            "symbol_key": [symbol_key] * len(rows),
            "period": [period] * len(rows),
            "adjust": [adjust] * len(rows),
            "dt": pd.to_datetime([r[0] for r in rows]),
        }
    )
    df.to_parquet(path / f"{market_code}_{code}.parquet")


ROWS = [
    ("2026-01-05", 10.0, 10.8, 9.8, 10.5, 1000.0),
    ("2026-01-06", 10.5, 11.2, 10.3, 11.0, 1200.0),
    ("2026-01-07", 11.0, 11.5, 10.6, 10.8, 900.0),
]


def test_source_reads_synthetic_parquet(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS)
    src = OpentdxParquetSource(data_dir=tmp_path)
    assert src.available_symbols() == ["SH.600000"]

    feed = src.load("SH.600000")
    assert isinstance(feed, BarFeed)
    assert len(feed.bars) == 3
    assert feed.bars[0].open == 10.0
    assert feed.bars[0].high == 10.8
    assert feed.bars[0].low == 9.8
    assert feed.bars[0].close == 10.5
    assert feed.bars[0].volume == 1000.0
    assert feed.bars[0].symbol == "SH.600000"
    # timestamp 使用日期序数，严格递增
    assert feed.bars[0].timestamp < feed.bars[1].timestamp < feed.bars[2].timestamp


def test_source_missing_symbol_raises(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS)
    src = OpentdxParquetSource(data_dir=tmp_path)
    with pytest.raises(KeyError):
        src.load("SZ.999999")


def test_source_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        OpentdxParquetSource(data_dir=tmp_path / "does_not_exist")


def test_source_date_range_filter(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS)
    src = OpentdxParquetSource(data_dir=tmp_path)
    feed = src.load("SH.600000", start="2026-01-06", end="2026-01-06")
    assert len(feed.bars) == 1
    assert feed.bars[0].close == 11.0


# ---------------------------------------------------------------------------
# 复权口径过滤（regression）
#
# 数仓内同一标的可能同时存在 NONE/QFQ/HFQ 多个文件；若不过滤会把不同口径的
# Bar 混成一条序列（曾出现「408 根但只有 400 个唯一时间戳」的静默污染）。
# ---------------------------------------------------------------------------
def test_source_filters_by_adjust_no_mixing(tmp_path):
    # 同一标的：不复权 3 根（收盘 10.x）与前复权 3 根（收盘 100.x），日期相同
    none_rows = [(d, o, h, l, c, v) for d, o, h, l, c, v in ROWS]
    qfq_rows = [(d, o * 10, h * 10, l * 10, c * 10, v) for d, o, h, l, c, v in ROWS]
    _write_kline(tmp_path, "SH.600000", none_rows, adjust=0)
    _write_kline(tmp_path, "SH.600000", qfq_rows, adjust=1)

    qfq = OpentdxParquetSource(data_dir=tmp_path, adjust="QFQ")
    assert qfq.available_symbols() == ["SH.600000"]
    fb = qfq.load("SH.600000")
    assert len(fb.bars) == 3, "不得把另一复权口径的 Bar 混进来"
    assert len({b.timestamp for b in fb.bars}) == 3, "时间戳必须唯一"
    assert fb.bars[0].close == 105.0, "应取 QFQ 口径"

    none = OpentdxParquetSource(data_dir=tmp_path, adjust="NONE")
    nb = none.load("SH.600000")
    assert len(nb.bars) == 3
    assert nb.bars[0].close == 10.5, "应取 NONE 口径"

    # 单次调用覆盖口径
    assert qfq.load("SH.600000", adjust="NONE").bars[0].close == 10.5
    assert qfq.available_adjusts("SH.600000") == ["NONE", "QFQ"]


def test_source_missing_adjust_raises(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS, adjust=0)
    src = OpentdxParquetSource(data_dir=tmp_path, adjust="QFQ")
    with pytest.raises(KeyError):
        src.load("SH.600000")   # 该标的没有 QFQ 口径


def test_source_unknown_period_adjust_raises(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS)
    with pytest.raises(ValueError):
        OpentdxParquetSource(data_dir=tmp_path, period="NOPE")
    with pytest.raises(ValueError):
        OpentdxParquetSource(data_dir=tmp_path, adjust="NOPE")


def test_source_filters_by_period(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS, period=4, adjust=1)   # 日线
    _write_kline(tmp_path, "SH.600000", ROWS, period=5, adjust=1)   # 周线
    daily = OpentdxParquetSource(data_dir=tmp_path, period="DAILY", adjust="QFQ")
    assert len(daily.load("SH.600000").bars) == 3
    weekly = OpentdxParquetSource(data_dir=tmp_path, period="WEEKLY", adjust="QFQ")
    assert len(weekly.load("SH.600000").bars) == 3


def test_source_universe_merges_symbols(tmp_path):
    _write_kline(tmp_path, "SH.600000", ROWS)
    _write_kline(tmp_path, "SZ.000001", [(d, o + 1, h + 1, l + 1, c + 1, v) for d, o, h, l, c, v in ROWS])
    src = OpentdxParquetSource(data_dir=tmp_path)
    feed = src.universe(["SH.600000", "SZ.000001"])
    assert set(feed.symbols()) == {"SH.600000", "SZ.000001"}
    assert len(feed.bars) == 6
    # 合并后仍按 timestamp 排序
    ts = [b.timestamp for b in feed.bars]
    assert ts == sorted(ts)


def test_source_backtest_integration(tmp_path):
    """合成 Parquet → 数据源 → 引擎 → 绩效，端到端。"""
    _write_kline(tmp_path, "SH.600000", ROWS)
    src = OpentdxParquetSource(data_dir=tmp_path)
    feed = src.load("SH.600000")
    bars = [
        {"timestamp": b.timestamp, "symbol": b.symbol, "open": b.open, "high": b.high,
         "low": b.low, "close": b.close, "volume": b.volume}
        for b in feed.bars
    ]
    res = run_backtest(bars, DualMAStrategy("SH.600000", short=1, long=2),
                       symbol="SH.600000", initial_cash=100000.0)
    assert res.frames >= 3
    assert res.final_equity() > 0


# ---------------------------------------------------------------------------
# 真实数仓冒烟（有数据才跑）
# ---------------------------------------------------------------------------
def _real_available() -> bool:
    try:
        return len(OpentdxParquetSource().available_symbols()) > 0
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _real_available(), reason="opentdx 数仓无 K 线数据")
def test_real_warehouse_loads_and_backtests():
    src = OpentdxParquetSource()
    symbols = src.available_symbols()
    symbol = symbols[0]
    feed = src.load(symbol)
    assert len(feed.bars) > 0

    bars = [
        {"timestamp": b.timestamp, "symbol": b.symbol, "open": b.open, "high": b.high,
         "low": b.low, "close": b.close, "volume": b.volume}
        for b in feed.bars
    ]
    res = run_backtest(bars, DualMAStrategy(symbol, short=1, long=2),
                       symbol=symbol, initial_cash=100000.0)
    assert res.frames > 0
    assert all(eq >= 0 for _k, eq in res.equity_curve)
    perf = Performance(res, period_frames=244)
    assert math.isfinite(perf.max_drawdown)
