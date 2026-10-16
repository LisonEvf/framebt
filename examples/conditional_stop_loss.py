"""示例 3：均线趋势 + 跟踪止损条件单（Phase 2 验收，§5.2）。

买入后由 ``on_fill`` 挂出「最新价跌破止损价自动卖出」的条件单。
条件单由引擎在帧间监控触发（§5.2），而非策略逐帧主动判断。

验证：
  - 条件单 resting 触发（LAST_LE）
  - on_fill 产生新订单并登记入在途（I2，延迟 1 帧）

运行：
    python examples/conditional_stop_loss.py
"""

from __future__ import annotations

from framebt import Performance, run_backtest
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import TrendWithStop


def main() -> None:
    src = OpentdxParquetSource()
    symbols = src.available_symbols()
    if not symbols:
        raise SystemExit("数据源无 K 线数据，请先运行 opentdx ETL 回填。")
    symbol = symbols[0]
    feed = src.load(symbol)
    print(f"标的 {symbol}：{len(feed.bars)} 根日线")

    bars = [
        {
            "timestamp": b.timestamp,
            "symbol": b.symbol,
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in feed.bars
    ]

    res = run_backtest(
        bars,
        TrendWithStop(symbol, ma_period=3, stop_pct=0.03),
        symbol=symbol,
        initial_cash=100_000.0,
    )
    perf = Performance(res, period_frames=244)

    print("\n=== 回测结果（趋势+跟踪止损）===")
    print(f"交易次数: {res.total_trades()}  最终权益: {res.final_equity():.2f}")
    print(f"总收益: {perf.total_return:.2%}  最大回撤: {perf.max_drawdown:.2%}  "
          f"Sharpe: {perf.sharpe:.2f}")
    print("\n=== 成交明细 ===")
    for f in res.fills:
        print(f"  帧 {f.frame_id:>3}  {f.side.value:>4}  {f.symbol}  "
              f"{f.size:.0f} @ {f.price:.2f}")


if __name__ == "__main__":
    main()
