"""示例 4：Bracket（入场 + OCO 止盈/止损，§5.1）。

入场成交后在 ``on_fill`` 中提交 OCO 组（止盈限价 + 止损条件单），
任一腿成交即自动撤销另一腿。

运行：
    python examples/bracket_oco.py
"""

from __future__ import annotations

from framebt import Performance, run_backtest
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import BracketStrategy


def main() -> None:
    src = OpentdxParquetSource()
    symbols = src.available_symbols()
    if not symbols:
        raise SystemExit("数据源无 K 线数据，请先运行 opentdx ETL 回填。")
    symbol = symbols[0]
    feed = src.load(symbol)
    print(f"标的 {symbol}：{len(feed.bars)} 根日线")

    bars = [
        {"timestamp": b.timestamp, "symbol": b.symbol, "open": b.open, "high": b.high,
         "low": b.low, "close": b.close, "volume": b.volume}
        for b in feed.bars
    ]

    res = run_backtest(
        bars,
        BracketStrategy(symbol, ma_period=2, take_profit_pct=0.02, stop_loss_pct=0.02),
        symbol=symbol,
        initial_cash=100_000.0,
    )
    perf = Performance(res, period_frames=244)

    print("\n=== 回测结果（Bracket）===")
    print(f"交易次数: {res.total_trades()}  最终权益: {res.final_equity():.2f}")
    print(f"总收益: {perf.total_return:.2%}  最大回撤: {perf.max_drawdown:.2%}")
    print("\n=== 成交明细 ===")
    for f in res.fills:
        print(f"  帧 {f.frame_id:>3}  {f.side.value:>4}  {f.symbol}  "
              f"{f.size:.0f} @ {f.price:.2f}")


if __name__ == "__main__":
    main()
