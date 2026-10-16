"""示例 1：真实数据双均线交叉回测（Phase 0 验收）。

用 opentdx 数仓的日线数据，对单标的跑双均线交叉策略，打印成交明细与绩效。

运行：
    python examples/dual_ma_real.py

若数仓无数据，先运行 opentdx 的 ETL 回填（见 opentdx/README）。
"""

from __future__ import annotations

from framebt import Performance, run_backtest
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import DualMAStrategy


def main() -> None:
    src = OpentdxParquetSource()
    symbols = src.available_symbols()
    if not symbols:
        raise SystemExit("数据源无 K 线数据，请先运行 opentdx ETL 回填。")
    symbol = symbols[0]
    print(f"数据源可用标的：{symbols}")

    feed = src.load(symbol)
    print(f"标的 {symbol}：{len(feed.bars)} 根日线，"
          f"{feed.bars[0].timestamp} ~ {feed.bars[-1].timestamp}")

    # 数仓为「自然落库」样例，样本较短；窗口取小以便产生交叉信号。
    # 真实回测请先用 opentdx ETL 回填更长历史（见 opentdx/db/README.md）。
    short, long = 2, 4
    if len(feed.bars) <= long:
        raise SystemExit(
            f"数据仅 {len(feed.bars)} 根，不足以计算 MA{long}；"
            "请先用 opentdx ETL 回填更长历史。"
        )
    print(f"使用窗口 MA{short}/MA{long}（样本较短，仅作管线演示）")

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
        DualMAStrategy(symbol, short=short, long=long),
        symbol=symbol,
        initial_cash=100_000.0,
    )
    perf = Performance(res, period_frames=244)

    print("\n=== 回测结果 ===")
    print(f"标的: {symbol}  窗口: {short}/{long}  初始资金: 100000.00")
    print(f"交易次数: {res.total_trades()}  最终权益: {res.final_equity():.2f}")
    print(f"总收益: {perf.total_return:.2%}  最大回撤: {perf.max_drawdown:.2%}  "
          f"Sharpe: {perf.sharpe:.2f}  胜率: {perf.win_rate():.0%}")

    print("\n=== 成交明细 ===")
    for f in res.fills:
        print(f"  帧 {f.frame_id:>3}  {f.side.value:>4}  {f.symbol}  "
              f"{f.size:.0f} @ {f.price:.2f}")


if __name__ == "__main__":
    main()
