"""示例 2：横截面动量选股（§4.5 / §4.3，框架差异化能力）。

在**选股池代表子集**（默认前 200 只，`--max-symbols` 可调）上，每帧用动量因子横截面
排名买入 TopK，并对比等权买入持有基准。全市场 5575 只的横截面是小时级，故默认降采样。
演示：
  - ``bus.cross_section`` 横截面请求（§4.5）
  - ``lag=1`` 排除当前帧，防同帧泄漏（§4.3）
  - 多标的 BarFeed 合并 + A 股成本/T+1/涨跌停（§7/§8）

运行：
    python examples/multi_factor_universe.py
"""

from __future__ import annotations

import argparse

from framebt import (
    Engine,
    Performance,
    StandardAshareRules,
    StandardCost,
)
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import MomentumCrossSection

TOPK = 5
LOOKBACK_SHORT, LOOKBACK_LONG = 20, 60


def main() -> None:
    ap = argparse.ArgumentParser(description="横截面动量选股示例")
    ap.add_argument("--max-symbols", type=int, default=200,
                    help="选股池上限（默认 200；全市场 5575 只会把示例跑成小时级）")
    args = ap.parse_args()

    src = OpentdxParquetSource()
    all_syms = src.available_symbols()
    if not all_syms:
        raise SystemExit("数据源无 K 线数据；请先运行 tools/backfill_history.py")
    universe = all_syms[: max(10, args.max_symbols)]
    print(f"选股池上限: {len(universe)} / 全市场 {len(all_syms)}"
          f"（--max-symbols 可调；全市场横截面是小时级）")

    feed = src.universe(universe)
    span_lo, span_hi = feed.span()
    print(f"选股池  : {len(universe)} 个标的")
    print(f"样本    : {len(feed.bars):,} 根日线（逻辑时间 {span_lo}..{span_hi}）")
    print(f"策略    : 动量 Top{TOPK}，窗口 MA{LOOKBACK_SHORT}/MA{LOOKBACK_LONG}，"
          f"每次调仓等权分配现金")

    cost = StandardCost(commission=0.0003, min_commission=5.0,
                        stamp_tax=0.001, transfer_fee=0.00002,
                        slippage="volume_based", k=0.001)

    eng = Engine(
        feed,
        MomentumCrossSection(universe, LOOKBACK_SHORT, LOOKBACK_LONG, TOPK),
        initial_cash=1_000_000.0,
        settlement_cycle=1,                       # T+1
        cost_model=cost,
        rules=StandardAshareRules(limit=0.10),    # A 股涨跌停
    )
    res = eng.run()
    perf = Performance(res)
    fees = sum(f.cost for f in res.fills)

    print("\n=== 策略结果 ===")
    print(f"帧数        : {res.frames:,}")
    print(f"成交笔数    : {res.total_trades():,}")
    print(f"累计费用    : {fees:,.0f}  ({fees / 1_000_000:.2%} 初始资金)")
    print(f"最终权益    : {res.final_equity():,.2f}")
    print(f"总收益      : {perf.total_return:+.2%}")
    print(f"年化收益    : {perf.annualized_return:+.2%}")
    print(f"最大回撤    : {perf.max_drawdown:.2%}")
    print(f"Sharpe      : {perf.sharpe:.2f}")

    # --- 基准：等权买入持有（同一建仓时点，不计成本）---
    bench = []
    for sym in universe:
        bars = src.load(sym).bars
        if len(bars) > LOOKBACK_LONG + 5:
            bench.append(bars[-1].close / bars[LOOKBACK_LONG].close - 1)
    bh = sum(bench) / len(bench) if bench else 0.0
    print("\n=== 基准对比 ===")
    print(f"等权买入持有: {bh:+.2%}  （{len(bench)} 标的，第 {LOOKBACK_LONG} 帧收盘 → 末帧收盘，不计成本）")
    print(f"超额        : {perf.total_return - bh:+.2%}")

    held = sorted({f.symbol for f in res.fills})
    print(f"\n历史持仓标的({len(held)}): {', '.join(held)}")


if __name__ == "__main__":
    main()
