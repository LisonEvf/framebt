"""示例：趋势选股 + 双均线轮动（§17）。

演示「完整的策略 = 选股 + 交易」，而不是在一个标的上死磕：

  1. 先选股：``TrendPresenceSelector`` 只在「收盘价站上 60 日均线」的标的里交易，
     自动规避下跌趋势——这是「挑选合适的标的」；
  2. 再交易：对每个选中标委托 ``DualMAStrategy`` 双均线交叉；
  3. 资金：``KellyBudgetPolicy`` 用策略即时胜率（凯利公式）决定总投入比例，
     并在活跃标的间等权分配。

用 ``RotationStrategy`` 把单标的的 DualMA 包装成多标的轮动策略。

运行：
    python examples/rotation_trend_ma.py

注：默认在「有足够历史、数量上限 ``MAX_SYMBOLS``」的标的子集上运行（演示速度）；
把 ``MAX_SYMBOLS`` 调大即可在全池 ``OpentdxParquetSource().available_symbols()`` 上跑。
"""

from __future__ import annotations

from framebt import (
    Engine, Performance, StandardAshareRules, StandardCost,
)
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import RotationStrategy, DualMAStrategy
from framebt.selection import TrendPresenceSelector, KellyBudgetPolicy


LOOKBACK_SHORT, LOOKBACK_LONG = 5, 20
MAX_SYMBOLS = 300      # 演示标的上限；全池运行调大即可
MIN_BARS = 260         # 至少这么多日线才纳入（保证 MA60 / 预热有数据）


class _All:
    """就地实现一个「全选」选择器（等价于 FixedUniverseSelector），避免额外导入。"""

    universe = None

    def select(self, frame, token, bus, host, universe):
        return list(universe)


def _prepare_universe(src, max_symbols: int, min_bars: int):
    """取有足够历史的标的子集（按成交额中位数排序取前 ``max_symbols``，保证流动性）。"""
    universe = src.available_symbols()
    feed = src.universe(universe)

    counts: dict[str, int] = {}
    amounts: dict[str, float] = {}
    for b in feed.bars:
        counts[b.symbol] = counts.get(b.symbol, 0) + 1
        amounts[b.symbol] = amounts.get(b.symbol, 0.0) + b.amount
    avg_amount = {s: a / max(1, counts[s]) for s, a in amounts.items()}

    eligible = [s for s in universe if counts.get(s, 0) >= min_bars]
    eligible.sort(key=lambda s: avg_amount.get(s, 0.0), reverse=True)
    return eligible[:max_symbols]


def run(universe, *, selector, label) -> None:
    feed = src.universe(universe)
    cost = StandardCost(commission=0.0003, min_commission=5.0,
                        stamp_tax=0.001, transfer_fee=0.00002,
                        slippage="volume_based", k=0.001)
    rot = RotationStrategy(
        universe, selector,
        make=lambda s: DualMAStrategy(s, short=LOOKBACK_SHORT, long=LOOKBACK_LONG),
        budget_policy=KellyBudgetPolicy(fraction=0.5, min_samples=20),
    )
    eng = Engine(
        feed, rot, initial_cash=1_000_000.0,
        settlement_cycle=1, cost_model=cost, rules=StandardAshareRules(limit=0.10),
    )
    res = eng.run()
    perf = Performance(res)
    held = rot.held_symbols()

    print(f"\n=== {label} ===")
    print(f"选股池    : {len(universe)} 个标的")
    print(f"轮动交易  : {len(held)} 个标的（历史过）")
    print(f"成交笔数  : {res.total_trades():,}")
    print(f"总收益    : {perf.total_return:+.2%}")
    print(f"年化      : {perf.annualized_return:+.2%}")
    print(f"最大回撤  : {perf.max_drawdown:.2%}")
    print(f"Sharpe    : {perf.sharpe:.2f}")


def main() -> None:
    global src
    src = OpentdxParquetSource()
    universe = _prepare_universe(src, MAX_SYMBOLS, MIN_BARS)
    if not universe:
        raise SystemExit("数据源无足够 K 线数据；请先运行 tools/backfill_history.py")

    span_lo, span_hi = src.universe(universe).span()
    print(f"选股池规模：{len(universe)} 个标的（逻辑时间 {span_lo}..{span_hi}）")
    print(f"交易逻辑  : 双均线 MA{LOOKBACK_SHORT}/MA{LOOKBACK_LONG}（单标的）")
    print(f"资金策略  : 半凯利 + 即时胜率")

    # 基准 A：不选股，整个池子轮动
    run(universe, selector=_All(), label="基准：全池轮动（不选股）")

    # 方案 B：只趋势向上选股
    run(universe, selector=TrendPresenceSelector(period=60, min_ratio=1.0),
        label="方案：趋势选股（价 > MA60）+ 双均线轮动")

    print("\n解读：两种方案仅「选股」不同，其余（双均线交易 + 半凯利资金）完全一致，"
          "差异即选股能力的边际贡献。"
          "注意：单标的策略（如本例 DualMA）在轮动下资金由凯利按活跃标的等权分配，"
          "若趋势选择器在某一阶段活跃集合过小，资金会相对集中——"
          "这是「选多少标的」与「资金如何分配」的耦合，正是本框架把这两件事显式拆开、可各自调优的原因。")


if __name__ == "__main__":
    main()
