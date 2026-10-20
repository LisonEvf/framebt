"""示例：中期动量轮动（本框架「选股 + 交易 + 资金」三合一的有效策略）。

思路（针对上一轮「只交易 ~5 个标的」过度集中的教训）：
  1. 充分分散 —— 用横截面动量 Top80 作为可交易池，活跃集合始终够大，
     Kelly 的 `available/n` 自动把资金摊薄到 80 个标的，而非少数押注；
  2. 有 edge —— 60 日中期动量排序 + MA20 趋势入场 + 5% 硬止损 + 均线离场；
  3. 保守资金 —— 半凯利，预热期 warmup_fraction=0.5（不是 1.0），
     样本不足时不过度下注；
  4. 等权分配 —— 活跃标的之间等权投入。

对比基准：等权买入持有（同一建仓时点）。

运行：
    python examples/rotation_momentum.py

会打印回测报告（绩效指标 + 基准对比 + 交易明细），并输出：
    tools/report/momentum_report.md
    tools/report/momentum_trades.csv
    tools/report/momentum_equity.png
"""

from __future__ import annotations

import datetime
import os

import pandas as pd

from framebt import (
    Engine, Performance, StandardAshareRules, StandardCost,
)
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import RotationStrategy, TrendWithStop
from framebt.selection import CrossSectionRankSelector, KellyBudgetPolicy

TOP_K = 80              # 动量可交易池大小（够大 → 分散）
MOM_PERIOD = 60         # 动量窗口（日）
MA_PERIOD = 20          # 趋势均线（日）
STOP_PCT = 0.05         # 硬止损
KELY_FRACTION = 0.5     # 半凯利
KELY_MIN_SAMPLES = 15   # 样本不足前用 warmup
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "tools", "report")


def _prepare_universe(src, max_symbols: int = 300, min_bars: int = 260):
    """按成交额中位数取前 ``max_symbols`` 个流动性标的（保证可交易与分散）。"""
    universe = src.available_symbols()
    feed = src.universe(universe)
    counts, amounts = {}, {}
    for b in feed.bars:
        counts[b.symbol] = counts.get(b.symbol, 0) + 1
        amounts[b.symbol] = amounts.get(b.symbol, 0.0) + b.amount
    avg_amt = {s: a / max(1, counts[s]) for s, a in amounts.items()}
    eligible = [s for s in universe if counts.get(s, 0) >= min_bars]
    eligible.sort(key=lambda s: avg_amt.get(s, 0.0), reverse=True)
    return eligible[:max_symbols]


def _round_trips(fills):
    """把 fills 配对成「每标的完整回合」：entry/exit 日期·价格·金额·盈亏。"""
    from framebt.core.events import OrderSide

    open_notional, open_side, rows = {}, {}, []
    for f in fills:
        if f.side is OrderSide.BUY:
            open_notional[f.symbol] = open_notional.get(f.symbol, 0.0) + f.size * f.price
            open_side[f.symbol] = OrderSide.BUY
            rows.append(("BUY", f.symbol, f.timestamp, f.size, f.price, f.cost))
        else:
            if open_side.get(f.symbol) is OrderSide.BUY:
                pnl = f.size * f.price - open_notional.pop(f.symbol, 0.0)
                rows.append(("SELL", f.symbol, f.timestamp, f.size, f.price, 0.0))
            else:
                rows.append(("SELL", f.symbol, f.timestamp, f.size, f.price, 0.0))
    return rows


def _fmt_date(ts):
    # 数仓 timestamp 为格里高利历序号（toordinal），用 fromordinal 反解
    return datetime.date.fromordinal(int(ts)).isoformat()


def main() -> None:
    src = OpentdxParquetSource()
    universe = _prepare_universe(src)
    if not universe:
        raise SystemExit("数据源无足够 K 线数据；请先运行 tools/backfill_history.py")
    feed = src.universe(universe)
    span_lo, span_hi = feed.span()

    cost = StandardCost(commission=0.0003, min_commission=5.0,
                        stamp_tax=0.001, transfer_fee=0.00002,
                        slippage="volume_based", k=0.001)

    selector = CrossSectionRankSelector(factor_name=f"mom{MOM_PERIOD}",
                                        key="close", period=MOM_PERIOD, top_k=TOP_K)
    budget = KellyBudgetPolicy(fraction=KELY_FRACTION, min_samples=KELY_MIN_SAMPLES,
                               warmup_fraction=0.5)
    rot = RotationStrategy(
        universe, selector,
        make=lambda s: TrendWithStop(s, ma_period=MA_PERIOD, stop_pct=STOP_PCT),
        budget_policy=budget,
        rotate=True,  # 跌出动量池即卖出 → 真轮动（买新上榜 / 卖跌出池）
    )
    eng = Engine(feed, rot, initial_cash=1_000_000.0,
                 settlement_cycle=1, cost_model=cost, rules=StandardAshareRules(limit=0.10))
    res = eng.run()
    perf = Performance(res)
    fees = sum(f.cost for f in res.fills)

    # ---- 回合盈亏（用于胜率 / 盈亏比 / 盈利因子）----
    from framebt.core.events import OrderSide
    pnls, wins, losses = [], [], []
    open_n, open_s = {}, {}
    for f in res.fills:
        if f.side is OrderSide.BUY:
            open_n[f.symbol] = open_n.get(f.symbol, 0.0) + f.size * f.price
            open_s[f.symbol] = OrderSide.BUY
        elif open_s.get(f.symbol) is OrderSide.BUY:
            pnl = f.size * f.price - open_n.pop(f.symbol, 0.0)
            pnls.append(pnl)
            if pnl > 0:
                wins.append(pnl)
            else:
                losses.append(pnl)
    profit_factor = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    calmar = (perf.annualized_return / abs(perf.max_drawdown)) if perf.max_drawdown > 0 else float("inf")

    held = sorted({f.symbol for f in res.fills})
    n_years = max(1e-9, res.frames / 252.0)

    # ================= 报告正文 =================
    print("=" * 68)
    print("中期动量轮动 · 回测报告")
    print("=" * 68)
    print(f"选股池    : {len(universe)} 个流动性标的（成交额前 300，≥260 日线）")
    print(f"样本区间  : {_fmt_date(span_lo)} .. {_fmt_date(span_hi)}  （{res.frames:,} 根日线，{n_years:.1f} 年）")
    print(f"策略逻辑  : 动量 Top{TOP_K}（{MOM_PERIOD} 日）→ MA{MA_PERIOD} 趋势入场 → {STOP_PCT:.0%} 止损 → 等权 + 半凯利")
    print(f"成交笔数  : {res.total_trades():,}（{len(pnls)} 个完整回合）")
    print(f"累计费用  : {fees:,.0f}  （{fees / 1_000_000:.2%} 初始资金）")
    print("-" * 68)
    print(f"最终权益    : {res.final_equity():,.0f}")
    print(f"总收益      : {perf.total_return:+.2%}")
    print(f"年化收益    : {perf.annualized_return:+.2%}")
    print(f"波动率      : {perf.volatility:.2%}")
    print(f"最大回撤    : {perf.max_drawdown:.2%}")
    print(f"Sharpe      : {perf.sharpe:.2f}")
    print(f"Calmar      : {calmar:.2f}")
    print(f"胜率        : {perf.win_rate():.1%}   盈亏比: {((sum(wins) / len(wins)) / (abs(sum(losses) / len(losses))) if wins and losses else 0):.2f}   盈利因子: {profit_factor:.2f}")
    print(f"持仓标的    : {len(held)} 个（历史过）")
    print("-" * 68)

    # 基准：等权买入持有（第 MOM_PERIOD 帧收盘 → 末帧收盘，不计成本）
    bench = []
    for sym in universe:
        bars = src.load(sym).bars
        if len(bars) > MOM_PERIOD + 5:
            bench.append(bars[-1].close / bars[MOM_PERIOD].close - 1)
    bh = sum(bench) / len(bench) if bench else 0.0
    print("基准对比    : 等权买入持有 {:+.2%}（{d} 标的）  超额 {:+.2%}".format(
        bh, perf.total_return - bh, d=len(bench)))
    print("=" * 68)

    # ================= 交易明细 =================
    os.makedirs(REPORT_DIR, exist_ok=True)
    rows = _round_trips(res.fills)
    trades_path = os.path.join(REPORT_DIR, "momentum_trades.csv")
    with open(trades_path, "w", encoding="utf-8") as fh:
        fh.write("side,symbol,date,size,price,cost\n")
        for side, sym, ts, size, price, cost_ in rows:
            fh.write(f"{side},{sym},{_fmt_date(ts)},{size:.0f},{price:.4f},{cost_:.2f}\n")

    equity_path = os.path.join(REPORT_DIR, "momentum_equity.png")
    _plot_equity(res, bh, equity_path)

    report_path = os.path.join(REPORT_DIR, "momentum_report.md")
    _write_markdown(report_path, locals())

    print(f"\n已输出：")
    print(f"  报告  : {os.path.abspath(report_path)}")
    print(f"  成交  : {os.path.abspath(trades_path)}（{len(rows)} 行）")
    print(f"  权益图 : {os.path.abspath(equity_path)}")


def _plot_equity(res, bh, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eq = [e for _, e in res.equity_curve]
    frames = [i for i, _ in res.equity_curve]
    peak = [max(eq[: i + 1]) for i in range(len(eq))]
    draw = [(p - e) / p if p > 0 else 0.0 for p, e in zip(peak, eq)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(frames, eq, lw=1.1, color="#1f77b4", label="策略权益")
    ax1.axhline(1_000_000, ls="--", lw=0.8, color="#888")
    ax1.set_ylabel("权益（元）")
    ax1.legend(loc="upper left")
    ax1.set_title("中期动量轮动 · 权益曲线")

    ax2.plot(frames, [d * 100 for d in draw], color="#d62728", lw=0.8)
    ax2.fill_between(frames, [d * 100 for d in draw], 0, color="#d62728", alpha=0.2)
    ax2.set_ylabel("回撤 (%)")
    ax2.axhline(0, color="#333", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _write_markdown(path, ctx):
    perf = ctx["perf"]
    rows = _round_trips(ctx["res"].fills)
    pnls, wins, losses = [], [], []
    open_n, open_s = {}, {}
    for f in ctx["res"].fills:
        if f.side.value == "buy":
            open_n[f.symbol] = open_n.get(f.symbol, 0.0) + f.size * f.price
            open_s[f.symbol] = "buy"
        elif open_s.get(f.symbol) == "buy":
            pnl = f.size * f.price - open_n.pop(f.symbol, 0.0)
            pnls.append(pnl)
            (wins if pnl > 0 else losses).append(pnl)
    profit_factor = (sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else float("inf")
    calmar = (perf.annualized_return / abs(perf.max_drawdown)) if perf.max_drawdown > 0 else float("inf")

    lines = [
        f"# 中期动量轮动 · 回测报告",
        "",
        f"- 样本区间：{_fmt_date(ctx['span_lo'])} .. {_fmt_date(ctx['span_hi'])}（{ctx['res'].frames:,} 根日线，{ctx['res'].frames/252:.1f} 年）",
        f"- 选股池：{len(ctx['universe'])} 个流动性标的",
        f"- 策略：动量 Top{TOP_K}（{MOM_PERIOD}日）→ MA{MA_PERIOD} 入场 → {STOP_PCT:.0%}止损 → 等权+半凯利",
        f"- 成交：{ctx['res'].total_trades():,} 笔（{len(pnls)} 回合）｜费用 {sum(f.cost for f in ctx['res'].fills):,.0f}",
        "",
        "## 绩效",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 最终权益 | {ctx['res'].final_equity():,.0f} |",
        f"| 总收益 | {perf.total_return:+.2%} |",
        f"| 年化收益 | {perf.annualized_return:+.2%} |",
        f"| 波动率 | {perf.volatility:.2%} |",
        f"| 最大回撤 | {perf.max_drawdown:.2%} |",
        f"| Sharpe | {perf.sharpe:.2f} |",
        f"| Calmar | {calmar:.2f} |",
        f"| 胜率 | {perf.win_rate():.1%} |",
        f"| 盈利因子 | {profit_factor:.2f} |",
        f"| 持仓标的 | {len(ctx['held'])} |",
        "",
        f"完整成交明细见 [`momentum_trades.csv`](momentum_trades.csv)，权益曲线见 [`momentum_equity.png`](momentum_equity.png)。",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"[report] wrote {path}")


if __name__ == "__main__":
    main()
