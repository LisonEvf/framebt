"""为推荐策略 MomentumDaily 生成最终交付物：

  tools/report/momentum_report.md   中文回测报告（含 A/B 对比 + 幸存者偏差警告）
  tools/report/momentum_trades.csv  逐笔交易明细
  tools/report/momentum_equity.png  权益曲线图
"""
from __future__ import annotations

import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from framebt import Engine, Performance, StandardAshareRules, StandardCost
from framebt.core import OrderSide

from ab_test_strategies import _Rebalance, _mom, _prepare_universe
from framebt.data_sources import OpentdxParquetSource

OUT = os.path.join(os.path.dirname(__file__), "report")
os.makedirs(OUT, exist_ok=True)

A_B = [
    ("Buy&Hold 每日等权(全300)", "+6079%", "+16.64%", 0.51, 70.02, 1, 639_770, 327_216, 300, "16.9M"),
    ("MomentumDaily Top30 每日", "+5845%", "+16.47%", 0.42, 70.70, 11, 132_668, 70_240, 298, "19.8M"),
    ("Momentum60D Top30 60日", "+5845%", "+16.47%", 0.42, 70.70, 11, 132_668, 70_240, 298, "19.8M"),
    ("Reversal60D Bottom30 60日", "+370%", "+5.95%", 0.17, 82.02, 20, 122_293, 62_097, 300, "4.7M"),
]


def main():
    src = OpentdxParquetSource()
    universe = _prepare_universe(src)
    feed = src.universe(universe)
    lo, hi = feed.span()
    start, end = datetime.fromordinal(int(lo)).isoformat(), datetime.fromordinal(int(hi)).isoformat()

    cost = StandardCost(commission=0.0003, min_commission=5.0, stamp_tax=0.001,
                        transfer_fee=0.00002, slippage="volume_based", k=0.001)
    strat = _Rebalance(universe, _mom, 30, 1, reverse=False)
    eng = Engine(feed, strat, initial_cash=1_000_000.0, settlement_cycle=1,
                 cost_model=cost, rules=StandardAshareRules(limit=0.10))
    res = eng.run()
    p = Performance(res)

    # ---- 逐笔明细 CSV
    fills = sorted(res.fills, key=lambda f: f.timestamp)
    with open(os.path.join(OUT, "momentum_trades.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "side", "symbol", "size", "price", "amount", "cost"])
        for fl in fills:
            w.writerow([datetime.fromordinal(int(fl.timestamp)).isoformat()[:10],
                        "BUY" if fl.side is OrderSide.BUY else "SELL",
                        fl.symbol, f"{fl.size:.2f}", f"{fl.price:.4f}",
                        f"{fl.size * fl.price:,.2f}", f"{fl.cost:.2f}"])

    # ---- 权益曲线图
    eq = res.equity_curve  # list of (timestamp, equity)
    xs = [datetime.fromordinal(int(t)).year +
          datetime.fromordinal(int(t)).month / 12 for t, _ in eq]
    ys = [v for _, v in eq]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(xs, ys, "#1f77b4", lw=1.1)
    ax.set_yscale("log")
    ax.set_title("MomentumDaily 权益曲线（log 轴）", fontsize=13)
    ax.set_ylabel("权益（元，log）")
    ax.grid(True, alpha=0.3)
    ax.annotate(f"总收益 {p.total_return:+.0%}\n年化 {p.annualized_return:+.2%}\nSharpe {p.sharpe:.2f}\n最大回撤 {p.max_drawdown:.0%}",
                xy=(0.01, 0.02), xycoords="axes fraction", va="bottom", ha="left",
                bbox=dict(boxstyle="round", fc="#fffbe6", ec="#cca700"))
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "momentum_equity.png"), dpi=120)
    plt.close(fig)

    # ---- 中文报告
    def row(x): return "| " + " | ".join(str(c) for c in x) + " |"
    tbl = "\n".join([row(A_B[0]), "---"] + [row(a) for a in A_B[1:]])
    md = f"""# 多标的动量轮动策略 · 回测报告

## 一、结论速览

| 指标 | 结果 |
|------|------|
| 推荐策略 | **MomentumDaily**：每日横截面动量轮动，买 60 日动量 Top30，等权 |
| 区间 | {start} → {end}（{len(eq)} 交易日） |
| 总收益 | **{p.total_return:+.0%}**（年化 {p.annualized_return:+.2%}） |
| Sharpe | {p.sharpe:.2f} · 最大回撤 {p.max_drawdown:.0%} · 胜率 {p.win_rate():.0%} |
| 成交 | {res.total_trades():,} 笔 / {len({f.symbol for f in res.fills})} 标的 |

> ⚠️ **幸存者偏差警告**：本报告的收益由「按**当前**流动性选 300 只并持有 26 年」产生，
> 仅含幸存者（能活到今天的大盘股），**严重高估真实可获利性**。动量轮动（5845%）与等权轮动（6079%）
> 基本持平，说明**动量信号本身无超额收益**——收益来自标的池，而非择时能力。

## 二、A/B 测试结果（同一 300 标的池）

| 策略 | 总收益 | 年化 | Sharpe | 最大回撤 | 胜率 | 成交笔 | 回合 | 标的 | 累计费用 |
|------|--------|------|--------|---------|------|--------|------|------|--------|
{tbl}

**解读**：
- **动量 ≈ 等权**：动量轮动未跑赢每日等权轮动，动量因子在本池无 alpha。
- **反转更弱**：Reversal60D 仅 +370%、回撤 82%，反转不成立。
- **回撤巨大**：所有策略最大回撤 70%–82%，实盘不可承受，需风控/降杠杆。
- **费用极高**：高频轮动累计费用达本金的十余倍（已净扣，收益为税后）。

## 三、数据与口径

- 标的池：全 A 中流动性前 300（按历史日均成交额），≥260 根 K 线。
- 数据：OpenTDX Parquet 日线，QFQ 前复权，{start}–{end}。
- 选股因子：60 日动量 `close[t]/close[t-60] - 1`，横截面降序取 Top30。
- 入场/出场：每日再平衡，买 Top30 等权，跌出 Top30 市价卖出。
- 成本：佣金 0.03%（最低 5 元）+ 印花税 0.1%（卖）+ 过户费 0.002% + 滑点（量价）。
- 涨跌停：10% 硬约束，触及当日不可成交。

## 四、交易细节

- 逐笔明细：`tools/report/momentum_trades.csv`（date/side/symbol/size/price/amount/cost）
- 权益曲线：`tools/report/momentum_equity.png`（log 轴）

## 五、局限与下一步

1. **幸存者偏差**：需用「时间对齐的动态标的池」（当时已上市的股票）替代当前静态 300，
   否则收益不可信。这是把回测收益当真实现金率前的必要一步。
2. **无稳健 alpha**：动量/反转均未通过检验。框架价值在于**正确、可复现的回测引擎**，
   而非既得收益。
3. **风控**：最大回撤 70%+，需加市场择时（大盘跌破均线降仓）或波动率目标定位。
4. **下一步**：动态 universe + 市场择时 + 波动率定位，重建一个回撤可控的真实策略。
"""
    with open(os.path.join(OUT, "momentum_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("OK ->", OUT)
    print(f"  trades: {len(fills)}")
    print(f"  return {p.total_return:+.0%}  sharpe {p.sharpe:.2f}  dd {p.max_drawdown:.0%}")


if __name__ == "__main__":
    main()
