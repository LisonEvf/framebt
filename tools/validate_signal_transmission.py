"""信号传导对照实验（修正版）：同一套代码，只翻转排序方向。

为什么单独有这个脚本
--------------------
旧版 ``validate_framework.py`` 的 6d 节有两个缺陷：
  1. ``RankStrategy`` 每帧只看持仓下单，不看 ``host.has_pending``；在 I2 延迟下会
     重复提交同一批买单。
  2. 默认 ``NoOpRisk`` 不拦超买，重复买单成交后现金可为负 —— 等于无息杠杆，
     会把收益放大到 ``+235,360%`` / ``-102%`` 这种不可解释的量级。

本脚本做三件事：
  - 修掉重复下单（买卖都检查在途单）；
  - 由成交明细独立重算「期末现金 / 期间最低现金」，负现金直接标红；
  - 把结果写成可复核的 Markdown。

定位：这是**机制合理性检查**，不是收益能力证明。样本是降采样的（默认 100 只），
且数据源存在存活偏差；不要引用绝对收益做业绩宣传。

用法：
    python tools/validate_signal_transmission.py
    python tools/validate_signal_transmission.py --max-symbols 200 --md reports/SIGNAL_TRANSMISSION.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

from framebt import Engine, Performance
from framebt.data_sources import OpentdxParquetSource

try:  # 直接 `python tools/xxx.py` 时 sys.path[0] 就是 tools/
    from validate_framework import RankStrategy, cash_path
except ImportError:  # pragma: no cover - 以模块方式运行时的兜底
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validate_framework import RankStrategy, cash_path

INITIAL_CASH = 1_000_000.0


def _benchmark(src: OpentdxParquetSource, syms: list[str]) -> float:
    rets = []
    for s in syms:
        bars = src.load(s).bars
        if len(bars) > 80:
            rets.append(bars[-1].close / bars[60].close - 1.0)
    return sum(rets) / len(rets) if rets else 0.0


def _pure_stats(src: OpentdxParquetSource, syms: list[str]) -> dict:
    pairs = []
    for s in syms:
        closes = [b.close for b in src.load(s).bars]
        for i in range(60, len(closes) - 20, 20):
            pairs.append((closes[i] / closes[i - 60] - 1.0,
                          closes[i + 20] / closes[i] - 1.0))
    if len(pairs) <= 10:
        return {}
    n = len(pairs)
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    cov = sum((a - mx) * (b - my) for a, b in pairs) / n
    sx = (sum((a - mx) ** 2 for a, _ in pairs) / n) ** 0.5
    sy = (sum((b - my) ** 2 for _, b in pairs) / n) ** 0.5
    q = max(1, n // 5)
    return {
        "n": n,
        "corr": cov / (sx * sy) if sx and sy else 0.0,
        "top": sum(b for _, b in sorted(pairs, reverse=True)[:q]) / q,
        "bot": sum(b for _, b in sorted(pairs)[:q]) / q,
    }


def run(max_symbols: int) -> dict:
    src = OpentdxParquetSource()
    all_syms = src.available_symbols()
    syms = all_syms[:max_symbols]
    if not syms:
        raise SystemExit("数据源无数据")
    feed = src.universe(syms)

    rows = []
    any_negative = False
    print(f"标的数: {len(syms)} / 全市场 {len(all_syms)}；bars: {len(feed.bars):,}")
    print(f"{'mode':<10}{'收益':>11}{'年化':>10}{'回撤':>9}"
          f"{'交易':>7}{'期末权益':>14}{'期末现金':>13}{'最低现金':>13}")
    for mode in ("momentum", "reversal", "random"):
        eng = Engine(feed, RankStrategy(syms, 3, mode), initial_cash=INITIAL_CASH)
        res = eng.run()
        perf = Performance(res)
        final_cash, min_cash = cash_path(res.fills, INITIAL_CASH)
        neg = min_cash < -1e-6
        any_negative = any_negative or neg
        rows.append(dict(mode=mode, ret=perf.total_return,
                         ann=perf.annualized_return, mdd=perf.max_drawdown,
                         trades=res.total_trades(), equity=res.final_equity(),
                         final_cash=final_cash, min_cash=min_cash, neg=neg))
        print(f"{mode:<10}{perf.total_return:>11.2%}{perf.annualized_return:>10.2%}"
              f"{perf.max_drawdown:>9.2%}{res.total_trades():>7d}"
              f"{res.final_equity():>14,.0f}{final_cash:>13,.0f}{min_cash:>13,.0f}"
              f"{'  NEG' if neg else ''}")

    bh = _benchmark(src, syms)
    stats = _pure_stats(src, syms)
    print(f"等权买入持有: {bh:+.2%}")
    if stats:
        print(f"纯统计: n={stats['n']} corr={stats['corr']:+.4f} "
              f"top={stats['top']:+.2%} bot={stats['bot']:+.2%}")
    return dict(rows=rows, bh=bh, stats=stats, n=len(syms), any_negative=any_negative,
                bars=len(feed.bars))


def render_md(r: dict) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 信号传导对照实验（修正版）",
        "",
        f"> 生成 {now} ｜ 降采样 {r['n']} 只 × DAILY/QFQ ｜ {r['bars']:,} 根 Bar ｜ "
        f"复现 `python tools/validate_signal_transmission.py --max-symbols {r['n']}`",
        "",
        "## 为什么重跑",
        "",
        "旧版 6d 的 `RankStrategy` 未检查 `host.has_pending`，在 I2 延迟下重复下单；"
        "配合默认 `NoOpRisk` 会让现金为负（隐含无息杠杆），从而产生 "
        "`+235,360%` / `−102%` 这类不可解释的结果。本脚本修正后重新生成对照表，"
        "并把「最低现金」作为有效性闸门：任何一组出现负现金，该组结果即判定无效。",
        "",
        "**定位**：机制对照，不是业绩证明。样本为降采样、且数据源不含退市股（存活偏差），"
        "绝对收益不得外推。",
        "",
        "## 结果",
        "",
        "| 排序模式 | 总收益 | 年化 | 最大回撤 | 交易数 | 期末权益 | 期末现金 | 最低现金 | 有效 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for x in r["rows"]:
        lines.append(
            f"| {x['mode']} | {x['ret']:+.2%} | {x['ann']:+.2%} | {x['mdd']:.2%} | "
            f"{x['trades']} | {x['equity']:,.0f} | {x['final_cash']:,.0f} | "
            f"{x['min_cash']:,.0f} | {'✗ 负现金' if x['neg'] else '✓'} |"
        )
    lines += [
        f"| 等权买入持有 | {r['bh']:+.2%} | — | — | — | — | — | — | — |",
        "",
    ]
    if r["stats"]:
        s = r["stats"]
        lines += [
            "## 不经框架的纯统计（60 日动量 → 未来 20 日收益）",
            "",
            f"- 样本数：{s['n']}",
            f"- 相关系数：{s['corr']:+.4f}",
            f"- 动量最高 20% 的未来 20 日收益：{s['top']:+.2%}",
            f"- 动量最低 20% 的未来 20 日收益：{s['bot']:+.2%}",
            "",
        ]
    if r["any_negative"]:
        lines += [
            "## 结论",
            "",
            "**至少一组出现负现金，本次对照无效。** 负现金意味着策略在透支/加杠杆，"
            "收益量级不能解释为信号传导；应先修策略的重复下单或启用 `StandardRisk`。",
        ]
    else:
        lines += [
            "## 结论",
            "",
            "三组均无负现金，成交与账本可复核。排序方向是否反转只说明该信号在本样本期"
            "的方向性；**不构成收益能力证明**，也不应引用绝对收益做宣传。",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="修正版信号传导对照实验")
    ap.add_argument("--max-symbols", type=int, default=100)
    ap.add_argument("--md", default="reports/SIGNAL_TRANSMISSION.md")
    args = ap.parse_args()
    result = run(max(10, args.max_symbols))
    md = render_md(result)
    p = Path(args.md)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(md, encoding="utf-8")
    print(f"\n报告已写入 {p}")


if __name__ == "__main__":
    main()
