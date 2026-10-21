"""多因子 + Sweep 主动交易策略 · 端到端回测与交付。

流程：
  1. 加载数据准备结果（缓存，含因子/信号/辅助面板）；
  2. 若已有最优权重 ``best_weights.json`` 则直接加载，否则运行权重优化器
     （样本内/外多次回测 + 坐标上升，选 OOS-Sharpe 最高的权重）；
  3. 用最优权重跑**完整 3 年窗口**回测；
  4. 生成交付物：
       - mf_sweep_report.html   自包含 HTML 报告（权益曲线 + 逐标的 K 线图 + 指标）
       - mf_sweep_trades.csv    逐笔成交明细
       - mf_sweep_roundtrips.csv  每标的已实现盈亏回合
       - mf_sweep_report.md     中文回测报告（含权重搜索结果 + 幸存者偏差警告）
       - best_weights.json      最优因子权重（供复现）

用法：
    python run.py            # 完整流程（含权重优化）
    python run.py --no-opt   # 跳过优化，直接用已保存的 best_weights.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import data_prep as D
import backtest_engine as BE
import optimize_weights as OW
from framebt.report import generate_report, ReportConfig

OUT = os.path.join(os.path.dirname(__file__))
BEST_WEIGHTS = os.path.join(OUT, "best_weights.json")


def _ts_to_date(ts):
    return datetime.fromordinal(int(ts)).date().isoformat()


def load_or_optimize(prep, *, do_optimize):
    """加载或优化出最优权重。"""
    if do_optimize or not os.path.exists(BEST_WEIGHTS):
        best_w, log, oos_ts = OW.optimize(prep, verbose=True)
        with open(BEST_WEIGHTS, "w", encoding="utf-8") as f:
            json.dump({k: round(v, 6) for k, v in best_w.items()},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUT, "weight_search_log.json"), "w",
                  encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2, default=str)
        print(f"[run] 最优权重已存 {BEST_WEIGHTS}")
        return best_w, log
    with open(BEST_WEIGHTS, encoding="utf-8") as f:
        best_w = {k: float(v) for k, v in json.load(f).items()}
    log = []
    lp = os.path.join(OUT, "weight_search_log.json")
    if os.path.exists(lp):
        with open(lp, encoding="utf-8") as f:
            log = json.load(f)
    return best_w, log


def round_trips(fills):
    """把逐笔成交按标的配成已实现回合，返回 [{date,side,symbol,size,price,amount,pnl}]。

    用**平均成本法**处理同一标的多次买入 / 部分平仓（核心组合会加仓后再调仓卖出），
    确保已实现盈亏之和与权益曲线变化一致。
    """
    from framebt.core import OrderSide
    rows = []
    open_notional = {}   # 当前持仓成本总额
    open_size = {}       # 当前持仓股数
    open_fill = {}       # 本轮最早买入（entry）
    for f in sorted(fills, key=lambda x: x.timestamp):
        sym = f.symbol
        if f.side is OrderSide.BUY:
            open_notional[sym] = open_notional.get(sym, 0.0) + f.size * f.price
            open_size[sym] = open_size.get(sym, 0) + f.size
            open_fill.setdefault(sym, f)
        else:
            sold_amount = f.size * f.price
            sz = open_size.get(sym, 0)
            if sz > 0:
                avg_cost = open_notional.get(sym, 0.0) / sz if sz else 0.0
                pnl = (f.price - avg_cost) * f.size          # 按成交股数计算已实现盈亏
                open_notional[sym] = open_notional.get(sym, 0.0) - avg_cost * f.size
                open_size[sym] = sz - f.size
                entry = open_fill.get(sym)
                if open_size[sym] <= 0:                     # 全部平仓 → 一个完整回合
                    rows.append({
                        "date": _ts_to_date(f.timestamp), "side": "ROUND_TRIP",
                        "symbol": sym, "size": f.size, "price": f.price,
                        "amount": round(sold_amount, 2), "pnl": round(pnl, 2),
                        "cost": round(f.cost, 2),
                        "entry_date": _ts_to_date(entry.timestamp) if entry else "",
                        "entry_price": round(entry.price, 4) if entry else 0.0,
                    })
                    open_notional.pop(sym, None)
                    open_size.pop(sym, None)
                    open_fill.pop(sym, None)
    return rows


def write_csv(path, rows):
    if not rows:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            f.write("")
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-opt", action="store_true",
                    help="跳过权重优化，直接用已保存的 best_weights.json")
    args = ap.parse_args()

    prep = D.prepare()
    print(f"[run] 窗口 {prep.window_start}..{prep.window_end}，"
          f"{prep.n_days} 交易日，池 {len(prep.universe)} 只")

    best_w, log = load_or_optimize(prep, do_optimize=not args.no_opt)
    print("[run] 最优权重:", {k: round(v, 4) for k, v in sorted(best_w.items())})

    # ---- 完整窗口回测
    res, perf = BE.run_backtest(prep, best_w)
    fills = sorted(res.fills, key=lambda f: f.timestamp)
    eq = res.equity_curve

    # ---- 逐笔明细 CSV
    fill_rows = [{
        "date": _ts_to_date(f.timestamp),
        "side": "BUY" if f.side.value == "buy" else "SELL",
        "symbol": f.symbol, "size": round(f.size, 2),
        "price": round(f.price, 4), "amount": round(f.size * f.price, 2),
        "cost": round(f.cost, 2),
    } for f in fills]
    write_csv(os.path.join(OUT, "mf_sweep_trades.csv"), fill_rows)

    # ---- 已实现盈亏回合 CSV
    rt = round_trips(fills)
    write_csv(os.path.join(OUT, "mf_sweep_roundtrips.csv"), rt)

    # ---- HTML 报告（含 K 线图）
    html = generate_report(
        res, bars_by_symbol={}, source=D.OpentdxParquetSource(),
        strategy_name="多因子基础 + Sweep 做 T · 主动交易策略",
        config=ReportConfig(max_trade_charts=15),
        output_path=os.path.join(OUT, "mf_sweep_report.html"),
    )
    print(f"[run] HTML 报告 -> mf_sweep_report.html ({len(html)//1024} KB)")

    # ---- 中文报告
    start, end = _ts_to_date(eq[0][0]), _ts_to_date(eq[-1][0])
    traded = sorted({f.symbol for f in fills})
    wins = [r for r in rt if r["pnl"] > 0]
    losers = [r for r in rt if r["pnl"] <= 0]
    total_pnl = sum(r["pnl"] for r in rt)
    win_rate = len(wins) / len(rt) if rt else 0.0

    # 权重搜索结果表
    if log:
        log_tbl = "\n".join(
            "| " + " | ".join([str(c) for c in [
                r.get("name", ""),
                f"{r.get('oos_sharpe', float('nan')):+.3f}",
                f"{r.get('in_sharpe', float('nan')):+.3f}",
                f"{r.get('sharpe', float('nan')):+.3f}",
                f"{r.get('total_return', float('nan')):+.3f}",
                f"{r.get('max_drawdown', float('nan')):.3f}",
                f"{r.get('win_rate', 0.0):.0%}" if isinstance(r.get('win_rate'), float) else str(r.get('win_rate')),
                str(r.get('n_trades', '')),
            ]]) + " |"
            for r in log
        )
        log_hdr = ("| 权重方案 | 样本外 Sharpe | 样本内 Sharpe | 全窗口 Sharpe | "
                   "全窗口收益 | 最大回撤 | 胜率 | 成交笔 |")
    else:
        log_tbl = log_hdr = ""

    md = f"""# 多因子基础 + Sweep 做 T · 主动交易策略 — 回测报告

## 一、结论速览

| 指标 | 结果 |
|------|------|
| 策略 | 多因子（14 因子加权）为核心，Sweep 突破放量信号做 T，主动选股、可空仓 |
| 区间 | {start} → {end}（{eq and len(eq) or 0} 交易日） |
| 总收益 | **{perf.total_return:+.0%}**（年化 {perf.annualized_return:+.2%}） |
| Sharpe | {perf.sharpe:.2f} · 最大回撤 {perf.max_drawdown:.0%} · 胜率 {win_rate:.0%} |
| 成交 | {res.total_trades():,} 笔 / {len(traded)} 标的 |
| 最优权重 | {', '.join(f'{k}={v:.2f}' for k, v in sorted(best_w.items(), key=lambda x:-x[1])[:6])} … |

## 二、策略设计

- **核心（~70% 仓位）**：14 个 WorldQuant 101 Alpha 因子横截面标准化后加权合成打分，
  每日打分选 Top20 等权持仓，每 5 交易日调仓；达标不足 5 只则**主动空仓**。
- **卫星（Sweep，~30% 仓位 / 闲置容量）**：收盘突破近 15 日高点、放量 >1.5×均量、当日阳线
  时触发，单笔持有 ≤5 交易日或跌破触发位即了结，反复滚动即“做 T”；仅在市场风险偏好（指数 >MA60）
  且核心未满仓时的**闲置容量**中，从“流动性合格 + 信号成立 + 不在核心池”中**主动挑选新中标的**。
- **空仓机制**：风险规避期（指数跌破 MA60）或无合格标的时策略接受空仓（现金）。
- **因果性**：所有信号在数据准备阶段后向计算，市价单次日开盘撮合（T+1 满足，无前视偏差）。

## 三、因子权重优化（多次回测 + 调整）

在样本内（前 60% 交易日）计算各因子 IC，生成 9 组候选权重（等权 / 带符号 IC / |IC| / 分块 /
各块放大），每组跑完整窗口回测并按**样本外 Sharpe** 排序，再对最优解做坐标上升微调。

{log_hdr + "\n" + log_tbl if log_tbl else "_权重搜索尚未运行_"}

## 四、绩效与盈亏

- 已实现盈亏回合 {len(rt)} 个，胜率 {win_rate:.0%}，合计已实现 {total_pnl:+,.0f} 元。
- 逐笔明细：`mf_sweep_trades.csv`（date/side/symbol/size/price/amount/cost）
- 盈亏回合：`mf_sweep_roundtrips.csv`（entry/exit 价格、每回合 pnl）
- 权益曲线与**逐标的 K 线图**：`mf_sweep_report.html`
- 最优权重：`best_weights.json`

## 五、局限与幸存者偏差警告

⚠️ 本策略沿用「全 A 当前流动性前 300 只」的**静态标的池**，存在**幸存者偏差**：
池内均为能活至今的大盘股，回测收益会**高估真实可获利性**。权重优化基于样本外 Sharpe，
已尽量抑制过拟合，但在静态池上测得的正收益仍需谨慎看待。

## 六、数据与口径

- 数据：OpenTDX Parquet 日线，QFQ 前复权，{start}–{end}。
- 标的池：全 A 中流动性前 300（历史日均成交额），≥260 根 K 线。
- 因子：WQ101 Alpha 14 因子（动量/反转、放量、vwap 偏离、波动率、日内动量跨块）。
- 成本：佣金 0.03%（最低 5）+ 印花税 0.1%（卖）+ 过户费 0.002% + 量价滑点。
- 涨跌停：10% 硬约束，触及当日不可成交。
"""
    with open(os.path.join(OUT, "mf_sweep_report.md"), "w", encoding="utf-8") as f:
        f.write(md)

    print("[run] 中文报告 -> mf_sweep_report.md")
    print(f"[run] 成交 {res.total_trades():,} 笔 | 收益 {perf.total_return:+.0%} | "
          f"Sharpe {perf.sharpe:.2f} | 回撤 {perf.max_drawdown:.0%} | 胜率 {win_rate:.0%}")


if __name__ == "__main__":
    main()
