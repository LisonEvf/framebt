"""快速上手 Demo：三步跑通 framebt 回测 + 图表化报告。

本演示**无需任何外部数据**即可运行：

- 若能找到 opentdx 数仓（`../opentdx/db/data`），则用真实日线数据；
- 否则自动用确定性随机过程合成一段日 K 线，保证演示可复现。

三步：
    1. 加载数据（真实 / 合成）
    2. 跑双均线交叉策略
    3. 生成一张含「收益曲线 + 回撤高亮 + 交易 K 线标注」的 HTML 报告

运行：
    python examples/demo_quickstart.py                 # 默认
    python examples/demo_quickstart.py --real          # 强制用真实数据
    python examples/demo_quickstart.py --synthetic     # 强制用合成数据
    python examples/demo_quickstart.py --short 10 --long 40
    python examples/demo_quickstart.py --symbol "SH.600519"

报告默认输出到 reports/demo_quickstart.html，用浏览器打开即可。
"""

from __future__ import annotations

import argparse
import math
import os

import numpy as np

from framebt import (
    Engine,
    Performance,
    run_backtest,
    StandardCost,
    StandardRisk,
    StandardAshareRules,
    generate_report,
)
from framebt.strategies import DualMAStrategy


def _synthetic_bars(symbol: str, n: int = 400, seed: int = 7) -> list[dict]:
    """确定性合成日 K 线（多周期正弦漂移 + 随机游走），供无数据时演示。

    用「低频正弦漂移 + 随机游走」构造有起伏的趋势，使 5/20 均线产生多轮交叉，
    从而在报告中呈现多笔完整交易。固定 seed 保证每次运行结果一致（I3 精神）。
    """
    rng = np.random.default_rng(seed)
    # 对数收益 = 低频正弦漂移（制造波峰波谷）+ 小噪声；积分得价格。
    phases = (2 * math.pi / 70, 2 * math.pi / 130, 2 * math.pi / 25)
    amps = (0.012, 0.008, 0.004)
    logp = math.log(50.0)
    logs = [logp]
    for i in range(1, n):
        drift = sum(a * math.sin(p * i) for a, p in zip(amps, phases))
        ret = drift + rng.normal(0, 0.012)
        logs.append(logs[-1] + ret)
    closes = [math.exp(z) for z in logs]

    bars = []
    for i in range(n):
        close = float(closes[i])
        open_ = float(closes[i - 1] if i > 0 else close)
        intrabar = abs(rng.normal(0, 0.015))
        high = float(max(open_, close) * (1 + intrabar))
        low = float(min(open_, close) * (1 - intrabar))
        low = max(low, 1.0)
        vol_shares = float(rng.integers(50_000, 200_000))
        bars.append(
            {
                "timestamp": 58_000 + i,  # 任意递增 ordinal，模拟交易日
                "symbol": symbol,
                "open": round(open_, 2),
                "high": round(high, 2),
                "low": round(low, 2),
                "close": round(close, 2),
                "volume": vol_shares,
                "amount": round(vol_shares * close, 2),
            }
        )
    return bars


def _try_real_source(symbol: str | None = None):
    """尝试加载 opentdx 真实数据；失败返回 None。"""
    try:
        from framebt.data_sources import OpentdxParquetSource

        src = OpentdxParquetSource()
        symbols = src.available_symbols()
        if not symbols:
            return None, None
        if symbol is not None and symbol not in symbols:
            print(f"〔提示〕数仓中无 {symbol}，自动选用首个可用标的 {symbols[0]}")
            symbol = symbols[0]
        elif symbol is None:
            symbol = symbols[0]
        feed = src.load(symbol)
        bars = [
            {
                "timestamp": b.timestamp,
                "symbol": b.symbol,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume,
                "amount": b.amount,
            }
            for b in feed.bars
        ]
        return symbol, bars
    except Exception:  # noqa: BLE001 数仓缺失 / 缺 pandas 等
        return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description="framebt 快速上手 Demo")
    ap.add_argument("--real", action="store_true", help="强制使用真实数据")
    ap.add_argument("--synthetic", action="store_true", help="强制使用合成数据")
    ap.add_argument("--symbol", default=None, help="指定标的（真实数据）")
    ap.add_argument("--short", type=int, default=5, help="短均线窗口")
    ap.add_argument("--long", type=int, default=20, help="长均线窗口")
    ap.add_argument("--cash", type=float, default=100_000.0, help="初始资金")
    ap.add_argument("--out", default="reports/demo_quickstart.html", help="报告输出路径")
    args = ap.parse_args()

    symbol, bars = _try_real_source(args.symbol)
    if args.synthetic or (args.real is False and bars is None):
        symbol = args.symbol or "SYN.000001"
        bars = _synthetic_bars(symbol)
        print(f"〔合成数据〕{symbol}：{len(bars)} 根日 K 线（无 opentdx 数仓时的兜底）")
    elif bars is not None:
        print(f"〔真实数据〕{symbol}：{len(bars)} 根日 K 线（opentdx 数仓）")
    else:
        raise SystemExit("未指定 --synthetic 且无真实数据，无法运行。")

    if len(bars) <= args.long:
        raise SystemExit(
            f"数据仅 {len(bars)} 根，不足以计算 MA{args.long}；"
            "请用 --synthetic 或回填更长历史数据。"
        )

    # ---- 跑回测：双均线交叉 + A 股成本 / 风控 / 规则 ----
    res = run_backtest(
        bars,
        DualMAStrategy(symbol, short=args.short, long=args.long),
        symbol=symbol,
        initial_cash=args.cash,
        settlement_cycle=1,
        cost_model=StandardCost(
            commission=0.0003, min_commission=5.0, stamp_tax=0.001
        ),
        risk=StandardRisk(max_position_pct=0.95),
        rules=StandardAshareRules(limit=0.10),
    )

    # ---- 绩效 ----
    perf = Performance(res, period_frames=244)
    print("\n=== 回测结果 ===")
    print(f"标的: {symbol}  窗口: {args.short}/{args.long}  初始资金: {args.cash:,.0f}")
    print(f"交易次数: {res.total_trades()}  最终权益: {res.final_equity():,.2f}")
    print(f"总收益: {perf.total_return:+.2%}  年化: {perf.annualized_return:+.2%}")
    print(f"最大回撤: {perf.max_drawdown:.2%}  Sharpe: {perf.sharpe:.2f}  胜率: {perf.win_rate():.0%}")

    # ---- 生成图表化报告（含收益曲线 / 回撤高亮 / 交易 K 线标注）----
    bars_by_symbol = {symbol: bars}
    html = generate_report(
        res,
        bars_by_symbol=bars_by_symbol,
        strategy_name=f"双均线 {args.short}/{args.long} · {symbol}",
        output_path=args.out,
    )
    print(f"\n报告已生成: {os.path.abspath(args.out)}")
    print("用浏览器打开即可查看：收益曲线、回撤高亮、逐笔交易 K 线标注、成交明细表。")


if __name__ == "__main__":
    main()
