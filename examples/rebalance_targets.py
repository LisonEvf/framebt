"""示例：目标权重调仓（60/40，每 21 帧再平衡）。

演示 ``host.rebalance_to(targets, frame)``：不用手写「算权益 → 目标股数 → 先卖后买 →
整手 → 现金缓冲」。数据是合成价格，无外部依赖。

运行：
    python examples/rebalance_targets.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from framebt import (  # noqa: E402
    BarFeed,
    Engine,
    FrameStrategy,
    NoOpCost,
    Performance,
    RebalanceConfig,
)


class SixtyForty(FrameStrategy):
    """每 ``rebalance_every`` 帧把权益调到 A 60% / B 40%。"""

    def __init__(self, a: str = "A", b: str = "B", *, rebalance_every: int = 21):
        self.a = a
        self.b = b
        self.rebalance_every = int(rebalance_every)
        self.config = RebalanceConfig(lot=100, min_trade_pct=0.01, cash_buffer=0.002)

    def on_frame(self, frame, token, bus, host):
        if frame.frame_id % self.rebalance_every != 0:
            return []
        return host.rebalance_to({self.a: 0.6, self.b: 0.4}, frame, config=self.config)


def synth_bars(n: int = 500) -> list[dict]:
    rows = []
    for i in range(n):
        # A：慢趋势 + 波动；B：反向波动，制造再平衡需求
        pa = 100.0 * (1.0 + 0.20 * math.sin(i / 37.0)) + i * 0.05
        pb = 50.0 * (1.0 - 0.15 * math.sin(i / 29.0)) + i * 0.03
        for sym, px in (("A", pa), ("B", pb)):
            rows.append({"timestamp": i, "symbol": sym, "open": px, "high": px * 1.01,
                         "low": px * 0.99, "close": px, "volume": 5_000_000.0})
    return rows


def main() -> None:
    strategy = SixtyForty(rebalance_every=21)
    engine = Engine(BarFeed.from_dicts(synth_bars()), strategy,
                    initial_cash=1_000_000.0, cost_model=NoOpCost())
    result = engine.run()
    perf = Performance(result)

    print("最终持仓：", engine.ledger.exposure())
    print("现金：    ", f"{engine.ledger.cash:,.2f}")
    print("总收益：  ", f"{perf.total_return:+.2%}")
    print("年化：    ", f"{perf.annualized_return:+.2%}")
    print("Sharpe：  ", f"{perf.sharpe:.2f}")
    print("最大回撤：", f"{perf.max_drawdown:.2%}")
    print("Calmar：  ", f"{perf.calmar:.2f}")
    print("成交笔数：", result.total_trades())


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
