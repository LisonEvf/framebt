"""共享回测引擎：策略构建、成本/规则、绩效计算。

供 ``optimize_weights``（权重搜索）与 ``run``（最终回测）复用，保证两者口径一致。
"""

from __future__ import annotations

import numpy as np

from framebt import Engine, Performance, StandardAshareRules, StandardCost
from framebt.strategies import MultiFactorSweepStrategy

# ---------------------------------------------------------------- 默认参数
DEFAULT_PARAMS = dict(
    top_k=20,
    rebalance_every=5,
    warmup=60,
    score_floor=0.0,
    min_hold=5,
    base_gross=0.70,        # 核心组合占权益 70%，留 30% 给 sweep
    regime_off_gross=0.0,   # 风险规避 → 完全空仓
    sweep_budget_pct=0.30,  # sweep 战术仓最大占比
    sweep_lookback=15,
    sweep_vol_mult=1.5,
    sweep_hold_max=5,
    max_sweep_positions=12,
    sweep_min_notional=30000.0,
    lot=100,
)

TRADING_DAYS_PER_YEAR = 244


def build_strategy(prep, weights, params=None):
    p = {**DEFAULT_PARAMS, **(params or {})}
    strat = MultiFactorSweepStrategy(
        prep.universe, prep.factor_panels, prep.close,
        factor_weights=weights, **p,
    )
    strat.inject_aux(eligibility=prep.eligibility, sweep_signal=prep.sweep_signal,
                     sweep_exit=prep.sweep_exit, regime=prep.regime)
    return strat


def make_cost():
    """A 股标准成本：佣金 0.03%(最低 5) + 印花税 0.1%(卖) + 过户费 0.002% + 量价滑点。"""
    return StandardCost(commission=0.0003, min_commission=5.0, stamp_tax=0.001,
                        transfer_fee=0.00002, slippage="volume_based", k=0.001)


def make_rules():
    return StandardAshareRules(limit=0.10)


def run_backtest(prep, weights, *, params=None, initial_cash=1_000_000.0,
                 cost=None, rules=None):
    """跑一次完整回测，返回 (result, Performance)。"""
    strat = build_strategy(prep, weights, params)
    engine = Engine(
        prep.feed, strat,
        initial_cash=initial_cash,
        settlement_cycle=1,
        delay_frames=1,
        cost_model=cost if cost is not None else make_cost(),
        rules=rules if rules is not None else make_rules(),
    )
    res = engine.run()
    perf = Performance(res, period_frames=TRADING_DAYS_PER_YEAR)
    return res, perf


def sharpe_from_equity(equity_pairs, *, period_frames=TRADING_DAYS_PER_YEAR):
    """从权益曲线切片计算年化 Sharpe（每日收益）。"""
    ys = [v for _, v in equity_pairs]
    if len(ys) < 5 or ys[0] <= 0:
        return 0.0
    r = [ys[i] / ys[i - 1] - 1.0 for i in range(1, len(ys)) if ys[i - 1] != 0]
    if len(r) < 5:
        return 0.0
    m = sum(r) / len(r)
    v = sum((x - m) ** 2 for x in r) / (len(r) - 1)
    sd = v ** 0.5
    if sd == 0:
        return 0.0
    return m / sd * (period_frames ** 0.5)


def oos_sharpe(res, *, start_frame, period_frames=TRADING_DAYS_PER_YEAR):
    """从完整回测中切出 OOS 区间（frame_id >= start_frame）计算 Sharpe。"""
    oos = [(t, e) for t, e in res.equity_curve if t >= start_frame]
    return sharpe_from_equity(oos, period_frames=period_frames)
