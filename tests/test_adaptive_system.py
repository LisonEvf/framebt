"""自适应多因子系统：集成测试与**缺陷回归**。

本文件锁死开发过程中通过真实数据回测暴露出来的一串缺陷：
  1. ``StandardRisk`` 把「现金」当「权益」→ 满仓后所有订单被判超额而拒绝
  2. ``reserved_cash`` 把**正在校验的订单自身**计入预留 → 全部买单被拒（零成交）
  3. 引擎从不调用 ``rules.observe(frame)`` → **涨跌停从未在任何回测中生效**
  4. 引擎未把 ``settlement_cycle`` 传给 ``Ledger`` → T+0 无法生效
  5. ``on_fill`` 未在默认唤醒集中 → 被动单静默失效（见 test_strategies）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framebt import (Bar, BarFeed, Engine, FillEvent, FrameStrategy, Ledger,
                     MarketFrame, MatchingEngine, NoOpRules, Order, OrderSide,
                     OrderType, StandardAshareRules, StandardRisk)
from framebt.factors.lifecycle import ACTIVE, OBSERVING, RETIRED, FactorRegistry
from framebt.monitor import FactorHealthMonitor
from framebt.strategies.adaptive_multi_factor import AdaptiveMultiFactorStrategy


def mkframe(fid: int, o: float, h: float, l: float, c: float, symbol: str = "S"):
    f = MarketFrame(fid, fid, fid + 1)
    f.add_bar(Bar(timestamp=fid, symbol=symbol, open=o, high=h, low=l, close=c, volume=1e6))
    return f


# ===========================================================================
# 1. 风控权益口径
# ===========================================================================
def test_risk_uses_total_equity_not_cash():
    """regression：曾用 Ledger.equity()（=现金）当权益 → 满仓后一切订单被拒。"""
    led = Ledger(cash=100_000.0)
    led.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                             size=9_900, price=10.0, frame_id=0, timestamp=0))
    # 现在：现金 1000，持仓 99,000 → 真实权益 100,000
    assert led.equity() == 1_000.0, "Ledger.equity() 本就只返回现金"

    risk = StandardRisk(max_order_pct=0.5)
    f = mkframe(1, 10, 10, 10, 10)
    # 卖单 notional = 3000×10 = 30,000 = 权益的 30% < 50% → 应放行
    assert risk.pre_trade_check(Order(side=OrderSide.SELL, symbol="S", size=3_000),
                                led, f) is None
    # 若误用现金(1000)，30,000 会远超 50%×1000 → 被拒
    assert StandardRisk._equity(led, f) == pytest.approx(100_000.0)


# ===========================================================================
# 2. 在途资金预留不得含自身
# ===========================================================================
def test_reserved_cash_excludes_current_order():
    """regression：预留含自身 → 每笔合法买单都被判资金不足 → 整场零成交。"""
    me = MatchingEngine(delay_frames=5)
    risk = StandardRisk(max_order_pct=10.0)
    risk.matcher = me
    led = Ledger(cash=10_000.0)
    f = mkframe(0, 10, 10, 10, 10)

    o1 = Order(side=OrderSide.BUY, symbol="S", size=500)     # 5,000
    me.register(o1, 0)
    assert risk.pre_trade_check(o1, led, f) is None, "自身的预留不应吃掉可用资金"

    o2 = Order(side=OrderSide.BUY, symbol="S", size=500)     # 再 5,000
    me.register(o2, 0)
    assert risk.pre_trade_check(o2, led, f) is None           # 恰好用满

    o3 = Order(side=OrderSide.BUY, symbol="S", size=200)     # 再 2,000 → 透支
    me.register(o3, 0)
    reason = risk.pre_trade_check(o3, led, f)
    assert reason is not None and "资金不足" in reason
    assert risk.reserved_cash(f, exclude=o3) == pytest.approx(10_000.0)


def test_risk_reserve_can_be_disabled():
    me = MatchingEngine(delay_frames=5)
    risk = StandardRisk(max_order_pct=10.0, reserve_pending=False)
    risk.matcher = me
    led = Ledger(cash=10_000.0)
    f = mkframe(0, 10, 10, 10, 10)
    me.register(Order(side=OrderSide.BUY, symbol="S", size=900), 0)
    assert risk.pre_trade_check(Order(side=OrderSide.BUY, symbol="S", size=900),
                                led, f) is None


# ===========================================================================
# 3. 引擎必须驱动 rules.observe（涨跌停真正生效）
# ===========================================================================
class _BuyAtFrame(FrameStrategy):
    def __init__(self, at: int = 0, size: float = 100.0):
        self.at, self.size = at, size

    def on_frame(self, frame, token, bus, host):
        if frame.frame_id == self.at:
            return [Order(side=OrderSide.BUY, symbol="S", size=self.size,
                          order_type=OrderType.MARKET)]
        return []


def _limit_up_feed():
    # 帧0 收盘 10.0 → 帧1 涨停价 11.0；帧1 开盘 11.5 越界
    return BarFeed.from_dicts([
        {"timestamp": 0, "symbol": "S", "open": 9.8, "high": 10.0, "low": 9.7,
         "close": 10.0, "volume": 1e6},
        {"timestamp": 1, "symbol": "S", "open": 11.5, "high": 11.6, "low": 11.4,
         "close": 11.5, "volume": 1e6},
    ], symbol="S")


def test_engine_enforces_price_limit_up():
    """regression：引擎曾从不调用 rules.observe → 涨跌停完全失效。"""
    eng = Engine(_limit_up_feed(), _BuyAtFrame(0), initial_cash=1_000_000.0,
                 rules=StandardAshareRules(limit=0.10))
    res = eng.run()
    assert res.total_trades() == 0, "开盘价越过涨停，买单必须被拒"


def test_same_setup_fills_without_rules():
    """同一场景改用 NoOpRules 应成交 —— 证明上一条确实由涨跌停规则拦住。"""
    eng = Engine(_limit_up_feed(), _BuyAtFrame(0), initial_cash=1_000_000.0,
                 rules=NoOpRules())
    res = eng.run()
    assert res.total_trades() == 1
    assert res.fills[0].price == 11.5


def test_rules_observe_tracks_prev_close():
    rules = StandardAshareRules(limit=0.10)
    f0 = mkframe(0, 9.8, 10.0, 9.7, 10.0)
    rules.observe(f0)
    upper, lower = rules.price_limit("S", f0)
    assert np.isclose(upper, 11.0) and np.isclose(lower, 9.0)


# ===========================================================================
# 4. settlement_cycle 必须贯通到 Ledger
# ===========================================================================
def test_engine_passes_settlement_cycle_to_ledger():
    """regression：引擎曾只把 settlement_cycle 给规则，Ledger 恒定 T+1。"""
    feed = _limit_up_feed()
    assert Engine(feed, _BuyAtFrame(0), settlement_cycle=1).ledger.settlement_cycle == 1
    assert Engine(feed, _BuyAtFrame(0), settlement_cycle=0).ledger.settlement_cycle == 0


def test_t0_allows_same_frame_sell_in_ledger():
    led = Ledger(cash=100_000.0, settlement_cycle=0)
    led.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                             size=100, price=10.0, frame_id=5, timestamp=5))
    assert led.position("S").available_to_sell(5) == 100
    led1 = Ledger(cash=100_000.0, settlement_cycle=1)
    led1.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                              size=100, price=10.0, frame_id=5, timestamp=5))
    assert led1.position("S").available_to_sell(5) == 0


# ===========================================================================
# 5. 策略端到端
# ===========================================================================
def _synth_panel(n_symbols=10, n_times=220, seed=3):
    from framebt.factors import Panel

    rng = np.random.default_rng(seed)
    rows = []
    starts = 20.0 * np.exp(0.25 * rng.normal(size=n_symbols))
    for k in range(n_symbols):
        s = f"S{k:02d}"
        p = float(starts[k])
        for t in range(n_times):
            p = max(0.5, p * (1 + 0.02 * rng.normal()))
            vol = 1e6 * (1 + abs(rng.normal()))
            o = p * (1 + 0.004 * rng.normal())
            # 换手率必须随标的变化（常数截面会让 rank 退化为零方差行）
            rows.append({"timestamp": t, "symbol": s, "open": o,
                         "high": max(o, p) * 1.005, "low": min(o, p) * 0.995,
                         "close": p, "volume": vol, "amount": p * vol,
                         "float_shares": 1e9,
                         "turnover": float(0.3 + 0.5 * abs(rng.normal())),
                         "vwap": p, "market_cap": p * 1e9})
    return Panel.from_records(rows)


def _synth_feed(panel):
    bars = []
    for s in panel.symbols:
        for t in panel.timestamps:
            bars.append(Bar(timestamp=t, symbol=s, open=float(panel.open.loc[t, s]),
                            high=float(panel.high.loc[t, s]), low=float(panel.low.loc[t, s]),
                            close=float(panel.close.loc[t, s]),
                            volume=float(panel.volume.loc[t, s]),
                            amount=float(panel.amount.loc[t, s])))
    return BarFeed(bars)


def _make_strategy(panel, **kw):
    from framebt.factors import compute, log_market_cap, preprocess

    raw = {n: compute(n, panel) for n in ("wq042", "wq060", "wq033")}
    expo = [log_market_cap(panel)]
    pre = {n: preprocess(f, exposures=expo) for n, f in raw.items()}
    return AdaptiveMultiFactorStrategy(
        panel.symbols, pre, panel.close, turnover_panel=panel.turnover,
        top_k=3, rebalance_every=5, warmup=30, **kw)


def test_adaptive_strategy_end_to_end():
    panel = _synth_panel()
    feed = _synth_feed(panel)
    st = _make_strategy(panel)
    eng = Engine(feed, st, initial_cash=1_000_000.0, settlement_cycle=1,
                 risk=StandardRisk(max_position_pct=0.98),
                 rules=StandardAshareRules(limit=0.10))
    res = eng.run()

    assert res.frames > 150
    assert res.total_trades() > 0, "策略应当实际成交"
    assert eng.ledger.cash >= -1e-6, f"现金不得为负：{eng.ledger.cash}"
    # 每笔成交数量必须是整手
    assert all(f.size > 0 and f.size % 100 == 0 for f in res.fills)
    # 持仓不超过 top_k
    held = {s for s, n in eng.ledger.exposure().items() if n > 0}
    assert len(held) <= 3
    # 在线学习确实发生了
    assert st.optimizer.step_count > 100


def test_adaptive_strategy_deterministic():
    """I3：同配置两次运行应逐位一致。"""
    def once():
        panel = _synth_panel()
        st = _make_strategy(panel)
        eng = Engine(_synth_feed(panel), st, initial_cash=1_000_000.0,
                     settlement_cycle=1, rules=StandardAshareRules())
        r = eng.run()
        return r, st

    r1, s1 = once()
    r2, s2 = once()
    assert r1.equity_curve == r2.equity_curve
    assert [(f.order_id, f.price, f.size, f.frame_id) for f in r1.fills] == \
           [(f.order_id, f.price, f.size, f.frame_id) for f in r2.fills]
    assert np.array_equal(s1.optimizer.weights, s2.optimizer.weights)


def test_adaptive_strategy_never_exceeds_cash():
    panel = _synth_panel()
    st = _make_strategy(panel)
    eng = Engine(_synth_feed(panel), st, initial_cash=500_000.0, settlement_cycle=1,
                 risk=StandardRisk(max_position_pct=0.98),
                 rules=StandardAshareRules())
    res = eng.run()
    eq = [e for _k, e in res.equity_curve]
    assert min(eq) > 0, "权益不得为负"
    assert eng.ledger.cash >= -1e-6


# ===========================================================================
# 6. 健康监控：三级预警 + 参数敏感性安全阀
# ===========================================================================
def test_health_monitor_green_for_stable_factor():
    """平稳序列不得被判 RED（曾因「历史 down 变点」被永久拉满 → 误降权）。"""
    mon = FactorHealthMonitor()
    # 完全无趋势的序列：delta≈0 ⇒ 无衰退证据 ⇒ GREEN
    flat = pd.Series(0.05 + 0.02 * np.sin(np.arange(300) / 7.0))
    m_flat = mon.evaluate("flat", ic_series=flat)
    assert m_flat.status == "GREEN", f"无趋势序列被判 {m_flat.status}"

    # 平稳噪声：允许 YELLOW（观察），但**不应** RED
    rng = np.random.default_rng(0)
    reds = 0
    for s in range(12):
        ic = pd.Series(0.05 + 0.02 * np.random.default_rng(s).normal(size=300))
        if mon.evaluate(f"n{s}", ic_series=ic).status == "RED":
            reds += 1
    assert reds == 0, f"平稳噪声序列仍有 {reds}/12 被误判 RED"


def test_health_monitor_detects_decay_and_downgrades():
    """强衰退因子：红灯 + 参数敏感性稳健 → 允许自动降权。"""
    mon = FactorHealthMonitor()
    rng = np.random.default_rng(1)
    ic = pd.Series(np.concatenate([0.06 + 0.02 * rng.normal(size=200),
                                   -0.10 + 0.02 * rng.normal(size=100)]))
    m = mon.evaluate("decayed", ic_series=ic)
    assert m.status == "RED"
    assert m.decay_prob > 0.6
    assert m.stable_across_params is True
    assert m.auto_downgrade is True
    assert m.downgrade == 0.0


def test_health_monitor_safety_valve_blocks_unstable_downgrade():
    """参数敏感性不稳健时绝不自动降权（设计文档「重要警示」）。

    这里打断言为**性质**：只要发生了自动降权，就必须同时满足
    「状态非 GREEN」「参数敏感性稳健」「降权系数确实被下调」。
    """
    mon = FactorHealthMonitor()
    rng = np.random.default_rng(11)
    cases = []
    cases.append(pd.Series(np.concatenate([0.06 + 0.02 * rng.normal(size=200),
                                           -0.12 + 0.02 * rng.normal(size=100)])))  # 真衰退
    cases.append(pd.Series(0.0 + 0.05 * rng.normal(size=300)))                        # 纯噪声
    cases.append(pd.Series(0.05 + 0.02 * np.sin(np.arange(300) / 9.0)))               # 周期
    cases.append(pd.Series([0.01, -0.02, 0.03, -0.01] * 60))                          # 高频抖动

    for i, ic in enumerate(cases):
        m = mon.evaluate(f"c{i}", ic_series=ic)
        if m.auto_downgrade:
            assert m.status != "GREEN"
            assert m.stable_across_params is True
            assert m.downgrade < 1.0
        else:
            assert m.downgrade == 1.0, "未获授权时不得下调权重"
    # 真衰退那一条必须被识别（红/黄）且允许降权
    m0 = mon.metrics["c0"]
    assert m0.status in ("YELLOW", "RED")
    assert m0.auto_downgrade is True and m0.downgrade < 1.0


def test_health_monitor_insufficient_samples():
    mon = FactorHealthMonitor()
    m = mon.evaluate("few", ic_series=pd.Series([0.1, 0.2, 0.3]))
    assert m.status == "GREEN" and m.auto_downgrade is False


def test_health_downgrade_weights_renormalizes():
    mon = FactorHealthMonitor()
    rng = np.random.default_rng(2)
    good = pd.Series(0.06 + 0.02 * rng.normal(size=300))
    bad = pd.Series(np.concatenate([0.06 + 0.02 * rng.normal(size=200),
                                    -0.12 + 0.02 * rng.normal(size=100)]))
    mon.evaluate("good", ic_series=good)
    mon.evaluate("bad", ic_series=bad)
    w = mon.downgrade_weights({"good": 0.5, "bad": 0.5})
    assert w["bad"] < w["good"]
    assert np.isclose(sum(w.values()), 1.0)
    assert mon.dashboard().shape[0] == 2


# ===========================================================================
# 7. 因子生命周期
# ===========================================================================
def test_lifecycle_register_and_promote():
    reg = FactorRegistry(observe_days=5, promote_icir=0.5)
    reg.register("f1", source="symbolic", birth_date=0, weight=0.5)
    assert reg.records["f1"].status == OBSERVING
    # 观察期权重上限 5%
    assert reg.records["f1"].current_weight <= 0.05
    for d in range(1, 7):
        reg.step(d, icir=0.9)
    assert reg.records["f1"].status == ACTIVE
    assert "f1" in reg.active_factors()


def test_lifecycle_retire_requires_cusum_alarm():
    reg = FactorRegistry(retire_days=3, retire_icir=0.2)
    reg.register("f1", birth_date=0)
    reg.records["f1"].status = ACTIVE
    # 仅 ICIR 低但 CUSUM 未报警 → 不退休
    for d in range(1, 6):
        reg.step(d, icir=0.05, cusum_alarm=False)
    assert reg.records["f1"].status == ACTIVE
    # CUSUM 报警后达到连续天数 → 退休
    for d in range(6, 12):
        reg.step(d, icir=0.05, cusum_alarm=True)
    assert reg.records["f1"].status == RETIRED
    assert reg.records["f1"].current_weight == 0.0


def test_lifecycle_rehabilitation():
    reg = FactorRegistry(retire_days=2, rehab_days=4, promote_icir=0.5)
    reg.register("f1", birth_date=0)
    reg.records["f1"].status = ACTIVE
    for d in range(1, 4):
        reg.step(d, icir=0.05, cusum_alarm=True)
    assert reg.records["f1"].status == RETIRED
    for d in range(4, 12):
        reg.step(d, icir=0.9)
    assert reg.records["f1"].status == ACTIVE


def test_lifecycle_cap_weights_for_observing():
    reg = FactorRegistry(observe_max_weight=0.05)
    reg.register("new", birth_date=0)
    reg.register("old", birth_date=0)
    reg.records["old"].status = ACTIVE
    capped = reg.cap_weights({"new": 0.9, "old": 0.1})
    assert capped["new"] <= 0.05 + 1e-9
    assert np.isclose(sum(capped.values()), 1.0)


def test_lifecycle_invalid_source():
    reg = FactorRegistry()
    with pytest.raises(ValueError):
        reg.register("x", source="bogus")


def test_lifecycle_summary_and_frame():
    reg = FactorRegistry()
    reg.register("a", source="manual")
    reg.register("b", source="symbolic")
    s = reg.summary()
    assert s["total"] == 2 and s["observing"] == 2
    assert not reg.to_frame().empty


# ===========================================================================
# 8. 符号回归发现 → 可直接作为因子使用
# ===========================================================================
def test_discovered_symbolic_factor_is_usable():
    """§6.2 产物接入 §2 因子体系：发现出的表达式可复用为因子。"""
    from framebt.discovery.symbolic import (SymbolicFactorDiscovery,
                                            evaluate_expression, spearman_ic)

    rng = np.random.default_rng(7)
    n_t, n_s = 60, 12
    close = np.cumprod(1 + 0.01 * rng.normal(size=(n_t, n_s)), axis=0) * 20
    open_ = close * (1 + 0.002 * rng.normal(size=(n_t, n_s)))
    panel = {"open": open_, "close": close, "high": np.maximum(open_, close),
             "low": np.minimum(open_, close),
             "volume": 1e6 * (1 + np.abs(rng.normal(size=(n_t, n_s))))}
    target = 3.0 * (close[1:] - close[:-1]) / close[:-1]
    target = np.vstack([target, np.full((1, n_s), np.nan)])

    d = SymbolicFactorDiscovery(population_size=60, generations=3, seed=0)
    results = d.fit(panel, target, top_k=3)
    assert results, "应至少发现一个表达式"
    best = results[0]
    assert best.expression and np.isfinite(best.abs_ic)

    # 表达式可独立复算，且 IC 与报告一致
    recomputed = evaluate_expression(best.expression, panel)
    assert np.isclose(spearman_ic(recomputed, target), best.ic, atol=1e-9)
