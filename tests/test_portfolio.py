"""组合层测试：在线权重优化 / 门控 / 因子合成与目标持仓。"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framebt.portfolio import (MarketStateGating, OnlineWeightOptimizer,
                               composite_score, select_portfolio, turnover,
                               weights_to_shares)


# ---------------------------------------------------------------------------
# 在线权重优化（§3.1 / §3.2）
# ---------------------------------------------------------------------------
def _xy(n_stocks=40, k=4, seed=0, signal=1.0):
    """构造一个「因子 j 对未来收益有预测力」的截面。"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_stocks, k))
    y = signal * X[:, 0] + 0.1 * rng.normal(size=n_stocks)
    return X, y


def test_optimizer_learns_predictive_factor():
    """只有第 0 个因子有预测力时，其权重应显著高于其余。"""
    opt = OnlineWeightOptimizer(4, lr=0.01, lambda_turnover=0.0, lambda_l1=0.0,
                                normalize="none")
    for i in range(60):
        X, y = _xy(seed=i)
        opt.observe(X, y)
    w = opt.weights
    assert w[0] == max(w), f"应把最大权重给预测因子，实际 {w}"
    assert w[0] > 0


def test_optimizer_deterministic():
    """I3：同输入序列 ⇒ 权重轨迹逐位相同（优化器本身不含随机数）。"""
    a = OnlineWeightOptimizer(4, lr=0.01)
    b = OnlineWeightOptimizer(4, lr=0.01)
    for i in range(30):
        X, y = _xy(seed=i)
        a.observe(X, y)
        b.observe(X, y)
    assert np.array_equal(a.weights, b.weights)
    assert [s.ic for s in a.history] == [s.ic for s in b.history]


def test_optimizer_l1_normalization():
    opt = OnlineWeightOptimizer(5, lr=0.01, normalize="l1")
    for i in range(20):
        opt.observe(*_xy(k=5, seed=i))
    assert np.isclose(np.abs(opt.weights).sum(), 1.0, atol=1e-9)


def test_optimizer_turnover_penalty_reduces_weight_jumps():
    """λ₁ 越大，权重单步变动应越小（换手惩罚生效）。"""
    def run(lam):
        opt = OnlineWeightOptimizer(4, lr=0.01, lambda_turnover=lam, normalize="none")
        turns = []
        for i in range(40):
            st = opt.observe(*_xy(seed=i))
            turns.append(st.turnover)
        return float(np.mean(turns[5:]))

    assert run(0.0) > run(2.0), "换手惩罚应压低权重变动"


def test_optimizer_l1_penalty_shrinks_weights():
    def norm(lam):
        opt = OnlineWeightOptimizer(6, lr=0.01, lambda_l1=lam, normalize="none")
        for i in range(40):
            opt.observe(*_xy(k=6, seed=i))
        return float(np.abs(opt.weights).sum())

    assert norm(0.0) > norm(0.5), "L1 正则应压缩权重绝对值"


def test_optimizer_loss_components():
    opt = OnlineWeightOptimizer(3, lambda_turnover=0.1, lambda_l1=0.01)
    opt.observe(*_xy(k=3, seed=1))
    v = opt.loss()
    assert np.isfinite(v)


def test_optimizer_rejects_bad_shape():
    opt = OnlineWeightOptimizer(3)
    with pytest.raises(ValueError):
        opt.observe(np.zeros((10, 5)), np.zeros(10))


def test_optimizer_top_factors_sorted():
    opt = OnlineWeightOptimizer(4, lr=0.01)
    for i in range(30):
        opt.observe(*_xy(seed=i))
    names = ["a", "b", "c", "d"]
    top = opt.top_factors(names, 4)
    assert [abs(w) for _n, w in top] == sorted([abs(w) for _n, w in top], reverse=True)
    assert set(opt.weights_dict(names)) == set(names)


# ---------------------------------------------------------------------------
# 门控（§3.3）
# ---------------------------------------------------------------------------
def test_gating_softmax_sums_to_one():
    g = MarketStateGating(6, n_state=3, hidden=8, seed=0)
    for state in ([0.1, 0.5, 0.2], [0.9, 0.1, 1.0], [0.0, 0.0, 0.0]):
        gates = g.gates(np.array(state))
        assert np.isclose(gates.sum(), 1.0)
        assert (gates >= 0).all()


def test_gating_temperature_controls_concentration():
    """温度越低，门控分配越集中（熵更小）。"""
    def ent(tau):
        g = MarketStateGating(8, n_state=3, hidden=8, temperature=tau, seed=1)
        gates = g.gates(np.array([0.7, 0.3, 0.5]))
        return float(-(gates * np.log(gates + 1e-12)).sum())

    assert ent(0.2) < ent(5.0)


def test_gating_training_reduces_loss():
    """用同一监督信号反复训练，交叉熵应下降。"""
    g = MarketStateGating(6, n_state=3, hidden=8, lr=0.05, seed=0)
    state = np.array([0.6, 0.4, 0.3])
    target = np.array([0.9, 0.5, -0.2, 0.1, 0.0, -0.4])
    first = g.observe(state, target)
    for _ in range(30):
        last = g.observe(state, target)
    assert last < first, f"损失未下降：{first:.4f} → {last:.4f}"


def test_gating_combine_normalizes():
    g = MarketStateGating(4, n_state=3, hidden=8, seed=0)
    base = np.array([0.4, 0.3, 0.2, 0.1])
    w = g.combine(base, np.array([0.5, 0.5, 0.2]))
    assert np.isclose(np.abs(w).sum(), 1.0)
    assert len(w) == 4


def test_gating_market_state_columns():
    idx, cols = list(range(60)), ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(0)
    close = pd.DataFrame(20 + np.cumsum(rng.normal(size=(60, 5)), axis=0),
                         index=idx, columns=cols)

    class P:
        pass

    p = P()
    p.close = close
    st = MarketStateGating.market_state(p, vol_window=10, lookback=30)
    assert list(st.columns) == ["vol_pct", "breadth", "dispersion"]
    assert st["breadth"].dropna().between(0, 1).all()


def test_gating_state_vector_handles_nan():
    g = MarketStateGating(4, n_state=3)
    v = g.state_vector(pd.Series({"vol_pct": np.nan, "breadth": 0.5, "dispersion": np.nan}))
    assert np.all(np.isfinite(v))


# ---------------------------------------------------------------------------
# 合成与目标持仓
# ---------------------------------------------------------------------------
def test_composite_score_weighted():
    idx, cols = [0, 1], ["A", "B"]
    f1 = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], index=idx, columns=cols)
    f2 = pd.DataFrame([[3.0, 0.0], [3.0, 0.0]], index=idx, columns=cols)
    s = composite_score({"f1": f1, "f2": f2}, {"f1": 1.0, "f2": 1.0})
    # A: (1+3)/2 = 2 ; B: (2+0)/2 = 1
    assert np.isclose(s.loc[0, "A"], 2.0) and np.isclose(s.loc[0, "B"], 1.0)


def test_composite_score_skips_missing_factor():
    idx, cols = [0], ["A", "B"]
    f1 = pd.DataFrame([[1.0, 2.0]], index=idx, columns=cols)
    f2 = pd.DataFrame([[np.nan, np.nan]], index=idx, columns=cols)
    s = composite_score({"f1": f1, "f2": f2}, {"f1": 1.0, "f2": 1.0})
    assert np.isclose(s.loc[0, "A"], 1.0)   # 只由 f1 贡献，且被归一


def test_select_portfolio_long():
    row = pd.Series({"A": 5.0, "B": 4, "C": 3, "D": 2, "E": 1})
    t = select_portfolio(row, top_k=3, mode="long", gross=1.0)
    assert set(t) == {"A", "B", "C"}
    assert np.isclose(sum(t.values()), 1.0)


def test_select_portfolio_buffer_reduces_turnover():
    """缓冲带：排名边缘小幅变动时不应换仓。"""
    cols = list("ABCDEFGH")
    # 原始：A..H 递减
    row1 = pd.Series({c: 8.0 - i for i, c in enumerate(cols)})
    t1 = select_portfolio(row1, top_k=3, mode="long")
    # 新排名：原第 4 名 D 微升到第 3，C 掉到第 4（边界附近互换）
    row2 = row1.copy()
    row2["C"], row2["D"] = row1["D"] + 0.1, row1["C"] - 0.1
    no_buffer = select_portfolio(row2, top_k=3, mode="long")
    with_buffer = select_portfolio(row2, top_k=3, mode="long",
                                   current=set(t1), buffer_mult=2.0)
    assert turnover(no_buffer, t1) > 0
    assert turnover(with_buffer, t1) == 0, "缓冲带应避免边界互换导致的换手"


def test_select_portfolio_long_short():
    # A=0 最低 … F=5 最高
    row = pd.Series({c: float(i) for i, c in enumerate("ABCDEF")})
    t = select_portfolio(row, top_k=2, mode="long_short", gross=1.0)
    assert t["F"] > 0 and t["E"] > 0      # 多头 = 打分最高
    assert t["A"] < 0 and t["B"] < 0      # 空头 = 打分最低
    assert np.isclose(sum(t.values()), 0.0, atol=1e-9)


def test_select_portfolio_empty():
    assert select_portfolio(pd.Series(dtype=float), top_k=3) == {}


def test_turnover_metric():
    assert np.isclose(turnover({"A": 0.5, "B": 0.5}, {"A": 1.0}), 0.5)
    assert turnover({}, {}) == 0.0


def test_weights_to_shares_lot_rounding():
    prices = {"A": 10.0, "B": 3.0, "C": 7.0}
    sh = weights_to_shares({"A": 0.5, "B": 0.25, "C": 0.25}, 100_000.0, prices, lot=100)
    # A: 0.5*100000/10 = 5000 → 5000
    assert sh["A"] == 5000
    # B: 0.25*100000/3 = 8333.3 → 8300（整手向下）
    assert sh["B"] == 8300
    assert sh["B"] % 100 == 0


def test_weights_to_shares_skips_bad_price():
    sh = weights_to_shares({"A": 0.5, "B": 0.5}, 100_000.0,
                           {"A": 10.0, "B": 0.0}, lot=100)
    assert "B" not in sh and sh.get("A") == 5000
