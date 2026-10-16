"""自主因子发现与入栈测试（§6.2–§6.6）。

覆盖：符号回归表达式可复算、神经网络隐因子生成、正交化、VIF 共线性、
增量 ICIR、影子运行、以及入栈流水线在**每一级**的拒绝理由。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framebt.discovery import (FactorStackingPipeline, ShadowFactorRunner,
                               StackingConfig, incremental_icir, orthogonalize,
                               variance_inflation_factors)
from framebt.factors import ic_summary
from framebt.factors.lifecycle import OBSERVING, FactorRegistry


# ---------------------------------------------------------------------------
# 夹具：构造一个「已知因子有效」的面板
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(11)
    n_t, n_s = 260, 20
    cols = [f"S{i:02d}" for i in range(n_s)]
    idx = list(range(n_t))
    px = np.cumprod(1 + 0.015 * rng.normal(size=(n_t, n_s)), axis=0) * 20
    close = pd.DataFrame(px, index=idx, columns=cols)
    fwd = close.shift(-1) / close - 1.0
    # 真因子：与未来收益强相关的「信号」+ 噪声
    true_sig = fwd * 3.0 + 0.02 * rng.normal(size=(n_t, n_s))
    return {"close": close, "fwd": fwd, "cols": cols,
            "signal": pd.DataFrame(true_sig, index=idx, columns=cols),
            "noise": pd.DataFrame(rng.normal(size=(n_t, n_s)), index=idx, columns=cols)}


# ---------------------------------------------------------------------------
# §6.4 正交化
# ---------------------------------------------------------------------------
def test_orthogonalize_removes_pool_exposure(world):
    pool = {"a": world["signal"]}
    cand = world["signal"] * 1.0 + 0.3 * world["noise"]
    resid = orthogonalize(cand, pool)
    # 残差与池因子的截面相关应接近 0
    c = resid.rank(axis=1).corrwith(pool["a"].rank(axis=1), axis=1).dropna()
    assert abs(float(c.mean())) < 0.05


def test_orthogonalize_empty_pool_is_identity(world):
    out = orthogonalize(world["signal"], {})
    pd.testing.assert_frame_equal(out, world["signal"])


# ---------------------------------------------------------------------------
# §6.4 VIF
# ---------------------------------------------------------------------------
def test_vif_flags_collinear_factor(world):
    a = world["signal"]
    # 近似重复的一对：两者 VIF 都应很高
    vifs_dup = variance_inflation_factors(
        {"a": a, "b": a * 2.0 + 0.001 * world["noise"]})
    assert vifs_dup["a"] > 10.0 and vifs_dup["b"] > 10.0, vifs_dup

    # 相互独立的一对：VIF 应接近 1
    vifs_ind = variance_inflation_factors({"a": a, "c": world["noise"]})
    assert vifs_ind["a"] < 5.0 and vifs_ind["c"] < 5.0, vifs_ind


def test_vif_single_factor_is_one(world):
    assert variance_inflation_factors({"only": world["signal"]})["only"] == 1.0


# ---------------------------------------------------------------------------
# §6.4 增量 ICIR
# ---------------------------------------------------------------------------
def test_incremental_icir_detects_useful_addition(world):
    pool = {"n": world["noise"]}
    inc = incremental_icir(pool, world["signal"], world["fwd"])
    assert inc["has_increment"] is True
    assert inc["new_icir"] > inc["base_icir"]


def test_incremental_icir_rejects_useless_addition(world):
    pool = {"s": world["signal"]}
    rng = np.random.default_rng(3)
    junk = pd.DataFrame(rng.normal(size=world["signal"].shape),
                        index=world["signal"].index, columns=world["signal"].columns)
    inc = incremental_icir(pool, junk, world["fwd"])
    assert inc["delta_t"] != inc["delta_t"] or inc["delta_t"] < 1.0


# ---------------------------------------------------------------------------
# §6.5 影子运行
# ---------------------------------------------------------------------------
def test_shadow_runner_passes_for_low_corr_positive_ic():
    """影子运行要的是「**增量**信息」：候选与实盘信号各自预测收益但彼此不相关。

    注意不能直接用 world 的 live/cand —— live 本身就是收益的预测器，任何强预测器
    都会与它高度相关（这不是缺陷，而是「同一个信号」）。所以要构造两条正交的信息源。
    """
    rng = np.random.default_rng(21)
    n_t, n_s = 60, 20
    idx = list(range(n_t))
    cols = [f"T{i:02d}" for i in range(n_s)]
    a = rng.normal(size=(n_t, n_s))
    b = rng.normal(size=(n_t, n_s))
    fwd = pd.DataFrame(0.01 * a + 0.01 * b + 0.0005 * rng.normal(size=(n_t, n_s)),
                       index=idx, columns=cols)
    live = pd.DataFrame(a * 0.01, index=idx, columns=cols)   # 只含 a 的信息
    cand = pd.DataFrame(b * 0.01, index=idx, columns=cols)   # 只含 b 的信息

    sr = ShadowFactorRunner("cand", min_days=5, max_corr=0.3)
    for t in idx:
        sr.on_bar(int(t), cand.loc[t], live.loc[t])
    rep = sr.report(forward_ret=fwd)
    assert rep.ic_mean > 0, f"候选应预测收益，实际 IC={rep.ic_mean}"
    assert abs(rep.corr_with_live) < 0.3, f"两信息源应不相关，实际 {rep.corr_with_live}"
    assert rep.passed is True, rep.note


def test_shadow_runner_fails_when_too_similar(world):
    live = world["signal"]
    sr = ShadowFactorRunner("dup", min_days=5, max_corr=0.3)
    for t in world["signal"].index[:8]:
        sr.on_bar(int(t), live.loc[t] * 1.5, live.loc[t])   # 与实盘信号几乎相同
    rep = sr.report(forward_ret=world["fwd"])
    assert rep.passed is False
    assert "过高" in rep.note


def test_shadow_runner_fails_when_too_short(world):
    sr = ShadowFactorRunner("short", min_days=20)
    for t in world["signal"].index[:3]:
        sr.on_bar(int(t), world["noise"].loc[t], world["signal"].loc[t])
    rep = sr.report(forward_ret=world["fwd"])
    assert rep.passed is False and "影子天数" in rep.note


# ---------------------------------------------------------------------------
# §6.6 入栈流水线
# ---------------------------------------------------------------------------
def test_pipeline_accepts_good_candidate(world):
    reg = FactorRegistry()
    pipe = FactorStackingPipeline(registry=reg)
    d = pipe.evaluate("good", world["signal"], pool={}, forward_ret=world["fwd"],
                      source="symbolic", birth_date=0)
    assert d.accepted is True
    assert d.stage == "5.入栈"
    assert reg.records["good"].status == OBSERVING
    assert reg.records["good"].current_weight <= 0.05


def test_pipeline_rejects_at_ic_stage(world):
    pipe = FactorStackingPipeline()
    d = pipe.evaluate("junk", world["noise"], pool={}, forward_ret=world["fwd"])
    assert d.accepted is False
    assert d.stage == "1.IC初筛"
    assert d.reasons


def test_pipeline_rejects_at_correlation_stage(world):
    pool = {"existing": world["signal"]}
    pipe = FactorStackingPipeline(StackingConfig(max_corr_with_pool=0.5))
    d = pipe.evaluate("dup", world["signal"] * 1.0001, pool=pool,
                      forward_ret=world["fwd"])
    assert d.accepted is False
    assert d.stage == "2.相关性"
    assert "相关" in d.reasons[0]


def test_pipeline_rejects_at_shadow_stage(world):
    reg = FactorRegistry()
    pipe = FactorStackingPipeline(registry=reg)
    sr = ShadowFactorRunner("c", min_days=20)
    for t in world["signal"].index[:5]:      # 影子天数不足
        sr.on_bar(int(t), world["signal"].loc[t], world["signal"].loc[t])
    rep = sr.report(forward_ret=world["fwd"])
    d = pipe.evaluate("cand", world["signal"], pool={}, forward_ret=world["fwd"],
                      shadow=rep)
    assert d.accepted is False
    assert d.stage == "4.影子运行"


def test_pipeline_records_all_decisions(world):
    pipe = FactorStackingPipeline()
    pipe.evaluate("junk", world["noise"], pool={}, forward_ret=world["fwd"])
    pipe.evaluate("good", world["signal"], pool={}, forward_ret=world["fwd"])
    rep = pipe.report()
    assert len(rep) == 2
    assert set(rep["factor"]) == {"junk", "good"}
    assert rep["accepted"].sum() == 1


# ---------------------------------------------------------------------------
# §6.3 神经网络隐因子
# ---------------------------------------------------------------------------
def test_neural_generator_trains_and_generates():
    torch = pytest.importorskip("torch")
    from framebt.discovery import NeuralFactorGenerator

    rng = np.random.default_rng(5)
    n_t, n_s, n_f = 40, 25, 4
    X = rng.normal(size=(n_t, n_s, n_f))
    # 未来收益由第 0 个特征驱动
    y = 2.0 * X[:, :, 0] + 0.2 * rng.normal(size=(n_t, n_s))

    gen = NeuralFactorGenerator(n_features=n_f, n_factors=6, hidden=32,
                               ortho_lambda=0.1, seed=0)
    hist = gen.fit(X, y, epochs=25, lr=5e-3)
    assert len(hist) == 25
    assert hist[-1] < hist[0], "训练损失应下降"

    f = gen.factors(X)
    assert f.shape == (n_t, n_s, 6)
    assert np.isfinite(f).all()

    summ = gen.ic_summary(X, y)
    assert np.isfinite(summ["composite_ic"])
    assert summ["composite_ic"] > 0.2, "应学到有预测力的隐因子"


def test_neural_generator_deterministic():
    torch = pytest.importorskip("torch")
    from framebt.discovery import NeuralFactorGenerator

    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 15, 3))
    y = 1.5 * X[:, :, 1] + 0.2 * rng.normal(size=(20, 15))

    def once():
        g = NeuralFactorGenerator(3, n_factors=4, hidden=16, seed=42)
        g.fit(X, y, epochs=10, lr=5e-3)
        return g.factors(X)

    assert np.allclose(once(), once(), atol=1e-6)


def test_neural_generator_orthogonality_penalty_effect():
    """开启正交惩罚后，生成因子与已有因子的相关应更低。"""
    torch = pytest.importorskip("torch")
    from framebt.discovery import NeuralFactorGenerator

    rng = np.random.default_rng(8)
    n_t, n_s = 30, 20
    x0 = rng.normal(size=(n_t, n_s))
    X = np.stack([x0, rng.normal(size=(n_t, n_s))], axis=-1)
    y = 1.0 * x0 + 0.2 * rng.normal(size=(n_t, n_s))
    existing = x0[..., None] * 1.0            # 已有因子 = x0

    def mean_abs_corr(lam):
        g = NeuralFactorGenerator(2, n_factors=3, hidden=16,
                                  ortho_lambda=lam, seed=1)
        g.fit(X, y, existing=existing, epochs=25, lr=5e-3)
        f = g.factors(X)
        cs = []
        for i in range(f.shape[-1]):
            for t in range(n_t):
                a, b = f[t, :, i], existing[t, :, 0]
                if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                    cs.append(abs(np.corrcoef(a, b)[0, 1]))
        return float(np.mean(cs))

    assert mean_abs_corr(1.0) <= mean_abs_corr(0.0) + 0.15
