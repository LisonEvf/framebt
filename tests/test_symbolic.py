"""符号回归因子发现（§6.2）测试。

覆盖：
  - ``spearman_ic`` 手算可验证情形（完全单调 ⇒ +1，反向 ⇒ -1）；
  - ``evaluate_expression`` 回放手写表达式并与手算结果比对；
  - 受保护除法/对数产生 NaN 而非 inf/异常；
  - 确定性：同 seed 两次 ``fit`` 表达式列表逐位一致；
  - 可发现性：目标为数据的已知函数时，``|IC|`` 显著高于随机基线；
  - ``max_depth`` / ``max_nodes`` 约束在全部返回结果中被遵守。
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from framebt.discovery import (
    GPResult,
    SymbolicFactorDiscovery,
    evaluate_expression,
    spearman_ic,
)


# ===========================================================================
# 工具
# ===========================================================================

def _panel(rng: np.random.Generator, t: int = 120, s: int = 12) -> dict[str, np.ndarray]:
    """构造一个合成价格面板 (n_times, n_symbols)。"""
    close = np.cumsum(rng.normal(0.0, 0.02, size=(t, s)), axis=0) + 10.0
    open_ = close + rng.normal(0.0, 0.005, size=(t, s))
    volume = np.abs(rng.normal(1e6, 1e5, size=(t, s)))
    returns = rng.normal(0.0, 0.01, size=(t, s))
    return {
        "open": open_,
        "high": np.maximum(open_, close) + 0.01,
        "low": np.minimum(open_, close) - 0.01,
        "close": close,
        "volume": volume,
        "returns": returns,
    }


def _easy_problem(t: int = 400, s: int = 24, seed: int = 7):
    """目标 = 数据的已知函数：``3 * ts_delta(close, 2) + 小噪声``。

    窗口 2 属于窗口候选集 ``{2,3,5,6,10,20}``，因此该因子是可被发现的：
    ``close`` 为随机游走，``ts_delta(close, 2)`` 与目标秩相关接近 1，
    而相邻窗口（1/3/5）只共享部分增量，秩相关明显更低。
    噪声量级刻意取小（0.05），使 |IC| 接近 1 而目标又**不是**数据的确定性函数。
    """
    rng = np.random.default_rng(seed)
    close = np.cumsum(rng.normal(0.0, 0.02, size=(t, s)), axis=0) + 20.0
    open_ = close + rng.normal(0.0, 0.002, size=(t, s))
    high = np.maximum(open_, close) + 0.01
    low = np.minimum(open_, close) - 0.01
    volume = np.abs(rng.normal(1e6, 1e5, size=(t, s)))
    panel = {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    delta2 = np.full_like(close, np.nan)
    delta2[2:] = close[2:] - close[:-2]
    # 噪声标准差 0.005，远小于 delta2 自身量级（~0.028），故信号主导但仍非确定性
    target = 3.0 * delta2 + 0.005 * rng.normal(size=(t, s))
    return panel, target


# ===========================================================================
# spearman_ic
# ===========================================================================

def test_spearman_ic_perfect_monotone_is_positive_one():
    rng = np.random.default_rng(0)
    panel = _panel(rng)
    factor = panel["close"]
    assert spearman_ic(factor, factor * 3.0 + 7.0) == pytest.approx(1.0, abs=1e-12)
    # 逐行完全单调（非线性、单调递增）仍是 +1
    assert spearman_ic(factor, np.exp(factor) * 1000.0) == pytest.approx(1.0, abs=1e-12)


def test_spearman_ic_reversed_is_negative_one():
    rng = np.random.default_rng(1)
    factor = _panel(rng)["close"]
    assert spearman_ic(factor, -2.0 * factor) == pytest.approx(-1.0, abs=1e-12)


def test_spearman_ic_hand_checkable_tiny_case():
    # 行 0：完全同序 ⇒ +1；行 1：完全反序 ⇒ -1；均值 = 0
    factor = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    target = np.array([[10.0, 20.0, 30.0], [30.0, 20.0, 10.0]])
    assert spearman_ic(factor, target) == pytest.approx(0.0, abs=1e-12)

    # 单行：2/3 同序 ⇒ 秩相关 0.5
    f1 = np.array([[1.0, 2.0, 3.0]])
    t1 = np.array([[1.0, 3.0, 2.0]])
    # 秩：f=[1,2,3]，t=[1,3,2] ⇒ Pearson([1,2,3],[1,3,2]) = 0.5
    assert spearman_ic(f1, t1) == pytest.approx(0.5, abs=1e-12)


def test_spearman_ic_ignores_nan_and_needs_enough_samples():
    # NaN 格被忽略：剩余三格完全同序 ⇒ +1
    factor = np.array([[1.0, 2.0, np.nan, 4.0]])
    target = np.array([[1.0, 2.0, 5.0, 4.0]])
    assert spearman_ic(factor, target) == pytest.approx(1.0, abs=1e-12)

    # 只忽略 factor 的 NaN（target 在别处为 NaN 同样被忽略），且 NaN 不参与并列判定
    f2 = np.array([[np.nan, 3.0, 1.0, 2.0]])
    t2 = np.array([[9.0, 30.0, 10.0, 20.0]])
    assert spearman_ic(f2, t2) == pytest.approx(1.0, abs=1e-12)

    # 有效样本 < 3 ⇒ 该行不参与 ⇒ 全无有效行 ⇒ nan
    f3 = np.array([[np.nan, np.nan, 1.0, 2.0]])
    t3 = np.array([[np.nan, np.nan, 1.0, 2.0]])
    assert math.isnan(spearman_ic(f3, t3))

    # 常数截面（方差为 0）⇒ 秩相关无定义 ⇒ 跳过
    f4 = np.array([[1.0, 1.0, 1.0, 1.0]])
    assert math.isnan(spearman_ic(f4, np.array([[1.0, 2.0, 3.0, 4.0]])))


def test_spearman_ic_raises_on_shape_mismatch():
    with pytest.raises(ValueError):
        spearman_ic(np.zeros((3, 4)), np.zeros((3, 5)))


def test_spearman_ic_ties_and_row_wise_masking():
    """并列取平均秩；且各行有效样本数不同时逐行独立计算。"""
    # 两行有效样本数不同：行 0 有 4 个、行 1 只有 3 个有效格
    factor = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, np.nan]])
    target = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, np.nan]])
    # 行 0 完全同序 ⇒ +1；行 1：[1,2,3] vs [4,1,2] 秩 [1,2,3] vs [3,1,2] ⇒ -0.5
    # 故均值 = (1.0 + (-0.5)) / 2 = 0.25（两行都被纳入，不因样本数不同而丢行）
    assert spearman_ic(factor, target) == pytest.approx(0.25, abs=1e-12)

    # 并列：factor 全相等 ⇒ 该行方差为 0 ⇒ 跳过，只留行 1 的 +1
    f_tie = np.array([[5.0, 5.0, 5.0], [1.0, 2.0, 3.0]])
    t_tie = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    assert spearman_ic(f_tie, t_tie) == pytest.approx(1.0, abs=1e-12)

    # 单调（非线性）变换不改变秩相关：使用不会溢出的单调变换
    rng = np.random.default_rng(5)
    x = rng.normal(size=(6, 9))
    assert spearman_ic(x, np.sign(x) * np.log1p(np.abs(x)) * 10.0) == pytest.approx(1.0, abs=1e-12)
    # NaN 只出现在部分行时，逐行独立判定样本数是否足够
    x2 = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, np.nan, np.nan]])
    y2 = np.array([[4.0, 3.0, 2.0, 1.0], [9.0, 8.0, np.nan, np.nan]])
    assert spearman_ic(x2, y2) == pytest.approx(-1.0, abs=1e-12)


# ===========================================================================
# evaluate_expression
# ===========================================================================

def test_evaluate_expression_round_trip_ts_delta_hand_computed():
    # 每步 +3（reshape 后仍为 3 列等差），便于手算差分
    close = 3.0 * np.arange(1.0, 13.0).reshape(4, 3)
    panel = {"close": close}
    out = evaluate_expression("ts_delta(close,1)", panel)
    assert out.shape == (4, 3)
    assert np.all(np.isnan(out[0]))
    assert np.allclose(out[1:], 9.0 * np.ones((3, 3)))
    # 与手算等价
    assert np.allclose(out[1:, :], close[1:, :] - close[:-1, :])
    # ts_delay 手算
    delay = evaluate_expression("ts_delay(close,2)", panel)
    assert np.all(np.isnan(delay[:2]))
    assert np.allclose(delay[2:], close[:-2])


def test_evaluate_expression_arithmetic_and_parser_equivalence():
    rng = np.random.default_rng(3)
    panel = _panel(rng, t=20, s=6)
    a = evaluate_expression("div(sub(close,open),open)", panel)
    b = (panel["close"] - panel["open"]) / panel["open"]
    assert np.allclose(a, b, equal_nan=True)

    # 一元负号 / 数字字面量 / 中缀运算等价的解析
    c = evaluate_expression("neg(add(close,-0.5000))", panel)
    d = evaluate_expression("-(close + -0.5)", panel)
    assert np.allclose(c, d, equal_nan=True)

    # 空白容错
    e = evaluate_expression("  ts_mean( close , 3 ) ".strip(), panel)
    f = evaluate_expression("ts_mean(close,3)", panel)
    assert np.allclose(e, f, equal_nan=True)


def test_evaluate_expression_rejects_untrusted_syntax():
    rng = np.random.default_rng(4)
    panel = _panel(rng, t=6, s=3)
    for bad in (
        "__import__('os').system('echo hi')",
        "close.__class__",
        "close[0]",
        "open if close else 0",
        "close > 0",
        "2 ** 64",
        "(lambda: 1)()",
        "ts_mean(close)",          # 缺窗口
        "ts_corr(close,open,2,3)", # 参数过多
        "ts_mean(close,close)",    # 窗口必须是字面量
        "unknown_fn(close)",
        "close; close",
    ):
        with pytest.raises(Exception):
            evaluate_expression(bad, panel)


def test_evaluate_expression_unknown_variable_raises():
    panel = {"close": np.ones((4, 2))}
    with pytest.raises(KeyError):
        evaluate_expression("open", panel)


# ===========================================================================
# 受保护数学
# ===========================================================================

def test_protected_division_returns_nan_not_inf():
    denom = np.array([[0.0, 1e-15, 2.0], [np.nan, -1e-300, -4.0]])
    num = np.ones((2, 3))
    panel = {"num": num, "den": denom}
    out = evaluate_expression("div(num,den)", panel)
    assert np.all(np.isnan(out[:, 0]))              # 分母精确 0
    assert np.all(np.isnan(out[:, 1]))              # 分母 < 1e-12
    assert np.all(np.isnan(out[1, :2]))             # 分母 NaN
    assert out[0, 2] == pytest.approx(0.5)
    assert out[1, 2] == pytest.approx(-0.25)
    assert not np.any(np.isinf(out))
    assert np.all(np.isnan(out) | np.isfinite(out))


def test_protected_log_sqrt_and_inv_never_emit_inf():
    panel = {
        "x": np.array([[0.0, 1e-20, -1e-20], [1.0, -1.0, np.nan]]),
    }
    log_out = evaluate_expression("log(x)", panel)
    assert np.all(np.isnan(log_out[0, :3]))                        # 0 与 ±1e-20 ⇒ NaN
    assert log_out[1, 0] == pytest.approx(0.0)                     # log|1| = 0
    assert log_out[1, 1] == pytest.approx(0.0)                     # log|-1| = 0，不抛错
    assert np.isnan(log_out[1, 2])

    inv_out = evaluate_expression("inv(x)", panel)
    assert np.all(np.isnan(inv_out[0, :3]))
    assert inv_out[1, 0] == pytest.approx(1.0)
    assert inv_out[1, 1] == pytest.approx(-1.0)

    sqrt_out = evaluate_expression("sqrt(x)", panel)
    assert not np.any(np.isinf(sqrt_out))
    assert sqrt_out[1, 1] == pytest.approx(1.0)                    # sqrt|x|，负数不报错

    for expr in ("div(x,sub(x,x))", "mul(inv(x),x)", "log(mul(x,x))"):
        out = evaluate_expression(expr, panel)
        assert not np.any(np.isinf(out))


def test_ts_rank_is_in_unit_interval_and_hand_checkable():
    x = np.array([[5.0], [1.0], [3.0], [9.0]])
    out = evaluate_expression("ts_rank(x,2)", {"x": x})
    assert np.isnan(out[0, 0])
    # t=1 窗口 [5,1]，末值 1 最小 ⇒ 0
    assert out[1, 0] == pytest.approx(0.0)
    # t=2 窗口 [1,3]，末值 3 最大 ⇒ 1
    assert out[2, 0] == pytest.approx(1.0)
    # t=3 窗口 [3,9]，末值 9 最大 ⇒ 1
    assert out[3, 0] == pytest.approx(1.0)
    assert np.all((out[np.isfinite(out)] >= 0.0) & (out[np.isfinite(out)] <= 1.0))


def test_ts_corr_and_cov_hand_checkable():
    a = np.array([[1.0], [2.0], [3.0], [4.0]])
    b = np.array([[2.0], [4.0], [6.0], [8.0]])   # 与 a 完全线性相关
    panel = {"a": a, "b": b}
    corr = evaluate_expression("ts_corr(a,b,3)", panel)
    assert np.isnan(corr[0, 0]) and np.isnan(corr[1, 0])
    assert corr[2, 0] == pytest.approx(1.0)
    assert corr[3, 0] == pytest.approx(1.0)

    c = evaluate_expression("ts_cov(a,a,3)", panel)
    # 窗口 [2,3,4] 的样本方差（ddof=1）= 1.0
    assert c[3, 0] == pytest.approx(1.0)


# ===========================================================================
# fit：确定性
# ===========================================================================

def test_fit_is_deterministic_for_same_seed():
    panel, target = _easy_problem(t=120, s=10, seed=11)
    a = SymbolicFactorDiscovery(population_size=60, generations=3, seed=42, tournament_size=5)
    b = SymbolicFactorDiscovery(population_size=60, generations=3, seed=42, tournament_size=5)
    ra = a.fit(panel, target, top_k=10)
    rb = b.fit(panel, target, top_k=10)
    assert [r.expression for r in ra] == [r.expression for r in rb]
    assert [r.abs_ic for r in ra] == [r.abs_ic for r in rb]
    assert a.best is not None and b.best is not None
    assert a.best.expression == b.best.expression


def test_fit_different_seed_explores_differently_but_still_valid():
    panel, target = _easy_problem(t=80, s=8, seed=12)
    r1 = SymbolicFactorDiscovery(population_size=50, generations=2, seed=1).fit(panel, target)
    r2 = SymbolicFactorDiscovery(population_size=50, generations=2, seed=2).fit(panel, target)
    assert r1 and r2
    for res in r1 + r2:
        assert isinstance(res, GPResult)
        assert math.isfinite(res.abs_ic)


# ===========================================================================
# fit：可发现性
# ===========================================================================

def test_fit_discovers_known_factor_and_beats_random_baseline():
    panel, target = _easy_problem(t=400, s=24, seed=7)
    engine = SymbolicFactorDiscovery(
        population_size=300, generations=4, max_depth=4, max_nodes=20, seed=0
    )
    results = engine.fit(panel, target, top_k=5)

    assert results, "fit 未返回任何结果"
    best = engine.best
    assert best is not None
    # 目标与数据强相关：最优 |IC| 应远超 0.5
    assert best.abs_ic > 0.5, f"best.abs_ic={best.abs_ic}"
    # 结果按 |IC| 降序
    assert [r.abs_ic for r in results] == sorted((r.abs_ic for r in results), reverse=True)
    # 优于第 0 代随机个体适应度的中位数
    gen0 = engine.generation0_fitness
    median0 = float(np.median(gen0[np.isfinite(gen0)]))
    assert best.abs_ic > median0, f"best={best.abs_ic} median_gen0={median0}"
    # 结构合理：|IC| 接近 1 说明确实找回了 ts_delta(close,2) 的单调变换
    assert best.abs_ic > 0.9


def test_fit_can_recover_expression_shape_on_easy_problem():
    """最优表达式应是 close 的短程时序差分（或其单调变换）。"""
    panel, target = _easy_problem(t=300, s=20, seed=3)
    engine = SymbolicFactorDiscovery(
        population_size=200, generations=3, max_depth=4, max_nodes=20, seed=5
    )
    results = engine.fit(panel, target, top_k=10)
    assert results
    exprs = [r.expression for r in results]
    # 语义检查（不假设具体下标）：最优因子与其 ts_delta 形式高度一致
    best_factor = evaluate_expression(exprs[0], panel)
    ref = evaluate_expression("ts_delta(close,2)", panel)
    ic_pair = spearman_ic(evaluate_expression("rank(close)", panel), panel["close"])
    assert abs(spearman_ic(best_factor, ref)) > 0.9
    assert abs(ic_pair) > 0.9  # rank 保序


# ===========================================================================
# fit：约束与去重
# ===========================================================================

def test_fit_respects_max_depth_and_max_nodes():
    panel, target = _easy_problem(t=150, s=8, seed=21)
    for max_depth, max_nodes in ((3, 8), (4, 12), (6, 20)):
        engine = SymbolicFactorDiscovery(
            population_size=80,
            generations=3,
            max_depth=max_depth,
            max_nodes=max_nodes,
            seed=0,
        )
        results = engine.fit(panel, target, top_k=20)
        assert results
        for r in results:
            assert r.depth <= max_depth, (max_depth, max_nodes, r)
            assert r.nodes <= max_nodes, (max_depth, max_nodes, r)
        assert engine.best is not None
        assert engine.best.depth <= max_depth
        assert engine.best.nodes <= max_nodes
        # 返回的表达式必须能回放（与引擎同一语法）
        for r in results:
            out = evaluate_expression(r.expression, panel)
            assert out.shape == panel["close"].shape


def test_fit_deduplicates_expressions_and_skips_nan_ic():
    panel, target = _easy_problem(t=100, s=6, seed=31)
    results = SymbolicFactorDiscovery(
        population_size=120, generations=2, seed=9
    ).fit(panel, target, top_k=50)
    exprs = [r.expression for r in results]
    assert len(exprs) == len(set(exprs))
    assert all(math.isfinite(r.abs_ic) for r in results)
    assert all(r.abs_ic >= 0.0 for r in results)


def test_fit_all_nan_factor_is_discarded():
    """全 NaN 因子必须被剔除（fitness=-inf）而不是崩溃或污染结果。"""
    # 1) 目标全 NaN ⇒ 没有任何有效 IC ⇒ 结果为空，best 为 None
    panel, _ = _easy_problem(t=60, s=6, seed=41)
    target_nan = np.full_like(panel["close"], np.nan)
    engine_nan = SymbolicFactorDiscovery(population_size=40, generations=2, seed=0)
    results_nan = engine_nan.fit(panel, target_nan, top_k=10)
    assert results_nan == []
    assert engine_nan.best is None
    assert np.all(np.isneginf(engine_nan.generation0_fitness))
    assert math.isnan(spearman_ic(panel["close"], target_nan))

    # 2) 面板本身全 NaN ⇒ 同样无有效因子，且不抛异常
    panel_all_nan = {k: np.full_like(v, np.nan) for k, v in panel.items()}
    engine2 = SymbolicFactorDiscovery(population_size=40, generations=2, seed=0)
    assert engine2.fit(panel_all_nan, panel_all_nan["close"], top_k=10) == []
    assert engine2.best is None

    # 3) 正常面板下，所有返回结果都必须有有限 |IC|
    engine3 = SymbolicFactorDiscovery(population_size=40, generations=2, seed=0)
    results = engine3.fit(panel, panel["close"], top_k=10)
    for r in results:
        assert math.isfinite(r.abs_ic)


def test_fit_validates_shapes_and_params():
    panel, target = _easy_problem(t=20, s=5, seed=51)
    engine = SymbolicFactorDiscovery(population_size=20, generations=1, seed=0)
    with pytest.raises(ValueError):
        engine.fit({}, target)
    with pytest.raises(ValueError):
        engine.fit(panel, target[:, :2])
    with pytest.raises(ValueError):
        engine.fit({"a": panel["close"], "b": panel["close"][:, :2]}, target)
    with pytest.raises(ValueError):
        engine.fit(panel, target, top_k=0)
    with pytest.raises(ValueError):
        SymbolicFactorDiscovery(function_set=["nope"])
    with pytest.raises(ValueError):
        SymbolicFactorDiscovery(population_size=1)
    with pytest.raises(RuntimeError):
        SymbolicFactorDiscovery().generation0_fitness


def test_fit_rejects_unknown_function_set_member_and_keeps_literals_valid():
    """窗口算子必须携带整数字面量；function_set 只接受白名单算子。"""
    panel, target = _easy_problem(t=80, s=6, seed=61)
    engine = SymbolicFactorDiscovery(population_size=60, generations=2, seed=0)
    results = engine.fit(panel, target, top_k=30)
    assert results
    # 每个结果的表达式都能被 evaluate_expression 回放（与引擎同语法）
    for r in results:
        assert evaluate_expression(r.expression, panel).shape == panel["close"].shape
    # 限制 function_set 后不应再出现被禁算子
    restricted = SymbolicFactorDiscovery(
        population_size=40, generations=2, seed=0, function_set=["ts_delta", "sub", "abs"]
    )
    res2 = restricted.fit(panel, target, top_k=10)
    assert res2
    for r in res2:
        for forbidden in ("ts_mean(", "ts_corr(", "rank(", "log(", "sqrt("):
            assert forbidden not in r.expression


def test_symbolic_engine_uses_no_global_rng():
    """同 seed 的结果不受全局 numpy 随机状态影响。"""
    panel, target = _easy_problem(t=80, s=8, seed=71)
    np.random.seed(1234)
    a = SymbolicFactorDiscovery(population_size=50, generations=2, seed=3).fit(panel, target)
    np.random.seed(9999)
    np.random.random(1000)
    b = SymbolicFactorDiscovery(population_size=50, generations=2, seed=3).fit(panel, target)
    assert [r.expression for r in a] == [r.expression for r in b]


# ===========================================================================
# 性能（守卫性断言，阈值宽松）
# ===========================================================================

def test_fit_is_fast_enough_for_realistic_panel():
    panel, target = _easy_problem(t=500, s=24, seed=7)
    engine = SymbolicFactorDiscovery(population_size=200, generations=3, seed=0)
    start = time.perf_counter()
    results = engine.fit(panel, target)
    elapsed = time.perf_counter() - start
    assert results
    assert elapsed < 60.0, f"fit 过慢：{elapsed:.2f}s"
