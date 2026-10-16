"""CUSUM 变点检测 / 卡尔曼平滑 / 衰退概率 的回归测试（§5.2 / §5.3）。

覆盖点：
  - 结构性下滑序列 ⇒ 检出变点且方向为 down
  - 常数序列 ⇒ 不报警、无 NaN、不抛异常（退化输入）
  - NaN 空洞 ⇒ 不污染累积和、不改变变点结论
  - 卡尔曼滤波降噪且缺失观测只做预测（方差单调不减）
  - decay_probability 双向判别（明显衰退 > 0.6 / 稳定 < 0.3）
  - parameter_sensitivity 稳健性判定与自动降权建议
  - 全流程确定性（无随机数）
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from framebt.monitor.changepoint import (
    CUSUMResult,
    KalmanICFilter,
    cusum_detect,
    decay_probability,
    parameter_sensitivity,
)

# 固定种子：测试必须可复现（模块本身不用随机数，随机只用于造数据）
SEED = 7


def _shift_series(seed: int = SEED, n_before: int = 200, n_after: int = 100,
                  shift: float = 1.5) -> np.ndarray:
    """前 n_before 天标准正态，之后均值下移 shift 的结构性衰退 IC。"""
    rng = np.random.default_rng(seed)
    return np.concatenate(
        [rng.standard_normal(n_before), rng.standard_normal(n_after) - shift]
    )


def _stable_series(n: int = 200) -> np.ndarray:
    """均值恒定、无结构漂移的稳定 IC 序列（低幅交替）。"""
    out = np.empty(n, dtype=np.float64)
    out[0::2] = 0.12
    out[1::2] = -0.08
    return out


def _decayed_series(seed: int = SEED) -> np.ndarray:
    """基线 IC≈0.06，末 20 天整体转为负值 ⇒ 明显衰退。"""
    rng = np.random.default_rng(seed)
    baseline = 0.06 + rng.standard_normal(180) * 0.1
    recent = rng.standard_normal(20) * 0.1 - 0.12
    return np.concatenate([baseline, recent])


# ---------------------------------------------------------------------------
# cusum_detect
# ---------------------------------------------------------------------------
def test_cusum_detects_downward_shift_near_200():
    """t=200 起强负向漂移 ⇒ [195,215] 内出现变点，方向为 down。"""
    series = _shift_series()
    res = cusum_detect(series)

    assert isinstance(res, CUSUMResult)
    assert res.changepoints, "结构性下滑必须触发报警"
    assert any(195 <= cp <= 215 for cp in res.changepoints), res.changepoints
    assert res.direction == "down"
    assert res.s_high.shape == series.shape
    assert res.s_low.shape == series.shape
    assert np.isfinite(res.s_high).all() and np.isfinite(res.s_low).all()
    assert (res.s_high >= 0.0).all() and (res.s_low >= 0.0).all()


def test_cusum_debounce_respects_min_gap():
    """去抖：相邻报警点间隔不小于 min_gap。"""
    series = _shift_series()
    res = cusum_detect(series, min_gap=20)
    gaps = np.diff(res.changepoints)
    assert gaps.size == 0 or int(gaps.min()) >= 20


def test_cusum_constant_series_never_alarms():
    """常数序列：无变点、不报警、累积和非 NaN（退化输入安全）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # 任何告警都算失败
        res = cusum_detect(np.full(300, 0.03))

    assert res.changepoints == []
    assert res.alarm is False
    assert res.direction == "none"
    assert not np.isnan(res.s_high).any()
    assert not np.isnan(res.s_low).any()
    assert np.array_equal(res.s_high, np.zeros(300))
    assert np.array_equal(res.s_low, np.zeros(300))


def test_cusum_empty_and_all_nan_are_safe():
    """空序列 / 全 NaN：返回退化结果而非抛异常。"""
    for series in (np.array([]), np.full(50, np.nan)):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = cusum_detect(series)
        assert res.changepoints == []
        assert res.alarm is False
        assert res.direction == "none"
        assert not np.isnan(res.s_high).any()
        assert not np.isnan(res.s_low).any()


def test_cusum_nan_holes_do_not_corrupt_accumulators():
    """NaN 空洞：不污染累积和、不重置累积器、不改变变点结论。

    构造「平台 + 单点结构性下移」序列：t=150 起 10 天均值下移 0.5，随后回到平台。
    平台段偏差对参考均值恰好为 0（参考均值取全样本有效观测均值），
    因此基线累积和恒为 0、报警唯一、去抖不参与比较，比较结果只反映 NaN 处理本身。
    NaN 会改变有效观测数、从而改变 MAD 尺度，因此把阈值固定为
    「无空洞版本的阈值」再比较；并额外断言空洞当天累积和确实是「按偏差 0 递推」
    的结果（缺失行未被当成观测）。把空洞放在平台段上，此时空洞是否被当成
    「偏差 0 的观测」与「被跳过」在数值上等价，从而可做严格的逐点比较。
    """
    series = np.concatenate([
        np.zeros(150),                              # 平台
        np.full(10, -0.5),                          # 结构性下移
        np.zeros(20),                               # 回到平台
    ])

    threshold = 5.0 * _sigma_for_test(series)     # 固定为无空洞版本的阈值
    clean = cusum_detect(series, h=threshold)     # 与空洞版本使用同一阈值

    holes = np.array([5, 6, 40, 90, 120, 170])
    holed = series.copy()
    holed[holes] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = cusum_detect(holed, h=threshold)

    # 同一阈值下，变点列表与无空洞版本完全一致
    assert res.changepoints == clean.changepoints
    assert any(150 <= cp <= 165 for cp in res.changepoints)
    assert res.direction == "down"

    # 空洞当天：S = max(0, S_prev - slack)（偏差按 0 处理，而不是被当成新观测）
    slack = 0.5 * _sigma_for_test(series)
    for i in holes:
        if i == 0:
            continue
        expected = max(0.0, float(clean.s_low[i - 1]) - slack)
        assert float(res.s_low[i]) == pytest.approx(expected, abs=1e-12)

    # 空洞不改变「这些点本身不触发报警」的事实
    assert all(cp not in set(holes.tolist()) for cp in res.changepoints)


def test_robust_scale_falls_back_when_degenerate():
    """use_robust_scale：MAD 退化时回退 std，两者都不至于让常数序列报警。"""
    series = _stable_series()
    robust = cusum_detect(series, use_robust_scale=True)
    plain = cusum_detect(series, use_robust_scale=False)
    assert robust.changepoints == [] and plain.changepoints == []
    assert np.isfinite(robust.s_low).all() and np.isfinite(plain.s_low).all()


def test_cusum_rejects_bad_parameters():
    """非法参数显式报错，而不是静默产生 NaN。"""
    series = _stable_series()
    with pytest.raises(ValueError):
        cusum_detect(series, k=-1.0)
    with pytest.raises(ValueError):
        cusum_detect(series, h=float("nan"))
    with pytest.raises(ValueError):
        cusum_detect(series, min_gap=0)


# ---------------------------------------------------------------------------
# KalmanICFilter
# ---------------------------------------------------------------------------
def test_kalman_reduces_noise_rmse():
    """滤波后序列相对真实信号的 RMSE 必须严格低于原始观测。"""
    rng = np.random.default_rng(11)
    t = np.arange(400, dtype=np.float64)
    signal = 0.5 * np.sin(2.0 * np.pi * t / 60.0)
    noisy = signal + rng.standard_normal(400) * 0.5

    filtered = KalmanICFilter(process_var=1e-4, obs_var=1e-2).filter(noisy)
    finite = np.isfinite(filtered)

    rmse_raw = float(np.sqrt(np.mean((noisy[finite] - signal[finite]) ** 2)))
    rmse_filtered = float(np.sqrt(np.mean((filtered[finite] - signal[finite]) ** 2)))
    assert rmse_filtered < rmse_raw
    assert rmse_filtered < 0.8 * rmse_raw       # 显著降噪（一维常增益滤波）


def test_kalman_nan_update_only_predicts_and_variance_non_decreasing():
    """obs=NaN 不抛异常、状态保留、方差单调不减。"""
    flt = KalmanICFilter()
    for obs in (0.1, 0.12, 0.11):
        flt.update(obs)

    var_before = flt.variance
    state_before = flt.state
    out = flt.update(np.nan)

    assert out == pytest.approx(state_before)
    assert flt.variance >= var_before
    assert np.isfinite(flt.state)
    assert np.isfinite(flt.variance)

    # 连续缺失：方差继续单调不减
    prev = flt.variance
    for _ in range(5):
        flt.update(float("nan"))
        assert flt.variance >= prev
        prev = flt.variance

    # 观测回来后仍能继续收缩
    flt.update(0.1)
    assert flt.variance < prev


def test_kalman_filter_outputs_nan_at_missing_rows():
    """filter：缺失位置输出 NaN，长度与输入一致，其余位置有限。"""
    series = np.array([0.1, 0.2, np.nan, 0.15, np.nan, 0.12])
    out = KalmanICFilter().filter(series)
    assert out.shape == series.shape
    assert np.isnan(out[2]) and np.isnan(out[4])
    assert np.isfinite(out[[0, 1, 3, 5]]).all()


def test_kalman_deterministic_and_reusable():
    """同一输入两次滤波结果完全一致；reset 后可复用实例。"""
    series = _shift_series()
    flt = KalmanICFilter()
    first = flt.filter(series)
    flt.reset()
    second = flt.filter(series)
    assert np.array_equal(first, second)


# ---------------------------------------------------------------------------
# decay_probability
# ---------------------------------------------------------------------------
def test_decay_probability_high_for_decayed_series():
    """明显衰退 ⇒ P > 0.6。"""
    prob = decay_probability(_decayed_series())
    assert np.isfinite(prob)
    assert 0.0 <= prob <= 1.0
    assert prob > 0.6


def test_decay_probability_low_for_stable_series():
    """稳定序列（常数 / 低幅交替 / 无漂移噪声）⇒ P < 0.3。"""
    rng = np.random.default_rng(3)
    cases = {
        "constant": np.full(300, 0.03),
        "alternating": _stable_series(),
        "noise_no_drift": 0.02 + rng.standard_normal(200) * 0.1,
    }
    for name, series in cases.items():
        prob = decay_probability(series)
        assert np.isfinite(prob), name
        assert prob < 0.3, f"{name}: P={prob}"


def test_decay_probability_monotone_in_shift_size():
    """落差越大概率越高（单调性检查），且两档阈值语义相反。

    用**确定性**的基线 + 单点可调落差，避免随机噪声把 delta 的符号搅乱：
    无落差时 delta 恰为 0 ⇒ P 应为基线值 0.2（< 0.3），强落差 ⇒ P > 0.6。
    """
    base = np.tile(np.array([0.1, -0.1]), 90)     # 180 天、均值 0、sigma ≈ 0.148
    zero = np.tile(np.array([0.06, -0.06]), 10)   # 20 天、均值 0（无落差）

    probs = [
        decay_probability(np.concatenate([base, zero - s]))
        for s in (0.0, 0.05, 0.12, 0.30)
    ]

    assert probs[0] == pytest.approx(0.16, abs=0.06)   # delta ≈ 0 ⇒ 无衰退证据
    assert probs[0] < 0.3                              # 未衰退
    assert probs[-1] > 0.6                             # 明显衰退
    assert probs == sorted(probs)                      # 单调不减
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_decay_probability_degenerate_inputs_are_safe():
    """空 / 全 NaN / 单点 / 短序列：不抛异常、不返回 NaN。"""
    for series in (
        np.array([]),
        np.full(50, np.nan),
        np.array([0.05]),
        np.array([0.05, 0.06]),
        np.zeros(5),
    ):
        prob = decay_probability(series)
        assert np.isfinite(prob)
        assert 0.0 <= prob <= 1.0

    # 参数校验
    with pytest.raises(ValueError):
        decay_probability(_stable_series(), recent_window=0)


def test_decay_probability_uses_kalman_flag():
    """use_kalman 两个分支都必须有限且落在 [0,1]。"""
    series = _decayed_series()
    with_k = decay_probability(series, use_kalman=True)
    without = decay_probability(series, use_kalman=False)
    for prob in (with_k, without):
        assert np.isfinite(prob)
        assert 0.0 <= prob <= 1.0
    assert without > 0.6


# ---------------------------------------------------------------------------
# parameter_sensitivity
# ---------------------------------------------------------------------------
def test_parameter_sensitivity_strong_decay_recommends_downgrade():
    """强衰退：网格结论一致 ⇒ stable 且建议自动降权。"""
    out = parameter_sensitivity(_decayed_series())

    assert isinstance(out["grid"], list)
    assert len(out["grid"]) == 3 * 4
    for item in out["grid"]:
        kv, hv, alarmed, direction = item
        assert isinstance(kv, float) and isinstance(hv, float)
        assert isinstance(alarmed, bool)
        assert direction in {"up", "down", "none"}
        assert alarmed in (True, False)

    assert out["stable_across_parameters"] is True
    assert out["alarm_rate"] == pytest.approx(1.0)
    assert out["recommend_auto_downgrade"] is True


def test_parameter_sensitivity_stable_series_never_recommends():
    """稳定 / 常数序列：不得建议自动降权，也不得崩溃。"""
    for series in (_stable_series(), np.full(300, 0.03), np.full(40, np.nan), np.array([])):
        out = parameter_sensitivity(series)
        assert out["recommend_auto_downgrade"] is False
        assert 0.0 <= out["alarm_rate"] <= 1.0
        assert isinstance(out["stable_across_parameters"], bool)


def test_parameter_sensitivity_custom_grid_and_threshold():
    """自定义网格 / 门槛（含空网格报错）与报警方向一致性。"""
    out = parameter_sensitivity(_decayed_series(), k_grid=(0.5,), h_grid=(4.0, 5.0))
    assert len(out["grid"]) == 2
    assert [g[2] for g in out["grid"]] == [True, True]
    assert {g[3] for g in out["grid"]} == {"down"}

    # 门槛提高到 1.0（要求全网格报警）时结论不变
    strict = parameter_sensitivity(_decayed_series(), threshold=1.0)
    assert strict["recommend_auto_downgrade"] is True

    with pytest.raises(ValueError):
        parameter_sensitivity(_stable_series(), k_grid=())


# ---------------------------------------------------------------------------
# 确定性
# ---------------------------------------------------------------------------
def test_determinism_of_all_entry_points():
    """两个完全相同的调用必须产生完全相同的输出（无随机数）。"""
    series = _shift_series()
    r1, r2 = cusum_detect(series), cusum_detect(series)
    assert r1.changepoints == r2.changepoints
    assert r1.direction == r2.direction
    assert r1.alarm == r2.alarm
    assert np.array_equal(r1.s_high, r2.s_high)
    assert np.array_equal(r1.s_low, r2.s_low)

    assert np.array_equal(KalmanICFilter().filter(series), KalmanICFilter().filter(series))

    assert decay_probability(series) == decay_probability(series)
    assert parameter_sensitivity(series) == parameter_sensitivity(series)


# ---------------------------------------------------------------------------
# 测试内辅助：复算参考尺度（与被测模块同定义，用于「固定阈值」对照实验）
# ---------------------------------------------------------------------------
def _sigma_for_test(series: np.ndarray) -> float:
    """复算模块文档定义的稳健尺度 sigma = 1.4826 * MAD（有效观测，MAD 以中位数为中心）。"""
    x = np.asarray(series, dtype=np.float64)
    mask = np.isfinite(x)
    valid = x[mask]
    mad = float(np.median(np.abs(valid - np.median(valid))))
    sigma = 1.4826 * mad
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(valid))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    return sigma
