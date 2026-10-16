"""Changepoint：CUSUM 变点检测与因子衰退判定（§5.2 / §5.3）。

本模块回答一个问题：**因子 IC 序列是否已经开始衰退？**
  1. ``cusum_detect``       §5.2 双侧 CUSUM 变点检测（上/下侧累积和 + 去抖）
  2. ``KalmanICFilter``     §5.2 一维卡尔曼平滑，抑制单日噪声导致的假报警
  3. ``decay_probability``  §5.3 因子「正在衰退」的概率（[0,1]）
  4. ``parameter_sensitivity`` §5.3 (k,h) 网格敏感性：自动降权前置条件

设计约束（与仓库其余模块一致）：
  - 只依赖 **numpy + 标准库**（不引入 sklearn / statsmodels / ruptures）；
  - **确定性**：不含任何随机数、不使用随机初始化；
  - **NaN 安全**：IC 序列常因停牌/新股/缺失因子暴露而出现空洞，
    缺失观测不得污染累积和、不得重置累积器、不得破坏卡尔曼状态；
  - **无告警/无 inf**：所有除法都做保护，退化输入返回确定性的退化结果。

------------------------------------------------------------------
CUSUM 约定（k / h 均以 **标准差 sigma 为单位**）
------------------------------------------------------------------
设 x 为 IC 序列，参考水平 mu = mean(有效观测)，稳健尺度

    sigma = 1.4826 * median(|x - median(x)|)   （use_robust_scale=True，仅在有效观测上）

若 MAD 退化（为 0 / 非有限）则回退到标准差 std(x - mu)；若仍退化则回退为 1.0。
松弛量 slack = k * sigma，决策阈值 thr = h * sigma（默认 k=0.5, h=5.0，与设计文档一致）。

双侧累积和（标准 CUSUM 递推）：

    S_high[t] = max(0, S_high[t-1] + (x[t] - mu) - slack)   # 上侧：水平抬升
    S_low [t] = max(0, S_low [t-1] - (x[t] - mu) - slack)   # 下侧：水平下滑（衰退）

某一侧越过阈值 thr 即触发报警；触发后该侧（以及同时触发的另一侧）归零重新累积，
因此 ``changepoints`` 是这些「报警起点」经 ``min_gap`` 去抖后的索引。

缺失值语义：把缺失行替换为参考水平 mu，即该行的偏差恰为 0
（递推增量 = -slack，只会让累积和单调下降并被 0 截断，**不会重置、不会虚增**）。
由于 mu / sigma / MAD 全部只在有效观测上估计，这保证「有 NaN 空洞」与
「去掉这些空洞」两个序列给出**完全相同**的变点列表与累积和。

方向判定：``direction == "down"`` 表示 S_low 报警（IC 下滑 ⇒ 因子衰退），
``"up"`` 表示 S_high 报警（IC 抬升），``"none"`` 表示序列末端两侧均未处于报警态。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CUSUMResult",
    "cusum_detect",
    "KalmanICFilter",
    "decay_probability",
    "parameter_sensitivity",
]

# MAD → 正态一致尺度因子（1 / Φ⁻¹(0.75)）
_MAD_TO_SIGMA = 1.4826
# 尺度退化（有限但为 0）时的兜底尺度
_FALLBACK_SCALE = 1.0
# z 分数兜底尺度下限：避免极小波动把「标准误」压成 0 而让 z 爆炸
_Z_SCALE_FLOOR = 0.25
# decay_probability 的 logistic 斜率与中点（见函数 docstring 的阈值语义）
_LOGIT_SLOPE = 4.0
_LOGIT_CENTER = 0.5


# ---------------------------------------------------------------------------
# 输入校验 / 基础工具
# ---------------------------------------------------------------------------
def _as_1d(series: np.ndarray) -> np.ndarray:
    """把输入规整为 float64 一维副本（不改动调用方数据）。"""
    arr = np.asarray(series, dtype=np.float64)
    if arr.ndim != 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise ValueError(f"IC 序列必须是一维数组，收到 shape={np.shape(series)}")
    return np.array(arr, dtype=np.float64, copy=True)


def _finite_positive(value: float, name: str, *, allow_zero: bool = True) -> float:
    """参数校验：必须是有限数，且按 ``allow_zero`` 决定是否允许 0。"""
    val = float(value)
    if not math.isfinite(val):
        raise ValueError(f"{name} 必须是有限数，收到 {value!r}")
    if allow_zero:
        if val < 0.0:
            raise ValueError(f"{name} 必须 >= 0，收到 {value!r}")
    elif val <= 0.0:
        raise ValueError(f"{name} 必须 > 0，收到 {value!r}")
    return val


def _robust_scale(x: np.ndarray, mask: np.ndarray) -> float:
    """稳健尺度：1.4826 * MAD(有效观测)，退化时回退 std，再退化回 1.0。

    MAD 以 ``median(x)`` 为中心，与累积和的参考水平同源（均为稳健统计量）。
    有效观测不足 2 个时直接回退到兜底尺度。
    """
    if int(mask.sum()) < 2:
        return _FALLBACK_SCALE
    valid = x[mask]
    med = float(np.median(valid))
    mad = float(np.median(np.abs(valid - med)))
    sigma = _MAD_TO_SIGMA * mad
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = float(np.std(valid))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = _FALLBACK_SCALE
    return sigma


def _plain_scale(x: np.ndarray, mask: np.ndarray) -> float:
    """非稳健尺度：有效观测的总体标准差，退化时回退 1.0。"""
    if int(mask.sum()) < 2:
        return _FALLBACK_SCALE
    sigma = float(np.std(x[mask]))
    if not math.isfinite(sigma) or sigma <= 0.0:
        sigma = _FALLBACK_SCALE
    return sigma


def _direction_of(res: CUSUMResult) -> str:
    """报警方向：优先看序列末端是否仍处报警态，末端已复位则回看最后一次报警。"""
    if res.direction in ("up", "down"):
        return res.direction
    if not res.changepoints:
        return "none"
    last = res.changepoints[-1]
    if float(res.s_low[last]) > 0.0 and float(res.s_high[last]) <= 0.0:
        return "down"
    return "up"


# ---------------------------------------------------------------------------
# §5.2 / §5.3 公共 API
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CUSUMResult:
    """CUSUM 检测结果。

    changepoints：触发报警的索引（已按 ``min_gap`` 去抖）。
    s_high / s_low：上侧 / 下侧累积和（与输入等长，无 NaN）。
    alarm：序列**末端**是否处于报警态。
    direction："up"（上侧报警）/ "down"（下侧报警）/ "none"。
    """

    changepoints: list[int]
    s_high: np.ndarray
    s_low: np.ndarray
    alarm: bool
    direction: str


def cusum_detect(
    series: np.ndarray,
    k: float = 0.5,
    h: float = 5.0,
    use_robust_scale: bool = True,
    min_gap: int = 5,
) -> CUSUMResult:
    """双侧 CUSUM 变点检测。

    k / h 以 **标准差为单位**：松弛量 = k*sigma，决策阈值 = h*sigma
    （默认 k=0.5, h=5.0，与设计文档一致）。
    min_gap: 相邻报警点最小间隔，用于去抖。

    参数：
      series：因子日频 IC 序列（一维；可含 NaN，NaN 视作「无观测」）。
      k：松弛量系数（>=0，单位 sigma）。
      h：决策阈值系数（>=0，单位 sigma）。
      use_robust_scale：True 用 1.4826*MAD，False 用总体 std；退化时一律回退 1.0。
      min_gap：报警去抖最小间隔（>=1）。

    返回：:class:`CUSUMResult`。

    备注：参考水平 mu 取**全样本有效观测均值**（回测里等价于「用整段历史标定基线」，
    因此回看时该检测不引入未来信息以外的额外假设；实盘监控应把 mu/sigma 冻结在
    历史标定窗口上，本模块不擅自决定标定窗口长度）。
    """
    k = _finite_positive(k, "k")
    h = _finite_positive(h, "h")
    if int(min_gap) < 1:
        raise ValueError(f"min_gap 必须 >= 1，收到 {min_gap!r}")
    min_gap = int(min_gap)

    x = _as_1d(series)
    n = int(x.shape[0])
    s_high = np.zeros(n, dtype=np.float64)
    s_low = np.zeros(n, dtype=np.float64)
    if n == 0:
        return CUSUMResult([], s_high, s_low, False, "none")

    mask = np.isfinite(x)
    if not bool(mask.any()):                      # 全缺失：无观测 → 无报警
        return CUSUMResult([], s_high, s_low, False, "none")

    mu = float(np.mean(x[mask]))
    sigma = _robust_scale(x, mask) if use_robust_scale else _plain_scale(x, mask)
    slack = k * sigma
    thr = h * sigma
    if not math.isfinite(thr):                    # 极端参数兜底：不报警
        return CUSUMResult([], s_high, s_low, False, "none")

    # 缺失行填为参考水平 ⇒ 偏差为 0（见模块 docstring 的 NaN 语义）
    dev = np.where(mask, x - mu, 0.0)
    hi = 0.0
    lo = 0.0
    cps: list[int] = []
    dirs: list[str] = []
    last_cp = -(min_gap + 1)
    for t in range(n):
        d = float(dev[t])
        hi = max(0.0, hi + d - slack)
        lo = max(0.0, lo - d - slack)
        s_high[t] = hi
        s_low[t] = lo

        trig_hi = hi > thr
        trig_lo = lo > thr
        if trig_hi or trig_lo:
            # 双侧同时触发时以「偏离更大」的一侧定向；相等则偏向衰退（保守）
            direction = "down" if (trig_lo and (not trig_hi or lo >= hi)) else "up"
            if t - last_cp >= min_gap:            # 去抖：丢弃 min_gap 内的重复报警
                cps.append(t)
                dirs.append(direction)
                last_cp = t
            # 触发后归零重新累积（该侧及另一侧同时归零，避免同一事件重复触发）
            if trig_hi:
                hi = 0.0
                s_high[t] = 0.0
            if trig_lo:
                lo = 0.0
                s_low[t] = 0.0

    alarm_hi = s_high[-1] > thr
    alarm_lo = s_low[-1] > thr
    if alarm_hi or alarm_lo:
        direction = "down" if (alarm_lo and (not alarm_hi or s_low[-1] >= s_high[-1])) else "up"
    elif dirs:
        # 序列末端已复位：报「最后一次报警」的方向，避免丢掉已经发生过的衰退
        direction = dirs[-1]
    else:
        direction = "none"

    return CUSUMResult(cps, s_high, s_low, bool(alarm_hi or alarm_lo), direction)


class KalmanICFilter:
    """一维卡尔曼滤波，用于估计因子的「当前真实 IC 水平」。

    状态空间（随机游走 + 观测噪声）：

        x[t] = x[t-1] + w,   w ~ N(0, process_var)     # 真实 IC 缓慢漂移
        z[t] = x[t]     + v,   v ~ N(0, obs_var)       # 观测 IC 含噪声

    每次 ``update`` 先做时间更新（P ← P + process_var）再做观测更新；
    ``obs`` 为 NaN（或非有限）时**跳过观测更新、只做预测**，
    因此状态被保留而方差单调不减（标准「缺失观测」处理，不会污染状态）。
    递推为纯顺序前向、无随机数，完全确定性。
    """

    def __init__(
        self,
        process_var: float = 1e-4,
        obs_var: float = 1e-2,
        init_state: float | None = None,
        init_var: float = 1.0,
    ) -> None:
        self.process_var = _finite_positive(process_var, "process_var")
        self.obs_var = _finite_positive(obs_var, "obs_var")
        self.init_state = 0.0 if init_state is None else float(init_state)
        if not math.isfinite(self.init_state):
            raise ValueError(f"init_state 必须是有限数，收到 {init_state!r}")
        init_v = _finite_positive(init_var, "init_var", allow_zero=False)
        self._init_var = init_v
        self._state = self.init_state
        self._var = init_v

    # -- 内部 ---------------------------------------------------------------
    def reset(self) -> None:
        """恢复初始状态（便于复用同一实例处理多条序列）。"""
        self._state = self.init_state
        self._var = self._init_var

    # -- 公共 API -----------------------------------------------------------
    def update(self, obs: float) -> float:
        """前进一步：返回后验状态估计；``obs`` 为 NaN 时只做预测。"""
        # 1) 时间更新（预测）
        self._var = self._var + self.process_var
        # 2) 观测更新（仅在观测有限时）
        obs_val = float(obs)
        if math.isfinite(obs_val):
            denom = self._var + self.obs_var
            if denom > 0.0:
                gain = self._var / denom
                self._state = self._state + gain * (obs_val - self._state)
                self._var = (1.0 - gain) * self._var
                if self._var < 0.0:               # 数值兜底（理论不会发生）
                    self._var = 0.0
        return float(self._state)

    def filter(self, series: np.ndarray) -> np.ndarray:
        """整段平滑（顺序前向）：返回与输入等长的后验状态估计序列。

        缺失位置输出 NaN（保持「无观测」语义），其余位置输出当时的后验估计。
        滤波后的序列用于后续降噪比较，缺失点的 RMSE 计算会被自动跳过。
        """
        x = _as_1d(series)
        out = np.empty(x.shape[0], dtype=np.float64)
        for t in range(int(x.shape[0])):
            obs = float(x[t])
            out[t] = self.update(obs) if math.isfinite(obs) else np.nan
        return out

    @property
    def state(self) -> float:
        """当前后验状态估计（真实 IC 水平）。"""
        return float(self._state)

    @property
    def variance(self) -> float:
        """当前后验状态方差（只读；缺失观测会让它单调不减）。"""
        return float(self._var)


def _valid_window(valid_idx: np.ndarray, size: int) -> np.ndarray:
    """从序列末尾向前取 ``size`` 个**有效观测**的索引。

    有效观测不足时返回全部可用索引（保证窗口不因空洞而变空）。
    """
    count = int(min(size, valid_idx.size))
    if count <= 0:
        return valid_idx[:0]
    return valid_idx[-count:]


def decay_probability(
    ic_series: np.ndarray,
    *,
    baseline_window: int = 120,
    recent_window: int = 20,
    use_kalman: bool = True,
) -> float:
    """估计因子「正在衰退」的概率，返回 [0,1]。

    采用的定义（完全确定性、无随机数）：

    1. 取近期窗口 R（默认末尾 20 个**有效观测**）与基线窗口 B（默认较早的
       120 个有效观测；两个窗口不重叠，各自跳过 NaN，只在有效观测上统计）。
       ``use_kalman=True`` 时先用 :class:`KalmanICFilter` 前向平滑再统计，
       以抑制单日噪声造成的误判。记

           delta   = mean(B) - mean(R)            （>0 表示近期水平走低）
           sigma_B = std(B)（为 0 或非有限时取 1.0）
           scale   = max(sigma_B, 0.25)           （下限只用于防止极小波动放大 z）
           n_R     = 近期窗口有效观测数

    2. 两个**在 delta = 0 处取 0** 的分量：

           z        = max(0, delta) * sqrt(n_R) / scale      # 均值差的 t 量级
           evidence = 1 - exp(-0.5 * z)
           effect   = min(1, max(0, delta) / sigma_B)        # 相对基线的效应量，封顶 1
           score    = 0.5 * effect + 0.5 * evidence          ∈ [0, 1)

    3. CUSUM 强证据：在默认参数 (k=0.5, h=5.0) 上跑 :func:`cusum_detect`，
       若最近一次报警方向为 ``"down"``（IC 下滑）则 ``score ← max(score, 0.95)``。

    4. logistic 压缩（斜率 4.0、中点 0.5）：

           P = 1 / (1 + exp(-4 * (score - 0.5)))

    性质：``delta <= 0`` ⇒ ``score = 0`` ⇒ ``P = 0.2``（无衰退证据的基线概率）；
    ``delta`` 越大、越持续（z 越大），P 单调不减；CUSUM 报警时 P >= 0.95。

    阈值 0.3 / 0.6 的含义（可直接用于 §5.3 的降权决策）：
      - ``P < 0.3``：等价于 score < 0.095 ⇒ 近期均值未低于基线（或落差远小于噪声）
        且 CUSUM 未报警 ⇒ 判为**未衰退**。
      - ``P > 0.6``：等价于 score > 0.905 ⇒ 落差既具统计显著性（z 大）又具实际
        幅度（effect 高），或 CUSUM 已报警 ⇒ 判为**正在衰退**。
      - ``0.3 <= P <= 0.6``：证据不足或落差尚小 ⇒ 建议「观察 + 缩短再评估周期」，
        而非直接自动降权（自动降权还需 ``parameter_sensitivity`` 通过）。

    退化情形：全 NaN / 空序列 / 有效观测不足 2 个一律返回 0.0；常数序列
    ``delta = 0`` 走 score=0 分支返回 0.2（均低于 0.3 阈值，即「判为未衰退」），
    全程不抛异常、不产生 NaN。
    """
    baseline_window = int(_finite_positive(baseline_window, "baseline_window", allow_zero=False))
    recent_window = int(_finite_positive(recent_window, "recent_window", allow_zero=False))

    x = _as_1d(ic_series)
    mask = np.isfinite(x)
    if x.size == 0 or int(mask.sum()) < 2:
        return 0.0

    series = x
    if use_kalman:
        series = KalmanICFilter().filter(x)       # 缺失处输出 NaN，统计时自动跳过

    smask = np.isfinite(series)
    if int(smask.sum()) < 2:
        return 0.0
    valid_idx = np.flatnonzero(smask)

    # 近期窗口：末尾 recent_window 个有效观测
    recent_vals = series[valid_idx[-min(recent_window, valid_idx.size):]]
    # 基线窗口：在「近期窗口最早索引」之前，尽量取满 baseline_window 个有效观测
    recent_start = int(valid_idx[-min(recent_window, valid_idx.size)])
    base_idx = valid_idx[valid_idx < recent_start]
    if base_idx.size == 0:                        # 有效观测太少：早段作基线
        order = np.argsort(valid_idx, kind="stable")
        split = max(1, valid_idx.size - min(recent_window, max(1, valid_idx.size // 2)))
        base_idx = valid_idx[order][:split]
    else:
        base_idx = _valid_window(base_idx, baseline_window)

    base_vals = series[base_idx]
    n_base = int(base_vals.size)
    n_recent = int(recent_vals.size)
    if n_base < 2 or n_recent < 1:
        return 0.0

    mean_base = float(np.mean(base_vals))
    mean_recent = float(np.mean(recent_vals))
    delta = mean_base - mean_recent
    sigma_base = float(np.std(base_vals))
    if not math.isfinite(sigma_base) or sigma_base <= 0.0:
        sigma_base = _FALLBACK_SCALE

    # 证据分（delta <= 0 时两项均为 0，保证「无落差 ⇒ 无衰退证据」）
    delta_pos = max(0.0, delta)
    z = delta_pos * math.sqrt(float(n_recent)) / max(sigma_base, _Z_SCALE_FLOOR)
    effect = min(1.0, delta_pos / sigma_base)
    evidence = 1.0 - math.exp(-0.5 * z)
    score = 0.5 * effect + 0.5 * evidence

    # CUSUM 下侧报警：强证据 —— 但**必须要求报警在序列末端仍然成立**。
    #
    # 早期实现用「只要历史上出现过 down 方向变点」就拉满 score，实测在**完全平稳**
    # 的 IC 序列上（均值/方差恒定，CUSUM 从不报警 0/30）仍有 33%~53% 被判为衰退
    # （P>=0.6），会造成大量误降权。原因是一次中途下探后即便已回升，历史变点仍会被
    # 反复采信。这里收紧为「末端仍处 down 报警态」，使判定反映**当前**是否在衰退。
    res = cusum_detect(x)                          # 默认 k=0.5, h=5.0
    if res.alarm and _direction_of(res) == "down":
        score = max(score, 0.95)

    if not math.isfinite(score):
        return 0.0
    prob = 1.0 / (1.0 + math.exp(-_LOGIT_SLOPE * (score - _LOGIT_CENTER)))
    if not math.isfinite(prob):
        return 0.0
    return float(min(1.0, max(0.0, prob)))


def parameter_sensitivity(
    ic_series: np.ndarray,
    *,
    k_grid: tuple[float, ...] = (0.25, 0.5, 1.0),
    h_grid: tuple[float, ...] = (3.0, 4.0, 5.0, 7.0),
    threshold: float = 0.5,
) -> dict:
    """参数敏感性分析（§5.3 强制要求：自动降权前必须做）。

    对 (k,h) 网格逐一跑 :func:`cusum_detect`，返回：

      {
        "grid": [(k, h, alarm, direction), ...],   # alarm 表示该组参数下「检出变点」
        "alarm_rate": float,                       # 检出变点的参数组合占比
        "stable_across_parameters": bool,          # 是否稳健（报警结论在网格上一致）
        "recommend_auto_downgrade": bool,          # 只有 stable 且报警才建议自动降权
      }

    判据：
      - ``alarm`` 取 ``result.alarm``（序列末端仍处报警态）**或** ``changepoints`` 非空
        （末端已复位但过程中确实触发过报警）；
      - ``stable_across_parameters``：要么所有组合都不报警，要么所有组合都报警
        且方向一致；只要出现「部分报警」或方向互相矛盾即为 False；
      - ``recommend_auto_downgrade = stable_across_parameters and alarm_rate >= threshold``。

    退化输入（空 / 全 NaN / 常数序列）不报警、不抛异常，返回确定性的四元组。
    ``threshold`` 是报警比例门槛（默认 0.5），不是 IC 阈值。
    """
    k_list = tuple(_finite_positive(kv, "k_grid 元素") for kv in k_grid)
    h_list = tuple(_finite_positive(hv, "h_grid 元素") for hv in h_grid)
    thr = float(threshold)
    if not math.isfinite(thr):
        raise ValueError(f"threshold 必须是有限数，收到 {threshold!r}")
    if not k_list or not h_list:
        raise ValueError("k_grid / h_grid 不能为空")

    x = _as_1d(ic_series)
    grid: list[tuple[float, float, bool, str]] = []
    for kv in k_list:
        for hv in h_list:
            res = cusum_detect(x, k=kv, h=hv)
            alarmed = bool(res.alarm or len(res.changepoints) > 0)
            direction = _direction_of(res) if alarmed else "none"
            grid.append((float(kv), float(hv), alarmed, direction))

    n = len(grid)
    hits = [g for g in grid if g[2]]
    alarm_rate = len(hits) / n if n else 0.0
    if not hits:
        stable = True                             # 全体一致「不报警」
    elif len(hits) == n:
        dirs = {g[3] for g in hits}
        stable = len(dirs) == 1                   # 全体一致报警且方向一致
    else:
        stable = False
    recommend = bool(stable and alarm_rate >= thr)

    return {
        "grid": grid,
        "alarm_rate": float(alarm_rate),
        "stable_across_parameters": bool(stable),
        "recommend_auto_downgrade": recommend,
    }
