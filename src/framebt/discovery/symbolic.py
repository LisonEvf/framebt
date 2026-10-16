"""Symbolic Discovery：符号回归（遗传规划）自主因子发现（§6.2）。

设计文档 §6.2 要求「机器自主发现因子表达式」而不是人工枚举。本模块用**遗传规划
（Genetic Programming, GP）**在价格面板上演化表达式树，适应度为**横截面 Spearman
IC**：

    fitness = |mean_t IC_spearman(factor[t, :], target[t, :])| - c_parsimony * nodes

数据约定
--------
面板 ``panel`` 是 ``dict[str, np.ndarray]``，每个数组形状 ``(n_times, n_symbols)``、
float64（例如 ``open/high/low/close/volume/vwap/returns``）。``target`` 同形状，语义为
**未来收益**。因子在**整个面板**上求值，适应度是「逐行（横截面）Spearman IC 沿时间
求均值」的绝对值最大化。

实现要点
--------
  1. **确定性**（I3）：唯一随机源是 ``np.random.Generator(PCG64(seed))``，构造后沿调用
     链显式传递；不使用任何全局 RNG。同 seed + 同数据 ⇒ 逐位相同的结果。
  2. **受保护数学**：全部算术在 ``np.errstate(all="ignore")`` 下进行，非有限值（inf/nan）
     统一替换为 NaN；``div``/``inv``/``log`` 走受保护分支。全 NaN 因子的适应度为 ``-inf``。
  3. **简约压力**（parsimony）：适应度扣除 ``parsimony_coefficient * nodes``，抑制表达式膨胀。
  4. **选择**：规模 ``tournament_size`` 的锦标赛 + 精英保留（每代最优个体原样进入下一代）。
  5. **多样性**：子树交叉、子树变异、提升（hoist）变异、点变异；每个遗传算子都强制校验
     ``max_depth``/``max_nodes``，越界后代直接拒绝并回退到亲本。
  6. **表达与解析**：``evaluate_expression`` 用 ``ast`` + **严格白名单**递归求值（不使用
     ``eval``），可回放引擎产出的表达式字符串。

术语
----
  - ``ts_*`` 为沿时间轴（axis=0）的滚动算子，窗口 ``n`` 是语法上的整数字面量：
    ``ts_mean(<expr>, <n>)``、``ts_corr(<expr>, <expr>, <n>)``。
  - ``rank(x)`` 为逐行（横截面）平均秩，NaN 保持 NaN。

窗口字面量取值集合：``{2, 3, 5, 6, 10, 20}``。
"""

from __future__ import annotations

import ast
import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

__all__ = [
    "GPResult",
    "SymbolicFactorDiscovery",
    "evaluate_expression",
    "spearman_ic",
]

# ---------------------------------------------------------------------------
# 常量与函数集
# ---------------------------------------------------------------------------

#: 窗口整数字面量候选集（§6.2）。
WINDOW_CHOICES: tuple[int, ...] = (2, 3, 5, 6, 10, 20)

#: 逐元素二元算子。
_BINARY_FUNCS: tuple[str, ...] = ("add", "sub", "mul", "div", "max", "min")

#: 逐元素一元算子。
_UNARY_FUNCS: tuple[str, ...] = ("sqrt", "log", "abs", "neg", "inv", "sign")

#: 横截面算子（逐行）。
_CROSS_FUNCS: tuple[str, ...] = ("rank",)

#: 一元时序算子（滚动窗口）。
_TS_FUNCS: tuple[str, ...] = (
    "ts_mean", "ts_std", "ts_min", "ts_max", "ts_sum",
    "ts_rank", "ts_delta", "ts_delay",
)

#: 二元时序算子（滚动窗口，两个表达式）。
_TS_BINARY_FUNCS: tuple[str, ...] = ("ts_corr", "ts_cov")

#: 默认函数集（恰好覆盖设计文档 §6.2 要求的全部算子）。
DEFAULT_FUNCTION_SET: tuple[str, ...] = (
    _BINARY_FUNCS + _UNARY_FUNCS + _CROSS_FUNCS + _TS_FUNCS + _TS_BINARY_FUNCS
)

#: 受保护除法/对数的下界（|分母| < 该值 ⇒ NaN）。
_EPS = 1e-12

#: 浮点安全下溢保护（避免 cumsum 抵消出现 inf）。
_MAX_FLOAT = 1e300


# ---------------------------------------------------------------------------
# 受保护数学原语（永不产生 inf，非有限值统一为 NaN）
# ---------------------------------------------------------------------------


def _as_float_array(x: Any) -> np.ndarray:
    """把标量/序列/数组统一成 float64 数组（不复制已有 float64 数组）。"""
    arr = np.asarray(x, dtype=np.float64)
    return arr


def _finite(a: np.ndarray) -> np.ndarray:
    """把 ±inf/NaN 统一替换为 NaN。"""
    with np.errstate(all="ignore"):
        return np.where(np.isfinite(a), a, np.nan)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把两个操作数广播到同一形状（标量 → 全 NaN 无关的 0 维常量）。"""
    a2 = _as_float_array(a)
    b2 = _as_float_array(b)
    if a2.shape == b2.shape:
        return a2, b2
    shape = np.broadcast_shapes(a2.shape, b2.shape)
    return np.broadcast_to(a2, shape), np.broadcast_to(b2, shape)


# ---- 逐元素算子 ----


def _f_div(a: Any, b: Any) -> np.ndarray:
    """受保护除法：|分母| < 1e-12 或分母 NaN ⇒ 该格 NaN。"""
    a2, b2 = _align(a, b)
    with np.errstate(all="ignore"):
        bad = ~(np.abs(b2) >= _EPS)  # 同时覆盖 NaN 与接近 0
        out = np.where(bad, np.nan, a2 / np.where(bad, 1.0, b2))
    return _finite(out)


def _f_inv(a: Any) -> np.ndarray:
    """受保护倒数 1/x。"""
    a2 = _as_float_array(a)
    with np.errstate(all="ignore"):
        bad = ~(np.abs(a2) >= _EPS)
        out = np.where(bad, np.nan, 1.0 / np.where(bad, 1.0, a2))
    return _finite(out)


def _f_log(a: Any) -> np.ndarray:
    """受保护对数：取 |x|；|x| < 1e-12 ⇒ NaN。"""
    a2 = _as_float_array(a)
    with np.errstate(all="ignore"):
        mag = np.abs(a2)
        bad = ~(mag >= _EPS)
        out = np.where(bad, np.nan, np.log(np.where(bad, 1.0, mag)))
    return _finite(out)


def _f_sqrt(a: Any) -> np.ndarray:
    """受保护平方根：取 |x|。"""
    with np.errstate(all="ignore"):
        out = np.sqrt(np.abs(_as_float_array(a)))
    return _finite(out)


def _f_abs(a: Any) -> np.ndarray:
    return _finite(np.abs(_as_float_array(a)))


def _f_neg(a: Any) -> np.ndarray:
    return _finite(-_as_float_array(a))


def _f_sign(a: Any) -> np.ndarray:
    with np.errstate(all="ignore"):
        out = np.sign(_as_float_array(a))
    return _finite(out)


def _f_add(a: Any, b: Any) -> np.ndarray:
    a2, b2 = _align(a, b)
    return _finite(a2 + b2)


def _f_sub(a: Any, b: Any) -> np.ndarray:
    a2, b2 = _align(a, b)
    return _finite(a2 - b2)


def _f_mul(a: Any, b: Any) -> np.ndarray:
    a2, b2 = _align(a, b)
    return _finite(a2 * b2)


def _f_max(a: Any, b: Any) -> np.ndarray:
    a2, b2 = _align(a, b)
    with np.errstate(all="ignore"):
        out = np.fmax(a2, b2)  # NaN 语义：任一为 NaN 则 NaN
        out = np.where(np.isnan(a2) | np.isnan(b2), np.nan, out)
    return _finite(out)


def _f_min(a: Any, b: Any) -> np.ndarray:
    a2, b2 = _align(a, b)
    with np.errstate(all="ignore"):
        out = np.fmin(a2, b2)
        out = np.where(np.isnan(a2) | np.isnan(b2), np.nan, out)
    return _finite(out)


# ---- 横截面算子 ----


def _row_ranks_dense(mat: np.ndarray) -> np.ndarray:
    """无 NaN 时的全向量化逐行平均秩（内部快路径，语义同 ``_row_ranks``）。

    做法：逐行稳定排序（列轴向量化）→ 以「与前一格是否相同」标记并列组首尾 →
    ``平均秩 = (左边界 + 右边界)/2 + 1`` → 散射回原列序。
    """
    n_rows, n_cols = mat.shape
    if n_cols == 1:
        return np.ones((n_rows, 1), dtype=np.float64)
    order = np.argsort(mat, axis=1, kind="stable")
    sorted_vals = np.take_along_axis(mat, order, axis=1)
    # 每格是否为并列组首格 / 组尾格
    is_start = np.empty((n_rows, n_cols), dtype=bool)
    is_start[:, 0] = True
    is_start[:, 1:] = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    is_end = np.empty((n_rows, n_cols), dtype=bool)
    is_end[:, -1] = True
    is_end[:, :-1] = sorted_vals[:, 1:] != sorted_vals[:, :-1]
    pos = np.broadcast_to(np.arange(n_cols, dtype=np.float64)[None, :], (n_rows, n_cols))
    # 组左边界：向前填充最近一个组首位置
    left = np.maximum.accumulate(
        np.where(is_start, pos, 0.0), axis=1, out=np.empty((n_rows, n_cols), dtype=np.float64)
    )
    # 组右边界：反向填充最近一个组尾位置
    right = np.minimum.accumulate(
        np.where(is_end, pos, np.float64(n_cols - 1))[:, ::-1], axis=1,
        out=np.empty((n_rows, n_cols), dtype=np.float64),
    )[:, ::-1]
    ranks_sorted = 0.5 * (left + right) + 1.0
    out = np.empty((n_rows, n_cols), dtype=np.float64)
    np.put_along_axis(out, order, ranks_sorted, axis=1)
    return out


def _row_ranks_masked(mat: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """只对 ``mask`` 为真的格子排序的逐行平均秩；其余格子为 NaN（不占秩）。

    与 ``_row_ranks(np.where(mask, mat, np.nan))`` 的区别：这里 NaN 槽位**不参与**
    秩标度，故每行的秩都是 ``1..k``（k = 该行有效格数），与用 ``mask`` 压缩后再排序
    的结果完全一致；亦与 ``scipy.stats.rankdata(row[mask], method="average")`` 一致。
    若各行有效格数一致（GP 的常态：NaN 呈整行/整列结构），则走全向量化快路径。
    """
    arr = np.asarray(mat, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    n_rows, n_cols = arr.shape
    counts = np.count_nonzero(m, axis=1)
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows == 0 or n_cols == 0:
        return out

    if n_rows > 0 and np.all(counts == counts[0]) and counts[0] == n_cols:
        return _row_ranks(arr)                      # 全有效
    if n_rows > 0 and counts[0] > 0 and np.all(counts == counts[0]):
        # 各行有效格数相同（NaN 结构整齐）：整块向量化，NaN 槽位不占秩
        big = np.where(m, np.clip(arr, -1e308, 1e308), 1e308)
        ranks = _row_ranks(big)
        out = np.where(m, ranks, np.nan)
        return out

    for i in range(n_rows):
        if counts[i] == 0:
            continue
        out[i, m[i]] = _column_ranks(arr[i, m[i]])
    return out


def _row_ranks(mat: np.ndarray) -> np.ndarray:
    """对二维数组**逐行**求平均秩（升序，最小 = 1），NaN 保持 NaN。

    纯 numpy 实现（无 scipy/sklearn 依赖），每行步骤如下：
      1) 该行 NaN 填 ``+inf`` 后按稳定排序，NaN 自然落到末尾；
      2) 由「与前一格是否相同」求出并列组（C 语义游程）的左右边界；
      3) ``平均秩 = (左边界 + 右边界) / 2 + 1``；
      4) 散射回原列序，并把 NaN 槽位改回 NaN。

    语义与 ``scipy.stats.rankdata(method="average")`` 一致。
    实现取向：**正确性优先**。逐行循环，但每行只用常数条 numpy 语句（粒度为「行」
    而非「元素」）；在典型横截面宽度下（数十~数百只标的）这点开销可以忽略。
    """
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"_row_ranks 需要二维输入，实得 {arr.shape}")
    n_rows, n_cols = arr.shape
    out = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    if n_rows == 0 or n_cols == 0:
        return out

    nan_mask = ~np.isfinite(arr)
    # 常用快路径：整块无 NaN ⇒ 全向量化（大多数因子在预热后都是这种情况）
    if not nan_mask.any():
        return _row_ranks_dense(arr)

    idx = np.arange(n_cols, dtype=np.float64)
    for i in range(n_rows):
        row = arr[i]
        nan_mask = ~np.isfinite(row)
        if nan_mask.all():
            continue
        vals = np.where(nan_mask, np.inf, row)
        order = np.argsort(vals, kind="stable")
        sorted_vals = vals[order]
        if n_cols == 1:
            out[i, 0] = 1.0
            continue
        # 并列组首格标记（首格恒为 True）
        new_group = np.empty(n_cols, dtype=bool)
        new_group[0] = True
        np.not_equal(sorted_vals[1:], sorted_vals[:-1], out=new_group[1:])
        group_of = np.cumsum(new_group) - 1              # 每个位置所属组编号（0 基）
        starts = idx[new_group]                          # 各组左边界（0 基）
        ends = np.empty(starts.size, dtype=np.float64)   # 各组右边界（0 基，含）
        ends[:-1] = starts[1:] - 1.0
        ends[-1] = n_cols - 1.0
        ranks_sorted = 0.5 * (starts[group_of] + ends[group_of]) + 1.0   # 平均秩（1 基）
        ranks = np.empty(n_cols, dtype=np.float64)
        ranks[order] = ranks_sorted
        ranks[nan_mask] = np.nan
        out[i] = ranks
    return out


def _column_ranks(col: np.ndarray) -> np.ndarray:
    """对 1 维数据求平均秩（升序，最小 = 1），NaN 保持 NaN。"""
    arr = np.asarray(col, dtype=np.float64)
    return _row_ranks(arr.reshape(1, -1))[0]


def _f_rank(a: Any) -> np.ndarray:
    """横截面（逐行）平均秩；NaN 保持 NaN。"""
    arr = _as_float_array(a)
    if arr.ndim <= 1:
        return _column_ranks(np.atleast_1d(arr))
    flat = arr.reshape(arr.shape[0], -1)
    return _row_ranks(flat).reshape(arr.shape)


# ---- 滚动（时序）原语 ----


def _rolling_count_sum(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
    """一次 cumsum 同时得到滚动有效计数与滚动和（尾部对齐窗口 ``[t-w+1, t]``）。

    返回 ``(count, total)``：``count[t]`` 为窗口内非 NaN 个数，``total[t]`` 为窗口内
    非 NaN 值之和；窗口不足 w 个观测时 ``total`` 为有效部分和（由 ``count`` 配合使用）。
    """
    n = x.shape[0]
    valid = np.isfinite(x)
    csum_c = np.cumsum(valid, axis=0, dtype=np.float64)
    filled = np.where(valid, np.clip(x, -_MAX_FLOAT, _MAX_FLOAT), 0.0)
    csum_s = np.cumsum(filled, axis=0)
    count = np.empty_like(csum_c)
    total = np.empty_like(csum_s)
    count[: w - 1] = csum_c[: w - 1]
    total[: w - 1] = csum_s[: w - 1]
    count[w - 1 :] = csum_c[w - 1 :]
    total[w - 1 :] = csum_s[w - 1 :]
    if n > w:
        count[w:] -= csum_c[:-w]
        total[w:] -= csum_s[:-w]
    return count, total


def _rolling_count(x: np.ndarray, w: int) -> np.ndarray:
    """逐格有效（非 NaN）计数，尾部对齐：out[t] 统计窗口 [t-w+1, t]。"""
    return _rolling_count_sum(x, w)[0]


def _rolling_sum(x: np.ndarray, w: int) -> np.ndarray:
    """逐格滚动求和；尾部不足 w 个观测的位置为有效部分和。"""
    return _rolling_count_sum(x, w)[1]


def _ts_reduce(x: Any, w: int, fn: str) -> np.ndarray:
    """一元滚动算子统一入口（``ts_mean/std/min/max/sum``）。"""
    arr = _as_float_array(x)
    n = int(arr.shape[0]) if arr.ndim else 0
    if arr.ndim == 0:
        return np.asarray(np.nan, dtype=np.float64)
    if n < w:
        # 窗口大于样本数：全部不足窗口 ⇒ NaN
        return np.full(arr.shape, np.nan, dtype=np.float64)
    cnt, total = _rolling_count_sum(arr, w)
    enough = cnt >= w
    with np.errstate(all="ignore"):
        if fn == "ts_sum":
            out = np.where(enough, total, np.nan)
        elif fn == "ts_mean":
            out = np.where(enough, total / w, np.nan)
        elif fn == "ts_std":
            _c2, sq = _rolling_count_sum(arr * arr, w)  # 窗口内平方和
            var = sq / w - (total / w) ** 2
            out = np.where(enough, np.sqrt(np.clip(var, 0.0, None)), np.nan)
        elif fn in ("ts_min", "ts_max"):
            win = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)
            # 全 NaN 窗口会触发 numpy 的 "All-NaN slice" 警告，这里显式压掉
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                red = np.nanmin(win, axis=-1) if fn == "ts_min" else np.nanmax(win, axis=-1)
            out = np.concatenate(
                [np.full((w - 1,) + tuple(arr.shape[1:]), np.nan, dtype=np.float64), red],
                axis=0,
            )
            out = np.where(enough, out, np.nan)
        else:  # pragma: no cover - 分派表已限定取值
            raise ValueError(f"未知窗口算子 {fn}")
    return _finite(out)


def _ts_delta(x: Any, w: int) -> np.ndarray:
    """``x[t] - x[t-w]``（前 w 行为 NaN）。"""
    arr = _as_float_array(x)
    n = arr.shape[0] if arr.ndim else 0
    if arr.ndim == 0 or n == 0:
        return np.full((n,), np.nan, dtype=np.float64) if arr.ndim == 0 else arr * np.nan
    out = np.full_like(arr, np.nan)
    if n > w:
        with np.errstate(all="ignore"):
            out[w:] = arr[w:] - arr[:-w]
    return _finite(out)


def _ts_delay(x: Any, w: int) -> np.ndarray:
    """``x[t-w]``（前 w 行为 NaN）。"""
    arr = _as_float_array(x)
    n = arr.shape[0] if arr.ndim else 0
    if arr.ndim == 0 or n == 0:
        return np.full((n,), np.nan, dtype=np.float64) if arr.ndim == 0 else arr * np.nan
    out = np.full_like(arr, np.nan)
    if n > w:
        out[w:] = arr[:-w]
    return _finite(out)


def _ts_rank_last(x: Any, w: int) -> np.ndarray:
    """滚动窗口内**最后值**的平均秩，归一化到 [0, 1]；不足 w 个有效观测 ⇒ NaN。

    归一化取平均秩 ``(less + (equal+1)/2 - 1) / (cnt - 1)``：
    窗口内唯一最小值 ⇒ 0，唯一最大值 ⇒ 1，全相等（cnt>1）⇒ 0.5。
    """
    arr = _as_float_array(x)
    n = arr.shape[0] if arr.ndim else 0
    if arr.ndim == 0:
        return np.asarray(np.nan, dtype=np.float64)
    if arr.ndim < 2:
        arr = arr.reshape(n, 1)
        squeeze = True
    else:
        squeeze = False
    if n < w:
        return np.full(arr.shape, np.nan, dtype=np.float64)
    win = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)  # (n-w+1, S..., w)
    last = win[..., -1:]  # (n-w+1, S..., 1)
    valid = np.isfinite(win)
    with np.errstate(all="ignore"):
        less = np.count_nonzero(valid & (win < last), axis=-1).astype(np.float64)
        equal = np.count_nonzero(valid & (win == last), axis=-1).astype(np.float64)
        cnt = np.count_nonzero(valid, axis=-1).astype(np.float64)
        denom = cnt - 1.0
        good = denom > 0.0
        r = np.where(good, (less + 0.5 * (equal + 1.0) - 1.0) / np.where(good, denom, 1.0), np.nan)
        out = np.concatenate(
            [np.full((w - 1,) + tuple(arr.shape[1:]), np.nan, dtype=np.float64), r],
            axis=0,
        )
    out = _finite(out)
    return out[:, 0] if squeeze else out


def _ts_corr_cov(x: Any, y: Any, w: int, *, kind: str) -> np.ndarray:
    """滚动 Pearson 相关 / 协方差（要求两个操作数形状一致）。"""
    a = _as_float_array(x)
    b = _as_float_array(y)
    if a.shape != b.shape:
        if a.ndim == 0:
            a = np.broadcast_to(a, b.shape)
        elif b.ndim == 0:
            b = np.broadcast_to(b, a.shape)
        else:
            raise ValueError(f"{kind} 的两个输入形状不一致：{a.shape} vs {b.shape}")
    if a.ndim == 0:
        # 两侧都是标量常量：不存在时序维度 ⇒ 该因子无意义，返回 NaN
        return np.asarray(np.nan, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False

    n = a.shape[0]
    if n < w:
        return np.full(a.shape[:1] if squeeze else a.shape, np.nan, dtype=np.float64)

    pair = np.isfinite(a) & np.isfinite(b)
    af = np.where(pair, np.clip(a, -_MAX_FLOAT, _MAX_FLOAT), 0.0)
    bf = np.where(pair, np.clip(b, -_MAX_FLOAT, _MAX_FLOAT), 0.0)

    cnt, _ = _rolling_count_sum(np.where(pair, 1.0, np.nan), w)
    _, sx = _rolling_count_sum(af, w)
    _, sy = _rolling_count_sum(bf, w)
    _, sxx = _rolling_count_sum(af * af, w)
    _, syy = _rolling_count_sum(bf * bf, w)
    _, sxy = _rolling_count_sum(af * bf, w)

    with np.errstate(all="ignore"):
        cxx = sxx - sx * sx / w
        cyy = syy - sy * sy / w
        cxy = sxy - sx * sy / w
        if kind == "ts_cov":
            out = cxy / (w - 1) if w > 1 else np.full(a.shape, np.nan, dtype=np.float64)
        else:
            denom = np.sqrt(np.clip(cxx, 0.0, None) * np.clip(cyy, 0.0, None))
            out = cxy / np.where(denom > 0.0, denom, np.nan)
        out = np.where(cnt >= w, out, np.nan)
    out = _finite(out)
    return out[:, 0] if squeeze else out


#: 逐元素二元算子分派表。
_BINARY_FUNCS_MAP: dict[str, Callable[[Any, Any], np.ndarray]] = {
    "add": _f_add,
    "sub": _f_sub,
    "mul": _f_mul,
    "div": _f_div,
    "max": _f_max,
    "min": _f_min,
}

#: 逐元素一元算子分派表。
_UNARY_FUNCS_MAP: dict[str, Callable[[Any], np.ndarray]] = {
    "sqrt": _f_sqrt,
    "log": _f_log,
    "abs": _f_abs,
    "neg": _f_neg,
    "inv": _f_inv,
    "sign": _f_sign,
    "rank": _f_rank,
}

#: 一元滚动算子分派表。
_TS_FUNCS_MAP: dict[str, Callable[[Any, int], np.ndarray]] = {
    "ts_mean": lambda x, w: _ts_reduce(x, w, "ts_mean"),
    "ts_std": lambda x, w: _ts_reduce(x, w, "ts_std"),
    "ts_min": lambda x, w: _ts_reduce(x, w, "ts_min"),
    "ts_max": lambda x, w: _ts_reduce(x, w, "ts_max"),
    "ts_sum": lambda x, w: _ts_reduce(x, w, "ts_sum"),
    "ts_rank": _ts_rank_last,
    "ts_delta": _ts_delta,
    "ts_delay": _ts_delay,
}

#: 二元滚动算子分派表。
_TS_BINARY_MAP: dict[str, Callable[[Any, Any, int], np.ndarray]] = {
    "ts_corr": lambda x, y, w: _ts_corr_cov(x, y, w, kind="ts_corr"),
    "ts_cov": lambda x, y, w: _ts_corr_cov(x, y, w, kind="ts_cov"),
}


def _call_op(name: str, args: list[Any]) -> np.ndarray:
    """按名称分派算子（表达式求值的唯一入口，名字必须是白名单成员）。"""
    if name in _BINARY_FUNCS_MAP:
        if len(args) != 2:
            raise ValueError(f"{name} 需要 2 个参数，实得 {len(args)}")
        return _BINARY_FUNCS_MAP[name](args[0], args[1])
    if name in _UNARY_FUNCS_MAP:
        if len(args) != 1:
            raise ValueError(f"{name} 需要 1 个参数，实得 {len(args)}")
        return _UNARY_FUNCS_MAP[name](args[0])
    if name in _TS_FUNCS_MAP:
        if len(args) != 2:
            raise ValueError(f"{name} 需要 2 个参数（表达式, 窗口），实得 {len(args)}")
        return _TS_FUNCS_MAP[name](args[0], _as_window(args[1]))
    if name in _TS_BINARY_MAP:
        if len(args) != 3:
            raise ValueError(f"{name} 需要 3 个参数（表达式, 表达式, 窗口），实得 {len(args)}")
        return _TS_BINARY_MAP[name](args[0], args[1], _as_window(args[2]))
    raise ValueError(f"未知函数 {name!r}")


def _as_window(v: Any) -> int:
    """把窗口参数校验成正整数。"""
    arr = np.asarray(v, dtype=np.float64)
    if arr.size != 1 or not np.isfinite(arr.ravel()[0]):
        raise ValueError(f"窗口参数必须是整数标量，实得 {v!r}")
    w = int(arr.ravel()[0])
    if w < 1 or float(w) != float(arr.ravel()[0]):
        raise ValueError(f"窗口参数必须是正整数，实得 {v!r}")
    return w


# ---------------------------------------------------------------------------
# 表达式树
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Var:
    """变量终端（面板中的一条序列，如 ``close``）。"""

    name: str


@dataclass(frozen=True)
class _Const:
    """常量终端（``const_range`` 内的随机浮点，渲染保留 4 位小数）。"""

    value: float


@dataclass(frozen=True)
class _Unary:
    """一元算子节点。"""

    op: str
    child: "_Node"


@dataclass(frozen=True)
class _Binary:
    """逐元素二元算子节点。"""

    op: str
    left: "_Node"
    right: "_Node"


@dataclass(frozen=True)
class _TsUnary:
    """一元滚动算子节点：``op(expr, window)``。"""

    op: str
    child: "_Node"
    window: int


@dataclass(frozen=True)
class _TsBinary:
    """二元滚动算子节点：``op(expr, expr, window)``。"""

    op: str
    left: "_Node"
    right: "_Node"
    window: int


_Node = _Var | _Const | _Unary | _Binary | _TsUnary | _TsBinary

_VARIABLE_NODE = _Var
_CONST_NODE = _Const
_UNARY_NODE = _Unary
_BINARY_NODE = _Binary
_TS_UNARY_NODE = _TsUnary
_TS_BINARY_NODE = _TsBinary


def _render_const(value: float) -> str:
    """常量渲染：4 位小数。"""
    return f"{float(value):.4f}"


def _fmt(value: float) -> str:
    """表达式后缀渲染（去尾零，保证可读性；与解析器兼容）。"""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _node_to_string(node: _Node) -> str:
    """渲染可读表达式。"""
    if isinstance(node, _Var):
        return node.name
    if isinstance(node, _Const):
        return _render_const(node.value)
    if isinstance(node, _Unary):
        return f"{node.op}({_node_to_string(node.child)})"
    if isinstance(node, _Binary):
        return f"{node.op}({_node_to_string(node.left)},{_node_to_string(node.right)})"
    if isinstance(node, _TsUnary):
        return f"{node.op}({_node_to_string(node.child)},{node.window})"
    if isinstance(node, _TsBinary):
        return (
            f"{node.op}({_node_to_string(node.left)},"
            f"{_node_to_string(node.right)},{node.window})"
        )
    raise TypeError(f"未知节点类型 {type(node)!r}")  # pragma: no cover - 防御


def _node_children(node: _Node) -> tuple[_Node, ...]:
    """子节点元组（终端为空元组）。"""
    if isinstance(node, _Binary):
        return (node.left, node.right)
    if isinstance(node, _TsBinary):
        return (node.left, node.right)
    if isinstance(node, (_Unary, _TsUnary)):
        return (node.child,)
    return ()


def _count_nodes(node: _Node) -> int:
    """节点总数（含终端与窗口参数不单独计数）。"""
    return 1 + sum(_count_nodes(c) for c in _node_children(node))


def _tree_depth(node: _Node) -> int:
    """树深：单节点树深为 0。"""
    children = _node_children(node)
    if not children:
        return 0
    return 1 + max(_tree_depth(c) for c in children)


def _evaluate_node(node: _Node, panel: dict[str, np.ndarray]) -> np.ndarray:
    """求值表达式树（受保护数学；返回 float64 数组）。"""
    if isinstance(node, _Var):
        if node.name not in panel:
            raise KeyError(f"面板中不存在变量 {node.name!r}")
        return _as_float_array(panel[node.name])
    if isinstance(node, _Const):
        return np.asarray(float(node.value), dtype=np.float64)
    if isinstance(node, _Unary):
        return _UNARY_FUNCS_MAP[node.op](_evaluate_node(node.child, panel))
    if isinstance(node, _Binary):
        left = _evaluate_node(node.left, panel)
        right = _evaluate_node(node.right, panel)
        return _BINARY_FUNCS_MAP[node.op](left, right)
    if isinstance(node, _TsUnary):
        return _TS_FUNCS_MAP[node.op](_evaluate_node(node.child, panel), int(node.window))
    if isinstance(node, _TsBinary):
        left = _evaluate_node(node.left, panel)
        right = _evaluate_node(node.right, panel)
        return _TS_BINARY_MAP[node.op](left, right, int(node.window))
    raise TypeError(f"未知节点类型 {type(node)!r}")  # pragma: no cover - 防御


def _subtree_paths(node: _Node, prefix: tuple[int, ...] = ()) -> list[tuple[int, ...]]:
    """返回全部子树路径（根为 ``()``）。"""
    out = [prefix]
    for i, child in enumerate(_node_children(node)):
        out.extend(_subtree_paths(child, prefix + (i,)))
    return out


def _node_at(node: _Node, path: tuple[int, ...]) -> _Node:
    """按路径取子树。"""
    for idx in path:
        node = _node_children(node)[idx]
    return node


def _replace_at(node: _Node, path: tuple[int, ...], new: _Node) -> _Node:
    """按路径替换子树，返回新树（结构共享，原树不变）。"""
    if not path:
        return new
    idx, rest = path[0], path[1:]
    children = list(_node_children(node))
    children[idx] = _replace_at(children[idx], rest, new)
    if isinstance(node, _Binary):
        return _Binary(node.op, children[0], children[1])
    if isinstance(node, _TsBinary):
        return _TsBinary(node.op, children[0], children[1], node.window)
    if isinstance(node, _Unary):
        return _Unary(node.op, children[0])
    if isinstance(node, _TsUnary):
        return _TsUnary(node.op, children[0], node.window)
    raise TypeError(f"未知节点类型 {type(node)!r}")  # pragma: no cover - 防御


# ---------------------------------------------------------------------------
# 表达式解析（ast + 严格白名单，不使用 eval）
# ---------------------------------------------------------------------------


class ExpressionError(ValueError):
    """表达式语法/白名单校验失败。"""


_SAFE_BIN_OPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
}
_SAFE_UNARY_OPS: dict[type, Callable[[float], float]] = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}


def _parse_expression(expr: str) -> _Node:
    """把表达式字符串解析成表达式树。

    只接受本模块函数集与数字字面量；``ast`` 节点类型严格白名单，
    禁止属性访问 / 下标 / 比较 / 布尔运算 / 幂运算 / 关键字参数。
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ExpressionError("表达式必须是非空字符串")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:  # pragma: no cover - 由调用方测试覆盖
        raise ExpressionError(f"表达式语法错误：{expr!r}") from exc
    return _parse_node(tree.body, expr)


def _parse_node(node: ast.AST, src: str) -> _Node:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ExpressionError(f"表达式 {src!r} 中只允许数字字面量")
        return _Const(float(node.value))
    if isinstance(node, ast.Name):
        return _Var(node.id)
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_UNARY_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"表达式 {src!r} 中含不允许的一元运算符")
        operand = node.operand
        if isinstance(node.op, ast.USub) and isinstance(operand, ast.Constant) and isinstance(
            operand.value, (int, float)
        ) and not isinstance(operand.value, bool):
            return _Const(-float(operand.value))
        inner = _parse_node(operand, src)
        if isinstance(node.op, ast.USub):
            return _Unary("neg", inner)
        return inner
    if isinstance(node, ast.BinOp):
        op = _SAFE_BIN_OPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"表达式 {src!r} 中含不允许的二元运算符")
        left = _parse_node(node.left, src)
        right = _parse_node(node.right, src)
        name = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}[type(node.op)]
        return _Binary(name, left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError(f"表达式 {src!r} 的函数名必须是白名单标识符")
        name = node.func.id
        if name not in DEFAULT_FUNCTION_SET:
            raise ExpressionError(f"表达式 {src!r} 使用了未支持的函数 {name!r}")
        if node.keywords:
            raise ExpressionError(f"表达式 {src!r} 不支持关键字参数")
        args = [_parse_node(a, src) for a in node.args]
        if name in _TS_BINARY_FUNCS:
            if len(args) != 3:
                raise ExpressionError(f"表达式 {src!r} 中 {name} 需要 3 个参数")
            return _TsBinary(name, args[0], args[1], _as_window(_const_of(args[2], name, src)))
        if name in _TS_FUNCS:
            if len(args) != 2:
                raise ExpressionError(f"表达式 {src!r} 中 {name} 需要 2 个参数（表达式, 窗口）")
            return _TsUnary(name, args[0], _as_window(_const_of(args[1], name, src)))
        if name in _BINARY_FUNCS_MAP:
            if len(args) != 2:
                raise ExpressionError(f"表达式 {src!r} 中 {name} 需要 2 个参数")
            return _Binary(name, args[0], args[1])
        if name in _UNARY_FUNCS_MAP:
            if len(args) != 1:
                raise ExpressionError(f"表达式 {src!r} 中 {name} 需要 1 个参数")
            return _Unary(name, args[0])
        raise ExpressionError(f"表达式 {src!r} 使用了未支持的函数 {name!r}")  # pragma: no cover
    raise ExpressionError(f"表达式 {src!r} 中含不允许的语法：{type(node).__name__}")


def _const_of(node: _Node, name: str, src: str) -> float:
    """取窗口参数（必须是数字字面量）。"""
    if not isinstance(node, _Const):
        raise ExpressionError(f"表达式 {src!r} 中 {name} 的窗口必须是整数字面量")
    return float(node.value)


def evaluate_expression(expr: str, panel: dict[str, np.ndarray]) -> np.ndarray:
    """重新求值一个表达式字符串（返回 ``(n_times, n_symbols)``）。

    与 GP 引擎产出的表达式语法一致；用 ``ast`` + 严格白名单解析，
    **不执行** ``eval``。任何非法语法/未支持函数/未定义变量都会抛
    ``ExpressionError``（或 ``KeyError``）。
    """
    node = _parse_expression(expr)
    with np.errstate(all="ignore"):
        out = _evaluate_node(node, panel)
    return _finite(_as_float_array(out))


# ---------------------------------------------------------------------------
# 横截面 Spearman IC
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _TargetRanks:
    """target 的预计算缓存（同一 target 在整轮进化中被反复使用）。

    ``ranks`` 为逐行平均秩：**只对 target 自身有效的格子**排序，因此秩标度与因子侧
    完全一致（NaN 槽位不占用秩）。``mask`` = target 有效掩码。
    """

    ranks: np.ndarray
    valid_rows: np.ndarray
    mask: np.ndarray


def _target_ranks(target: np.ndarray) -> _TargetRanks:
    """预计算 target 的逐行平均秩缓存（NaN 不占秩）。"""
    mask = np.isfinite(target)
    valid_rows = np.flatnonzero(np.count_nonzero(mask, axis=1) >= 3)
    ranks = _row_ranks_masked(target, mask)
    return _TargetRanks(ranks=ranks, valid_rows=valid_rows, mask=mask)


def spearman_ic(
    factor: np.ndarray,
    target: np.ndarray,
    *,
    _target_cache: _TargetRanks | None = None,
) -> float:
    """按行计算截面 Spearman IC 后取均值；忽略 NaN，样本不足返回 nan。

    - 逐行取 ``factor`` 与 ``target`` 同时有效（非 NaN）的格子；
    - 有效样本 < 3 或任一侧方差为 0（全相等）时该行不参与；
    - 无任何有效行时返回 ``nan``。

    实现上先做**平均秩**再做 Pearson（Spearman 的定义），数值上比直接对原始量做
    Pearson 稳定得多（原始量动态范围极大时后者会出现抵消误差）。

    ``_target_cache`` 为内部优化入口（GP 引擎自带 target 秩缓存，避免每个个体重复排序），
    属私有参数，不影响本函数的公开语义。
    """
    f = np.asarray(factor, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    if f.shape != t.shape:
        raise ValueError(f"factor/target 形状不一致：{f.shape} vs {t.shape}")
    if f.size == 0:
        return float("nan")
    if f.ndim == 0:
        return float("nan")  # 标量因子没有横截面，秩相关无定义

    f2 = f.reshape(f.shape[0], -1) if f.ndim > 1 else f.reshape(1, -1)
    t2 = t.reshape(t.shape[0], -1) if t.ndim > 1 else t.reshape(1, -1)

    cache = _target_cache
    if cache is None or cache.ranks.shape != t2.shape:
        cache = _target_ranks(t2)

    f_ok = np.isfinite(f2)
    mask = cache.mask & f_ok
    k = np.count_nonzero(mask, axis=1)
    valid_rows = np.flatnonzero(k >= 3)
    if valid_rows.size == 0:
        return float("nan")

    # 两端都在**交集行**上取平均秩，且秩标度一致（每行 1..k，NaN 槽位不占秩）：
    #   - 因子侧：对该行有效格排序；
    #   - target 侧：取「交集位置」上的 target 秩（交集内的相对次序即 target 秩的次序）。
    rows = mask[valid_rows]
    a = _row_ranks_masked(f2[valid_rows], rows)
    b = _row_ranks_masked(cache.ranks[valid_rows], rows)

    # 无效格在两端都置 0；随后所有求和都显式乘 ``rows``：置 0 已经使无效格对求和
    # 与内积无贡献，最后再乘掩码是为了让「无效格」在数值上绝无残余影响。
    a = np.where(rows, a, 0.0)
    b = np.where(rows, b, 0.0)
    w = rows.astype(np.float64)

    kk = np.count_nonzero(rows, axis=1).astype(np.float64)
    with np.errstate(all="ignore"):
        da = (a - np.sum(a, axis=1, keepdims=True) / kk[:, None]) * w
        db = (b - np.sum(b, axis=1, keepdims=True) / kk[:, None]) * w
        na = np.sqrt(np.sum(da * da, axis=1))
        nb = np.sqrt(np.sum(db * db, axis=1))
        denom = na * nb
        good = np.isfinite(denom) & (denom > 0.0)
        if not np.any(good):
            return float("nan")
        ics = np.sum(da[good] * db[good], axis=1) / denom[good]
        na = np.sqrt(np.sum(da * da, axis=1))
        nb = np.sqrt(np.sum(db * db, axis=1))
        denom = na * nb
        good = np.isfinite(denom) & (denom > 0.0)
        if not np.any(good):
            return float("nan")
        ics = np.sum(da[good] * db[good], axis=1) / denom[good]
    if ics.size == 0:  # pragma: no cover - good 非空时不会发生
        return float("nan")
    return float(np.mean(ics))


# ---------------------------------------------------------------------------
# 结果与引擎
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GPResult:
    """一个进化出来的因子表达式及其训练集表现（§6.2）。"""

    expression: str      # 可读表达式，如 "ts_rank(div(sub(close,open),open),5)"
    ic: float            # 训练集上的截面 IC 均值（带符号）
    abs_ic: float
    depth: int
    nodes: int


class SymbolicFactorDiscovery:
    """符号回归因子发现引擎（§6.2 遗传规划）。

    参数
    ----
      population_size：种群规模。
      generations：进化代数。
      max_depth：表达式树最大深度（根深度 0）。
      max_nodes：表达式树最大节点数。
      parsimony_coefficient：适应度惩罚系数，``fitness = |IC| - c * nodes``。
      p_crossover / p_subtree_mutation / p_hoist_mutation / p_point_mutation：
        各遗传算子概率；和 ≤ 1（不足 1 的部分为「复制」），超出 1 时自动归一化。
      tournament_size：锦标赛选择规模。
      seed：唯一随机种子（PCG64），保证逐位可复现。
      function_set：允许的算子集合（None ⇒ 设计文档 §6.2 全量算子集）。
      const_range：常量终端取值范围。

    设计说明
    --------
      - 初始种群 = 随机树 + 一份**原子基因种子池**（对每个变量构造
        ``ts_mean/ts_std/ts_rank/ts_delta(x, w)`` 等单算子表达式）。这只提供搜索起点，
        不编码任何答案；它让「单算子 + 单窗口」这一最常见 alpha 形态必然进入第 0 代，
        避免 GP 把预算浪费在随机树上（标准 GP 的 ramped half-and-half 做法变体）。
      - 精英保留：每代最优个体原样进入下一代（不施加任何算子）。
      - 每个遗传算子产出的后代都会校验 ``max_depth``/``max_nodes``，越界即回退亲本。
    """

    def __init__(
        self,
        population_size: int = 500,
        generations: int = 10,
        max_depth: int = 6,
        max_nodes: int = 20,
        parsimony_coefficient: float = 0.001,
        p_crossover: float = 0.7,
        p_subtree_mutation: float = 0.1,
        p_hoist_mutation: float = 0.05,
        p_point_mutation: float = 0.1,
        tournament_size: int = 20,
        seed: int = 0,
        function_set: list[str] | None = None,
        const_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        if population_size < 2:
            raise ValueError("population_size 至少为 2")
        if generations < 1:
            raise ValueError("generations 至少为 1")
        if max_depth < 1:
            raise ValueError("max_depth 至少为 1")
        if max_nodes < 1:
            raise ValueError("max_nodes 至少为 1")
        if tournament_size < 1:
            raise ValueError("tournament_size 至少为 1")
        lo, hi = float(const_range[0]), float(const_range[1])
        if lo > hi:
            raise ValueError("const_range 必须满足 lower <= upper")

        probs = {
            "crossover": max(0.0, float(p_crossover)),
            "subtree": max(0.0, float(p_subtree_mutation)),
            "hoist": max(0.0, float(p_hoist_mutation)),
            "point": max(0.0, float(p_point_mutation)),
        }
        total = sum(probs.values())
        if total > 1.0:  # 归一化到 <= 1（剩余概率为复制）
            probs = {k: v / total for k, v in probs.items()}

        funcs = list(DEFAULT_FUNCTION_SET) if function_set is None else list(function_set)
        for name in funcs:
            if name not in DEFAULT_FUNCTION_SET:
                raise ValueError(f"function_set 含未支持的算子 {name!r}")
        if not funcs:
            raise ValueError("function_set 不能为空")
        allowed = set(funcs)

        self.population_size = int(population_size)
        self.generations = int(generations)
        self.max_depth = int(max_depth)
        self.max_nodes = int(max_nodes)
        self.parsimony_coefficient = float(parsimony_coefficient)
        self.p_crossover = probs["crossover"]
        self.p_subtree_mutation = probs["subtree"]
        self.p_hoist_mutation = probs["hoist"]
        self.p_point_mutation = probs["point"]
        self.p_reproduction = max(0.0, 1.0 - sum(probs.values()))
        self.tournament_size = int(tournament_size)
        self.seed = int(seed)
        self.function_set = tuple(funcs)
        self.const_range = (lo, hi)

        self._allowed_binary = [f for f in _BINARY_FUNCS if f in allowed]
        self._allowed_unary = [f for f in _UNARY_FUNCS if f in allowed]
        self._allowed_cross = [f for f in _CROSS_FUNCS if f in allowed]
        self._allowed_ts = [f for f in _TS_FUNCS if f in allowed]
        self._allowed_ts_binary = [f for f in _TS_BINARY_FUNCS if f in allowed]

        # 生成器权重表（固定顺序，保证同 seed 下随机流逐位一致）
        weights: list[tuple[str, float]] = []
        for name, weight in (
            ("add", 1.0), ("sub", 1.0), ("mul", 1.0), ("div", 1.0),
            ("max", 0.5), ("min", 0.5),
        ):
            if name in self._allowed_binary:
                weights.append((name, weight))
        for name, weight in (
            ("sqrt", 1.0), ("log", 1.0), ("abs", 1.0),
            ("neg", 1.0), ("inv", 1.0), ("sign", 0.5),
        ):
            if name in self._allowed_unary:
                weights.append((name, weight))
        for name in self._allowed_cross:
            weights.append((name, 1.0))
        for name in self._allowed_ts:
            weights.append((name, 1.0 + (0.5 if name.startswith("ts_delta") else 0.0)))
        for name in self._allowed_ts_binary:
            weights.append((name, 0.5))
        self._op_names = tuple(w[0] for w in weights)
        self._op_p = np.asarray([w[1] for w in weights], dtype=np.float64)
        self._op_p = self._op_p / self._op_p.sum()

        # 状态（fit 后填充）
        self._best: GPResult | None = None
        self._population: list[_Node] = []
        self._fitness: np.ndarray = np.empty(0, dtype=np.float64)
        self._fitness_ref: np.ndarray = np.empty(0, dtype=np.float64)
        self._generation0_fitness: np.ndarray = np.empty(0, dtype=np.float64)
        self._panel: dict[str, np.ndarray] = {}
        self._var_names: tuple[str, ...] = ()

    # ---- 公共属性 ----

    @property
    def best(self) -> GPResult | None:
        """最近一次 ``fit`` 的最优结果（未训练过为 None）。"""
        return self._best

    @property
    def generation0_fitness(self) -> np.ndarray:
        """第 0 代（随机初始种群）的适应度副本，供诊断/基线对比。

        适应度 = ``|IC| - parsimony_coefficient * nodes``；无有效 IC 的个体为 ``-inf``。
        """
        if self._generation0_fitness.size == 0:
            raise RuntimeError("尚未调用 fit()")
        return self._generation0_fitness.copy()

    # ---- 主流程 ----

    def fit(
        self,
        panel: dict[str, np.ndarray],
        target: np.ndarray,
        top_k: int = 10,
    ) -> list[GPResult]:
        """在面板上进化表达式树，返回按 ``|IC|`` 降序去重的前 ``top_k`` 个结果。"""
        if not isinstance(panel, dict) or not panel:
            raise ValueError("panel 必须是非空 dict[str, np.ndarray]")
        var_names = tuple(panel.keys())
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in panel.items()}
        shape = arrays[var_names[0]].shape
        if len(shape) != 2:
            raise ValueError(f"panel 数组必须是二维 (n_times, n_symbols)，实得 {shape}")
        for name, arr in arrays.items():
            if arr.shape != shape:
                raise ValueError(f"panel[{name!r}] 形状 {arr.shape} 与 {shape} 不一致")

        tgt = np.asarray(target, dtype=np.float64)
        if tgt.shape != shape:
            raise ValueError(f"target 形状 {tgt.shape} 与 panel {shape} 不一致")
        if top_k < 1:
            raise ValueError("top_k 至少为 1")

        rng = np.random.Generator(np.random.PCG64(self.seed))
        self._panel = arrays
        self._var_names = var_names
        target_cache = _target_ranks(tgt.reshape(tgt.shape[0], -1))  # target 固定 ⇒ 只排一次

        population = self._init_population(var_names, rng)
        fitness = self._score_population(population, arrays, tgt, target_cache)
        self._generation0_fitness = fitness.copy()

        for _gen in range(self.generations):
            elite = self._select_elite(population, fitness)
            new_pop: list[_Node] = [elite]
            self._fitness_ref = fitness  # 当前代适应度（锦标赛选择用）
            while len(new_pop) < self.population_size:
                child = self._breed(population, fitness, rng, var_names)
                if not self._valid(child):
                    child = elite  # 兜底：极端参数下保证种群可填满
                new_pop.append(child)
            population = new_pop
            fitness = self._score_population(population, arrays, tgt, target_cache)

        self._population = population
        self._fitness = fitness

        # 去重 + 过滤 NaN + 按 |IC| 降序
        seen: dict[str, GPResult] = {}
        for tree, _fit_value in zip(population, fitness):
            ic = self._ic_of(tree, arrays, tgt, target_cache)
            if not math.isfinite(ic):
                continue
            expr = _node_to_string(tree)
            res = GPResult(
                expression=expr,
                ic=float(ic),
                abs_ic=float(abs(ic)),
                depth=int(_tree_depth(tree)),
                nodes=int(_count_nodes(tree)),
            )
            prev = seen.get(expr)
            if prev is None or res.abs_ic > prev.abs_ic:
                seen[expr] = res
        ranked = sorted(seen.values(), key=lambda r: r.abs_ic, reverse=True)
        self._best = ranked[0] if ranked else None
        return ranked[: int(top_k)]

    # ---- 适应度 ----

    def _evaluate_tree(
        self,
        tree: _Node,
        panel: dict[str, np.ndarray],
        target: np.ndarray,
        target_cache: _TargetRanks | None = None,
    ) -> tuple[float, float, float]:
        """返回 ``(ic, abs_ic, fitness)``；无有效 IC 时 fitness 为 ``-inf``。"""
        try:
            with np.errstate(all="ignore"):
                factor = _evaluate_node(tree, panel)
        except (ValueError, KeyError, FloatingPointError, ZeroDivisionError, IndexError):
            return float("nan"), float("nan"), float("-inf")
        factor = _as_float_array(factor)
        if factor.ndim == 0:
            # 只含常量的退化表达式：常量截面 ⇒ 秩相关无定义，直接判为无效
            return float("nan"), float("nan"), float("-inf")
        if factor.shape != target.shape:
            try:
                factor = np.broadcast_to(factor, target.shape)
            except ValueError:
                return float("nan"), float("nan"), float("-inf")
        if not np.any(np.isfinite(factor)):
            return float("nan"), float("nan"), float("-inf")
        ic = spearman_ic(factor, target, _target_cache=target_cache)
        if not math.isfinite(ic):
            return float("nan"), float("nan"), float("-inf")
        fit_value = abs(ic) - self.parsimony_coefficient * _count_nodes(tree)
        return float(ic), float(abs(ic)), float(fit_value)

    def _ic_of(
        self,
        tree: _Node,
        panel: dict[str, np.ndarray],
        target: np.ndarray,
        target_cache: _TargetRanks | None = None,
    ) -> float:
        return self._evaluate_tree(tree, panel, target, target_cache)[0]

    def _score_population(
        self,
        population: Sequence[_Node],
        panel: dict[str, np.ndarray],
        target: np.ndarray,
        target_cache: _TargetRanks | None = None,
    ) -> np.ndarray:
        out = np.empty(len(population), dtype=np.float64)
        for i, tree in enumerate(population):
            out[i] = self._evaluate_tree(tree, panel, target, target_cache)[2]
        return out

    # ---- 树构造 ----

    def _random_const(self, rng: np.random.Generator) -> _Const:
        lo, hi = self.const_range
        val = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        return _Const(val)

    def _random_terminal(self, var_names: Sequence[str], rng: np.random.Generator) -> _Node:
        if var_names and rng.random() < 0.7:
            return _Var(str(var_names[int(rng.integers(len(var_names)))]))
        return self._random_const(rng)

    def _random_window(self, rng: np.random.Generator) -> int:
        return int(WINDOW_CHOICES[int(rng.integers(len(WINDOW_CHOICES)))])

    def _random_op(self, rng: np.random.Generator) -> str | None:
        if self._op_names:
            idx = int(rng.choice(len(self._op_names), p=self._op_p))
            return self._op_names[idx]
        return None

    def _gen_random(
        self,
        var_names: Sequence[str],
        rng: np.random.Generator,
        depth: int = 0,
        max_depth: int | None = None,
        budget: int = 0,
    ) -> _Node:
        """随机生成表达式树（受深度与节点预算约束）。"""
        md = self.max_depth if max_depth is None else max_depth
        node_budget = self.max_nodes if budget <= 0 else budget
        # 深度到顶 / 预算耗尽 / 无算子可用 ⇒ 终端
        if depth >= md or node_budget <= 1 or not self._op_names:
            return self._random_terminal(var_names, rng)
        name = self._random_op(rng)
        if name is None:  # pragma: no cover - _op_names 非空时不会发生
            return self._random_terminal(var_names, rng)

        if name in self._allowed_ts_binary:
            remaining = max(1, node_budget - 1 - 1)  # 去掉本节点与窗口
            left_budget = int(rng.integers(1, remaining + 1)) if remaining > 1 else 1
            right_budget = remaining - left_budget + 1
            left = self._gen_random(var_names, rng, depth + 1, md, left_budget)
            right = self._gen_random(var_names, rng, depth + 1, md, max(1, right_budget))
            return _TsBinary(name, left, right, self._random_window(rng))
        if name in self._allowed_ts:
            remaining = max(1, node_budget - 1)
            child = self._gen_random(var_names, rng, depth + 1, md, remaining)
            return _TsUnary(name, child, self._random_window(rng))
        if name in self._allowed_binary:
            remaining = max(1, node_budget - 1)
            left_budget = int(rng.integers(1, remaining + 1)) if remaining > 1 else 1
            right_budget = remaining - left_budget + 1
            left = self._gen_random(var_names, rng, depth + 1, md, left_budget)
            right = self._gen_random(var_names, rng, depth + 1, md, max(1, right_budget))
            return _Binary(name, left, right)
        # 一元/横截面算子
        child = self._gen_random(var_names, rng, depth + 1, md, max(1, node_budget - 1))
        return _Unary(name, child)

    def _make_tree(self, var_names: Sequence[str], rng: np.random.Generator) -> _Node:
        """生成一棵满足约束的树（带重试与最小树兜底）。"""
        for _ in range(24):
            tree = self._gen_random(var_names, rng)
            if self._valid(tree):
                return tree
        # 兜底：单变量终端必然合法
        return _Var(str(var_names[0])) if var_names else _Const(0.0)

    def _init_population(
        self, var_names: Sequence[str], rng: np.random.Generator
    ) -> list[_Node]:
        """第 0 代：原子基因种子池 + 随机树填充（按表达式去重）。"""
        population: list[_Node] = []
        seen: set[str] = set()
        for tree in self._gene_seeds(var_names):
            if len(population) >= self.population_size:
                break
            expr = _node_to_string(tree)
            if self._valid(tree) and expr not in seen:
                seen.add(expr)
                population.append(tree)
        guard = 0
        while len(population) < self.population_size and guard < 20 * self.population_size:
            guard += 1
            tree = self._make_tree(var_names, rng)
            expr = _node_to_string(tree)
            if expr not in seen:
                seen.add(expr)
                population.append(tree)
        while len(population) < self.population_size:  # pragma: no cover - 防御
            population.append(self._make_tree(var_names, rng))
        return population

    def _gene_seeds(self, var_names: Sequence[str]) -> list[_Node]:
        """原子基因种子池：单算子 × 单变量 × 单窗口（无答案信息，仅作搜索起点）。"""
        seeds: list[_Node] = []
        vs = [str(v) for v in var_names]
        for v in vs:
            seeds.append(_Var(v))
        for v in vs:
            node = _Var(v)
            for fn in self._allowed_ts:
                for w in WINDOW_CHOICES:
                    seeds.append(_TsUnary(fn, node, int(w)))
        if self._allowed_cross:
            for v in vs:
                seeds.append(_Unary(self._allowed_cross[0], _Var(v)))
        if self._allowed_ts_binary:
            pair = vs[:6]
            for i, a in enumerate(pair):
                for b in pair[i + 1 :]:
                    seeds.append(
                        _TsBinary(self._allowed_ts_binary[0], _Var(a), _Var(b), int(WINDOW_CHOICES[0]))
                    )
        return seeds

    def _valid(self, tree: _Node) -> bool:
        """约束校验：``max_depth`` 与 ``max_nodes``。"""
        try:
            return _tree_depth(tree) <= self.max_depth and _count_nodes(tree) <= self.max_nodes
        except (TypeError, RecursionError):  # pragma: no cover - 防御
            return False

    # ---- 选择与遗传算子 ----

    def _select_elite(self, population: Sequence[_Node], fitness: np.ndarray) -> _Node:
        """精英保留：最优个体（并列取首个出现者，保证确定性）。"""
        best_i = 0
        best_v = fitness[0] if len(fitness) else float("-inf")
        for i in range(1, len(fitness)):
            if fitness[i] > best_v:
                best_v = float(fitness[i])
                best_i = i
        return population[best_i]

    def _tournament(
        self, population: Sequence[_Node], fitness: np.ndarray, rng: np.random.Generator
    ) -> _Node:
        """锦标赛选择（规模 ``tournament_size``，无放回抽样）。"""
        n = len(population)
        k = min(self.tournament_size, n)
        idx = rng.choice(n, size=k, replace=False)
        best_i = int(idx[0])
        best_v = float(fitness[best_i])
        for j in idx[1:]:
            v = float(fitness[int(j)])
            if v > best_v:
                best_v = v
                best_i = int(j)
        return population[best_i]

    def _breed(
        self,
        population: Sequence[_Node],
        fitness: np.ndarray,
        rng: np.random.Generator,
        var_names: Sequence[str],
    ) -> _Node:
        """按概率抽取遗传算子产生一个后代。"""
        u = float(rng.random())
        if u < self.p_crossover:
            parent_a = self._tournament(population, fitness, rng)
            parent_b = self._tournament(population, fitness, rng)
            child = self._crossover(parent_a, parent_b, var_names, rng)
        elif u < self.p_crossover + self.p_subtree_mutation:
            parent = self._tournament(population, fitness, rng)
            child = self._subtree_mutation(parent, var_names, rng)
        elif u < self.p_crossover + self.p_subtree_mutation + self.p_hoist_mutation:
            parent = self._tournament(population, fitness, rng)
            child = self._hoist_mutation(parent, rng)
        elif u < (
            self.p_crossover
            + self.p_subtree_mutation
            + self.p_hoist_mutation
            + self.p_point_mutation
        ):
            parent = self._tournament(population, fitness, rng)
            child = self._point_mutation(parent, rng)
        else:
            child = self._tournament(population, fitness, rng)
        return child

    def _crossover(
        self,
        parent_a: _Node,
        parent_b: _Node,
        var_names: Sequence[str],
        rng: np.random.Generator,
    ) -> _Node:
        """子树交叉：把 A 的随机子树嫁接到 B 的随机位置。"""
        path_a = _subtree_paths(parent_a)
        path_b = _subtree_paths(parent_b)
        if len(path_a) < 2 or not path_b:
            return self._make_tree(var_names, rng)
        cut_a = path_a[int(rng.integers(len(path_a)))]
        cut_b = path_b[int(rng.integers(len(path_b)))]
        donor = _node_at(parent_a, cut_a)
        child = _replace_at(parent_b, cut_b, donor)
        if self._valid(child):
            return child
        return self._make_tree(var_names, rng)

    def _subtree_mutation(
        self, parent: _Node, var_names: Sequence[str], rng: np.random.Generator
    ) -> _Node:
        """子树变异：随机位置替换为随机生成的新子树。"""
        paths = _subtree_paths(parent)
        cut = paths[int(rng.integers(len(paths)))]
        for _ in range(8):
            new_sub = self._gen_random(var_names, rng)
            child = _replace_at(parent, cut, new_sub)
            if self._valid(child):
                return child
        return parent

    def _hoist_mutation(self, parent: _Node, rng: np.random.Generator) -> _Node:
        """提升变异：随机位置替换为其某个后代子树（缩短表达式）。"""
        paths = _subtree_paths(parent)
        cut = paths[int(rng.integers(len(paths)))]
        subtree = _node_at(parent, cut)
        inner = _subtree_paths(subtree)
        if len(inner) < 2:
            return parent
        pick = inner[int(rng.integers(1, len(inner)))]  # 排除自身，保证真正缩小
        child = _replace_at(parent, cut, _node_at(subtree, pick))
        return child if self._valid(child) else parent

    def _point_mutation(self, parent: _Node, rng: np.random.Generator) -> _Node:
        """点变异：改写某节点的算子（保持元数）或终端取值。"""
        paths = _subtree_paths(parent)
        # 随机起点 + 环形扫描：避免无谓的整树遍历，且随机流与路径顺序无关
        start = int(rng.integers(len(paths)))
        order = paths[start:] + paths[:start]
        parent_expr = _node_to_string(parent)
        for path in order:
            node = _node_at(parent, path)
            new_node = self._mutate_node(node, rng)
            if new_node is None:
                continue
            child = _replace_at(parent, path, new_node)
            if self._valid(child) and _node_to_string(child) != parent_expr:
                return child
        return parent

    def _mutate_node(self, node: _Node, rng: np.random.Generator) -> _Node | None:
        """对单个节点做等价元数的扰动（无候选时返回 None）。"""
        if isinstance(node, _Var):
            if self._var_names and len(self._var_names) > 1:
                others = [v for v in self._var_names if v != node.name]
                if others:
                    return _Var(str(others[int(rng.integers(len(others)))]))
            return None
        if isinstance(node, _Const):
            return self._random_const(rng)
        if isinstance(node, _Binary):
            cands = [f for f in self._allowed_binary if f != node.op]
            if not cands:
                return None
            return _Binary(cands[int(rng.integers(len(cands)))], node.left, node.right)
        if isinstance(node, _Unary):
            cands = [f for f in (self._allowed_unary + self._allowed_cross) if f != node.op]
            if not cands:
                return None
            return _Unary(cands[int(rng.integers(len(cands)))], node.child)
        if isinstance(node, _TsUnary):
            return _TsUnary(node.op, node.child, self._random_window(rng))
        if isinstance(node, _TsBinary):
            return _TsBinary(node.op, node.left, node.right, self._random_window(rng))
        return None  # pragma: no cover - 防御
