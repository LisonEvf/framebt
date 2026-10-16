"""因子评估（设计文档 §2.3 / §5.1）：IC / ICIR / t 统计量 / 分组单调性 / 相关性。

术语与因果性说明
----------------
- **预测性 IC**（``cross_section_ic(factor, forward_return)``）：用 t 期因子与
  t→t+1 的未来收益求截面 Spearman 相关。用于**事后评估因子质量**，本身不可用于
  交易决策（含未来信息）。
- **已实现 IC**（在线学习用）：用 ``factor.shift(1)`` 与 t−1→t 的**已实现**收益求
  相关 —— 两段信息在 t 时刻都已知，因此可用于逐期更新权重（设计文档 §3.2 的
  「先预测、后观察标签、再更新」范式）。

两者共用同一个 ``cross_section_ic``，只是入参不同。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ops import corrwith_rows

MIN_NAMES = 5  # 截面 IC 至少需要的标的本数


# ===========================================================================
# 收益与 IC
# ===========================================================================

def forward_returns(close: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """前瞻收益 ``close[t+h]/close[t] - 1``（**含未来信息**，仅用于评估）。"""
    return close.shift(-horizon) / close - 1.0


def realized_returns(close: pd.DataFrame) -> pd.DataFrame:
    """已实现收益 ``close[t]/close[t-1] - 1``（t 时刻已知）。"""
    return close / close.shift(1) - 1.0


def cross_section_ic(factor: pd.DataFrame, target: pd.DataFrame) -> pd.Series:
    """逐期截面 Spearman IC 序列。

    实现：先把 factor / target **按行转秩**，再求逐行 Pearson 相关
    （秩上的 Pearson 即 Spearman）。有效标的数 < ``MIN_NAMES`` 的期置为 NaN；
    零方差截面（全体同值）返回 NaN 而非 inf。
    """
    rf = factor.rank(axis=1)
    rt = target.rank(axis=1)
    return corrwith_rows(rf, rt, min_valid=MIN_NAMES)


# ===========================================================================
# 统计量
# ===========================================================================

def newey_west_tstat(x: pd.Series | np.ndarray, lag: int | None = None) -> float:
    """Newey-West 调整的 t 统计量（检验均值是否为 0，容忍自相关）。

    ``lag`` 缺省用经验公式 ``floor(4*(n/100)^(2/9))``；核权重为 Bartlett。
    """
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 3:
        return float("nan")
    mu = a.mean()
    e = a - mu
    if lag is None:
        lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    gamma0 = float(e @ e) / n
    s = gamma0
    for l in range(1, lag + 1):
        if l >= n:
            break
        w = 1.0 - l / (lag + 1.0)
        s += 2.0 * w * float(e[l:] @ e[:-l]) / n
    if s <= 0:
        return float("nan")
    return float(mu / np.sqrt(s / n))


def ic_summary(ic: pd.Series) -> dict:
    """IC 序列的汇总统计：均值 / 标准差 / ICIR / t / 胜率 / 样本数。"""
    a = ic.dropna()
    if a.empty:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan"),
                "t_stat": float("nan"), "positive_rate": float("nan"), "n": 0}
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else float("nan")
    icir = mean / std if std and std > 0 else float("nan")
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "t_stat": newey_west_tstat(a),
        "positive_rate": float((a > 0).mean()),
        "n": int(len(a)),
    }


# ===========================================================================
# 分组单调性
# ===========================================================================

def group_returns(factor: pd.DataFrame, forward_ret: pd.DataFrame,
                  n_groups: int = 10) -> pd.Series:
    """按因子值分组的**等权前瞻收益**（group 1 = 因子最小，group N = 最大）。

    实现为**向量化**：逐行取百分位秩 → 分箱 → 对每箱求行均值再对时间求均值。
    （早期版本逐行 ``qcut``+``groupby``，在 57 因子 × 1500 期上耗时百秒级。）
    """
    common = factor.notna() & forward_ret.notna()
    n_valid = common.sum(axis=1)
    ok = n_valid >= max(n_groups, MIN_NAMES)
    if not ok.any():
        return pd.Series(dtype=float)

    f = factor.where(common)[ok]
    r = forward_ret.where(common)[ok]
    # 行内百分位秩 → 分箱 1..n_groups
    pct = f.rank(axis=1, pct=True)
    grp = np.ceil(pct.to_numpy() * n_groups)
    grp = np.clip(grp, 1, n_groups)

    rv = r.to_numpy(dtype=float)
    out = {}
    for g in range(1, n_groups + 1):
        mask = grp == g
        if not mask.any():
            out[g] = np.nan
            continue
        vals = np.where(mask, rv, np.nan)
        # 每期该组的等权平均 → 再对时间求平均（忽略全 NaN 期）
        with np.errstate(invalid="ignore"):
            per_time = np.nanmean(vals, axis=1)
        out[g] = float(np.nanmean(per_time)) if np.isfinite(per_time).any() else np.nan
    return pd.Series(out)


def monotonicity_score(gr: pd.Series) -> float:
    """分组单调性打分 ∈ [-1, 1]：组序与收益序的 Spearman 相关。"""
    if len(gr) < 3:
        return float("nan")
    g = gr.index.to_numpy(dtype=float)
    r = gr.to_numpy(dtype=float)
    m = np.isfinite(r)
    if m.sum() < 3:
        return float("nan")
    gr_rank = pd.Series(g[m]).rank()
    r_rank = pd.Series(r[m]).rank()
    return float(gr_rank.corr(r_rank))


def top_bottom_spread(gr: pd.Series) -> float:
    """多空价差：最高组 − 最低组 的平均收益。"""
    if len(gr) < 2:
        return float("nan")
    return float(gr.iloc[-1] - gr.iloc[0])


# ===========================================================================
# 因子池筛选（设计文档 §2.3 阈值表）
# ===========================================================================

DEFAULT_SCREEN = {
    "min_abs_ic": 0.02,      # |IC 均值|
    "min_abs_t": 2.0,        # |t 统计量|
    "min_monotonicity": 0.5,  # 分组单调性
    "min_samples": 200,      # IC 样本数
}


def screen_factor(factor: pd.DataFrame, forward_ret: pd.DataFrame,
                  n_groups: int = 10, thresholds: dict | None = None) -> dict:
    """对单个因子做完整评估 + 达标判定。"""
    th = {**DEFAULT_SCREEN, **(thresholds or {})}
    ic = cross_section_ic(factor, forward_ret)
    summ = ic_summary(ic)
    gr = group_returns(factor, forward_ret, n_groups)
    mono = monotonicity_score(gr)
    spread = top_bottom_spread(gr)
    reason = []
    if not (abs(summ["ic_mean"]) > th["min_abs_ic"]):
        reason.append(f"|IC|={abs(summ['ic_mean']):.4f} ≤ {th['min_abs_ic']}")
    if not (abs(summ["t_stat"]) > th["min_abs_t"]):
        reason.append(f"|t|={abs(summ['t_stat']):.2f} ≤ {th['min_abs_t']}")
    if np.isfinite(mono) and abs(mono) < th["min_monotonicity"]:
        reason.append(f"单调性={mono:.2f} < {th['min_monotonicity']}")
    if summ["n"] < th["min_samples"]:
        reason.append(f"样本={summ['n']} < {th['min_samples']}")
    return {
        "ic": ic,
        "ic_mean": summ["ic_mean"],
        "icir": summ["icir"],
        "t_stat": summ["t_stat"],
        "positive_rate": summ["positive_rate"],
        "n": summ["n"],
        "groups": gr,
        "monotonicity": mono,
        "spread": spread,
        "passed": not reason,
        "reject_reason": "; ".join(reason),
    }


def screen_pool(factors: dict[str, pd.DataFrame], forward_ret: pd.DataFrame,
                n_groups: int = 10, thresholds: dict | None = None) -> pd.DataFrame:
    """批量评估因子池，返回按 |ICIR| 降序的评估表。"""
    rows = []
    for name, fac in factors.items():
        r = screen_factor(fac, forward_ret, n_groups, thresholds)
        rows.append({
            "factor": name, "ic_mean": r["ic_mean"], "icir": r["icir"],
            "t_stat": r["t_stat"], "positive_rate": r["positive_rate"],
            "monotonicity": r["monotonicity"], "spread": r["spread"],
            "n": r["n"], "passed": r["passed"], "reject_reason": r["reject_reason"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.reindex(df["icir"].abs().sort_values(ascending=False).index).reset_index(drop=True)
