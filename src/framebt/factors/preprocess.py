"""因子预处理流水线（设计文档 §2.2）。

对**每个截面**（每一行 = 同一时刻的跨标的观测）按固定顺序执行：

  1. **MAD 去极值**：``median ± n × 1.4826 × MAD`` 截尾（1.4826 使 MAD 与正态
     标准差一致；n 默认 3）
  2. **标准化**：截面 z-score
  3. **中性化**：对暴露（行业哑变量 / log 市值）做截面 OLS，取残差
     —— 本数据集无行业分类，故实际暴露为 **log 市值**（可能再加市场 beta）
  4. **相关性去冗余**：两两截面 Spearman 相关 > 阈值（默认 0.7）的因子簇中，
     保留评分（如 |ICIR|）最高者

所有函数都是**面板进、面板出**（``DataFrame``，index=时间, columns=标的），
因此研究（整段）与回测（逐帧截面）共用同一实现。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ops import corrwith_rows

MAD_SCALE = 1.4826  # MAD → 正态标准差的一致化常数


# ===========================================================================
# 1. MAD 去极值
# ===========================================================================

def mad_winsorize(factor: pd.DataFrame, n: float = 3.0) -> pd.DataFrame:
    """截面 MAD 去极值（逐行）。

    上界 = median + n × 1.4826 × MAD，下界 = median − n × 1.4826 × MAD；
    超出者截断到边界。MAD = median(|x − median(x)|)。
    """
    med = factor.median(axis=1)
    mad = factor.sub(med, axis=0).abs().median(axis=1)
    half = n * MAD_SCALE * mad
    upper = med + half
    lower = med - half
    return factor.clip(lower=lower, upper=upper, axis=0)


# ===========================================================================
# 2. 标准化
# ===========================================================================

def zscore(factor: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score。"""
    mu = factor.mean(axis=1)
    sd = factor.std(axis=1)
    return factor.sub(mu, axis=0).div(sd.where(sd.abs() > 1e-12), axis=0)


# ===========================================================================
# 3. 中性化
# ===========================================================================

def neutralize(factor: pd.DataFrame, exposures: list[pd.DataFrame],
               add_const: bool = True) -> pd.DataFrame:
    """截面 OLS 中性化：``factor ~ exposures``，返回残差面板。

    逐行做最小二乘（对缺失做成对剔除）。暴露通常为 ``[log(市值)]``；
    若有行业哑变量也应一并传入。样本数不足（有效观测 ≤ 暴露数+1）时该行返回 NaN。
    """
    if not exposures:
        return factor.copy()

    y_all = factor.to_numpy(dtype=float)
    x_list = [e.reindex_like(factor).to_numpy(dtype=float) for e in exposures]
    out = np.full_like(y_all, np.nan)

    for i in range(y_all.shape[0]):
        y = y_all[i]
        X = np.column_stack([x[i] for x in x_list])
        if add_const:
            X = np.column_stack([np.ones(X.shape[0]), X])
        mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
        if mask.sum() <= X.shape[1]:
            continue
        Xm, ym = X[mask], y[mask]
        try:
            beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
        except np.linalg.LinAlgError:  # pragma: no cover - 退化设计矩阵
            continue
        out[i, mask] = ym - Xm @ beta
    return pd.DataFrame(out, index=factor.index, columns=factor.columns)


def log_market_cap(panel) -> pd.DataFrame:
    """规模暴露：``log(市值)``（市值缺失时为 NaN）。"""
    cap = panel.market_cap.where(panel.market_cap > 0)
    return np.log(cap)


# ===========================================================================
# 4. 相关性去冗余
# ===========================================================================

def pairwise_ic_correlation(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """因子两两的**截面 Spearman 相关**（按行求相关后取时间均值）。

    只保留两因子在同一时刻都有值的观测，避免缺失结构造成的伪相关。
    """
    names = sorted(factors)
    mat = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            fa, fb = factors[a], factors[b]
            # 逐行 Spearman：先按行转秩再求 Pearson（秩上的 Pearson = Spearman）
            corr = corrwith_rows(fa.rank(axis=1), fb.rank(axis=1))
            c = float(corr.mean()) if corr.notna().any() else np.nan
            mat.loc[a, b] = mat.loc[b, a] = c
    return mat


def dedup_by_correlation(factors: dict[str, pd.DataFrame],
                         scores: dict[str, float],
                         threshold: float = 0.7) -> tuple[list[str], pd.DataFrame, list[tuple[str, str, float]]]:
    """按相关性去冗余，保留每簇中 |score| 最大者。

    参数：
      factors：因子名 → 面板
      scores ：因子名 → 评分（如 ICIR，绝对值越大越好）
      threshold：相关性阈值，超过则视为冗余

    返回 ``(保留的因子名, 相关性矩阵, 被剔除的 (保留者, 被剔除者, 相关性) 列表)``。
    """
    corr = pairwise_ic_correlation(factors)
    order = sorted(factors, key=lambda n: abs(scores.get(n, 0.0)), reverse=True)
    kept: list[str] = []
    dropped: list[tuple[str, str, float]] = []
    for name in order:
        redundant_with = None
        for k in kept:
            c = corr.loc[name, k]
            if np.isfinite(c) and abs(c) > threshold:
                redundant_with = (k, float(c))
                break
        if redundant_with is None:
            kept.append(name)
        else:
            dropped.append((redundant_with[0], name, redundant_with[1]))
    return sorted(kept), corr, dropped


# ===========================================================================
# 流水线封装
# ===========================================================================

def preprocess(factor: pd.DataFrame, *, winsor_n: float = 3.0,
               exposures: list[pd.DataFrame] | None = None,
               do_zscore: bool = True,
               do_neutralize: bool = True) -> pd.DataFrame:
    """完整预处理：去极值 → 标准化 → 中性化 → （再标准化）。

    最后再做一次 z-score：中性化后的残差量纲会变，重新标准化便于跨因子合成。
    """
    out = mad_winsorize(factor, winsor_n)
    if do_zscore:
        out = zscore(out)
    if do_neutralize and exposures:
        out = neutralize(out, exposures)
        if do_zscore:
            out = zscore(out)
    return out
