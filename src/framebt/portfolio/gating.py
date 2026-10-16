"""门控机制：市场状态驱动的因子权重调节（设计文档 §3.3）。

思路
----
基础权重 ``w_base`` 由在线梯度下降学到「长期哪类因子有效」；门控网络根据
**当前市场状态**（波动率分位、涨跌家数比、离散度）输出一组门控系数 ``g``，
最终权重为

    w_final = normalize(w_base ⊙ g)

门控温度 τ：``g = softmax(logits / τ)``。τ 越大分配越均匀，τ 越小越集中。

训练目标（可微、可在线）
------------------------
门控要回答的是「**在这种市场状态下，哪些因子管用**」。因此用**最近的因子 IC
向量**作为监督信号：把近期 IC 经 softmax 变成「理想门控分布」``p``，最小化
交叉熵 ``−Σ p·log g``。梯度有闭式解：

    ∂L/∂logits = (g − p) / τ

再经标准 BP 回传至 MLP 参数。

实证结论（设计文档）：门控加权对 **alpha 因子**稳定性更好，风险因子更适合固定
暴露，故门控应主要用于 alpha 因子 —— 本实现即作用于因子权重。

**确定性**：纯 numpy + 固定种子初始化，无随机训练过程（I3）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def _softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    z = np.asarray(x, dtype=float) / max(tau, EPS)
    z = z - np.nanmax(z)
    e = np.exp(z)
    s = e.sum()
    return e / s if s > EPS else np.full_like(z, 1.0 / z.size)


class MarketStateGating:
    """市场状态门控网络（单隐层 MLP + softmax）。

    参数：
      n_factors  ：因子个数 K
      n_state    ：市场状态变量个数（默认 3）
      hidden     ：隐层宽度
      temperature：门控温度 τ
      lr         ：门控学习率
      seed       ：初始化种子（保证可复现）
    """

    def __init__(self, n_factors: int, n_state: int = 3, hidden: int = 16,
                 temperature: float = 1.0, lr: float = 0.01, seed: int = 0):
        if n_factors <= 0:
            raise ValueError("n_factors 必须为正")
        self.k = int(n_factors)
        self.s = int(n_state)
        self.h = int(hidden)
        self.tau = float(temperature)
        self.lr = float(lr)

        rng = np.random.default_rng(seed)
        # 小初始化（He 风格），保证初始门控接近均匀
        self.W1 = rng.normal(0, np.sqrt(2.0 / max(1, self.s)), (self.h, self.s))
        self.b1 = np.zeros(self.h)
        self.W2 = rng.normal(0, np.sqrt(2.0 / max(1, self.h)), (self.k, self.h))
        self.b2 = np.zeros(self.k)

        self.steps = 0
        self.last_loss = float("nan")

    # ---- 前向 ----

    def gates(self, state: np.ndarray) -> np.ndarray:
        """市场状态 → 门控系数（K 维，和为 1）。"""
        s = np.asarray(state, dtype=float).reshape(-1)
        if s.size != self.s:
            raise ValueError(f"state 维度应为 {self.s}，得到 {s.size}")
        logits, _cache = self._logits(s)
        return _softmax(logits, self.tau)

    def _logits(self, s: np.ndarray):
        a1 = self.W1 @ s + self.b1
        h1 = np.maximum(a1, 0.0)
        logits = self.W2 @ h1 + self.b2
        return logits, (a1, h1)

    def combine(self, base_weights: np.ndarray, state: np.ndarray) -> np.ndarray:
        """基础权重 ⊙ 门控 → 归一化后的最终权重。"""
        g = self.gates(state)
        w = np.asarray(base_weights, dtype=float) * g
        s = np.abs(w).sum()
        return w / s if s > EPS else w

    # ---- 在线训练 ----

    def observe(self, state: np.ndarray, recent_ic: np.ndarray) -> float:
        """用「近期因子 IC 向量」监督训练一步，返回交叉熵损失。

        ``recent_ic``：每个因子的近期 IC（越好 → 该因子在此状态下应获得更高门控）。
        """
        s = np.asarray(state, dtype=float).reshape(-1)
        ic = np.asarray(recent_ic, dtype=float)
        mask = np.isfinite(ic)
        if mask.sum() < 2:
            return float("nan")

        logits, (a1, h1) = self._logits(s)
        g = _softmax(logits, self.tau)
        # 理想门控分布：近期 IC 的 softmax（NaN 因子给极小概率）
        tgt_raw = np.where(mask, ic, -1e9)
        p = _softmax(tgt_raw, self.tau)

        loss = float(-np.sum(np.where(p > 0, p * np.log(g + EPS), 0.0)))
        self.last_loss = loss

        # ∂L/∂logits = (g − p)/τ
        dlogits = (g - p) / max(self.tau, EPS)
        gW2 = np.outer(dlogits, h1)
        gb2 = dlogits
        dh1 = self.W2.T @ dlogits
        da1 = dh1 * (a1 > 0)
        gW1 = np.outer(da1, s)
        gb1 = da1

        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        self.steps += 1
        return loss

    # ---- 市场状态提取 ----

    @staticmethod
    def market_state(panel, vol_window: int = 20,
                     lookback: int = 250) -> pd.DataFrame:
        """从面板提取市场状态变量（逐时刻）。

        设计文档原定三变量为「波动率分位、涨跌家数比、行业集中度」。
        本数据集**无行业分类**，故第三项以**截面收益离散度**（横截面标准差）替代，
        它同样刻画「个股分化 / 抱团」程度。

        返回 ``DataFrame``，index 与面板一致，columns = [vol_pct, breadth, dispersion]。
        """
        ret = panel.close / panel.close.shift(1) - 1.0
        # 1) 市场波动率（截面等权收益的滚动标准差）的历史分位
        mkt = ret.mean(axis=1)
        vol = mkt.rolling(vol_window, min_periods=max(2, vol_window // 2)).std()
        vol_pct = vol.rolling(lookback, min_periods=max(20, lookback // 5)).apply(
            lambda x: float(pd.Series(x).rank(pct=True).iloc[-1]), raw=False)
        # 2) 涨跌家数比
        breadth = (ret > 0).sum(axis=1) / ret.notna().sum(axis=1).replace(0, np.nan)
        # 3) 截面离散度（替代行业集中度）
        dispersion = ret.std(axis=1)
        return pd.DataFrame({
            "vol_pct": vol_pct, "breadth": breadth, "dispersion": dispersion,
        })

    def state_vector(self, state_row: pd.Series) -> np.ndarray:
        """把一行状态转成网络输入（缺失填 0.5 / 0）。"""
        defaults = {"vol_pct": 0.5, "breadth": 0.5, "dispersion": 0.0}
        vals = []
        for name in ("vol_pct", "breadth", "dispersion")[: self.s]:
            v = state_row.get(name, np.nan) if hasattr(state_row, "get") else np.nan
            vals.append(float(v) if np.isfinite(v) else defaults.get(name, 0.0))
        return np.asarray(vals, dtype=float)

    def params(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}

    def set_params(self, p: dict) -> None:
        self.W1, self.b1 = p["W1"].copy(), p["b1"].copy()
        self.W2, self.b2 = p["W2"].copy(), p["b2"].copy()
