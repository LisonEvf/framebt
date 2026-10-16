"""权重优化层：复合损失 + 在线梯度下降（设计文档 §3.1 / §3.2）。

复合损失
--------
    L = −ICIR + λ₁·Turnover + λ₂·‖w‖₁

- **ICIR**：组合打分与已实现收益的截面相关（IC）的滚动信息比。用**秩相关**更稳健，
  但为可微与确定性，这里用「截面已标准化的因子矩阵 × w」与「截面已标准化的收益」
  的 Pearson 相关作为 IC 的可微代理（因子在预处理层已 z-score，故该代理与
  Spearman IC 高度一致）。
- **Turnover**：‖w − w_prev‖₁ —— 抑制权重频繁跳动（换手成本）。
- **‖w‖₁**：因子稀疏化，避免把权重摊到大量弱因子上。

在线更新（§3.2）
----------------
「先预测、后观察标签、再更新」：每一轮先用当前 w 打分，随后市场揭晓已实现收益，
据损失对 w 的梯度反方向更新。为体现「距离当前较远的小批量降权」，维护一个
小批量缓冲区，按折扣 ``β^age`` 加权累加梯度：

    W_{t+1} ← W_t − η · Σ_j β^{age(j)} · ∇_W L(f^(j), y)

**确定性**：纯 numpy，无随机数；给定输入序列与超参，权重轨迹逐位可复现（I3）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


@dataclass
class WeightState:
    """一次权重更新的快照（用于监控与审计）。"""

    step: int
    ic: float
    icir: float
    turnover: float
    l1: float
    loss: float
    weights: np.ndarray


def _ic_and_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """截面 IC 及其对 w 的解析梯度。

    IC = corr(Xc·w, yc)，其中 Xc/yc 为**去均值**后的截面。
    ∇_w IC = Xcᵀ (ŷn − ûn·IC) / ‖Xc·w‖

    其中 ûn = Xc·w/‖Xc·w‖，ŷn = yc/‖yc‖。
    """
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    if mask.sum() < 3:
        return float("nan"), np.zeros_like(w)
    Xm, ym = X[mask], y[mask]
    Xc = Xm - Xm.mean(axis=0, keepdims=True)
    yc = ym - ym.mean()

    u = Xc @ w
    nu = float(np.linalg.norm(u))
    ny = float(np.linalg.norm(yc))
    if nu < EPS or ny < EPS:
        return float("nan"), np.zeros_like(w)

    un = u / nu
    yn = yc / ny
    ic = float(un @ yn)
    grad = Xc.T @ (yn - un * ic) / nu
    return ic, grad


class OnlineWeightOptimizer:
    """在线梯度下降权重优化器（§3.1 / §3.2）。

    参数：
      n_factors      ：因子个数 K
      lambda_turnover：λ₁，换手惩罚
      lambda_l1      ：λ₂，L1 正则
      lr             ：学习率 η（Adam 一阶/二阶动量）
      discount       ：折扣率 β，缓冲区中越旧的小批量权重越小
      buffer         ：保留多少个历史小批量参与梯度累加
      normalize      ：``'l1'`` 每步把 ‖w‖₁ 归一到 1；``'none'`` 不归一
      init           ：初始权重（缺省等权 1/K）
    """

    def __init__(
        self,
        n_factors: int,
        lambda_turnover: float = 0.1,
        lambda_l1: float = 0.01,
        lr: float = 0.001,
        discount: float = 0.95,
        buffer: int = 20,
        beta1: float = 0.9,
        beta2: float = 0.999,
        normalize: str = "l1",
        init: np.ndarray | None = None,
    ):
        if n_factors <= 0:
            raise ValueError("n_factors 必须为正")
        self.n = int(n_factors)
        self.lambda_turnover = lambda_turnover
        self.lambda_l1 = lambda_l1
        self.lr = lr
        self.discount = discount
        self.beta1, self.beta2 = beta1, beta2
        self.normalize = normalize

        self.w = np.full(self.n, 1.0 / self.n) if init is None else np.asarray(init, dtype=float).copy()
        self.w_prev = self.w.copy()
        self._buf: deque = deque(maxlen=buffer)
        self._m = np.zeros(self.n)
        self._v = np.zeros(self.n)
        self._t = 0

        # 折扣后的 IC 均值/方差（用于 ICIR）
        self._ic_mean = 0.0
        self._ic_var = 0.0
        self._ic_w = 0.0
        self.step_count = 0
        self.history: list[WeightState] = []
        self._last_ic = float("nan")

    # ---- 在线更新 ----

    def observe(self, X: np.ndarray, y: np.ndarray) -> WeightState:
        """观察一个新截面：先用当前 w 打分并记录 IC，再更新权重。

        X：(n_stocks, K) 已预处理的因子截面；y：(n_stocks,) 已实现收益。
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n:
            raise ValueError(f"X 形状应为 (n_stocks, {self.n})，得到 {X.shape}")

        # 1) 先预测：用当前权重记录 IC
        ic, grad_now = _ic_and_grad(X, y, self.w)
        self._last_ic = ic
        self._update_ic_stats(ic)
        icir = self._icir()

        # 2) 入缓冲区（供折扣梯度累加）
        self._buf.append((X, y))

        # 3) 再更新：折扣累加历史梯度
        grad = self._discounted_grad(icir)

        # 4) Adam 步（只用 IC 梯度 + 换手次梯度）
        self._t += 1
        self._m = self.beta1 * self._m + (1 - self.beta1) * grad
        self._v = self.beta2 * self._v + (1 - self.beta2) * grad * grad
        mhat = self._m / (1 - self.beta1 ** self._t)
        vhat = self._v / (1 - self.beta2 ** self._t)
        w_new = self.w - self.lr * mhat / (np.sqrt(vhat) + EPS)

        # 5) L1 用**近端算子**（软阈值）而非次梯度项。
        #    原因：Adam 会把梯度按二阶动量归一化，次梯度 λ₂·sign(w) 的幅度被抹平，
        #    L1 实际不产生任何收缩/稀疏效果（实测惩罚系数放大 50 倍也无影响）。
        #    软阈值 w ← sign(w)·max(0,|w|−ηλ₂) 与优化器无关，能真正压缩权重。
        if self.lambda_l1 > 0:
            w_new = np.sign(w_new) * np.maximum(0.0, np.abs(w_new) - self.lr * self.lambda_l1)

        if self.normalize == "l1":
            s = np.abs(w_new).sum()
            if s > EPS:
                w_new = w_new / s

        turnover = float(np.abs(w_new - self.w).sum())
        self.w_prev = self.w.copy()
        self.w = w_new
        self.step_count += 1

        state = WeightState(
            step=self.step_count, ic=ic, icir=icir, turnover=turnover,
            l1=float(np.abs(self.w).sum()), loss=self.loss(icir, turnover),
            weights=self.w.copy(),
        )
        self.history.append(state)
        return state

    def _update_ic_stats(self, ic: float) -> None:
        if not np.isfinite(ic):
            return
        if self._ic_w == 0.0:
            self._ic_mean = ic
            self._ic_var = 0.0
        else:
            # 指数加权均值/方差（折扣率即 discount）
            d = self.discount
            delta = ic - self._ic_mean
            self._ic_mean += (1 - d) * delta
            self._ic_var = d * self._ic_var + (1 - d) * delta * delta
        self._ic_w += 1.0

    def _icir(self) -> float:
        sd = float(np.sqrt(max(self._ic_var, 0.0)))
        return self._ic_mean / sd if sd > EPS else float("nan")

    def _discounted_grad(self, icir: float) -> np.ndarray:
        """折扣累加缓冲区内的 IC 梯度 + 换手次梯度。

        注意：**不含 L1 项** —— L1 由 :meth:`observe` 中的近端软阈值处理，
        放进这里会被 Adam 的梯度归一化抹平而失效。
        """
        total = np.zeros(self.n)
        n = len(self._buf)
        for age, (X, y) in enumerate(reversed(self._buf)):   # age=0 最新
            ic, g = _ic_and_grad(X, y, self.w)
            if not np.isfinite(ic):
                continue
            total += (self.discount ** age) * (-g)
        # Turnover 次梯度（换手惩罚是「上一权重」的函数，适合用次梯度）
        total += self.lambda_turnover * np.sign(self.w - self.w_prev)
        return total / max(1, n)

    # ---- 报告 ----

    def loss(self, icir: float | None = None, turnover: float | None = None) -> float:
        icir = self._icir() if icir is None else icir
        turnover = 0.0 if turnover is None else turnover
        core = -icir if np.isfinite(icir) else 0.0
        return float(core + self.lambda_turnover * turnover
                     + self.lambda_l1 * float(np.abs(self.w).sum()))

    @property
    def weights(self) -> np.ndarray:
        return self.w.copy()

    @property
    def ic(self) -> float:
        return self._last_ic

    @property
    def icir(self) -> float:
        return self._icir()

    def weights_dict(self, names: list[str]) -> dict[str, float]:
        return {n: float(w) for n, w in zip(names, self.w)}

    def top_factors(self, names: list[str], k: int = 10) -> list[tuple[str, float]]:
        pairs = sorted(zip(names, self.w), key=lambda t: abs(t[1]), reverse=True)
        return [(n, float(w)) for n, w in pairs[:k]]
