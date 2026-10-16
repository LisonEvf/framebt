"""神经网络因子挖掘（设计文档 §6.3）。

通过可微网络**直接生成因子**并端到端优化，与符号回归（§6.2）互补：

  - 符号回归产出**可读表达式**（人可审查、可解释），但搜索空间受算子集合限制；
  - 神经网络产出**隐因子**（不可读），但能表达算子集合之外的非线性结构。

结构（与设计文档一致）::

    encoder:  x (n_features) → 64 → 32        # 逐标的、逐时刻的前向网络
    factor_head: 32 → n_factors               # 生成 n_factors 个隐因子
    predictor:  n_factors → 1                 # 用隐因子预测收益

损失：``−IC(pred, target) + λ · 正交惩罚``

正交惩罚促使新因子与**现有因子池**不相关（§6.4 的正交化要求），避免生成一堆
彼此高度相关的伪因子。

因果性：网络只用**输入特征**（由历史行情构造）预测**同截面的未来收益**；训练在
样本内进行，产出的是「因子函数」，其调用仍受 :class:`~framebt.data_bus.CausalToken`
约束（与库内因子走同一条通道）。

依赖：PyTorch（可选）。未安装时本模块不可用，但其余部分不受影响。
"""

from __future__ import annotations

import numpy as np

try:  # torch 为可选依赖
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - 环境无 torch
    _HAS_TORCH = False

EPS = 1e-8


def _require_torch() -> None:
    if not _HAS_TORCH:
        raise ImportError(
            "神经网络因子挖掘需要 PyTorch（pip install torch）；"
            "未安装时可改用 framebt.discovery.symbolic 的符号回归路径。"
        )


def _rank_ic(pred, target, mask=None):
    """可微的截面 IC 代理：对去均值后的预测/真实值求 Pearson 相关。

    用「逐截面 Pearson」而非秩相关，是为了让梯度可回传（秩不可微）。
    输入均为预处理后的标准化量时，它与 Spearman IC 高度一致。
    """
    if mask is not None:
        pred = pred * mask
        target = target * mask
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    num = (p * t).sum(dim=-1)
    den = p.pow(2).sum(dim=-1).sqrt() * t.pow(2).sum(dim=-1).sqrt() + EPS
    return num / den


class NeuralFactorGenerator:
    """隐因子生成器（§6.3）。

    参数：
      n_features  ：输入特征个数
      n_factors   ：生成的隐因子个数（设计文档建议 10–20）
      hidden      ：编码器隐层宽度
      ortho_lambda：正交惩罚权重
      seed        ：随机种子（保证可复现）
      device      ：'cpu' / 'cuda'
    """

    def __init__(self, n_features: int, n_factors: int = 10, hidden: int = 64,
                 ortho_lambda: float = 0.1, seed: int = 0, device: str = "cpu"):
        _require_torch()
        if n_features <= 0 or n_factors <= 0:
            raise ValueError("n_features / n_factors 必须为正")
        self.n_features = int(n_features)
        self.n_factors = int(n_factors)
        self.ortho_lambda = float(ortho_lambda)
        self.device = device

        torch.manual_seed(seed)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
        ).to(device)
        self.factor_head = nn.Linear(hidden // 2, self.n_factors).to(device)
        self.predictor = nn.Linear(self.n_factors, 1).to(device)
        self.history: list[float] = []

    # ---- 前向 ----

    def forward(self, x):
        """x: (..., n_features) → (factors (..., n_factors), pred (...))"""
        h = self.encoder(x)
        f = self.factor_head(h)
        p = self.predictor(f).squeeze(-1)
        return f, p

    def factors(self, x: np.ndarray) -> np.ndarray:
        """对输入特征生成隐因子（numpy 进出，便于接入因子体系）。"""
        self.encoder.eval(); self.factor_head.eval()
        with torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float32, device=self.device)
            f, _ = self.forward(xt)
        return f.cpu().numpy()

    # ---- 训练 ----

    def fit(self, X: np.ndarray, y: np.ndarray,
            existing: np.ndarray | None = None,
            epochs: int = 60, lr: float = 1e-3, verbose: bool = False) -> list[float]:
        """训练。

        X：(n_times, n_symbols, n_features)；y：(n_times, n_symbols) 未来收益。
        existing：(n_times, n_symbols, n_existing) 现有因子池（用于正交惩罚），可选。
        """
        _require_torch()
        self.encoder.train(); self.factor_head.train(); self.predictor.train()
        opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.factor_head.parameters())
            + list(self.predictor.parameters()), lr=lr)

        Xt = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        yt = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        Et = (torch.as_tensor(existing, dtype=torch.float32, device=self.device)
              if existing is not None else None)

        finite = torch.isfinite(yt)
        self.history = []
        for ep in range(epochs):
            opt.zero_grad()
            f, p = self.forward(Xt)
            ic = _rank_ic(p, torch.nan_to_num(yt), finite.float()).mean()
            loss = -ic
            if Et is not None and self.ortho_lambda > 0:
                # 正交惩罚：新因子与现有因子的截面相关绝对值
                pen = 0.0
                for j in range(Et.shape[-1]):
                    ej = torch.nan_to_num(Et[..., j])
                    for i in range(self.n_factors):
                        fi = f[..., i]
                        m = torch.isfinite(Et[..., j]) & torch.isfinite(fi)
                        if m.sum() < 2:
                            continue
                        c = _rank_ic(fi, ej, m.float()).mean()
                        pen = pen + torch.abs(c)
                loss = loss + self.ortho_lambda * pen / max(1, self.n_factors)
            loss.backward()
            opt.step()
            self.history.append(float(loss.detach().cpu()))
            if verbose and (ep % 10 == 0 or ep == epochs - 1):
                print(f"  epoch {ep:3d}  loss={self.history[-1]:+.5f}  IC={float(ic):+.4f}")
        return self.history

    # ---- 报告 ----

    def ic_summary(self, X: np.ndarray, y: np.ndarray) -> dict:
        """训练后的样本内 IC 汇总（每个隐因子与预测组合）。"""
        f = self.factors(X)
        pred = f @ self.predictor.weight.detach().cpu().numpy().ravel()
        def _ic(a, b):
            ics = []
            for t in range(a.shape[0]):
                m = np.isfinite(a[t]) & np.isfinite(b[t])
                if m.sum() < 5:
                    continue
                av, bv = a[t][m], b[t][m]
                av = av - av.mean(); bv = bv - bv.mean()
                d = np.sqrt((av ** 2).sum() * (bv ** 2).sum())
                if d > EPS:
                    ics.append(float((av * bv).sum() / d))
            return float(np.mean(ics)) if ics else float("nan")

        per_factor = [_ic(f[:, :, i], y) for i in range(self.n_factors)]
        return {"composite_ic": _ic(pred, y), "per_factor_ic": per_factor,
                "mean_abs_ic": float(np.nanmean(np.abs(per_factor)))}
