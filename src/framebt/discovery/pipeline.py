"""新因子正交化入栈流水线与影子运行（设计文档 §6.4 / §6.5 / §6.6）。

流水线（§6.6）::

    [符号回归/神经网络生成候选]
            ↓
    [IC/ICIR 初筛]            → 淘汰大部分
            ↓
    [与现有因子池相关性检验]   → 淘汰冗余
            ↓
    [增量信息检验 VIF]        → 淘汰无增量者
            ↓
    [影子运行 N 个交易日]      → 淘汰与实盘信号高度重合者
            ↓
    [正式入栈，初始权重 ≤ 5%]  → 交给 FactorRegistry（观察期）

设计原则：**每一步的门槛都显式可配、可复现**，并且拒绝理由完整记录，
避免「黑箱入栈」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..factors.evaluate import cross_section_ic, ic_summary
from ..factors.lifecycle import OBSERVE_MAX_WEIGHT, FactorRegistry
from ..factors.preprocess import neutralize
from ..factors.ops import corrwith_rows

EPS = 1e-12


# ===========================================================================
# §6.4 Step 1 —— 正交化
# ===========================================================================

def orthogonalize(new_factor: pd.DataFrame,
                  existing: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """对现有因子池做**逐截面回归取残差**（§6.4 Step1）。

    返回与 ``new_factor`` 同形的残差面板：其中的信息是现有因子池**无法解释**的部分。
    若现有池为空，原样返回。
    """
    if not existing:
        return new_factor.copy()
    ex = [f.reindex_like(new_factor) for f in existing.values()]
    return neutralize(new_factor, ex, add_const=True)


# ===========================================================================
# §6.4 Step 2 —— 增量信息检验
# ===========================================================================

def variance_inflation_factors(factors: dict[str, pd.DataFrame]) -> dict[str, float]:
    """逐因子 VIF = 1/(1−R²)，R² 来自该因子对其余因子的**截面**回归（时间平均）。

    VIF > 10 通常视为多重共线性严重（经验阈值）。
    """
    names = list(factors)
    if len(names) < 2:
        return {n: 1.0 for n in names}
    out: dict[str, float] = {}
    for target in names:
        others = [factors[n] for n in names if n != target]
        y = factors[target]
        r2s = []
        for t in y.index:
            yy = y.loc[t]
            X = pd.concat([o.loc[t] for o in others], axis=1)
            m = yy.notna() & X.notna().all(axis=1)
            if m.sum() <= X.shape[1] + 1:
                continue
            ym = yy[m].to_numpy(float)
            Xm = X[m].to_numpy(float)
            Xm = np.column_stack([np.ones(len(Xm)), Xm])
            try:
                beta, *_ = np.linalg.lstsq(Xm, ym, rcond=None)
            except np.linalg.LinAlgError:  # pragma: no cover
                continue
            resid = ym - Xm @ beta
            ss_res = float(resid @ resid)
            ss_tot = float(((ym - ym.mean()) ** 2).sum())
            if ss_tot > EPS:
                r2s.append(1.0 - ss_res / ss_tot)
        r2 = float(np.mean(r2s)) if r2s else 0.0
        r2 = min(max(r2, 0.0), 0.999999)
        out[target] = float(1.0 / (1.0 - r2))
    return out


def incremental_icir(existing: dict[str, pd.DataFrame],
                     new_factor: pd.DataFrame,
                     forward_ret: pd.DataFrame,
                     weights: dict[str, float] | None = None) -> dict:
    """新因子是否带来**增量预测力**（§6.4 Step2 的 Granger 式检验）。

    做法：比较「现有因子池等权/加权合成」与「加入新因子后」的 ICIR，
    并检验增量是否为正、是否超过噪声水平（用 IC 序列差的 t 统计量近似）。
    """
    from ..portfolio.combine import composite_score

    w = weights or {n: 1.0 for n in existing}
    base_score = composite_score(existing, w) if existing else None
    merged = dict(existing)
    merged["__new__"] = new_factor
    w2 = dict(w)
    w2["__new__"] = 1.0
    new_score = composite_score(merged, w2)

    fwd = forward_ret
    if base_score is None:
        ic_new = cross_section_ic(new_factor, fwd).dropna()
        s_new = ic_summary(ic_new)
        return {"base_icir": float("nan"), "new_icir": s_new["icir"],
                "delta_icir": float("nan"), "delta_t": float("nan"),
                "has_increment": bool(np.isfinite(s_new["icir"]) and s_new["icir"] > 0)}

    ic_base = cross_section_ic(base_score, fwd).dropna()
    ic_new = cross_section_ic(new_score, fwd).dropna()
    common = ic_base.index.intersection(ic_new.index)
    s_base = ic_summary(ic_base)
    s_new = ic_summary(ic_new)

    delta = (ic_new.loc[common] - ic_base.loc[common]) if len(common) > 5 else pd.Series(dtype=float)
    d_t = float(delta.mean() / (delta.std(ddof=1) / np.sqrt(len(delta)))) if len(delta) > 5 and delta.std(ddof=1) > EPS else float("nan")
    return {
        "base_icir": s_base["icir"], "new_icir": s_new["icir"],
        "delta_icir": (s_new["icir"] - s_base["icir"])
        if np.isfinite(s_base["icir"]) and np.isfinite(s_new["icir"]) else float("nan"),
        "delta_t": d_t,
        "has_increment": bool(np.isfinite(d_t) and d_t > 1.0),
    }


# ===========================================================================
# §6.5 影子运行
# ===========================================================================

@dataclass
class ShadowReport:
    """影子运行报告（§6.5）。"""

    factor: str
    days: int
    corr_with_live: float
    ic_mean: float
    passed: bool
    note: str = ""


class ShadowFactorRunner:
    """影子运行器：候选因子在实盘/回测中**只记录信号、不参与交易**。

    运行 ``min_days`` 个交易日后评估：
      - 与现有实盘信号的相关性应 ``< max_corr``（否则没有新增信息）
      - 自身 IC 应持续为正
    设计文档建议至少 20 个交易日。
    """

    def __init__(self, factor_name: str, min_days: int = 20, max_corr: float = 0.3):
        self.factor_name = factor_name
        self.min_days = int(min_days)
        self.max_corr = float(max_corr)
        self.signals: list[tuple[int, pd.Series]] = []
        self.live: list[tuple[int, pd.Series]] = []

    def on_bar(self, ts: int, signal: pd.Series, live_signal: pd.Series) -> None:
        self.signals.append((ts, signal))
        self.live.append((ts, live_signal))

    def report(self, forward_ret: pd.DataFrame | None = None) -> ShadowReport:
        n = len(self.signals)
        if n == 0:
            return ShadowReport(self.factor_name, 0, float("nan"), float("nan"),
                                False, "无影子数据")
        S = pd.DataFrame({ts: s for ts, s in self.signals}).T
        L = pd.DataFrame({ts: s for ts, s in self.live}).T
        corr = corrwith_rows(S.rank(axis=1), L.reindex_like(S).rank(axis=1))
        corr_mean = float(corr.mean()) if corr.notna().any() else float("nan")

        ic_mean = float("nan")
        if forward_ret is not None:
            ic = cross_section_ic(S, forward_ret.reindex(S.index))
            ic_mean = float(ic.mean()) if ic.notna().any() else float("nan")

        enough = n >= self.min_days
        low_corr = (not np.isfinite(corr_mean)) or abs(corr_mean) < self.max_corr
        pos_ic = (not np.isfinite(ic_mean)) or ic_mean > 0
        passed = bool(enough and low_corr and pos_ic)
        note = []
        if not enough:
            note.append(f"影子天数 {n} < {self.min_days}")
        if not low_corr:
            note.append(f"与实盘信号相关 {corr_mean:+.3f} 过高")
        if not pos_ic:
            note.append(f"影子 IC {ic_mean:+.4f} 非正")
        return ShadowReport(self.factor_name, n, corr_mean, ic_mean, passed,
                            "; ".join(note) or "通过")


# ===========================================================================
# §6.6 自动入栈流水线
# ===========================================================================

@dataclass
class StackingConfig:
    min_abs_ic: float = 0.02
    min_abs_t: float = 2.0
    max_corr_with_pool: float = 0.5
    max_vif: float = 10.0
    min_delta_t: float = 1.0
    shadow_days: int = 20
    shadow_max_corr: float = 0.3
    initial_weight_cap: float = OBSERVE_MAX_WEIGHT


@dataclass
class StackingDecision:
    """入栈决策 + 完整拒绝理由（可审计）。"""

    factor: str
    accepted: bool
    stage: str
    reasons: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


class FactorStackingPipeline:
    """§6.6 自动入栈流水线：候选因子 → 逐级筛选 → 注册为观察期因子。"""

    def __init__(self, config: StackingConfig | None = None,
                 registry: FactorRegistry | None = None):
        self.cfg = config or StackingConfig()
        self.registry = registry or FactorRegistry()
        self.decisions: list[StackingDecision] = []

    def evaluate(self, name: str, candidate: pd.DataFrame,
                 pool: dict[str, pd.DataFrame], forward_ret: pd.DataFrame,
                 *, source: str = "symbolic", birth_date: int = 0,
                 shadow: ShadowReport | None = None) -> StackingDecision:
        """逐级评估一个候选因子；通过则注册为观察期因子。"""
        cfg = self.cfg
        m: dict = {}

        # ---- 1. IC/ICIR 初筛 ----
        ic = cross_section_ic(candidate, forward_ret)
        s = ic_summary(ic)
        m.update({"ic_mean": s["ic_mean"], "icir": s["icir"], "t_stat": s["t_stat"],
                  "n": s["n"]})
        reason = []
        if not (abs(s["ic_mean"]) > cfg.min_abs_ic):
            reason.append(f"|IC|={abs(s['ic_mean']):.4f} ≤ {cfg.min_abs_ic}")
        if not (abs(s["t_stat"]) > cfg.min_abs_t):
            reason.append(f"|t|={abs(s['t_stat']):.2f} ≤ {cfg.min_abs_t}")
        if reason:
            return self._decide(name, False, "1.IC初筛", reason, m)

        # ---- 2. 与现有池的相关性 ----
        max_corr, worst = 0.0, None
        for pname, pf in pool.items():
            c = corrwith_rows(candidate.rank(axis=1), pf.rank(axis=1))
            cv = float(c.mean()) if c.notna().any() else float("nan")
            if np.isfinite(cv) and abs(cv) > max_corr:
                max_corr, worst = abs(cv), pname
        m["max_corr_with_pool"] = max_corr
        if pool and max_corr > cfg.max_corr_with_pool:
            return self._decide(name, False, "2.相关性", 
                                [f"与 {worst} 相关 {max_corr:+.3f} > {cfg.max_corr_with_pool}"], m)

        # ---- 3. 增量信息（VIF + 增量 ICIR）----
        vifs = variance_inflation_factors({**pool, name: candidate}) if pool else {name: 1.0}
        vif = vifs.get(name, 1.0)
        inc = incremental_icir(pool, candidate, forward_ret) if pool else {
            "delta_t": float("nan"), "has_increment": True, "delta_icir": float("nan")}
        m.update({"vif": vif, **inc})
        reason = []
        if vif > cfg.max_vif:
            reason.append(f"VIF={vif:.1f} > {cfg.max_vif}（与现有池共线）")
        if pool and not inc["has_increment"] and not (inc["delta_t"] != inc["delta_t"]):
            reason.append(f"增量 t={inc['delta_t']:.2f} ≤ {cfg.min_delta_t}（无增量预测力）")
        if reason:
            return self._decide(name, False, "3.增量信息", reason, m)

        # ---- 4. 影子运行 ----
        if shadow is not None:
            m.update({"shadow_days": shadow.days,
                      "shadow_corr": shadow.corr_with_live,
                      "shadow_ic": shadow.ic_mean})
            if shadow.days < cfg.shadow_days:
                return self._decide(name, False, "4.影子运行",
                                    [f"影子天数 {shadow.days} < {cfg.shadow_days}"], m)
            if np.isfinite(shadow.corr_with_live) and abs(shadow.corr_with_live) >= cfg.shadow_max_corr:
                return self._decide(name, False, "4.影子运行",
                                    [f"与实盘信号相关 {shadow.corr_with_live:+.3f} 过高"], m)
            if np.isfinite(shadow.ic_mean) and shadow.ic_mean <= 0:
                return self._decide(name, False, "4.影子运行",
                                    [f"影子 IC {shadow.ic_mean:+.4f} 非正"], m)

        # ---- 5. 入栈（观察期，权重受限）----
        self.registry.register(name, expression=f"discovered:{name}", source=source,
                              birth_date=birth_date, weight=cfg.initial_weight_cap)
        return self._decide(name, True, "5.入栈", [], m)

    def _decide(self, name, accepted, stage, reasons, metrics) -> StackingDecision:
        d = StackingDecision(name, accepted, stage, list(reasons), dict(metrics))
        self.decisions.append(d)
        return d

    def report(self) -> pd.DataFrame:
        if not self.decisions:
            return pd.DataFrame()
        return pd.DataFrame([{
            "factor": d.factor, "accepted": d.accepted, "stage": d.stage,
            "reasons": "; ".join(d.reasons), **d.metrics,
        } for d in self.decisions])
