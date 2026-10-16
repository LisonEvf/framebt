"""因子健康监控与自动降权（设计文档 §5.1 / §5.3）。

监控指标
--------
| 指标 | 含义 | 预警阈值 |
|---|---|---|
| 截面 IC | 因子与下期收益的 Spearman 相关 | 均值连续 20 日 < 0 |
| ICIR（60 日滚动） | mean(IC)/std(IC) | ICIR < 0.2 |
| FSC | 因子排名的自相关（稳定性） | FSC < 0.5 |
| 拥挤度 | 因子暴露与「已走强/高换手」标的重合度 | 分位 > 90% |

三级预警与降权（§5.3）
----------------------
- 绿灯（衰退概率 < 0.3）：维持权重
- 黄灯（0.3 ≤ p < 0.6）：权重 × 0.5，进入观察期
- 红灯（p ≥ 0.6）：权重 → 0，移出活跃池并入康复观察池

**强制安全阀**：自动降权前必须做参数敏感性分析；若结论在 (k, h) 网格上不稳定
（``stable_across_parameters=False``），则只输出看板提示，**不执行**自动降权。
这条规则直接来自设计文档的「重要警示」。

拥挤度的诚实替代
----------------
设计文档原定义为「因子暴露与公募持仓的重合度」，但本数据集**没有基金持仓数据**。
故以可计算的**拥挤交易代理**替代：因子暴露与①近 20 日涨幅、②换手率 的平均截面
相关。取值越高，说明该因子当前押注的是「已经涨过、且交易拥挤」的标的。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .changepoint import (
    KalmanICFilter, cusum_detect, decay_probability, parameter_sensitivity,
)
from ..factors.ops import corrwith_rows

EPS = 1e-12
GREEN, YELLOW, RED = "GREEN", "YELLOW", "RED"
YELLOW_THRESHOLD = 0.3
RED_THRESHOLD = 0.6
DOWNGRADE_FACTOR = {GREEN: 1.0, YELLOW: 0.5, RED: 0.0}


# ===========================================================================
# 指标计算
# ===========================================================================

def factor_stability(factor: pd.DataFrame, lag: int = 1) -> float:
    """FSC（因子稳定性系数）：因子**截面排名**的自相关（时间平均）。

    排名自相关高 ⇒ 因子的相对排序稳定（换手低）；低 ⇒ 排名频繁翻转。
    """
    r = factor.rank(axis=1, pct=True)
    corr = corrwith_rows(r.shift(lag), r)
    return float(corr.mean()) if corr.notna().any() else float("nan")


def crowding_proxy(factor: pd.DataFrame, panel, window: int = 20) -> float:
    """拥挤度代理：因子暴露与「近期涨幅、换手率」的截面平均相关。

    无基金持仓数据（见模块 docstring），故用可计算的拥挤交易代理。
    内部会把面板对齐到因子的 index/columns，避免窗口不一致导致错位相关。
    """
    close = panel.close.reindex(index=factor.index, columns=factor.columns)
    turnover_df = panel.turnover.reindex(index=factor.index, columns=factor.columns)

    ret = close / close.shift(1) - 1.0
    trail = (1.0 + ret).rolling(window, min_periods=max(2, window // 2)).apply(
        np.prod, raw=True) - 1.0

    out = []
    for target in (trail, turnover_df):
        corr = corrwith_rows(factor.rank(axis=1, pct=True),
                             target.rank(axis=1, pct=True))
        if corr.notna().any():
            out.append(float(corr.mean()))
    return float(np.mean(out)) if out else float("nan")


def rolling_icir(ic: pd.Series, window: int = 60) -> pd.Series:
    """滚动 ICIR：``mean(IC, w) / std(IC, w)``。"""
    m = ic.rolling(window, min_periods=max(5, window // 3)).mean()
    s = ic.rolling(window, min_periods=max(5, window // 3)).std()
    return m / s.where(s.abs() > EPS)


# ===========================================================================
# 健康度评估
# ===========================================================================

@dataclass
class HealthMetrics:
    """单个因子的健康度快照。"""

    factor: str
    ic_mean: float
    icir: float
    icir_60d: float
    fsc: float
    crowding: float
    decay_prob: float
    status: str
    cusum_alarm: bool
    stable_across_params: bool
    downgrade: float          # 权重乘数（1.0 / 0.5 / 0.0）
    auto_downgrade: bool      # 是否真正执行自动降权
    note: str = ""

    @property
    def is_healthy(self) -> bool:
        return self.status == GREEN


@dataclass
class HealthMonitorConfig:
    icir_floor: float = 0.2
    fsc_floor: float = 0.5
    crowding_ceiling: float = 0.9
    yellow: float = YELLOW_THRESHOLD
    red: float = RED_THRESHOLD
    baseline_window: int = 120
    recent_window: int = 20
    #: 参数敏感性网格上「报警比例」门槛（透传给 parameter_sensitivity.threshold）
    sensitivity_alarm_rate: float = 0.5


class FactorHealthMonitor:
    """逐因子健康监控器：累积 IC 序列 → 三级预警 + （经敏感性检验的）自动降权。"""

    def __init__(self, config: HealthMonitorConfig | None = None):
        self.cfg = config or HealthMonitorConfig()
        self.ic_history: dict[str, list[float]] = {}
        self.metrics: dict[str, HealthMetrics] = {}
        self._kalman: dict[str, KalmanICFilter] = {}

    def update_ic(self, factor: str, ic: float) -> None:
        self.ic_history.setdefault(factor, []).append(float(ic))

    def evaluate(self, factor: str, *, ic_series: pd.Series | None = None,
                 fsc: float = float("nan"), crowding: float = float("nan"),
                 apply_downgrade: bool = True) -> HealthMetrics:
        """评估一个因子的健康度并给出降权系数。"""
        cfg = self.cfg
        if ic_series is None:
            ic_series = pd.Series(self.ic_history.get(factor, []), dtype=float)
        ic_series = ic_series.dropna()

        if len(ic_series) < 10:
            m = HealthMetrics(factor, float("nan"), float("nan"), float("nan"),
                              fsc, crowding, float("nan"), GREEN, False, True,
                              1.0, False, "IC 样本不足，暂不评估")
            self.metrics[factor] = m
            return m

        ic_mean = float(ic_series.mean())
        icir = (ic_mean / float(ic_series.std(ddof=1))
                if float(ic_series.std(ddof=1)) > EPS else float("nan"))
        icir60 = float(rolling_icir(ic_series, 60).iloc[-1])

        # 卡尔曼平滑后的衰退概率（避免单日噪声触发误报）
        p = decay_probability(ic_series.to_numpy(), baseline_window=cfg.baseline_window,
                              recent_window=cfg.recent_window, use_kalman=True)
        status = GREEN if p < cfg.yellow else (YELLOW if p < cfg.red else RED)

        cusum = cusum_detect(ic_series.to_numpy())
        # 参数敏感性：只有结论稳健（网格上报警结论一致）才允许自动降权
        sens = parameter_sensitivity(ic_series.to_numpy(),
                                     threshold=cfg.sensitivity_alarm_rate)
        stable = bool(sens.get("stable_across_parameters", False))
        recommend = bool(sens.get("recommend_auto_downgrade", False))

        note = ""
        downgrade = DOWNGRADE_FACTOR[status]
        if status != GREEN and not recommend:
            downgrade = 1.0
            note = ("参数敏感性不稳健 → 仅看板提示，不自动降权"
                    if not stable else "网格报警比例不足 → 仅看板提示")

        m = HealthMetrics(
            factor=factor, ic_mean=ic_mean, icir=icir, icir_60d=icir60,
            fsc=fsc, crowding=crowding, decay_prob=p, status=status,
            cusum_alarm=bool(cusum.alarm), stable_across_params=stable,
            downgrade=downgrade,
            auto_downgrade=bool(apply_downgrade and status != GREEN and recommend),
            note=note,
        )
        self.metrics[factor] = m
        return m

    def downgrade_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """按健康度对权重降权并重新归一（红灯因子权重归零）。"""
        out = {}
        for name, w in weights.items():
            m = self.metrics.get(name)
            mult = m.downgrade if m is not None else 1.0
            out[name] = w * mult
        s = sum(abs(v) for v in out.values())
        return {k: v / s for k, v in out.items()} if s > EPS else out

    def dashboard(self) -> pd.DataFrame:
        """监控看板：所有已评估因子的健康指标。"""
        if not self.metrics:
            return pd.DataFrame()
        rows = [{
            "factor": m.factor, "status": m.status, "ic_mean": m.ic_mean,
            "icir": m.icir, "icir_60d": m.icir_60d, "fsc": m.fsc,
            "crowding": m.crowding, "decay_prob": m.decay_prob,
            "cusum_alarm": m.cusum_alarm, "stable": m.stable_across_params,
            "downgrade": m.downgrade, "auto": m.auto_downgrade, "note": m.note,
        } for m in self.metrics.values()]
        df = pd.DataFrame(rows)
        order = {GREEN: 0, YELLOW: 1, RED: 2}
        df["_o"] = df["status"].map(order)
        return df.sort_values(["_o", "decay_prob"], ascending=[False, False]) \
                 .drop(columns="_o").reset_index(drop=True)
