"""因子生命周期管理（设计文档 §7.1）。

每个因子从入栈到退休的完整轨迹都被记录，并可按规则自动流转：

    birth ──▶ observing ──▶ active ──▶ retired ──▶（康复）──▶ active
                (观察期)                  (退休池)

流转规则
--------
- **入栈**：新因子以 ``observing`` 状态进入，初始权重受上限约束（观察期内 ≤ 5%）。
- **提权**：观察期（默认 3 个月 ≈ 60 交易日）内 ICIR 持续 > 0.5 → 转 ``active``。
- **退休**：连续 ``retire_days``（默认 60）个交易日 ICIR < 0.2 **且** CUSUM 检出
  显著衰退 → 转 ``retired``，权重归零。
- **康复**：退休 ``rehab_days``（默认 120）交易日后若 IC 恢复至阈值以上 → 重新
  ``active``。

**注意**：退休只说明「近期数据支持降权」，不证明因子永久失效（设计文档警示），
因此退休池保留完整历史，随时可康复。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Optional

import pandas as pd

ACTIVE, OBSERVING, RETIRED = "active", "observing", "retired"
SOURCES = ("manual", "symbolic", "neural", "event")

OBSERVE_MAX_WEIGHT = 0.05     # 观察期内权重上限（设计文档 §6.4）
PROMOTE_ICIR = 0.5
RETIRE_ICIR = 0.2


@dataclass
class FactorRecord:
    """单因子的生命周期档案（对应设计文档 §7.1 字段表）。"""

    factor_id: str
    expression: str = ""
    source: str = "manual"
    birth_date: int = 0
    status: str = OBSERVING
    current_weight: float = 0.0
    icir_60d: float = float("nan")
    ic_mean: float = float("nan")
    fsc: float = float("nan")
    crowding: float = float("nan")
    observe_days: int = 0
    icir_above_streak: int = 0
    icir_below_streak: int = 0
    days_since_retire: int = 0
    retire_date: Optional[int] = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["notes"] = "; ".join(self.notes[-3:])
        return d


class FactorRegistry:
    """因子注册表 + 生命周期状态机。"""

    def __init__(self, *, observe_days: int = 60, retire_days: int = 60,
                 rehab_days: int = 120, promote_icir: float = PROMOTE_ICIR,
                 retire_icir: float = RETIRE_ICIR,
                 observe_max_weight: float = OBSERVE_MAX_WEIGHT):
        self.observe_days = observe_days
        self.retire_days = retire_days
        self.rehab_days = rehab_days
        self.promote_icir = promote_icir
        self.retire_icir = retire_icir
        self.observe_max_weight = observe_max_weight
        self.records: dict[str, FactorRecord] = {}
        self.events: list[tuple[int, str, str]] = []   # (date, factor, event)

    # ---- 入栈 ----

    def register(self, factor_id: str, expression: str = "", source: str = "manual",
                 birth_date: int = 0, weight: float = 0.0) -> FactorRecord:
        if source not in SOURCES:
            raise ValueError(f"source 应为 {SOURCES} 之一，得到 {source!r}")
        rec = FactorRecord(
            factor_id=factor_id, expression=expression, source=source,
            birth_date=birth_date, status=OBSERVING,
            current_weight=min(abs(weight), self.observe_max_weight),
            notes=[f"入栈（{source}），初始权重 ≤ {self.observe_max_weight:.0%}"],
        )
        self.records[factor_id] = rec
        self.events.append((birth_date, factor_id, "register"))
        return rec

    # ---- 逐日推进 ----

    def step(self, date: int, *, icir: float = float("nan"),
             ic_mean: float = float("nan"), fsc: float = float("nan"),
             crowding: float = float("nan"), cusum_alarm: bool = False,
             weight: float | None = None) -> None:
        """按一个交易日推进所有因子的状态机。"""
        for rec in self.records.values():
            rec.icir_60d = icir if factor_matches(rec, icir) else rec.icir_60d
            if rec.factor_id and icir == icir:
                rec.icir_60d = icir
                rec.ic_mean = ic_mean
                rec.fsc = fsc
                rec.crowding = crowding
            if weight is not None:
                rec.current_weight = weight

            if rec.status == OBSERVING:
                rec.observe_days += 1
                if rec.icir_60d > self.promote_icir:
                    rec.icir_above_streak += 1
                else:
                    rec.icir_above_streak = 0
                if (rec.observe_days >= self.observe_days
                        and rec.icir_above_streak >= self.observe_days):
                    rec.status = ACTIVE
                    rec.notes.append(f"{date}: 观察期满且 ICIR 持续 > {self.promote_icir} → 转 active")
                    self.events.append((date, rec.factor_id, "promote"))

            elif rec.status == ACTIVE:
                if rec.icir_60d < self.retire_icir:
                    rec.icir_below_streak += 1
                else:
                    rec.icir_below_streak = 0
                # 退休需「ICIR 持续低于阈值」且「CUSUM 检出显著衰退」
                if rec.icir_below_streak >= self.retire_days and cusum_alarm:
                    rec.status = RETIRED
                    rec.retire_date = date
                    rec.days_since_retire = 0
                    rec.current_weight = 0.0
                    rec.notes.append(
                        f"{date}: ICIR 连续 {rec.icir_below_streak} 日 < {self.retire_icir} "
                        f"且 CUSUM 报警 → 退休")
                    self.events.append((date, rec.factor_id, "retire"))

            elif rec.status == RETIRED:
                rec.days_since_retire += 1
                if (rec.days_since_retire >= self.rehab_days
                        and rec.icir_60d > self.promote_icir):
                    rec.status = ACTIVE
                    rec.icir_below_streak = 0
                    rec.notes.append(f"{date}: 退休满 {self.rehab_days} 日且 ICIR 恢复 → 康复")
                    self.events.append((date, rec.factor_id, "rehabilitate"))

    # ---- 查询 / 导出 ----

    def active_factors(self) -> list[str]:
        return [f for f, r in self.records.items() if r.status == ACTIVE]

    def observing_factors(self) -> list[str]:
        return [f for f, r in self.records.items() if r.status == OBSERVING]

    def retired_factors(self) -> list[str]:
        return [f for f, r in self.records.items() if r.status == RETIRED]

    def tradable_factors(self) -> list[str]:
        """可参与实盘加权的因子（active + observing）。"""
        return [f for f, r in self.records.items() if r.status in (ACTIVE, OBSERVING)]

    def cap_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """对观察期因子施加权重上限（§6.4），并保持权重和为 1。

        实现要点：**不能「先截断再归一」** —— 归一化会把被压下去的部分又按比例放大
        回去（例：new 截到 0.05、old 0.1 → 归一后 new 又变成 0.333，上限形同虚设）。
        正确做法是「归一 → 截断 → 把释放出的预算按比例分给未受限因子 → 再归一」，
        并迭代若干次直至收敛。
        """
        total = sum(abs(v) for v in weights.values())
        if total <= 0:
            return dict(weights)
        w = {k: float(v) / total for k, v in weights.items()}

        capped = {k for k in w
                  if (rec := self.records.get(k)) is not None and rec.status == OBSERVING}
        if not capped:
            return w

        cap = self.observe_max_weight
        free = [k for k in w if k not in capped and abs(w[k]) > 0]

        for _ in range(10):
            excess = 0.0
            for k in capped:
                if abs(w[k]) > cap:
                    excess += abs(w[k]) - cap
                    w[k] = math.copysign(cap, w[k])
            if excess <= 1e-15:
                break
            fsum = sum(abs(w[k]) for k in free)
            if fsum <= 1e-15:
                break
            for k in free:                       # 把释放的预算按比例转给未受限因子
                w[k] += excess * (w[k] / fsum)

        s = sum(abs(v) for v in w.values())
        return {k: v / s for k, v in w.items()} if s > 0 else w

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.records.values()])

    def summary(self) -> dict:
        return {
            "total": len(self.records),
            "active": len(self.active_factors()),
            "observing": len(self.observing_factors()),
            "retired": len(self.retired_factors()),
        }


def factor_matches(rec: FactorRecord, icir: float) -> bool:
    """占位：``step`` 目前对全体因子用同一 icir（研究模式下的简化）。

    实盘/精细化模式下应传入 ``{factor: icir}`` 映射逐因子推进；此函数保留扩展点。
    """
    return False
