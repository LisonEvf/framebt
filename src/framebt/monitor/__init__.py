"""framebt.monitor：因子健康度监控（§5.1 指标 / §5.2 CUSUM 变点 / §5.3 三级预警与降权）。

- changepoint.cusum_detect        ：双侧 CUSUM 变点检测（因子 IC 衰减报警）
- changepoint.KalmanICFilter      ：一维卡尔曼滤波（估计「当前真实 IC 水平」）
- changepoint.decay_probability   ：因子正在衰退的概率（[0,1]）
- changepoint.parameter_sensitivity：(k,h) 网格参数敏感性（自动降权前置条件）
- health.FactorHealthMonitor      ：IC/ICIR/FSC/拥挤度看板 + 三级预警 + 自动降权
- health.factor_stability / crowding_proxy / rolling_icir：单项指标

仅依赖 numpy + 标准库（无 sklearn / statsmodels / ruptures），全流程确定性、无随机数。
"""

from __future__ import annotations

from .changepoint import (
    CUSUMResult,
    KalmanICFilter,
    cusum_detect,
    decay_probability,
    parameter_sensitivity,
)
from .health import (
    GREEN, RED, YELLOW,
    FactorHealthMonitor,
    HealthMetrics,
    HealthMonitorConfig,
    crowding_proxy,
    factor_stability,
    rolling_icir,
)

__all__ = [
    "CUSUMResult",
    "KalmanICFilter",
    "cusum_detect",
    "decay_probability",
    "parameter_sensitivity",
    "FactorHealthMonitor",
    "HealthMetrics",
    "HealthMonitorConfig",
    "factor_stability",
    "crowding_proxy",
    "rolling_icir",
    "GREEN",
    "YELLOW",
    "RED",
]
