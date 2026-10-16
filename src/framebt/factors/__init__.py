"""framebt.factors：多因子体系的因子层（设计文档 §2）。

- ``ops``        ：因子算子（WQ101 算子集合）
- ``panel``      ：Panel 容器（研究整段 / 回测逐帧共用的数据形状）
- ``library``    ：因子库（WQ101 / GTJA191 风格）
- ``preprocess`` ：预处理流水线（MAD → 标准化 → 中性化 → 去冗余）
- ``evaluate``   ：IC / ICIR / t / 分组单调性 / 因子池筛选
"""

from __future__ import annotations

from .panel import Panel, SERIES
from .library import FACTOR_LIBRARY, compute, compute_many, list_factors
from .preprocess import (
    dedup_by_correlation, log_market_cap, mad_winsorize, neutralize,
    pairwise_ic_correlation, preprocess, zscore,
)
from .evaluate import (
    DEFAULT_SCREEN, cross_section_ic, forward_returns, group_returns, ic_summary,
    monotonicity_score, newey_west_tstat, realized_returns, screen_factor,
    screen_pool, top_bottom_spread,
)

__all__ = [
    "Panel", "SERIES",
    "FACTOR_LIBRARY", "compute", "compute_many", "list_factors",
    "mad_winsorize", "zscore", "neutralize", "log_market_cap", "preprocess",
    "pairwise_ic_correlation", "dedup_by_correlation",
    "cross_section_ic", "forward_returns", "realized_returns", "ic_summary",
    "newey_west_tstat", "group_returns", "monotonicity_score", "top_bottom_spread",
    "screen_factor", "screen_pool", "DEFAULT_SCREEN",
]
