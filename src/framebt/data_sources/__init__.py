"""framebt.data_sources：数据源接入（§12）。

把「外部市场数据」转换为引擎可消费的 ``BarFeed``。默认数据源为 opentdx 数仓
的 Parquet K 线。
"""

from __future__ import annotations

from .base import DataSource
from .opentdx import OpentdxParquetSource, default_data_dir

__all__ = [
    "DataSource",
    "OpentdxParquetSource",
    "default_data_dir",
]
