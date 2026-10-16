"""数据源抽象（§12 / 数据接入）。

``DataSource`` 是把「外部市场数据」接入引擎的统一入口。实现类负责把原始数据
（本仓库为 opentdx 数仓的 Parquet）转换为引擎可消费的 ``BarFeed``。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.events import Bar
from ..feed import BarFeed


class DataSource(ABC):
    """数据源抽象基类。"""

    name: str = "base"

    @abstractmethod
    def available_symbols(self) -> list[str]:
        """返回数据源中已有的标的列表。"""

    @abstractmethod
    def load(
        self,
        symbol: str,
        start=None,
        end=None,
        **params,
    ) -> BarFeed:
        """加载单标的的 K 线（返回按时间升序的 BarFeed）。"""

    def universe(
        self,
        symbols: list[str],
        start=None,
        end=None,
        **params,
    ) -> BarFeed:
        """加载多标的合并的 BarFeed（用于横截面回测）。"""
        all_bars: list[Bar] = []
        for sym in symbols:
            try:
                feed = self.load(sym, start=start, end=end, **params)
            except Exception:  # noqa: BLE001 - 单标的缺失不影响其余
                continue
            all_bars.extend(feed.bars)
        return BarFeed(all_bars)
