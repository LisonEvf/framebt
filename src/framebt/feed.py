"""BarFeed：原始 K 线数据源（§2 / §12）。

数据加载阶段：
  - 按 ``(timestamp, sequence_id)`` 排序（I4），sequence_id 预分配保证全序。
  - 支持从「dict 列表」或「Bar 列表」构造。
  - 多标的：timestamp 为跨标的统一逻辑时间轴。

数据源接入见 ``framebt.data_sources``（opentdx 数仓 Parquet 读取）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .core.events import Bar


class BarFeed:
    def __init__(self, bars: Sequence[Bar]):
        self.bars = self._sort(list(bars))

    @staticmethod
    def _sort(bars: list[Bar]) -> list[Bar]:
        # I4：(timestamp, sequence_id) 严格升序
        return sorted(bars, key=lambda b: (b.timestamp, b.sequence_id))

    @classmethod
    def from_dicts(
        cls,
        rows: Iterable[dict],
        symbol: str = "S",
        sequence_base: int = 0,
    ) -> "BarFeed":
        """从字典列表构造。每行需含 open/high/low/close；timestamp/volume 有默认。"""
        bars: list[Bar] = []
        for i, r in enumerate(rows):
            bars.append(
                Bar(
                    timestamp=int(r.get("timestamp", i)),
                    symbol=r.get("symbol", symbol),
                    open=float(r["open"]),
                    high=float(r.get("high", r["open"])),
                    low=float(r.get("low", r["open"])),
                    close=float(r["close"]),
                    volume=float(r.get("volume", 0.0)),
                    amount=float(r.get("amount", 0.0)),
                    float_shares=float(r.get("float_shares", 0.0)),
                    turnover=float(r.get("turnover", 0.0)),
                    sequence_id=sequence_base + i,
                )
            )
        return cls(bars)

    def timestamps(self) -> list[int]:
        return [b.timestamp for b in self.bars]

    def symbols(self) -> list[str]:
        return sorted({b.symbol for b in self.bars})

    def span(self) -> tuple[int, int]:
        if not self.bars:
            return (0, 0)
        return self.bars[0].timestamp, self.bars[-1].timestamp
