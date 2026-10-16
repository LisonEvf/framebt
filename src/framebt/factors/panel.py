"""面板（Panel）容器：因子计算的统一数据形状。

``Panel`` 是一组对齐的 ``pandas.DataFrame``，``index=时间戳``、``columns=标的``。
因子函数只认 ``Panel``，因此**同一份因子定义**既能用于：

  - **研究/评估**：``Panel.from_feed(feed)`` 取整段历史，向量化算全panel；
  - **回测执行**：``Panel.from_bus(bus, symbols, token, window=...)`` 每帧从
    因果总线取「截至当前帧」的窗口 —— 受 ``CausalToken`` 约束，天然无前视。

这就是设计文档「因子定义、权重计算、回测执行三者共享同一套逻辑」的落地点：
不存在第二份因子实现，只有数据来源不同。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterable, Optional

import pandas as pd

# 面板包含的序列名（与引擎 Phase 1 注入 DataBus 的 key 一致）
SERIES = ("open", "high", "low", "close", "volume", "amount",
          "vwap", "market_cap", "turnover", "float_shares")


@dataclass
class Panel:
    """对齐的多序列面板。"""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    amount: pd.DataFrame
    vwap: pd.DataFrame
    market_cap: pd.DataFrame
    turnover: pd.DataFrame
    float_shares: pd.DataFrame

    # ---- 构造 ----

    @classmethod
    def from_feed(cls, feed) -> "Panel":
        """由 ``BarFeed`` 构造整段面板（研究用）。"""
        rows = []
        for b in feed.bars:
            rows.append({
                "timestamp": b.timestamp, "symbol": b.symbol,
                "open": b.open, "high": b.high, "low": b.low, "close": b.close,
                "volume": b.volume, "amount": b.amount,
                "vwap": b.vwap if b.vwap > 0 else (
                    (b.amount / b.volume) if b.volume > 0 else b.close),
                "market_cap": b.market_cap if b.market_cap > 0 else b.close * b.float_shares,
                "turnover": b.turnover, "float_shares": b.float_shares,
            })
        return cls.from_records(rows)

    @classmethod
    def from_records(cls, rows: Iterable[dict]) -> "Panel":
        df = pd.DataFrame(list(rows))
        if df.empty:
            empty = pd.DataFrame()
            return cls(**{k: empty for k in SERIES})
        return cls(**{
            name: df.pivot_table(index="timestamp", columns="symbol",
                                 values=name, aggfunc="last").sort_index()
            for name in SERIES
        })

    @classmethod
    def from_bus(cls, bus, symbols: list[str], token,
                 window: Optional[int] = None) -> "Panel":
        """由因果总线取「截至 token.issued_at」的窗口面板（回测用）。

        ``window`` 为回溯期数；``None`` 表示取全部可见历史。
        总线自身会按 ``token`` 过滤未来数据，故本方法构造的面板必然因果安全。
        """
        series: dict[str, pd.DataFrame] = {}
        for name in SERIES:
            cols = {}
            for sym in symbols:
                seq = bus.series(name, sym, token)
                if not seq:
                    continue
                ts = [t for t, _ in seq]
                vals = [v for _, v in seq]
                if window is not None:
                    ts, vals = ts[-window:], vals[-window:]
                cols[sym] = pd.Series(vals, index=ts, dtype=float)
            series[name] = pd.DataFrame(cols).sort_index() if cols else pd.DataFrame()
        return cls(**series)

    # ---- 基本属性 ----

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def timestamps(self) -> list[int]:
        return list(self.close.index)

    @property
    def n_times(self) -> int:
        return len(self.close.index)

    def returns(self) -> pd.DataFrame:
        """日简单收益率面板。"""
        c = self.close
        return c / c.shift(1) - 1.0

    def adv(self, n: int = 20) -> pd.DataFrame:
        """adv{n}：n 日平均成交量。"""
        return self.volume.rolling(n, min_periods=max(2, n // 2)).mean()

    # ---- 因果切片 ----

    def slice_upto(self, ts: int) -> "Panel":
        """截断到 ``timestamp <= ts``（因果审计用）。"""
        return self._map(lambda d: d.loc[d.index <= ts])

    def slice_window(self, ts: int, n: int) -> "Panel":
        """取 ``timestamp <= ts`` 的最近 n 期。"""
        return self._map(lambda d: d.loc[d.index <= ts].tail(n))

    def _map(self, fn) -> "Panel":
        return Panel(**{f.name: fn(getattr(self, f.name))
                        for f in fields(self) if isinstance(getattr(self, f.name), pd.DataFrame)})

    def last_row(self) -> dict[str, pd.Series]:
        """取每个序列的最后一期（截面）。"""
        return {f.name: getattr(self, f.name).iloc[-1]
                for f in fields(self) if isinstance(getattr(self, f.name), pd.DataFrame)
                and not getattr(self, f.name).empty}

    def __repr__(self) -> str:  # pragma: no cover
        return (f"Panel(times={self.n_times}, symbols={len(self.symbols)}, "
                f"span={self.timestamps[0] if self.n_times else '-'}"
                f"..{self.timestamps[-1] if self.n_times else '-'})")
