"""opentdx 数仓 Parquet 数据源（数据接入）。

读取 ``opentdx/db/data`` 下的 Parquet K 线（``fakt_kline/{period}/{adjust}/*.parquet``），
转换为引擎可消费的 ``BarFeed``。这是本框架默认的数据源。

两个关键的数据正确性处理
------------------------

**1. 按 ``period`` / ``adjust`` 过滤。**
数仓内同一标的可能同时存在不复权/前复权/后复权多个文件（路径带 ``{period}/{adjust}``，
行内也有权威的 ``period``/``adjust`` 列）。若不过滤，会把不同复权口径的 Bar 混成
一条序列，静默污染回测。本实现以**行内列**为准过滤，并按 ``datetime`` 去重保最后。

**2. vwap / 市值的口径。**
``open/high/low/close`` 是**复权后**价格；``amount``/``vol`` 是**未复权**原始
成交值，故 ``amount/vol`` 就是真实成交均价（原始 vwap）。本数据源统一采用：

    vwap       = amount / vol                   # 真实 vwap（全市场一致）
    market_cap = close × float_shares           # 与复权 close 同口径

不额外解复权因子把 vwap 拉到复权价同口径：原始 twin 仅占全市场约 0.5%，
按因子兜底会让 twin/非 twin 的 vwap/mcap 口径分裂，横截面因子混入「数据可用性」
信号；原始 vwap 已是真实成交均价，且与 ``Panel.from_feed`` 的兜底同源。

依赖：pandas + pyarrow（可选依赖 ``data``）。
"""

from __future__ import annotations

import os
from pathlib import Path

from .base import DataSource
from ..core.events import Bar

# period / adjust 名称 → 数仓内的整型编码（与 opentdx.const 及 db/config.py 对齐）
PERIOD_CODE = {
    "MIN_1": 7, "MIN_5": 0, "MIN_15": 1, "MIN_30": 2, "MIN_60": 3,
    "DAILY": 4, "WEEKLY": 5, "MONTHLY": 6,
    "MINS": 8, "DAYS": 9, "QUARTERLY": 10, "YEARLY": 11, "SECONDS": 13,
}
ADJUST_CODE = {"NONE": 0, "QFQ": 1, "HFQ": 2}

DEFAULT_PERIOD = "DAILY"
DEFAULT_ADJUST = "QFQ"   # 前复权是回测首选


def default_data_dir() -> Path:
    """默认数据目录：opentdx 数仓的 data/。"""
    env = os.environ.get("FRAMEBT_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    project_root = here.parents[3]  # .../TRADE/framebt
    return (project_root.parent / "opentdx" / "db" / "data").resolve()


def _period_code(period) -> int:
    if isinstance(period, int):
        return period
    p = str(period).upper()
    if p not in PERIOD_CODE:
        raise ValueError(f"未知 period {period!r}；可选 {sorted(PERIOD_CODE)}")
    return PERIOD_CODE[p]


def _adjust_code(adjust) -> int:
    if isinstance(adjust, int):
        return adjust
    a = str(adjust).upper()
    if a not in ADJUST_CODE:
        raise ValueError(f"未知 adjust {adjust!r}；可选 {sorted(ADJUST_CODE)}")
    return ADJUST_CODE[a]


class OpentdxParquetSource(DataSource):
    """从 opentdx 数仓 Parquet 读取 K 线。

    用法：
        src = OpentdxParquetSource()                      # 默认 DAILY + QFQ
        src = OpentdxParquetSource(adjust="NONE")         # 不复权
        feed = src.load("SH.600519")
        feed = src.universe(["SH.600519", "SZ.000001"])
        src.available_adjusts("SH.600519")                # 该标的可用的复权口径
    """

    name = "opentdx_parquet"

    # 文件名前缀的市场码 → 市场前缀（opentdx 数仓约定：0=SZ / 1=SH / 2=BJ）。
    # 文件名形如 ``{market_code}_{code>.parquet``，例如 ``1_600519.parquet`` → ``SH.600519``。
    _MARKET = {0: "SZ", 1: "SH", 2: "BJ"}

    def __init__(self, data_dir=None, period=DEFAULT_PERIOD, adjust=DEFAULT_ADJUST):
        self.data_dir = Path(data_dir).expanduser().resolve() if data_dir else default_data_dir()
        if not self.data_dir.is_dir():
            raise FileNotFoundError(
                f"数据目录不存在：{self.data_dir}。"
                "可通过 FRAMEBT_DATA_DIR 环境变量或 data_dir= 参数指定 opentdx 数仓的 data/ 目录。"
            )
        self._kline_root = self.data_dir / "fakt_kline"
        self.period = _period_code(period)
        self.adjust = _adjust_code(adjust)
        # 文件索引：adjust 子目录路径(str) -> {symbol_key -> Path}。
        # 每个 parquet 文件只含单个标的、单一 (period, adjust) 口径，故可纯靠文件名
        # 解析出 symbol_key（无需读 pandas），实现「按标的按需读取」而非扫全库。
        self._file_index: dict[str, dict[str, Path]] = {}
        # Bar 缓存：(period, adjust, symbol) -> list[Bar]。
        self._bar_cache: dict[tuple[int, int, str], list[Bar]] = {}

    def _adjust_dir(self, period: int, adjust: int) -> Path:
        """``fakt_kline/{period}/{adjust}`` 子目录（每个 parquet 落在其中）。"""
        return self._kline_root / str(period) / str(adjust)

    def _build_index(self, period: int, adjust: int) -> dict[str, Path]:
        """解析文件名建 ``symbol_key -> 文件路径`` 索引（纯字符串，不读 pandas）。

        索引按 adjust 子目录缓存；首次构建后秒回。文件名 ``{market_code}_{code>.parquet``
        中的市场码与代码即可唯一还原 ``symbol_key``（见 ``_MARKET``）。
        """
        adj_dir = self._adjust_dir(period, adjust)
        idx = self._file_index.get(str(adj_dir))
        if idx is not None:
            return idx
        idx = {}
        if adj_dir.is_dir():
            for name in adj_dir.glob("*.parquet"):
                stem = name.stem  # 如 "1_600519"
                if "_" not in stem:
                    continue
                lead, code = stem.split("_", 1)
                try:
                    market = self._MARKET[int(lead)]
                except ValueError:
                    continue
                idx[f"{market}.{code}"] = name
        self._file_index[str(adj_dir)] = idx
        return idx

    @property
    def all_files(self) -> list[Path]:
        """所有 parquet 文件路径（兼容旧代码对 ``src._files`` 的引用）。"""
        files: list[Path] = []
        for idx in self._file_index.values():
            files.extend(idx.values())
        return files

    # ---- 单文件读取 ----

    def _read_df(self, path: Path) -> "object":
        """读取单个 parquet 文件，按 ``datetime`` 去重保最后、升序排列（未转 Bar）。

        每个文件只含单一标的、单一 ``(period, adjust)`` 口径（已由文件名/目录保证），
        故无需按 symbol 分组、无需再按列过滤口径——直接读整文件即可。
        """
        import pandas as pd

        df = pd.read_parquet(path)
        if df.empty or "datetime" not in df.columns:
            return df
        merged = df.drop_duplicates(subset=["datetime"], keep="last")
        return merged.sort_values("datetime").reset_index(drop=True)

    def _to_bars(self, df) -> list[Bar]:
        """把单文件的行转成 Bar。

        口径全市场统一（**不依赖原始 twin**）：
          - ``vwap = amount / vol``：原始成交均价（真实 vwap），与 ``Panel.from_feed``
            的兜底同源，且无需 raw 数据即可算；
          - ``market_cap = close × float_shares``：与（复权）close 同口径。

        不再解复权因子把 vwap 拉到复权价同口径。原始 twin 仅占全市场约 0.5%
        （QFQ-only 占 99.5%），按因子兜底会让 twin/非 twin 的 vwap/mcap 口径分裂，
        横截面因子混入「数据可用性」信号；原始 vwap 已是真实成交均价，全市场一致更优。
        """
        import pandas as pd

        bars: list[Bar] = []
        for _, row in df.iterrows():
            try:
                ordinal = int(pd.Timestamp(row["dt"]).toordinal())
            except Exception:  # noqa: BLE001
                continue
            close = float(row["close"])
            vol = float(row.get("vol", 0.0) or 0.0)
            amount = float(row.get("amount", 0.0) or 0.0)
            fsh = float(row.get("float_shares", 0.0) or 0.0)
            turn = float(row.get("turnover", 0.0) or 0.0)

            vwap = (amount / vol) if vol > 0 else close
            market_cap = close * fsh if fsh > 0 else 0.0

            bars.append(
                Bar(
                    timestamp=ordinal,
                    symbol=str(row["symbol_key"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=close,
                    volume=vol,
                    amount=amount,
                    float_shares=fsh,
                    turnover=turn,
                    vwap=float(vwap),
                    market_cap=float(market_cap),
                )
            )
        return bars

    def _bars_for(self, symbol: str, period: int, adjust: int) -> list[Bar]:
        """读取单个标的的 Bar（按 datetime 去重），并按 (period, adjust, symbol) 缓存。

        只读该标的**一个**文件（目标口径）——vwap/mcap 口径已全市场统一，不再需要
        原始 twin。缺标的返回空列表（由 ``load`` 抛 ``KeyError``）。
        """
        cached = self._bar_cache.get((period, adjust, symbol))
        if cached is not None:
            return cached

        idx = self._build_index(period, adjust)
        path = idx.get(symbol)
        if path is None:
            bars: list[Bar] = []
        else:
            bars = self._to_bars(self._read_df(path))

        self._bar_cache[(period, adjust, symbol)] = bars
        return bars

    def available_symbols(self, period=None, adjust=None) -> list[str]:
        """返回该口径下所有标的（纯解析文件名，不读任何 parquet）。"""
        p = period if period is not None else self.period
        a = adjust if adjust is not None else self.adjust
        if not isinstance(p, int):
            p = _period_code(p)
        if not isinstance(a, int):
            a = _adjust_code(a)
        return sorted(self._build_index(p, a).keys())

    def available_adjusts(self, symbol: str) -> list[str]:
        """返回该标的在数仓中实际存在的复权口径（纯解析文件名，不读 parquet）。"""
        found: set[int] = set()
        for code in ADJUST_CODE.values():
            if symbol in self._build_index(self.period, code):
                found.add(code)
        inv = {v: k for k, v in ADJUST_CODE.items()}
        return sorted(inv.get(a, str(a)) for a in found)

    def load(self, symbol, start=None, end=None, period=None, adjust=None, **params) -> "BarFeed":
        from ..feed import BarFeed

        p = period if period is not None else self.period
        a = adjust if adjust is not None else self.adjust
        if not isinstance(p, int) or not isinstance(a, int):
            p, a = _period_code(p), _adjust_code(a)

        if symbol not in self._build_index(p, a):
            raise KeyError(
                f"数据源中无标的 {symbol}（period={p}, adjust={a}）。"
                f"可用标的：{sorted(self._build_index(p, a))}"
            )
        bars = self._bars_for(symbol, p, a)
        return BarFeed(self._filter(bars, start, end))

    @staticmethod
    def _filter(bars, start, end):
        import pandas as pd

        if start is not None:
            start_ord = pd.Timestamp(start).toordinal() if not isinstance(start, int) else start
            bars = [b for b in bars if b.timestamp >= start_ord]
        if end is not None:
            end_ord = pd.Timestamp(end).toordinal() if not isinstance(end, int) else end
            bars = [b for b in bars if b.timestamp <= end_ord]
        return bars
