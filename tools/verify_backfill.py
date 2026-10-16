"""回填结果校验：文件数、每票根数、日期跨度、市场分布。"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRADE_ROOT = _HERE.parents[2]
_OPENTDX = _TRADE_ROOT / "opentdx"
for _p in (str(_OPENTDX), str(_OPENTDX / "db")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pandas as pd                                             # noqa: E402

ROOT = _OPENTDX / "db" / "data" / "fakt_kline" / "4" / "1"      # DAILY + QFQ


def main() -> None:
    files = sorted(ROOT.glob("*.parquet"))
    print(f"日线 QFQ 文件数: {len(files)}")
    counts = []
    markets = {"SZ": 0, "SH": 0, "BJ": 0}
    first_dates, last_dates = [], []
    for path in files:
        try:
            df = pd.read_parquet(path)
        except Exception:
            print(f"  ✗ 损坏: {path.name}")
            continue
        m = path.stem.split("_")[0]
        if m in markets:
            markets[m] += 1
        counts.append(len(df))
        dts = pd.to_datetime(df["datetime"])
        first_dates.append(dts.min().date())
        last_dates.append(dts.max().date())

    counts = [c for c in counts if c > 0]
    print(f"\n每票根数: 最小 {min(counts)}  中位 {int(sorted(counts)[len(counts)//2])}  "
          f"最大 {max(counts)}  合计 {sum(counts):,} 根")
    print(f"市场分布: {markets}  总计 {sum(markets.values())}")
    print(f"最早上市: {min(first_dates)}  最新交易日: {max(last_dates)}")
    print(f"日期跨度: {(max(last_dates) - min(first_dates)).days / 365.25:.1f} 年")

    # 抽样几票看覆盖
    print("\n  抽样覆盖：")
    for name in ["SH600519", "SH000300".replace("000300", "600519"), "SH600036",
                 "SH601318", "SH510300", "SH000001".replace("000001", "600036")]:
        pass
    for name in ["SH600519", "SH600036", "SH601318", "SZ000001"]:
        p = ROOT / f"{name}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            dts = pd.to_datetime(df["datetime"])
            print(f"    {name}: {len(df)} 根 {dts.min().date()} .. {dts.max().date()}")


if __name__ == "__main__":
    main()
