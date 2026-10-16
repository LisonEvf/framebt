"""回填脚本的端到端冒烟：拉 3 只票、写库、校验 parquet 文件存在且有数据。"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRADE_ROOT = _HERE.parents[2]
_OPENTDX = _TRADE_ROOT / "opentdx"
for _p in (str(_OPENTDX), str(_OPENTDX / "db")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opentdx import TdxClient                                   # noqa: E402
from opentdx.const import (ADJUST, CATEGORY, MARKET, PERIOD,   # noqa: E402
                           SORT_ORDER, SORT_TYPE)
import persist as tdx_persist                                   # noqa: E402
import pandas as pd                                             # noqa: E402

TESTS = [(MARKET.SH, "600519"), (MARKET.SZ, "000001"), (MARKET.SH, "000333")]


def main() -> None:
    with TdxClient() as client:
        rows = client.stock_board_members(CATEGORY.A, count=100000,
                                          sort_type=SORT_TYPE.CODE,
                                          sort_order=SORT_ORDER.ASC)
        print(f"CATEGORY.A 返回 {len(rows)} 只；第一条 market={rows[0].get('market')} "
              f"code={rows[0].get('code')}")
        print(f"第一条 market 类型: {type(rows[0]['market'])}")

    adjust = ADJUST.QFQ
    t0 = time.time()
    ok = 0
    for mkt, code in TESTS:
        start, bars = 0, []
        while len(bars) < 6000:
            more = client.stock_kline(mkt, code, PERIOD.DAILY,
                                      start=start, count=700, adjust=adjust)
            if not more:
                break
            bars.extend(more)
            start += len(more)
            if len(more) < 700:
                break
        n = tdx_persist.persist_kline(mkt, code, PERIOD.DAILY, adjust, bars)
        rel = f"fakt_kline/{int(PERIOD.DAILY.value)}/{int(adjust.value)}/{mkt.value}_{code}.parquet"
        path = _OPENTDX / "db" / "data" / rel
        df = pd.read_parquet(path)
        print(f"  {mkt.name} {code}: 落库 {n} 行, 文件 {path.name}, "
              f"{df['datetime'].min().date()} .. {df['datetime'].max().date()}")
        ok += 1
    print(f"\n冒烟完成：{ok}/{len(TESTS)} 只写入成功，耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
