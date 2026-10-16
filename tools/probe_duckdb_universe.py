"""测试 duckdb 一次性查询：从全 fakt_kline parquet 计算前 N 成交额并集。"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRADE_ROOT = _HERE.parents[2]
_OPENTDX = _TRADE_ROOT / "opentdx"
for _p in (str(_OPENTDX), str(_OPENTDX / "db")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ROOT = _OPENTDX / "db" / "data" / "fakt_kline" / "4" / "1"


def main(top_n: int = 500) -> None:
    import duckdb
    glob = str((ROOT / "*.parquet").as_posix())
    conn = duckdb.connect()
    q = f"""
        WITH all_data AS (
            SELECT symbol_key, CAST(datetime AS DATE) AS d, CAST(amount AS DOUBLE) AS amount
            FROM read_parquet('{glob}')
            WHERE amount IS NOT NULL AND amount > 0
        ),
        avg_amt AS (
            SELECT symbol_key, AVG(amount) AS avg_amount, COUNT(*) AS n_days
            FROM all_data
            GROUP BY symbol_key
        )
        SELECT symbol_key FROM avg_amt
        ORDER BY avg_amount DESC
        LIMIT {top_n}
    """
    t0 = __import__("time").time()
    rows = conn.execute(q).fetchall()
    syms = [r[0] for r in rows]
    print(f"前 {top_n} 日均成交额 = {len(syms)} 只  ({time.time() - t0:.1f}s)")
    print("  抽样:", syms[:10])
    print("  含 SH.600519:", "SH.600519" in syms, " 含 SZ.000001:", "SZ.000001" in syms)


if __name__ == "__main__":
    import time
    main()
