"""全市场日线回填：把 CATEGORY.A 的全部 5580 只 A 股的历史日线拉进 opentdx 数仓。

设计目标（对应 docs/PATH_200K.md 阶段 0）
----------------------------------------
- 广度：全 A（SZ/SH/BJ），为横截面因子研究和「按调仓日成交额取前 N」的股票池提供数据；
- 深度：每票拉到历史尽头（实测 6000+ 根 ≈ 25 年），不是 2000 根；
- 幂等 + 断点续跑：已填满（bars 数 ≥ 阈值）的票跳过，可中断后重跑；
- 口径：默认 QFQ（回测首选）。--none 则写不复权。

复用 opentdx 自有的 db/persist.py::persist_kline（schema / merge 语义与数仓完全一致）。
不修改 opentdx 任何源码。

用法
----
    python tools/backfill_universe.py                 # 全 A，QFQ，每票拉到历史尽头
    python tools/backfill_universe.py --max-bars 4000 # 每票上限 4000 根
    python tools/backfill_universe.py --none           # 写不复权口径
    python tools/backfill_universe.py --resume         # 跳过已填满的票（默认开启）
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRADE_ROOT = _HERE.parents[2]
_OPENTDX = _TRADE_ROOT / "opentdx"
_OPENTDX_DB = _OPENTDX / "db"
for _p in (str(_OPENTDX), str(_OPENTDX_DB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opentdx import TdxClient                                   # noqa: E402
from opentdx.const import (ADJUST, CATEGORY, MARKET, PERIOD,    # noqa: E402
                           SORT_ORDER, SORT_TYPE)
import persist as tdx_persist                                  # noqa: E402

PAGE = 700
DEFAULT_MAX_BARS = 6000       # 实测 6000+ 根 ≈ 25 年，覆盖多段市场周期
RESUME_MIN_BARS = 1500        # 已填满 ≥ 此数视为已完成，跳过


def _bar_count(market, code, adjust) -> int:
    """已落库的根数（用于断点续跑）。"""
    rel = f"fakt_kline/{int(PERIOD.DAILY.value)}/{int(adjust.value)}/{market.value}_{code}.parquet"
    data_dir = _OPENTDX_DB / "data"
    path = Path(rel) if Path(rel).is_absolute() else data_dir / rel
    if not path.exists():
        return 0
    try:
        import pandas as pd
        df = pd.read_parquet(path)
        return len(df)
    except Exception:  # noqa: BLE001
        return 0


def fetch_paged(client, market, code, max_bars, adjust) -> list[dict]:
    """分页拉取日线（旧→新），直到耗尽或达到 max_bars。"""
    collected: dict = {}
    start = 0
    while len(collected) < max_bars:
        want = min(PAGE, max_bars - len(collected))
        bars = client.stock_kline(market, code, PERIOD.DAILY,
                                  start=start, count=want, adjust=adjust)
        if not bars:
            break
        before = len(collected)
        for b in bars:
            collected[b["datetime"]] = b
        if len(collected) == before:
            break
        start += len(bars)
    return sorted(collected.values(), key=lambda b: b["datetime"])


def main() -> None:
    ap = argparse.ArgumentParser(description="全市场日线回填（全 A，QFQ）")
    ap.add_argument("--max-bars", type=int, default=DEFAULT_MAX_BARS,
                    help="每标的最多 K 线数（默认 6000 ≈ 25 年）")
    ap.add_argument("--none", action="store_true", help="写不复权口径（默认 QFQ）")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="跳过已填满的票（默认开启）")
    args = ap.parse_args()

    adjust = ADJUST.NONE if args.none else ADJUST.QFQ
    print(f"全 A 日线回填，口径={adjust.name}，每票上限 {args.max_bars} 根")
    print(f"目标数仓: {_OPENTDX_DB / 'data'}\n")

    with TdxClient() as client:
        print("正在获取全 A 代码表（CATEGORY.A）...")
        t0 = time.time()
        rows = client.stock_board_members(CATEGORY.A, count=100000,
                                          sort_type=SORT_TYPE.CODE,
                                          sort_order=SORT_ORDER.ASC)
        print(f"  取到 {len(rows)} 只，耗时 {time.time() - t0:.1f}s")
        universe = [(r["market"], str(r["code"])) for r in rows]

    ok, skipped, failed = [], [], []
    t0 = time.time()
    with TdxClient() as client:
        for i, (mkt_int, code) in enumerate(universe, 1):
            mkt = mkt_int if isinstance(mkt_int, MARKET) else MARKET(int(mkt_int))
            mkt_name = {"0": "SZ", "1": "SH", "2": "BJ"}.get(
                str(mkt.value), str(mkt_int))

            if args.resume:
                have = _bar_count(mkt, code, adjust)
                if have >= RESUME_MIN_BARS:
                    skipped.append((mkt_name, code, have))
                    continue

            try:
                bars = fetch_paged(client, mkt, code, args.max_bars, adjust)
            except Exception as e:  # noqa: BLE001
                print(f"[{i}/{len(universe)}] {mkt_name} {code} 拉取失败: {type(e).__name__}: {e}")
                failed.append((mkt_name, code))
                continue
            if not bars:
                failed.append((mkt_name, code))
                continue

            n = tdx_persist.persist_kline(mkt, code, PERIOD.DAILY, adjust, bars)
            span = f"{bars[0]['datetime'].date()} .. {bars[-1]['datetime'].date()}"
            if i % 200 == 0 or i == len(universe):
                done = i - len(skipped) - len(failed)
                print(f"[{i:4d}/{len(universe)}] 完成 {done}/{len(universe)} "
                      f"(跳过 {len(skipped)}, 失败 {len(failed)}) "
                      f"耗时 {time.time() - t0:.0f}s")
            ok.append((mkt_name, code, n, span))

    print(f"\n完成：成功 {len(ok)}，跳过(已填满) {len(skipped)}，失败 {len(failed)}，"
          f"总耗时 {time.time() - t0:.0f}s")
    if failed:
        print("失败清单:", ", ".join(f"{m} {c}" for m, c in failed[:50]))


if __name__ == "__main__":
    main()
