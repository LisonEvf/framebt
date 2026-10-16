"""指数与基准可达性探针：确认能否取到沪深300等基准（做「20万机会成本」的对照）。"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRADE_ROOT = _HERE.parents[2]
_OPENTDX = _TRADE_ROOT / "opentdx"
for _p in (str(_OPENTDX), str(_OPENTDX / "db")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opentdx import TdxClient                                    # noqa: E402
from opentdx.const import ADJUST, BOARD_TYPE, CATEGORY, MARKET, PERIOD  # noqa: E402

CANDIDATES = [
    (MARKET.SH, "000300", "沪深300"),
    (MARKET.SH, "000905", "中证500"),
    (MARKET.SH, "000852", "中证1000"),
    (MARKET.SH, "000001", "上证指数"),
    (MARKET.SZ, "399001", "深证成指"),
    (MARKET.SZ, "399006", "创业板指"),
    (MARKET.SH, "000016", "上证50"),
    (MARKET.SH, "000688", "科创50"),
    (MARKET.SH, "510300", "沪深300ETF(华泰柏瑞)"),
    (MARKET.SH, "510500", "中证500ETF"),
    (MARKET.SH, "512100", "中证1000ETF"),
    (MARKET.SH, "588000", "科创50ETF"),
]


def main() -> None:
    with TdxClient() as client:
        print("=" * 72)
        print("  1. 指数 / ETF 可达性")
        print("=" * 72)
        for mkt, code, note in CANDIDATES:
            try:
                b = client.stock_kline(mkt, code, PERIOD.DAILY,
                                       start=0, count=3, adjust=ADJUST.NONE)
                if b:
                    print(f"  ✓ {mkt.name} {code} {note}: 最新 {b[-1]['datetime'].date()} "
                          f"close={b[-1].get('close')}")
                else:
                    print(f"  ✗ {mkt.name} {code} {note}: 无数据")
            except Exception as e:  # noqa: BLE001
                print(f"  ! {mkt.name} {code} {note}: {type(e).__name__}: {e}")

        print()
        print("=" * 72)
        print("  2. 指数历史深度（沪深300）")
        print("=" * 72)
        for mkt, code, note in [(MARKET.SH, "000300", "沪深300"),
                                (MARKET.SH, "000001", "上证指数")]:
            collected = {}
            start = 0
            pages = 0
            while pages < 12:
                b = client.stock_kline(mkt, code, PERIOD.DAILY,
                                       start=start, count=700, adjust=ADJUST.NONE)
                pages += 1
                if not b:
                    break
                for x in b:
                    collected[x["datetime"]] = x
                start += len(b)
                if len(b) < 700:
                    break
            srt = sorted(collected.values(), key=lambda x: x["datetime"])
            if srt:
                print(f"  {note}: {len(srt)} 根  {srt[0]['datetime'].date()} .. "
                      f"{srt[-1]['datetime'].date()}")

        print()
        print("=" * 72)
        print("  3. 指数成分股板块能不能取（用于构造可投资universe）")
        print("=" * 72)
        try:
            rows = client.stock_board_members(CATEGORY.ZS, count=60)
            print(f"  CATEGORY.ZS 返回 {len(rows)} 行")
            for r in rows[:12]:
                print(f"    {r.get('code')} {r.get('name')}")
        except Exception as e:  # noqa: BLE001
            print(f"  CATEGORY.ZS 失败: {type(e).__name__}: {e}")
        try:
            bl = client.stock_board_list(BOARD_TYPE.ALL, count=10000)
            print(f"  板块总数: {len(bl or [])}")
            idx = [r for r in (bl or []) if "指数" in str(r.get("name", ""))]
            print(f"  名称含“指数”的板块 {len(idx)} 个，样例：")
            for r in idx[:10]:
                print(f"    {r.get('code')} {r.get('name')} type={r.get('market')}")
        except Exception as e:  # noqa: BLE001
            print(f"  stock_board_list 失败: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
