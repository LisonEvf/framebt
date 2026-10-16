"""数据可达性探针：为「20 万本金的可行路径」回答五个必须先知道的事实。

1. 全 A 代码表能不能一次拿到（决定能不能做全市场横截面，而不是 24 只票）；
2. 日线历史能拉多深（决定样本外能有多长）；
3. **退市股能不能拉到**（决定存活偏差能不能消除 —— 这是全部结论的前提）；
4. 分页拉取的**真实速度**（决定 5400 只票回填要几小时还是几天）；
5. 日内数据的可得性边界。

只读，不写数仓。
"""

from __future__ import annotations

import sys
import time
from datetime import date, timedelta
from pathlib import Path

_HERE = Path(__file__).resolve()
_TRADE_ROOT = _HERE.parents[2]
_OPENTDX = _TRADE_ROOT / "opentdx"
for _p in (str(_OPENTDX), str(_OPENTDX / "db")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opentdx import TdxClient                                   # noqa: E402
from opentdx.const import (ADJUST, CATEGORY, MARKET, PERIOD,    # noqa: E402
                           SORT_ORDER, SORT_TYPE)


def hr(t: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {t}")
    print("=" * 72)


def main() -> None:
    with TdxClient() as client:
        # ---- 1. 全 A 代码表 ----
        hr("1. 全 A 代码表（CATEGORY.A）")
        t0 = time.time()
        try:
            rows = client.stock_board_members(CATEGORY.A, count=100000,
                                              sort_type=SORT_TYPE.CODE,
                                              sort_order=SORT_ORDER.ASC)
        except Exception as e:  # noqa: BLE001
            rows = []
            print(f"  取全 A 失败: {type(e).__name__}: {e}")
        dt = time.time() - t0
        print(f"  返回 {len(rows)} 行，耗时 {dt:.1f}s")
        if rows:
            print(f"  字段: {sorted(rows[0].keys())}")
            s = rows[0]
            print(f"  样例: code={s.get('code')} market={s.get('market')} "
                  f"name={s.get('name')} price={s.get('price')}")
            mkts = {}
            for r in rows:
                mkts[r.get("market")] = mkts.get(r.get("market"), 0) + 1
            print(f"  市场分布: {mkts}")
            codes = {str(r.get("code")) for r in rows}
        else:
            codes = set()

        # ---- 2. 单票历史深度 ----
        hr("2. 日线历史深度（贵州茅台，向前翻页直到耗尽）")
        total, start, PAGE = 0, 0, 700
        first, last = None, None
        t0 = time.time()
        pages = 0
        while pages < 20:
            bars = client.stock_kline(MARKET.SH, "600519", PERIOD.DAILY,
                                      start=start, count=PAGE, adjust=ADJUST.QFQ)
            pages += 1
            if not bars:
                break
            total += len(bars)
            if first is None:
                first = bars[0]["datetime"]
            last = bars[-1]["datetime"]
            if len(bars) < PAGE:
                break
            start += len(bars)
        dt = time.time() - t0
        if first:
            print(f"  拉到 {total} 根，{first.date()} .. {last.date()}")
            print(f"  {pages} 页，耗时 {dt:.1f}s（{dt / max(pages, 1):.2f}s/页）")
            print(f"  → 单票全量历史约 {total} 根 ≈ {total / 244:.1f} 年")
        else:
            print("  未取到数据")

        # ---- 3. 退市股（存活偏差的命门） ----
        hr("3. 退市 / 老股能否取到（存活偏差）")
        probes = [
            (MARKET.SZ, "000003", "PT金田A（已退市）"),
            (MARKET.SH, "600001", "邯郸钢铁（已退市）"),
            (MARKET.SH, "600087", "退市长油（已退市）"),
            (MARKET.SZ, "000033", "新都退（已退市）"),
            (MARKET.SH, "600519", "贵州茅台（对照，在市）"),
        ]
        for mkt, code, note in probes:
            try:
                bars = client.stock_kline(mkt, code, PERIOD.DAILY,
                                          start=0, count=5, adjust=ADJUST.NONE)
                if bars:
                    print(f"  {mkt.name} {code} {note}: {len(bars)} 根，"
                          f"最新 {bars[-1]['datetime'].date()} "
                          f"收 {bars[-1].get('close')}")
                else:
                    print(f"  {mkt.name} {code} {note}: 无数据")
            except Exception as e:  # noqa: BLE001
                print(f"  {mkt.name} {code} {note}: 异常 {type(e).__name__}: {e}")

        # ---- 4. 分页回填速度实测 ----
        hr("4. 分页拉取速度（决定 5400 票回填要多久）")
        for code in ["000001", "000725"]:
            t0 = time.time()
            n = 0
            st = 0
            while st < 2100:
                bars = client.stock_kline(MARKET.SZ, code, PERIOD.DAILY,
                                          start=st, count=700, adjust=ADJUST.QFQ)
                if not bars:
                    break
                n += len(bars)
                st += len(bars)
                if len(bars) < 700:
                    break
            dt = time.time() - t0
            print(f"  SZ {code}: {n} 根 / {dt:.2f}s  → {n / max(dt, 1e-9):.0f} 根/秒")
        if codes:
            est = 5400 * 3 * 1.0  # 粗估：5400 票 × 3 页
            print(f"  粗估：全 A 约 5400 票 × 3 页 ≈ {est:.0f} 次请求")

        # ---- 5. 日内数据边界 ----
        hr("5. 日内数据可得性")
        today = date(2026, 10, 15)
        for back in (1, 5, 30, 120):
            d = today - timedelta(days=back)
            try:
                tr = client.stock_transaction(MARKET.SH, "600519", query_date=d, count=50)
                ic = client.stock_tick_chart(MARKET.SH, "600519", query_date=d)
                print(f"  {d}: 逐笔 {len(tr or [])} 条, 分时 {len(ic or [])} 点")
            except Exception as e:  # noqa: BLE001
                print(f"  {d}: 异常 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
