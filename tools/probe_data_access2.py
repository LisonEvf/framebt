"""数据可达性探针 · 第二轮：把第一轮的两个关键发现钉死。

发现 A：日线历史能拉到 2003 年（约 6000 根 ≈ 24.7 年），远超文档假设的 2000 根/8 年。
发现 B：CATEGORY.A 一次返回 5580 只（SZ 2907 / SH 2320 / BJ 353）。

本轮要钉死的三件事：
1. 分页返回顺序与去重 —— 第一轮 first..last 出现 (2023-11-22..2003-06-11) 的可疑区间，
   要确认是"页内降序"还是"数据错乱"，并给出排序去重后的真实跨度与连续性；
2. 退市股到底能不能取到（用全 A 列表里没有、但历史上存在过的代码，逐个试）；
3. 日内历史数据的真实深度（逐笔 / 分时分别能回溯几天）。
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


def fetch_all(client, mkt, code, adjust=ADJUST.QFQ, max_pages=30):
    """逐页拉取所有历史，返回 (排序去重后的 bars, 原始页数, 页内首尾顺序样本)。"""
    collected: dict = {}
    order_sample = []
    page = 0
    start = 0
    while page < max_pages:
        bars = client.stock_kline(mkt, code, PERIOD.DAILY,
                                  start=start, count=700, adjust=adjust)
        page += 1
        if not bars:
            break
        if page <= 2:
            order_sample.append((bars[0]["datetime"].date(), bars[-1]["datetime"].date()))
        for b in bars:
            collected[b["datetime"]] = b
        start += len(bars)
        if len(bars) < 700:
            break
    srt = sorted(collected.values(), key=lambda b: b["datetime"])
    return srt, page, order_sample


def main() -> None:
    with TdxClient() as client:
        # ---- 1. 分页顺序与连续性 ----
        hr("1. 分页返回顺序 / 去重后真实跨度（SH 600519）")
        bars, pages, sample = fetch_all(client, MARKET.SH, "600519")
        print(f"  页数 {pages}，去重后 {len(bars)} 根")
        print(f"  前两页的 (首,尾) 顺序样本: {sample}")
        print(f"  排序后区间: {bars[0]['datetime'].date()} .. {bars[-1]['datetime'].date()}")
        # 连续性：相邻交易日的间隔分布（找有没有大洞）
        gaps = [(bars[i + 1]["datetime"] - bars[i]["datetime"]).days
                for i in range(len(bars) - 1)]
        big = [(bars[i]["datetime"].date(), bars[i + 1]["datetime"].date(), g)
               for i, g in enumerate(gaps) if g > 30]
        print(f"  相邻间隔 >30 天的断点: {len(big)} 个")
        for a, b, g in big[:10]:
            print(f"    {a} → {b}  ({g} 天)")
        yrs = (bars[-1]["datetime"] - bars[0]["datetime"]).days / 365.25
        print(f"  跨度 {yrs:.1f} 年，平均 {len(bars) / yrs:.0f} 根/年（A 股约 242 根/年）")

        # ---- 2. 复权口径自检 ----
        hr("2. 复权口径自检（NONE vs QFQ 最新收盘）")
        for adj in (ADJUST.NONE, ADJUST.QFQ, ADJUST.HFQ):
            b = client.stock_kline(MARKET.SH, "600519", PERIOD.DAILY,
                                   start=0, count=2, adjust=adj)
            if b:
                print(f"  {adj.name:5s}: 最新 {b[-1]['datetime'].date()} "
                      f"close={b[-1].get('close')}  vol={b[-1].get('vol')}")

        # ---- 3. 退市股（存活偏差命门）----
        hr("3. 退市股 / 老代码可达性（存活偏差）")
        # 用全 A 现役代码表做差集，再从"历史上有过、现在没有"的代码里试
        rows = client.stock_board_members(CATEGORY.A, count=100000,
                                          sort_type=SORT_TYPE.CODE,
                                          sort_order=SORT_ORDER.ASC)
        live = {str(r.get("code")) for r in rows}
        print(f"  现役全 A {len(live)} 只；下面逐个试“不在现役表里”的历史代码：")
        probes = [
            (MARKET.SZ, "000003", "PT金田A"),
            (MARKET.SZ, "000013", "*ST石化A"),
            (MARKET.SZ, "000033", "新都退"),
            (MARKET.SZ, "000406", "石油大明"),
            (MARKET.SH, "600001", "邯郸钢铁"),
            (MARKET.SH, "600087", "退市长油"),
            (MARKET.SH, "600002", "齐鲁石化"),
            (MARKET.SH, "600625", "PT水仙"),
            (MARKET.SZ, "000024", "招商地产"),
            (MARKET.SZ, "000562", "宏源证券"),
        ]
        got = []
        for mkt, code, note in probes:
            flag = "现役" if code in live else "非现役"
            try:
                b = client.stock_kline(mkt, code, PERIOD.DAILY,
                                       start=0, count=3, adjust=ADJUST.NONE)
                if b:
                    got.append((mkt.name, code))
                    print(f"  [{flag}] {mkt.name} {code} {note}: ✓ {len(b)} 根, "
                          f"最新 {b[-1]['datetime'].date()} close={b[-1].get('close')}")
                else:
                    print(f"  [{flag}] {mkt.name} {code} {note}: 无数据")
            except Exception as e:  # noqa: BLE001
                print(f"  [{flag}] {mkt.name} {code} {note}: 异常 {type(e).__name__}: {e}")

        # 对退市股拉全历史，看它的最后一天是不是真的退市日（决定能否做"死票"回测）
        for mkt, code in got:
            if code in live:
                continue
            b, pg, _ = fetch_all(client, mkt, code, adjust=ADJUST.NONE, max_pages=12)
            if b:
                print(f"    → {mkt} {code} 全历史 {len(b)} 根: "
                      f"{b[0]['datetime'].date()} .. {b[-1]['datetime'].date()} "
                      f"(末日收盘 {b[-1].get('close')})")

        # ---- 4. 日内历史深度 ----
        hr("4. 日内数据历史深度（逐笔 / 分时）")
        today = date(2026, 10, 15)
        for back in (1, 2, 3, 5, 10, 20, 40):
            d = today - timedelta(days=back)
            if d.weekday() >= 5:
                print(f"  {d} (周末) 跳过")
                continue
            try:
                tr = client.stock_transaction(MARKET.SH, "600519", query_date=d, count=200)
                ic = client.stock_tick_chart(MARKET.SH, "600519", query_date=d)
                print(f"  {d}: 逐笔 {len(tr or []):4d} 条, 分时 {len(ic or []):4d} 点")
            except Exception as e:  # noqa: BLE001
                print(f"  {d}: 异常 {type(e).__name__}: {e}")

        # ---- 5. 回填吞吐实测（写库口径）----
        hr("5. 纯网络侧吞吐（不含落库）")
        syms = [(MARKET.SZ, c) for c in ("000001", "000002", "000063", "000333", "000538")]
        t0 = time.time()
        tot = 0
        for mkt, c in syms:
            b, _, _ = fetch_all(client, mkt, c)
            tot += len(b)
        dt = time.time() - t0
        print(f"  {len(syms)} 票共 {tot} 根 / {dt:.2f}s → {tot / dt:.0f} 根/秒, "
              f"{len(syms) / dt:.2f} 票/秒")
        print(f"  外推全 A 5580 票：约 {5580 / (len(syms) / dt) / 60:.1f} 分钟（纯网络，不含落库）")


if __name__ == "__main__":
    main()
