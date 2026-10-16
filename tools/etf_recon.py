"""ETF 侦察：拉全表 + 探测候选标的的历史深度。

产出 tools/etf_universe.json（按成交额排序的 ETF 清单，供后续 PIT 选池）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from opentdx import TdxClient
from opentdx.const import ADJUST, MARKET, PERIOD, SORT_ORDER, SORT_TYPE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "etf_universe.json"


def fetch_all(client, market, code, hard_cap: int = 20000) -> list[dict]:
    """把一只标的的全部日线拉全（start 从新往旧偏移）。"""
    out: list[dict] = []
    start = 0
    while len(out) < hard_cap:
        try:
            bars = list(client.stock_kline(market, code, PERIOD.DAILY,
                                           start=start, count=700, adjust=ADJUST.QFQ))
        except Exception:
            break
        if not bars:
            break
        out.extend(bars)
        start += len(bars)
        if len(bars) < 700:
            break
    out.sort(key=lambda b: b["datetime"])
    return out


def main() -> None:
    t0 = time.time()
    with TdxClient() as c:
        rows = c.stock_board_members("11005", count=600,
                                     sort_type=SORT_TYPE.TOTAL_AMOUNT,
                                     sort_order=SORT_ORDER.DESC)
        print(f"ETF 全表(按成交额前 {len(rows)}): {time.time()-t0:.1f}s")
        uni = [{"market": "SH" if r["market"] == MARKET.SH else "SZ",
                "code": str(r["code"]), "name": r["name"],
                "amount": float(r.get("amount") or 0.0)} for r in rows]
        OUT.write_text(json.dumps(uni, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"写入 {OUT}")
        print("\n前 40 只（代码 / 名称 / 成交额万）:")
        for u in uni[:40]:
            print(f"  {u['market']}.{u['code']}  {u['name']:<22} {u['amount']/1e4:>12,.0f}")

        # 候选：宽基/资产类别/长历史优先
        cand = [
            ("SH", "510050", "上证50ETF"), ("SH", "510300", "沪深300ETF"),
            ("SH", "510500", "中证500ETF"), ("SZ", "159915", "创业板ETF"),
            ("SH", "512100", "中证1000ETF"), ("SH", "588000", "科创50ETF"),
            ("SH", "511010", "国债ETF"), ("SH", "511260", "十年国债ETF"),
            ("SH", "518880", "黄金ETF"), ("SH", "513100", "纳指ETF"),
            ("SH", "513500", "标普500ETF"), ("SZ", "159920", "恒生ETF"),
            ("SH", "512880", "证券ETF"), ("SH", "512400", "有色ETF"),
            ("SH", "512170", "医疗ETF"), ("SH", "512690", "酒ETF"),
            ("SH", "515030", "新能源车ETF"), ("SH", "512480", "半导体ETF"),
            ("SZ", "159949", "创业板50ETF"), ("SH", "515000", "科技ETF"),
            ("SH", "512660", "军工ETF"), ("SH", "512800", "银行ETF"),
            ("SH", "515180", "红利ETF"), ("SH", "510880", "红利ETF"),
            ("SH", "511880", "银华日利"), ("SH", "511990", "华宝添益"),
        ]
        print("\n候选历史深度:")
        print(f"  {'标的':<16}{'市场':<5}{'首日':<13}{'末日':<13}{'根数':>7}")
        info = []
        for mkt, code, label in cand:
            mk = MARKET.SH if mkt == "SH" else MARKET.SZ
            bars = fetch_all(c, mk, code)
            if not bars:
                print(f"  {code} {label:<10} 无数据")
                continue
            f, l = bars[0]["datetime"], bars[-1]["datetime"]
            print(f"  {code} {label:<10} {mkt:<5}{str(f.date()):<13}"
                  f"{str(l.date()):<13}{len(bars):>7}")
            info.append({"market": mkt, "code": code, "label": label,
                         "first": str(f.date()), "last": str(l.date()), "n": len(bars)})
        (ROOT / "etf_candidates.json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
