"""历史回填工具：为 framebt 验证回填 opentdx 数仓的日线历史。

背景
----
opentdx 的 ``db/etl.py`` 以 ``count=800`` 单次拉取，但实测该值命中协议边界
（返回 800 行却是错乱区间）；而 ``count<=700`` 正常，且 ``start`` 偏移分页可用。

因此本工具以 **分页** 方式拉取长历史（每页 700 根，按 ``start`` 递增直到耗尽），
并复用 opentdx 自有的 ``db/persist.py::persist_kline`` 写库 —— 保证 schema 与
merge 语义（按 ``datetime`` 去重、keep-last、幂等）与 opentdx 数仓完全一致。

不修改 opentdx 任何源码，只做数据接入。

用法
----
    python tools/backfill_history.py                       # 默认验证池，最多 2000 根
    python tools/backfill_history.py --max-bars 3000       # 约 12 年
    python tools/backfill_history.py --symbols "SH 600519, SZ 000001"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---- 把 opentdx 及其 db/ 平铺模块加入 import 路径 ----
_HERE = Path(__file__).resolve()
_FRAMEBT_ROOT = _HERE.parents[1]                 # .../TRADE/framebt
_TRADE_ROOT = _FRAMEBT_ROOT.parent               # .../TRADE
_OPENTDX = _TRADE_ROOT / "opentdx"
_OPENTDX_DB = _OPENTDX / "db"
for _p in (str(_OPENTDX), str(_OPENTDX_DB)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from opentdx import TdxClient                                  # noqa: E402
from opentdx.const import ADJUST, MARKET, PERIOD               # noqa: E402
import persist as tdx_persist                                  # noqa: E402  (opentdx/db/persist.py)

PAGE = 700          # 单次请求安全上限（800 命中协议边界，返回错乱区间）

# 冒烟样例：仅用于快速验证回填链路（20 只，秒级完成）。
# ⚠️ 这是「冒烟测试」，不是验证池、更不是「全市场」。opentdx 数仓现成已有
# 5500+ 只 QFQ 日线（见 OpentdxParquetSource.available_symbols()），框架校验默认
# 加载全市场而非此子集。回填工具只用于「补齐数仓里还没有的新标的」，不要把它当成
# 定义验证样本的入口——那样会引入幸存者偏差（以下全是 2015 年前上市且至今存续的大盘股）。
DEFAULT_UNIVERSE = [
    ("SH", "600519"),  # 贵州茅台
    ("SH", "600036"),  # 招商银行
    ("SH", "600000"),  # 浦发银行
    ("SH", "601318"),  # 中国平安
    ("SH", "600030"),  # 中信证券
    ("SH", "601166"),  # 兴业银行
    ("SH", "600276"),  # 恒瑞医药
    ("SH", "600887"),  # 伊利股份
    ("SH", "601988"),  # 中国银行
    ("SH", "600028"),  # 中国石化
    ("SH", "600585"),  # 海螺水泥
    ("SH", "600309"),  # 万华化学
    ("SH", "601899"),  # 紫金矿业
    ("SH", "600690"),  # 海尔智家
    ("SZ", "000001"),  # 平安银行
    ("SZ", "000002"),  # 万科A
    ("SZ", "000651"),  # 格力电器
    ("SZ", "000858"),  # 五粮液
    ("SZ", "000333"),  # 美的集团
    ("SZ", "002415"),  # 海康威视
    ("SZ", "000063"),  # 中兴通讯
    ("SZ", "002304"),  # 洋河股份
    ("SZ", "000538"),  # 云南白药
    ("SZ", "000725"),  # 京东方A
]

_NAME_TO_MARKET = {"SZ": MARKET.SZ, "SH": MARKET.SH, "BJ": MARKET.BJ}


def parse_symbols(spec: str) -> list[tuple[str, str]]:
    out = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        m, c = token.split()
        out.append((m.upper(), c.strip()))
    return out


def fetch_paged(client: TdxClient, market, code: str, max_bars: int, adjust) -> list[dict]:
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
            collected[b["datetime"]] = b          # 按 datetime 去重
        if len(collected) == before:
            break                                   # 无新增 → 历史耗尽
        start += len(bars)
    return sorted(collected.values(), key=lambda b: b["datetime"])


def main() -> None:
    ap = argparse.ArgumentParser(description="回填 opentdx 数仓日线历史（分页）")
    ap.add_argument("--symbols", default=None, help="'SH 600519, SZ 000001'（默认仅 20 只冒烟样例，非验证池）")
    ap.add_argument("--max-bars", type=int, default=2000, help="每标的最多 K 线数（默认 2000≈8年）")
    ap.add_argument("--adjust", type=int, default=int(ADJUST.QFQ.value), help="0 不权/1 前复权/2 后复权")
    args = ap.parse_args()

    universe = (parse_symbols(args.symbols) if args.symbols else DEFAULT_UNIVERSE)
    adjust = ADJUST(args.adjust)

    print(f"回填 {len(universe)} 个标的，每标的最多 {args.max_bars} 根日线，"
          f"复权={adjust.name}，页大小={PAGE}")
    print(f"目标数仓: {_OPENTDX_DB / 'data'}\n")

    ok, fail = [], []
    t0 = time.time()
    with TdxClient() as client:
        for i, (mkt_name, code) in enumerate(universe, 1):
            market = _NAME_TO_MARKET[mkt_name]
            try:
                bars = fetch_paged(client, market, code, args.max_bars, adjust)
            except Exception as e:  # noqa: BLE001
                print(f"[{i:2d}/{len(universe)}] {mkt_name} {code} 拉取失败: {type(e).__name__}: {e}")
                fail.append((mkt_name, code))
                continue
            if not bars:
                print(f"[{i:2d}/{len(universe)}] {mkt_name} {code} 无数据，跳过")
                fail.append((mkt_name, code))
                continue
            n = tdx_persist.persist_kline(market, code, PERIOD.DAILY, adjust, bars)
            span = f"{bars[0]['datetime'].date()} .. {bars[-1]['datetime'].date()}"
            print(f"[{i:2d}/{len(universe)}] {mkt_name} {code} 拉到 {len(bars):4d} 根 → 落库 {n:4d} 行  {span}")
            ok.append((mkt_name, code, len(bars)))

    print(f"\n完成：成功 {len(ok)}，失败/无数据 {len(fail)}，耗时 {time.time() - t0:.1f}s")
    if fail:
        print("失败清单:", ", ".join(f"{m} {c}" for m, c in fail))


if __name__ == "__main__":
    main()
