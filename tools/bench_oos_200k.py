"""阶段 1 · 第 0 步：把 ADAPTIVE 的样本外收益对齐到「可投资基准」。

回答 PATH_200K.md 第 3.1 节的问题：
  +42.47% 的样本外超额，对手到底是「24 只存活大盘股等权」还是「你能买到的 ETF」？

方法（不依赖任何框架内部数字，纯日线收盘价）：
  - ADAPTIVE 样本外窗口 = 交易日序号 738014..739904（即 pd.Timestamp.toordinal()，
    739904 == 2026-10-15，末帧）；
  - 拉取 510300（沪深300ETF）、510500（中证500ETF）全历史日线（QFQ）；
  - 直接读 opentdx 数仓里已有的 24 只票 QFQ 日线，算等权买入持有；
  - 三者同窗口并排，与 +42.47% 对照。
"""

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

from opentdx import TdxClient                                    # noqa: E402
from opentdx.const import ADJUST, MARKET, PERIOD                 # noqa: E402

import pandas as pd                                              # noqa: E402

STRATEGY_OOS = 0.4247            # +42.47%
BENCH_24_OOS = 0.1671            # +16.71%（ADAPTIVE §5 报的 24 票等权基准）
WIN_START_ORD = 738014           # 样本外起点（ordinal）
WIN_END_ORD = 739904             # 样本外终点（= 2026-10-15）


def _dts(df: pd.DataFrame) -> pd.Series:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"])
    if "dt" in df.columns:
        return pd.to_datetime(df["dt"])
    raise KeyError("no datetime column")


def buy_hold(df: pd.DataFrame, start_ord: int, end_ord: int):
    """[start_ord, end_ord] 上买入持有收益（起点首日收盘买入，终点收盘卖出）。"""
    dts = _dts(df)
    cs = df["close"].to_numpy(dtype=float)
    start = pd.Timestamp.fromordinal(start_ord)
    end = pd.Timestamp.fromordinal(end_ord)
    mask = (dts >= start) & (dts <= end)
    sub = cs[mask]
    if len(sub) < 2:
        return float("nan"), None, None
    return float(sub[-1] / sub[0] - 1.0), dts[mask].iloc[0], dts[mask].iloc[-1]


def main() -> None:
    with TdxClient() as client:
        print("=" * 72)
        print(f"  样本外窗口: 交易日序号 {WIN_START_ORD} .. {WIN_END_ORD}  "
              f"(739904 == 2026-10-15)")
        print("=" * 72)

        etfs = [("510300", "沪深300ETF"), ("510500", "中证500ETF")]
        etf_returns = {}
        for code, name in etfs:
            t0 = time.time()
            bars = client.stock_kline(MARKET.SH, code, PERIOD.DAILY,
                                      start=0, count=700, adjust=ADJUST.QFQ)
            all_bars = list(bars)
            start = 700
            while len(all_bars) < 12000:
                more = client.stock_kline(MARKET.SH, code, PERIOD.DAILY,
                                          start=start, count=700, adjust=ADJUST.QFQ)
                if not more:
                    break
                all_bars.extend(more)
                start += len(more)
                if len(more) < 700:
                    break
            df = pd.DataFrame(all_bars).sort_values("datetime")
            dt = time.time() - t0
            ret, d0, d1 = buy_hold(df, WIN_START_ORD, WIN_END_ORD)
            etf_returns[code] = ret
            print(f"  {code} {name}: {len(df)} 根, {d0.date()} .. {d1.date()}")
            print(f"      样本外买入持有收益 = {ret:+.2%}   ({dt:.1f}s)")

        # ---- 24 只票等权（直接读数仓 QFQ 日线）----
        print("\n  24 只等权基准（直接读数仓 QFQ 日线，重算）：")
        kline_root = _OPENTDX / "db" / "data" / "fakt_kline" / "4" / "1"
        closes = {}
        if kline_root.is_dir():
            for path in sorted(kline_root.glob("*.parquet")):
                try:
                    df = pd.read_parquet(path)
                except Exception:
                    continue
                if "symbol_key" not in df.columns or "dt" not in df.columns:
                    continue
                df = df.sort_values("dt")
                sym = str(df["symbol_key"].iloc[0])
                closes[sym] = pd.to_datetime(df["dt"]).to_numpy(), df["close"].to_numpy(dtype=float)

        if closes:
            syms = sorted(closes)
            print(f"    读取 {len(syms)} 只 QFQ 日线")
            start = pd.Timestamp.fromordinal(WIN_START_ORD)
            end = pd.Timestamp.fromordinal(WIN_END_ORD)
            per_ret = []
            for sym in syms:
                dts, cs = closes[sym]
                mask = (dts >= start) & (dts <= end)
                sub = cs[mask]
                if len(sub) >= 2:
                    per_ret.append(float(sub[-1] / sub[0] - 1.0))
            ew = float(sum(per_ret) / len(per_ret)) if per_ret else float("nan")
            print(f"    24 只等权样本外平均收益 = {ew:+.2%}")

    # ---- 对照表 ----
    print("\n" + "=" * 72)
    print("  对照（样本外窗口 738014..739904）")
    print("=" * 72)
    print(f"  {'项目':<32} {'收益':>10}  {'备注':<24}")
    print(f"  {'自适应多因子（ADAPTIVE 报告）':<32} {STRATEGY_OOS:+10.2%}  {'报告值':<24}")
    print(f"  {'24 只存活大盘股等权（报告）':<32} {BENCH_24_OOS:+10.2%}  {'报告值':<24}")
    for code, name in etfs:
        if code in etf_returns:
            vs = etf_returns[code] - STRATEGY_OOS
            tag = "跑赢策略" if vs > 0 else "跑输策略"
            print(f"  {name:<32} {etf_returns[code]:+10.2%}  {'vs 策略 ' + f'{vs:+.2%} ({tag})':<24}")

    print("\n  ── 净超额（策略 − ETF）──")
    for code, name in etfs:
        if code in etf_returns:
            ex = STRATEGY_OOS - etf_returns[code]
            if ex <= 0:
                print(f"  >>> {name} 同期买入持有 ≥ 策略收益，策略未产生可归因的超额")
            else:
                print(f"  >>> 策略同期跑赢 {name} {ex:+.2%}（仍需看波动与回撤是否值得承担）")


if __name__ == "__main__":
    main()
