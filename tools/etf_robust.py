"""ETF 等权多资产基准的稳健性：菜单成分 / 子区间 / 起始日敏感性。

目的：检验"等权多资产 10%/年"是否只由单一资产（黄金）或单一行情驱动。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "etf_cache"
MENU = ["SH.510050", "SH.510300", "SH.510500", "SZ.159915", "SH.512100",
        "SZ.159920", "SH.513100", "SH.513500", "SH.511010", "SH.511260",
        "SH.518880"]
LABEL = {"SH.510050": "上证50", "SH.510300": "沪深300", "SH.510500": "中证500",
         "SZ.159915": "创业板", "SH.512100": "中证1000", "SZ.159920": "恒生",
         "SH.513100": "纳指", "SH.513500": "标普500", "SH.511010": "国债",
         "SH.511260": "十年国债", "SH.518880": "黄金"}


def load(sym: str) -> pd.Series:
    df = pd.read_parquet(CACHE / f"{sym.replace('.', '_')}.parquet")
    return df.set_index("dt_ord")["close"].astype(float)


def ew(syms: list[str], start: int, end: int | None = None,
       every: int = 21, vol_target: float | None = None,
       vol_window: int = 60) -> dict:
    cols = {s: load(s) for s in syms}
    P = pd.DataFrame(cols).sort_index().ffill()
    if end is not None:
        P = P[P.index <= end]
    P = P[P.index >= start]
    if len(P) < 60:
        return {}
    R = P.pct_change().fillna(0.0)
    avail = P.notna()
    w = np.zeros(P.shape[1])
    eq = [1.0]
    for i in range(1, len(P)):
        ok = avail.iloc[i].to_numpy()
        if i % every == 0 or w.sum() <= 0:
            w = ok.astype(float) / max(ok.sum(), 1)
        base = w / w.sum()
        scale = 1.0
        if vol_target is not None and i >= vol_window + 1:
            sub = R.iloc[i - vol_window:i].to_numpy()
            pr = sub @ base
            sd = float(np.std(pr, ddof=1)) * math.sqrt(252.0)
            if sd > 1e-9:
                scale = min(1.0, vol_target / sd)
        ri = R.iloc[i].fillna(0.0).to_numpy()
        eq.append(eq[-1] * (1.0 + float(ri @ base) * scale))
        w = w * (1.0 + ri * scale)
        tot = w.sum()
        if tot > 0:
            w = w / tot
    px = pd.Series(eq, index=P.index)
    r = px.pct_change().dropna()
    tot = float(px.iloc[-1] / px.iloc[0] - 1.0)
    yrs = len(px) / 252.0
    sd = float(r.std(ddof=1)) * math.sqrt(252.0)
    return {"total": tot,
            "ann": float((1 + tot) ** (1 / max(yrs, 1e-9)) - 1),
            "sharpe": float(r.mean() * 252 / sd) if sd > 1e-12 else 0.0,
            "mdd": float(((px.cummax() - px) / px.cummax()).max()),
            "vol": sd, "years": yrs}


def row(label: str, st: dict) -> None:
    if not st:
        print(f"  {label:<30} 样本不足")
        return
    print(f"  {label:<30}{st['total']:>+11.2%}{st['ann']:>+9.2%}"
          f"{st['sharpe']:>8.2f}{st['mdd']:>9.2%}{st['vol']:>8.2%}")


def main() -> None:
    S14 = pd.Timestamp("2014-01-01").toordinal()
    S20 = pd.Timestamp("2020-01-01").toordinal()
    E19 = pd.Timestamp("2019-12-31").toordinal()
    print(f"  {'配置':<28}{'总收益':>11}{'年化':>9}{'Sharpe':>8}{'回撤':>9}{'波动':>8}")
    print("\n[菜单成分敏感性] 全区间 2014-01 → 2026-10")
    row("全菜单（11 只）", ew(MENU, S14))
    row("剔除黄金", ew([s for s in MENU if s != "SH.518880"], S14))
    row("剔除海外(恒生/纳指/标普)",
        ew([s for s in MENU if s not in ("SZ.159920", "SH.513100", "SH.513500")], S14))
    row("剔除债券", ew([s for s in MENU if s not in ("SH.511010", "SH.511260")], S14))
    row("仅 A 股宽基(5 只)",
        ew(["SH.510050", "SH.510300", "SH.510500", "SZ.159915", "SH.512100"], S14))
    row("仅 A 股 + 债券", ew(["SH.510050", "SH.510300", "SH.510500", "SZ.159915",
                              "SH.512100", "SH.511010", "SH.511260"], S14))
    row("全菜单 + 12% 波动目标", ew(MENU, S14, vol_target=0.12))

    print("\n[子区间稳定性]")
    row("2014-2019", ew(MENU, S14, E19))
    row("2020-2026", ew(MENU, S20))
    row("2014-2019 剔除黄金", ew([s for s in MENU if s != "SH.518880"], S14, E19))
    row("2020-2026 剔除黄金", ew([s for s in MENU if s != "SH.518880"], S20))

    print("\n[起始日敏感性] 全菜单")
    for y in (2013, 2014, 2015, 2016, 2017, 2018):
        st = ew(MENU, pd.Timestamp(f"{y}-01-01").toordinal())
        if st:
            print(f"  起 {y}-01-01{'':<14}{st['total']:>+11.2%}{st['ann']:>+9.2%}"
                  f"{st['sharpe']:>8.2f}{st['mdd']:>9.2%}{st['vol']:>8.2%}")

    print("\n[单资产对照] 2014-01 → 2026-10 买入持有")
    for s in MENU:
        px = load(s)
        px = px[px.index >= S14]
        if len(px) < 60:
            continue
        r = px.pct_change().dropna()
        tot = float(px.iloc[-1] / px.iloc[0] - 1.0)
        yrs = len(px) / 252.0
        sd = float(r.std(ddof=1)) * math.sqrt(252.0)
        mdd = float(((px.cummax() - px) / px.cummax()).max())
        print(f"  {LABEL[s]:<30}{tot:>+11.2%}"
              f"{(1+tot)**(1/yrs)-1:>+9.2%}"
              f"{r.mean()*252/sd if sd>1e-12 else 0:>8.2f}{mdd:>9.2%}{sd:>8.2%}")


if __name__ == "__main__":
    main()
