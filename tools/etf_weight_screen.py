"""ETF 权重方案筛选（近似口径，不含引擎）。

回答一个问题：在 11 只 ETF + 货币 ETF 现金腿上，「按资产类别等权」是否比
「按标的等权」更值得进引擎做精确回测？

近似口径：月末（每 21 交易日）调仓、收盘价、单边 3bp 成本、可选 12% 波动率目标
（只降不升）。精确逐笔结果以 ``examples/risk_budget_etf.py`` 为准。

用法：
    python tools/etf_weight_screen.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "tools" / "etf_cache"

MENU = {
    "SH.510050": ("A", "SSE50"),
    "SH.510300": ("A", "CSI300"),
    "SH.510500": ("A", "CSI500"),
    "SZ.159915": ("A", "ChiNext"),
    "SH.512100": ("A", "CSI1000"),
    "SZ.159920": ("O", "HSI"),
    "SH.513100": ("O", "NDX"),
    "SH.513500": ("O", "SPX"),
    "SH.511010": ("B", "CGB"),
    "SH.511260": ("B", "CGB10"),
    "SH.518880": ("G", "Gold"),
}
CASH = "SH.511990"
START = pd.Timestamp("2014-01-01").toordinal()


def load() -> tuple[pd.DataFrame, pd.Series]:
    cols = {}
    for sym in list(MENU) + [CASH]:
        df = pd.read_parquet(CACHE / f"{sym.replace('.', '_')}.parquet")
        cols[sym] = df.set_index("dt_ord")["close"].astype(float)
    P = pd.DataFrame(cols).sort_index()
    P = P[P.index >= START]
    return P[list(MENU)], P[CASH]


def erc_weights(cov: np.ndarray) -> np.ndarray:
    """等风险贡献（循环坐标下降）。"""
    n = cov.shape[0]
    w = np.full(n, 1.0 / n)
    target = 1.0 / n
    for _ in range(300):
        for i in range(n):
            a = cov[i, i]
            b = float(np.delete(w, i) @ np.delete(cov[i], i))
            c = target * float(w @ cov @ w)
            disc = b * b + 4.0 * a * c
            w[i] = (-b + math.sqrt(disc)) / (2.0 * a) if a > 0 else 0.0
        s = w.sum()
        if s > 0:
            w /= s
    return w


def weights_for(kind: str, rets: np.ndarray, syms: list[str]) -> np.ndarray:
    n = len(syms)
    if kind == "equal":
        return np.full(n, 1.0 / n)
    if kind == "inv_vol":
        v = rets.std(axis=0, ddof=1) * math.sqrt(252)
        v = np.where(v > 1e-9, v, np.nan)
        inv = 1.0 / np.nan_to_num(v, nan=np.nanmax(v))
        return inv / inv.sum()
    if kind == "erc":
        return erc_weights(np.cov(rets, rowvar=False, ddof=1))
    classes = [MENU[s][0] for s in syms]
    uniq = list(dict.fromkeys(classes))
    if kind == "class_equal":
        w = np.zeros(n)
        for cl in uniq:
            idx = [i for i, c in enumerate(classes) if c == cl]
            w[idx] = 1.0 / len(uniq) / len(idx)
        return w
    if kind == "class_erc":
        cov = np.cov(rets, rowvar=False, ddof=1)
        w = np.zeros(n)
        for cl in uniq:
            idx = [i for i, c in enumerate(classes) if c == cl]
            w[idx] = erc_weights(cov[np.ix_(idx, idx)]) / len(uniq)
        return w
    raise ValueError(kind)


def run(kind: str, P: pd.DataFrame, cash: pd.Series, *, vol_window: int = 60,
        target_vol: float | None = None, every: int = 21,
        cost_bps: float = 3.0) -> dict:
    syms = list(P.columns)
    avail = P.notna().to_numpy()
    px = P.ffill().to_numpy()
    n = len(px)
    r = np.zeros_like(px)
    r[1:] = px[1:] / px[:-1] - 1.0
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    cash_r = cash.pct_change().fillna(0.0).to_numpy()
    first = np.array([np.argmax(avail[:, j]) if avail[:, j].any() else n
                      for j in range(len(syms))])

    w = np.zeros(len(syms))
    eq = [1.0]
    turnover = 0.0
    for i in range(1, n):
        if i % every == 0 and i > vol_window + 1:
            ok = avail[i] & (i - first >= vol_window)
            if ok.sum() >= 2:
                sub = [syms[j] for j in range(len(syms)) if ok[j]]
                hist = r[i - vol_window:i][:, ok]
                base_sub = weights_for(kind, hist, sub)
                scale = 1.0
                if target_vol is not None:
                    pv = (hist @ base_sub).std(ddof=1) * math.sqrt(252)
                    scale = min(1.0, target_vol / pv) if pv > 1e-9 else 1.0
                new = np.zeros(len(syms))
                new[ok] = base_sub * scale
                turnover += float(np.abs(new - w).sum()) / 2.0
                if cost_bps:
                    eq[-1] *= 1.0 - cost_bps * 1e-4 * float(np.abs(new - w).sum())
                w = new
        gross = float(w @ (1.0 + r[i])) + float(1.0 - w.sum()) * (1.0 + cash_r[i])
        eq.append(eq[-1] * gross)
        w = w * (1.0 + r[i]) / max(gross, 1e-12)

    e = pd.Series(np.asarray(eq), index=P.index)
    rr = e.pct_change().dropna()
    yrs = len(rr) / 252.0
    total = float(e.iloc[-1] / e.iloc[0] - 1.0)
    sd = float(rr.std(ddof=1)) * math.sqrt(252)
    return {"ann": float((1 + total) ** (1 / yrs) - 1),
            "sharpe": float(rr.mean() * 252 / sd) if sd > 1e-12 else 0.0,
            "mdd": float(((e.cummax() - e) / e.cummax()).max()),
            "vol": sd, "turnover": turnover / yrs, "total": total}


def main() -> None:
    P, cash = load()
    print(f"range {pd.Timestamp.fromordinal(P.index.min()).date()} .. "
          f"{pd.Timestamp.fromordinal(P.index.max()).date()}  {len(P)} td\n")

    for tv, label in ((None, "no vol target"), (0.12, "12% vol target")):
        print(f"== {label} ==  (cost 3bp)")
        print(f"{'scheme':<14}{'ann':>9}{'Sharpe':>8}{'MDD':>8}{'vol':>8}{'turn':>7}")
        for kind in ("equal", "inv_vol", "erc", "class_equal", "class_erc"):
            m = run(kind, P, cash, target_vol=tv)
            print(f"{kind:<14}{m['ann']:>+9.2%}{m['sharpe']:>8.2f}"
                  f"{m['mdd']:>8.2%}{m['vol']:>8.2%}{m['turnover']:>7.2f}")
        print()

    print("== robustness: class budget vs instrument equal (no vol target) ==")
    print(f"{'window':<11}{'subset':<14}{'scheme':<13}"
          f"{'ann':>9}{'Sharpe':>8}{'MDD':>8}")
    splits = (("2014-2019", "2014-01-01", "2019-12-31"),
              ("2020-2026", "2020-01-01", "2026-12-31"))
    subsets = (("all", []),
               ("no gold", ["SH.518880"]),
               ("no overseas", ["SZ.159920", "SH.513100", "SH.513500"]),
               ("no bonds", ["SH.511010", "SH.511260"]),
               ("A-share only", ["SZ.159920", "SH.513100", "SH.513500",
                                 "SH.511010", "SH.511260", "SH.518880"]))
    for name, a, b in splits:
        Ps = P[(P.index >= pd.Timestamp(a).toordinal()) & (P.index <= pd.Timestamp(b).toordinal())]
        cs = cash.reindex(Ps.index)
        for label, drop in subsets:
            Ps2 = Ps.drop(columns=[c for c in drop if c in Ps.columns])
            if Ps2.shape[1] < 2:
                continue
            for kind in ("equal", "class_equal", "class_erc"):
                m = run(kind, Ps2, cs, target_vol=None)
                print(f"{name:<11}{label:<14}{kind:<13}"
                      f"{m['ann']:>+9.2%}{m['sharpe']:>8.2f}{m['mdd']:>8.2%}")


if __name__ == "__main__":
    main()
