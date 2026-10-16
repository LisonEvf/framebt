"""ETF 多资产趋势跟踪回测（用本项目 framebt 引擎）。

预登记设计见 docs/ETF_TREND.md。核心对照：
  - 主策略：TSMOM + 趋势过滤 + 波动率目标（低换手多资产）
  - **消融**：--no-trend（去掉择时，永远持有同一菜单）→ 检验"择时是否产生价值"
  - 基准：买入持有 510300（沪深300ETF）

成本：ETF 免印花税、免过户费；佣金 万1.5（最低 5 元）+ 比例滑点。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from framebt import Engine, StandardCost, StandardRisk
from framebt.core import Bar
from framebt.feed import BarFeed
from framebt.strategies.etf_trend import EtfTrendStrategy
from opentdx import TdxClient
from opentdx.const import ADJUST, MARKET, PERIOD

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "etf_cache"
LINES: list[str] = []

# 结构性资产类别菜单（非按业绩事后挑选）
MENU: dict[str, str] = {
    "SH.510050": "上证50",
    "SH.510300": "沪深300",
    "SH.510500": "中证500",
    "SZ.159915": "创业板",
    "SH.512100": "中证1000",
    "SZ.159920": "恒生",
    "SH.513100": "纳指",
    "SH.513500": "标普500",
    "SH.511010": "国债",
    "SH.511260": "十年国债",
    "SH.518880": "黄金",
}
CASH = "SH.511990"          # 华宝添益（货币 ETF）
FEED_START = pd.Timestamp("2012-01-01").toordinal()
REPORT_START = pd.Timestamp("2014-01-01").toordinal()


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(t: str) -> None:
    out("")
    out("=" * 74)
    out(f"  {t}")
    out("=" * 74)


# ---------------- 数据 ----------------

def fetch(sym: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    p = CACHE / f"{sym.replace('.', '_')}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    mkt, code = sym.split(".")
    mk = MARKET.SH if mkt == "SH" else MARKET.SZ
    rows: list[dict] = []
    with TdxClient() as c:
        start = 0
        while True:
            try:
                got = list(c.stock_kline(mk, code, PERIOD.DAILY,
                                         start=start, count=700, adjust=ADJUST.QFQ))
            except Exception:
                break
            if not got:
                break
            rows.extend(got)
            start += len(got)
            if len(got) < 700:
                break
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"{sym} 无数据")
    df["dt_ord"] = pd.to_datetime(df["datetime"]).map(lambda d: d.toordinal())
    df = df.sort_values("dt_ord").drop_duplicates("dt_ord")
    df = df[["dt_ord", "open", "high", "low", "close", "vol", "amount"]]
    df.to_parquet(p, index=False)
    return df


def build_feed() -> tuple[list[Bar], dict[str, pd.DataFrame]]:
    dfs: dict[str, pd.DataFrame] = {}
    bars: list[Bar] = []
    for sym in list(MENU) + [CASH]:
        df = fetch(sym)
        df = df[df["dt_ord"] >= FEED_START]
        dfs[sym] = df
        for r in df.to_dict("records"):
            c = float(r["close"])
            if not (c > 0):
                continue
            vol = float(r["vol"] or 0.0)
            amt = float(r["amount"] or 0.0)
            bars.append(Bar(
                timestamp=int(r["dt_ord"]), symbol=sym,
                open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
                close=c, volume=vol, amount=amt,
                float_shares=0.0, turnover=0.0,
                vwap=(amt / vol) if vol > 0 else c, market_cap=0.0,
            ))
    return bars, dfs


# ---------------- 指标 ----------------

def metrics(eq: pd.Series, fills, cash0: float) -> dict:
    eq = eq[eq.index >= REPORT_START]
    if len(eq) < 30:
        return {}
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann = float((1.0 + total) ** (1.0 / max(yrs, 1e-9)) - 1.0)
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * math.sqrt(252.0)) if sd > 1e-12 else 0.0
    cummax = eq.cummax()
    mdd = float(((cummax - eq) / cummax).max())
    traded = sum(f.size * f.price for f in fills)
    fees = sum(f.cost for f in fills)
    avg_eq = float(eq.mean())
    turnover = traded / avg_eq / max(yrs, 1e-9) / 2.0
    return {"total": total, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "turnover": turnover, "fees": fees, "traded": traded,
            "final": float(eq.iloc[-1]), "start": float(eq.iloc[0]),
            "years": yrs, "n": len(eq)}


def _stats(px: pd.Series) -> dict:
    r = px.pct_change().dropna()
    total = float(px.iloc[-1] / px.iloc[0] - 1.0)
    yrs = len(px) / 252.0
    sd = float(r.std(ddof=1)) * math.sqrt(252.0)
    sharpe = float(r.mean() * 252.0 / sd) if sd > 1e-12 else 0.0
    cummax = px.cummax()
    mdd = float(((cummax - px) / cummax).max())
    return {"total": total,
            "ann": float((1 + total) ** (1 / max(yrs, 1e-9)) - 1),
            "sharpe": sharpe, "mdd": mdd, "vol": sd, "n": len(px), "years": yrs}


def buyhold(dfs: dict[str, pd.DataFrame], sym: str) -> dict:
    df = dfs[sym]
    df = df[df["dt_ord"] >= REPORT_START]
    return _stats(df.set_index("dt_ord")["close"].astype(float))


def ew_series(dfs: dict[str, pd.DataFrame], syms: list[str],
              every: int = 21) -> pd.Series:
    """等权多资产、每 every 日调回等权的买入持有净值序列（不含成本）。"""
    cols = {}
    for s in syms:
        d = dfs[s]
        d = d[d["dt_ord"] >= REPORT_START]
        cols[s] = d.set_index("dt_ord")["close"].astype(float)
    P = pd.DataFrame(cols).sort_index().ffill()
    R = P.pct_change().fillna(0.0)
    avail = P.notna()
    w = np.zeros(P.shape[1])
    eq = [1.0]
    for i in range(1, len(P)):
        ok = avail.iloc[i].to_numpy()
        if i % every == 0 or w.sum() <= 0:
            w = ok.astype(float) / max(ok.sum(), 1)
        w = w * (1.0 + R.iloc[i].fillna(0.0).to_numpy())
        tot = w.sum()
        if tot > 0:
            w = w / tot
        eq.append(eq[-1] * float(tot))      # 累乘，不是单日因子
    return pd.Series(eq, index=P.index)


def ew_multiasset(dfs: dict[str, pd.DataFrame], syms: list[str],
                  every: int = 21) -> dict:
    return _stats(ew_series(dfs, syms, every))


def paired_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 2000,
                     block: int = 21, seed: int = 0) -> tuple[float, float, float]:
    """配对块自举：Sharpe(a) − Sharpe(b) 的中位数与 5%/95% 分位。"""
    df = pd.DataFrame({"a": a.pct_change(), "b": b.pct_change()}).dropna()
    ra, rb = df["a"].to_numpy(), df["b"].to_numpy()
    n = len(ra)
    if n < 60:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    nb = max(1, n // block)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, max(1, n - block), size=nb)
        ia = np.concatenate([np.arange(i, i + block) for i in idx])
        sa, sb = ra[ia], rb[ia]
        da, db = sa.std(ddof=1), sb.std(ddof=1)
        if da <= 1e-12 or db <= 1e-12:
            continue
        diffs.append((sa.mean() / da - sb.mean() / db) * math.sqrt(252.0))
    d = np.asarray(diffs)
    if d.size == 0:
        return (float("nan"),) * 3
    return float(np.median(d)), float(np.percentile(d, 5)), float(np.percentile(d, 95))


def bootstrap_sharpe(eq: pd.Series, n_boot: int = 2000, block: int = 21,
                     seed: int = 0) -> tuple[float, float, float]:
    r = eq[eq.index >= REPORT_START].pct_change().dropna().to_numpy()
    if len(r) < 60:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    n = len(r)
    nb = max(1, n // block)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, max(1, n - block), size=nb)
        s = np.concatenate([r[i:i + block] for i in idx])
        sd = s.std(ddof=1)
        stats.append(s.mean() / sd * math.sqrt(252.0) if sd > 1e-12 else 0.0)
    a = np.asarray(stats)
    return float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95))


# ---------------- 主流程 ----------------

def make_cost(commission: float, slip: float) -> StandardCost:
    return StandardCost(commission=commission, min_commission=5.0,
                        stamp_tax=0.0, transfer_fee=0.0,
                        slippage="proportional", k=slip)


def run(args) -> None:
    t0 = time.time()
    hr("0. 数据")
    bars, dfs = build_feed()
    out(f"  ETF 菜单 {len(MENU)} 只 + 现金腿 {CASH}（华宝添益）")
    out(f"  总 bars {len(bars):,}   数据区间 "
        f"{pd.Timestamp.fromordinal(min(b.timestamp for b in bars)).date()} .. "
        f"{pd.Timestamp.fromordinal(max(b.timestamp for b in bars)).date()}")
    out(f"  报告区间起点 {pd.Timestamp.fromordinal(REPORT_START).date()}")

    feed = BarFeed(bars)
    strat = EtfTrendStrategy(
        list(MENU), cash_symbol=CASH,
        lookback=args.lookback, ma_window=args.ma_window,
        vol_window=args.vol_window, target_vol=args.target_vol,
        rebalance_every=args.rebalance,
        use_trend=not args.no_trend, use_vol_target=not args.no_vol_target,
        weighting=args.weighting, top_k=args.top_k,
        min_amount=args.min_amount, max_weight=args.max_weight,
    )
    hr("1. 回测（本项目 framebt 引擎 · T+1 · 100 份整手 · ETF 免印花税）")
    eng = Engine(feed, strat, initial_cash=args.cash, settlement_cycle=1,
                 cost_model=make_cost(args.commission, args.slippage),
                 risk=StandardRisk(max_position_pct=0.99))
    res = eng.run()
    eq = pd.Series({int(k): float(v) for k, v in res.equity_curve}).sort_index()
    out(f"  回测耗时 {time.time()-t0:.1f}s   帧数 {res.frames}   成交 {res.total_trades()} 笔")
    m = metrics(eq, res.fills, args.cash)
    if not m:
        out("  样本不足")
        return

    hr("2. 主策略业绩（报告区间）")
    out(f"  起 {m['start']:,.0f} → 终 {m['final']:,.0f}   跨度 {m['years']:.2f} 年")
    out(f"  总收益 {m['total']:+.2%}   年化 {m['ann']:+.2%}")
    out(f"  Sharpe {m['sharpe']:.2f}   最大回撤 {m['mdd']:.2%}")
    out(f"  年化单边换手 {m['turnover']:.2f}×   费用 {m['fees']:,.0f}"
        f"（{m['fees']/args.cash:.2%} 本金）")
    med, lo, hi = bootstrap_sharpe(eq)
    out(f"  bootstrap Sharpe 中位 {med:.2f}（5% {lo:.2f} .. 95% {hi:.2f}）")

    hr("3. 基准对比（同区间，含风险调整）")
    out(f"  {'组合':<26}{'总收益':>11}{'年化':>9}{'Sharpe':>8}{'回撤':>9}{'波动':>8}{'IR':>7}")
    out(f"  {'主策略（趋势+波动目标）':<22}{m['total']:>+11.2%}{m['ann']:>+9.2%}"
        f"{m['sharpe']:>8.2f}{m['mdd']:>9.2%}{m['ann']/m['sharpe'] if m['sharpe'] else float('nan'):>8.2%}{'—':>7}")

    def _row(label: str, st: dict, ir: float | None = None) -> None:
        out(f"  {label:<24}{st['total']:>+11.2%}{st['ann']:>+9.2%}"
            f"{st['sharpe']:>8.2f}{st['mdd']:>9.2%}{st['vol']:>8.2%}"
            f"{('—' if ir is None else f'{ir:+.2f}'):>7}")

    eqd = eq[eq.index >= REPORT_START]
    eqr = eqd.pct_change().dropna()
    bench = {
        "买入持有 510300 沪深300": buyhold(dfs, "SH.510300"),
        "买入持有 510050 上证50": buyhold(dfs, "SH.510050"),
        "买入持有 510500 中证500": buyhold(dfs, "SH.510500"),
        "买入持有 518880 黄金": buyhold(dfs, "SH.518880"),
        "买入持有 511010 国债": buyhold(dfs, "SH.511010"),
        "等权多资产（月度调回）": ew_multiasset(dfs, list(MENU)),
    }
    irs: dict[str, float] = {}
    for label, st in bench.items():
        sym = {"买入持有 510300 沪深300": "SH.510300",
               "买入持有 510050 上证50": "SH.510050",
               "买入持有 510500 中证500": "SH.510500",
               "买入持有 518880 黄金": "SH.518880",
               "买入持有 511010 国债": "SH.511010"}.get(label)
        ir = None
        if sym:
            d = dfs[sym]
            d = d[d["dt_ord"] >= REPORT_START]
            px = d.set_index("dt_ord")["close"].astype(float)
            br = px.pct_change().reindex(eqr.index).dropna()
            ex = eqr.reindex(br.index) - br
            if ex.std(ddof=1) > 1e-12:
                ir = float(ex.mean() / ex.std(ddof=1) * math.sqrt(252.0))
                irs[label] = ir
        _row(label, st, ir)
    bh = bench["买入持有 510300 沪深300"]
    ew = bench["等权多资产（月度调回）"]
    out(f"  → 相对沪深300ETF 年化超额 {m['ann']-bh['ann']:+.2%}")
    out(f"  → 相对等权多资产     年化超额 {m['ann']-ew['ann']:+.2%}"
        f"   Sharpe 差 {m['sharpe']-ew['sharpe']:+.2f}")

    hr("4. kill criteria")
    checks = [
        ("年化 ≥ 10%", m["ann"] >= 0.10, f"{m['ann']:+.2%}"),
        ("Sharpe ≥ 0.5", m["sharpe"] >= 0.5, f"{m['sharpe']:.2f}"),
        ("bootstrap Sharpe 下界 > 0", lo > 0, f"{lo:.2f}"),
        ("最大回撤 ≤ 25%", m["mdd"] <= 0.25, f"{m['mdd']:.2%}"),
        ("年化换手 ≤ 4×", m["turnover"] <= 4.0, f"{m['turnover']:.2f}×"),
        ("年化 > 买入持有510300", m["ann"] > bh["ann"],
         f"{m['ann']:+.2%} vs {bh['ann']:+.2%}"),
    ]
    for name, ok, val in checks:
        out(f"  [{'✓' if ok else '✗'}] {name:<26} {val}")
    out(f"  通过 {sum(1 for _, ok, _ in checks if ok)}/{len(checks)}")

    hr("5. 择时诊断")
    d = strat.diag
    if d:
        risky = np.array([x["n_risky"] for x in d])
        cashw = np.array([x["cash_w"] for x in d])
        sc = np.array([x["scale"] for x in d])
        out(f"  调仓次数 {len(d)}")
        out(f"  持有风险资产数: 均值 {risky.mean():.2f}  最小 {risky.min()}  最大 {risky.max()}")
        out(f"  现金腿权重: 均值 {cashw.mean():.2%}  中位 {np.median(cashw):.2%}  "
            f"最大 {cashw.max():.2%}")
        out(f"  波动率缩放: 均值 {sc.mean():.3f}  <1 的比例 {float((sc<0.999).mean()):.1%}")
        out(f"  空仓（无风险资产）次数 {int((risky==0).sum())}/{len(d)}")

    hr("6. 决定性检验：本策略 vs 等权多资产（配对块自举）")
    ews = ew_series(dfs, list(MENU))
    ew_al = ews.reindex(eqd.index).ffill()
    med_d, lo_d, hi_d = paired_bootstrap(eqd, ew_al)
    out(f"  Sharpe 差（策略 − 等权多资产）中位 {med_d:+.2f}"
        f"（5% {lo_d:+.2f} .. 95% {hi_d:+.2f}）")
    if lo_d > 0:
        out("  ✓ 差异显著为正 —— 择时/波动目标确实提升风险调整收益")
    elif hi_d < 0:
        out("  ✗ 差异显著为负 —— 组合比朴素等权更差")
    else:
        out("  ✗ 置信区间跨 0 —— **无法区分于朴素等权**，择时未证明有价值")
    out(f"  年化差 {m['ann'] - ew['ann']:+.2%}   "
        f"回撤差 {m['mdd'] - ew['mdd']:+.2%}   换手 {m['turnover']:.2f}× vs 0.49×")
    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("```\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
        print(f"报告已写入 {p}")
    out(f"\n总耗时 {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cash", type=float, default=200_000.0)
    ap.add_argument("--lookback", type=int, default=252)
    ap.add_argument("--ma-window", type=int, default=200)
    ap.add_argument("--vol-window", type=int, default=60)
    ap.add_argument("--target-vol", type=float, default=0.12)
    ap.add_argument("--rebalance", type=int, default=21)
    ap.add_argument("--weighting", default="equal")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--min-amount", type=float, default=2e7)
    ap.add_argument("--max-weight", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=0.00015)
    ap.add_argument("--slippage", type=float, default=0.0003)
    ap.add_argument("--no-trend", action="store_true")
    ap.add_argument("--no-vol-target", action="store_true")
    ap.add_argument("--md", default=None)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
