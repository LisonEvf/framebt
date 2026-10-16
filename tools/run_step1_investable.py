"""阶段 1 · 可投资池回测：8 因子自适应策略在「前 N 日均成交额」池上的样本外表现。

回答 PATH_200K.md 阶段 1 的问题：
  「这个 8 因子策略，在 20 万实际能买到的池子里（按日均成交额取前 N），
   滚动样本外到底赚不赚钱、风险调整后值不值得承担？」

关键设计（避免自我欺骗 + 避免慢）
------------------------------------
1. 可投资池 = 全历史「日均成交额取前 N」。用 duckdb 单条 SQL 0.5s 算出（不物化全面板）；
2. 因子在「可投资池」上计算 + 样本内（前 --train 日）筛选去冗余；
3. 样本外 = 剩余全部交易日（约 2008→2026），framebt 引擎在线学习实跑；
4. 基准 = 沪深300ETF(510300) / 中证500ETF(510500)（20 万真实机会成本）
   + 前 N 等权（stock-selection 策略的应有基准）；
5. 输出：分年度超额、Sharpe/回撤/换手、bootstrap 置信区间（5/50/95 分位）。

慢点规避（已缓解）：早期 available_symbols() 会读全部 5575 个 parquet 并 pivot，
    故本工具用 duckdb 单条 SQL 查池 + 只读池内文件构建长表（派生 Panel / OOS Bars）。
    数据源已改为 symbol→文件索引 + 按需单文件读取（available_symbols 现仅解析文件名，<0.05s），
    此规避退化为可选优化；保留 duckdb 是因为它可在不物化全面板的情况下算出可投资池。

用法
----
    python tools/run_step1_investable.py --top-n 500 --cash 200000
    python tools/run_step1_investable.py --top-n 300 --md reports/STEP1.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from framebt import (Engine, Performance, StandardAshareRules,
                     StandardCost, StandardRisk)
from framebt.core import Bar
from framebt.feed import BarFeed
from framebt.factors import (Panel, compute_many, dedup_by_correlation,
                             forward_returns, list_factors, log_market_cap,
                             preprocess, screen_pool)
from framebt.factors.lifecycle import FactorRegistry
from framebt.strategies.adaptive_multi_factor import AdaptiveMultiFactorStrategy

LINES: list[str] = []
_KLINE_DIR = Path(__file__).resolve().parents[2] / "opentdx" / "db" / "data" \
    / "fakt_kline" / "4" / "1"


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(t: str) -> None:
    out("")
    out("=" * 74)
    out(f"  {t}")
    out("=" * 74)


def top_n_universe(root: Path, top_n: int) -> list[str]:
    """duckdb 单查询：全历史日均成交额前 N 的 symbol_key 列表。"""
    import duckdb
    glob = str((root / "*.parquet").as_posix())
    conn = duckdb.connect()
    q = f"""
        WITH all_data AS (
            SELECT symbol_key, CAST(datetime AS DATE) AS d, CAST(amount AS DOUBLE) AS amount
            FROM read_parquet('{glob}')
            WHERE amount IS NOT NULL AND amount > 0
        ),
        avg_amt AS (
            SELECT symbol_key, AVG(amount) AS avg_amount FROM all_data GROUP BY symbol_key
        )
        SELECT symbol_key FROM avg_amt ORDER BY avg_amount DESC LIMIT {top_n}
    """
    return [r[0] for r in conn.execute(q).fetchall()]


def read_long(symbols: list[str], root: Path) -> pd.DataFrame:
    """只读池内文件，拼接为长表（一列一行一根 K 线）。

    文件名 = {market_value}_{code}.parquet（如 SH.600519 → 1_600519.parquet），
    与文件内 symbol_key（SH.600519）格式不同，需转换。
    """
    mkt_int = {"SZ": 0, "SH": 1, "BJ": 2}
    dfs = []
    for sym in symbols:
        mkt, code = sym.split(".")
        f = root / f"{mkt_int[mkt]}_{code}.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        df["symbol_key"] = sym
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["dt_ord"] = all_df["dt"].apply(lambda d: pd.Timestamp(d).toordinal())
    return all_df


def long_to_panel(all_df: pd.DataFrame) -> Panel:
    """长表 → Panel（index=ordinal, columns=symbol）。"""
    df = all_df.copy()
    df["vwap"] = df["amount"] / df["vol"].replace(0.0, np.nan)
    df["market_cap"] = df["close"] * df["float_shares"]
    src = {"open": "open", "high": "high", "low": "low", "close": "close",
           "volume": "vol", "amount": "amount", "vwap": "vwap",
           "market_cap": "market_cap", "turnover": "turnover",
           "float_shares": "float_shares"}
    out_cols = ["open", "high", "low", "close", "volume", "amount",
                "vwap", "market_cap", "turnover", "float_shares"]
    dics = {}
    for name in out_cols:
        col = src[name]
        if col not in df.columns:
            dics[name] = pd.DataFrame()
            continue
        piv = df.pivot_table(index="dt_ord", columns="symbol_key",
                             values=col, aggfunc="last").sort_index()
        dics[name] = piv
    idx = dics["close"].index
    for k, v in dics.items():
        if not v.empty:
            dics[k] = v.reindex(idx)
    return Panel(**dics)


def long_to_bars(all_df: pd.DataFrame, oos_start_ts: int) -> list[Bar]:
    """长表 → OOS 帧的 Bar 列表（dt_ord >= oos_start_ts）。"""
    sub = all_df[all_df["dt_ord"] >= oos_start_ts].sort_values("dt_ord")
    recs = sub.to_dict("records")
    bars: list[Bar] = []
    for r in recs:
        c = r["close"]
        if not (c > 0):
            continue
        vol = r["vol"] or 0.0
        amt = r["amount"] or 0.0
        fsh = r["float_shares"] or 0.0
        vwap = (amt / vol) if vol > 0 else c
        mcap = (c * fsh) if fsh > 0 else 0.0
        bars.append(Bar(
            timestamp=int(r["dt_ord"]), symbol=str(r["symbol_key"]),
            open=float(r["open"]), high=float(r["high"]), low=float(r["low"]),
            close=float(c), volume=float(vol), amount=float(amt),
            float_shares=float(fsh), turnover=float(r["turnover"] or 0.0),
            vwap=float(vwap), market_cap=float(mcap),
        ))
    return bars


def run_once(feed: BarFeed, strategy, cash: float):
    eng = Engine(
        feed, strategy, initial_cash=cash, settlement_cycle=1,
        cost_model=StandardCost(commission=0.0003, min_commission=5.0,
                                stamp_tax=0.001, transfer_fee=0.00002,
                                slippage="volume_based", k=0.001),
        risk=StandardRisk(max_position_pct=0.98),
        rules=StandardAshareRules(limit=0.10),
    )
    return eng.run()


def bootstrap_daily(excess_daily: pd.Series, n_boot: int = 2000) -> dict:
    x = excess_daily.dropna().to_numpy()
    if len(x) < 20:
        return {k: float("nan") for k in
                ["excess_ann", "excess_lo", "excess_hi",
                 "sharpe_ann", "sharpe_lo", "sharpe_hi"]}
    trading_days = 244
    ex_s = np.empty(n_boot)
    sh_s = np.empty(n_boot)
    base = pd.Series(x)
    for i in range(n_boot):
        s = base.sample(n=len(x), replace=True).to_numpy()
        mean = float(np.mean(s))
        sd = float(np.std(s))
        ex_s[i] = mean * trading_days
        sh_s[i] = (mean / sd) * np.sqrt(trading_days) if sd > 0 else 0.0
    q5, q50, q95 = np.percentile(ex_s, [5, 50, 95])
    s5, s50, s95 = np.percentile(sh_s, [5, 50, 95])
    return {"excess_ann": q50, "excess_lo": q5, "excess_hi": q95,
            "sharpe_ann": s50, "sharpe_lo": s5, "sharpe_hi": s95}


def main() -> None:
    ap = argparse.ArgumentParser(description="阶段1：可投资池滚动外测")
    ap.add_argument("--top-n", type=int, default=500)
    ap.add_argument("--cash", type=float, default=200_000.0)
    ap.add_argument("--train", type=int, default=800)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--md", default=None)
    ap.add_argument("--fixed-weights", action="store_true",
                    help="消融：固定等权因子权重，旁路在线GD/健康降权/生命周期/门控")
    args = ap.parse_args()

    t_start = time.time()
    t0 = time.time()
    investable = top_n_universe(_KLINE_DIR, args.top_n)
    out(f"阶段 1 · 可投资池回测（前 {args.top_n} 日均成交额）")
    out(f"可投资池: {len(investable)} 只（日均成交额前 {args.top_n}）  "
        f"池构建 {time.time() - t0:.1f}s")
    out(f"  抽样: {investable[:8]}")

    t0 = time.time()
    all_df = read_long(investable, _KLINE_DIR)
    out(f"池内 K 线: {len(all_df)} 行  ({time.time() - t0:.1f}s)")

    # 只取 2013 以后的 dense 面板：早期（1998–2012）稀疏，会触发 numpy masked-array
    # nanmedian 的越界 bug；且 20 万可投资池在 2013 后已完整，足够样本内筛选。
    cutoff = pd.Timestamp("2013-01-01").toordinal()
    all_df = all_df[all_df["dt_ord"] >= cutoff]

    t0 = time.time()
    panel = long_to_panel(all_df)
    ts = panel.timestamps
    out(f"面板: {panel.n_times} 交易日 × {len(panel.symbols)} 标的 "
        f"（{ts[0]}..{ts[-1]}）  构建 {time.time() - t0:.1f}s")

    # ---- 因子池 ----
    names = list_factors()
    t0 = time.time()
    raw = compute_many(names, panel)
    expo = [log_market_cap(panel)]
    pre = {n: preprocess(f, exposures=expo) for n, f in raw.items()}
    out(f"因子库 {len(names)} 个（在池上计算+预处理）  {time.time() - t0:.1f}s")

    # ---- 样本内筛选 + 去冗余 ----
    # 可投资池（前 500 日均成交额）多为近年上市标的，早期（1998–2005）面板稀疏，
    # 截面样本数不足 → 用「近期样本外」：2018-01-01 起池子已完整。
    oos_date = pd.Timestamp("2018-01-01")
    oos_idx = min(range(len(ts)), key=lambda i: abs(ts[i] - oos_date.toordinal()))
    oos_start_ts = ts[oos_idx]
    train_end = max(0, oos_idx - args.train)
    train_idx = ts[:train_end]
    out(f"样本外起点 {oos_start_ts}（距 {oos_date} 最近交易日），"
        f"样本内 {train_end} 交易日（{ts[0]}..{oos_start_ts}）")
    fwd = forward_returns(panel.close, 1)
    rep = screen_pool({n: pre[n].loc[train_idx] for n in names},
                      fwd.loc[train_idx], n_groups=5)
    passed = rep[rep["passed"]]["factor"].tolist()
    kept, corr, dropped = dedup_by_correlation(
        {n: pre[n].loc[train_idx] for n in passed},
        dict(zip(rep["factor"], rep["icir"])), threshold=args.corr_threshold)
    out(f"样本内筛选去冗余后保留 {len(kept)} 因子: {kept}")
    if not kept:
        out("  筛选失败明细（ic_mean/t/单调性/样本数/reject）：")
        for r in rep.itertuples(index=False):
            out(f"    {r.factor:<9} IC={r.ic_mean:+.4f} t={r.t_stat:+.2f} "
                f"mono={r.monotonicity:+.2f} n={r.n}  {r.reject_reason}")
        raise SystemExit("没有因子通过筛选")

    # ---- 样本外回测 ----
    bars_oos = long_to_bars(all_df, oos_start_ts)
    feed_oos = BarFeed(bars_oos)
    out(f"样本外区间: {len(bars_oos)} bars（{oos_start_ts}..{ts[-1]}）")

    fp = {n: pre[n] for n in kept}
    registry = FactorRegistry(observe_days=40, retire_days=40, rehab_days=80)
    for n in kept:
        registry.register(n, expression=f"library:{n}", source="manual",
                          birth_date=int(oos_start_ts))
    strategy = AdaptiveMultiFactorStrategy(
        investable, fp, panel.close, turnover_panel=panel.turnover,
        top_k=args.top_k, rebalance_every=args.rebalance,
        warmup=60, mode="long", gross=1.0,
        lr=0.002, lambda_turnover=0.1, lambda_l1=0.01, discount=0.95, buffer=20,
        use_gating=True, use_health=True, health_every=20,
        buffer_mult=2.0, min_order_pct=0.01, registry=registry,
    )
    if args.fixed_weights:
        # 消融对照：因子权重锁死为等权 1/K，绕开 ①GD ②健康 ③生命周期 ④门控。
        # 优化器仍照常 observe（数据管线不变），只是其权重不再进入 composite_score。
        _eqw = 1.0 / len(kept)
        strategy._current_weights = lambda _ts: {n: _eqw for n in kept}
        out(f"  [消融模式] 因子权重固定等权 {_eqw:.4f}，"
            f"旁路 GD/健康/生命周期/门控")
    t0 = time.time()
    res = run_once(feed_oos, strategy, args.cash)
    dt = time.time() - t0
    perf = Performance(res)
    out(f"回测耗时 {dt:.1f}s   帧数 {res.frames}   成交 {res.total_trades()} 笔")
    out("")
    out(f"  最终权益   : {res.final_equity():,.2f}")
    out(f"  总收益     : {perf.total_return:+.2%}")
    out(f"  年化收益   : {perf.annualized_return:+.2%}")
    out(f"  最大回撤   : {perf.max_drawdown:.2%}")
    out(f"  Sharpe     : {perf.sharpe:.2f}")
    out(f"  胜率       : {perf.win_rate():.0%}")

    # ---- 诊断：权益曲线 ----
    eq_df = pd.DataFrame(res.equity_curve, columns=["ts", "eq"])
    eq_df = eq_df[eq_df["ts"].astype(int) >= 1]
    eq_df["dt"] = pd.to_datetime(
        eq_df["ts"].map(lambda t: pd.Timestamp.fromordinal(int(t))))
    eq_df["year"] = eq_df["dt"].dt.year
    out("")
    out("  [诊断] 年末权益 / 年内回撤:")
    for y in sorted(eq_df["year"].unique()):
        sub = eq_df[eq_df["year"] == y]
        hi = sub["eq"].max()
        dd = hi - sub["eq"].min()
        out(f"    {y}: 初 {sub['eq'].iloc[0]:,.0f} 末 {sub['eq'].iloc[-1]:,.0f}  "
            f"峰值 {hi:,.0f} 年内回撤 {dd:,.0f} ({dd / hi:.0%})")
    # 最大回撤段（峰值→谷底）
    cummax = eq_df["eq"].cummax()
    dd_seq = cummax - eq_df["eq"]
    m = dd_seq.idxmax()
    peak_i = eq_df.loc[:m, "eq"].idxmax()
    out(f"    最大回撤 {perf.max_drawdown:.0%}: 峰值 {eq_df.loc[peak_i,'eq']:,.0f} "
        f"({eq_df.loc[peak_i,'dt'].date()}) → 谷底 {eq_df.loc[m,'eq']:,.0f} "
        f"({eq_df.loc[m,'dt'].date()})")
    out(f"    持仓最小时期: 权益 {eq_df['eq'].min():,.0f} @ "
        f"{eq_df.loc[eq_df['eq'].idxmin(),'dt'].date()}")

    # ---- 基准：ETF + 池等权 ----
    hr("5. 基准对比（同样本外区间）")
    from opentdx import TdxClient
    from opentdx.const import ADJUST, MARKET, PERIOD
    etf_daily = {}
    with TdxClient() as client:
        for code, name in [("510300", "沪深300ETF"), ("510500", "中证500ETF")]:
            bars = client.stock_kline(MARKET.SH, code, PERIOD.DAILY,
                                      start=0, count=700, adjust=ADJUST.QFQ)
            all_b = list(bars)
            s = 700
            while len(all_b) < 12000:
                more = client.stock_kline(MARKET.SH, code, PERIOD.DAILY,
                                          start=s, count=700, adjust=ADJUST.QFQ)
                if not more:
                    break
                all_b.extend(more)
                s += len(more)
                if len(more) < 700:
                    break
            df = pd.DataFrame(all_b).sort_values("datetime")
            dts = pd.to_datetime(df["datetime"]).to_numpy()
            cs = df["close"].to_numpy(dtype=float)
            start = pd.Timestamp.fromordinal(oos_start_ts)
            end = pd.Timestamp.fromordinal(ts[-1])
            m = (dts >= start.to_pydatetime()) & (dts <= end.to_pydatetime())
            etf_daily[name] = pd.Series(cs[m], index=pd.to_datetime(dts[m]))

    # 池等权：各日横截面收益均值（≥20 样本），再连乘
    ret = panel.close.pct_change().replace([np.inf, -np.inf], np.nan).reindex(ts)
    oos_ret = ret[ret.index >= oos_start_ts]
    valid = oos_ret[oos_ret.notna().sum(axis=1) >= 20]
    bench_ew_ret = valid.mean(axis=1, skipna=True).dropna()
    bench_ew_total = float((1 + bench_ew_ret).prod() - 1)   # 原始总收益
    # 超额计算：过滤 -100% 日 + winsorize 极端日（避免 bootstrap 溢出）
    bench_ew_ret = bench_ew_ret[bench_ew_ret > -0.99]
    lo, hi = bench_ew_ret.quantile(0.01), bench_ew_ret.quantile(0.99)
    bench_ew_ret = bench_ew_ret.clip(lo, hi)

    out(f"  池等权（{len(investable)} 标的）: 样本外 {bench_ew_total:+.2%}")
    for name, s in etf_daily.items():
        out(f"  {name:<12}: 样本外 {float(s.iloc[-1] / s.iloc[0] - 1):+.2%}")

    # ---- 策略日收益 ----
    eq = pd.Series([e for _k, e in res.equity_curve],
                   index=[pd.Timestamp.fromordinal(int(t)).to_pydatetime()
                          for t, _e in res.equity_curve])
    strat_daily = eq.pct_change().dropna()
    # 策略日收益含极端离群（5 只持仓单日巨震），clip 后年度超额才可读
    lo, hi = strat_daily.quantile(0.01), strat_daily.quantile(0.99)
    strat_daily = strat_daily.clip(lo, hi)

    # ---- 分年度超额 vs 沪深300ETF + bootstrap ----
    hr("6. 分年度超额（策略 − 沪深300ETF）与 bootstrap 置信区间")
    for name, s in etf_daily.items():
        common = strat_daily.index.intersection(s.index)
        ex = (strat_daily[common] - s[common].pct_change()).dropna()
        grp = ex.groupby(ex.index.year).sum()
        out(f"  ── vs {name} ──")
        out(f"    {'年份':<8} {'年度超额':>10} {'累计超额':>10}")
        cum = 0.0
        for y, v in grp.items():
            cum += float(v)
            out(f"    {int(y):<8} {float(v):+10.2%} {cum:+10.2%}")
        bs = bootstrap_daily(ex)
        out(f"    超额年化 50 分位 = {bs['excess_ann']:+.2%}  "
            f"(5% {bs['excess_lo']:+.2%} .. 95% {bs['excess_hi']:+.2%})")
        out(f"    Sharpe  50 分位 = {bs['sharpe_ann']:.2f}  "
            f"(5% {bs['sharpe_lo']:.2f} .. 95% {bs['sharpe_hi']:.2f})")
        if not np.isnan(bs["sharpe_lo"]) and bs["sharpe_lo"] > 0:
            out(f"    ✓ bootstrap 下界 Sharpe>0 —— 超额统计显著（非运气）")
        elif not np.isnan(bs["sharpe_lo"]):
            out(f"    ✗ bootstrap 下界 Sharpe<=0 —— 超额不显著，可能是过拟合/运气")

    # ---- 分年度超额 vs 池等权 ----
    hr("7. 分年度超额（策略 − 池等权）")
    # bench_ew_ret 索引为 ordinal（panel.timestamps），strat_daily 为日期 → 对齐到 ordinal
    strat_daily_ord = pd.Series(strat_daily.values,
                                index=[int(t.toordinal()) for t in strat_daily.index])
    common2 = strat_daily_ord.index.intersection(bench_ew_ret.index)
    ex2 = (strat_daily_ord[common2] - bench_ew_ret[common2]).dropna()
    grp2 = ex2.groupby(
        [pd.Timestamp.fromordinal(int(t)).year for t in ex2.index]).sum()
    out(f"    {'年份':<8} {'年度超额':>10} {'累计超额':>10}")
    cum2 = 0.0
    for y, v in grp2.items():
        cum2 += float(v)
        out(f"    {int(y):<8} {float(v):+10.2%} {cum2:+10.2%}")
    bs2 = bootstrap_daily(ex2)
    out(f"    超额年化 50 分位 = {bs2['excess_ann']:+.2%}  "
        f"(5% {bs2['excess_lo']:+.2%} .. 95% {bs2['excess_hi']:+.2%})")

    hr("8. 换手与成本")
    traded = sum(f.size * f.price for f in res.fills)
    eqs = [e for _k, e in res.equity_curve]
    avg_eq = float(np.mean(eqs)) if eqs else args.cash
    years = res.frames / 244.0
    out(f"  累计成交额 {traded:,.0f}（{traded / avg_eq:.1f}× 平均权益）")
    out(f"  年化单边换手约 {traded / avg_eq / max(years, 1e-9) / 2:.1f}×")
    fees = sum(f.cost for f in res.fills)
    out(f"  费用合计 {fees:,.0f}（{fees / args.cash:.2%} 占本金）")

    # ---- 9. 自适应层诊断：动态因子到底动了没有 ----
    hr("9. 自适应层诊断（设计里的动态因子，实际轨迹）")
    eq_w = 1.0 / len(kept)
    try:
        rw = strategy.weights_report()
        out(f"  在线 GD 优化器: 步数 {rw['steps']}   末帧 IC {rw['ic']:+.4f}   "
            f"ICIR {rw['icir']:+.3f}")
        out(f"  优化器最终权重（L1 归一到 1；等权基准 = {eq_w:.3f}）:")
        for n, wv in sorted(rw["weights"].items(), key=lambda t: -abs(t[1])):
            out(f"    {n:<9} {wv:+.4f}   （偏离等权 {wv - eq_w:+.4f}）")
    except Exception as exc:
        out(f"  [优化器诊断失败] {type(exc).__name__}: {exc}")

    try:
        hist = getattr(strategy.optimizer, "history", [])
        if hist:
            W = np.array([h.weights for h in hist], dtype=float)
            out(f"  权重轨迹 {len(W)} 步（首帧 / 末帧 / 极值 / 波动）:")
            out(f"    {'因子':<9} {'首帧':>9} {'末帧':>9} {'最小':>9} {'最大':>9} {'std':>8}")
            for i, n in enumerate(kept):
                out(f"    {n:<9} {W[0, i]:+9.4f} {W[-1, i]:+9.4f} "
                    f"{W[:, i].min():+9.4f} {W[:, i].max():+9.4f} {W[:, i].std():8.4f}")
            spread = W.max(axis=1) - W.min(axis=1)
            out(f"    权重极差 max-min: 首帧 {spread[0]:.4f}  末帧 {spread[-1]:.4f}  "
                f"全程均值 {spread.mean():.4f}  历史最大 {spread.max():.4f}")
            out("    等权时极差 = 0；极差全程贴着 0 说明自适应层在空转")
    except Exception as exc:
        out(f"  [权重轨迹诊断失败] {type(exc).__name__}: {exc}")

    if getattr(strategy, "registry", None) is not None:
        try:
            s = strategy.registry.summary()
            out(f"  因子生命周期: 共 {s['total']}  active {s['active']}  "
                f"observing {s['observing']}  retired {s['retired']}")
            ev = strategy.registry.events
            out(f"    状态迁移事件 {len(ev)} 条"
                + (f"（前 3: {ev[:3]}）" if ev else " —— 全程零迁移"))
            if s["observing"] == s["total"] and s["total"] > 0:
                out("    ⚠ 全部滞留 observing → cap_weights 无 free 因子可分配，")
                out("      退化为纯重归一，设计的 5% 权重上限完全没生效")
        except Exception as exc:
            out(f"  [生命周期诊断失败] {type(exc).__name__}: {exc}")

    if getattr(strategy, "gating", None) is not None:
        try:
            g = np.asarray(strategy.gating.gates(strategy._state_vector(ts[-1])),
                           dtype=float)
            out(f"  市场状态门控: 末帧 gate（等权基准 = {eq_w:.3f}，训练步数 "
                f"{getattr(strategy.gating, 'steps', '?')}）")
            for n, gv in zip(kept, g):
                out(f"    {n:<9} {gv:.4f}   （偏离等权 {gv - eq_w:+.4f}）")
        except Exception as exc:
            out(f"  [门控诊断失败] {type(exc).__name__}: {exc}")

    try:
        fin = strategy._current_weights(ts[-1])
        out("  最终生效权重（GD → 健康降权 → 生命周期 → 门控，四层叠加后真正决定持仓）:")
        for n, wv in sorted(fin.items(), key=lambda t: -abs(t[1])):
            out(f"    {n:<9} {wv:+.4f}   （偏离等权 {wv - eq_w:+.4f}）")
    except Exception as exc:
        out(f"  [生效权重诊断失败] {type(exc).__name__}: {exc}")

    out("")
    out(f"总耗时 {time.time() - t_start:.1f}s")

    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("```\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
        print(f"\n报告已写入 {p}")


if __name__ == "__main__":
    main()
