"""自适应多因子系统：在真实数据上的完整验证。

方法论（避免自我欺骗）
----------------------
1. **样本内选因子**：只用前 ``--train`` 个交易日做因子筛选与去冗余；
2. **样本外回测**：在其余区间上用 framebt 引擎跑自适应策略（在线学习 + 门控 + 健康监控）；
3. **因果审计**：抽样验证「预计算因子面板的第 t 行」== 「只用 ≤t 数据重算的第 t 行」，
   证明预计算没有前视；
4. **基准对比**：等权买入持有、静态等权因子组合（不学习）。

用法：
    python tools/validate_adaptive.py
    python tools/validate_adaptive.py --train 800 --top-k 5 --md reports/ADAPTIVE.md

注意：OOS 回测每帧对截面矩阵做 numpy 矩阵运算，本机 OpenBLAS 多线程在较大
截面（~500 标）上会触发访问越界崩溃（0xC0000005）。如遇崩溃，先在外部设
置单线程再跑：$env:OMP_NUM_THREADS="1"（macOS/Linux 为 export OMP_NUM_THREADS=1）。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from framebt import (BarFeed, Engine, Performance, StandardAshareRules,
                     StandardCost, StandardRisk)
from framebt.data_sources import OpentdxParquetSource
from framebt.factors import (Panel, compute_many, dedup_by_correlation,
                             forward_returns, list_factors, log_market_cap,
                             preprocess, screen_pool)
from framebt.factors.lifecycle import FactorRegistry
from framebt.strategies.adaptive_multi_factor import AdaptiveMultiFactorStrategy

LINES: list[str] = []


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(t: str) -> None:
    out("")
    out("=" * 74)
    out(f"  {t}")
    out("=" * 74)


# ---------------------------------------------------------------------------
def causality_audit(panel: Panel, names: list[str], pre: dict, samples: int = 4) -> bool:
    """因果审计：预计算面板的第 t 行 == 只用 ≤t 数据重算的第 t 行。"""
    expo_fn = lambda p: [log_market_cap(p)]  # noqa: E731
    ts_list = panel.timestamps
    idxs = np.linspace(len(ts_list) // 2, len(ts_list) - 2, samples).astype(int)
    ok = True
    for i in idxs:
        t = ts_list[i]
        sub = panel.slice_upto(t)
        raw_sub = compute_many(names, sub)
        for n in names:
            a = pre[n].loc[t]
            b = preprocess(raw_sub[n], exposures=expo_fn(sub)).loc[t]
            both = a.notna() & b.notna()
            if both.sum() == 0:
                continue
            if not np.allclose(a[both].to_numpy(float), b[both].to_numpy(float),
                               rtol=1e-9, atol=1e-9, equal_nan=True):
                out(f"  ✗ 因果审计失败: factor={n} ts={t}")
                ok = False
            elif bool(a.notna().sum() != b.notna().sum()):
                out(f"  ✗ 因果审计缺失结构不一致: factor={n} ts={t}")
                ok = False
    return ok


def run_backtest(feed: BarFeed, strategy, cash: float, with_cost: bool = True):
    eng = Engine(
        feed, strategy, initial_cash=cash, settlement_cycle=1,
        cost_model=StandardCost(commission=0.0003, min_commission=5.0,
                                stamp_tax=0.001, transfer_fee=0.00002,
                                slippage="volume_based", k=0.001) if with_cost else None,
        risk=StandardRisk(max_position_pct=0.98),
        rules=StandardAshareRules(limit=0.10),
    )
    return eng, eng.run()


def main() -> None:
    ap = argparse.ArgumentParser(description="自适应多因子系统验证")
    ap.add_argument("--train", type=int, default=800, help="样本内（选因子）交易日数")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--rebalance", type=int, default=5)
    ap.add_argument("--cash", type=float, default=1_000_000.0)
    ap.add_argument("--corr-threshold", type=float, default=0.7)
    ap.add_argument("--md", default=None)
    ap.add_argument("--fixed-weights", action="store_true",
                    help="消融：固定等权因子权重，旁路在线GD/健康降权/生命周期/门控")
    ap.add_argument("--symbols", default=None,
                    help="限定标的代码（逗号分隔），在读取前按文件名过滤，避免加载全市场")
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="因子池与样本外回测用的代表样本上限（缺省=全市场）。"
                         "全市场 5575 标的的 57 因子池计算约 30min，"
                         "降采样可大幅提速且统计量稳定。")
    args = ap.parse_args()

    t_start = time.time()
    src = OpentdxParquetSource()
    codes = [c.strip() for c in args.symbols.split(",") if c.strip()] if args.symbols else []
    syms = src.available_symbols()
    if codes:
        # load() 已改为按标的单文件读取（秒级），此处只需按代码过滤 symbol 列表即可，
        # 不再需要过滤 src._files 预读。symbol_key 形如 SH.600519，取点后缀匹配代码。
        syms = [s for s in syms if any(c == s.split(".")[-1] or c == s for c in codes)]
        out(f"  [限定标的] {len(codes)} 个代码 → 匹配 {len(syms)} 个标的")
    if args.max_symbols:
        # 真正截断：因子池与样本外回测用代表样本（否则仍是全市场 5575，57 因子池 ~30min）。
        if len(syms) > args.max_symbols:
            out(f"  [--max-symbols] {len(syms)} → {args.max_symbols} 标代表样本")
        syms = syms[:args.max_symbols]
    feed_full = src.universe(syms)
    panel = Panel.from_feed(feed_full)
    names = list_factors()

    out("自适应多因子系统 · 真实数据验证")
    out(f"数据源 : opentdx 数仓 · {len(syms)} 标的 · DAILY/QFQ")
    out(f"样本   : {panel.n_times} 交易日（{panel.timestamps[0]}..{panel.timestamps[-1]}）")

    # ---- 1. 因子池：计算 + 预处理（逐行因果）----
    hr("1. 因子池构建（§2）")
    t0 = time.time()
    raw = compute_many(names, panel)
    expo = [log_market_cap(panel)]
    pre = {n: preprocess(f, exposures=expo) for n, f in raw.items()}
    out(f"因子库 {len(names)} 个；计算+预处理耗时 {time.time() - t0:.1f}s")

    # ---- 2. 样本内筛选 + 去冗余（§2.3 / §2.2）----
    ts = panel.timestamps
    train_end = min(args.train, len(ts) - 200)
    train_idx = ts[:train_end]
    fwd = forward_returns(panel.close, 1)

    hr("2. 样本内因子筛选与去冗余（§2.3 / §2.2）")
    out(f"样本内区间: 前 {train_end} 日（{train_idx[0]}..{train_idx[-1]}）")
    t0 = time.time()
    rep = screen_pool({n: pre[n].loc[train_idx] for n in names},
                      fwd.loc[train_idx], n_groups=5)
    out(f"通过 IC/t/单调性/样本量 门槛: {int(rep['passed'].sum())}/{len(rep)}  （{time.time() - t0:.1f}s）")
    out("  通过清单（按 |ICIR| 排序）：")
    for r in rep[rep["passed"]].itertuples():
        out(f"    {r.factor:<9} IC={r.ic_mean:+.4f}  ICIR={r.icir:+.4f}  "
            f"t={r.t_stat:+.2f}  单调={r.monotonicity:+.1f}  spread={r.spread:+.2%}")

    passed = rep[rep["passed"]]["factor"].tolist()
    scores = dict(zip(rep["factor"], rep["icir"]))
    kept, corr, dropped = dedup_by_correlation(
        {n: pre[n].loc[train_idx] for n in passed}, scores, threshold=args.corr_threshold)
    out(f"  去冗余（|corr|>{args.corr_threshold}）后保留 {len(kept)} 个: {kept}")
    for keeper, loser, c in dropped:
        out(f"    剔除 {loser:<9} ← 与 {keeper} 相关 {c:+.3f}")

    if not kept:
        raise SystemExit("没有因子通过筛选，请放宽阈值或检查数据")

    # ---- 3. 因果审计 ----
    hr("3. 因果审计（预计算面板 vs 截断重算）")
    t0 = time.time()
    ok = causality_audit(panel, kept, pre)
    out(f"抽样 {4} 个时间点 × {len(kept)} 因子："
        f"{'✓ 逐位一致（无前视）' if ok else '✗ 存在不一致'}")
    out(f"耗时 {time.time() - t0:.1f}s")

    # ---- 4. 样本外回测 ----
    hr("4. 样本外回测（framebt 引擎 · 在线学习 + 门控 + 健康监控）")
    oos_start_ts = ts[train_end]
    feed_oos = BarFeed([b for b in feed_full.bars if b.timestamp >= oos_start_ts])
    out(f"样本外区间: {len(feed_oos.bars)} bars（{oos_start_ts}..{ts[-1]}）")

    fp = {n: pre[n] for n in kept}
    registry = FactorRegistry(observe_days=40, retire_days=40, rehab_days=80)
    # 因子入栈（§7.1）：新因子以 observing 状态进入，初始权重受上限约束
    for n in kept:
        registry.register(n, expression=f"library:{n}", source="manual",
                          birth_date=int(oos_start_ts))

    strategy = AdaptiveMultiFactorStrategy(
        syms, fp, panel.close, turnover_panel=panel.turnover,
        top_k=args.top_k, rebalance_every=args.rebalance,
        warmup=60, mode="long", gross=1.0,
        lr=0.002, lambda_turnover=0.1, lambda_l1=0.01, discount=0.95, buffer=20,
        use_gating=True, use_health=True, health_every=20,
        buffer_mult=2.0, min_order_pct=0.01,
        registry=registry,
    )
    if args.fixed_weights:
        # 消融对照：因子权重锁死为等权 1/K，绕开 ①GD ②健康 ③生命周期 ④门控。
        _eqw = 1.0 / len(kept)
        strategy._current_weights = lambda _ts: {n: _eqw for n in kept}
        out(f"  [消融模式] 因子权重固定等权 {_eqw:.4f}，"
            f"旁路 GD/健康/生命周期/门控")
    t0 = time.time()
    eng, res = run_backtest(feed_oos, strategy, args.cash)
    dt = time.time() - t0
    perf = Performance(res)
    fees = sum(f.cost for f in res.fills)
    out(f"回测耗时 {dt:.1f}s   帧数 {res.frames}   成交 {res.total_trades()} 笔   费用 {fees:,.0f}")
    # 换手与成本
    traded = sum(f.size * f.price for f in res.fills)
    eq = [e for _k, e in res.equity_curve]
    avg_eq = float(np.mean(eq)) if eq else args.cash
    years = res.frames / 244.0
    out(f"  累计成交额 {traded:,.0f}（{traded / avg_eq:.1f}× 平均权益）"
        f"  → 年化单边换手约 {traded / avg_eq / max(years, 1e-9) / 2:.1f}×")
    out(f"  费用占初始资金 {fees / args.cash:.2%}")
    out("")
    out(f"  最终权益   : {res.final_equity():,.2f}")
    out(f"  总收益     : {perf.total_return:+.2%}")
    out(f"  年化收益   : {perf.annualized_return:+.2%}")
    out(f"  最大回撤   : {perf.max_drawdown:.2%}")
    out(f"  Sharpe     : {perf.sharpe:.2f}")
    out(f"  胜率       : {perf.win_rate():.0%}")

    # ---- 5. 基准对比 ----
    hr("5. 基准对比（同样本外区间）")
    bars_oos = [b.timestamp for b in feed_oos.bars]
    bh = []
    for s in syms:
        c = panel.close[s].reindex(bars_oos).dropna()
        if len(c) > 2:
            bh.append(c.iloc[-1] / c.iloc[0] - 1.0)
    out(f"  等权买入持有        : {np.mean(bh):+.2%}  （{len(bh)} 标的）")
    out(f"  自适应多因子（本策略）: {perf.total_return:+.2%}")
    out(f"  超额                : {perf.total_return - np.mean(bh):+.2%}")

    # ---- 6. 权重与健康度 ----
    hr("6. 在线学习结果与因子健康度")
    wr = strategy.weights_report()
    ics = [v for lst in strategy._ic_history.values() for v in lst]
    out(f"在线更新步数: {wr['steps']}   滚动 ICIR={wr['icir']:+.4f}   "
        f"累计已实现 IC 均值={np.mean(ics):+.4f}（{len(ics)} 个因子-日样本）")
    out("  权重 Top10（学习所得）：")
    for n, w in wr["top"]:
        out(f"    {n:<9} {w:+.4f}")

    dash = strategy.health_dashboard()
    if not dash.empty:
        out("")
        out("  健康度看板（§5.1）：")
        cols = ["factor", "status", "ic_mean", "icir", "fsc", "crowding",
                "decay_prob", "downgrade", "auto"]
        out("    " + "".join(f"{c:>11}" for c in cols))
        for r in dash[cols].itertuples(index=False):
            cells = []
            for c, v in zip(cols, r):
                cells.append(f"{v:>11.3f}" if isinstance(v, (int, float)) else f"{str(v):>11}")
            out("    " + "".join(cells))

    if strategy.registry is not None:
        out("")
        out(f"  因子生命周期（§7.1）: {strategy.registry.summary()}")

    out("")
    out(f"总耗时 {time.time() - t_start:.1f}s")

    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("```\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
        print(f"\n报告已写入 {p}")


if __name__ == "__main__":
    main()
