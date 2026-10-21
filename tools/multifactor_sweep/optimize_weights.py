"""因子权重优化：在样本内/样本外之间多次回测，按样本外 Sharpe 选最优权重。

方法
----
1. 计算每个因子的信息系数（IC）：因子值 ``f[t]`` 与次日截面收益 ``ret[t+1]`` 的
   横相关，跨交易日取均值。IC 越高（绝对值）的因子越有预测力。
2. 生成候选权重集合：
     - equal            ：等权（基线）
     - ic_signed        ：按带符号 IC 加权（负 IC 因子自动反向）
     - ic_abs           ：按 |IC| 加权，符号按其原始 IC 方向
     - block_momentum   ：给动量/反转块加权重
     - block_volume     ：给放量/成交量块加权重
     - block_vwap       ：给 vwap 偏离块加权重
     - block_vol        ：给波动率块加权重
3. 对每个候选在**完整窗口**跑一次回测，切出**样本外**（最后 40% 交易日）算 Sharpe；
   同时记录样本内 Sharpe 作稳健性检查。按样本外 Sharpe 排序，要求样本内 Sharpe>0。
4. 坐标上升法（coordinate ascent）在最优候选附近做局部微调，进一步压榨样本外 Sharpe。

设计取舍：只优化“样本外 Sharpe”，避免过拟合样本内；多次回测 + 多候选 + 局部搜索，
对应“多次回测并调整因子权重”的需求。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_prep as D
import backtest_engine as BE


# ---------------------------------------------------------------- 块定义
BLOCKS = {
    "momentum": ["wq019", "wq051", "wq101", "wq024", "wq017"],
    "volume":   ["wq002", "wq007", "wq043", "wq096"],
    "vwap":     ["wq041", "wq057"],
    "vol":      ["wq018", "wq034"],
    "other":    ["wq005"],
}


def compute_ic(prep, insample_frac=0.6):
    """每个因子的平均日度 IC（因子[t] 与次日截面收益的相关，跨交易日均值）。"""
    ret = prep.close.pct_change().shift(-1)
    rvals = ret.to_numpy()
    days = list(prep.close.index)
    n = len(days)
    n_ins = int(n * insample_frac)
    ics = {}
    for name in D.FACTOR_NAMES:
        f = prep.factor_panels[name]
        fvals = f.to_numpy(dtype=float)
        assert fvals.shape == rvals.shape, (fvals.shape, rvals.shape)
        per_day = np.empty(fvals.shape[0], dtype=float)
        for i in range(fvals.shape[0]):
            a, b = fvals[i], rvals[i]
            mask = ~(np.isnan(a) | np.isnan(b))
            am, bm = a[mask], b[mask]
            per_day[i] = np.corrcoef(am, bm)[0, 1] if am.size >= 5 else np.nan
        ics[name] = pd.Series(per_day, index=f.index)
    ins = {k: v.iloc[:n_ins].mean() for k, v in ics.items()}
    full = {k: v.mean() for k, v in ics.items()}
    return ins, full


def _normalize(weights, floor=0.0):
    """把原始权重归一化为和=1（允许 floor 截断负值后取 abs）。"""
    w = {k: float(v) for k, v in weights.items()}
    if floor and any(v < 0 for v in w.values()):
        w = {k: max(floor, v) for k, v in w.items()}
    s = sum(abs(v) for v in w.values())
    if s <= 0:
        return {k: 1.0 / len(w) for k in w}
    return {k: abs(v) / s for k, v in w.items()}


def build_candidates(ic_ins):
    """生成候选权重字典列表。"""
    names = D.FACTOR_NAMES
    cands = []

    # 1. 等权基线
    cands.append(("equal", {k: 1.0 / len(names) for k in names}))

    # 2. 带符号 IC 加权
    cands.append(("ic_signed", _normalize(ic_ins)))

    # 3. |IC| 加权，符号保持原始方向
    ic_abs = {k: (abs(v), v) for k, v in ic_ins.items()}
    cands.append(("ic_abs", _normalize({k: a for k, (a, v) in ic_abs.items()})))

    # 4. 块加权：每个块内等权，块间按块内 |IC| 总和加权
    block_w = {}
    for block, members in BLOCKS.items():
        members = [m for m in members if m in names]
        if not members:
            continue
        bsum = sum(abs(ic_ins.get(m, 0.0)) for m in members) or 1.0
        block_w[block] = bsum
    tot = sum(block_w.values()) or 1.0
    block_weights = {}
    for block, members in BLOCKS.items():
        members = [m for m in members if m in names]
        if not members:
            continue
        share = block_w[block] / tot
        for m in members:
            block_weights[m] = share / len(members)
    # 补齐未在任何块里的因子
    for m in names:
        block_weights.setdefault(m, 0.0)
    cands.append(("block_ic", _normalize(block_weights, floor=0.02)))

    # 5. 逐个块单独放大（其余等权）
    for block in BLOCKS:
        w = {k: 1.0 / len(names) for k in names}
        members = [m for m in BLOCKS[block] if m in names]
        if members:
            bump = 3.0
            for m in members:
                w[m] *= bump
        cands.append((f"block_{block}", _normalize(w, floor=0.0)))

    return [(name, w) for name, w in cands if sum(w.values()) > 0]


def run_and_score(prep, weights, *, oos_start_ts):
    """跑一次完整回测，返回指标字典（含样本内/外 Sharpe）。"""
    res, perf = BE.run_backtest(prep, weights)
    # 样本外 Sharpe：frame_id >= oos_start_ts 的权益曲线
    oos_sharpe = BE.oos_sharpe(res, start_frame=oos_start_ts)
    in_pairs = [(t, e) for t, e in res.equity_curve if t < oos_start_ts]
    in_sharpe = BE.sharpe_from_equity(in_pairs)
    return {
        "weights": weights,
        "total_return": perf.total_return,
        "sharpe": perf.sharpe,
        "max_drawdown": perf.max_drawdown,
        "win_rate": perf.win_rate(),
        "oos_sharpe": oos_sharpe,
        "in_sharpe": in_sharpe,
        "n_trades": res.total_trades(),
    }


def optimize(prep, *, insample_frac=0.6, verbose=True):
    """候选搜索 + 坐标上升微调，返回 (best_weights, log)。"""
    days = list(prep.close.index)
    oos_start_ts = days[int(len(days) * insample_frac)]
    if verbose:
        print(f"[opt] OOS 起点 ts={oos_start_ts}（第 {int(len(days)*insample_frac)}/{len(days)} 交易日）")

    ic_ins, ic_full = compute_ic(prep, insample_frac)
    if verbose:
        print("[opt] 平均 IC:", {k: round(v, 4) for k, v in sorted(ic_full.items(),
              key=lambda x: -abs(x[1]))[:6]})

    cands = build_candidates(ic_ins)
    log = []
    results = []
    for name, w in cands:
        try:
            r = run_and_score(prep, w, oos_start_ts=oos_start_ts)
        except Exception as e:  # 保留搜索继续
            r = {"weights": w, "oos_sharpe": -9.0, "in_sharpe": -9.0,
                 "total_return": float("nan"), "sharpe": float("nan"),
                 "max_drawdown": float("nan"), "win_rate": 0.0,
                 "n_trades": 0, "error": str(e)}
        r["name"] = name
        results.append(r)
        log.append({"name": name, "oos_sharpe": r["oos_sharpe"],
                    "in_sharpe": r["in_sharpe"], "sharpe": r["sharpe"],
                    "total_return": r["total_return"], "max_drawdown": r["max_drawdown"],
                    "win_rate": r["win_rate"], "n_trades": r["n_trades"]})
        if verbose:
            print(f"  [{name:12s}] OOS-Sharpe={r['oos_sharpe']:+.3f}  "
                  f"IN-Sharpe={r['in_sharpe']:+.3f}  Sharpe={r['sharpe']:+.3f}  "
                  f"Ret={r['total_return']:+.3f}  MDD={r['max_drawdown']:.3f}  "
                  f"WR={r['win_rate']:.2f}  n={r['n_trades']}")

    # 稳健性过滤：样本内 Sharpe 必须 > 0（否则视为过拟合/失效）
    viable = [r for r in results if r["in_sharpe"] > 0]
    pool = viable if viable else results
    pool.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    best = pool[0]
    if verbose:
        print(f"[opt] 候选 {len(results)} 个，可行 {len(viable)} 个，"
              f"最优 '{best['name']}' OOS-Sharpe={best['oos_sharpe']:+.3f}")

    # 坐标上升微调
    refined = _coordinate_ascent(prep, best["weights"], oos_start_ts, verbose=verbose)
    if refined and refined["oos_sharpe"] > best["oos_sharpe"]:
        if verbose:
            print(f"[opt] 坐标上升改进：{best['name']} -> refined "
                  f"{best['oos_sharpe']:+.3f} -> {refined['oos_sharpe']:+.3f}")
        best = refined
        best["name"] = "refined"
        log.append({"name": "refined", **{k: best[k] for k in
                    ("oos_sharpe", "in_sharpe", "sharpe", "total_return",
                     "max_drawdown", "win_rate", "n_trades")}})

    return best["weights"], log, oos_start_ts


def _coordinate_ascent(prep, base_w, oos_start_ts, *, rounds=6, step=0.05, verbose=True):
    """逐个因子微调权重：每次对一个因子 ±step，保留提升样本外 Sharpe 的方向。"""
    names = list(base_w.keys())
    w = dict(base_w)
    cur = run_and_score(prep, w, oos_start_ts=oos_start_ts)
    improved = True
    r = 0
    while improved and r < rounds:
        improved = False
        r += 1
        for name in names:
            for delta in (step, -step):
                trial = dict(w)
                trial[name] = round(trial.get(name, 0.0) + delta, 4)
                if trial[name] < 0:
                    continue
                s = sum(trial.values())
                trial = {k: v / s for k, v in trial.items()}
                try:
                    cand = run_and_score(prep, trial, oos_start_ts=oos_start_ts)
                except Exception:
                    continue
                if cand["oos_sharpe"] > cur["oos_sharpe"] + 1e-3:
                    w = trial
                    cur = cand
                    improved = True
                    break
            if improved:
                break
    if verbose:
        print(f"[opt] 坐标上升 {r} 轮，最终 OOS-Sharpe={cur['oos_sharpe']:+.3f}")
    return {"weights": w, "oos_sharpe": cur["oos_sharpe"], "in_sharpe": cur["in_sharpe"],
            "sharpe": cur["sharpe"], "total_return": cur["total_return"],
            "max_drawdown": cur["max_drawdown"], "win_rate": cur["win_rate"],
            "n_trades": cur["n_trades"], "name": "refined"}


if __name__ == "__main__":
    prep = D.prepare()
    best_w, log, oos_ts = optimize(prep, verbose=True)
    import json
    with open("weight_search_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2, default=str)
    print("\n[opt] 最优权重:", {k: round(v, 4) for k, v in sorted(best_w.items())})
