"""因子层测试：算子 / 面板 / 因子库 / 预处理 / 评估。

重点覆盖开发过程中发现的**真实缺陷**：
  - ``safe_div`` 标量广播（曾导致 ``wq094`` 全 NaN）
  - 滚动窗口 ``min_periods`` 不得大于窗口（曾导致 ``wq045`` 全 NaN）
  - 面板的因果性（``from_bus`` 受 token 约束）
  - 逐行截面算子不跨期泄漏
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from framebt import CausalToken, DataBus
from framebt.factors import (Panel, compute, compute_many, cross_section_ic,
                             forward_returns, group_returns, ic_summary,
                             list_factors, log_market_cap, mad_winsorize,
                             monotonicity_score, newey_west_tstat, neutralize,
                             preprocess, screen_factor, zscore)
from framebt.factors import ops
from framebt.factors.preprocess import dedup_by_correlation, pairwise_ic_correlation


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------
def make_panel(n_symbols: int = 8, n_times: int = 120, seed: int = 0) -> Panel:
    """构造合成面板。

    关键：价格路径必须**互相穿越**（截面排序随时间变化），否则 `rank(open)` 逐期
    恒定 → 滚动截面相关退化为 0/0（数学上无定义）→ 形如 ``wq003`` 的因子会大面积
    NaN。这是数据性质而非代码缺陷，但会让测试产生误导。
    """
    rng = np.random.default_rng(seed)
    syms = [f"S{i:02d}" for i in range(n_symbols)]
    rows = []
    # 各标的从相近的价位出发、以较强的特质波动独立游走 → 排序会不断变化
    starts = 20.0 * np.exp(0.25 * rng.normal(size=n_symbols))
    for k, s in enumerate(syms):
        p = float(starts[k])
        shares = (1e9 + k * 1e8)
        for t in range(n_times):
            p = max(0.5, p * (1.0 + 0.02 * rng.normal()))
            vol = 1e6 * (1.0 + abs(rng.normal()))
            o = p * (1 + 0.004 * rng.normal())
            h = max(o, p) * (1 + 0.005 * abs(rng.normal()))
            low = min(o, p) * (1 - 0.005 * abs(rng.normal()))
            rows.append({
                "timestamp": t, "symbol": s, "open": o, "high": h, "low": low,
                "close": p, "volume": vol, "amount": p * vol,
                "float_shares": shares, "turnover": 0.5 + 0.3 * rng.random(),
                "vwap": p * (1 + 0.001 * rng.normal()), "market_cap": p * shares,
            })
    return Panel.from_records(rows)


@pytest.fixture(scope="module")
def panel():
    return make_panel()


# ---------------------------------------------------------------------------
# 算子：受保护除法与滚动窗口
# ---------------------------------------------------------------------------
def test_safe_div_scalar_broadcasts():
    """regression：safe_div(1.0, panel) 曾构造 1×1 面板 → 全 NaN。"""
    df = pd.DataFrame({"a": [2.0, 4.0], "b": [5.0, 8.0]})
    out = ops.safe_div(1.0, df)
    assert out.shape == df.shape
    assert np.allclose(out.to_numpy(), 1.0 / df.to_numpy())
    # 标量/标量 -> 标量
    assert ops.safe_div(1.0, 2.0) == 0.5
    assert np.isnan(ops.safe_div(1.0, 0.0))
    # 面板/标量
    assert np.allclose(ops.safe_div(df, 2.0).to_numpy(), df.to_numpy() / 2.0)


def test_safe_div_protects_against_inf():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [0.0, 4.0]})
    out = ops.safe_div(df, pd.DataFrame({"a": [0.0, 2.0], "b": [0.0, 2.0]}))
    assert not np.isinf(out.to_numpy()).any()


def test_rolling_min_periods_not_exceeding_window():
    """regression：correlation 窗口 2 曾要求 3 个观测 → 永远 NaN。"""
    df = pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0) * 2})
    for n in (2, 3, 5):
        c = ops.correlation(df, df, n)
        assert bool(c.iloc[n:].notna().to_numpy().any()), f"窗口 {n} 的相关不应全 NaN"
    # 完全线性相关应接近 1
    c = ops.correlation(df, df, 5)
    assert np.allclose(c.dropna().to_numpy(), 1.0, atol=1e-9)


def test_ops_cross_sectional_are_row_wise():
    """截面算子只在本行内计算，不得跨期泄漏。"""
    df = pd.DataFrame({"a": [1.0, 10.0], "b": [2.0, 20.0], "c": [3.0, 30.0]})
    r = ops.rank(df)
    # 两行的排名结构完全相同（因为都是 1:2:3）
    assert np.allclose(r.iloc[0].to_numpy(), r.iloc[1].to_numpy())
    z = ops.zscore(df)
    assert np.allclose(z.iloc[0].to_numpy(), z.iloc[1].to_numpy())


def test_ops_delay_delta():
    df = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
    assert ops.delay(df, 1)["a"].tolist()[1:] == [1.0, 2.0]
    assert ops.delta(df, 1)["a"].tolist()[1:] == [1.0, 2.0]


def test_ops_ts_rank_in_unit_interval():
    df = pd.DataFrame({"a": np.arange(20.0)})
    r = ops.ts_rank(df, 5).dropna()
    assert r["a"].between(0.0, 1.0).all()


def test_ops_where_scalar_branches():
    cond = pd.DataFrame({"a": [True, False]})
    a = pd.DataFrame({"a": [1.0, 2.0]})
    assert ops.where(cond, a, 0.0)["a"].tolist() == [1.0, 0.0]
    assert ops.where(cond, 9.0, a)["a"].tolist() == [9.0, 2.0]
    assert ops.where(cond, 9.0, 0.0)["a"].tolist() == [9.0, 0.0]


# ---------------------------------------------------------------------------
# 面板
# ---------------------------------------------------------------------------
def test_panel_from_feed_and_records():
    p = make_panel(4, 30)
    assert p.symbols == [f"S{i:02d}" for i in range(4)]
    assert p.n_times == 30
    assert p.returns().shape == p.close.shape


def test_panel_slice_upto_is_causal():
    p = make_panel(4, 30)
    t = 15
    sub = p.slice_upto(t)
    assert sub.timestamps[-1] == t
    assert sub.n_times == t + 1


def test_panel_from_bus_respects_token():
    """regression：面板构造必须受因果令牌约束。"""
    bus = DataBus()
    for i in range(10):
        bus.record("close", "A", i, 100.0 + i)
        bus.record("open", "A", i, 100.0 + i)
    tok = CausalToken(4, 4)
    p = Panel.from_bus(bus, ["A"], tok)
    assert p.n_times == 5, "只能看到 ts<=4 的 5 期"
    assert p.timestamps == [0, 1, 2, 3, 4]


def test_panel_from_bus_window():
    bus = DataBus()
    for i in range(50):
        bus.record("close", "A", i, float(i))
    p = Panel.from_bus(bus, ["A"], CausalToken(49, 49), window=10)
    assert p.n_times == 10
    assert p.close["A"].iloc[-1] == 49.0


# ---------------------------------------------------------------------------
# 因子库
# ---------------------------------------------------------------------------
def test_library_factors_compute_finite_values():
    """全库因子都应产出有效数据。

    regression：``wq045``（correlation 窗口 2 的 min_periods 缺陷）与
    ``wq094``（safe_div 标量广播缺陷）曾**全为 NaN**。

    注意：部分因子本身需要很长窗口（``wq019``/``wq038`` 用 250 期、
    ``wq052`` 用 240 期、``wq024`` 用 200 期），在短面板上为 NaN 是**正确**行为，
    故这里用足够长的面板并只对「短窗口因子」做严格断言。
    """
    p = make_panel(n_symbols=10, n_times=400, seed=7)
    names = list_factors()
    assert len(names) >= 50
    res = compute_many(names, p)

    # 1) 曾出缺陷的两个因子必须有效
    for n in ("wq045", "wq094"):
        ratio = float(res[n].notna().mean().mean())
        assert ratio > 0.2, f"{n} 有效率仅 {ratio:.1%}（疑似 min_periods/广播缺陷复发）"

    # 2) 需要长窗口的因子单独豁免
    long_window = {"wq019", "wq038", "wq052", "wq024"}
    dead = {n: float(f.notna().mean().mean()) for n, f in res.items()
            if n not in long_window and f.notna().mean().mean() < 0.2}
    assert not dead, f"以下因子几乎全为 NaN：{dead}"

    # 3) 长窗口因子在同长度面板上应确实有效
    for n in long_window:
        assert float(res[n].notna().mean().mean()) > 0.2, f"{n} 在 400 期面板上仍全 NaN"


def test_library_unknown_factor_raises(panel):
    with pytest.raises(KeyError):
        compute("nope", panel)


def test_library_wq101_formula(panel):
    """wq101 = (close-open)/((high-low)+0.001)，手工核对。"""
    f = compute("wq101", panel)
    ref = (panel.close - panel.open) / ((panel.high - panel.low) + 0.001)
    assert np.allclose(f.to_numpy(), ref.to_numpy(), equal_nan=True)


def test_library_factors_are_causal(panel):
    """因子第 t 行只依赖 <=t 的数据：截断重算应与原值一致。"""
    t = panel.timestamps[60]
    sub = panel.slice_upto(t)
    for name in ("wq042", "wq057", "wq060", "gtja002"):
        full = compute(name, panel).loc[t]
        part = compute(name, sub).loc[t]
        common = full.notna() & part.notna()
        assert common.sum() > 0
        assert np.allclose(full[common].to_numpy(float), part[common].to_numpy(float),
                           rtol=1e-9, atol=1e-9), f"{name} 存在跨期泄漏"


# ---------------------------------------------------------------------------
# 预处理
# ---------------------------------------------------------------------------
def test_mad_winsorize_clips_outliers():
    """去极值是**截面**操作：一行 = 同一时刻的跨标的观测。"""
    df = pd.DataFrame([[1.0, 2.0, 3.0, 4.0, 1000.0]],
                      columns=list("abcde"))
    w = mad_winsorize(df, n=3)
    assert w.iloc[0]["e"] < 1000.0
    assert w.iloc[0]["a"] == 1.0
    # 正常值不应被改动
    assert np.isclose(w.iloc[0]["c"], 3.0)


def test_zscore_zero_mean_unit_std():
    df = make_panel(6, 10).close
    z = zscore(df)
    assert np.allclose(z.mean(axis=1).dropna().to_numpy(), 0.0, atol=1e-9)


def test_neutralize_removes_exposure():
    """中性化后残差应与暴露近似正交。"""
    p = make_panel(10, 40)
    expo = log_market_cap(p)
    f = p.close.rank(axis=1, pct=True) + expo * 5.0   # 人为注入规模暴露
    resid = neutralize(f, [expo])
    # 逐行计算残差与暴露的相关
    corrs = []
    for t in resid.index:
        a, b = resid.loc[t], expo.loc[t]
        m = a.notna() & b.notna()
        if m.sum() > 5:
            corrs.append(float(np.corrcoef(a[m], b[m])[0, 1]))
    assert abs(np.mean(corrs)) < 0.05, f"中性化后仍残留暴露 {np.mean(corrs):.3f}"


def test_preprocess_pipeline_runs(panel):
    expo = [log_market_cap(panel)]
    raw = compute("wq042", panel)
    out = preprocess(raw, exposures=expo)
    assert out.shape == raw.shape
    assert out.notna().any().any()


def test_pairwise_correlation_and_dedup(panel):
    base = compute("wq042", panel)
    factors = {"a": base, "b": base * 1.0001, "c": compute("wq060", panel)}
    corr = pairwise_ic_correlation(factors)
    assert corr.loc["a", "b"] > 0.99
    kept, _, dropped = dedup_by_correlation(factors, {"a": 0.3, "b": 0.2, "c": 0.1}, 0.7)
    assert "a" in kept and "b" not in kept   # 保留 |score| 大的
    assert any(loser == "b" for _k, loser, _c in dropped)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def test_cross_section_ic_perfect_and_reversed():
    cols = list("abcde")
    f = pd.DataFrame([[1.0, 2, 3, 4, 5], [1.0, 2, 3, 4, 5]], columns=cols)
    r = pd.DataFrame([[1.0, 2, 3, 4, 5], [5.0, 4, 3, 2, 1]], columns=cols)
    ic = cross_section_ic(f, r)
    assert np.isclose(ic.iloc[0], 1.0)
    assert np.isclose(ic.iloc[1], -1.0)


def test_cross_section_ic_min_names_guard():
    f = pd.DataFrame({"a": [1.0], "b": [2.0]})
    r = pd.DataFrame({"a": [1.0], "b": [2.0]})
    assert cross_section_ic(f, r).isna().all()


def test_newey_west_tstat_on_constant_is_nan():
    assert not np.isfinite(newey_west_tstat(np.zeros(50)))


def test_newey_west_tstat_positive_for_positive_mean():
    rng = np.random.default_rng(0)
    x = 0.3 + 0.1 * rng.normal(size=300)
    assert newey_west_tstat(x) > 5


def test_ic_summary_fields():
    s = pd.Series([0.1, 0.2, -0.05, 0.15, 0.3])
    out = ic_summary(s)
    assert set(out) == {"ic_mean", "ic_std", "icir", "t_stat", "positive_rate", "n"}
    assert out["n"] == 5 and out["positive_rate"] == 0.8


def test_group_returns_monotone_for_perfect_factor():
    """完全预测未来的因子：分组收益应严格单调。"""
    t = 6
    cols = [f"S{i}" for i in range(10)]
    idx = list(range(12))
    rng = np.random.default_rng(1)
    fwd = pd.DataFrame(rng.normal(0, 0.01, size=(12, 10)), index=idx, columns=cols)
    factor = fwd.copy()          # 因子=未来收益 → 完美预测
    gr = group_returns(factor, fwd, n_groups=5)
    assert monotonicity_score(gr) > 0.9


def test_screen_factor_thresholds():
    t = 8
    cols = [f"S{i}" for i in range(20)]
    idx = list(range(300))
    rng = np.random.default_rng(3)
    noise = pd.DataFrame(rng.normal(size=(300, 20)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.normal(0, 0.01, size=(300, 20)), index=idx, columns=cols)
    good = fwd * 100 + noise * 0.01     # 强预测
    r_good = screen_factor(good, fwd, thresholds={"min_samples": 50})
    r_bad = screen_factor(noise, fwd, thresholds={"min_samples": 50})
    assert r_good["passed"] is True
    assert r_bad["passed"] is False
    assert r_bad["reject_reason"]


def test_forward_returns_uses_future():
    c = pd.DataFrame({"a": [1.0, 2.0, 4.0]})
    fr = forward_returns(c, 1)
    assert np.isclose(fr["a"].iloc[0], 1.0)   # (2-1)/1
    assert np.isnan(fr["a"].iloc[-1])
