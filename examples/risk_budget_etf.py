"""风险预算多资产 ETF 策略（原型 + 回测）。

设计要点
--------
1. **不做选股、不做趋势择时**：本项目实测已证伪「横截面选股」与「价格趋势择时」
   （见 docs/PATH_200K.md §8、docs/ETF_TREND.md §6.3）。
2. **风险预算按资产类别，而不是按标的数量**：A 股宽基 5 只、海外 3 只、债券 2 只、
   黄金 1 只，若按标的等权，A 股会天然占 45%；按类别等权则每类 25%。
3. **波动率目标**：组合 60 日实现波动 > 12% 时按比例降仓，不足部分放货币 ETF；
   只降不升，不加杠杆。
4. **月度调仓 + 交易带**：每 21 个交易日重算目标，只有偏离超过 band 才下单，
   把换手压在 1×/年以内（ETF 免印花税、免过户费）。

运行：python examples/risk_budget_etf.py [--vol-target 0.12] [--no-vol-target]
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from framebt import Engine, StandardCost, StandardRisk  # noqa: E402
from framebt.core import Bar, FrameStrategy, Order, OrderSide, OrderType  # noqa: E402
from framebt.feed import BarFeed  # noqa: E402

CACHE = ROOT / "tools" / "etf_cache"
CASH = "SH.511990"          # 华宝添益（货币 ETF）
REPORT_START = pd.Timestamp("2014-01-01").toordinal()

# 结构性菜单：按资产类别分组，不按历史业绩挑选
CLASSES: dict[str, list[str]] = {
    "A股": ["SH.510050", "SH.510300", "SH.510500", "SZ.159915", "SH.512100"],
    "海外": ["SZ.159920", "SH.513100", "SH.513500"],
    "债券": ["SH.511010", "SH.511260"],
    "黄金": ["SH.518880"],
}
UNIVERSE = [s for members in CLASSES.values() for s in members]
CLS_OF = {s: c for c, members in CLASSES.items() for s in members}


class RiskBudgetStrategy(FrameStrategy):
    """类别等权 + 波动率目标 + 货币 ETF 现金腿。"""

    def __init__(
        self,
        universe: list[str],
        classes: dict[str, str],
        *,
        cash_symbol: str,
        vol_window: int = 60,
        target_vol: float = 0.12,
        rebalance_every: int = 21,
        weighting: str = "class_equal",
        min_amount: float = 2e7,
        amount_window: int = 20,
        band: float = 0.01,
        lot: int = 100,
    ):
        self.universe = list(universe)
        self.classes = dict(classes)
        self.cash_symbol = cash_symbol
        self.vol_window = int(vol_window)
        self.target_vol = float(target_vol)
        self.rebalance_every = int(rebalance_every)
        self.weighting = weighting
        self.min_amount = float(min_amount)
        self.amount_window = int(amount_window)
        self.band = float(band)
        self.lot = int(lot)

        self._frame = 0
        self._target: dict[str, float] = {}
        self.diag: list[dict] = []

    # ---- 主循环 ----

    def on_frame(self, frame, token, bus, host) -> list:
        if not frame.bars:
            return []
        self._frame += 1
        if self._frame <= self.vol_window + 1:
            return []

        rebalance = self._frame % self.rebalance_every == 0
        if rebalance:
            tgt = self._build_target(frame, bus, token)
            if tgt:
                self._target = tgt
        if not self._target:
            return []
        return self._orders(frame, host, allow_sells=rebalance)

    # ---- 目标权重 ----

    def _eligible(self, sym: str, frame, bus, token) -> bool:
        if not frame.has_bar(sym):
            return False
        hist = np.asarray(bus.get("close", sym, window=self.vol_window + 1, token=token),
                          dtype=float)
        if hist.size < self.vol_window + 1 or not np.isfinite(hist).all() or (hist <= 0).any():
            return False
        amt = np.asarray(bus.get("amount", sym, window=self.amount_window, token=token),
                         dtype=float)
        if amt.size < self.amount_window or not np.isfinite(amt).all():
            return False
        return float(amt.mean()) >= self.min_amount

    def _build_target(self, frame, bus, token) -> dict[str, float]:
        eligible = [s for s in self.universe if self._eligible(s, frame, bus, token)]
        if not eligible:
            return {self.cash_symbol: 1.0}

        if self.weighting == "equal":
            base = {s: 1.0 / len(eligible) for s in eligible}
        else:
            # 每个类别分配 1/N 资本；类别内等权。缺失类别 -> 现金（不把风险挤到别的类别）
            by_class: dict[str, list[str]] = {}
            for sym in eligible:
                by_class.setdefault(self.classes[sym], []).append(sym)
            n_cls = len(CLASSES)
            base = {}
            for cls, members in by_class.items():
                w_each = 1.0 / n_cls / len(members)
                for sym in members:
                    base[sym] = w_each

        # 波动率目标（只降不升）
        scale = 1.0
        port_vol = float("nan")
        if self.target_vol > 0 and len(base) >= 2:
            rets = []
            for sym in base:
                h = np.asarray(bus.get("close", sym, window=self.vol_window + 1, token=token),
                               dtype=float)
                rets.append(h[1:] / h[:-1] - 1.0)
            R = np.column_stack(rets)
            wv = np.array([base[s] for s in base], dtype=float)
            port_vol = float(np.std(R @ wv, ddof=1) * math.sqrt(252.0))
            if port_vol > 1e-9:
                scale = min(1.0, self.target_vol / port_vol)

        tgt = {s: w * scale for s, w in base.items()}
        cash_w = 1.0 - sum(tgt.values())
        if cash_w > 1e-6:
            tgt[self.cash_symbol] = cash_w

        cls_counts: dict[str, int] = {}
        for sym in eligible:
            cls_counts[self.classes[sym]] = cls_counts.get(self.classes[sym], 0) + 1
        self.diag.append({
            "frame": self._frame,
            "classes": cls_counts,
            "scale": scale,
            "port_vol": port_vol,
            "cash_w": max(cash_w, 0.0),
            "weights": dict(tgt),
        })
        return tgt

    # ---- 下单 ----

    def _orders(self, frame, host, allow_sells: bool) -> list:
        prices: dict[str, float] = {}
        for sym in set(self._target) | set(host.portfolio.exposure()):
            if frame.has_bar(sym):
                px = float(frame.close(sym))
                if px > 0 and math.isfinite(px):
                    prices[sym] = px
        if not prices:
            return []

        equity = float(host.portfolio.cash)
        for sym, size in host.portfolio.exposure().items():
            equity += size * prices.get(sym, 0.0)
        if equity <= 0:
            return []

        want: dict[str, int] = {}
        for sym, w in self._target.items():
            px = prices.get(sym)
            if px and w > 1e-6:
                sh = int(w * equity / px // self.lot * self.lot)
                if sh > 0:
                    want[sym] = sh

        pending = {o.symbol for o in host.pending_orders()}
        held = {s: int(host.portfolio.position(s).net_size())
                for s in set(want) | set(host.portfolio.exposure())}
        min_notional = equity * self.band
        orders: list = []

        if allow_sells:
            for sym, cur in held.items():
                if cur <= 0 or sym in pending:
                    continue
                tgt = want.get(sym, 0)
                if cur > tgt and (cur - tgt) * prices.get(sym, 0.0) >= min_notional:
                    orders.append(Order(side=OrderSide.SELL, symbol=sym,
                                        size=float(cur - tgt), order_type=OrderType.MARKET))

        under = [(s, n - held.get(s, 0)) for s, n in want.items()
                 if n > held.get(s, 0) and s not in pending]
        if not under:
            return orders

        reserved = sum(o.size * prices.get(o.symbol, 0.0) for o in host.pending_orders()
                       if o.side is OrderSide.BUY and math.isfinite(o.size))
        cash = max(0.0, float(host.portfolio.cash) - reserved) * 0.995
        need = {s: n * prices[s] for s, n in under}
        total = sum(need.values())
        if total <= 0 or cash <= 0:
            return orders

        for sym, _n in under:
            px = prices[sym]
            alloc = min(need[sym], cash * need[sym] / total)
            sh = int(alloc / px // self.lot * self.lot)
            if sh > 0 and sh * px >= min_notional:
                orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                    size=float(sh), order_type=OrderType.MARKET))
        return orders


# ---------------- 数据 / 指标 ----------------

def load_data() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in UNIVERSE + [CASH]:
        p = CACHE / f"{sym.replace('.', '_')}.parquet"
        if not p.exists():
            raise SystemExit(f"缺少缓存 {p}；先跑 tools/run_etf_trend.py 拉取数据")
        out[sym] = pd.read_parquet(p)
    return out


def build_feed(dfs: dict[str, pd.DataFrame]) -> list[Bar]:
    feed_start = pd.Timestamp("2012-01-01").toordinal()
    bars: list[Bar] = []
    for sym, df in dfs.items():
        for r in df.to_dict("records"):
            if int(r["dt_ord"]) < feed_start:
                continue
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
    return bars


def metrics(eq: pd.Series, fills) -> dict:
    eq = eq[eq.index >= REPORT_START]
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    ann = float((1.0 + total) ** (1.0 / max(yrs, 1e-9)) - 1.0)
    sd = float(r.std(ddof=1))
    sharpe = float(r.mean() / sd * math.sqrt(252.0)) if sd > 1e-12 else 0.0
    mdd = float(((eq.cummax() - eq) / eq.cummax()).max())
    fills = [f for f in fills if int(f.timestamp) >= REPORT_START]
    traded = sum(f.size * f.price for f in fills)
    fees = sum(f.cost for f in fills)
    turnover = traded / float(eq.mean()) / max(yrs, 1e-9) / 2.0
    return {"total": total, "ann": ann, "sharpe": sharpe, "mdd": mdd,
            "vol": sd * math.sqrt(252.0), "turnover": turnover, "fees": fees,
            "final": float(eq.iloc[-1]), "start": float(eq.iloc[0]), "years": yrs,
            "n_fills": len(fills)}


def bootstrap_sharpe(eq: pd.Series, n_boot: int = 2000, block: int = 21,
                     seed: int = 0) -> tuple[float, float, float]:
    r = eq[eq.index >= REPORT_START].pct_change().dropna().to_numpy()
    rng = np.random.default_rng(seed)
    nb = max(1, len(r) // block)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, max(1, len(r) - block), size=nb)
        s = np.concatenate([r[i:i + block] for i in idx])
        sd = s.std(ddof=1)
        stats.append(s.mean() / sd * math.sqrt(252.0) if sd > 1e-12 else 0.0)
    a = np.asarray(stats)
    return float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95))


def buyhold(dfs: dict[str, pd.DataFrame], sym: str) -> dict:
    px = dfs[sym].set_index("dt_ord")["close"].astype(float)
    px = px[px.index >= REPORT_START]
    r = px.pct_change().dropna()
    yrs = len(px) / 252.0
    total = float(px.iloc[-1] / px.iloc[0] - 1.0)
    sd = float(r.std(ddof=1)) * math.sqrt(252.0)
    return {"total": total,
            "ann": float((1 + total) ** (1 / max(yrs, 1e-9)) - 1),
            "sharpe": float(r.mean() * 252.0 / sd) if sd > 1e-12 else 0.0,
            "mdd": float(((px.cummax() - px) / px.cummax()).max()),
            "vol": sd}


def ew_curve(dfs: dict[str, pd.DataFrame], syms: list[str], every: int = 21) -> pd.Series:
    P = pd.DataFrame({s: dfs[s].set_index("dt_ord")["close"].astype(float) for s in syms})
    P = P.sort_index()
    P = P[P.index >= REPORT_START]
    avail = P.notna().to_numpy()
    P = P.ffill()
    R = P.pct_change().fillna(0.0)
    w = np.zeros(P.shape[1])
    eq = [1.0]
    for i in range(1, len(P)):
        if i % every == 0 or w.sum() <= 0:
            ok = avail[i]
            w = np.where(ok, 1.0 / max(ok.sum(), 1), 0.0)
        gross = float(w @ (1.0 + R.iloc[i].to_numpy()))
        eq.append(eq[-1] * gross)
        w = w * (1.0 + R.iloc[i].to_numpy()) / max(gross, 1e-12)
    return pd.Series(eq, index=P.index)


def ew_metrics(dfs: dict[str, pd.DataFrame]) -> dict:
    eq = ew_curve(dfs, UNIVERSE)
    r = eq.pct_change().dropna()
    yrs = len(eq) / 252.0
    total = eq.iloc[-1] / eq.iloc[0] - 1.0
    sd = r.std(ddof=1) * math.sqrt(252.0)
    return {"total": total, "ann": (1 + total) ** (1 / yrs) - 1,
            "sharpe": r.mean() * 252 / sd, "mdd": ((eq.cummax() - eq) / eq.cummax()).max(),
            "vol": sd}


def run_one(dfs, *, vol_target: float, weighting: str = "class_equal") -> dict:
    feed = BarFeed(build_feed(dfs))
    strat = RiskBudgetStrategy(
        UNIVERSE, CLS_OF, cash_symbol=CASH,
        target_vol=vol_target, rebalance_every=21, weighting=weighting,
    )
    cost = StandardCost(commission=0.00015, min_commission=5.0, stamp_tax=0.0,
                        transfer_fee=0.0, slippage="proportional", k=0.0003)
    eng = Engine(feed, strat, initial_cash=200_000.0, settlement_cycle=1,
                 cost_model=cost, risk=StandardRisk(max_position_pct=0.99))
    res = eng.run()
    eq = pd.Series({int(k): float(v) for k, v in res.equity_curve}).sort_index()
    m = metrics(eq, res.fills)
    m["eq"] = eq
    m["diag"] = strat.diag
    m["trades"] = res.total_trades()
    m["frames"] = res.frames
    return m


def show(label: str, m: dict) -> None:
    print(f"  {label:<28}{m['total']:>+10.2%}{m['ann']:>+9.2%}"
          f"{m['sharpe']:>8.2f}{m['mdd']:>9.2%}{m['vol']:>8.2%}"
          f"{m.get('turnover', float('nan')):>9.2f}")


def subperiod(eq: pd.Series, a: str, b: str) -> dict:
    lo, hi = pd.Timestamp(a).toordinal(), pd.Timestamp(b).toordinal()
    e = eq[(eq.index >= lo) & (eq.index <= hi)]
    r = e.pct_change().dropna()
    yrs = len(e) / 252.0
    total = float(e.iloc[-1] / e.iloc[0] - 1.0)
    sd = float(r.std(ddof=1)) * math.sqrt(252.0)
    return {"total": total, "ann": float((1 + total) ** (1 / max(yrs, 1e-9)) - 1),
            "sharpe": float(r.mean() * 252.0 / sd) if sd > 1e-12 else 0.0,
            "mdd": float(((e.cummax() - e) / e.cummax()).max()), "vol": sd}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vol-target", type=float, default=0.12)
    ap.add_argument("--md", default=None)
    ap.add_argument("--png", default=None, help="可选：把权益曲线画成 PNG")
    args = ap.parse_args()

    dfs = load_data()
    print(f"数据区间 {pd.Timestamp.fromordinal(min(d['dt_ord'].min() for d in dfs.values())).date()}"
          f" .. {pd.Timestamp.fromordinal(max(d['dt_ord'].max() for d in dfs.values())).date()}")
    print(f"报告区间 {pd.Timestamp.fromordinal(REPORT_START).date()} 起\n")

    variants = {
        "类别等权 + 12%波动目标": run_one(dfs, vol_target=args.vol_target),
        "类别等权（无波动目标）": run_one(dfs, vol_target=0.0),
        "全标的等权 + 12%波动目标": run_one(dfs, vol_target=args.vol_target, weighting="equal"),
        "全标的等权（基线）": run_one(dfs, vol_target=0.0, weighting="equal"),
    }

    print(f"  {'组合':<26}{'总收益':>10}{'年化':>9}{'Sharpe':>8}{'回撤':>9}{'波动':>8}{'换手':>9}")
    for label, m in variants.items():
        show(label, m)
    show("沪深300ETF 买入持有", {**buyhold(dfs, "SH.510300"), "turnover": 0.0})
    show("11 只等权（月度调回）", {**ew_metrics(dfs), "turnover": 0.5})

    best = variants["类别等权 + 12%波动目标"]
    med, lo, hi = bootstrap_sharpe(best["eq"])
    print(f"\n  主策略 bootstrap Sharpe 中位 {med:.2f}（5% {lo:.2f} .. 95% {hi:.2f}）")
    print(f"  成交 {best['trades']} 笔  费用 {best['fees']:,.0f} 元"
          f"（{best['fees']/200_000:.2%} 本金）  帧 {best['frames']}")

    d = best["diag"]
    if d:
        cash = np.array([x["cash_w"] for x in d])
        scale = np.array([x["scale"] for x in d])
        print(f"  调仓 {len(d)} 次  现金腿均值 {cash.mean():.1%}  最大 {cash.max():.1%}"
              f"  波动缩放均值 {scale.mean():.3f}（<1 占 {(scale < 0.999).mean():.1%}）")

    # 消融：波动目标的边际贡献
    base = variants["类别等权（无波动目标）"]
    print(f"\n  波动目标边际：年化 {best['ann']-base['ann']:+.2%}  "
          f"Sharpe {best['sharpe']-base['sharpe']:+.2f}  "
          f"回撤 {best['mdd']-base['mdd']:+.2%}  换手 {best['turnover']-base['turnover']:+.2f}×")

    print("\n  子区间（同一回测路径切片）：")
    print(f"  {'区间':<12}{'组合':<18}{'年化':>9}{'Sharpe':>8}{'回撤':>9}{'波动':>8}")
    for name, a, b in (("2014-2019", "2014-01-01", "2019-12-31"),
                       ("2020-2026", "2020-01-01", "2026-12-31")):
        for label, key in (("类别等权+波动目标", "类别等权 + 12%波动目标"),
                           ("全标的等权（基线）", "全标的等权（基线）")):
            st = subperiod(variants[key]["eq"], a, b)
            print(f"  {name:<12}{label:<18}{st['ann']:>+9.2%}{st['sharpe']:>8.2f}"
                  f"{st['mdd']:>9.2%}{st['vol']:>8.2%}")

    if args.png:
        _plot(variants, dfs, Path(args.png))


def _plot(variants: dict, dfs: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series = {
        "Risk budget (class-equal + 12% vol)": variants["类别等权 + 12%波动目标"]["eq"],
        "Equal weight (11 ETFs)": variants["全标的等权（基线）"]["eq"],
    }
    series["CSI300 ETF buy & hold"] = dfs["SH.510300"].set_index("dt_ord")["close"].astype(float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6.8), dpi=140, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]})
    for label, e in series.items():
        e = e[e.index >= REPORT_START]
        idx = [pd.Timestamp.fromordinal(int(i)) for i in e.index]
        growth = e.to_numpy() / e.iloc[0]
        ax1.plot(idx, growth, label=label, linewidth=1.3)
        dd = growth / np.maximum.accumulate(growth) - 1.0
        ax2.plot(idx, dd, linewidth=1.0)

    ax1.set_title("Risk-budget multi-asset ETF vs equal weight (2014-2026, costs included)")
    ax1.set_ylabel("growth of 1.0")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8)
    ax2.set_ylabel("drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    print(f"\n  图已保存 {path}")


if __name__ == "__main__":
    main()
