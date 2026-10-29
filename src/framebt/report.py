"""回测报告：图表化展示（收益曲线 / 回撤高亮 / 典型交易 K 线标注）。

本模块把 :class:`~framebt.core.result.BacktestResult` 变成一份**自包含的 HTML 报告**：

- **收益曲线**：策略权益 vs 基准（买入持有），含最终收益 / 超额标注；
- **回撤高亮**：权益回撤面积图，最大回撤区间自动着色并标注起点/终点与幅度；
- **交易 K 线**：以日 K 线图（K 线蜡烛 + 成交量）逐标的展示，买入/卖出成交点箭头标注；
- **绩效面板**：总收益 / 年化 / Sharpe / 最大回撤 / 胜率 / 盈亏比 / 交易明细表。

使用：

    from framebt import Engine, run_backtest
    from framebt.report import generate_report, save_report

    res = run_backtest(bars, strategy, symbol="SH.600519", initial_cash=100_000.0)
    html = generate_report(res, bars_by_symbol={"SH.600519": bars})
    save_report(html, "reports/dual_ma.html")

依赖：matplotlib（可选依赖 `report`）。核心引擎不依赖它；`framebt/__init__.py`
通过模块级 ``__getattr__`` 惰性导入本模块，故 `import framebt` 在无 matplotlib
的环境也能工作，只有真正调用报告 API 时才需要安装。
"""

from __future__ import annotations

import base64
import bisect
import datetime as _dt
import io
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # 无显示环境（headless / 回测）也用 Agg 后端

# 中文字体：默认字体 DejaVu Sans 不含 CJK，中文会显示为方框。优先使用系统已装的
# 微软雅黑；若未被 matplotlib 字体管理器注册，则定位其 .ttf 显式注册后再选用。
import matplotlib.font_manager as _fm
from matplotlib.font_manager import FontProperties

_CJK_FONTS = ("Microsoft YaHei", "Microsoft JhengHei", "SimHei", "FangSong", "KaiTi")
_registered = False
for _fam in _CJK_FONTS:
    try:
        _fm.fontManager.findfont(FontProperties(family=_fam), fallback_to_default=False)
        break
    except OSError:
        continue
else:
    # 未注册：从系统字体目录定位第一个存在的 .ttf 显式注册。
    import glob
    import os

    for _base in (
        "C:\\Windows\\Fonts",
        os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts"),
    ):
        for _fp in glob.glob(os.path.join(_base, "*.ttf")):
            if any(_fam.lower() in os.path.basename(_fp).lower() for _fam in _CJK_FONTS):
                try:
                    _fm.fontManager.addfont(_fp)
                    _registered = True
                    break
                except Exception:  # noqa: BLE001
                    continue
        if _registered:
            break

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from .core.events import Bar, FillEvent, OrderSide
from .core.result import BacktestResult, Performance

# ---- 配色（A 股惯例：涨红跌绿）----
UP_COLOR = "#d64045"        # 涨 / 阳线
DOWN_COLOR = "#1daf7f"      # 跌 / 阴线
BENCH_COLOR = "#6c7a89"     # 基准
EQ_COLOR = "#2b6df6"        # 策略权益
DD_COLOR = "#e74c3c"        # 回撤
HIGHLIGHT = "#f5b041"       # 最大回撤区间高亮
BG = "#ffffff"
GRID = "#eef1f5"
TEXT = "#2c3e50"
MUTE = "#7f8c99"

# ---- K 线蜡烛颜色 ----
_C_UP = UP_COLOR
_C_DOWN = DOWN_COLOR
_C_WICK = "#596069"


def _attr(b, name):
    """统一取 Bar 对象或字典字段的值（报告同时接受这两种输入）。"""
    v = getattr(b, name, None)
    if v is not None:
        return v
    if isinstance(b, dict):
        return b.get(name)
    return None


def _ordinal_to_date(ordi: float) -> _dt.date:
    """逻辑时间戳（交易日 ordinal）转日历日期。"""
    return _dt.date.fromordinal(int(ordi))


def _fmt_date(ordi: float) -> str:
    return _ordinal_to_date(ordi).strftime("%Y-%m-%d")


def _fmt_money(v: float) -> str:
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:,.0f}"


def _b64_png(fig: plt.Figure) -> str:
    """把 Figure 渲染为 PNG 的 base64 data URI（供 HTML 内嵌）。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _tight(fig: plt.Figure) -> None:
    """tight_layout 的封装：忽略「共享 x 轴次面板」等已知无害警告。

    堆叠共享 x 轴的「K线+成交量」布局会触发 matplotlib 的
    “Axes not compatible with tight_layout” 提示，属渲染提示而非错误，
    仅针对该特定信息做静默处理。
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not compatible with tight_layout.*",
            category=UserWarning,
        )
        fig.tight_layout()


# ---------------------------------------------------------------------------
# 交易重构
# ---------------------------------------------------------------------------

@dataclass
class RoundTrip:
    """一个完整回合：开仓 → 平仓的已实现盈亏。"""

    symbol: str
    entry_ts: float
    exit_ts: float
    entry_frame: int
    exit_frame: int
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    reason: str = ""

    @property
    def direction(self) -> str:
        return "多" if self.shares > 0 else "空"

    @property
    def hold_frames(self) -> int:
        return max(0, self.exit_frame - self.entry_frame)


def _round_trips(fills: Sequence[FillEvent], symbol: str) -> list[RoundTrip]:
    """按标的把成交重构成一个完整回合（FIFO 配对）。"""
    trips: list[RoundTrip] = []
    open_shares = 0.0
    open_cost = 0.0  # 未平腿的累计成本（shares × price）
    entry_ts = entry_frame = entry_price = 0.0
    entry_reason = ""

    def _close(shares: float, fill: FillEvent) -> None:
        nonlocal open_shares, open_cost
        # 加权平均成本法：平仓部分的已实现盈亏。
        avg = (open_cost / open_shares) if open_shares else entry_price
        pnl = (fill.price - avg) * shares * (1 if shares > 0 else -1)
        # 方向由开仓时决定；这里用开仓累计符号。
        trips.append(
            RoundTrip(
                symbol=symbol,
                entry_ts=entry_ts,
                exit_ts=fill.timestamp,
                entry_frame=int(entry_frame),
                exit_frame=int(fill.frame_id),
                entry_price=avg,
                exit_price=fill.price,
                shares=shares,
                pnl=pnl,
                reason=fill.reason,
            )
        )

    fs = [f for f in fills if f.symbol == symbol]
    for f in fs:
        size = f.size
        if f.side is OrderSide.BUY:
            if open_shares <= 0:  # 开多
                open_shares = size
                open_cost = size * f.price
                entry_ts, entry_frame, entry_price = f.timestamp, f.frame_id, f.price
                entry_reason = f.reason
            else:  # 加仓
                open_shares += size
                open_cost += size * f.price
        else:  # SELL 平多
            if open_shares > 0:
                close = min(size, open_shares)
                _close(close, f)
                open_shares -= close
                open_cost -= close * (open_cost / open_shares if open_shares else f.price)
                if open_shares <= 1e-9:
                    open_shares = 0.0
                    open_cost = 0.0
    return trips


# ---------------------------------------------------------------------------
# 绩效聚合
# ---------------------------------------------------------------------------

@dataclass
class TradeStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    profit_factor: float = 0.0
    avg_hold: float = 0.0


def _trade_stats(result: BacktestResult) -> TradeStats:
    stats = TradeStats()
    symbols = sorted({f.symbol for f in result.fills})
    trips: list[RoundTrip] = []
    for s in symbols:
        trips.extend(_round_trips(result.fills, s))
    stats.trades = len(trips)
    if not trips:
        return stats
    gains = [t.pnl for t in trips if t.pnl > 0]
    losses = [t.pnl for t in trips if t.pnl < 0]
    stats.wins = len(gains)
    stats.losses = len(losses)
    stats.win_rate = stats.wins / len(trips)
    stats.total_pnl = sum(t.pnl for t in trips)
    gross_win = sum(gains)
    gross_loss = -sum(losses)
    stats.profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    stats.avg_hold = sum(t.hold_frames for t in trips) / len(trips)
    return stats


def _benchmark_series(
    equity_x: Sequence[float],
    bars_by_symbol: dict[str, list[Bar]],
    start_equity: float,
) -> list[tuple[float, float]]:
    """等权买入持有基准：每个标的独立 B&H，等权求和，对齐到策略权益曲线的时间轴。"""
    syms = list(bars_by_symbol)
    if not syms or start_equity <= 0:
        return [(x, start_equity) for x in equity_x]
    alloc = start_equity / len(syms)
    # 每个标的预计算 (ordinal, 累计收益乘数)，二分对齐。
    indexed: list[tuple[str, list[tuple[float, float]]]] = []
    for s in syms:
        bars = bars_by_symbol[s]
        if not bars:
            continue
        if len(bars) < 2:
            indexed.append((s, [(_attr(bars[0], "timestamp"), 1.0)]))
            continue
        ts = [_attr(b, "timestamp") for b in bars]
        closes = [_attr(b, "close") for b in bars]
        cum = 1.0
        series = [1.0]
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                cum *= (closes[i] / closes[i - 1])
            series.append(cum)
        indexed.append((s, list(zip(ts, series, strict=False))))

    xs = list(equity_x)
    out = []
    for x in xs:
        val = 0.0
        for _, series in indexed:
            arr_ts = [p[0] for p in series]
            i = bisect.bisect_right(arr_ts, x) - 1
            if i < 0:
                continue
            val += alloc * series[i][1]
        out.append((x, val))
    return out


# ---------------------------------------------------------------------------
# 图表 1：收益曲线 + 回撤高亮
# ---------------------------------------------------------------------------

def render_equity_chart(
    result: BacktestResult,
    *,
    benchmark: Optional[Sequence[tuple[float, float]]] = None,
    title: str = "权益曲线与回撤",
    initial_cash: float | None = None,
) -> str:
    """生成「权益曲线（上）+ 回撤面积（下）」图，返回 base64 PNG。"""
    eq = result.equity_curve
    if not eq:
        raise ValueError("权益曲线为空，无法生成图表")
    xs = [x for x, _ in eq]
    ys = [y for _, y in eq]
    start_equity = ys[0] or (initial_cash if initial_cash else 1.0)

    fig = plt.figure(figsize=(12, 7), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.18)
    ax_eq = fig.add_subplot(gs[0, 0])
    ax_dd = fig.add_subplot(gs[1, 0], sharex=ax_eq)
    for ax in (ax_eq, ax_dd):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTE)

    # 权益曲线
    ax_eq.plot(xs, ys, color=EQ_COLOR, linewidth=1.6, label="策略权益", zorder=3)
    if benchmark:
        bx = [x for x, _ in benchmark]
        by = [y for _, y in benchmark]
        ax_eq.plot(bx, by, color=BENCH_COLOR, linewidth=1.2,
                   linestyle=(0, (4, 2)), alpha=0.9, label="基准(买入持有)", zorder=2)

    # 终值标注
    last_x, last_y = xs[-1], ys[-1]
    eq_ret = (last_y / start_equity - 1) if start_equity else 0.0
    ax_eq.annotate(
        f"终值 {_fmt_money(last_y)}\n({eq_ret:+.1%})",
        xy=(last_x, last_y), xytext=(8, 6), textcoords="offset points",
        fontsize=10, color=EQ_COLOR,
        arrowprops=dict(arrowstyle="->", color=EQ_COLOR),
    )
    ax_eq.set_ylabel("权益", color=TEXT)
    ax_eq.legend(loc="upper left", framealpha=0.9, fontsize=10,
                 handlelength=2.2, labelcolor=TEXT)
    ax_eq.set_title(title, color=TEXT, fontsize=13, fontweight="bold", loc="left")

    # 回撤面积（回撤为负值，向下填充）
    dd = []
    peak = -math.inf
    for y in ys:
        if y > peak:
            peak = y
        dd.append((peak - y) / peak if peak > 0 else 0.0)
    ax_dd.plot(xs, [-d for d in dd], color=DD_COLOR, linewidth=1.0, zorder=3)
    ax_dd.fill_between(xs, 0, [-d for d in dd], color=DD_COLOR, alpha=0.25, zorder=2)
    ax_dd.set_ylabel("回撤", color=DD_COLOR)
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # 最大回撤区间高亮
    mdd, mdd_i = 0.0, 0
    trough_i = 0
    peak_v = -math.inf
    for i, (y, d) in enumerate(zip(ys, dd)):
        if y > peak_v:
            peak_v = y
            trough_i = i
        if d > mdd:
            mdd = d
            mdd_i = i
    # 回撤起点（峰值日）：[0..trough_i] 中权益最大的点。
    peak_i = max(range(trough_i + 1), key=lambda j: ys[j])
    ax_eq.axvspan(xs[peak_i], xs[trough_i], color=HIGHLIGHT, alpha=0.18, zorder=1)
    ax_dd.axvspan(xs[peak_i], xs[trough_i], color=HIGHLIGHT, alpha=0.25, zorder=1)
    mid = (xs[peak_i] + xs[trough_i]) / 2
    ax_dd.annotate(
        f"最大回撤 {mdd:.1%}",
        xy=(mid, -mdd),
        xytext=(0, 8), textcoords="offset points", ha="center",
        fontsize=10, color=DD_COLOR, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=DD_COLOR),
    )
    ax_dd.text(xs[peak_i], ax_dd.get_ylim()[1] * 0.7,
               f"{_fmt_date(xs[peak_i])} → {_fmt_date(xs[trough_i])}",
               fontsize=8, color=MUTE, va="top")

    ax_eq.xaxis.set_major_formatter(FuncFormatter(_xdate_fmt))
    ax_dd.xaxis.set_major_formatter(FuncFormatter(_xdate_fmt))
    fig.autofmt_xdate()
    _tight(fig)
    return _b64_png(fig)


def _xdate_fmt(x, pos) -> str:
    try:
        return _ordinal_to_date(x).strftime("%m-%d")
    except Exception:
        return str(int(x))


# ---------------------------------------------------------------------------
# 图表 2：K 线 + 交易标注
# ---------------------------------------------------------------------------

def render_trade_chart(
    bars: Sequence[Bar],
    trips: Sequence[RoundTrip],
    fills: Sequence[FillEvent],
    *,
    symbol: str,
    max_bars: int = 160,
    ma_periods: tuple[int, ...] = (5, 20),
) -> str:
    """逐标的日 K 线图（蜡烛 + 成交量），标注买入/卖出成交点。返回 base64 PNG。"""
    bars = list(bars)
    if not bars:
        return ""
    # 取最近 max_bars 根，避免蜡烛重叠。
    window = bars[-max_bars:] if len(bars) > max_bars else bars
    ts = [_attr(b, "timestamp") for b in window]
    o = [_attr(b, "open") for b in window]
    h = [_attr(b, "high") for b in window]
    l = [_attr(b, "low") for b in window]
    c = [_attr(b, "close") for b in window]
    v = [_attr(b, "volume") for b in window]

    fig = plt.figure(figsize=(13, 6.5), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    axv = fig.add_subplot(gs[1, 0], sharex=ax)
    for axi in (ax, axv):
        axi.set_facecolor(BG)
        for sp in axi.spines.values():
            sp.set_color(GRID)
        axi.grid(True, color=GRID, linewidth=0.8)
        axi.tick_params(colors=MUTE)

    # ---- K 线蜡烛 ----
    wick_w = 0.6
    body_w = 0.6
    for i in range(len(ts)):
        color = _C_UP if c[i] >= o[i] else _C_DOWN
        # 影线
        ax.vlines(ts[i], ymin=min(o[i], l[i]), ymax=max(o[i], h[i]),
                 color=_C_WICK, linewidth=0.7, zorder=1)
        # 实体
        body_lo = min(o[i], c[i])
        body_hi = max(o[i], c[i])
        height = max(body_hi - body_lo, 0.01)
        ax.add_patch(
            Rectangle((ts[i] - body_w / 2, body_lo), body_w, height,
                      facecolor=color, edgecolor=color, linewidth=0.6, zorder=2)
        )

    # ---- 均线 ----
    for p in ma_periods:
        if len(c) < p:
            continue
        ma = [sum(c[max(0, i - p + 1):i + 1]) / min(p, i + 1) for i in range(len(c))]
        ax.plot(ts, ma, linewidth=1.0, alpha=0.9,
                color="#f2994a" if p == 5 else "#5352bd", label=f"MA{p}")

    # ---- 成交量 ----
    for i in range(len(ts)):
        axv.bar(ts[i], v[i], width=wick_w,
                color=_C_UP if c[i] >= o[i] else _C_DOWN, alpha=0.6, zorder=2)

    # ---- 交易标注：本标的在窗口内的成交 ----
    sym_fills = [f for f in fills if f.symbol == symbol
                 and _attr(window[0], "timestamp") <= f.timestamp
                 <= _attr(window[-1], "timestamp")]
    for f in sym_fills:
        if f.side is OrderSide.BUY:
            ax.scatter(f.timestamp, f.price, marker="^", s=130,
                       color=_C_UP, edgecolor="white", linewidth=0.8, zorder=5)
            ax.annotate("买", (f.timestamp, f.price),
                        xytext=(0, 10), textcoords="offset points",
                        ha="center", fontsize=8.5, color=_C_UP, fontweight="bold")
        else:
            ax.scatter(f.timestamp, f.price, marker="v", s=130,
                       color=_C_DOWN, edgecolor="white", linewidth=0.8, zorder=5)
            ax.annotate("卖", (f.timestamp, f.price),
                        xytext=(0, -16), textcoords="offset points",
                        ha="center", fontsize=8.5, color=_C_DOWN, fontweight="bold")

    # 提示成交笔数
    n_buy = sum(1 for f in sym_fills if f.side is OrderSide.BUY)
    n_sell = sum(1 for f in sym_fills if f.side is OrderSide.SELL)
    ax.set_title(
        f"{symbol}  ·  K 线交易明细  ·  买 {n_buy} 笔 / 卖 {n_sell} 笔",
        color=TEXT, fontsize=12, fontweight="bold", loc="left",
    )
    ax.set_ylabel("价格", color=TEXT)
    axv.set_ylabel("量", color=MUTE)
    axv.tick_params(labeltop=True, labelcolor=MUTE)
    for lab in ax.yaxis.get_majorticklabels() + axv.yaxis.get_majorticklabels():
        lab.set_color(MUTE)

    ax.xaxis.set_major_formatter(FuncFormatter(_xdate_fmt))
    fig.autofmt_xdate()
    # bars 少于 MA 窗口时无均线图例，避免「No artists with labels」警告。
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9, handlelength=2, labelcolor=TEXT)
    _tight(fig)
    return _b64_png(fig)


# ---------------------------------------------------------------------------
# HTML 组装
# ---------------------------------------------------------------------------

def _metric_card(label: str, value: str, sub: str = "", color: str = TEXT) -> str:
    return (
        f'<div class="card"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def build_html(
    *,
    result: BacktestResult,
    strategy_name: str,
    symbols: Sequence[str],
    equity_img: str,
    trade_imgs: dict[str, str],
    perf: dict,
    stats: TradeStats,
    fills: Sequence[FillEvent],
    start_date: str,
    end_date: str,
) -> str:
    cards = [
        _metric_card("总收益", f"{perf['total_return']:+.2%}"),
        _metric_card("年化收益", f"{perf['annualized_return']:+.2%}"),
        _metric_card("Sharpe", f"{perf['sharpe']:.2f}"),
        _metric_card("最大回撤", f"{perf['max_drawdown']:.2%}", color=DD_COLOR),
        _metric_card("胜率", f"{perf['win_rate']:.0%}"),
        _metric_card("盈亏比", f"{stats.profit_factor if math.isfinite(stats.profit_factor) else '∞'}"),
        _metric_card("交易回合", f"{stats.trades}"),
        _metric_card("平均持仓", f"{stats.avg_hold:.0f} 帧"),
    ]
    cards_html = "\n".join(cards)

    # 交易明细表
    rows = []
    for f in fills:
        rows.append(
            f"<tr><td>{f.frame_id}</td><td>{_fmt_date(f.timestamp)}</td>"
            f"<td style='color:{UP_COLOR if f.side is OrderSide.BUY else DOWN_COLOR}'>"
            f"{'买入' if f.side is OrderSide.BUY else '卖出'}</td>"
            f"<td>{f.symbol}</td><td>{f.size:.0f}</td>"
            f"<td>{f.price:.3f}</td><td>{f.notional:,.0f}</td>"
            f"<td>{f.cost:,.3f}</td><td>{f.reason}</td></tr>"
        )
    trades_tbl = (
        "<table class='tbl'><thead><tr>"
        "<th>帧</th><th>日期</th><th>方向</th><th>标的</th><th>数量</th>"
        "<th>成交价</th><th>金额</th><th>费用</th><th>原因</th>"
        "</tr></thead><tbody>" + ("\n".join(rows) or "<tr><td colspan='9' class='empty'>无成交</td></tr>")
        + "</tbody></table>"
    )

    # 交易 K 线图
    charts_html = []
    for s in symbols:
        if s in trade_imgs and trade_imgs[s]:
            charts_html.append(
                f"<div class='chart-box'><div class='chart-title'>{s} · 交易 K 线</div>"
                f"<img src='data:image/png;base64,{trade_imgs[s]}'></div>"
            )
    charts_html = "\n".join(charts_html) or "<p class='empty'>未提供 K 线数据，跳过交易 K 线图（可传 bars_by_symbol 补全）</p>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>回测报告 · {strategy_name}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#f4f6f9; color:{TEXT};
    font-family:-apple-system,"Microsoft YaHei","Segoe UI",sans-serif; }}
  .wrap {{ max-width:1100px; margin:0 auto; padding:28px 20px 60px; }}
  header {{ background:linear-gradient(135deg,#2b6df6,#1daf7f); color:#fff;
    border-radius:14px; padding:26px 30px; margin-bottom:22px; }}
  header h1 {{ margin:0 0 6px; font-size:22px; letter-spacing:.5px; }}
  header .meta {{ font-size:13px; opacity:.92; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
    gap:14px; margin-bottom:26px; }}
  .card {{ background:#fff; border-radius:12px; padding:16px 18px;
    box-shadow:0 1px 4px rgba(44,62,80,.08); }}
  .card .label {{ font-size:12px; color:{MUTE}; margin-bottom:6px; }}
  .card .value {{ font-size:22px; font-weight:700; }}
  .card .sub {{ font-size:11px; color:{MUTE}; margin-top:3px; }}
  h2 {{ font-size:16px; margin:26px 0 12px; padding-left:10px;
    border-left:4px solid {EQ_COLOR}; }}
  .chart-box {{ background:#fff; border-radius:12px; padding:14px;
    box-shadow:0 1px 4px rgba(44,62,80,.08); margin-bottom:16px; }}
  .chart-title {{ font-size:13px; font-weight:600; margin-bottom:6px; color:{MUTE}; }}
  .chart-box img {{ width:100%; display:block; border-radius:6px; }}
  table.tbl {{ width:100%; border-collapse:collapse; background:#fff;
    border-radius:10px; overflow:hidden;
    box-shadow:0 1px 4px rgba(44,62,80,.08); font-size:13px; }}
  table.tbl th {{ background:#f0f3f7; text-align:left; padding:10px 12px;
    font-weight:600; color:{MUTE}; }}
  table.tbl td {{ padding:8px 12px; border-top:1px solid {GRID}; }}
  table.tbl tr:hover td {{ background:#fafbfc; }}
  .empty {{ color:{MUTE}; text-align:center; padding:18px; }}
  footer {{ text-align:center; color:{MUTE}; font-size:12px; margin-top:30px; }}
</style></head>
<body><div class="wrap">
<header>
  <h1>回测报告 · {strategy_name}</h1>
  <div class="meta">标的 {', '.join(symbols)} · 区间 {_fmt_date(start_date) if isinstance(start_date,(int,float)) else start_date} ~ {_fmt_date(end_date) if isinstance(end_date,(int,float)) else end_date} · 共 {result.frames} 帧 · {result.total_trades()} 笔成交</div>
</header>

<div class="cards">{cards_html}</div>

<h2>收益曲线与回撤</h2>
<div class="chart-box"><img src='data:image/png;base64,{equity_img}'></div>

<h2>交易 K 线（买入/卖出标注）</h2>
{charts_html}

<h2>成交明细</h2>
{trades_tbl}

<footer>framebt · 帧刷新因果回测框架 · 报告由 framebt.report 自动生成</footer>
</div></body></html>"""


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

@dataclass
class ReportConfig:
    """报告配置。"""

    strategy_name: str = "策略回测"
    initial_cash: float | None = None
    benchmark: bool = True            # 是否画买入持有基准
    max_trade_charts: int = 8         # 最多渲染多少只标的的 K 线图
    max_bars_per_chart: int = 160     # 单张 K 线最多根数
    ma_periods: tuple = (5, 20)
    period_frames: float = 244        # 一年帧数（年化用，日线≈244）


def generate_report(
    result: BacktestResult,
    *,
    bars_by_symbol: Optional[dict[str, list[Bar]]] = None,
    source=None,
    strategy_name: str | None = None,
    output_path: str | None = None,
    config: Optional[ReportConfig] = None,
) -> str:
    """生成回测报告的 HTML 字符串（图表以 base64 内嵌，文件自包含）。

    参数：
      result：回测结果（含 fills / equity_curve）。
      bars_by_symbol：{标的: [Bar,...]}，用于画交易 K 线图。缺省时若传入
        ``source``，则从它加载所有成交标的的 K 线；仍缺省则跳过 K 线图。
      source：数据源（DataSource），用于按需加载 K 线。
      strategy_name：报告标题。缺省取策略类名。
      output_path：若给出，额外把 HTML 写到此路径；返回 HTML 字符串。
      config：ReportConfig，控制指标口径与图表粒度。

    返回：自包含 HTML 字符串。
    """
    from .data_sources.base import DataSource

    if strategy_name is None:
        strat = getattr(result, "_strategy", None)
        strategy_name = type(strat).__name__ if strat else "策略回测"

    cfg = config or ReportConfig()
    perf = Performance(result, period_frames=cfg.period_frames)
    stats = _trade_stats(result)

    eq = result.equity_curve
    xs = [x for x, _ in eq]
    ys = [y for _, y in eq]
    start_date = xs[0] if xs else 0
    end_date = xs[-1] if xs else 0
    start_equity = ys[0] if ys else (cfg.initial_cash or 1.0)

    # 基准：等权买入持有
    benchmark = None
    if cfg.benchmark and bars_by_symbol:
        benchmark = _benchmark_series(xs, bars_by_symbol, start_equity)

    equity_img = render_equity_chart(
        result, benchmark=benchmark,
        title="权益曲线与回撤", initial_cash=cfg.initial_cash,
    )

    # 交易 K 线图：按成交数量取前 max_trade_charts 只标的
    trade_imgs: dict[str, str] = {}
    if bars_by_symbol is not None:
        # 补齐缺标的的 bars（按需从 source 加载）
        symbols = sorted({f.symbol for f in result.fills})
        for s in symbols:
            if s not in bars_by_symbol:
                if source is not None:
                    try:
                        bars_by_symbol[s] = source.load(s).bars
                    except Exception:
                        bars_by_symbol[s] = []
            if s not in bars_by_symbol:
                bars_by_symbol[s] = []
        top = sorted(symbols, key=lambda s: sum(1 for f in result.fills if f.symbol == s),
                     reverse=True)[: cfg.max_trade_charts]
        for s in top:
            trips = _round_trips(result.fills, s)
            if not bars_by_symbol[s]:
                continue
            trade_imgs[s] = render_trade_chart(
                bars_by_symbol[s], trips, result.fills, symbol=s,
                max_bars=cfg.max_bars_per_chart, ma_periods=cfg.ma_periods,
            )

    perf_dict = {
        "total_return": perf.total_return,
        "annualized_return": perf.annualized_return,
        "volatility": perf.volatility,
        "sharpe": perf.sharpe,
        "sharpe_cagr": perf.sharpe_cagr,
        "sortino": perf.sortino,
        "max_drawdown": perf.max_drawdown,
        "calmar": perf.calmar,
        "win_rate": perf.win_rate(),
    }

    html = build_html(
        result=result,
        strategy_name=strategy_name,
        symbols=sorted({f.symbol for f in result.fills}) or list(bars_by_symbol),
        equity_img=equity_img,
        trade_imgs=trade_imgs,
        perf=perf_dict,
        stats=stats,
        fills=result.fills,
        start_date=start_date,
        end_date=end_date,
    )

    if output_path:
        import pathlib
        p = pathlib.Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(html, encoding="utf-8")
    return html


def save_report(
    html: str,
    output_path: str,
) -> str:
    """把 HTML 字符串落盘，返回文件路径。"""
    import pathlib
    p = pathlib.Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return str(p)
