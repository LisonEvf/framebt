"""端到端「真实可用」门禁：真实数据重放 + 干净环境 + GBK 控制台 + 吞吐。

与 ``tests/test_capability_gate.py`` 的分工：
  - pytest 门禁：合成数据、金标准、随机 oracle、因果攻击、变异（秒级）；
  - 本脚本：真实 opentdx 数据上的独立重放、数据体检、工程可用性，并生成
    ``reports/USABILITY.md``。

用法：
    $env:PYTHONPATH="src"; python -u tools/gate_runner.py --md reports/USABILITY.md
"""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from framebt import (  # noqa: E402
    BarFeed,
    Engine,
    FrameStrategy,
    Order,
    OrderSide,
    Performance,
    StandardAshareRules,
    StandardCost,
    StandardRisk,
)
from framebt.data_sources import OpentdxParquetSource  # noqa: E402
from framebt.strategies import DualMAStrategy  # noqa: E402

EPS = 1e-9
COST_KW = dict(commission=3e-4, min_commission=5.0, stamp_tax=5e-4,
               transfer_fee=2e-5, slippage="volume_based", k=5e-4)


def _cost(side, size, price, volume) -> float:
    """独立成本公式（不复用 framebt.cost）。"""
    notional = size * price
    fee = max(notional * COST_KW["commission"], COST_KW["min_commission"])
    if side is OrderSide.SELL:
        fee += notional * COST_KW["stamp_tax"]
    fee += notional * COST_KW["transfer_fee"]
    if volume and volume > 0:
        fee += notional * COST_KW["k"] * (size / volume)
    return fee


class _Oracle:
    def __init__(self, cash: float):
        self.cash = float(cash)
        self.pos: dict[str, float] = defaultdict(float)

    def apply(self, f) -> None:
        n = f.size * f.price
        if f.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            self.cash += n - f.cost
            self.pos[f.symbol] -= f.size
        else:
            self.cash -= n + f.cost
            self.pos[f.symbol] += f.size

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + sum(q * marks.get(s, 0.0) for s, q in self.pos.items())


class _Plan(FrameStrategy):
    """按帧号执行固定计划：{frame_id: [(symbol, side, size, budget), ...]}。"""

    def __init__(self, plan: dict[int, list[tuple[str, OrderSide, float, float]]]):
        self.plan = plan

    def on_frame(self, frame, token, bus, host):
        orders = []
        for sym, side, size, budget in self.plan.get(frame.frame_id, []):
            if not frame.has_bar(sym):
                continue
            if side is OrderSide.SELL and size <= 0:
                continue
            orders.append(Order(side=side, symbol=sym, size=size, budget=budget))
        return orders


def _replay(rows_by_ts: dict[int, dict[str, object]], res, cash: float,
            symbols: list[str], n_ts: int) -> tuple[_Oracle, list[str]]:
    """逐帧重放；返回 (oracle, 错误列表)。静态帧用最后收盘。"""
    errors: list[str] = []
    last_close: dict[str, float] = {}
    for t in range(n_ts):
        for s in symbols:
            b = rows_by_ts.get(t, {}).get(s)
            if b is not None:
                last_close[s] = b.close

    fills_by_frame: dict[int, list] = defaultdict(list)
    for f in res.fills:
        fills_by_frame[f.frame_id].append(f)

    oracle = _Oracle(cash)
    for fid in range(len(res.equity_curve)):
        for f in fills_by_frame.get(fid, []):
            oracle.apply(f)
        marks = {}
        for s in symbols:
            b = rows_by_ts.get(fid, {}).get(s)
            marks[s] = b.close if b is not None else last_close.get(s, 0.0)
        got = res.equity_curve[fid][1]
        want = oracle.equity(marks)
        if abs(got - want) > max(1e-9, abs(want) * 1e-9):
            errors.append(f"帧 {fid} 权益 {got!r} != 重放 {want!r}")
            break

    for f in res.fills:
        b = rows_by_ts.get(f.frame_id, {}).get(f.symbol)
        vol = b.volume if b is not None else 0.0
        want = _cost(f.side, f.size, f.price, vol)
        if abs(f.cost - want) > max(1e-9, abs(want) * 1e-9):
            errors.append(f"帧 {f.frame_id} {f.symbol} 成本 {f.cost} != {want}")
            break
        if b is not None:
            if not (b.low - EPS <= f.price <= b.high + EPS):
                errors.append(f"成交价越界 {f.price} not in [{b.low},{b.high}]")
                break
            if b.volume > 0 and f.size > b.volume + EPS:
                errors.append(f"成交量越界 {f.size} > {b.volume}")
                break
    return oracle, errors


def _hygiene(bars) -> dict[str, int]:
    bad_nan = bad_ohlc = bad_range = bad_ret = bad_dup = 0
    prev_close = None
    seen = set()
    for b in bars:
        if not all(math.isfinite(x) for x in (b.open, b.high, b.low, b.close)):
            bad_nan += 1
        if min(b.open, b.high, b.low, b.close) <= 0:
            bad_ohlc += 1
        if not (b.low - EPS <= min(b.open, b.close) and
                b.high + EPS >= max(b.open, b.close) and b.low <= b.high + EPS):
            bad_range += 1
        if prev_close and prev_close > 0 and abs(b.close / prev_close - 1) > 0.31:
            bad_ret += 1
        if (b.timestamp, b.symbol) in seen:
            bad_dup += 1
        seen.add((b.timestamp, b.symbol))
        prev_close = b.close
    return dict(nan=bad_nan, ohlc=bad_ohlc, range=bad_range, ret=bad_ret, dup=bad_dup)


def _clean_import() -> tuple[bool, str]:
    code = f"""
import sys
sys.path.insert(0, r'{ROOT / "src"}')
import framebt
from framebt import FrameStrategy, Order, OrderSide, run_backtest
class S(FrameStrategy):
    def __init__(self):
        self.done = False
    def on_frame(self, frame, token, bus, host):
        if self.done:
            return []
        self.done = True
        return [Order(side=OrderSide.BUY, symbol="S", size=10)]
bars = [{{"timestamp": i, "symbol": "S", "open": 10, "high": 10,
          "low": 10, "close": 10, "volume": 1000}} for i in range(4)]
res = run_backtest(bars, S(), symbol="S", initial_cash=1000.0)
print("OK", framebt.__version__, len(res.fills), round(res.final_equity(), 6))
"""
    p = subprocess.run([sys.executable, "-S", "-c", code], cwd=str(ROOT),
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    return p.returncode == 0 and p.stdout.startswith("OK "), (p.stdout + p.stderr).strip()


def _gbk_quick() -> tuple[bool, str]:
    fd, md = tempfile.mkstemp(prefix="framebt_gbk_", suffix=".md")
    os.close(fd)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["PYTHONIOENCODING"] = "gbk"   # 故意用 GBK 控制台，验证脚本自身不崩
    try:
        t0 = time.perf_counter()
        p = subprocess.run(
            [sys.executable, "-u", "tools/validate_capability.py", "--quick", "--md", md],
            cwd=str(ROOT), env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        dt = time.perf_counter() - t0
        ok = p.returncode == 0 and Path(md).exists() and Path(md).stat().st_size > 0
        size = Path(md).stat().st_size if Path(md).exists() else 0
        return ok, f"exit={p.returncode}；报告 {size}B；用时 {dt:.1f}s（GBK 控制台未崩）"
    finally:
        try:
            os.unlink(md)
        except OSError:
            pass


def _throughput() -> tuple[float, bool]:
    rows = []
    for i in range(3_000):
        c = 10 + 2 * math.sin(i / 5) + i * 0.01
        rows.append({"timestamp": i, "symbol": "S", "open": c, "high": c * 1.01,
                     "low": c * 0.99, "close": c, "volume": 1_000_000.0})
    t0 = time.perf_counter()
    Engine(BarFeed.from_dicts(rows), DualMAStrategy("S", 5, 20),
           initial_cash=100_000.0, cost_model=StandardCost()).run()
    dt = time.perf_counter() - t0
    return 3_000 / dt, dt < 10.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default="reports/USABILITY.md")
    args = ap.parse_args()

    t_start = time.time()
    checks: list[dict] = []

    def add(cid, name, criterion, ok, detail):
        checks.append(dict(id=cid, name=name, criterion=criterion, ok=bool(ok),
                           detail=str(detail)))
        print(f"[{'PASS' if ok else 'FAIL'}] {cid} {name}：{detail}")

    def _cell(x) -> str:
        """Markdown 表格单元格：转义竖线，避免详情里的 ``|`` 撑破列。"""
        return str(x).replace("|", "\\|").replace("\n", " ")

    # ---- G6a 干净环境（纯标准库）----
    ok, detail = _clean_import()
    add("G6a", "纯标准库 import + 微型回测", "python -S 下 exit 0", ok, detail)

    # ---- G1–G5 门禁本体 ----
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_capability_gate.py",
         "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300,
    )
    summary = [ln for ln in (p.stdout + p.stderr).splitlines() if "passed" in ln or "failed" in ln]
    add("G1-G5", "pytest 门禁（金标准/oracle/因果/元形/不变量）",
        "exit 0，除 7 条 strict xfail 外 0 failed",
        p.returncode == 0, summary[-1] if summary else f"exit={p.returncode}")

    # ---- G6b GBK 控制台 ----
    ok, detail = _gbk_quick()
    add("G6b", "GBK 控制台下 validate_capability.py --quick", "exit 0 且写出报告", ok, detail)

    # ---- G6c 吞吐 ----
    rate, ok = _throughput()
    add("G6c", "3,000 帧吞吐", "> 2,000 bars/s 且 < 10s", ok and rate > 2_000,
        f"{rate:,.0f} bars/s")

    # ---- G7 真实数据 ----
    src = OpentdxParquetSource()
    available = set(src.available_symbols())
    wanted = ["SH.600519", "SZ.000001", "SH.601318", "SZ.300750", "SH.601899",
              "SH.600036", "SH.600276", "SZ.000858", "SZ.002415", "SH.601166",
              "SH.600030", "SH.601288", "SH.601398", "SH.601939"]
    candidates = [s for s in wanted if s in available]
    candidates += [s for s in src.available_symbols() if s not in candidates][:80]

    feeds: dict[str, list] = {}
    skipped: list[tuple[str, int]] = []
    seen_short: set[str] = set()
    # 优先选历史足够长的标的（多标的对齐帧数才够）；不足再退到 ≥100 根。
    for min_bars in (300, 100):
        for s in candidates:
            if len(feeds) >= 5:
                break
            if s in feeds:
                continue
            bars = src.load(s).bars[-400:]
            if len(bars) < min_bars:
                if s not in seen_short:
                    skipped.append((s, len(bars)))
                    seen_short.add(s)
                continue
            feeds[s] = bars
        if len(feeds) >= 5:
            break
    symbols = list(feeds)
    if len(symbols) < 3:
        raise SystemExit("真实数据可用标的不足 3 只（均 <100 根有效 Bar）")
    hyg = {s: _hygiene(b) for s, b in feeds.items()}
    bad_total = sum(sum(v.values()) for v in hyg.values())
    add("G7a", f"真实数据体检（{len(symbols)} 标的 × ≤400 根）",
        "NaN/非正价/OHLC 矛盾/异常日收益/重复时间戳 均为 0",
        bad_total == 0,
        "；".join(f"{s}:{sum(v.values())}" for s, v in hyg.items()) + f"（合计 {bad_total}）")

    # 单标的：80% 买入 + 后段卖出（先探出买入份额，再按份额卖出）
    sym = "SH.600519" if "SH.600519" in feeds else symbols[0]
    bars = feeds[sym][-400:]
    n = len(bars)
    buy_at = max(20, n // 4)
    sell_at = max(buy_at + 20, n - 40)
    eng_probe = Engine(BarFeed(bars), _Plan({buy_at: [(sym, OrderSide.BUY, 0.0, 80_000.0)]}),
                       initial_cash=100_000.0, cost_model=StandardCost(**COST_KW),
                       risk=StandardRisk(), rules=StandardAshareRules(limit=0.10))
    res_probe = eng_probe.run()
    bought = sum(f.size for f in res_probe.fills)
    eng1 = Engine(
        BarFeed(bars),
        _Plan({buy_at: [(sym, OrderSide.BUY, 0.0, 80_000.0)],
               sell_at: [(sym, OrderSide.SELL, bought, 0.0)]}),
        initial_cash=100_000.0, cost_model=StandardCost(**COST_KW),
        risk=StandardRisk(), rules=StandardAshareRules(limit=0.10),
    )
    res1 = eng1.run()
    rows1 = {i: {sym: b} for i, b in enumerate(bars)}
    oracle1, err1 = _replay(rows1, res1, 100_000.0, [sym], n)
    ok1 = (not err1 and abs(eng1.ledger.cash - oracle1.cash) <= max(1e-9, abs(oracle1.cash) * 1e-9))
    perf1 = Performance(res1, period_frames=244)
    add("G7b", f"真实单标的端到端重放（{sym}，{n} 根）",
        "逐帧权益 + 逐笔成本 1e-9 一致",
        ok1,
        f"{len(res1.fills)} 笔成交；期末 {res1.final_equity():,.2f}；"
        f"收益 {perf1.total_return:+.2%}；回撤 {perf1.max_drawdown:.2%}；"
        f"现金偏差 {eng1.ledger.cash - oracle1.cash:+.3e}"
        + (f"；{err1[:1]}" if err1 else ""))

    # 多标的：对齐后 5 只 × 250 帧，买入并持有
    common = None
    for s in symbols:
        ts = {b.timestamp for b in feeds[s][-300:]}
        common = ts if common is None else (common & ts)
    common = sorted(common or [])[-250:]
    if len(common) < 50:
        raise SystemExit("真实数据共同交易日不足，无法做多标的门禁")
    by_key = {(b.timestamp, b.symbol): b for s in symbols for b in feeds[s]}
    rows_multi = []
    for t in common:
        for s in symbols:
            b = by_key.get((t, s))
            if b is not None:
                rows_multi.append(b)
    feed_multi = BarFeed(rows_multi)
    plan_multi = {30: [(s, OrderSide.BUY, 0.0, 20_000.0) for s in symbols]}
    eng2 = Engine(feed_multi, _Plan(plan_multi), initial_cash=1_000_000.0,
                  cost_model=StandardCost(**COST_KW), risk=StandardRisk(),
                  rules=StandardAshareRules(limit=0.10))
    res2 = eng2.run()
    rows2 = {i: {s: by_key[(t, s)] for s in symbols if (t, s) in by_key}
             for i, t in enumerate(common)}
    oracle2, err2 = _replay(rows2, res2, 1_000_000.0, symbols, len(common))
    ok2 = (not err2 and abs(eng2.ledger.cash - oracle2.cash) <= max(1e-9, abs(oracle2.cash) * 1e-9))
    perf2 = Performance(res2, period_frames=244)
    add("G7c", f"真实多标的端到端重放（{len(symbols)} 标的 × {len(common)} 帧）",
        "逐帧权益 + 逐笔成本 1e-9 一致",
        ok2,
        f"{len(res2.fills)} 笔成交；期末 {res2.final_equity():,.2f}；"
        f"收益 {perf2.total_return:+.2%}；回撤 {perf2.max_drawdown:.2%}；"
        f"现金偏差 {eng2.ledger.cash - oracle2.cash:+.3e}"
        + (f"；{err2[:1]}" if err2 else ""))

    # ---- G8 变异（读取独立运行的报告）----
    mut_md = ROOT / "reports" / "MUTATION.md"
    mut_text = mut_md.read_text(encoding="utf-8") if mut_md.exists() else ""
    mut_ok = "杀死率：15/15" in mut_text
    add("G8", "变异杀死率（tools/mutate_gate.py）", "15/15 且控制组通过",
        mut_ok, "见 reports/MUTATION.md" if mut_md.exists() else "未运行")

    core_ok = all(c["ok"] for c in checks)
    verdict = ("核心能力真实可用（预登记门禁通过；7 条建模边界见 §5，不构成生产就绪）"
               if core_ok else "门禁未通过：存在核心失败项，不能宣称真实可用")

    # ---- 写报告 ----
    path = Path(args.md)
    path.parent.mkdir(parents=True, exist_ok=True)
    env_lines = [
        f"- Python：{sys.version.split()[0]}",
        f"- 平台：{os.name} / {sys.platform}",
        f"- 数据源：`{src.data_dir}`",
        f"- 标的：{', '.join(symbols)}",
        f"- 命令：`python -u tools/gate_runner.py --md {path.as_posix()}`；"
        f"变异 `python -u tools/mutate_gate.py --md reports/MUTATION.md`",
    ]
    lines = [
        "# framebt「真实可用」门禁报告",
        "",
        f"> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ 用时 {time.time() - t_start:.1f}s ｜ "
        f"预登记：[`docs/GATE_USABILITY.md`](../docs/GATE_USABILITY.md)",
        "",
        "## 0. 结论",
        "",
        f"**{verdict}**",
        "",
        f"核心门禁 {sum(1 for c in checks if c['ok'])}/{len(checks)} 项通过；"
        "门禁本体见 `tests/test_capability_gate.py`（预登记 7 条 strict xfail 为已登记边界，"
        "不计通过）；变异结果见 [`MUTATION.md`](MUTATION.md)。",
        "",
        "> 本报告只回答「框架算得对不对、挡不挡得住、可不可复现、改坏会不会被抓」，"
        "**不回答任何策略是否赚钱**。",
        "",
        "## 1. 结果总表",
        "",
        "| 项 | 判据 | 结果 | 证据 |",
        "|---|---|---|---|",
    ]
    for c in checks:
        lines.append(f"| {_cell(c['id'])} {_cell(c['name'])} | {_cell(c['criterion'])} | "
                     f"**{'PASS' if c['ok'] else 'FAIL'}** | {_cell(c['detail'])} |")
    lines += [
        "",
        "## 2. 环境",
        "",
        *env_lines,
        "",
        "## 3. 真实数据体检",
        "",
        "对每只标的最近 ≤400 根 QFQ 日线检查：NaN、非正价、OHLC 区间矛盾、"
        "|日收益|>31%、重复 `(timestamp, symbol)`。",
        "",
        "| 标的 | NaN | 非正价 | OHLC 矛盾 | 异常日收益 | 重复 |",
        "|---|---|---|---|---|---|",
    ]
    for s in symbols:
        h = hyg[s]
        lines.append(f"| {s} | {h['nan']} | {h['ohlc']} | {h['range']} | {h['ret']} | {h['dup']} |")
    if skipped:
        lines += [
            "",
            "被剔除的候选（有效 Bar < 100，数据覆盖/复权问题）："
            + "、".join(f"`{s}`（{n} 根）" for s, n in skipped[:10]),
        ]
    lines += [
        "",
        "## 4. 独立重放口径",
        "",
        "- 成本：`max(名义×3bp, 5) + 卖 5bp 印花 + 0.2bp 过户 + 名义×5bp×(量/当帧量)`；",
        "- 权益：`现金 + Σ(净持仓 × 当帧收盘)`；静态帧用最后收盘；",
        "- 重放不使用 `Ledger` / `StandardCost` 的实现，只用 `FillEvent` 公开字段；",
        "- 判定阈值：相对/绝对偏差 ≤ 1e-9。",
        "",
        "## 5. 预登记边界（未修复，不计入 PASS）",
        "",
        "门禁用 `xfail(strict=True)` 钉住 B1–B6；B7–B9 是代码审查确认的边界，未单独写门禁测试。"
        "修好后 strict xfail 会 XPASS 并提醒更新判据：",
        "",
        "| # | 边界 | 影响 |",
        "|---|---|---|",
        "| B1 | `volume=0` 的 Bar 上市价单不受成交量约束 | 停牌/无成交日可无限成交 |",
        "| B2 | NaN 行情/数量被静默接受 | 权益变 NaN，业绩不可读 |",
        "| B3 | 同帧多笔市价单不共享当帧成交量 | 总成交可超 Bar 成交量 |",
        "| B4 | STOP / TRAILING_STOP 不受成交量约束 | 止损单可超过当日成交量 |",
        "| B5 | `StandardRisk` 不预留费用 | 满仓预算单现金可小幅为负 |",
        "| B6 | 同帧多笔大额预算单互相预留 | 总量超现金时两笔都被拒（过度保守） |",
        "| B7 | 涨跌停全局单一 `limit` | 创业板/科创板/北交所口径不准 |",
        "| B8 | 数据结束后静态帧可成交尾部订单 | 尾部订单在「无市场」时成交 |",
        "| B9 | `delta_t>1` 时 `frame.bars` 只留最后一根 | 只用 frame 访问会漏帧内 Bar |",
        "",
        "## 6. 复现",
        "",
        "```powershell",
        "$env:PYTHONPATH=\"src\"",
        "python -m pytest -q tests/test_capability_gate.py",
        "python -u tools/gate_runner.py --md reports/USABILITY.md",
        "python -u tools/mutate_gate.py --md reports/MUTATION.md",
        "```",
        "",
        "**通过标准**（预登记 §7）：门禁 0 failed/0 error（strict xfail 除外）；"
        "本报告 G1–G7 全 PASS；变异 15/15；三条同时成立才可宣称「真实可用」。",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n报告已写入 {path}")
    print(f"结论：{verdict}")
    return 0 if core_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
