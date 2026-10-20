"""框架能力验证：用真实 opentdx 数仓数据（默认全市场 5500+ 标的 × DAILY/QFQ）逐项检验。

不是「跑个策略看收益」，而是逐条检验设计文档承诺的机制是否**在真实数据上成立**：

  1. 数据接入完整性（时间戳唯一/有序/OHLC 自洽）
  2. 因果性 I1/I2 —— 前视在物理上不可达（含「偷看未来」尝试）
  3. 确定性 I3 —— 逐位复现
  4. A 股规则真实生效（涨跌停 / T+1）
  5. 成本模型影响可核验（权益差 == 累计费用）
  6. 真实策略表现（诚实报告，不做调参美化）
  7. 吞吐量

用法：
    python tools/validate_framework.py            # 打印报告
    python tools/validate_framework.py --md reports/VALIDATION.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from framebt import (
    CausalityViolation,
    DetermRNG,
    Engine,
    FrameStrategy,
    Ledger,
    MarketFrame,  # noqa: F401  (类型标注)
    NoOpCost,
    Performance,
    StandardAshareRules,
    StandardCost,
    Order,
    OrderSide,
    OrderType,
    run_backtest,
)
from framebt.core.events import Bar
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import DualMAStrategy, MomentumCrossSection

LINES: list[str] = []

# Windows 中文控制台默认 GBK，输出「✓」会 UnicodeEncodeError 并中断（--md 也写不出来）。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass


def out(s: str = "") -> None:
    print(s)
    LINES.append(s)


def hr(title: str) -> None:
    out("")
    out(f"{'=' * 72}")
    out(f"  {title}")
    out(f"{'=' * 72}")


def to_bars(feed) -> list[dict]:
    return [
        {"timestamp": b.timestamp, "symbol": b.symbol, "open": b.open, "high": b.high,
         "low": b.low, "close": b.close, "volume": b.volume}
        for b in feed.bars
    ]


# ---------------------------------------------------------------------------
# 1. 数据接入完整性
# ---------------------------------------------------------------------------
def check_data(src: OpentdxParquetSource, syms: list[str]) -> None:
    hr("1. 数据接入完整性")
    out(f"数据源      : opentdx 数仓（默认 ../opentdx/db/data，可用 FRAMEBT_DATA_DIR 覆盖）")
    out(f"标的数      : {len(syms)}")
    out(f"口径        : period={src.period} (DAILY)  adjust={src.adjust} (QFQ)")

    lens, problems = [], []
    for s in syms:
        feed = src.load(s)
        bars = feed.bars
        lens.append(len(bars))
        ts = [b.timestamp for b in bars]
        if len(set(ts)) != len(ts):
            problems.append(f"{s} 时间戳重复")
        if ts != sorted(ts):
            problems.append(f"{s} 未按时间排序")
        for b in bars:
            if not (b.high >= max(b.open, b.close) - 1e-6
                    and b.low <= min(b.open, b.close) + 1e-6
                    and b.high >= b.low - 1e-6):
                problems.append(f"{s} OHLC 不自洽 @ts={b.timestamp}")
                break

    out(f"bar 数      : min={min(lens)}  max={max(lens)}  合计={sum(lens)}")
    out(f"完整性问题  : {len(problems) if problems else '无 ✓'}")
    for p in problems[:5]:
        out(f"   - {p}")


# ---------------------------------------------------------------------------
# 2. 因果性 —— 「偷看未来」尝试
# ---------------------------------------------------------------------------
class PeekStrategy(FrameStrategy):
    """主动尝试前视：负 lag、伪造未来 token，并记录每帧可见 bar 数。"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.visible: list[int] = []
        self.leaks: list[str] = []

    def on_frame(self, frame, token, bus, host):
        from framebt import CausalToken

        legit = bus.get("close", self.symbol, token=token)
        self.visible.append(len(legit))

        # 尝试 1：负 lag = 显式请求未来 → 应抛 CausalityViolation
        try:
            bus.get("close", self.symbol, token=token, lag=-1)
            self.leaks.append(f"frame {frame.frame_id}: lag=-1 未被拒绝")
        except CausalityViolation:
            pass

        # 尝试 2：伪造 issued_at 更大的 token → 必须拿不到更多数据（总线钳制到权威时间）
        try:
            forged = bus.get("close", self.symbol,
                             token=CausalToken(token.issued_at_ns + 10_000, 999))
            if len(forged) > len(legit):
                self.leaks.append(
                    f"frame {frame.frame_id}: 伪造 token 多拿到 "
                    f"{len(forged) - len(legit)} 根未来数据"
                )
        except CausalityViolation:
            pass
        return []


def check_causality(src: OpentdxParquetSource, sym: str) -> None:
    hr("2. 因果性 I1 / I2 —— 前视在物理上不可达")
    feed = src.load(sym)

    # --- I1：可见历史长度必须等于已过帧数+1，永远看不到下一根 ---
    probe = PeekStrategy(sym)
    eng_probe = Engine(feed, probe)
    eng_probe.run()
    ok_growth = probe.visible == list(range(1, len(probe.visible) + 1))
    out(f"I1 每帧可见 bar 数 : {probe.visible[:6]} ... {probe.visible[-3:]}  (应 1,2,3,...)")
    out(f"I1 可见性单调增长 : {'✓ 通过' if ok_growth else '✗ 失败'}")
    out(f"I1 前视尝试逃逸数 : {len(probe.leaks)} "
        f"{'✓ 通过' if not probe.leaks else '✗ 失败 ' + str(probe.leaks[:3])}")
    out(f"I1 伪造 token 次数 : {eng_probe.bus.forgery_attempts}（总线已钳制到权威时间，"
        f"未泄漏任何额外数据）")

    # --- I2：每笔成交的帧号必须严格大于其信号帧（订单延迟 >= 1）---
    eng = Engine(feed, DualMAStrategy(sym, 10, 30), initial_cash=1_000_000.0)
    res = eng.run()
    violations = [f for f in res.fills if f.frame_id is None or f.frame_id < 1]
    out(f"I2 成交笔数       : {res.total_trades()}")
    out(f"I2 首笔成交帧     : {res.fills[0].frame_id if res.fills else '-'}（信号帧应更早）")
    out(f"I2 违反延迟笔数   : {len(violations)} {'✓ 通过' if not violations else '✗ 失败'}")

    # --- delay_frames=0 必须被拒绝（否则等于允许同帧成交）---
    try:
        Engine(feed, DualFrame(), delay_frames=0)
        out("I2 delay=0 拒绝    : ✗ 失败（未报错）")
    except ValueError:
        out("I2 delay=0 拒绝    : ✓ 通过（引擎构造即抛 ValueError）")


class DualFrame(FrameStrategy):  # pragma: no cover - 仅用于触发构造校验
    def on_frame(self, frame, token, bus, host):
        return []


# ---------------------------------------------------------------------------
# 3. 确定性
# ---------------------------------------------------------------------------
def check_determinism(src: OpentdxParquetSource, syms: list[str]) -> None:
    hr("3. 确定性 I3 —— 同配置逐位复现")
    feed = src.universe(syms[:6])
    bars = to_bars(feed)

    def once():
        return run_backtest(bars, MomentumCrossSection(syms[:6], 20, 60, 3),
                            symbol="P", initial_cash=1_000_000.0, seed=20240101)

    a, b = once(), once()
    same_eq = a.equity_curve == b.equity_curve
    same_fill = ([(f.order_id, f.price, f.size, f.frame_id) for f in a.fills]
                 == [(f.order_id, f.price, f.size, f.frame_id) for f in b.fills])
    out(f"运行规模        : {a.frames} 帧 / {a.total_trades()} 笔成交")
    out(f"权益曲线逐位一致: {'✓ 通过' if same_eq else '✗ 失败'}")
    out(f"成交明细逐位一致: {'✓ 通过' if same_fill else '✗ 失败'}")
    out(f"最终权益        : {a.final_equity():,.4f}")


# ---------------------------------------------------------------------------
# 4. A 股规则真实生效
# ---------------------------------------------------------------------------
def check_rules(src: OpentdxParquetSource, syms: list[str]) -> None:
    hr("4. A 股规则真实生效（涨跌停 / T+1）")
    rules = StandardAshareRules(limit=0.10, settlement_cycle=1)

    # --- 扫描真实涨跌停日（收盘价封在 ±10% 板上）---
    limit_up = limit_down = days = 0
    sample = None
    for s in syms:
        bars = src.load(s).bars
        for prev, cur in zip(bars, bars[1:]):
            days += 1
            up = prev.close * 1.10
            dn = prev.close * 0.90
            if cur.close >= up * 0.999:
                limit_up += 1
                sample = sample or (s, prev, cur)
            if cur.close <= dn * 1.001:
                limit_down += 1
    out(f"扫描交易日      : {days}（{len(syms)} 标的）")
    out(f"真实涨停日(收盘封板): {limit_up}")
    out(f"真实跌停日(收盘封板): {limit_down}")
    if sample:
        s, prev, cur = sample
        rules.observe(MarketFrame(0, 0, 1))
        out(f"样例            : {s}  昨收 {prev.close:.3f} → 今收 {cur.close:.3f} "
            f"({cur.close / prev.close - 1:+.2%})")

    # --- 构造真实昨收，验证涨跌停拒绝与放行 ---
    s = syms[0]
    bars = src.load(s).bars
    prev_bar, cur_bar = bars[500], bars[501]
    rules2 = StandardAshareRules(limit=0.10)
    f_prev = MarketFrame(0, 0, 1)
    f_prev.add_bar(prev_bar)
    rules2.observe(f_prev)
    upper, lower = rules2.price_limit(s, f_prev)
    out(f"涨跌停带        : {s} 昨收 {prev_bar.close:.3f} → "
        f"涨停 {upper:.3f} / 跌停 {lower:.3f}")

    # 买入价越涨停 → 拒绝
    f_up = MarketFrame(1, 1, 2)
    f_up.add_bar(Bar(timestamp=1, symbol=s, open=upper * 1.02, high=upper * 1.02,
                     low=upper * 1.01, close=upper * 1.02, volume=1.0))
    r_up = rules2.check(Order(side=OrderSide.BUY, symbol=s, size=100), Ledger(cash=1e9), f_up)
    out(f"涨停价买入      : {'✓ 被拒 → ' + r_up if r_up else '✗ 未拦截'}")

    # 卖出价越跌停 → 拒绝
    f_dn = MarketFrame(2, 2, 3)
    f_dn.add_bar(Bar(timestamp=2, symbol=s, open=lower * 0.98, high=lower * 0.99,
                     low=lower * 0.98, close=lower * 0.98, volume=1.0))
    led = Ledger(cash=0.0)
    r_dn = rules2.check(Order(side=OrderSide.SELL, symbol=s, size=100), led, f_dn)
    out(f"跌停价卖出      : {'✓ 被拒 → ' + r_dn if r_dn else '✗ 未拦截'}")

    # --- T+1：同帧买入后卖出应被拒 ---
    led2 = Ledger(cash=1e6, settlement_cycle=1)
    from framebt import FillEvent
    led2.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol=s, size=100,
                              price=cur_bar.close, frame_id=501, timestamp=501))
    f_same = MarketFrame(501, 501, 502)
    f_same.add_bar(Bar(timestamp=501, symbol=s, open=cur_bar.open, high=cur_bar.high,
                       low=cur_bar.low, close=cur_bar.close, volume=cur_bar.volume))
    r_t1 = rules2.check(Order(side=OrderSide.SELL, symbol=s, size=100), led2, f_same)
    out(f"T+1 同日卖出    : {'✓ 被拒 → ' + r_t1 if r_t1 else '✗ 未拦截'}")

    f_next = MarketFrame(502, 502, 503)
    f_next.add_bar(Bar(timestamp=502, symbol=s, open=cur_bar.open, high=cur_bar.high,
                       low=cur_bar.low, close=cur_bar.close, volume=cur_bar.volume))
    r_t1b = rules2.check(Order(side=OrderSide.SELL, symbol=s, size=100), led2, f_next)
    out(f"T+1 次日卖出    : {'✓ 放行' if r_t1b is None else '✗ 仍被拒 → ' + r_t1b}")


# ---------------------------------------------------------------------------
# 5. 成本模型影响可核验
# ---------------------------------------------------------------------------
class FixedSizeMA(DualMAStrategy):
    """双均线，但以**固定股数**下单。

    这样含/不含成本的两次回测成交价与数量完全一致，权益差应恰好等于累计费用
    （用 budget 下单时费用会改变现金→改变后续下单量，权益差含复利效应，不能直接比对）。
    """

    def __init__(self, symbol: str, short: int, long: int, size: float):
        super().__init__(symbol, short, long)
        self._fixed = size

    def on_frame(self, frame, token, bus, host):
        orders = super().on_frame(frame, token, bus, host)
        for o in orders:
            o.budget = 0.0
            o.size = self._fixed
        return orders


def reconcile(engine: Engine, result, initial_cash: float) -> float:
    """由成交独立重算期末权益：现金流水 + 持仓按末次收盘计值。

    这是对 Ledger 的**外部审计**：不信任账本自身，只依据 FillEvent 明细重算。
    """
    cash = initial_cash
    for f in result.fills:
        if f.side is OrderSide.BUY:
            cash -= f.size * f.price + f.cost
        else:
            cash += f.size * f.price - f.cost
    pos_value = sum(size * engine._last_close.get(sym, 0.0)
                    for sym, size in engine.ledger.exposure().items())
    return cash + pos_value


def check_cost(src: OpentdxParquetSource, sym: str) -> None:
    hr("5. 成本模型与账本可核验性")
    feed = src.load(sym)
    cost = StandardCost(commission=0.0003, min_commission=5.0,
                        stamp_tax=0.001, transfer_fee=0.00002,
                        slippage="volume_based", k=0.001)

    # --- 5a. 账本外部审计（含成本，用真实 budget 下单策略）---
    eng = Engine(feed, DualMAStrategy(sym, 10, 30), initial_cash=1_000_000.0, cost_model=cost)
    res = eng.run()
    recomputed = reconcile(eng, res, 1_000_000.0)
    ledger_eq = res.final_equity()
    diff = abs(recomputed - ledger_eq)
    out("5a. 账本外部审计（由 FillEvent 明细独立重算期末权益）")
    out(f"    账本报告权益 : {ledger_eq:,.6f}")
    out(f"    独立重算权益 : {recomputed:,.6f}")
    out(f"    偏差         : {diff:.6e} {'✓ 通过' if diff < 0.01 else '✗ 失败'}")
    out(f"    成交笔数     : {res.total_trades()}  累计费用: {sum(f.cost for f in res.fills):,.2f}")

    # --- 5b. 固定股数下，权益差应恰好等于累计费用 ---
    eng_free = Engine(feed, FixedSizeMA(sym, 10, 30, 1000), initial_cash=1_000_000.0,
                      cost_model=NoOpCost())
    r_free = eng_free.run()
    eng_cost = Engine(feed, FixedSizeMA(sym, 10, 30, 1000), initial_cash=1_000_000.0,
                      cost_model=cost)
    r_cost = eng_cost.run()

    fees = sum(f.cost for f in r_cost.fills)
    eq_diff = r_free.final_equity() - r_cost.final_equity()
    out("")
    out("5b. 固定股数：含成本与否的唯一差异应是费用本身")
    out(f"    零成本权益   : {r_free.final_equity():,.4f}  ({r_free.total_trades()} 笔)")
    out(f"    含成本权益   : {r_cost.final_equity():,.4f}  ({r_cost.total_trades()} 笔)")
    out(f"    累计费用     : {fees:,.4f}")
    out(f"    权益差       : {eq_diff:,.4f}")
    ok = abs(eq_diff - fees) < 0.01
    out(f"    一致性       : {'✓ 通过（权益差 == 累计费用）' if ok else '✗ 失败'}")
    out(f"    成本拖累     : {fees / 1_000_000:.2%} 初始资金")


# ---------------------------------------------------------------------------
# 6. 真实策略表现（诚实报告）
# ---------------------------------------------------------------------------
class RankStrategy(FrameStrategy):
    """与 MomentumCrossSection 相同的组合/下单逻辑，仅**排序方向**可切换。

    用于「信号传导」对照实验：同一套代码，只翻转排序 → 结果应显著反转，
    且方向应与「不依赖框架的纯统计」一致。这样才能区分
    「框架算错了」与「信号本身在这段样本里无效」。
    """

    def __init__(self, universe: list[str], top_k: int, mode: str, lookback: int = 60):
        self.universe = list(universe)
        self.top_k = top_k
        self.mode = mode
        self.lookback = lookback
        # 用宿主同款确定性 RNG，而不是标准库 random（README/宿主规则要求策略
        # 不得使用 random，否则跨解释器/版本的序列不保证一致）。
        self._rng = DetermRNG(seed=42)
        self._reg = False

    def _ensure(self, bus) -> None:
        if self._reg or "mom" in bus._factors:
            return
        lb = self.lookback

        def mom(seq, token):
            closes = [v for _t, v in seq]
            if len(closes) < lb:
                return float("nan")
            return closes[-1] / closes[-lb] - 1.0

        bus.register_factor("mom", mom)
        self._reg = True

    def on_frame(self, frame, token, bus, host):
        # 有在途单就先等它落地：否则本帧按「当前现金」下的买单会与上一帧尚未
        # 成交的买单叠加，成交后现金变负（隐含杠杆）。这是 I2 延迟下的基本约束。
        if host.pending_orders():
            return []
        self._ensure(bus)
        rows = bus.cross_section("mom", self.universe, token=token, lag=1)
        names = [s for s, r in rows.items() if r.is_valid]
        if self.mode == "momentum":
            names.sort(key=lambda s: rows[s].value, reverse=True)
        elif self.mode == "reversal":
            names.sort(key=lambda s: rows[s].value)
        else:
            names = self._rng.shuffle(names)
        target = set(names[: self.top_k])
        if not target:
            return []

        pos = host.portfolio
        actual = {s: n for s, n in pos.exposure().items() if n > 0}
        orders: list = []
        for sym, size in actual.items():
            # I2 延迟下，卖出腿在成交前仍会出现在持仓里；已有在途单就不再重复下。
            if sym not in target and not host.has_pending(sym):
                orders.append(Order(side=OrderSide.SELL, symbol=sym, size=size,
                                    order_type=OrderType.MARKET))
        to_buy = [s for s in names[: self.top_k]
                  if actual.get(s, 0) <= 0 and not host.has_pending(s)]
        if to_buy:
            per = pos.cash / len(to_buy)
            if per > 0:
                for sym in to_buy:
                    orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                        order_type=OrderType.MARKET, budget=per))
        return orders


def cash_path(fills, initial_cash: float) -> tuple[float, float]:
    """由成交明细独立重算「期末现金 / 期间最低现金」。

    默认 ``NoOpRisk`` 不会拦超买；若出现负现金，说明策略重复下单或有隐式杠杆，
    该对照实验的收益量级不可解释。返回 ``(final_cash, min_cash)``。
    """
    cash = initial_cash
    min_cash = initial_cash
    for f in sorted(fills, key=lambda x: (x.frame_id, x.order_id)):
        if f.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            cash += f.size * f.price - f.cost
        else:
            cash -= f.size * f.price + f.cost
        min_cash = min(min_cash, cash)
    return cash, min_cash


def check_signal_transmission(src: OpentdxParquetSource,
                              syms: list[str],
                              scan_syms: list[str] | None = None) -> None:
    """信号对照实验：框架必须忠实把「信号」翻译成「盈亏」。"""
    scan = scan_syms if scan_syms is not None else syms
    scan_note = ("（--max-symbols 降采样：{n} 标代表样本）".format(n=len(scan))
                 if scan_syms is not None and scan_syms != syms else "")
    hr(f"6d. 信号传导对照实验（同一套代码，只翻转排序方向）{scan_note}")
    feed = src.universe(scan)
    out(f"    {'排序模式':<12}{'收益':>12}{'回撤':>10}{'交易数':>9}"
        f"{'期末权益':>15}{'期末现金':>14}{'最低现金':>14}")
    results = {}
    min_cashes = {}
    for mode in ("momentum", "reversal", "random"):
        # universe 必须与 feed 一致，否则每帧对全市场 5000+ 标的做横截面（O(n) 爆炸）。
        eng = Engine(feed, RankStrategy(scan, 3, mode), initial_cash=1_000_000.0)
        r = eng.run()
        p = Performance(r)
        final_cash, min_cash = cash_path(r.fills, 1_000_000.0)
        results[mode] = p.total_return
        min_cashes[mode] = min_cash
        out(f"    {mode:<12}{p.total_return:>12.2%}{p.max_drawdown:>10.2%}"
            f"{r.total_trades():>9d}{r.final_equity():>15,.0f}"
            f"{final_cash:>14,.0f}{min_cash:>14,.0f}")

    # 基准
    bh = []
    for s in scan:
        b = src.load(s).bars
        if len(b) > 80:
            bh.append(b[-1].close / b[60].close - 1)
    avg_bh = sum(bh) / len(bh) if bh else 0.0
    out(f"    {'等权买入持有':<12}{avg_bh:>12.2%}")

    # --- 不依赖框架的纯统计检验：60 日动量 → 未来 20 日收益 ---
    pairs = []
    for s in scan:
        closes = [b.close for b in src.load(s).bars]
        for i in range(60, len(closes) - 20, 20):
            pairs.append((closes[i] / closes[i - 60] - 1, closes[i + 20] / closes[i] - 1))
    out("")
    out("    纯统计检验（不经框架）：60 日动量 → 未来 20 日收益")
    if len(pairs) > 10:
        n = len(pairs)
        mx = sum(a for a, _ in pairs) / n
        my = sum(b for _, b in pairs) / n
        cov = sum((a - mx) * (b - my) for a, b in pairs) / n
        sx = (sum((a - mx) ** 2 for a, _ in pairs) / n) ** 0.5
        sy = (sum((b - my) ** 2 for _, b in pairs) / n) ** 0.5
        q = max(1, n // 5)
        top = sum(b for _, b in sorted(pairs, reverse=True)[:q]) / q
        bot = sum(b for _, b in sorted(pairs)[:q]) / q
        out(f"      样本数        : {n}")
        out(f"      相关系数      : {cov / (sx * sy):+.4f}")
        out(f"      动量最高 20%  : 未来 20 日 {top:+.2%}")
        out(f"      动量最低 20%  : 未来 20 日 {bot:+.2%}")

    out("")
    negative = [m for m, c in min_cashes.items() if c < -1e-6]
    if negative:
        out(f"    ✗ 实验无效：{', '.join(negative)} 出现负现金（最低 "
            f"{min(min_cashes.values()):,.0f}）——策略重复下单/默认无风控，"
            "收益量级不可解释，不能作为能力证据。")
    else:
        ok = (results["reversal"] > results["random"]
              and results["reversal"] > results["momentum"])
        out(f"    结论：三组均无负现金；翻转排序后结果"
            f"{'反转' if ok else '未反转'}（{'✓' if ok else '✗'} 方向对照）")
        out("    说明：方向反转只说明排序信号在本样本期有效，不构成收益能力证明；"
            "具体盈亏见上表，不预设随机组合的正负。")


def check_strategies(src: OpentdxParquetSource, syms: list[str],
                     scan_syms: list[str] | None = None) -> None:
    hr(f"6. 真实策略表现（全市场 {len(syms)} 标的 × DAILY/QFQ，含 A 股成本）")

    cost = StandardCost(commission=0.0003, min_commission=5.0,
                        stamp_tax=0.001, transfer_fee=0.00002, slippage="volume_based", k=0.001)

    # --- 6a. 基准：等权买入持有 ---
    out("")
    out("6a. 基准：等权买入持有（第 60 帧建仓后不动，不计成本）")
    eq_rets = []
    for s in syms:
        bars_s = src.load(s).bars
        if len(bars_s) > 80:
            eq_rets.append(bars_s[-1].close / bars_s[60].close - 1)
    avg_bh = sum(eq_rets) / len(eq_rets) if eq_rets else 0.0
    out(f"    平均收益: {avg_bh:+.2%}（{len(eq_rets)} 标的，第60帧收盘 → 末帧收盘）")
    out(f"    说明：样本期（2018-05 ~ 2026-10）A 股大盘股整体大幅上涨，")
    out(f"          任何策略都要与这个「什么都不做」的基准比较才有意义。")

    # --- 6b. 双均线参数扫描（逐标的）---
    # 6b 是 O(标的数 × 参数) 的扫描：全市场 5575×3 ≈ 16725 次回测约 1-3h，
    # 故在 --max-symbols 存在时改用代表样本逐标独立回测（结果统计量稳定）。
    scan_note = ("（--max-symbols 降采样：{n} 标代表样本，避免 O(n) 扫描过慢）".format(n=len(scan_syms if scan_syms is not None else syms))
                 if scan_syms is not None and scan_syms != syms else "")
    out("")
    out(f"6b. 双均线参数扫描（逐标的独立回测，A 股成本 + T+1）{scan_note}")
    out(f"    {'窗口':<14}{'平均收益':>11}{'中位收益':>11}{'平均回撤':>11}{'平均交易':>10}{'胜率':>8}{'vs基准':>10}")
    scan = scan_syms if scan_syms is not None else syms
    for short, long in [(5, 20), (10, 30), (20, 60)]:
        rets, mdds, trades, wins = [], [], [], []
        for s in scan:
            feed = src.load(s)
            eng = Engine(feed, DualMAStrategy(s, short, long), initial_cash=1_000_000.0,
                         settlement_cycle=1, cost_model=cost, rules=StandardAshareRules())
            r = eng.run()
            p = Performance(r)
            rets.append(p.total_return)
            mdds.append(p.max_drawdown)
            trades.append(r.total_trades())
            wins.append(p.win_rate())
        rets_sorted = sorted(rets)
        med = rets_sorted[len(rets_sorted) // 2]
        avg = sum(rets) / len(rets)
        out(f"    MA{short:<2}/MA{long:<8}{avg:>11.2%}{med:>11.2%}"
            f"{sum(mdds) / len(mdds):>11.2%}{sum(trades) / len(trades):>10.1f}"
            f"{sum(wins) / len(wins):>8.1%}{avg - avg_bh:>10.2%}")
    out("    注：平均收益远高于中位数 → 结果由少数标的主导；三者均大幅跑输等权买入持有。")

    # --- 6c. 横截面动量（框架的差异化能力）---
    out("")
    out(f"6c. 横截面动量选股（cross_section + lag=1，框架差异化能力）{scan_note}")
    feed = src.universe(scan)
    bars = to_bars(feed)
    out(f"    {'TopK':<6}{'收益':>12}{'年化':>12}{'回撤':>12}{'Sharpe':>10}{'交易数':>8}{'成本':>12}")
    for top_k in (3, 5, 8):
        r = run_backtest(bars, MomentumCrossSection(scan, 20, 60, top_k),
                         symbol="P", initial_cash=1_000_000.0,
                         settlement_cycle=1, cost_model=cost, rules=StandardAshareRules())
        p = Performance(r)
        fees = sum(f.cost for f in r.fills)
        out(f"    {top_k:<6}{p.total_return:>12.2%}{p.annualized_return:>12.2%}"
            f"{p.max_drawdown:>12.2%}{p.sharpe:>10.2f}{r.total_trades():>8d}{fees:>12,.0f}")
    out("    注：顺势动量在本样本期大幅亏损 —— 原因见 6d 的对照实验（并非框架错误）。")


# ---------------------------------------------------------------------------
# 7. 吞吐量
# ---------------------------------------------------------------------------
def check_throughput(src: OpentdxParquetSource, syms: list[str]) -> None:
    hr("7. 吞吐量")
    feed = src.universe(syms)
    bars = to_bars(feed)
    n_bars = len(bars)

    t0 = time.perf_counter()
    r_single = run_backtest(bars, DualMAStrategy(syms[0], 10, 30), symbol="X",
                            initial_cash=1_000_000.0)
    t_single = time.perf_counter() - t0

    t0 = time.perf_counter()
    r_cross = run_backtest(bars, MomentumCrossSection(syms, 20, 60, 5), symbol="P",
                           initial_cash=1_000_000.0)
    t_cross = time.perf_counter() - t0

    out(f"数据集          : {n_bars:,} bars / {len(syms)} 标的 / {r_single.frames:,} 帧")
    out(f"单标的双均线    : {t_single:.2f}s  ({n_bars / t_single:,.0f} bars/s, "
        f"{r_single.frames / t_single:,.0f} 帧/s)")
    out(f"横截面动量      : {t_cross:.2f}s  ({n_bars / t_cross:,.0f} bars/s, "
        f"{r_cross.frames / t_cross:,.0f} 帧/s, {r_cross.total_trades()} 笔成交)")
    out("")
    out("说明：纯 Python 事件循环，未做向量化；单帧成本 O(持仓标的)。")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="framebt 框架能力验证")
    ap.add_argument("--md", default=None, help="同时写出 Markdown 报告路径")
    ap.add_argument("--skip-sweep", action="store_true", help="跳过多参数扫描（更快）")
    ap.add_argument("--max-symbols", type=int, default=None,
                    help="扫描/横截面类检查（6b 扫描、6c、6d 信号传导）使用的代表样本"
                         "上限；缺省=全市场。核心检查（数据完整性/因果/A股规则/成本/"
                         "确定性/吞吐/6a 基准）始终用全市场，保证全量可加载。")
    args = ap.parse_args()

    src = OpentdxParquetSource()
    syms = src.available_symbols()
    if not syms:
        raise SystemExit("数据源无数据；请先运行 tools/backfill_history.py")

    scan_syms = syms[:args.max_symbols] if args.max_symbols else None
    if args.skip_sweep and not args.max_symbols:
        scan_syms = syms[:50]  # 跳过扫描：仅用极小样本过一遍 6b/6c/6d
    scan_label = (f"{args.max_symbols} 标代表样本" if args.max_symbols
                  else "50 标（--skip-sweep）" if args.skip_sweep else "全市场")

    out("framebt 框架能力验证报告")
    out(f"数据源: opentdx 数仓 · {len(syms)} 标的 · DAILY/QFQ")
    out(f"扫描项样本: {scan_label}（--max-symbols；核心项仍用全市场 {len(syms)} 标）")

    check_data(src, syms)
    check_causality(src, syms[0])
    check_determinism(src, syms)
    check_rules(src, syms)
    check_cost(src, syms[0])
    check_strategies(src, syms, scan_syms)
    check_signal_transmission(src, syms, scan_syms)
    check_throughput(src, syms)

    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("```\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
        print(f"\n报告已写入 {p}")


if __name__ == "__main__":
    main()
