"""能力门禁：验证 framebt「真实可用」——不是能跑出数字，而是**算得对、挡得住、可复现、改坏会被抓**。

预登记与判据：``docs/GATE_USABILITY.md``。本文件的纪律：

- oracle（独立复算）**不 import** ``framebt.ledger/cost/execution`` 的实现，只用 FillEvent
  的公开字段按文档公式重写；
- 每个断言都对应预登记中的 G1–G6 之一；
- 已知建模边界用 ``xfail(strict=True)`` 显式登记（G7），不是「通过」；
- 本文件必须能被 ``tools/mutate_gate.py`` 当作变异体杀手（G8）。

运行：
    $env:PYTHONPATH="src"; python -m pytest -q tests/test_capability_gate.py
"""

from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import pytest

from framebt import (
    BacktestResult,
    Bar,
    BarFeed,
    CausalToken,
    CausalityViolation,
    CorporateAction,
    DataBus,
    Engine,
    FillEvent,
    FrameStrategy,
    Ledger,
    MarketFrame,
    MatchingEngine,
    NoOpCost,
    NoOpRisk,
    NoOpRules,
    Order,
    OrderSide,
    OrderState,
    OrderType,
    Performance,
    StandardAshareRules,
    StandardCost,
    StandardRisk,
    run_backtest,
)
from framebt.strategies import DualMAStrategy

ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-9


# ===========================================================================
# 辅助：数据、订单、独立 oracle
# ===========================================================================

def _mk(t: int, o: float, h: float, l: float, c: float, v: float = 1_000_000.0,
        s: str = "S") -> dict:
    return {"timestamp": t, "symbol": s, "open": o, "high": h, "low": l,
            "close": c, "volume": v}


def _flat(n: int, px: float = 10.0, v: float = 1_000_000.0, s: str = "S") -> list[dict]:
    return [_mk(i, px, px, px, px, v, s) for i in range(n)]


def _feed(rows: list[dict], symbol: str = "S") -> BarFeed:
    return BarFeed.from_dicts(rows, symbol=symbol)


def _oracle_cost(side, size, price, volume, *, commission=3e-4, min_commission=5.0,
                 stamp_tax=5e-4, transfer_fee=2e-5, slip_k=5e-4, borrow_fee=0.0) -> float:
    """按文档口径独立重写成本公式（不调用 framebt.cost）。"""
    notional = size * price
    fee = max(notional * commission, min_commission)
    if side is OrderSide.SELL:
        fee += notional * stamp_tax
    fee += notional * transfer_fee
    if side is OrderSide.SELL_SHORT:
        fee += notional * borrow_fee
    if volume and volume > 0:
        fee += notional * slip_k * (size / volume)
    return fee


class _Oracle:
    """独立账本：现金 + 净持仓；不依赖 framebt.ledger。"""

    def __init__(self, cash: float):
        self.cash = float(cash)
        self.pos: dict[str, float] = defaultdict(float)

    def apply(self, f: FillEvent) -> None:
        notional = f.size * f.price
        if f.side in (OrderSide.SELL, OrderSide.SELL_SHORT):
            self.cash += notional - f.cost
            self.pos[f.symbol] -= f.size
        else:
            self.cash -= notional + f.cost
            self.pos[f.symbol] += f.size

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + sum(q * marks.get(s, 0.0) for s, q in self.pos.items())


class _BuyOnce(FrameStrategy):
    def __init__(self, size: float = 0.0, budget: float = 0.0,
                 otype: OrderType = OrderType.MARKET, price: float = 0.0,
                 symbol: str = "S"):
        self.order = Order(side=OrderSide.BUY, symbol=symbol, size=size,
                           budget=budget, order_type=otype, price=price)
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return [self.order]


class _BuyThenSell(FrameStrategy):
    """帧 0 买入；成交回报时挂出等量卖单（下一帧才可成交，T+1 天然满足）。"""

    def __init__(self, size: float, symbol: str = "S"):
        self.size = size
        self.symbol = symbol
        self.sell_order: Order | None = None
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return [Order(side=OrderSide.BUY, symbol=self.symbol, size=self.size)]

    def on_fill(self, fill):
        if fill.side is OrderSide.BUY:
            self.sell_order = Order(side=OrderSide.SELL, symbol=fill.symbol,
                                    size=fill.size)
            return [self.sell_order]
        return []


class _BuyThenStop(FrameStrategy):
    def __init__(self, trigger: float, size: float = 100.0):
        self.trigger = trigger
        self.size = size
        self.stop_order: Order | None = None
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return [Order(side=OrderSide.BUY, symbol="S", size=self.size)]

    def on_fill(self, fill):
        if fill.side is OrderSide.BUY:
            self.stop_order = Order(side=OrderSide.SELL, symbol=fill.symbol,
                                    size=fill.size, order_type=OrderType.STOP,
                                    price=self.trigger)
            return [self.stop_order]
        return []


class _BuyThenTrailing(FrameStrategy):
    def __init__(self, trail_pct: float, reference: float, size: float = 100.0):
        self.trail_pct = trail_pct
        self.reference = reference
        self.size = size
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return [Order(side=OrderSide.BUY, symbol="S", size=self.size)]

    def on_fill(self, fill):
        if fill.side is OrderSide.BUY:
            return [Order(side=OrderSide.SELL, symbol=fill.symbol, size=fill.size,
                          order_type=OrderType.TRAILING_STOP,
                          trail_pct=self.trail_pct, price=self.reference)]
        return []


class _ThreeLimits(FrameStrategy):
    def __init__(self, price: float, size: float = 100.0, n: int = 3):
        self.price = price
        self.size = size
        self.n = n
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return [Order(side=OrderSide.BUY, symbol="S", size=self.size,
                      order_type=OrderType.LIMIT, price=self.price)
                for _ in range(self.n)]


class _NOrders(FrameStrategy):
    """帧 0 一次性提交 n 笔同向订单（用于拆单/成交量共享检查）。"""

    def __init__(self, orders: list[Order]):
        self.orders = orders
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return self.orders


class _BuySellSameFrame(FrameStrategy):
    def __init__(self, size: float = 100.0):
        self.size = size
        self.sent = False

    def on_frame(self, frame, token, bus, host):
        if self.sent:
            return []
        self.sent = True
        return [Order(side=OrderSide.BUY, symbol="S", size=self.size),
                Order(side=OrderSide.SELL, symbol="S", size=self.size)]


def _trend_rows(n: int, seed: int = 0, symbol: str = "S", scale: float = 1.0) -> list[dict]:
    rng = random.Random(seed)
    px = 10.0
    rows = []
    for t in range(n):
        c = px * (1 + rng.uniform(-0.02, 0.025))
        o = px
        h = max(o, c) * (1 + rng.uniform(0.0, 0.01))
        l = min(o, c) * (1 - rng.uniform(0.0, 0.01))
        rows.append(_mk(t, o * scale, h * scale, l * scale, c * scale, 1_000_000.0, symbol))
        px = c
    return rows


def _replay(rows: list[dict], res: BacktestResult, initial_cash: float,
            symbols: list[str]) -> tuple[_Oracle, dict[int, list[FillEvent]]]:
    """独立重放：逐帧应用 FillEvent，按当帧收盘（静态帧用最后收盘）计值。"""
    bars = {(r["timestamp"], r["symbol"]): r for r in rows}
    last_close: dict[str, float] = {}
    for t in range(max((r["timestamp"] for r in rows), default=-1) + 1):
        for s in symbols:
            r = bars.get((t, s))
            if r is not None:
                last_close[s] = r["close"]

    def mark(fid: int, sym: str) -> float:
        r = bars.get((fid, sym))
        return r["close"] if r is not None else last_close.get(sym, 0.0)

    fills_by_frame: dict[int, list[FillEvent]] = defaultdict(list)
    for f in res.fills:
        fills_by_frame[f.frame_id].append(f)

    oracle = _Oracle(initial_cash)
    for fid in range(len(res.equity_curve)):
        for f in fills_by_frame.get(fid, []):
            oracle.apply(f)
        marks = {s: mark(fid, s) for s in symbols}
        assert oracle.equity(marks) == pytest.approx(
            res.equity_curve[fid][1], rel=1e-9, abs=1e-9
        ), f"帧 {fid} 权益与独立重放不一致"
    return oracle, fills_by_frame


# ===========================================================================
# G1 手算金标准
# ===========================================================================

def test_g1_buy_and_hold_exact_cash_cost_equity():
    rows = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 10, 11, 9.8, 10.5),
            _mk(2, 11, 12, 10.5, 11.5), _mk(3, 12, 12.5, 11.5, 12)]
    cost = StandardCost(commission=3e-4, min_commission=5.0, stamp_tax=5e-4,
                        transfer_fee=2e-5, slippage="proportional", k=5e-4)
    eng = Engine(_feed(rows), _BuyOnce(budget=10_000.0), initial_cash=100_000.0,
                 cost_model=cost, risk=NoOpRisk(), rules=NoOpRules())
    res = eng.run()

    assert len(res.fills) == 1
    f = res.fills[0]
    assert (f.frame_id, f.size, f.price) == (1, 1_000.0, 10.0)
    # 佣金 max(10000×3bp, 5) + 过户 0.2 + 比例滑点 5 = 10.2
    assert f.cost == pytest.approx(10.2, abs=EPS)
    assert eng.ledger.cash == pytest.approx(89_989.8, abs=EPS)
    assert [e for _, e in res.equity_curve] == pytest.approx(
        [100_000.0, 100_489.8, 101_489.8, 101_989.8], abs=EPS
    )


def test_g1_roundtrip_realized_pnl_and_cash():
    rows = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 10, 11, 9.8, 10.5),
            _mk(2, 11, 12, 10.5, 11.5), _mk(3, 11, 12, 11, 11.5)]
    cost = StandardCost(commission=3e-4, min_commission=5.0, stamp_tax=5e-4,
                        transfer_fee=2e-5, slippage="proportional", k=5e-4)
    eng = Engine(_feed(rows), _BuyThenSell(size=1_000.0), initial_cash=100_000.0,
                 cost_model=cost, risk=NoOpRisk(), rules=NoOpRules())
    res = eng.run()

    assert [(f.frame_id, f.side.value, f.size, f.price) for f in res.fills] == [
        (1, "buy", 1_000.0, 10.0), (2, "sell", 1_000.0, 11.0)
    ]
    # 卖出成本 = max(11000×3bp,5) + 印花 5.5 + 过户 0.22 + 滑点 5.5 = 16.22
    assert res.fills[1].cost == pytest.approx(16.22, abs=EPS)
    assert eng.ledger.cash == pytest.approx(100_973.58, abs=EPS)
    assert eng.ledger.position("S").realized_pnl == pytest.approx(1_000.0, abs=EPS)
    assert eng.ledger.position("S").net_size() == 0


def test_g1_delist_cash_settlement():
    led = Ledger(cash=100_000.0)
    led.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                             size=100, price=10.0, frame_id=0, timestamp=0))
    assert led.cash == pytest.approx(99_000.0)
    led.apply_corporate_action(
        CorporateAction(symbol="S", kind="delist", date=1, ratio=8.0)
    )
    assert led.cash == pytest.approx(99_800.0)
    assert led.position("S").net_size() == 0
    assert led.mark_to_market({}) == pytest.approx(99_800.0)


def test_g1_split_and_dividend():
    led = Ledger(cash=100_000.0)
    led.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                             size=100, price=10.0, frame_id=0, timestamp=0))
    led.apply_corporate_action(
        CorporateAction(symbol="S", kind="split", date=1, ratio=2.0)
    )
    pos = led.position("S")
    assert pos.net_size() == 200
    assert pos.avg_price() == pytest.approx(5.0)
    led.apply_corporate_action(
        CorporateAction(symbol="S", kind="dividend", date=2, cash_per_share=0.5)
    )
    assert led.cash == pytest.approx(99_100.0)
    assert led.mark_to_market({"S": 5.5}) == pytest.approx(100_200.0)


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL, OrderSide.SELL_SHORT])
@pytest.mark.parametrize("size", [1_000.0, 10_000.0, 100_000.0])
def test_g1_fee_formula_matrix(side, size):
    cfg = dict(commission=2.5e-4, min_commission=5.0, stamp_tax=5e-4,
               transfer_fee=1e-5, borrow_fee=8e-4)
    cm = StandardCost(slippage="volume_based", k=3e-4, **cfg)
    frame = MarketFrame(0, 0, 1)
    frame.add_bar(Bar(timestamp=0, symbol="S", open=10, high=10, low=10,
                      close=10, volume=50_000.0))
    f = FillEvent(order_id=1, side=side, symbol="S", size=size, price=10.0,
                  frame_id=0, timestamp=0)
    expected = _oracle_cost(side, size, 10.0, 50_000.0, slip_k=3e-4,
                            commission=2.5e-4, min_commission=5.0,
                            stamp_tax=5e-4, transfer_fee=1e-5, borrow_fee=8e-4)
    assert cm.cost_for(f, frame) == pytest.approx(expected, rel=1e-12, abs=EPS)


def test_g1_performance_metrics_known_answer():
    res = BacktestResult(
        equity_curve=[(0, 100.0), (1, 110.0), (2, 99.0), (3, 120.0)],
        fills=[
            FillEvent(1, OrderSide.BUY, "S", 100, 10.0, 0, 0),
            FillEvent(2, OrderSide.SELL, "S", 100, 12.0, 1, 1),
        ],
        frames=3,
    )
    p = Performance(res, period_frames=3)
    assert p.total_return == pytest.approx(0.2, abs=EPS)
    assert p.annualized_return == pytest.approx(0.2, abs=EPS)
    assert p.max_drawdown == pytest.approx(0.1, abs=EPS)
    assert p.volatility == pytest.approx(0.27386, rel=1e-4)
    # 2026-10-29 起 Sharpe 统一为标准定义 mean/std×√ppy；旧口径保留为 sharpe_cagr。
    assert p.sharpe == pytest.approx(0.77458, rel=1e-4)
    assert p.sharpe_cagr == pytest.approx(0.20 / 0.27386, rel=1e-4)
    assert p.calmar == pytest.approx(2.0, rel=1e-9)
    assert p.win_rate() == pytest.approx(1.0)

    loss = BacktestResult(fills=[
        FillEvent(1, OrderSide.BUY, "S", 100, 10.0, 0, 0),
        FillEvent(2, OrderSide.SELL, "S", 100, 9.0, 1, 1),
    ])
    assert Performance(loss).win_rate() == pytest.approx(0.0)


def test_g1_limit_fills_at_limit_not_daily_low():
    rows = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 12, 12, 9.0, 10.5),
            _mk(2, 10, 10.5, 9.5, 10)]
    res = run_backtest(rows, _BuyOnce(size=100.0, otype=OrderType.LIMIT, price=10.0),
                       symbol="S", initial_cash=100_000.0, rules=NoOpRules())
    assert [(f.frame_id, f.price) for f in res.fills] == [(1, 10.0)]


@pytest.mark.parametrize("frame2, expected", [
    ((9.0, 9.4, 8.5, 8.8), 9.0),    # 跳空低开：按开盘价，不按触发价
    ((10.0, 10.2, 9.2, 9.3), 9.5),  # 帧内跌破：按触发价
])
def test_g1_stop_gap_and_breakdown(frame2, expected):
    rows = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 10, 10.5, 9.5, 10),
            _mk(2, *frame2), _mk(3, 9.5, 9.6, 9.4, 9.5)]
    res = run_backtest(rows, _BuyThenStop(trigger=9.5, size=100.0),
                       symbol="S", initial_cash=100_000.0,
                       rules=NoOpRules(), cost_model=NoOpCost())
    sells = [f for f in res.fills if f.side is OrderSide.SELL]
    assert len(sells) == 1
    assert sells[0].frame_id == 2
    assert sells[0].price == pytest.approx(expected)


def test_g1_trailing_stop_trigger_price():
    rows = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 10, 10.5, 9.5, 10),
            _mk(2, 11.0, 12.0, 11.5, 11.8), _mk(3, 11.6, 11.7, 11.2, 11.3)]
    res = run_backtest(rows, _BuyThenTrailing(trail_pct=0.05, reference=10.0),
                       symbol="S", initial_cash=100_000.0,
                       rules=NoOpRules(), cost_model=NoOpCost())
    sells = [f for f in res.fills if f.side is OrderSide.SELL]
    assert len(sells) == 1
    assert sells[0].frame_id == 3
    assert sells[0].price == pytest.approx(11.4)


# ===========================================================================
# G2 独立 oracle + 随机订单
# ===========================================================================

class _Chaos(FrameStrategy):
    """确定性随机订单策略：每标的每帧最多一单，覆盖 MARKET/LIMIT/IOC/FOK。"""

    def __init__(self, seed: int, symbols: list[str]):
        self.rng = random.Random(seed)
        self.symbols = symbols
        self.records: list[tuple[Order, int]] = []

    def on_frame(self, frame, token, bus, host):
        orders: list[Order] = []
        for sym in self.symbols:
            if not frame.has_bar(sym):
                continue
            px = frame.close(sym)
            pos = host.portfolio.position(sym).net_size()
            u = self.rng.random()
            if u < 0.15 and pos > 0:
                size = min(pos, float(self.rng.choice([100, 200, 300])))
                orders.append(Order(side=OrderSide.SELL, symbol=sym, size=size))
            elif u < 0.32:
                typ = self.rng.choice([OrderType.MARKET, OrderType.LIMIT,
                                       OrderType.IOC, OrderType.FOK])
                size = float(self.rng.choice([100, 200, 400]))
                kwargs = {}
                if typ is OrderType.LIMIT:
                    kwargs["price"] = round(px * (0.995 + 0.01 * self.rng.random()), 2)
                orders.append(Order(side=OrderSide.BUY, symbol=sym, size=size,
                                    order_type=typ, **kwargs))
        for o in orders:
            self.records.append((o, frame.frame_id))
        return orders


def _chaos_rows(seed: int, n: int, symbols: list[str]) -> list[dict]:
    rng = random.Random(10_000 + seed)
    px = {s: 10.0 + 5 * rng.random() for s in symbols}
    rows: list[dict] = []
    for t in range(n):
        for s in symbols:
            o = round(px[s], 2)
            c = round(max(1.0, o * (1 + rng.uniform(-0.03, 0.03))), 2)
            h = round(max(o, c) * (1 + rng.uniform(0.0, 0.01)), 2)
            l = round(min(o, c) * (1 - rng.uniform(0.0, 0.01)), 2)
            v = float(rng.choice([2_000, 5_000, 10_000, 50_000]))
            rows.append(_mk(t, o, h, l, c, v, s))
            px[s] = c
    return rows


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5])
def test_g2_fuzz_replay_cash_equity_cost_positions(seed):
    symbols = ["A", "B", "C"]
    rows = _chaos_rows(seed, 50, symbols)
    strat = _Chaos(seed, symbols)
    eng = Engine(
        BarFeed.from_dicts(rows), strat,
        initial_cash=1_000_000.0,
        cost_model=StandardCost(commission=3e-4, min_commission=5.0,
                                stamp_tax=5e-4, transfer_fee=2e-5,
                                slippage="volume_based", k=5e-4),
        risk=StandardRisk(max_position_pct=0.8, max_order_pct=0.5),
        rules=StandardAshareRules(limit=0.10),
    )
    res = eng.run()

    oracle, fills_by_frame = _replay(rows, res, 1_000_000.0, symbols)
    assert eng.ledger.cash == pytest.approx(oracle.cash, rel=1e-9, abs=EPS)
    assert eng.ledger.exposure() == {
        s: pytest.approx(q, abs=EPS) for s, q in oracle.pos.items() if abs(q) > EPS
    }

    bars = {(r["timestamp"], r["symbol"]): r for r in rows}
    # 逐笔成本 / 成交价区间 / 单笔量 / I2 / T+1
    placement = {o.id: fid for o, fid in strat.records}
    assert res.fills, "随机策略应至少产生一笔成交"
    for f in res.fills:
        r = bars.get((f.frame_id, f.symbol))
        vol = r["volume"] if r is not None else 0.0
        expected_cost = _oracle_cost(f.side, f.size, f.price, vol)
        assert f.cost == pytest.approx(expected_cost, rel=1e-12, abs=EPS), \
            f"帧 {f.frame_id} {f.symbol} 成本不符"
        if r is not None:
            assert r["low"] - EPS <= f.price <= r["high"] + EPS, \
                f"成交价越出 Bar 区间：{f.price} not in [{r['low']},{r['high']}]"
            if vol > 0:
                assert f.size <= vol + EPS
        assert f.frame_id >= placement[f.order_id] + 1, "违反 I2 延迟"

    legs: dict[str, list[list[float]]] = defaultdict(list)
    for f in res.fills:
        if f.side is OrderSide.BUY:
            legs[f.symbol].append([f.size, f.frame_id + 1])  # T+1
        else:
            remaining = f.size
            for leg in sorted(legs[f.symbol], key=lambda x: x[1]):
                if remaining <= 0:
                    break
                if leg[1] <= f.frame_id and leg[0] > 0:
                    take = min(leg[0], remaining)
                    leg[0] -= take
                    remaining -= take
            assert remaining <= EPS, f"T+1 违约：帧 {f.frame_id} 卖出了未结算持仓"

    # 每帧每标的成交量不得被同一组订单超卖（本策略每标的每帧至多一单）
    per_frame = defaultdict(float)
    for f in res.fills:
        per_frame[(f.frame_id, f.symbol)] += f.size
    for (fid, sym), total in per_frame.items():
        r = bars.get((fid, sym))
        if r is not None and r["volume"] > 0:
            assert total <= r["volume"] + EPS, \
                f"帧 {fid} {sym} 成交 {total} 超过成交量 {r['volume']}"


# ===========================================================================
# G3 因果性攻击 + 前缀不变性
# ===========================================================================

def _preloaded_bus(n: int = 20) -> DataBus:
    bus = DataBus()
    for i in range(n):
        bus.record("close", "S", i, float(i))
    bus.set_now(5)
    return bus


def test_g3_forged_token_cannot_read_future():
    bus = _preloaded_bus(20)
    forged = CausalToken(10**9, 999)
    assert bus.get("close", "S", token=forged) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [ts for ts, _ in bus.series("close", "S", forged)] == [0, 1, 2, 3, 4, 5]
    assert bus.cross_section("close", ["S"], forged)["S"].value == 5.0
    assert bus.forgery_attempts == 3


def test_g3_negative_lag_is_rejected_on_every_read_api():
    bus = _preloaded_bus(20)
    tok = CausalToken(5, 5)
    with pytest.raises(CausalityViolation):
        bus.get("close", "S", token=tok, lag=-1)
    with pytest.raises(CausalityViolation):
        bus.series("close", "S", tok, lag=-1)
    with pytest.raises(CausalityViolation):
        bus.cross_section("close", ["S"], tok, lag=-1)


class _Attacker(FrameStrategy):
    def __init__(self):
        self.errors: list[str] = []

    def on_frame(self, frame, token, bus, host):
        if frame.frame_id != 0:
            return []
        for fn in (
            lambda: bus.record("close", "S", 0, 9999.0),
            lambda: bus.set_now(10**9),
            lambda: bus.reset(),
            lambda: bus.enable_write_guard(object()),
        ):
            try:
                fn()
            except CausalityViolation as exc:
                self.errors.append(str(exc))
        return []


def test_g3_write_guard_blocks_history_and_clock_tampering():
    rows = _flat(5, 10.0)
    strat = _Attacker()
    eng = Engine(_feed(rows), strat)
    res = eng.run()
    assert len(strat.errors) == 4, strat.errors
    # 历史未被追加、权威时间未被改写
    assert eng.bus.get("close", "S", token=CausalToken(10**9, 0)) == [10.0] * 5
    assert eng.bus._now_ns == res.end_ns
    assert len(res.equity_curve) == 5


def test_g3_prefix_invariance_truncation():
    rows = _trend_rows(120, seed=3)
    full = run_backtest(rows, DualMAStrategy("S", 5, 20), symbol="S",
                        initial_cash=100_000.0, cost_model=StandardCost())
    prefix = run_backtest(rows[:80], DualMAStrategy("S", 5, 20), symbol="S",
                          initial_cash=100_000.0, cost_model=StandardCost())
    assert full.equity_curve[:80] == prefix.equity_curve[:80]
    assert [f for f in full.fills if f.frame_id < 80] == \
           [f for f in prefix.fills if f.frame_id < 80]


def test_g3_future_shock_does_not_change_past():
    rows = _trend_rows(120, seed=4)
    shocked = [dict(r) for r in rows]
    for r in shocked[80:]:
        for k in ("open", "high", "low", "close"):
            r[k] *= 5.0
    base = run_backtest(rows, DualMAStrategy("S", 5, 20), symbol="S",
                        initial_cash=100_000.0, cost_model=StandardCost())
    alt = run_backtest(shocked, DualMAStrategy("S", 5, 20), symbol="S",
                       initial_cash=100_000.0, cost_model=StandardCost())
    assert base.equity_curve[:80] == alt.equity_curve[:80]
    assert [f for f in base.fills if f.frame_id < 80] == \
           [f for f in alt.fills if f.frame_id < 80]


# ===========================================================================
# G4 元形不变
# ===========================================================================

def test_g4_input_order_invariance():
    rows = _trend_rows(60, seed=5)
    shuffled = [dict(r) for r in rows]
    random.Random(7).shuffle(shuffled)
    a = run_backtest(rows, DualMAStrategy("S", 5, 20), symbol="S",
                     initial_cash=100_000.0, cost_model=StandardCost())
    b = run_backtest(shuffled, DualMAStrategy("S", 5, 20), symbol="S",
                     initial_cash=100_000.0, cost_model=StandardCost())
    assert a.equity_curve == b.equity_curve
    assert a.fills == b.fills


def test_g4_symbol_rename_invariance():
    rows_s = _trend_rows(60, seed=6, symbol="S")
    rows_x = _trend_rows(60, seed=6, symbol="X")
    a = run_backtest(rows_s, DualMAStrategy("S", 5, 20), symbol="S",
                     initial_cash=100_000.0, cost_model=StandardCost())
    b = run_backtest(rows_x, DualMAStrategy("X", 5, 20), symbol="X",
                     initial_cash=100_000.0, cost_model=StandardCost())
    assert a.equity_curve == b.equity_curve
    assert [(f.frame_id, f.side, f.size, f.price, f.cost) for f in a.fills] == \
           [(f.frame_id, f.side, f.size, f.price, f.cost) for f in b.fills]


def test_g4_split_order_equivalence_zero_cost():
    rows = _flat(5, 10.0)
    one = run_backtest(rows, _BuyOnce(size=200.0), symbol="S",
                       initial_cash=100_000.0, cost_model=NoOpCost())
    two = run_backtest(rows, _NOrders([
        Order(side=OrderSide.BUY, symbol="S", size=100.0),
        Order(side=OrderSide.BUY, symbol="S", size=100.0),
    ]), symbol="S", initial_cash=100_000.0, cost_model=NoOpCost())
    assert one.final_equity() == pytest.approx(two.final_equity(), abs=EPS)
    assert sum(f.size for f in one.fills) == 200.0
    assert sum(f.size for f in two.fills) == 200.0


def test_g4_price_scaling():
    base = _trend_rows(6, seed=8, scale=1.0)
    scaled = _trend_rows(6, seed=8, scale=10.0)
    # 比例成本（无最低佣金/固定费用）下，价格与本金 ×10 ⇒ 权益 ×10
    pct = StandardCost(commission=0.0, min_commission=0.0, stamp_tax=0.0,
                       transfer_fee=0.0, slippage="proportional", k=5e-4)
    a = run_backtest(base, _BuyThenSell(size=100.0), symbol="S",
                     initial_cash=100_000.0, cost_model=pct,
                     rules=NoOpRules())
    b = run_backtest(scaled, _BuyThenSell(size=100.0), symbol="S",
                     initial_cash=1_000_000.0, cost_model=pct,
                     rules=NoOpRules())
    assert a.final_equity() * 10.0 == pytest.approx(b.final_equity(), rel=1e-9)
    assert [f.price * 10.0 for f in a.fills] == pytest.approx(
        [f.price for f in b.fills], rel=1e-9
    )


def test_g4_irrelevant_symbol_does_not_change_result():
    rows = _trend_rows(40, seed=9)
    extra = [dict(r, symbol="Z", open=r["open"] * 5, high=r["high"] * 5,
                  low=r["low"] * 5, close=r["close"] * 5) for r in rows]
    a = run_backtest(rows, _BuyOnce(size=100.0), symbol="S",
                     initial_cash=100_000.0, cost_model=StandardCost())
    b = run_backtest(rows + extra, _BuyOnce(size=100.0), symbol="S",
                     initial_cash=100_000.0, cost_model=StandardCost())
    assert a.equity_curve == b.equity_curve
    assert a.fills == b.fills


def test_g4_cross_process_determinism():
    code = f"""
import hashlib, math, sys
sys.path.insert(0, r'{ROOT / "src"}')
from framebt import run_backtest
from framebt.strategies import DualMAStrategy
bars = [{{"timestamp": i, "symbol": "S",
          "open": 10 + 2 * math.sin(i / 5) + i * 0.01,
          "high": (10 + 2 * math.sin(i / 5) + i * 0.01) * 1.01,
          "low": (10 + 2 * math.sin(i / 5) + i * 0.01) * 0.99,
          "close": 10 + 2 * math.sin(i / 5) + i * 0.01,
          "volume": 1000000.0}} for i in range(200)]
res = run_backtest(bars, DualMAStrategy("S", 5, 20), symbol="S", initial_cash=100000.0)
payload = repr((res.equity_curve, [(f.frame_id, f.side.value, f.size, f.price, f.cost)
                                   for f in res.fills]))
print("HASH=" + hashlib.sha256(payload.encode()).hexdigest())
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("HASH=")][-1]
    remote = line.split("=", 1)[1]

    rows = [{"timestamp": i, "symbol": "S",
             "open": 10 + 2 * math.sin(i / 5) + i * 0.01,
             "high": (10 + 2 * math.sin(i / 5) + i * 0.01) * 1.01,
             "low": (10 + 2 * math.sin(i / 5) + i * 0.01) * 0.99,
             "close": 10 + 2 * math.sin(i / 5) + i * 0.01,
             "volume": 1_000_000.0} for i in range(200)]
    res = run_backtest(rows, DualMAStrategy("S", 5, 20), symbol="S",
                       initial_cash=100_000.0)
    payload = repr((res.equity_curve,
                    [(f.frame_id, f.side.value, f.size, f.price, f.cost)
                     for f in res.fills]))
    assert hashlib.sha256(payload.encode()).hexdigest() == remote


# ===========================================================================
# G5 撮合 / 账户不变量
# ===========================================================================

@pytest.mark.parametrize("delay", [1, 2, 3])
def test_g5_i2_fill_frame_is_exactly_placement_plus_delay(delay):
    res = run_backtest(_flat(8, 10.0), _BuyOnce(size=100.0), symbol="S",
                       initial_cash=100_000.0, delay_frames=delay,
                       cost_model=NoOpCost())
    assert [f.frame_id for f in res.fills] == [delay]


def test_g5_limit_up_buy_rejected_and_limit_down_sell_rejected():
    up = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 11.1, 11.1, 11.1, 11.1),
          _mk(2, 11.1, 11.1, 11.1, 11.1)]
    strat = _BuyOnce(size=100.0)
    res = run_backtest(up, strat, symbol="S", initial_cash=100_000.0,
                       rules=StandardAshareRules(limit=0.10))
    assert res.fills == []
    assert strat.order.state is OrderState.REJECTED

    down = [_mk(0, 10, 10.5, 9.5, 10), _mk(1, 10, 10.5, 9.5, 10),
            _mk(2, 8.9, 8.9, 8.9, 8.9), _mk(3, 8.9, 8.9, 8.9, 8.9)]
    strat2 = _BuyThenSell(size=100.0)
    res2 = run_backtest(down, strat2, symbol="S", initial_cash=100_000.0,
                        rules=StandardAshareRules(limit=0.10))
    assert [f.side for f in res2.fills] == [OrderSide.BUY]
    assert strat2.sell_order is not None
    assert strat2.sell_order.state is OrderState.REJECTED


def test_g5_t1_same_frame_sell_rejected_next_frame_allowed():
    rows = _flat(5, 10.0)
    same = run_backtest(rows, _BuySellSameFrame(size=100.0), symbol="S",
                        initial_cash=100_000.0, rules=StandardAshareRules())
    assert [(f.frame_id, f.side) for f in same.fills] == [(1, OrderSide.BUY)]

    nxt = _BuyThenSell(size=100.0)
    res = run_backtest(rows, nxt, symbol="S", initial_cash=100_000.0,
                       rules=StandardAshareRules())
    assert [(f.frame_id, f.side) for f in res.fills] == [
        (1, OrderSide.BUY), (2, OrderSide.SELL)
    ]


def test_g5_ledger_settlement_cycle_is_really_t_plus_1():
    """直接钉住账本的可卖日：买入当日不可卖、次帧可卖（T+0 配置则当日可卖）。"""
    led = Ledger(cash=100_000.0, settlement_cycle=1)
    led.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                             size=100, price=10.0, frame_id=5, timestamp=5))
    assert led.position("S").available_to_sell(5) == 0
    assert led.position("S").available_to_sell(6) == 100

    led0 = Ledger(cash=100_000.0, settlement_cycle=0)
    led0.apply_fill(FillEvent(order_id=1, side=OrderSide.BUY, symbol="S",
                              size=100, price=10.0, frame_id=5, timestamp=5))
    assert led0.position("S").available_to_sell(5) == 100


def test_g5_partial_fill_fifo_and_volume_allocation():
    rows = [_mk(i, 10, 10.5, 9.0, 10, 250.0) for i in range(3)]
    res = run_backtest(rows, _ThreeLimits(price=10.0, size=100.0, n=3),
                       symbol="S", initial_cash=100_000.0,
                       rules=NoOpRules(), cost_model=NoOpCost())
    assert [(f.frame_id, f.size) for f in res.fills] == [
        (1, 100.0), (1, 100.0), (1, 50.0), (2, 50.0)
    ]
    assert sum(f.size for f in res.fills if f.frame_id == 1) == 250.0
    assert sum(f.size for f in res.fills) == 300.0


def test_g5_standard_risk_keeps_cash_non_negative():
    rows = _flat(5, 10.0)
    res = run_backtest(rows, _BuyOnce(size=9_000.0), symbol="S",
                       initial_cash=100_000.0, risk=StandardRisk(),
                       cost_model=StandardCost(), rules=StandardAshareRules())
    assert len(res.fills) == 1
    assert res.fills[0].cost > 0
    # 现金在成交后仍非负（90% 仓位 + 费用）
    assert 100_000.0 - 9_000.0 * 10.0 - res.fills[0].cost > 0


def test_g5_missing_bar_marks_last_close_and_does_not_crash():
    rows = [_mk(0, 10, 10.5, 9.5, 10, s="A"), _mk(1, 10, 10.5, 9.5, 10, s="A"),
            _mk(2, 11, 11.5, 10.5, 12, s="A")]
    b = [_mk(i, 50, 50, 50, 50, s="B") for i in range(4)]
    res = run_backtest(rows + b, _BuyOnce(size=100.0, symbol="A"), symbol="A",
                       initial_cash=100_000.0, cost_model=NoOpCost(),
                       rules=NoOpRules())
    # A 在第 3 帧无 Bar：按最后收盘 12 计值
    assert res.equity_curve[-1][1] == pytest.approx(99_000.0 + 100 * 12.0, abs=EPS)


def test_g5_cancelled_order_never_fills():
    class CancelIt(FrameStrategy):
        def __init__(self):
            self.done = False

        def on_frame(self, frame, token, bus, host):
            if self.done:
                return []
            self.done = True
            o = Order(side=OrderSide.BUY, symbol="S", size=100.0,
                      order_type=OrderType.LIMIT, price=5.0)
            host.matcher.submit(o, frame.frame_id)
            host.matcher.cancel(o.id, frame.frame_id)
            return []

    res = run_backtest(_flat(5, 10.0), CancelIt(), symbol="S",
                       initial_cash=100_000.0)
    assert res.fills == []


# ===========================================================================
# G6 工程可用性
# ===========================================================================

def test_g6_public_api_exports_resolve():
    import framebt

    optional_report = {"generate_report", "save_report", "render_equity_chart",
                       "render_trade_chart", "ReportConfig"}
    for name in framebt.__all__:
        try:
            obj = getattr(framebt, name)
        except ImportError:
            # 报告 API 依赖可选的 matplotlib；未装 report extra 时不算核心 API 缺失。
            if name in optional_report:
                continue
            raise
        assert obj is not None, f"公开 API 缺失：{name}"


def test_g6_invalid_configurations_raise_clearly():
    with pytest.raises(ValueError):
        Engine(_feed(_flat(3)), _BuyOnce(size=1.0), delay_frames=0)
    with pytest.raises(ValueError):
        Engine(_feed(_flat(3)), _BuyOnce(size=1.0), delta_t=0)
    with pytest.raises(ValueError):
        Engine(_feed(_flat(3)), _BuyOnce(size=1.0), sub_steps=0)
    with pytest.raises(ValueError):
        MatchingEngine(delay_frames=0)
    with pytest.raises(ValueError):
        Order(side=OrderSide.BUY, symbol="S", size=100.0,
              order_type=OrderType.LIMIT)  # 缺 price
    with pytest.raises(ValueError):
        Order(side=OrderSide.SELL, symbol="S", size=100.0,
              order_type=OrderType.TRAILING_STOP)  # 缺 trail_pct


def test_g6_empty_and_single_bar_feeds_do_not_crash():
    empty = run_backtest([], _BuyOnce(size=1.0), symbol="S", initial_cash=1_000.0)
    assert empty.frames == 0 and empty.equity_curve == []

    single = run_backtest(_flat(1, 10.0), _BuyOnce(size=1.0), symbol="S",
                          initial_cash=1_000.0)
    assert single.frames >= 1
    assert all(math.isfinite(e) for _t, e in single.equity_curve)


def test_g6_clean_stdlib_import_and_mini_backtest():
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
print("OK", framebt.__version__, len(res.fills), res.final_equity())
"""
    out = subprocess.run([sys.executable, "-S", "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("OK "), out.stdout


def test_g6_throughput_budget():
    rows = _trend_rows(3_000, seed=11)
    t0 = time.perf_counter()
    res = run_backtest(rows, DualMAStrategy("S", 5, 20), symbol="S",
                       initial_cash=100_000.0, cost_model=StandardCost())
    elapsed = time.perf_counter() - t0
    assert res.frames >= 3_000
    # 纯 Python 事件循环；3k 帧应远低于 10s（审计实测约 8k bars/s）
    assert elapsed < 10.0, f"吞吐过低：{elapsed:.2f}s / 3000 bars"
    print(f"\n[perf] 3000 bars in {elapsed:.3f}s = {3000 / elapsed:,.0f} bars/s")


# ===========================================================================
# G7 已登记边界（strict xfail：修好后会 XPASS 并提醒更新预登记）
# ===========================================================================

@pytest.mark.xfail(strict=True, reason="KNOWN GAP B1：volume=0 的 Bar 上市价单不受成交量约束")
def test_g7_b1_zero_volume_market_order_cannot_fill():
    rows = [_mk(0, 10, 11, 9, 10, 1_000.0), _mk(1, 10, 11, 9, 10, 0.0),
            _mk(2, 10, 11, 9, 10, 1_000.0)]
    res = run_backtest(rows, _BuyOnce(size=100_000.0), symbol="S",
                       initial_cash=1_000_000.0)
    assert res.fills == []


@pytest.mark.xfail(strict=True, reason="KNOWN GAP B2：NaN 行情静默污染权益曲线")
def test_g7_b2_nan_bar_does_not_silently_poison_equity():
    rows = [_mk(0, 10, 11, 9, 10), _mk(1, float("nan"), 11, 9, 10),
            _mk(2, 10, 11, 9, 10)]
    res = run_backtest(rows, _BuyOnce(size=100.0), symbol="S",
                       initial_cash=100_000.0)
    assert all(math.isfinite(e) for _t, e in res.equity_curve)


@pytest.mark.xfail(strict=True, reason="KNOWN GAP B2：NaN 数量/价格未被 Order 校验拒绝")
def test_g7_b2_nan_order_is_rejected():
    with pytest.raises(ValueError):
        Order(side=OrderSide.BUY, symbol="S", size=float("nan"))


@pytest.mark.xfail(strict=True, reason="KNOWN GAP B3：同帧多笔市价单不共享 Bar 成交量")
def test_g7_b3_market_orders_share_single_bar_volume():
    rows = [_mk(i, 10, 11, 9, 10, 1_000.0) for i in range(4)]
    res = run_backtest(rows, _NOrders([
        Order(side=OrderSide.BUY, symbol="S", size=1_000.0),
        Order(side=OrderSide.BUY, symbol="S", size=1_000.0),
    ]), symbol="S", initial_cash=1_000_000.0)
    assert sum(f.size for f in res.fills) <= 1_000.0


@pytest.mark.xfail(strict=True, reason="KNOWN GAP B4：STOP 单不受成交量约束")
def test_g7_b4_stop_order_respects_volume():
    rows = [_mk(0, 10, 11, 9, 10, 1_000_000.0),
            _mk(1, 10, 11, 9, 10, 1_000_000.0),
            _mk(2, 9.6, 9.7, 9.4, 9.5, 100.0),
            _mk(3, 9.5, 9.6, 9.4, 9.5, 100.0)]
    res = run_backtest(rows, _BuyThenStop(trigger=9.5, size=1_000.0), symbol="S",
                       initial_cash=100_000.0, rules=StandardAshareRules(),
                       risk=StandardRisk())
    sells = [f for f in res.fills if f.side is OrderSide.SELL]
    assert len(sells) == 1 and sells[0].size <= 100.0


@pytest.mark.xfail(strict=True, reason="KNOWN GAP B5：StandardRisk 不预留费用，预算单现金可小幅为负")
def test_g7_b5_budget_order_with_risk_never_negative_cash():
    eng = Engine(_feed(_flat(5, 10.0)), _BuyOnce(budget=100_000.0),
                 initial_cash=100_000.0, risk=StandardRisk(),
                 rules=StandardAshareRules(), cost_model=StandardCost())
    eng.run()
    assert eng.ledger.cash >= -EPS


@pytest.mark.xfail(strict=True, reason="KNOWN GAP B6：同帧两笔大额预算单互相预留，两笔都被拒")
def test_g7_b6_two_large_budget_orders_at_least_one_fills():
    res = run_backtest(_flat(5, 10.0), _NOrders([
        Order(side=OrderSide.BUY, symbol="S", budget=60_000.0),
        Order(side=OrderSide.BUY, symbol="S", budget=60_000.0),
    ]), symbol="S", initial_cash=100_000.0, risk=StandardRisk(),
        rules=StandardAshareRules())
    assert len(res.fills) >= 1
