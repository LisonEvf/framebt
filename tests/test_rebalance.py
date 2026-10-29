"""Target-weight 调仓原语（``framebt.rebalance``）的回归测试。"""

from __future__ import annotations

import pytest

from framebt import (
    BarFeed,
    Engine,
    FrameStrategy,
    Ledger,
    NoOpCost,
    Order,
    OrderSide,
    OrderType,
    RebalanceConfig,
    rebalance_orders,
    target_shares,
)
from framebt.core.events import FillEvent
from framebt.core.market_frame import MarketFrame


def _bars(n: int = 6, price_a: float = 10.0, price_b: float = 20.0):
    rows = []
    for i in range(n):
        rows.append({"timestamp": i, "symbol": "A", "open": price_a, "high": price_a,
                     "low": price_a, "close": price_a, "volume": 1e6})
        rows.append({"timestamp": i, "symbol": "B", "open": price_b, "high": price_b,
                     "low": price_b, "close": price_b, "volume": 1e6})
    return rows


class Plan(FrameStrategy):
    """在指定帧执行一次调仓，并记录返回的订单。"""

    def __init__(self, at_frame: int, targets, config=None):
        self.at_frame = at_frame
        self.targets = targets
        self.config = config
        self.orders: list[Order] = []

    def on_frame(self, frame, token, bus, host):
        if frame.frame_id != self.at_frame:
            return []
        self.orders = host.rebalance_to(self.targets, frame, config=self.config)
        return self.orders


class BuyThenPlan(FrameStrategy):
    """第 0 帧买入 A（默认全仓）；第 2 帧调仓到目标。"""

    def __init__(self, targets, config=None, size: float = 10000.0):
        self.targets = targets
        self.config = config
        self.size = size
        self.rebalance_orders: list[Order] = []

    def on_frame(self, frame, token, bus, host):
        if frame.frame_id == 0:
            return [Order(side=OrderSide.BUY, symbol="A",
                          order_type=OrderType.MARKET, size=self.size)]
        if frame.frame_id == 2:
            self.rebalance_orders = host.rebalance_to(
                self.targets, frame, config=self.config)
            return self.rebalance_orders
        return []


def _run(rows, strat, cash=100_000.0):
    engine = Engine(BarFeed.from_dicts(rows), strat, initial_cash=cash,
                    cost_model=NoOpCost())
    return engine, engine.run()


# ---------------------------------------------------------------------------
# 纯函数
# ---------------------------------------------------------------------------

def test_target_shares_lot_and_missing_price():
    s = target_shares({"A": 0.4, "B": 0.5, "C": 0.1}, 100_000.0,
                      {"A": 10.0, "B": 30.0})
    assert s == {"A": 4000, "B": 1600}  # 1666→1600（整手）；C 无价被跳过


def test_weight_validation():
    with pytest.raises(ValueError):
        target_shares({"A": 1.2}, 100_000.0, {"A": 10.0})
    with pytest.raises(ValueError):
        target_shares({"A": 0.6, "B": 0.6}, 100_000.0, {"A": 10.0, "B": 10.0})
    with pytest.raises(ValueError):
        target_shares({"A": -0.1}, 100_000.0, {"A": 10.0})
    with pytest.raises(ValueError):
        target_shares({"A": float("nan")}, 100_000.0, {"A": 10.0})


def test_held_position_without_price_raises():
    led = Ledger(cash=0.0)
    led.apply_fill(FillEvent(1, OrderSide.BUY, "A", 100, 10.0, 0, 0))
    frame = MarketFrame(0, 0, 1)  # 无任何 bar

    class Host:
        portfolio = led

    with pytest.raises(ValueError):
        rebalance_orders({"B": 0.5}, frame, Host(), prices={"B": 10.0})


# ---------------------------------------------------------------------------
# 端到端
# ---------------------------------------------------------------------------

def test_rebalance_hits_target_weights():
    strat = Plan(1, {"A": 0.5, "B": 0.5},
                 RebalanceConfig(cash_buffer=0.0, min_trade_pct=0.0))
    engine, res = _run(_bars(), strat)
    fills = [(f.frame_id, f.symbol, f.side, f.size) for f in res.fills]
    assert fills == [(2, "A", OrderSide.BUY, 5000.0), (2, "B", OrderSide.BUY, 2500.0)]
    assert engine.ledger.cash == pytest.approx(0.0, abs=1e-9)
    assert engine.ledger.exposure() == {"A": 5000.0, "B": 2500.0}


def test_sells_come_before_buys():
    strat = BuyThenPlan({"B": 1.0}, RebalanceConfig(cash_buffer=0.0, min_trade_pct=0.0))
    engine, res = _run(_bars(), strat)
    sides = [o.side for o in strat.rebalance_orders]
    assert sides == [OrderSide.SELL, OrderSide.BUY]
    assert engine.ledger.exposure() == {"B": 5000.0}
    assert engine.ledger.cash == pytest.approx(0.0, abs=1e-9)


def test_min_trade_pct_suppresses_tiny_leg():
    # 已持 A 5000（半仓）；目标 A 0.501 → 只差 10 股（100 元），低于 2% 权益门槛 → 不发 A 单
    strat_buy = BuyThenPlan({"A": 0.501, "B": 0.499},
                            RebalanceConfig(cash_buffer=0.0, min_trade_pct=0.02),
                            size=5000.0)
    engine, res = _run(_bars(), strat_buy)
    syms = {o.symbol for o in strat_buy.rebalance_orders}
    assert "A" not in syms
    assert "B" in syms


def test_cash_buffer_leaves_cash():
    strat = Plan(1, {"A": 1.0}, RebalanceConfig(cash_buffer=0.01, min_trade_pct=0.0))
    engine, res = _run(_bars(), strat)
    # 目标 1.0 但因 1% 缓冲只买 99% 权益；余下 ~1000 元现金
    assert engine.ledger.cash == pytest.approx(1000.0, rel=1e-6)
    assert engine.ledger.exposure() == {"A": 9900.0}


def test_unspecified_holdings_are_liquidated():
    strat = BuyThenPlan({"B": 1.0}, RebalanceConfig(cash_buffer=0.0, min_trade_pct=0.0))
    engine, _ = _run(_bars(), strat)
    assert "A" not in engine.ledger.exposure()
