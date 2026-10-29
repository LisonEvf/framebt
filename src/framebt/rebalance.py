"""Target-weight 调仓原语：把「目标权重」翻译成一组成交订单。

为什么需要它
------------
按目标权重调仓是资产配置/轮动策略的日常动作，但在此之前每个策略都要手写：

1. 用当前收盘价算出权益；
2. 目标权重 → 目标股数（A 股 100 股整手）；
3. 先卖后买（卖出所得同日可用），并按可用现金约束买入；
4. 过滤掉「为了 0.1% 权重差而交易」的碎单；
5. 留出现金缓冲，避免下一帧开盘跳空/费用把现金打成负数。

这五步在多个策略与复刻脚本里各写了一遍，且很容易漏掉「先卖后买」或「现金约束」。
本模块把它收成一个纯函数 + 宿主便捷方法。

因果性
------
- 决策价：``frame`` 的收盘价（或调用方显式传入的 ``prices``）；
- 订单：帧 t 生成、最早帧 t+1 开盘成交（I2），因此「收盘定价 → 次日开盘成交」；
- 不读取任何未来数据；持仓/现金来自 ``host.portfolio``（已成交部分）。

用法
----
    def on_frame(self, frame, token, bus, host):
        if not self.is_rebalance_day(frame):
            return []
        targets = {"SH.510300": 0.6, "SH.511010": 0.4}   # 权重和 ≤ 1
        return host.rebalance_to(targets, frame)

    # 或直接用函数
    from framebt import rebalance_orders
    orders = rebalance_orders(targets, frame, host, prices={"SH.510300": 4.2})
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .core.events import Order, OrderSide, OrderType
from .core.market_frame import MarketFrame

__all__ = ["RebalanceConfig", "target_shares", "rebalance_orders", "weights_from_positions"]


@dataclass(frozen=True)
class RebalanceConfig:
    """调仓参数。

    lot            ：最小交易单位（A 股 100 股）。
    min_trade_pct  ：单腿最小交易金额占权益比例，低于此不交易（抑制碎单/换手）。
    cash_buffer    ：买入前预留的现金占权益比例（覆盖费用与次日开盘跳空）。
    enforce_cash   ：是否按「现金 + 卖出所得 − 缓冲」缩放买入腿。关闭则完全按目标下单
                     （可能透支，只建议在复刻历史口径时使用）。
    order_type     ：订单类型，默认市价（次日开盘成交）。
    """

    lot: int = 100
    min_trade_pct: float = 0.005
    cash_buffer: float = 0.002
    enforce_cash: bool = True
    order_type: OrderType = OrderType.MARKET

    def __post_init__(self) -> None:
        if self.lot <= 0:
            raise ValueError("lot 必须为正")
        if not 0.0 <= self.min_trade_pct < 1.0:
            raise ValueError("min_trade_pct 必须在 [0, 1)")
        if not 0.0 <= self.cash_buffer < 1.0:
            raise ValueError("cash_buffer 必须在 [0, 1)")


def _validate_weights(targets: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for sym, w in targets.items():
        try:
            x = float(w)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"目标权重 {sym!r} 不是数值：{w!r}") from exc
        if not math.isfinite(x) or x < 0:
            raise ValueError(f"目标权重 {sym!r} 必须是非负有限数：{w!r}")
        if x > 1.0 + 1e-9:
            raise ValueError(f"目标权重 {sym!r}={x} > 1（本原语只支持多头、单腿权重 ≤ 1）")
        if x > 0:
            out[sym] = x
    total = sum(out.values())
    if total > 1.0 + 1e-9:
        raise ValueError(f"目标权重之和 {total:.6f} > 1（现金腿需显式留出）")
    return out


def target_shares(
    targets: Mapping[str, float],
    equity: float,
    prices: Mapping[str, float],
    *,
    lot: int = 100,
) -> dict[str, int]:
    """目标权重 → 目标股数（向下取整到 ``lot``）。无有效价格的标的不出现在结果中。"""
    weights = _validate_weights(targets)
    if equity <= 0:
        return {}
    out: dict[str, int] = {}
    for sym, w in weights.items():
        px = float(prices.get(sym, 0.0))
        if px <= 0 or not math.isfinite(px):
            continue
        shares = int((w * equity) / px // lot) * lot
        if shares > 0:
            out[sym] = shares
    return out


def rebalance_orders(
    targets: Mapping[str, float],
    frame: MarketFrame,
    host,
    *,
    prices: Mapping[str, float] | None = None,
    config: RebalanceConfig | None = None,
) -> list[Order]:
    """把目标权重翻译成「先卖后买」的订单列表。

    参数：
      targets ：``{symbol: weight}``，权重和 ≤ 1；未列出的持仓会被清掉（目标 0）。
      frame   ：当前帧（提供收盘价与 ``has_bar``）。
      host    ：``StrategyHost``（提供 ``portfolio``：现金/持仓）。
      prices  ：可选价格覆盖（如停牌日用手工价）；缺省用 ``frame.close``。
      config  ：:class:`RebalanceConfig`。

    返回：``list[Order]``（先 SELL 后 BUY）。无可行交易时返回 ``[]``。
    """
    cfg = config or RebalanceConfig()
    weights = _validate_weights(targets)
    portfolio = host.portfolio

    # ---- 价格表：显式 prices 优先，其次本帧收盘；持仓无法定价则报错 ----
    px: dict[str, float] = {}
    if prices:
        for sym, p in prices.items():
            p = float(p)
            if p > 0 and math.isfinite(p):
                px[sym] = p
    held = {s: float(n) for s, n in portfolio.exposure().items() if n != 0}
    for sym in set(weights) | set(held):
        if sym not in px:
            if frame.has_bar(sym):
                px[sym] = float(frame.close(sym))
            elif sym in held:
                raise ValueError(
                    f"{sym} 有持仓但本帧无行情，无法估值；请通过 prices= 传入最近收盘价"
                )
    equity = float(portfolio.cash) + sum(n * px[s] for s, n in held.items() if s in px)
    if equity <= 0:
        return []

    want = target_shares(weights, equity, px, lot=cfg.lot)

    # ---- 逐腿差额（未列出的持仓目标为 0）----
    symbols = list(dict.fromkeys(list(weights) + list(held)))
    min_notional = equity * cfg.min_trade_pct
    sells: list[tuple[str, int]] = []
    buys: list[tuple[str, int]] = []
    for sym in symbols:
        p = px.get(sym, 0.0)
        if p <= 0:
            continue
        cur = int(held.get(sym, 0.0))
        tgt = int(want.get(sym, 0))
        delta = tgt - cur
        if delta == 0 or abs(delta) * p < min_notional:
            continue
        if delta < 0:
            sells.append((sym, -delta))
        else:
            buys.append((sym, delta))

    # ---- 现金约束：卖出所得同日可用，留 cash_buffer ----
    sell_proceeds = sum(n * px[s] for s, n in sells)
    available = float(portfolio.cash) + sell_proceeds - cfg.cash_buffer * equity
    if cfg.enforce_cash and buys:
        need = sum(n * px[s] for s, n in buys)
        if need > available:
            if available <= 0:
                buys = []
            else:
                ratio = available / need
                scaled: list[tuple[str, int]] = []
                for sym, n in buys:
                    k = int(n * ratio // cfg.lot) * cfg.lot
                    if k > 0 and k * px[sym] >= min_notional:
                        scaled.append((sym, k))
                buys = scaled
                # 缩放后再校验一次（lot 取整可能仍略超）
                while buys and sum(n * px[s] for s, n in buys) > available:
                    sym, n = buys[-1]
                    n -= cfg.lot
                    if n <= 0:
                        buys.pop()
                    else:
                        buys[-1] = (sym, n)

    orders: list[Order] = []
    for sym, n in sells:
        orders.append(Order(side=OrderSide.SELL, symbol=sym, size=float(n),
                            order_type=cfg.order_type))
    for sym, n in buys:
        orders.append(Order(side=OrderSide.BUY, symbol=sym, size=float(n),
                            order_type=cfg.order_type))
    return orders


def weights_from_positions(host, frame: MarketFrame) -> dict[str, float]:
    """当前持仓权重（按本帧收盘价估值；用于监控/日志，不产生订单）。"""
    held = {s: float(n) for s, n in host.portfolio.exposure().items() if n != 0}
    prices = {s: float(frame.close(s)) for s in held if frame.has_bar(s)}
    equity = float(host.portfolio.cash) + sum(n * prices.get(s, 0.0) for s, n in held.items())
    if equity <= 0:
        return {}
    return {s: n * prices[s] / equity for s, n in held.items() if s in prices}
