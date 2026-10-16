"""示例策略：横截面多因子选股（§4.5 / Phase 1 验收）。

在「多标的组合」上，每帧用 ``bus.cross_section`` 计算动量因子横截面排名，
买入排名最高的 K 只、卖出不在持仓 TopK 中的标的。

因果性要点：
  - 横截面因子用 ``lag=1`` 排除当前帧（§4.3），避免同帧泄漏。
  - 调仓订单经延迟管线于下一撮合（I2）。
  - 用 budget 均分权益到每只持仓，而非全仓单只。
"""

from __future__ import annotations

from ..core import FrameStrategy, Order, OrderSide, OrderType


class MomentumCrossSection(FrameStrategy):
    """动量横截面选股：买入动量排名 TopK，剔除出榜标的。

    参数：
      universe：标的列表（需数据源提供）。
      lookback_short / lookback_long：动量窗口（短/长）。
      top_k：持仓数量。
    """

    def __init__(
        self,
        universe: list[str],
        lookback_short: int = 20,
        lookback_long: int = 60,
        top_k: int = 3,
    ):
        if top_k <= 0:
            raise ValueError("top_k 必须为正")
        self.universe = list(universe)
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.top_k = top_k
        self._factor_registered = False

    def _ensure_factor(self, bus) -> None:
        """注册动量因子：close[-short] / close[-long] - 1（幂等）。"""
        if self._factor_registered or "momentum" in bus._factors:
            return
        sb, lb = self.lookback_short, self.lookback_long

        def momentum(seq, token):
            closes = [v for _ts, v in seq]
            if len(closes) < lb:
                return float("nan")
            return closes[-1] / closes[-lb] - 1.0

        bus.register_factor("momentum", momentum)
        self._factor_registered = True

    def on_frame(self, frame, token, bus, host) -> list:
        self._ensure_factor(bus)
        # 动量因子（§4.5 横截面），lag=1 排除当前帧，避免同帧泄漏（§4.3）。
        rows = bus.cross_section(
            factor="momentum",
            universe=self.universe,
            token=token,
            lag=1,
        )
        ranked = [s for s, r in rows.items() if r.is_valid]
        ranked.sort(key=lambda s: rows[s].value, reverse=True)
        target = set(ranked[: self.top_k])

        if not target:
            return []

        pos = host.portfolio
        orders: list = []

        # 以**账本实际敞口**为准（而非「意图持仓」），避免订单被拒后意图与持仓漂移。
        actual = {s: n for s, n in pos.exposure().items() if n > 0}

        # 卖出：实际持有但已跌出目标
        for sym, size in actual.items():
            if sym not in target:
                orders.append(
                    Order(side=OrderSide.SELL, symbol=sym, size=size,
                          order_type=OrderType.MARKET)
                )

        # 买入：目标中尚未持有的，用**可用现金**在新增标的间均分
        to_buy = [s for s in ranked[: self.top_k] if actual.get(s, 0) <= 0]
        if to_buy:
            per = pos.cash / len(to_buy)
            # 现金已全部占用（满仓）时 per<=0：不下单，等卖出腿释放资金（T+1/延迟）
            if per > 0:
                for sym in to_buy:
                    orders.append(
                        Order(side=OrderSide.BUY, symbol=sym,
                              order_type=OrderType.MARKET, budget=per)
                    )
        return orders
