"""Engine Core：时间模型 + 帧生命周期编排（§2 / §3）。

核心骨架：**固定 Δt 主动推进虚拟时间**。每帧四阶段（§2.2）：
   Phase 1 市场数据注入：把落在 (frame_start, frame_end] 的 Bar 写入 Bus
   Phase 2 唤醒分发：host.dispatch_frame → 策略 on_frame → 返回订单
   Phase 3 订单处理与撮合：matcher.process（延迟管线 + 价格-时间优先）
   Phase 4 状态快照与绩效记录：ledger.mark_to_market → 权益曲线

时间推进（累加器模式，§3.1）：
   当事件时间对齐 Δt 网格（日线最常见）时，每帧恰一根 Bar，累加器恒为 0，
   本循环即 ``while accumulator >= delta_t: step()`` 的特例。
   空闲交易日区间（周末/节假日）折叠跳过——不生成静态帧，避免无意义迭代；
   仅当「数据耗尽但仍有在途订单」时才补静态帧，让其以最后收盘价撮合。

不变式（§2.3）：
   I1  任何数据请求返回的记录 timestamp <= current_virtual_time
   I2  帧 t 生成的订单，最早可成交时间为帧 t+delay（延迟管线）
   I3  相同配置逐位可复现
   I4  帧内多 Bar 按 (timestamp, sequence_id) 严格排序
   I5  策略对已过去帧的状态修改请求被拒绝
"""

from __future__ import annotations

from typing import Optional, Sequence

from .core import Bar, FillEvent, Order
from .feed import BarFeed
from .ledger import Ledger
from .core import MarketFrame, StaticFrameBuilder
from .core import BacktestResult
from .rules import MarketRules, NoOpRules
from .core import FrameStrategy
from .cost import CostModel, NoOpCost
from .guard import RiskMiddleware, NoOpRisk
from .core import StrategyHost
from .data_bus import CausalToken, DataBus
from .execution import MatchingEngine


class Engine:
    """帧刷新回测引擎（单时钟域）。

    参数：
      feed：原始 K 线数据源（BarFeed，已按 (timestamp, sequence_id) 排序）。
      strategy：策略实例。
      delta_t：固定逻辑步长（默认 1）。日线数据 Δt=1 时每帧一根 Bar。
      delay_frames：订单从生成到可撮合的强制延迟帧数（默认 1，满足 I2）。
      initial_cash：初始现金。
      seed：确定性 RNG 种子。
      settlement_cycle：结算周期（T+1 → 1；T+0 → 0）。
      cost_model / risk / rules：可插拔的成本/风控/规则中间件（默认空实现）。
    """

    def __init__(
        self,
        feed: BarFeed,
        strategy: FrameStrategy,
        *,
        start_ns: int = 0,
        delta_t: int = 1,
        seed: int = 0,
        delay_frames: int = 1,
        initial_cash: float = 0.0,
        settlement_cycle: int = 1,
        cost_model: Optional[CostModel] = None,
        risk: Optional[RiskMiddleware] = None,
        rules: Optional[MarketRules] = None,
    ):
        if delta_t <= 0:
            raise ValueError("delta_t 必须为正")
        if delay_frames < 1:
            raise ValueError("delay_frames 必须 >= 1（I2：帧 t 订单最早帧 t+1 成交）")

        self.feed = feed
        self.start_ns = int(start_ns)
        self.delta_t = int(delta_t)

        self.bus = DataBus()
        # settlement_cycle 必须同时传给 Ledger：持仓腿的「可卖日」由它决定，
        # 否则 T+0 配置下账本仍按 T+1 冻结，规则与账本口径不一致。
        self.ledger = Ledger(cash=initial_cash, settlement_cycle=settlement_cycle)
        self.matcher = MatchingEngine(delay_frames=delay_frames)
        self.host = StrategyHost(seed=seed)
        self.host.register(strategy)
        self.host.portfolio = self.ledger
        self.host.matcher = self.matcher   # 供策略查询在途订单，避免重复下单

        self.cost_model = cost_model or NoOpCost()
        self.risk = risk or NoOpRisk()
        self.rules = rules or NoOpRules(settlement_cycle=settlement_cycle)
        # 让风控能看到在途订单（否则多笔在途买单各自「资金充足」却合计透支）
        self.risk.matcher = self.matcher

        self._static_builder = StaticFrameBuilder()
        self._last_close: dict[str, float] = {}
        self._bars = feed.bars
        self._idx = 0
        self._n = len(self._bars)

        self.result = BacktestResult(start_ns=start_ns)

    # ---- Phase 1：数据注入 ----

    def _inject(self, bar: Bar) -> None:
        self.bus.record("open", bar.symbol, bar.timestamp, bar.open)
        self.bus.record("high", bar.symbol, bar.timestamp, bar.high)
        self.bus.record("low", bar.symbol, bar.timestamp, bar.low)
        self.bus.record("close", bar.symbol, bar.timestamp, bar.close)
        self.bus.record("volume", bar.symbol, bar.timestamp, bar.volume)
        self.bus.record("amount", bar.symbol, bar.timestamp, bar.amount)
        self.bus.record("float_shares", bar.symbol, bar.timestamp, bar.float_shares)
        self.bus.record("turnover", bar.symbol, bar.timestamp, bar.turnover)
        # vwap / 市值：优先用数据源已对齐复权口径的值；否则退回近似
        vwap = bar.vwap if bar.vwap > 0 else (
            (bar.amount / bar.volume) if bar.volume > 0 else bar.close)
        cap = bar.market_cap if bar.market_cap > 0 else bar.close * bar.float_shares
        self.bus.record("vwap", bar.symbol, bar.timestamp, vwap)
        self.bus.record("market_cap", bar.symbol, bar.timestamp, cap)
        self._last_close[bar.symbol] = bar.close

    # ---- 盘前检查（风控 + 市场规则组合）----

    def _pre_trade(self, order: Order, frame: MarketFrame) -> Optional[str]:
        """组合风控与规则：返回拒绝原因字符串，或 None 表示放行。"""
        r = self.risk.pre_trade_check(order, self.ledger, frame)
        if r is not None:
            return r
        return self.rules.check(order, self.ledger, frame)

    def _register_items(self, items, frame_id: int) -> None:
        """登记策略返回的订单 / 订单组（组支持 all_or_none 原子性与 OCO 联动）。"""
        from .execution import OrderGroup

        for item in items or []:
            if item is None:
                continue
            if isinstance(item, OrderGroup):
                self.matcher.register_group(item, frame_id)
            else:
                self.matcher.register(item, frame_id)

    # ---- 帧编排 ----

    def run(self, max_frames: int = 10_000_000) -> BacktestResult:
        bars = self._bars
        n = self._n
        idx = 0
        k = 0
        logical_time = self.start_ns

        while True:
            frame_start = logical_time
            frame_end = logical_time + self.delta_t
            frame = MarketFrame(k, frame_start, frame_end)

            # ---- Phase 1：注入本帧 Bar（半开区间 [frame_start, frame_end)）----
            injected = 0
            while idx < n and bars[idx].timestamp < frame_end:
                bar = bars[idx]
                frame.add_bar(bar)
                self._inject(bar)
                injected += 1
                idx += 1

            if injected == 0:
                if not self.matcher.has_in_flight():
                    # 空闲区间折叠：无数据且无在途订单 → 跳到下一根 bar，不生成静态帧
                    if idx >= n:
                        break  # 数据耗尽
                    logical_time = bars[idx].timestamp
                    continue
                # 数据耗尽但仍有在途订单：以前帧 BBO 生成静态帧，让其撮合（§2.2 Phase 1）
                frame = self._static_builder.rebuild_static(k, frame_start, frame_end)

            # ---- Phase 2：唤醒分发 ----
            # 先登记权威虚拟时间：即使策略伪造 issued_at 更大的 token，
            # 总线也会钳制到此刻，拿不到未来数据（§4.1 / I1）。
            self.bus.set_now(frame_end)
            if injected > 0:
                token = CausalToken(frame_end, k)
                orders = self.host.dispatch_frame(frame, token, self.bus)
                self._register_items(orders, k)

            # ---- Phase 3：订单处理与撮合 ----
            raw_fills = self.matcher.process(frame, k, pre_trade_check=self._pre_trade)

            # 成本模型：在成交时计入费用
            fills: list[FillEvent] = []
            for fill in raw_fills:
                fill.cost = self.cost_model.cost_for(fill, frame)
                self.ledger.apply_fill(fill)
                # on_fill 回调可产生新订单/订单组（如止损条件单、OCO 止盈止损），登记入在途队列（I2）
                fill_orders = self.host.dispatch_fill(fill)
                self._register_items(fill_orders, k)
                fills.append(fill)

            # ---- Phase 4：状态快照与绩效记录 ----
            marks = self._marks(frame)
            equity = self.ledger.mark_to_market(marks)
            self.result.equity_curve.append((frame_start, equity))
            self.result.fills.extend(fills)
            self.result.frames = k + 1
            self.result.end_ns = frame_end

            # 规则逐帧钩子：在**本帧撮合之后**更新昨收，供下一帧计算涨跌停带。
            # （若在撮合前调用，涨跌停会以当帧收盘为基准，口径错误。）
            self.rules.observe(frame)

            if injected > 0:
                self._static_builder.refresh(frame)

            # ---- 推进虚拟时间 ----
            if injected == 0:
                # 静态帧（仅在途订单）：按 Δt 推进
                logical_time = frame_end
            elif idx < n and bars[idx].timestamp > frame_end:
                # 下一个 bar 不在本帧内：空闲区间折叠，不生成空帧
                logical_time = bars[idx].timestamp
            else:
                logical_time = frame_end
            k += 1

            if k > max_frames:
                raise RuntimeError(
                    f"引擎超过 max_frames={max_frames} 仍未终止"
                    "（可能存在未成交在途订单或逻辑死循环）"
                )

        return self.result

    def _marks(self, frame: MarketFrame) -> dict[str, float]:
        marks: dict[str, float] = {}
        for sym in self.ledger.exposure():
            if frame.has_bar(sym):
                marks[sym] = frame.close(sym)
            else:
                marks[sym] = self._last_close.get(sym, 0.0)
        return marks

    # ---- 便捷访问（供分析/测试）----

    def sma(self, values: Sequence[float], period: int) -> float:
        if period <= 0:
            raise ValueError("period 必须为正")
        if len(values) < period:
            raise ValueError("数据不足以计算 SMA")
        return sum(values[-period:]) / period


def run_backtest(
    bars: Sequence[dict],
    strategy: FrameStrategy,
    *,
    symbol: str = "S",
    start_ns: int = 0,
    delta_t: int = 1,
    seed: int = 0,
    delay_frames: int = 1,
    initial_cash: float = 0.0,
    settlement_cycle: int = 1,
    cost_model: Optional[CostModel] = None,
    risk: Optional[RiskMiddleware] = None,
    rules: Optional[MarketRules] = None,
) -> BacktestResult:
    """从「字典列表」快速启动一次回测（演示/测试友好）。"""
    feed = BarFeed.from_dicts(bars, symbol=symbol)
    engine = Engine(
        feed,
        strategy,
        start_ns=start_ns,
        delta_t=delta_t,
        seed=seed,
        delay_frames=delay_frames,
        initial_cash=initial_cash,
        settlement_cycle=settlement_cycle,
        cost_model=cost_model,
        risk=risk,
        rules=rules,
    )
    return engine.run()
