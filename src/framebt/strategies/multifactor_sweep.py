"""主动交易策略：多因子基础 + Sweep 信号做 T + 主动选股。

对应需求：

  1. **多因子策略作为基础**（core）
     用预计算好的、已预处理的因子面板合成截面打分，按打分选 TopK 建核心组合，
     等权持仓、定期调仓。核心组合只占权益的一部分（``base_gross``，默认 70%），
     留出容量给下面的 sweep 战术仓。

  2. **叠加 sweep 信号做 T**（satellite / T+0 滚动）
     ``sweep`` 是一个短周期（默认 ≤5 交易日）的突破/放量信号：收盘价突破近 N 日
     高点且放量、当日为阳线时触发（“扫掉”阻力位的激进突破）。对触发的标的做一个
     独立的短回合交易（买入 → 持有 ≤K 日 / 跌破触发位卖出），反复滚动即为“做 T”。

  3. **主动选股，用新命中标有的填补空仓期**
     sweep 仓只在**核心组合未满仓时的闲置容量**里、且**市场处于风险偏好（regime_on）
     时**开仓，标的从“当日满足流动性、突破信号成立、且不在核心池”里主动挑选——这些就是
     “新命中标的”。若核心筛选与 sweep 信号都无合格标的，策略**接受空仓**（``regime_off_gross``）。

因果性（I1/I2）
----------------
- 所有信号（composite / sweep_signal / sweep_exit / eligibility / regime）都在
  **数据准备阶段**预先算好（纯后向算子），策略每天只读取“当天已发生”的值。
- 订单为市价单，引擎在 **次日开盘价** 撮合（``delay_frames=1``、``fill_price_mode='open'``），
  即“用当天收盘信息决策、次日开盘执行”，无前视偏差。
- A 股 T+1：sweep 仓当日买入的份额次日才可卖，本策略的 T 回合持有期 ≥1 日，自然满足。

使用：

    from framebt import Engine, StandardCost, StandardAshareRules
    from framebt.strategies import MultiFactorSweepStrategy

    strat = MultiFactorSweepStrategy(universe, factor_panels, close_panel,
                                     factor_weights=W, top_k=20)
    res = Engine(feed, strat, initial_cash=1e6, settlement_cycle=1,
                 cost_model=StandardCost(...), rules=StandardAshareRules()).run()
"""

from __future__ import annotations

import numpy as np

from ..core import FrameStrategy, Order, OrderSide, OrderType


class MultiFactorSweepStrategy(FrameStrategy):
    """多因子基础 + sweep 做 T + 主动选股（详见模块文档）。

    参数
    ----
    universe              ：标的池（列对齐）
    factor_panels         ：{因子名: 已预处理面板}（index=交易日 ordinal, columns=标的）
    close_panel           ：收盘价面板（用于权益估值）
    factor_weights        ：{因子名: 权重}，合成截面打分 ``score = Σ wᵢ·fᵢ``
    top_k                 ：核心组合持仓只数
    rebalance_every       ：核心组合调仓间隔（交易日）
    warmup                ：预热帧数（不足则不交易）
    score_floor           ：核心标的最低composite打分（截面分位，默认 0 = 中位以上）
    min_hold              ：核心池至少多少只合格标的，否则视为“无标的”→空仓
    base_gross            ：核心组合目标仓位（风险偏好时）
    regime_off_gross      ：风险规避时总仓位（默认 0 → 完全空仓）
    sweep_budget_pct      ：sweep 战术仓最大权益占比
    sweep_lookback        ：sweep 突破回溯窗口（交易日）
    sweep_vol_mult        ：放量阈值 = 该倍数 × 均量
    sweep_hold_max        ：sweep 仓最长持有（交易日）
    max_sweep_positions   ：同时存在的 sweep 仓最大数量
    sweep_min_notional    ：单笔 sweep 最小名义金额
    lot                   ：最小交易单位（A 股 100 股）
    buffer_mult           ：核心组合缓冲带倍数（抑制换手）
    """

    def __init__(
        self,
        universe,
        factor_panels,
        close_panel,
        *,
        factor_weights,
        top_k=20,
        rebalance_every=5,
        warmup=60,
        score_floor=0.0,
        min_hold=5,
        base_gross=0.70,
        regime_off_gross=0.0,
        sweep_budget_pct=0.30,
        sweep_lookback=15,
        sweep_vol_mult=1.5,
        sweep_hold_max=5,
        max_sweep_positions=12,
        sweep_min_notional=30000.0,
        lot=100,
        buffer_mult=2.0,
        min_order_pct=0.01,
    ):
        self.universe = list(universe)
        self.factors = {k: v.reindex(columns=self.universe)
                        for k, v in factor_panels.items()}
        self.close = close_panel.reindex(columns=self.universe)
        self.weights = {k: float(w) for k, w in factor_weights.items()
                        if k in self.factors}

        # 数据准备阶段预计算的辅助面板（index=ordinal, columns=标的），由 run 前注入
        self.eligibility = None       # bool：当日是否流动性合格
        self.sweep_signal = None      # bool：当日是否触发突破放量信号
        self.sweep_exit = None        # bool：当日是否跌破触发位（退出信号）
        self.regime = None            # bool：市场是否风险偏好（>N 日均线）
        self._ts_list = None
        self._ts_index = {}

        self.top_k = int(top_k)
        self.rebalance_every = max(1, int(rebalance_every))
        self.warmup = int(warmup)
        self.score_floor = float(score_floor)
        self.min_hold = int(min_hold)
        self.base_gross = float(base_gross)
        self.regime_off_gross = float(regime_off_gross)
        self.sweep_budget_pct = float(sweep_budget_pct)
        self.sweep_lookback = int(sweep_lookback)
        self.sweep_vol_mult = float(sweep_vol_mult)
        self.sweep_hold_max = int(sweep_hold_max)
        self.max_sweep_positions = int(max_sweep_positions)
        self.sweep_min_notional = float(sweep_min_notional)
        self.lot = int(lot)
        self.buffer_mult = float(buffer_mult)
        self.min_order_pct = float(min_order_pct)

        # 运行时状态
        self._frame_count = 0
        self._base_target: dict[str, float] = {}
        self._base_held: set[str] = set()
        self._sweep_shares: dict[str, int] = {}
        self._sweep_enter_ts: dict[str, int] = {}
        self._sweep_used_notional = 0.0
        self._since_rebalance = 0
        self._events: list[dict] = []

    # ------------------------------------------------------------------ 注入
    def inject_aux(self, *, eligibility=None, sweep_signal=None,
                   sweep_exit=None, regime=None):
        """注入数据准备阶段算好的辅助面板（由运行器调用）。"""
        self.eligibility = eligibility
        self.sweep_signal = sweep_signal
        self.sweep_exit = sweep_exit
        self.regime = regime
        src = regime if regime is not None else self.close
        if src is not None and len(src.index):
            self._ts_list = list(src.index)
            self._ts_index = {int(t): i for i, t in enumerate(self._ts_list)}
        return self

    # ------------------------------------------------------------- 主循环
    def on_frame(self, frame, token, bus, host) -> list:
        bars = frame.bars
        if not bars:
            return []
        ts = int(next(iter(bars.values())).timestamp)
        self._frame_count += 1

        if self._ts_list is None or ts not in self._ts_index:
            return []

        # 预热：数据不足时不交易
        if self._frame_count <= self.warmup:
            self._since_rebalance += 1
            return []

        equity = self._equity(ts, host)
        if equity <= 0:
            return []

        # ---- 1. 核心组合调仓（定期）----
        self._since_rebalance += 1
        if self._since_rebalance >= self.rebalance_every:
            orders = self._base_rebalance(ts, host, equity)
        else:
            orders = []

        # ---- 2. sweep 做 T（每日，填充闲置容量）----
        orders += self._sweep_trades(ts, host, equity)
        return orders

    # ============================================================ 核心组合
    def _base_rebalance(self, ts, host, equity) -> list:
        """按 composite 打分选 TopK，等权，缓冲带抑制换手；无合格标的则空仓。"""
        score_row = self._composite(ts)
        self._since_rebalance = 0
        if score_row is None or len(score_row) == 0:
            self._base_target = {}
            self._base_held = set()
            return []

        elig = self._eligible_set(ts)
        scored = []
        for s, v in score_row.items():
            if s not in elig:
                continue
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            if v >= self.score_floor:
                scored.append((s, float(v)))
        scored.sort(key=lambda t: t[1], reverse=True)

        if len(scored) < self.min_hold:
            # 合格太少 → 视为“无法命中”，核心组合空仓
            self._base_target = {}
            self._base_held = set()
            self._events.append({"ts": ts, "kind": "base_empty",
                                 "n_scored": len(scored)})
            return self._base_orders(ts, host, equity, {})

        gross = self.base_gross if self._regime_on(ts) else self.regime_off_gross
        picks = [s for s, _ in scored[: self.top_k]]
        target = {s: gross / len(picks) for s in picks}
        self._base_target = target
        self._base_held = set(picks)
        self._events.append({"ts": ts, "kind": "base_rebalance",
                             "picks": picks, "gross": gross})
        return self._base_orders(ts, host, equity, target)

    def _base_orders(self, ts, host, equity, target) -> list:
        """向核心目标持仓靠拢：卖出跌出池者，买入按可用现金分批。"""
        prices = self.close.loc[ts].to_dict()
        target_shares = {}
        for sym, w in target.items():
            px = prices.get(sym, 0.0)
            if px <= 0 or not np.isfinite(px):
                continue
            notional = w * equity
            shares = int(abs(notional) / px // self.lot * self.lot)
            if shares > 0:
                target_shares[sym] = shares

        held = {s: int(host.portfolio.position(s).net_size())
                for s in self.universe}
        pending_list = host.pending_orders()
        pending = {o.symbol for o in pending_list}
        min_notional = equity * self.min_order_pct

        orders: list[Order] = []
        # 卖出：清掉跌出核心池的持仓
        for sym, cur in held.items():
            if cur <= 0 or sym in pending or sym in self._sweep_shares:
                continue
            want = target_shares.get(sym, 0)
            if cur > want and (cur - want) * float(prices.get(sym, 0.0)) >= min_notional:
                orders.append(Order(side=OrderSide.SELL, symbol=sym,
                                    size=float(cur - want), order_type=OrderType.MARKET))

        # 买入：按当前可用现金，在低配核心标的上按比例分配
        under = [(s, want - held.get(s, 0)) for s, want in target_shares.items()
                 if want > held.get(s, 0) and s not in pending]
        if not under:
            return orders
        reserved = sum(o.size * float(prices.get(o.symbol, 0.0))
                       for o in pending_list
                       if o.side is OrderSide.BUY and np.isfinite(o.size))
        cash = max(0.0, host.portfolio.cash - reserved) * 0.995
        need = {s: n * float(prices.get(s, 0.0)) for s, n in under}
        total_need = sum(need.values())
        if total_need <= 0 or cash <= 0:
            return orders
        for sym, need_sh in under:
            px = float(prices.get(sym, 0.0))
            if px <= 0 or not np.isfinite(px):
                continue
            alloc = min(need[sym], cash * need[sym] / total_need)
            shares = int(alloc / px // self.lot * self.lot)
            if shares > 0 and shares * px >= min_notional:
                orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                    size=float(shares), order_type=OrderType.MARKET))
        return orders

    # =============================================================== sweep
    def _sweep_trades(self, ts, host, equity) -> list:
        """每日处理 sweep 仓：先处理到期/止损退出，再用闲置容量开新仓。"""
        if self.sweep_signal is None or self.eligibility is None:
            return []
        prices = self.close.loc[ts].to_dict()
        orders: list[Order] = []

        # ---- 退出：持有满 K 日 或 跌破触发位 ----
        exited = []
        for sym, enter_ts in list(self._sweep_enter_ts.items()):
            exit_now = (ts - enter_ts) >= self.sweep_hold_max
            if (self.sweep_exit is not None and ts in self.sweep_exit.index
                    and sym in self.sweep_exit.columns
                    and bool(self.sweep_exit.loc[ts, sym])):
                exit_now = True
            if exit_now and self._sweep_shares.get(sym, 0) > 0:
                orders.append(Order(side=OrderSide.SELL, symbol=sym,
                                    size=float(self._sweep_shares[sym]),
                                    order_type=OrderType.MARKET))
                exited.append(sym)
        for sym in exited:
            self._sweep_shares.pop(sym, None)
            self._sweep_enter_ts.pop(sym, None)

        # ---- 开仓：用核心组合未占用的闲置容量 ----
        gross_used = self._gross_used(ts, host)
        spare = max(0.0, 1.0 - gross_used)
        budget = max(0.0, equity * self.sweep_budget_pct - self._sweep_used_notional)
        budget = min(budget, equity * spare)  # 不超过闲置容量
        if budget < self.sweep_min_notional or not self._regime_on(ts):
            return orders
        if len(self._sweep_shares) >= self.max_sweep_positions:
            return orders

        candidates = self._sweep_candidates(ts)
        if not candidates:
            return orders
        # 等权分配预算
        per = budget / len(candidates)
        for sym in candidates:
            px = prices.get(sym, 0.0)
            if px <= 0 or not np.isfinite(px):
                continue
            alloc = min(per, budget)
            shares = int(alloc / px // self.lot * self.lot)
            if shares > 0 and shares * px >= self.sweep_min_notional:
                orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                    size=float(shares), order_type=OrderType.MARKET))
                self._sweep_shares[sym] = shares
                self._sweep_enter_ts[sym] = ts
                self._sweep_used_notional += shares * px
                self._events.append({"ts": ts, "kind": "sweep_entry",
                                     "symbol": sym, "shares": shares, "alloc": alloc})
        return orders

    def _sweep_candidates(self, ts) -> list:
        """满足：流动性合格 + 突破信号成立 + 不在核心池 + 无在途 sweep 仓。"""
        elig = self._eligible_set(ts)
        signal = self.sweep_signal
        cand = []
        for sym in self.universe:
            if sym in self._base_held or sym in self._sweep_shares:
                continue
            if sym not in elig:
                continue
            if sym not in signal.columns or ts not in signal.index:
                continue
            if not bool(signal.loc[ts, sym]):
                continue
            cand.append(sym)
        return cand

    # ============================================================== 辅助
    def _composite(self, ts):
        """合成截面打分 ``Σ wᵢ·fᵢ``（因子面板已预处理）。"""
        if not self.weights:
            return None
        total = None
        wsum = None
        for name, w in self.weights.items():
            f = self.factors[name]
            if ts not in f.index:
                continue
            row = f.loc[ts]
            contrib = row * w
            total = contrib if total is None else total.add(contrib, fill_value=0.0)
            aw = row.notna().astype(float) * abs(w)
            wsum = aw if wsum is None else wsum.add(aw, fill_value=0.0)
        if total is None:
            return None
        total = total.div(wsum.where(wsum > 1e-12))
        return total.to_dict()

    def _eligible_set(self, ts):
        if self.eligibility is None or ts not in self.eligibility.index:
            return set(self.universe)
        row = self.eligibility.loc[ts]
        return {s for s, v in row.items() if bool(v)}

    def _regime_on(self, ts) -> bool:
        """风险偏好判定：市场处于风险偏好（数据准备阶段已算好 bool 面板）。"""
        if self.regime is None or ts not in self.regime.index:
            return True
        return bool(self.regime.loc[ts])

    def _gross_used(self, ts, host) -> float:
        """当前权益中已被占用的比例（核心 + sweep）。"""
        prices = self.close.loc[ts].to_dict()
        held = host.portfolio.exposure()
        used = 0.0
        for sym, n in held.items():
            px = prices.get(sym, 0.0)
            if np.isfinite(px) and n > 0:
                used += n * float(px)
        equity = self._equity(ts, host)
        return (used / equity) if equity > 0 else 0.0

    def _equity(self, ts, host) -> float:
        cash = host.portfolio.cash
        prices = self.close.loc[ts].to_dict()
        held = host.portfolio.exposure()
        val = 0.0
        for sym, n in held.items():
            px = prices.get(sym, 0.0)
            if np.isfinite(px):
                val += n * float(px)
        return float(cash + val)
