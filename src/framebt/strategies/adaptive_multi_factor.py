"""持续自适应的多因子策略：在 framebt 引擎内跑通全流水线。

对应设计文档的阶段三～五（在线权重 + 门控 + 健康监控），把「因子 → 权重 → 执行」
三层串在**同一个因果时钟**下：

每帧（因果顺序严格）：
  1. **在线学习**（§3.2）：用「上一帧的因子截面」与「本帧已实现收益」更新权重
     —— 二者在当帧都已知，符合「先预测、后观察标签、再更新」范式；
  2. **健康监控**（§5）：按滚动 IC 评估各因子健康度，黄灯降权 / 红灯移出
     （仅在参数敏感性稳健时才自动执行）；
  3. **门控**（§3.3）：按市场状态调节因子门控系数；
  4. **组合构建**：合成打分 → 选 TopK → 目标权重；
  5. **下单**：卖出腿先撤出，买入腿按**可用现金**分批建仓
     （订单有强制延迟 I2，同帧卖出的资金要到下一帧才可用，故不能一次性换仓）。

因子数据来源
------------
``factor_panels`` 为**预先算好的（已预处理）因子面板**。这在本框架下是可证明因果的：
每个因子的每一行只依赖该行及之前的行情（所有算子在 :mod:`framebt.factors.ops` 中
都是后向窗口），且预处理（去极值/标准化/中性化）都是**逐行截面**操作。
``tools/validate_adaptive.py`` 中的因果审计会抽样验证：
``Panel.slice_upto(t)`` 重算得到的第 t 行与预计算面板的第 t 行逐位相等。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import FrameStrategy, Order, OrderSide, OrderType
from ..factors.lifecycle import FactorRegistry
from ..monitor.health import FactorHealthMonitor
from ..portfolio.combine import select_portfolio, weights_to_shares
from ..portfolio.gating import MarketStateGating
from ..portfolio.weights import OnlineWeightOptimizer

EPS = 1e-12


class AdaptiveMultiFactorStrategy(FrameStrategy):
    """在线自适应的多因子组合策略。

    参数：
      universe       ：标的列表
      factor_panels  ：{因子名: 已预处理面板}（index=时间戳, columns=标的）
      close_panel    ：收盘价面板（用于计算已实现收益与估值）
      top_k          ：持仓只数
      rebalance_every：调仓间隔（帧）
      warmup         ：预热帧数（不足则不交易）
      mode           ：``'long'`` / ``'long_short'``
      gross          ：总仓位（1.0 = 满仓）
      use_gating     ：是否启用市场状态门控
      use_health     ：是否启用健康监控降权
      health_every   ：健康评估间隔（帧）
      lot            ：最小交易单位（A 股 100 股）
    """

    def __init__(
        self,
        universe: list[str],
        factor_panels: dict[str, pd.DataFrame],
        close_panel: pd.DataFrame,
        *,
        turnover_panel: pd.DataFrame | None = None,
        top_k: int = 5,
        rebalance_every: int = 5,
        warmup: int = 260,
        mode: str = "long",
        gross: float = 1.0,
        lr: float = 0.002,
        lambda_turnover: float = 0.1,
        lambda_l1: float = 0.01,
        discount: float = 0.95,
        buffer: int = 20,
        use_gating: bool = True,
        use_health: bool = True,
        health_every: int = 20,
        lot: int = 100,
        buffer_mult: float = 2.0,
        min_order_pct: float = 0.01,
        registry: FactorRegistry | None = None,
        gating_lr: float = 0.02,
        gating_tau: float = 1.0,
        seed: int = 0,
    ):
        self.universe = list(universe)
        self.factors = {k: v.reindex(columns=self.universe)
                        for k, v in factor_panels.items()}
        self.close = close_panel.reindex(columns=self.universe)
        self.turnover = (turnover_panel.reindex(columns=self.universe)
                         if turnover_panel is not None else None)
        self.factor_names = sorted(self.factors)

        if not self.factor_names:
            raise ValueError("至少需要一个因子面板")
        self.top_k = int(top_k)
        self.rebalance_every = max(1, int(rebalance_every))
        self.warmup = int(warmup)
        self.mode = mode
        self.gross = float(gross)
        self.lot = int(lot)
        self.use_gating = use_gating
        self.use_health = use_health
        self.health_every = max(1, int(health_every))
        #: 缓冲带倍数：当前持仓仍在前 top_k×buffer_mult 名内则保留（抑制换手）
        self.buffer_mult = float(buffer_mult)
        #: 单笔最小下单占比（占权益），低于此不下单，避免「滴水」产生大量碎单
        self.min_order_pct = float(min_order_pct)

        self.optimizer = OnlineWeightOptimizer(
            len(self.factor_names), lambda_turnover=lambda_turnover,
            lambda_l1=lambda_l1, lr=lr, discount=discount, buffer=buffer)
        self.gating = (MarketStateGating(len(self.factor_names), n_state=3, hidden=16,
                                         temperature=gating_tau, lr=gating_lr, seed=seed)
                       if use_gating else None)
        self.health = FactorHealthMonitor() if use_health else None
        self.registry = registry

        # 运行时状态
        self._frame_count = 0
        self._prev_ts: int | None = None
        self._prev_X: np.ndarray | None = None
        self._target: dict[str, float] = {}
        self._held_shares: dict[str, int] = {}
        self._ic_history: dict[str, list[float]] = {n: [] for n in self.factor_names}
        self._market_state: pd.DataFrame | None = None
        self._last_ts: int | None = None
        self.events: list[dict] = []

    # ---- 主循环 ----

    def on_frame(self, frame, token, bus, host) -> list:
        bars = frame.bars
        if not bars:
            return []
        ts = next(iter(bars.values())).timestamp
        self._frame_count += 1

        # 只在因子面板有的时间戳上工作
        if ts not in self.factors[self.factor_names[0]].index:
            return []

        # --- 1. 在线学习：用上一帧因子截面 + 本帧已实现收益更新权重 ---
        if self._prev_X is not None and self._prev_ts is not None:
            y = self._realized_return(self._prev_ts, ts)
            if y is not None:
                state = self.optimizer.observe(self._prev_X, y)
                self._record_ic(ts)
                if (self.gating is not None and self._frame_count % self.rebalance_every == 0):
                    recent = np.array([np.nanmean(self._ic_history[n][-60:])
                                       for n in self.factor_names])
                    self.gating.observe(self._state_vector(ts), recent)

        # 记录本帧的因子截面，供下一帧学习
        X = self._cross_section(ts)
        if X is not None:
            self._prev_X = X
            self._prev_ts = ts

        # --- 2/3. 健康监控（定期） ---
        if (self.use_health and self.health is not None
                and self._frame_count % self.health_every == 0
                and self._frame_count > self.warmup):
            self._refresh_health(ts)
            if self.registry is not None:
                self._advance_registry(ts)

        # --- 4/5. 调仓 ---
        if self._frame_count < self.warmup:
            return []
        # 先刷新当前持仓快照（供缓冲带判断与下单使用）
        self._held_shares = {s: int(host.portfolio.position(s).net_size())
                             for s in self.universe}
        rebalance_now = (self._frame_count % self.rebalance_every == 0)
        if rebalance_now:
            self._rebuild_target(ts)
        # 卖出腿只在调仓帧发出（否则目标未变时会每帧重复挂单）；
        # 买入腿每帧「滴水」执行，因为同帧卖出的资金下一帧才可用。
        return self._orders_towards_target(ts, host, allow_sells=rebalance_now)

    # ---- 因子与收益 ----

    def _cross_section(self, ts: int) -> np.ndarray | None:
        cols = []
        for n in self.factor_names:
            f = self.factors[n]
            cols.append(f.loc[ts].to_numpy(dtype=float) if ts in f.index
                        else np.full(len(self.universe), np.nan))
        X = np.column_stack(cols)
        return X if np.isfinite(X).any() else None

    def _realized_return(self, ts_prev: int, ts_now: int) -> np.ndarray | None:
        if ts_prev not in self.close.index or ts_now not in self.close.index:
            return None
        y = (self.close.loc[ts_now] / self.close.loc[ts_prev] - 1.0).to_numpy(dtype=float)
        return y if np.isfinite(y).any() else None

    def _record_ic(self, ts: int) -> None:
        """记录每个因子的**已实现 IC**（用于健康监控）。"""
        if self._prev_ts is None:
            return
        y = self.close.loc[ts] / self.close.loc[self._prev_ts] - 1.0
        y_rank = y.rank()
        for n in self.factor_names:
            f = self.factors[n]
            if self._prev_ts not in f.index:
                continue
            x = f.loc[self._prev_ts]
            common = x.notna() & y_rank.notna()
            if common.sum() < 5:
                continue
            ic = x[common].rank().corr(y_rank[common])
            if np.isfinite(ic):
                self._ic_history[n].append(float(ic))
    # ---- 权重 ----

    def _current_weights(self, ts: int) -> dict[str, float]:
        w = self.optimizer.weights_dict(self.factor_names)
        if self.use_health and self.health is not None and self.health.metrics:
            w = self.health.downgrade_weights(w)
        if self.registry is not None:
            w = self.registry.cap_weights(w)
        if self.gating is not None:
            base = np.array([w.get(n, 0.0) for n in self.factor_names])
            final = self.gating.combine(base, self._state_vector(ts))
            return dict(zip(self.factor_names, final))
        return w

    def _state_vector(self, ts: int) -> np.ndarray:
        if self.gating is None:
            return np.zeros(3)
        if self._market_state is None:
            from ..factors.panel import Panel  # 局部导入避免循环
            self._market_state = MarketStateGating.market_state(
                _MiniPanel(self.close))
        if ts in self._market_state.index:
            return self.gating.state_vector(self._market_state.loc[ts])
        return np.array([0.5, 0.5, 0.0])

    def _refresh_health(self, ts: int, window: int = 250) -> None:
        """用累积的已实现 IC 序列刷新各因子健康度（含 FSC 与拥挤度）。"""
        assert self.health is not None
        from ..monitor.health import crowding_proxy, factor_stability

        for n in self.factor_names:
            ic = pd.Series(self._ic_history[n], dtype=float)
            if len(ic) < 30:
                continue
            f_win = self.factors[n].loc[self.factors[n].index <= ts].tail(window)
            fsc = factor_stability(f_win) if len(f_win) > 20 else float("nan")
            if self.turnover is not None:
                crowding = crowding_proxy(f_win, _MiniPanel(
                    self.close.loc[self.close.index <= ts].tail(window),
                    self.turnover.loc[self.turnover.index <= ts].tail(window)))
            else:
                crowding = float("nan")
            self.health.evaluate(n, ic_series=ic, fsc=fsc, crowding=crowding)

    def _advance_registry(self, ts: int) -> None:
        """按最健康/最差的因子状态推进生命周期（研究模式下的简化推进）。"""
        assert self.registry is not None
        metrics = self.health.metrics if self.health else {}
        if not metrics:
            return
        worst = min(metrics.values(), key=lambda m: (m.decay_prob if np.isfinite(m.decay_prob) else 0.0))
        for n in self.factor_names:
            m = metrics.get(n)
            if m is None:
                continue
            self.registry.step(ts, icir=m.icir, ic_mean=m.ic_mean, fsc=m.fsc,
                               crowding=m.crowding, cusum_alarm=m.cusum_alarm,
                               weight=self.optimizer.weights_dict(self.factor_names).get(n))

    # ---- 组合构建与下单 ----

    def _rebuild_target(self, ts: int) -> None:
        from ..portfolio.combine import composite_score

        w = self._current_weights(ts)
        score_panel = composite_score(self.factors, w)
        if ts not in score_panel.index:
            return
        row = score_panel.loc[ts].replace([np.inf, -np.inf], np.nan)
        current = {s for s, n in self._held_shares.items() if n > 0}
        target = select_portfolio(row, top_k=self.top_k, mode=self.mode,
                                  gross=self.gross, current=current,
                                  buffer_mult=self.buffer_mult)
        self._target = target
        self.events.append({"ts": ts, "kind": "rebalance", "n_target": len(target),
                            "weights": self.optimizer.weights_dict(self.factor_names)})

    def _orders_towards_target(self, ts: int, host, allow_sells: bool = True) -> list:
        """向目标持仓靠拢：先卖超配，再用**当前可用现金**分批买入。

        三个必须处理的约束：
        1. **同帧卖出的资金下一帧才可用**（订单有强制延迟 I2，撮合在 Phase 3、
           入账在撮合之后）。因此不能用「卖出所得」在同一帧一次性建仓，只能逐帧
           「滴水」买入 —— 由于本方法**每帧**都被调用，闲置现金会在下一帧被投出。
        2. **不得重复下单**：在途订单未落地前持仓数字不变；若每帧无条件重发会产生
           重复订单。故用 ``host.pending_orders()`` 跳过已有在途单的标的。
        3. **卖出腿不要每帧重发**：目标未变时卖出腿的重发只是徒增订单（被拒后还会
           下一帧再来）。故仅在调仓帧发卖出腿。
        """
        if not self._target or ts not in self.close.index:
            return []
        prices = self.close.loc[ts].to_dict()
        equity = self._equity(ts, host)
        if equity <= 0:
            return []

        target_shares = weights_to_shares(self._target, equity, prices, lot=self.lot)
        held = dict(self._held_shares)
        pending_list = host.pending_orders()
        pending = {o.symbol for o in pending_list}
        min_notional = equity * self.min_order_pct

        orders: list = []
        # --- 卖出腿：清掉不在目标中的持仓，以及超配部分 ---
        if allow_sells:
            for sym, cur in held.items():
                if cur <= 0 or sym in pending:
                    continue
                want = max(0, target_shares.get(sym, 0))
                if cur > want and (cur - want) * float(prices.get(sym, 0.0)) >= min_notional:
                    orders.append(Order(side=OrderSide.SELL, symbol=sym,
                                        size=float(cur - want), order_type=OrderType.MARKET))

        # --- 买入腿：按当前可用现金，在低配标的上按缺口比例分配 ---
        under = [(sym, want - held.get(sym, 0))
                 for sym, want in target_shares.items()
                 if want > held.get(sym, 0) and sym not in pending]
        if not under:
            return orders

        # 可用现金必须扣除**前几帧仍在途的买单**（它们尚未入账，但资金已被占用）。
        # 否则策略会按未扣减的现金下单，与风控的预留口径不一致 → 大量「资金不足」拒绝。
        reserved = sum(o.size * float(prices.get(o.symbol, 0.0))
                       for o in pending_list
                       if o.side is OrderSide.BUY and np.isfinite(o.size))
        cash = max(0.0, host.portfolio.cash - reserved) * 0.995   # 预留费用缓冲
        need_notional = {s: n * float(prices.get(s, 0.0)) for s, n in under}
        total_need = sum(need_notional.values())
        if total_need <= 0 or cash <= 0:
            return orders

        for sym, need_sh in under:
            px = float(prices.get(sym, 0.0))
            if px <= 0 or not np.isfinite(px):
                continue
            alloc = min(need_notional[sym], cash * need_notional[sym] / total_need)
            shares = int(alloc / px // self.lot * self.lot)
            # 单笔过小则跳过：避免「滴水」式执行产生海量碎单（真实交易中也不经济）
            if shares > 0 and shares * px >= min_notional:
                orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                    size=float(shares), order_type=OrderType.MARKET))
        return orders

    def _equity(self, ts: int, host) -> float:
        """按当前帧价格估值权益（现金 + 持仓市值）。"""
        cash = host.portfolio.cash
        val = 0.0
        for sym, size in host.portfolio.exposure().items():
            px = self.close.loc[ts].get(sym, np.nan) if ts in self.close.index else np.nan
            if np.isfinite(px):
                val += size * float(px)
        return float(cash + val)

    # ---- 报告 ----

    def weights_report(self) -> dict:
        return {
            "ic": self.optimizer.ic,
            "icir": self.optimizer.icir,
            "weights": self.optimizer.weights_dict(self.factor_names),
            "top": self.optimizer.top_factors(self.factor_names, 10),
            "steps": self.optimizer.step_count,
        }

    def health_dashboard(self) -> pd.DataFrame:
        return self.health.dashboard() if self.health else pd.DataFrame()


class _MiniPanel:
    """仅含 close（+ 可选 turnover）的轻量面板，供市场状态/拥挤度复用同一份逻辑。"""

    def __init__(self, close: pd.DataFrame, turnover: pd.DataFrame | None = None):
        self.close = close
        self.turnover = (turnover if turnover is not None
                         else pd.DataFrame(np.nan, index=close.index, columns=close.columns))
