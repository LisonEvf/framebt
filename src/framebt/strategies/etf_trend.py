"""ETF 多资产趋势跟踪策略（预登记设计见 docs/ETF_TREND.md）。

针对本项目的实测教训逐条设计：
  1. **不做横截面选股** —— 用时间序列（绝对）动量决定"是否持有"，
     而非"持哪只"。横截面选股在最流动池子里已被证伪（见 PATH_200K.md §8）。
  2. **广度来自时间与资产类别** —— 10 个标的 × 4~5 个低相关资产类别，
     绕开「20 万 → 只能持 3~6 只」的广度锁死。
  3. **低换手** —— 月末调仓（rebalance_every≈21）；非调仓帧只做向目标靠拢。
  4. **ETF 免印花税、免过户费** —— 单边摩擦从股票策略的 0.0838% 降到约 0.03%。
  5. **波动率目标** —— Moreira & Muir (2017) 波动率管理组合；无杠杆，只降不升。
  6. **现金腿 = 货币 ETF** —— 有票息、T+0、无印花税，比空仓现金强。

因果纪律（I1/I2）：所有历史经 ``bus.get(token=...)`` 取，订单当帧生成、下一帧开盘成交。
"""

from __future__ import annotations

import math

import numpy as np

from ..core import FrameStrategy, Order, OrderSide, OrderType


class EtfTrendStrategy(FrameStrategy):
    def __init__(
        self,
        universe: list[str],
        *,
        cash_symbol: str,
        lookback: int = 252,
        ma_window: int = 200,
        vol_window: int = 60,
        target_vol: float = 0.12,
        rebalance_every: int = 21,
        use_trend: bool = True,
        use_vol_target: bool = True,
        weighting: str = "equal",
        top_k: int | None = None,
        min_amount: float = 2e7,
        amount_window: int = 20,
        max_weight: float = 1.0,
        lot: int = 100,
        min_order_pct: float = 0.004,
    ):
        if weighting not in ("equal", "inv_vol"):
            raise ValueError("weighting 应为 'equal' 或 'inv_vol'")
        self.universe = list(universe)
        self.cash_symbol = cash_symbol
        self.lookback = int(lookback)
        self.ma_window = int(ma_window)
        self.vol_window = int(vol_window)
        self.target_vol = float(target_vol)
        self.rebalance_every = int(rebalance_every)
        self.use_trend = bool(use_trend)
        self.use_vol_target = bool(use_vol_target)
        self.weighting = weighting
        self.top_k = top_k
        self.min_amount = float(min_amount)
        self.amount_window = int(amount_window)
        self.max_weight = float(max_weight)
        self.lot = int(lot)
        self.min_order_pct = float(min_order_pct)

        self._frame = 0
        self._target: dict[str, float] = {}
        self._last_target_frame = -10**9
        self.diag: list[dict] = []      # 每次调仓的诊断快照

    # ---------------- 主循环 ----------------

    def on_frame(self, frame, token, bus, host) -> list:
        if not frame.bars:
            return []
        self._frame += 1

        need = max(self.lookback, self.ma_window, self.vol_window) + 1
        if self._frame < need:
            return []

        rebalance_now = (self._frame % self.rebalance_every == 0)
        if rebalance_now:
            self._target = self._build_target(frame, bus, token, host)
            self._last_target_frame = self._frame
        if not self._target:
            return []
        return self._orders_towards_target(frame, host, allow_sells=rebalance_now)

    # ---------------- 目标构建 ----------------

    def _prices(self, frame) -> dict[str, float]:
        out = {}
        for sym in list(self.universe) + [self.cash_symbol]:
            if frame.has_bar(sym):
                px = float(frame.close(sym))
                if px > 0 and math.isfinite(px):
                    out[sym] = px
        return out

    def _build_target(self, frame, bus, token, host) -> dict[str, float]:
        cash_mom = self._momentum(bus, token, self.cash_symbol)
        cands: list[tuple[str, float, np.ndarray]] = []

        for sym in self.universe:
            if sym == self.cash_symbol:
                continue
            hist = np.asarray(
                bus.get("close", sym, window=self.lookback + 1, token=token),
                dtype=float)
            if hist.size < self.lookback + 1 or not np.isfinite(hist).all():
                continue                      # 历史不足（PIT：新上市即被排除）
            px = float(hist[-1])
            if px <= 0:
                continue
            # PIT 流动性：trailing 日均成交额
            amt = np.asarray(
                bus.get("amount", sym, window=self.amount_window, token=token),
                dtype=float)
            if amt.size < self.amount_window or not np.isfinite(amt).all():
                continue
            if float(amt.mean()) < self.min_amount:
                continue
            mom = px / float(hist[0]) - 1.0
            if self.use_trend:
                ma = float(hist[-self.ma_window:].mean())
                if not (px > ma):
                    continue
                if cash_mom is not None and mom <= cash_mom:
                    continue
            cands.append((sym, mom, hist))

        if not cands:
            return {self.cash_symbol: 1.0}

        if self.top_k is not None and len(cands) > self.top_k:
            cands.sort(key=lambda t: t[1], reverse=True)
            cands = cands[: self.top_k]

        syms = [c[0] for c in cands]

        # --- 基础权重 ---
        if self.weighting == "equal":
            base = np.full(len(syms), 1.0 / len(syms))
        else:
            vols = np.array([self._vol(bus, token, s) for s in syms], dtype=float)
            vols = np.where(np.isfinite(vols) & (vols > 1e-6), vols, np.nan)
            if not np.isfinite(vols).any():
                base = np.full(len(syms), 1.0 / len(syms))
            else:
                inv = 1.0 / np.nan_to_num(vols, nan=np.nanmax(vols))
                base = inv / inv.sum()

        base = np.minimum(base, self.max_weight)
        base = base / base.sum()

        # --- 波动率目标 ---
        scale = 1.0
        port_vol = float("nan")
        if self.use_vol_target and len(syms) >= 1:
            R = self._ret_matrix(bus, token, syms)
            if R is not None:
                port = R @ base
                port_vol = float(np.std(port, ddof=1)) * math.sqrt(252.0)
                if port_vol > 1e-9:
                    scale = min(1.0, self.target_vol / port_vol)
        scale = max(0.0, min(1.0, scale))

        tgt: dict[str, float] = {}
        for s, w in zip(syms, base):
            wv = float(w) * scale
            if wv > 1e-6:
                tgt[s] = wv
        cash_w = 1.0 - sum(tgt.values())
        if cash_w > 1e-6:
            tgt[self.cash_symbol] = cash_w

        self.diag.append({
            "frame": self._frame,
            "n_risky": len(syms),
            "selected": list(syms),
            "scale": scale,
            "port_vol": port_vol,
            "cash_w": cash_w,
            "weights": dict(tgt),
        })
        return tgt

    def _momentum(self, bus, token, sym) -> float | None:
        h = np.asarray(bus.get("close", sym, window=self.lookback + 1, token=token),
                       dtype=float)
        if h.size < self.lookback + 1 or not np.isfinite(h).all() or h[0] <= 0:
            return None
        return float(h[-1] / h[0] - 1.0)

    def _vol(self, bus, token, sym) -> float:
        h = np.asarray(bus.get("close", sym, window=self.vol_window + 1, token=token),
                       dtype=float)
        if h.size < self.vol_window + 1 or not np.isfinite(h).all() or (h <= 0).any():
            return float("nan")
        r = h[1:] / h[:-1] - 1.0
        return float(np.std(r, ddof=1) * math.sqrt(252.0))

    def _ret_matrix(self, bus, token, syms) -> np.ndarray | None:
        cols = []
        for s in syms:
            h = np.asarray(
                bus.get("close", s, window=self.vol_window + 1, token=token),
                dtype=float)
            if h.size < self.vol_window + 1 or not np.isfinite(h).all() or (h <= 0).any():
                return None
            cols.append(h[1:] / h[:-1] - 1.0)
        return np.column_stack(cols)

    # ---------------- 下单 ----------------

    def _orders_towards_target(self, frame, host, allow_sells: bool) -> list:
        prices = self._prices(frame)
        if not prices:
            return []
        equity = self._equity(frame, host)
        if equity <= 0:
            return []

        want_sh: dict[str, int] = {}
        for sym, w in self._target.items():
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue
            sh = int(abs(w) * equity / px // self.lot * self.lot)
            if sh > 0:
                want_sh[sym] = sh

        held = {s: int(host.portfolio.position(s).net_size())
                for s in set(list(want_sh) + list(self._target)
                             + [b.symbol for b in frame.bars.values()])}
        pending_list = host.pending_orders()
        pending = {o.symbol for o in pending_list}
        min_notional = equity * self.min_order_pct
        orders: list = []

        if allow_sells:
            for sym, cur in held.items():
                if cur <= 0 or sym in pending:
                    continue
                want = want_sh.get(sym, 0)
                if cur > want and (cur - want) * prices.get(sym, 0.0) >= min_notional:
                    orders.append(Order(side=OrderSide.SELL, symbol=sym,
                                        size=float(cur - want),
                                        order_type=OrderType.MARKET))

        under = [(s, w - held.get(s, 0)) for s, w in want_sh.items()
                 if w > held.get(s, 0) and s not in pending]
        if not under:
            return orders

        reserved = sum(o.size * prices.get(o.symbol, 0.0) for o in pending_list
                       if o.side is OrderSide.BUY and math.isfinite(o.size))
        cash = max(0.0, host.portfolio.cash - reserved) * 0.995
        need = {s: n * prices.get(s, 0.0) for s, n in under}
        total = sum(need.values())
        if total <= 0 or cash <= 0:
            return orders

        for sym, _n in under:
            px = prices.get(sym, 0.0)
            if px <= 0:
                continue
            alloc = min(need[sym], cash * need[sym] / total)
            sh = int(alloc / px // self.lot * self.lot)
            if sh > 0 and sh * px >= min_notional:
                orders.append(Order(side=OrderSide.BUY, symbol=sym,
                                    size=float(sh), order_type=OrderType.MARKET))
        return orders

    def _equity(self, frame, host) -> float:
        val = 0.0
        for sym, size in host.portfolio.exposure().items():
            if frame.has_bar(sym):
                px = float(frame.close(sym))
                if math.isfinite(px):
                    val += size * px
        return float(host.portfolio.cash + val)
