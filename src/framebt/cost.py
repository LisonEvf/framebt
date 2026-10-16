"""Cost：成本模型（§8）。

成本在撮合成交时扣除，计入 FillEvent.cost，由 Ledger 在入账时从现金中减去。

组件：
  - commission：佣金率 + 最低佣金
  - stamp_tax：印花税（仅卖出，A 股 0.1%）
  - transfer_fee：过户费
  - slippage：滑点模型（fixed / proportional / volume_based / sqrt）

可插拔：实现 ``CostModel.cost_for(fill, frame) -> float`` 即可。
"""

from __future__ import annotations

from .core.events import FillEvent, OrderSide
from .core.market_frame import MarketFrame


class CostModel:
    """成本模型接口。"""

    def cost_for(self, fill: FillEvent, frame: MarketFrame) -> float:
        raise NotImplementedError


class NoOpCost(CostModel):
    """零成本（基准/Phase 0）。"""

    def cost_for(self, fill: FillEvent, frame: MarketFrame) -> float:
        return 0.0


class StandardCost(CostModel):
    """A 股标准成本模型。

    参数：
      commission：佣金率（双边）。
      min_commission：最低佣金。
      stamp_tax：印花税（仅卖出）。
      transfer_fee：过户费。
      slippage：滑点模型，'fixed'/'proportional'/'volume_based'/'sqrt'。
      k：滑点系数。
      spread：fixed 模型用双边价差。
      impact：平方根冲击系数。
      adv：日均成交量（sqrt/volume 模型用；缺省用当帧成交量）。
    """

    def __init__(
        self,
        commission: float = 0.0003,
        min_commission: float = 5.0,
        stamp_tax: float = 0.001,
        transfer_fee: float = 0.00002,
        slippage: str = "volume_based",
        k: float = 0.001,
        spread: float = 0.0,
        impact: float = 0.1,
        adv: float | None = None,
    ):
        self.commission = commission
        self.min_commission = min_commission
        self.stamp_tax = stamp_tax
        self.transfer_fee = transfer_fee
        self.slippage = slippage
        self.k = k
        self.spread = spread
        self.impact = impact
        self.adv = adv

    def cost_for(self, fill: FillEvent, frame: MarketFrame) -> float:
        notional = fill.size * fill.price
        fee = 0.0

        # 佣金（取 佣金率 与 最低佣金 的较大者）
        commission = max(notional * self.commission, self.min_commission)
        fee += commission

        # 印花税（仅卖出）
        if fill.side is OrderSide.SELL and self.stamp_tax > 0:
            fee += notional * self.stamp_tax

        # 过户费
        if self.transfer_fee > 0:
            fee += notional * self.transfer_fee

        # 滑点
        fee += self._slippage_cost(fill, frame)
        return fee

    def _slippage_cost(self, fill: FillEvent, frame: MarketFrame) -> float:
        notional = fill.size * fill.price
        vol = self._volume(frame, fill.symbol)

        if self.slippage == "fixed":
            return notional * (self.spread / 2.0) if self.spread else 0.0
        if self.slippage == "proportional":
            return notional * self.k
        if self.slippage == "volume_based":
            if not vol or vol <= 0:
                return 0.0
            return notional * self.k * (fill.size / vol)
        if self.slippage == "sqrt":
            adv = self.adv if self.adv and self.adv > 0 else (vol or 0.0)
            if not adv or adv <= 0:
                return 0.0
            return notional * self.impact * (fill.size / adv) ** 0.5
        raise ValueError(f"未知滑点模型 {self.slippage}")

    def _volume(self, frame: MarketFrame, symbol: str) -> float:
        if frame.has_bar(symbol):
            return frame.volume(symbol)
        return 0.0
