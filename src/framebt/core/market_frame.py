"""MarketFrame：引擎以固定 Δt 推进时聚合出来的「一帧市场数据」。

一帧包含落在区间 ``(frame_start, frame_end]`` 内所有标的的 Bar。
区间内无数据时以前帧 BBO 填充并标记 ``is_static=True``（见 §2.1 Phase 1）。

对日线级 Bar：
  - open/high/low/close 直接来自 Bar
  - last  = close
  - bid   = low, ask = high   （一帧内的极差作为 BBO 近似）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .events import Bar


@dataclass
class Quote:
    """单标的 BBO / 报价快照。"""

    bid: float
    ask: float
    last: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0


@dataclass
class FormingBar:
    """多时间框架对齐（§4.4）：当前正在形成的更高周期 bar。

    例如在小时帧请求日线数据时，``closed`` 是最近一根已收盘日线，
    ``forming`` 是当前正在形成、high/low 随帧推进更新的日线。
    """

    closed: Optional[Bar]
    forming: Optional[Bar]

    def __post_init__(self) -> None:
        # 默认禁止在 forming bar 上触发信号（§4.4），由策略显式选择。
        pass


class MarketFrame:
    """单帧市场快照。

    策略通过 ``bars`` 拿到本帧注入的 Bar；通过 open/high/low/close/volume/bbo
    便捷访问器按标的取数。所有访问都建立在「本帧已注入数据」之上，天然满足 I1。
    """

    __slots__ = (
        "frame_id", "start_ns", "end_ns", "is_static", "is_trading", "_bars"
    )

    def __init__(
        self,
        frame_id: int,
        start_ns: int,
        end_ns: int,
        is_static: bool = False,
        is_trading: bool = True,
    ):
        self.frame_id = frame_id
        self.start_ns = start_ns
        self.end_ns = end_ns
        self.is_static = is_static
        self.is_trading = is_trading
        self._bars: dict[str, Bar] = {}

    def add_bar(self, bar: Bar) -> None:
        self._bars[bar.symbol] = bar

    @property
    def bars(self) -> dict[str, Bar]:
        return self._bars

    # ---- 逐标的便捷访问（无数据时抛 KeyError，策略需自行判断） ----

    def _bar(self, symbol: str) -> Bar:
        return self._bars[symbol]

    def open(self, symbol: str) -> float:
        return self._bars[symbol].open

    def high(self, symbol: str) -> float:
        return self._bars[symbol].high

    def low(self, symbol: str) -> float:
        return self._bars[symbol].low

    def close(self, symbol: str) -> float:
        return self._bars[symbol].close

    def last(self, symbol: str) -> float:
        return self._bars[symbol].close

    def volume(self, symbol: str) -> float:
        return self._bars[symbol].volume

    def amount(self, symbol: str) -> float:
        return self._bars[symbol].amount

    def has_bar(self, symbol: str) -> bool:
        return symbol in self._bars

    def symbols(self) -> list[str]:
        return sorted(self._bars)

    def bbo(self, symbol: str) -> Quote:
        """返回本帧 BBO：bid=low, ask=high, last=close。"""
        bar = self._bars[symbol]
        return Quote(bid=bar.low, ask=bar.high, last=bar.close)

    def mid(self, symbol: str) -> float:
        q = self.bbo(symbol)
        return q.mid

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"MarketFrame(id={self.frame_id}, "
            f"t=({self.start_ns},{self.end_ns}], "
            f"static={self.is_static}, trading={self.is_trading}, bars={self.symbols()})"
        )


class StaticFrameBuilder:
    """空帧填充工具：某帧无数据时，以前帧 BBO 生成 is_static 帧。

    由 Engine 在「数据已耗尽但仍有在途订单」或「跳过区间末帧」时调用。
    """

    def __init__(self):
        self._last_bbo: dict[str, Quote] = {}

    def refresh(self, frame: MarketFrame) -> None:
        """用本帧真实数据更新缓存的 BBO。"""
        for sym, bar in frame.bars.items():
            self._last_bbo[sym] = Quote(bid=bar.low, ask=bar.high, last=bar.close)

    def rebuild_static(self, frame_id: int, start_ns: int, end_ns: int) -> MarketFrame:
        """基于上次真实 BBO 重建一帧静态帧。

        静态帧代表「本帧无新信息」：以前帧收盘价为可成交价（open=close=last），
        high/low 保留前帧极差，供 BBO 类条件单判断。
        """
        fr = MarketFrame(frame_id, start_ns, end_ns, is_static=True)
        for sym, q in self._last_bbo.items():
            fr.add_bar(
                Bar(
                    timestamp=end_ns,
                    symbol=sym,
                    open=q.last,
                    high=q.ask,
                    low=q.bid,
                    close=q.last,
                    volume=0.0,
                )
            )
        return fr
