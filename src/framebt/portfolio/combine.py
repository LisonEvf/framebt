"""因子合成与目标持仓（设计文档 §3 的组合侧）。

- ``composite_score``：按权重把多个（已预处理的）因子面板合成一个打分为面板；
- ``select_portfolio``：把某一时刻的打分截面转成目标权重（多头 / 多空）；
- ``turnover``：目标权重与当前权重的单边换手。

同样保持「研究/回测共用」：这些函数既不读时钟也不读引擎，只吃面板与权重。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-12


def composite_score(factors: dict[str, pd.DataFrame],
                    weights: dict[str, float]) -> pd.DataFrame:
    """加权合成打分：``score = Σ_i w_i · f_i``（按因子取齐，缺失跳过）。

    只累加**该时刻有值**的因子，并按参与因子的 |w| 之和归一，避免缺失导致打分
    系统性偏小。
    """
    total = None
    wsum = None
    for name, w in weights.items():
        f = factors.get(name)
        if f is None or abs(w) < EPS:
            continue
        contrib = f * w
        total = contrib if total is None else total.add(contrib, fill_value=0.0)
        aw = f.notna().astype(float) * abs(w)
        wsum = aw if wsum is None else wsum.add(aw, fill_value=0.0)
    if total is None:
        first = next(iter(factors.values()))
        return pd.DataFrame(np.nan, index=first.index, columns=first.columns)
    return total.div(wsum.where(wsum > EPS))


def select_portfolio(score_row: pd.Series, top_k: int = 5,
                     mode: str = "long", gross: float = 1.0,
                     current: set[str] | None = None,
                     buffer_mult: float = 2.0) -> dict[str, float]:
    """把打分截面转成目标权重。

    ``mode='long'``  ：取打分最高的 top_k，等权（Σw = gross）
    ``mode='long_short'``：多 top_k / 空 bottom_k，各占 gross/2

    **缓冲带（buffer band）**：若给出 ``current``（当前持仓集合），则当前持仓中
    仍排在前 ``top_k × buffer_mult`` 名的标的**继续保留**，只把腾出的名额分给新进榜者。
    这是抑制换手的标准做法 —— 否则排名在边界附近反复穿越会导致组合天天大换血，
    交易成本吃掉全部 alpha。
    """
    s = score_row.dropna()
    if len(s) == 0:
        return {}
    ranked = s.sort_values(ascending=False)
    half = 2 if mode == "long_short" else 1
    k = max(1, min(top_k, len(s) // half or 1))

    if mode == "long":
        if current:
            keep_zone = int(max(k, round(k * buffer_mult)))
            holds = [sym for sym in ranked.index[:keep_zone] if sym in current]
            picks = holds[:k]
            for sym in ranked.index:
                if len(picks) >= k:
                    break
                if sym not in picks:
                    picks.append(sym)
        else:
            picks = list(ranked.index[:k])
        if not picks:
            return {}
        w = gross / len(picks)
        return {str(sym): float(w) for sym in picks}

    if mode == "long_short":
        longs = list(ranked.index[:k])
        shorts = list(ranked.index[-k:])
        wl = (gross / 2.0) / len(longs)
        ws = -(gross / 2.0) / len(shorts)
        out: dict[str, float] = {}
        for sym in longs:
            out[str(sym)] = float(wl)
        for sym in shorts:
            out[str(sym)] = out.get(str(sym), 0.0) + float(ws)
        return out

    raise ValueError(f"未知 mode {mode!r}；可选 'long' / 'long_short'")


def turnover(target: dict[str, float], current: dict[str, float]) -> float:
    """单边换手：``Σ|w_target − w_current| / 2``。"""
    keys = set(target) | set(current)
    return float(sum(abs(target.get(k, 0.0) - current.get(k, 0.0)) for k in keys) / 2.0)


def weights_to_shares(target: dict[str, float], equity: float,
                      prices: dict[str, float], lot: int = 100) -> dict[str, int]:
    """目标权重 → 目标股数（按 A 股 100 股整手向下取整）。

    价格为 0 或缺失的标的被跳过。
    """
    out: dict[str, int] = {}
    for sym, w in target.items():
        px = prices.get(sym, 0.0)
        if px <= 0 or not np.isfinite(px):
            continue
        notional = w * equity
        shares = int(abs(notional) / px // lot * lot)
        if shares > 0:
            out[sym] = shares if notional >= 0 else -shares
    return out
