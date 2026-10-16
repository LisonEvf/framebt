"""因子库：WorldQuant 101 Alpha 与国泰君安 191 Alpha 风格因子。

只收录**可由日频 OHLCV + vwap + 市值推导**的因子（原版中依赖行业分类
``indclass`` 的约 19 个 WQ 因子、以及依赖 L2/财务数据的因子不在其中 —— 本数据集
没有行业与财务 PIT 数据，见 README「实现边界」）。

每个因子是 ``(Panel) -> DataFrame`` 的纯函数，只调用 :mod:`framebt.factors.ops`
的算子。**同一份定义同时服务研究与回测**（见 ``factors.panel`` 的说明）。

命名：``wq001``..``wq101`` 对应 WorldQuant Alpha#N；``gtja001``.. 对应国泰君安 191。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ops import (
    abs_, adv, correlation, covariance, decay_linear, delay, delta, log, rank,
    returns, safe_div, scale, sign, signed_power, sqrt, ts_argmax, ts_max,
    ts_min, ts_rank, ts_std, ts_sum, where,
)

C = pd.DataFrame  # 便捷别名


def _const(p, v: float) -> pd.DataFrame:
    return pd.DataFrame(v, index=p.close.index, columns=p.close.columns)


# ===========================================================================
# WorldQuant 101 Alpha（可仅用价量数据计算的子集）
# ===========================================================================

def wq001(p):
    """rank(Ts_ArgMax(SignedPower(((returns<0)?stddev(returns,20):close),2),5)) - 0.5"""
    inner = where(p.returns() < 0, ts_std(p.returns(), 20), p.close)
    return rank(ts_argmax(signed_power(inner, 2), 5)) - 0.5


def wq002(p):
    """-1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)"""
    return -1 * correlation(rank(delta(log(p.volume), 2)),
                            rank(safe_div(p.close - p.open, p.open)), 6)


def wq003(p):
    """-1 * correlation(rank(open), rank(volume), 10)"""
    return -1 * correlation(rank(p.open), rank(p.volume), 10)


def wq004(p):
    """-1 * Ts_Rank(rank(low), 9)"""
    return -1 * ts_rank(rank(p.low), 9)


def wq005(p):
    """rank(open - sum(vwap,10)/10) * (-1 * abs(rank(close - vwap)))"""
    return rank(p.open - ts_sum(p.vwap, 10) / 10) * (-1 * abs_(rank(p.close - p.vwap)))


def wq006(p):
    """-1 * correlation(open, volume, 10)"""
    return -1 * correlation(p.open, p.volume, 10)


def wq007(p):
    """(adv20 < volume) ? (-1*ts_rank(abs(delta(close,7)),60)*sign(delta(close,7))) : -1"""
    a = -1 * ts_rank(abs_(delta(p.close, 7)), 60) * sign(delta(p.close, 7))
    return where(p.adv(20) < p.volume, a, _const(p, -1.0))


def wq008(p):
    """-1 * rank((sum(open,5)*sum(returns,5)) - delay(...,10))"""
    inner = ts_sum(p.open, 5) * ts_sum(p.returns(), 5)
    return -1 * rank(inner - delay(inner, 10))


def _wq09_10_inner(p):
    d = delta(p.close, 1)
    return d, where(ts_min(d, 5) > 0, d, where(ts_max(d, 5) < 0, d, -1 * d))


def wq009(p):
    """(0<ts_min(delta(close,1),5)) ? delta : ((ts_max<0) ? delta : -delta)"""
    return _wq09_10_inner(p)[1]


def wq010(p):
    """rank(与 alpha009 同构，窗口 4)"""
    d = delta(p.close, 1)
    return rank(where(ts_min(d, 4) > 0, d, where(ts_max(d, 4) < 0, d, -1 * d)))


def wq011(p):
    """(rank(ts_max(vwap-close,3)) + rank(ts_min(vwap-close,3))) * rank(delta(volume,3))"""
    return (rank(ts_max(p.vwap - p.close, 3)) + rank(ts_min(p.vwap - p.close, 3))) \
        * rank(delta(p.volume, 3))


def wq012(p):
    """sign(delta(volume,1)) * (-1 * delta(close,1))"""
    return sign(delta(p.volume, 1)) * (-1 * delta(p.close, 1))


def wq013(p):
    """-1 * rank(covariance(rank(close), rank(volume), 5))"""
    return -1 * rank(covariance(rank(p.close), rank(p.volume), 5))


def wq014(p):
    """(-1 * rank(delta(returns,3))) * correlation(open, volume, 10)"""
    return (-1 * rank(delta(p.returns(), 3))) * correlation(p.open, p.volume, 10)


def wq015(p):
    """-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
    return -1 * ts_sum(rank(correlation(rank(p.high), rank(p.volume), 3)), 3)


def wq016(p):
    """-1 * rank(covariance(rank(high), rank(volume), 5))"""
    return -1 * rank(covariance(rank(p.high), rank(p.volume), 5))


def wq017(p):
    """(-1*rank(ts_rank(close,10))) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume/adv20,5))"""
    return ((-1 * rank(ts_rank(p.close, 10)))
            * rank(delta(delta(p.close, 1), 1))
            * rank(ts_rank(safe_div(p.volume, p.adv(20)), 5)))


def wq018(p):
    """-1 * rank(stddev(abs(close-open),5) + (close-open) + correlation(close,open,10))"""
    return -1 * rank(ts_std(abs_(p.close - p.open), 5) + (p.close - p.open)
                     + correlation(p.close, p.open, 10))


def wq019(p):
    """(-1*sign((close-delay(close,7)) + delta(close,7))) * (1 + rank(1+sum(returns,250)))"""
    return (-1 * sign((p.close - delay(p.close, 7)) + delta(p.close, 7))) \
        * (1 + rank(1 + ts_sum(p.returns(), 250)))


def wq020(p):
    """(-1*rank(open-delay(high,1))) * rank(open-delay(close,1)) * rank(open-delay(low,1))"""
    return ((-1 * rank(p.open - delay(p.high, 1)))
            * rank(p.open - delay(p.close, 1))
            * rank(p.open - delay(p.low, 1)))


def wq022(p):
    """-1 * (delta(correlation(high,volume,5),5) * rank(stddev(close,20)))"""
    return -1 * (delta(correlation(p.high, p.volume, 5), 5) * rank(ts_std(p.close, 20)))


def wq023(p):
    """(sum(high,20)/20 < high) ? (-1*delta(high,2)) : 0"""
    return where(ts_sum(p.high, 20) / 20 < p.high, -1 * delta(p.high, 2), 0.0)


def wq024(p):
    """(delta(sum(close,100)/100,100)/delay(close,100) <= 0.05) ? -1*(close-ts_min(close,100)) : -1*delta(close,3)"""
    ratio = safe_div(delta(ts_sum(p.close, 100) / 100, 100), delay(p.close, 100))
    return where(ratio <= 0.05, -1 * (p.close - ts_min(p.close, 100)), -1 * delta(p.close, 3))


def wq026(p):
    """-1 * ts_max(correlation(ts_rank(volume,5), ts_rank(high,5), 5), 3)"""
    return -1 * ts_max(correlation(ts_rank(p.volume, 5), ts_rank(p.high, 5), 5), 3)


def wq028(p):
    """scale(correlation(adv20, low, 5) + (high+low)/2 - close)"""
    return scale(correlation(p.adv(20), p.low, 5) + (p.high + p.low) / 2 - p.close)


def wq033(p):
    """rank(-1 * (1 - open/close))"""
    return rank(-1 * (1 - safe_div(p.open, p.close)))


def wq034(p):
    """rank((1 - rank(stddev(returns,2)/stddev(returns,5))) + (1 - rank(delta(close,1))))"""
    return rank((1 - rank(safe_div(ts_std(p.returns(), 2), ts_std(p.returns(), 5))))
                + (1 - rank(delta(p.close, 1))))


def wq038(p):
    """(-1*rank(delta(close,7) * (1 - rank(decay_linear(volume/adv20,20))))) * (1+rank(sum(returns,250)))"""
    return (-1 * rank(delta(p.close, 7)
                      * (1 - rank(decay_linear(safe_div(p.volume, p.adv(20)), 20))))) \
        * (1 + rank(ts_sum(p.returns(), 250)))


def wq040(p):
    """(-1 * rank(stddev(high,10))) * correlation(high, volume, 10)"""
    return (-1 * rank(ts_std(p.high, 10))) * correlation(p.high, p.volume, 10)


def wq041(p):
    """sqrt(high*low) - vwap"""
    return sqrt(p.high * p.low) - p.vwap


def wq042(p):
    """rank(vwap-close) / rank(vwap+close)"""
    return safe_div(rank(p.vwap - p.close), rank(p.vwap + p.close))


def wq043(p):
    """ts_rank(volume/adv20, 20) * ts_rank(-1*delta(close,7), 8)"""
    return ts_rank(safe_div(p.volume, p.adv(20)), 20) * ts_rank(-1 * delta(p.close, 7), 8)


def wq044(p):
    """-1 * correlation(high, rank(volume), 5)"""
    return -1 * correlation(p.high, rank(p.volume), 5)


def wq045(p):
    """-1 * rank(sum(delay(close,5),20)/20) * correlation(close,volume,2) * rank(correlation(sum(close,5),sum(close,20),2))"""
    return -1 * (rank(ts_sum(delay(p.close, 5), 20) / 20)
                 * correlation(p.close, p.volume, 2)
                 * rank(correlation(ts_sum(p.close, 5), ts_sum(p.close, 20), 2)))


def _wq46_49_inner(p):
    return ((delay(p.close, 20) - delay(p.close, 10)) / 10
            - (delay(p.close, 10) - p.close) / 10)


def wq046(p):
    """(inner>0.25) ? -1 : ((inner<0) ? 1 : -1*(close-delay(close,1)))"""
    inner = _wq46_49_inner(p)
    return where(inner > 0.25, -1.0, where(inner < 0, 1.0, -1 * (p.close - delay(p.close, 1))))


def wq049(p):
    """(inner < -0.1) ? 1 : -1*(close-delay(close,1))"""
    inner = _wq46_49_inner(p)
    return where(inner < -0.1, 1.0, -1 * (p.close - delay(p.close, 1)))


def wq050(p):
    """-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)"""
    return -1 * ts_max(rank(correlation(rank(p.volume), rank(p.vwap), 5)), 5)


def wq051(p):
    """(inner < -0.05) ? 1 : -1*(close-delay(close,1))"""
    inner = _wq46_49_inner(p)
    return where(inner < -0.05, 1.0, -1 * (p.close - delay(p.close, 1)))


def wq052(p):
    """((-1*ts_min(low,5)) + delay(ts_min(low,5),5)) * rank((sum(returns,240)-sum(returns,20))/sum(returns,20))"""
    r = p.returns()
    return ((-1 * ts_min(p.low, 5)) + delay(ts_min(p.low, 5), 5)) \
        * rank(safe_div(ts_sum(r, 240) - ts_sum(r, 20), ts_sum(r, 20)))


def wq053(p):
    """-1 * delta(((close-low) - (high-close)) / (close-low), 9)"""
    return -1 * delta(safe_div((p.close - p.low) - (p.high - p.close),
                               p.close - p.low), 9)


def wq054(p):
    """-1 * ((low-close) * open^5) / ((low-high) * close^5)"""
    return -1 * safe_div((p.low - p.close) * (p.open ** 5),
                         (p.low - p.high) * (p.close ** 5))


def wq057(p):
    """0 - (close-vwap) / decay_linear(rank(ts_argmax(close,30)), 2)"""
    return 0 - safe_div(p.close - p.vwap, decay_linear(rank(ts_argmax(p.close, 30)), 2))


def wq060(p):
    """0 - (((2*close - high - low) * open) / ((high-low)*close)) * volume"""
    return 0 - safe_div((2 * p.close - p.high - p.low) * p.open,
                        (p.high - p.low) * p.close) * p.volume


def wq094(p):
    """(rank(1/close) * volume) / adv20"""
    return safe_div(rank(safe_div(1.0, p.close)) * p.volume, p.adv(20))


def wq096(p):
    """-1 * ts_rank(rank(correlation(close, volume, 6)), 5)"""
    return -1 * ts_rank(rank(correlation(p.close, p.volume, 6)), 5)


def wq101(p):
    """(close - open) / ((high - low) + 0.001)"""
    return safe_div(p.close - p.open, (p.high - p.low) + 0.001)


# ===========================================================================
# 国泰君安 191 Alpha（选取不依赖财务/行业数据的代表因子）
# ===========================================================================

def gtja001(p):
    """(-1 * CORR(RANK(DELTA(LOG(VOLUME),1)), RANK((CLOSE-OPEN)/OPEN), 6))"""
    return -1 * correlation(rank(delta(log(p.volume), 1)),
                            rank(safe_div(p.close - p.open, p.open)), 6)


def gtja002(p):
    """(-1 * DELTA(((CLOSE-LOW) - (HIGH-CLOSE)) / (HIGH-LOW), 1))"""
    return -1 * delta(safe_div((p.close - p.low) - (p.high - p.close),
                               p.high - p.low), 1)


def gtja005(p):
    """(-1 * TSMAX(CORR(TSRANK(VOLUME,5), TSRANK(HIGH,5), 5), 3))"""
    return -1 * ts_max(correlation(ts_rank(p.volume, 5), ts_rank(p.high, 5), 5), 3)


def gtja006(p):
    """(RANK(SIGN(DELTA((OPEN*0.85 + HIGH*0.15), 4))) * -1)"""
    return rank(sign(delta(p.open * 0.85 + p.high * 0.15, 4))) * -1


def gtja007(p):
    """((RANK(MAX(VWAP-CLOSE,3)) + RANK(MIN(VWAP-CLOSE,3))) * RANK(DELTA(VOLUME,3)))"""
    return (rank(ts_max(p.vwap - p.close, 3)) + rank(ts_min(p.vwap - p.close, 3))) \
        * rank(delta(p.volume, 3))


def gtja008(p):
    """RANK(DELTA((CLOSE*0.8 + OPEN*0.2), 4) * -1)"""
    return rank(delta(p.close * 0.8 + p.open * 0.2, 4) * -1)


def gtja012(p):
    """RANK(OPEN - (SUM(VWAP,10)/10)) * (-1 * RANK(ABS(CLOSE - VWAP)))"""
    return rank(p.open - ts_sum(p.vwap, 10) / 10) * (-1 * rank(abs_(p.close - p.vwap)))


def gtja014(p):
    """CLOSE - DELAY(CLOSE, 5)"""
    return p.close - delay(p.close, 5)


def gtja018(p):
    """CLOSE / DELAY(CLOSE, 5)"""
    return safe_div(p.close, delay(p.close, 5))


def gtja033(p):
    """RANK((-1 * ((1 - (OPEN/CLOSE))^1)))"""
    return rank(-1 * (1 - safe_div(p.open, p.close)))


def gtja041(p):
    """(-1 * RANK(TSMAX(DELTA(VWAP,3), 5)))"""
    return -1 * rank(ts_max(delta(p.vwap, 3), 5))


# ===========================================================================
# 注册表
# ===========================================================================

FACTOR_LIBRARY: dict[str, callable] = {
    f"wq{n:03d}": fn for n, fn in sorted(
        {1: wq001, 2: wq002, 3: wq003, 4: wq004, 5: wq005, 6: wq006, 7: wq007,
         8: wq008, 9: wq009, 10: wq010, 11: wq011, 12: wq012, 13: wq013, 14: wq014,
         15: wq015, 16: wq016, 17: wq017, 18: wq018, 19: wq019, 20: wq020,
         22: wq022, 23: wq023, 24: wq024, 26: wq026, 28: wq028, 33: wq033,
         34: wq034, 38: wq038, 40: wq040, 41: wq041, 42: wq042, 43: wq043,
         44: wq044, 45: wq045, 46: wq046, 49: wq049, 50: wq050, 51: wq051,
         52: wq052, 53: wq053, 54: wq054, 57: wq057, 60: wq060, 94: wq094,
         96: wq096, 101: wq101}.items())
}
FACTOR_LIBRARY.update({
    f"gtja{n:03d}": fn for n, fn in sorted(
        {1: gtja001, 2: gtja002, 5: gtja005, 6: gtja006, 7: gtja007, 8: gtja008,
         12: gtja012, 14: gtja014, 18: gtja018, 33: gtja033, 41: gtja041}.items())
})


def list_factors() -> list[str]:
    return sorted(FACTOR_LIBRARY)


def compute(name: str, panel) -> pd.DataFrame:
    """计算单个因子；未知名称抛 KeyError。"""
    if name not in FACTOR_LIBRARY:
        raise KeyError(f"未知因子 {name!r}；可用 {len(FACTOR_LIBRARY)} 个")
    out = FACTOR_LIBRARY[name](panel)
    return out.replace([np.inf, -np.inf], np.nan)


def compute_many(names: list[str], panel) -> dict[str, pd.DataFrame]:
    """批量计算；单个因子失败不影响其余（记为 NaN 面板）。"""
    out: dict[str, pd.DataFrame] = {}
    for n in names:
        try:
            out[n] = compute(n, panel)
        except Exception:  # noqa: BLE001 - 因子在部分数据上可能退化
            out[n] = pd.DataFrame(np.nan, index=panel.close.index,
                                  columns=panel.close.columns)
    return out
