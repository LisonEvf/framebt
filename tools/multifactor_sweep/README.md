# 经典案例 · 多因子基础 + Sweep 做 T 的主动交易策略

> 本文用一套**真实跑通、带完整回测报告**的策略，讲解「如何用 framebt 正确地搭建并回测
> 一个横截面多因子 + 战术叠加策略」。

## 本次成果

区间 **2023-07-07 → 2026-10-19**（近 3 年，约 725 交易日），**样本外**结果：

| 指标 | 结果 |
|---|---|
| 总收益 | **+13%**（年化 +3.74%） |
| Sharpe | 0.26 · 最大回撤 22% · 胜率 48% |
| 成交 | 1,407 笔 / 87 标的 |
| 已实现盈亏 | +208,017 元（620 回合，与权益曲线一致） |
| 最优权重 | wq043 0.17 · wq007 0.16 · wq002 0.16 · wq096 0.12（其余等权 ~0.04，详见 `best_weights.json`） |

> ⚠️ **定位**：本案例的价值在**「正确用法」**，不在 alpha。+13% 是一个「没有自欺、口径
> 正确」的结果，不是「稳赚」保证。同框架自适应多因子系统（`docs/MULTIFACTOR.md`）样本外是
> **−78%**。两相对照说明：**框架正确 ≠ 策略赚钱**——框架能消除「前视偏差 / 同帧泄漏 / 口径
> 错乱 / 不可复现」这类**技术性自欺**，消除不了「因子本身没有超额收益」这类**策略性失败**。
> 本文教你守住前者、诚实地呈现后者。

---

## 目录

- [本次成果](#本次成果)
- [1. 交付物](#1-交付物)
- [2. 策略架构：核心 + 卫星](#2-策略架构核心--卫星)
- [3. 正确用法（核心要点）](#3-正确用法核心要点)
- [4. 两个真实陷阱与修复（最大价值）](#4-两个真实陷阱与修复最大价值)
- [5. 幸存者偏差](#5-幸存者偏差)
- [6. 复现与调参](#6-复现与调参)
- [7. 文件与参考](#7-文件与参考)

---

## 1. 交付物

| 文件 | 用途 |
|---|---|
| [`mf_sweep_report.html`](mf_sweep_report.html) | **主交付物**：自包含 HTML（指标卡 + 权益曲线/回撤 + **15 张逐标的 K 线图** + 成交明细表） |
| [`mf_sweep_report.md`](mf_sweep_report.md) | 中文 Markdown 报告 |
| [`mf_sweep_trades.csv`](mf_sweep_trades.csv) | 1407 笔逐笔成交明细 |
| [`mf_sweep_roundtrips.csv`](mf_sweep_roundtrips.csv) | 620 个已实现盈亏回合 |
| [`best_weights.json`](best_weights.json) · [`weight_search_log.json`](weight_search_log.json) | 最优权重 + 权重搜索日志 |
| `data_prep.py` / `backtest_engine.py` / `optimize_weights.py` / `run.py` | 可复现的完整代码 |

---

## 2. 策略架构：核心 + 卫星

`MultiFactorSweepStrategy`（`src/framebt/strategies/multifactor_sweep.py`）由核心 + 卫星构成：

```
组合收益 = 核心（多因子，~70% 仓位） + 战术（Sweep 做 T，~30% 闲置容量）
```

```
┌─────────────────────────────────────────────────────────────────┐
│  核心：14 个 WQ101 Alpha 因子 → 截面加权合成打分 → Top20 等权      │
│       → 每 5 交易日调仓；达标不足 5 只 → 主动空仓                   │
├──────────────────────────────────────────────────────────────────┤
│  卫星：收盘突破近 15 日高点 + 放量>1.5×均量 + 当日阳线 → 触发        │
│       → 持有≤5 日 / 跌破触发位了结 → 反复滚动；仅在核心容量之外挑新标的 │
└─────────────────────────────────────────────────────────────────┘
                          ▼
   风险规避期（指数跌破 MA60）→ 总仓位归 0（完全空仓，现金）
```

解决两个真实问题：① 多因子单独用易「钝化」，合成 + 调仓 + 主动选股 + 可空仓才成系统；
② A 股只能 T+1，但可「底仓 + 日内 T」——核心仓长持，用闲置容量滚动做短周期 T。

---

## 3. 正确用法（核心要点）

### 3.1 因果信号：信号后向计算，策略只读「当天已发生」

避免前视偏差（I1）的第一道防线。所有信号（因子打分、sweep、eligibility、regime）都在
**数据准备阶段**预先算好：

```python
# data_prep.py —— 研究侧，向量化、可缓存
panel = Panel.from_feed(window_feed)
factor_panels = compute_factor_panels(panel, FACTOR_NAMES)   # 去极值→标准化→市值中性化
eligibility = compute_eligibility(panel, symbols=pool)       # 动态流动性合格
sweep_signal, sweep_exit = compute_sweep_signals(panel)      # 突破/放量 / 跌破
regime = compute_regime(panel)                               # 指数 > MA60 → 风险偏好
```

要点：① **因子只认 `Panel`**（时间×标的对齐面板），研究与回测共用同一份实现（「流批一体」）；
② **时间戳是日历 ordinal，不是 pandas 纳秒**——切窗口、算 OOS 起点用 `int(ts)` 直接比较；
③ 结果 pickle 缓存到 `cache/`（因子计算最耗时，全池 5575 只），改参数要 `force=True` 或删缓存；
④ 流动性筛选用「池内最高 120 日均额」的**相对比例**（非绝对阈值），否则排除掉几乎所有标的。

### 3.2 引擎装配：`delay_frames=1` 是防自欺命门

```python
# backtest_engine.py
engine = Engine(
    prep.feed, strat,
    initial_cash=1_000_000.0,
    settlement_cycle=1,       # A 股 T+1：买入次日才可卖
    delay_frames=1,           # ★ 帧 t 下单 → 帧 t+1 才成交
    cost_model=make_cost(),   # 佣金/印花税/过户费/量价滑点
    rules=make_rules(),       # 涨跌停 ±10%
)
```

绝大多数前视偏差不是「偷看明天 K 线」，而是「用 t 帧收盘价下单、假设以 t 帧价格成交」。
实盘收盘瞬间你根本买不进——事件驱动框架默认你能。framebt 把订单强制推到**下一帧开盘价**
撮合（不变式 I2），即「收盘信息决策、次日开盘执行」，物理上不可达前视。

A 股成本（`StandardCost`）：佣金 0.03%(最低 5)、印花税 0.1%(仅卖)、过户费 0.002%、
量价滑点 k=0.001。A 股规则（`StandardAshareRules`）：涨跌停 ±10%、T+1、停牌日历。
> 默认 `risk=NoOpRisk()` 不拦超买——漏配会让策略静默失效或现金变负（隐含杠杆），
> 严肃回测请显式传 `StandardRisk(...)`。

### 3.3 主动选股 + 可空仓

- **主动选股**：核心仓从 `eligibility=True` 的标的按 composite 打分取 Top20；Sweep 仓从
  「流动性合格 + 信号成立 + **不在核心池**」里主动挑「新中标的」，填补核心满仓之外的闲置容量。
- **可空仓**：达标核心标的不足 `min_hold`（默认 5 只）或进入风险规避期（`regime=False`）
  任一满足则接受空仓（现金）。与其在弱势市场硬持，不如持有现金——「最大回撤 22%」
  部分就来源于规避期空仓，这是框架的**市场状态门控**能力。

### 3.4 Sweep 做 T

```python
prev_high = ts_max(close, 15).shift(1)     # 近 15 日高点（不含当日）
breakout  = close > prev_high              # 收盘突破
volume_ok = vol > 1.5 * ts_mean(vol, 15)   # 放量 > 1.5× 均量
bullish   = close > open                   # 当日阳线
signal    = (breakout & volume_ok & bullish)
exit_sig  = close < prev_high              # 跌破触发位 → 了结
```

「突破 + 放量 + 阳线」三者同时成立才算激进突破，过滤假突破；持有 ≤5 日或跌破即了结。
T+1 自洽：sweep 仓持有期 ≥1 日，天然满足结算约束。

### 3.5 权重优化：按「样本外 Sharpe」选（不是样本内收益）

最容易过拟合的环节。四步：① 样本内（前 60% 交易日）算每个因子的平均日度 IC；
② 生成 9 组候选（等权 / 带符号 IC / |IC| / 分块 / 各块放大）；
③ **按样本外 Sharpe 排序选优**（切最后 40% 交易日），要求样本内 Sharpe>0（淘汰过拟合）；
④ 坐标上升在最优候选上逐个因子 ±0.05 微调。

```python
# optimize_weights.py —— 只优化「样本外 Sharpe」，避免过拟合样本内
best = min(pool, key=lambda r: r["oos_sharpe"], reverse=True)
refined = _coordinate_ascent(prep, best["weights"], oos_start_ts)
```

本案例 `block_volume` 经坐标上升从 OOS-Sharpe 0.284 升到 0.745，样本内 Sharpe 由负转正——
「样本外好、样本内也不差」的健康优化。

### 3.6 生成带 K 线图的报告

```python
from framebt.report import generate_report, ReportConfig

html = generate_report(
    res,
    bars_by_symbol={},                    # ★ 空字典 + source 触发按需加载
    source=OpentdxParquetSource(),        # 按成交标的从数仓加载 K 线
    strategy_name="多因子 + Sweep 做 T",
    config=ReportConfig(max_trade_charts=15),   # 最多渲染多少只标的的 K 线图
    output_path="mf_sweep_report.html",
)
```

报告自包含（图表 base64 内嵌，可离线打开）：指标卡、权益曲线 + 回撤、**逐标的 K 线图**
（蜡烛 + 成交量 + MA5/MA20 + 买/卖标注）、成交明细表。

---

## 4. 两个真实陷阱与修复（最大价值）

这两个 bug 都是「**结果看起来对、但口径错了**」——它们不会报错，只会让你**相信一条错误
的曲线**。这正是 framebt 要消除的「技术性自欺」。

### 陷阱 1：已实现盈亏对不上权益曲线

**症状**：报告说「已实现盈亏 −260 万元」，但权益曲线 +13%——差 2.7 亿。**根因**：核心组合
「加仓后再调仓卖出」，旧回合配对在**部分平仓**时把整个持仓成本一次性冲销，对没卖掉的份额
也算了盈亏。**修复**：改用**平均成本法**——卖出按当前持仓平均成本算已实现盈亏，逐步核减持仓，
**只有持仓归零才闭合一个回合**。修复后 620 个回合合计 **+208,017 元**，与权益曲线变化
（+127k 净收益 + ~51k 费用）精确吻合。

```python
# 修复后：平均成本法，部分平仓不再冲销未卖出份额
avg_cost = open_notional[sym] / open_size[sym]
pnl = (f.price - avg_cost) * f.size
open_notional[sym] -= avg_cost * f.size
open_size[sym]     -= f.size
if open_size[sym] <= 0:          # 持仓归零才闭合一个回合
    rows.append(回合)
```

> **自检**：任何回测报告，「已实现盈亏之和」必须与「权益曲线变化 − 费用」量级一致；对不上，
> 一定是回合口径错了——别信那条曲线。

### 陷阱 2：交易 K 线图静默缺失

**根因**：`generate_report` 用 `if bars_by_symbol:` 判断是否画 K 线图，空字典为假值，
只传 `source` 永不触发。**修复**：改为 `if bars_by_symbol is not None:`，调用方传
`bars_by_symbol={}` + `source=`。修复后报告嵌入 **15 张逐标的 K 线图**。

> **自检**：打开报告，确认「交易 K 线」章节有图且每只标的带「买/卖」标注；没有就是被
> 静默跳过了。

---

## 5. 幸存者偏差

本策略沿用「全 A 当前流动性前 300 只」的**静态标的池**，存在幸存者偏差：池内均为能活至今的
大盘股，回测收益会**高估真实可获利性**。权重优化基于样本外 Sharpe，已尽量抑制过拟合，但
静态池上测得的正收益仍需谨慎看待。这是所有「当前存活标的」回测的共有边界（`reports/
VALIDATION.md`、`docs/MULTIFACTOR.md` 都明确标注）。产品级策略应换成**时间对齐的动态标的池**
（当时已上市的股票）并做**多段滚动样本外**验证。

---

## 6. 复现与调参

```bash
cd framebt
export PYTHONPATH=src            # Windows: $env:PYTHONPATH="...\framebt\src"

python tools/multifactor_sweep/run.py          # 完整流程（含权重优化，约 10 分钟）
python tools/multifactor_sweep/run.py --no-opt # 跳过优化，直接用已保存的最优权重
python tools/multifactor_sweep/optimize_weights.py
python tools/multifactor_sweep/data_prep.py --force   # 强制重算因子
```

依赖：核心引擎零第三方依赖；数据源需 `pandas + pyarrow`；图表报告需 `matplotlib`；数据源
默认 `../opentdx/db/data`，可用 `FRAMEBT_DATA_DIR` 覆盖。

| 参数 | 默认 | 含义 | 参数 | 默认 | 含义 |
|---|---|---|---|---|---|
| `top_k` | 20 | 核心持仓只数 | `sweep_lookback` | 15 | Sweep 突破回溯 |
| `rebalance_every` | 5 | 核心调仓间隔 | `sweep_vol_mult` | 1.5 | 放量阈值倍数 |
| `base_gross` | 0.70 | 核心目标仓位 | `sweep_hold_max` | 5 | Sweep 最长持有 |
| `regime_off_gross` | 0.0 | 风险规避期仓位 | `sweep_budget_pct` | 0.30 | Sweep 最大占比 |

---

## 7. 文件与参考

- **源码**：`data_prep.py`（数据/因子/信号）· `backtest_engine.py`（引擎/成本/规则，优化器与运行器复用）· `optimize_weights.py`（IC/候选/样本外选优/坐标上升）· `run.py`（端到端编排）· `src/framebt/strategies/multifactor_sweep.py`（策略本身）
- **参考**：[`../USER_GUIDE.md`](../USER_GUIDE.md)（框架总指南）· [`../docs/MULTIFACTOR.md`](../docs/MULTIFACTOR.md)（负结果，对照）· [`../docs/INTRADAY_T_TUTORIAL.md`](../docs/INTRADAY_T_TUTORIAL.md)（扫单分时 T 技术手册）· [`../reports/INDEX.md`](../reports/INDEX.md)（验证报告索引）
