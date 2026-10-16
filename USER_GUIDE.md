# framebt 使用指南与社区邀请

> **回测不难，难的是你敢于相信回测结果。**
>
> 大多数回测框架默认「你的策略不骗自己」，于是前视偏差（look-ahead bias）悄悄渗进每一条权益曲线——
> 你以为抓住了 alpha，其实只是抓住了未来。`framebt` 把这件事做反了：它**默认环境在试图欺骗你**，
> 然后用一套物理层面的因果约束，让「看到未来」在工程上不可达。
>
> 本文档面向两类读者：**想立刻跑起来的新人**，和**想知道它到底哪里不一样、值不值得加入的老手**。
> 前者看 §1–§4，后者看 §5–§7，想一起做事情的看 §8。

---

## 目录

- [1. 一页看懂：framebt 是什么](#1-一页看懂framebt-是什么)
- [2. 为什么是 framebt：传统框架的三大隐痛](#2-为什么是-framebt传统框架的三大隐痛)
- [3. 三条公理与五个不变式](#3-三条公理与五个不变式)
- [4. 快速开始：5 分钟跑出第一个回测](#4-快速开始5-分钟跑出第一个回测)
- [5. 核心能力全景](#5-核心能力全景)
- [6. 真实数据：最有说服力的一项验证](#6-真实数据最有说服力的一项验证)
- [7. 与主流框架的差异对照](#7-与主流框架的差异对照)
- [8. 加入我们：这不是一个写完的工具，是一个正在生长的社区](#8-加入我们这不是一个写完的工具是一个正在生长的社区)

---

## 1. 一页看懂：framebt 是什么

`framebt`（**帧刷新量化回测框架**）是一个以「**固定逻辑步长 Δt 主动推进虚拟时间**」为骨架的、
**确定性、因果性**的回测引擎，面向 A 股（也支持 ETF / 多资产横截面）。它的工程实现对应设计文档
《帧刷新量化回测框架 v1.0》，以 [opentdx](../opentdx) 数仓（Parquet + DuckDB）为数据源。

一句话差异点：

> **传统事件驱动框架「跳到下一事件的时间戳」；framebt「以固定 Δt 主动推进时间，事件按真实时间戳注入」。**
> 前者把「时间推进」和「事件到达」焊死在一起，后者把它们解耦——这正是前视偏差滋生的缝隙，也是 framebt 修复的起点。

### 它解决的核心问题：回测可信度

| 回测中最常见的「自欺」 | 传统框架 | framebt |
|---|---|---|
| 用当前数据、以当前价格成交（同帧泄漏） | 靠「自觉」，框架不拦 | **因果令牌 + 执行层延迟**，物理上不可达 |
| 看到未来数据 | 靠预加载时机 | 总线钳制到权威虚拟时间，伪造 token 也拿不到 |
| 结果每次都不一样 | 随机种子没控 | **I3 逐位复现**（同机同解释器逐字节一致） |
| 日内多 Bar 顺序错乱 | 不保证 | **I4 按 (timestamp, sequence_id) 严格全序** |

---

## 2. 为什么是 framebt：传统框架的三大隐痛

### 隐痛 1：「能看到未来」往往不是因为看到了未来，而是因为「看到当前数据并以当前价格成交」

这是 framebt 最反直觉、也最重要的一个判断。绝大多数前视偏差不是来自「策略偷看了明天的 K 线」，
而是来自「策略在 t 帧用 t 帧的收盘价下单，并假设以 t 帧价格成交」。
实盘里 t 帧收盘的瞬间你根本买不进这个价——但事件驱动框架默认你能。

framebt 把防前视的主战场放在**执行层**：帧 t 生成的订单**最早在帧 t+delay 成交**（不变式 I2）。
延迟管线是默认开启的硬约束，不是可选项。

### 隐痛 2：时间推进被事件「绑架」

传统框架「跳到下一根 Bar 的时间戳」，于是「时间本身」消失了——
你无法表达「我每 5 分钟扫一次盘，即使中间没有成交」这种与事件节奏无关的感知策略。
framebt 用**固定 Δt 帧刷新**把时间从事件中解放出来，策略可以自主选择「多久醒一次、关心哪些事件」。

### 隐痛 3：「框架预定义静态数据集」 vs 「策略自由请求」

传统框架要么预加载全量矩阵（向量化），要么预定义静态集合（事件驱动）。
framebt 让**策略动态声明它关心的数据**，引擎在**访问层**按因果过滤返回——
既省内存，又保证「声明不构成白名单」：没声明的请求也会实时计算，但依然被钳制在当前虚拟时间之内。

---

## 3. 三条公理与五个不变式

framebt 的全部设计从三条公理推出（详见 `src/framebt/__init__.py` 顶部注释）：

| 公理 | 含义 |
|---|---|
| **公理一：因果性优先于可见性** | 引擎不规定策略「能看什么变量」，只保证策略「不能因果地使用尚未发生的信息」 |
| **公理二：执行层才是防前视的主战场** | 前视最常见的来源是「看到当前数据并以当前价格成交」，约束施加在**执行层**而非感知层 |
| **公理三：帧刷新是时间的底层节拍** | 策略的感知节奏由策略自主选择（多唤醒机制） |

由此硬编码进引擎的**五个不变式**（既是设计约束，也是测试判据，见 `tests/test_invariants.py`）：

| 不变式 | 含义 |
|---|---|
| **I1** | 任何数据请求返回的记录 `timestamp <= current_virtual_time` |
| **I2** | 帧 t 生成的订单，最早成交于帧 t+delay（延迟管线） |
| **I3** | 相同配置（数据/参数/种子/Δt 序列）逐位可复现 |
| **I4** | 帧内多 Bar 按 `(timestamp, sequence_id)` 严格排序 |
| **I5** | 策略无法访问/改写已过去帧与未来帧 |

> 这不是文档里的口号。`tests/` 有 **246 个测试**逐项校验它们，
> 真实数据验证里 I1 做了 **2000 次「伪造未来 token」攻击，0 次逃逸**。

---

## 4. 快速开始：5 分钟跑出第一个回测

### 4.1 安装

核心引擎**零第三方依赖（纯标准库）**，保证 I3 逐位复现与可移植性；
只有读取 opentdx 数仓 Parquet 时才需要 `pandas + pyarrow`。

```bash
cd framebt
pip install -e .              # 仅核心引擎
pip install -e ".[data,dev]"  # + 数据源 + 测试
python --version              # 需要 >= 3.12
```

数据源默认指向 `../opentdx/db/data`，可用环境变量 `FRAMEBT_DATA_DIR` 或 `data_dir=` 参数覆盖。

### 4.2 准备数据（回填历史）

opentdx 的 ETL 单次 `count=800` 会命中协议边界（返回错乱区间），因此本仓库提供**分页回填工具**，
复用 opentdx 自有的 `persist.py` 写库（schema 一致，不改动 opentdx 源码）：

```bash
python tools/backfill_history.py                  # 默认验证池 24 标的，每标的 2000 根（≈8 年）
python tools/backfill_history.py --max-bars 3000  # ≈12 年
python tools/backfill_history.py --symbols "SH 600519, SZ 000001"
```

回填后数仓含 **24 个 A 股大盘股 × 2000 根前复权日线（2018-05 ~ 2026-10）**。

> ⚠️ **复权口径**：同一标的可能同时存在 `NONE/QFQ/HFQ` 多个文件。数据源**按行内 `period`/`adjust` 列过滤**
> （默认 `DAILY + QFQ`），避免把不同口径混成一条序列。用 `src.available_adjusts(symbol)` 查看某标的实际口径。

### 4.3 第一个回测（真实数据）

```python
from framebt import Engine, Performance
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import DualMAStrategy

src  = OpentdxParquetSource()                # 默认读 opentdx/db/data，DAILY + QFQ
feed = src.load("SH.600519")                 # 2000 根前复权日线
res  = Engine(feed, DualMAStrategy("SH.600519", short=5, long=20),
              initial_cash=100_000.0).run()

print(res.final_equity(), Performance(res).sharpe)
```

### 4.4 第一个回测（合成数据，测试友好）

```python
from framebt import run_backtest
from framebt.strategies import DualMAStrategy

bars = [{"timestamp": i, "open": o, "high": h, "low": l, "close": c, "volume": v}, ...]
res = run_backtest(bars, DualMAStrategy("S", short=5, long=20),
                   symbol="S", initial_cash=100_000.0, seed=1234)
```

### 4.5 装配完整中间件（A 股全规则）

```python
from framebt import (Engine, OpentdxParquetSource, DualMAStrategy,
                      StandardCost, StandardRisk, StandardAshareRules)

engine = Engine(
    src.load("SZ.000001"),
    DualMAStrategy("SZ.000001", 5, 20),
    initial_cash=1_000_000.0,
    settlement_cycle=1,                                  # T+1
    cost_model=StandardCost(commission=0.0003, min_commission=5.0,
                            stamp_tax=0.001, slippage="sqrt"),  # 佣金/印花税/滑点
    risk=StandardRisk(max_position_pct=0.95),            # 风控中间件
    rules=StandardAshareRules(limit=0.10),               # A 股涨跌停 ±10%
)
result = engine.run()
```

### 4.6 运行内置示例

```bash
python examples/dual_ma_real.py          # 真实数据双均线
python examples/multi_factor_universe.py # 横截面动量选股
python examples/conditional_stop_loss.py # 趋势 + 止损条件单
python examples/bracket_oco.py           # 入场 + OCO 止盈止损
```

### 4.7 运行测试

```bash
python -m pytest                              # 246 passed
python -m pytest tests/test_invariants.py -v  # 只看不变式 I1–I5
python tools/validate_framework.py            # 真实数据逐项能力验证
```

---

## 5. 核心能力全景

### 5.1 因果数据总线（`data_bus.py`）

所有数据访问**必须附带引擎签发的 `CausalToken`**；引擎强制过滤 `timestamp > token.issued_at - lag` 的记录。

```python
def on_frame(self, frame, token, bus, host):
    closes = bus.get("close", "SH.600519", window=20, token=token)   # 单序列 + 窗口
    coef   = bus.get("coint_coef", window=100, lag=1, token=token)   # lag 强制排除当前帧
    rows   = bus.cross_section("momentum", universe, token=token, lag=1)  # 横截面
    top    = sorted(rows, key=lambda s: rows[s].value, reverse=True)[:3]
    closed, forming = bus.get_multi_tf("close", "SH.600519", token, bucket=5)  # 多周期
```

- **令牌不可伪造**：总线把 token 与引擎权威的虚拟时间取较小者，伪造 token 拿不到任何额外数据。
- **`lag` 原语**强制排除当前帧，防同帧泄漏；负 `lag` 视为显式请求未来，直接抛 `CausalityViolation`。
- **`forming_bar`** 的 high/low 随帧变化，默认不应用于触发信号。

### 5.2 订单与撮合（`execution/`）

| 订单类型 | 语义 |
|---|---|
| `MARKET` | 以当帧开盘价成交（因果最保守） |
| `LIMIT` | 限价挂单，未达价则 resting |
| `STOP` / `STOP_LIMIT` | 触发价达到后转市价/限价 |
| `TRAILING_STOP` | 跟踪止损，触发价随有利方向移动 |
| `IOC` / `FOK` | 立即成交或取消 / 全部成交或取消 |
| **条件单** | `Condition(trigger, level)`，引擎帧间监控 |
| **OCO / Bracket** | `OrderGroup.oco([...])`，任一腿成交自动撤销其余 |

**Bracket = 入场 + `on_fill` 返回 OCO 组**，无需专门订单类型：

```python
def on_fill(self, fill):
    if fill.side is OrderSide.BUY:
        tp = Order(SELL, fill.symbol, size=fill.size, order_type=LIMIT, price=fill.price * 1.05)
        sl = Order(SELL, fill.symbol, size=fill.size, order_type=MARKET,
                   condition=Condition(Trigger.LAST_LE, fill.price * 0.97))
        return [OrderGroup.oco([tp, sl])]   # 任一成交，另一自动撤销
```

### 5.3 账本与规则（`ledger.py` / `rules.py`）

- **多腿持仓**：金字塔加仓 / 分批止盈跟踪每笔成本与结算日，FIFO 计算已实现盈亏
- **T+1 结算**：买入腿在 `entry_frame + settlement_cycle` 前不可卖出
- **公司行动**：拆股 / 现金分红 / 退市现金结算
- **A 股规则**：涨跌停（默认 ±10%）、停牌、T+1、交易日历

### 5.4 成本与风控（`cost.py` / `guard.py`）

- **成本**：佣金（含最低佣金）、印花税（仅卖出）、过户费、滑点（fixed / proportional / volume_based / sqrt 冲击）
- **风控中间件**：盘前检查（单笔上限、仓位上限、资金充足、最小保留现金），独立于策略与撮合
- **增量风险**：逐帧最大回撤、历史模拟法 VaR、暴露度

### 5.5 策略宿主（`core/host.py`）

多唤醒机制，策略只注册它关心的类型：`on_frame / on_sub_step / on_event / on_condition / on_schedule / on_fill / on_lifecycle`。

> **`FILL` 默认开启很重要**：`on_fill` 是挂出止损/止盈等被动单的标准时机，若默认关闭，这类策略会**静默失效**。

**避免重复下单**：订单有强制延迟，本帧下单不会立刻反映到持仓。策略应先用 `host.has_pending(symbol)` 检查，
否则会在成交落地前每帧重复下单，导致仓位指数级增长。

确定性 RNG 由宿主注入（`host.rng`），策略**禁止**使用 `random` / `numpy.random`，否则破坏 I3。

### 5.6 自适应多因子系统（在 framebt 之上的完整落地）

在引擎之上实现了《即时事件驱动 + 持续自适应的多因子系统》中可由日频价量数据实现的部分：

| 层 | 模块 | 内容 |
|---|---|---|
| 因子体系 | `factors/` | 57 个 WQ101/GTJA191 风格因子 · MAD→标准化→中性化→去冗余 · IC/ICIR/Newey-West t/分组单调性 |
| 权重优化 | `portfolio/` | 复合损失 `−ICIR+λ₁·Turnover+λ₂·‖w‖₁` · 在线梯度下降 · 市场状态门控 |
| 监控降权 | `monitor/` | IC/ICIR/FSC/拥挤度 · CUSUM 变点 + 卡尔曼平滑 · 三级预警 · 参数敏感性安全阀 |
| 自主发现 | `discovery/` | 符号回归（自实现遗传规划）· 神经网络隐因子（PyTorch 可选）· 入栈流水线 |
| 生命周期 | `factors/lifecycle.py` | 观察期 → 提权 → 退休（ICIR+CUSUM 双条件）→ 康复 |
| 策略执行 | `strategies/adaptive_multi_factor.py` | 因果在线学习：每帧用 `factor[t−1]` 与 `realized(t−1→t)` 更新权重并调仓 |

**核心原则落地**：因子函数只认 `Panel`，因此**同一份因子定义**同时驱动研究（整段面板）与回测（逐帧因果切片），
不存在第二份实现。

**真实数据验证结果（严格样本内/外切分）**：57 候选因子 → 10 通过门槛 → 去冗余后 8 个；
样本外总收益 **+42.47%** vs 基准等权买入持有 +16.71%，超额 **+25.76%**（详见 `reports/ADAPTIVE.md`）。

> ⚠️ 保留意见：验证池仅 24 只且为存活至今的大盘股（存活偏差）；样本外仅一段 5 年历史；换手偏高使结果对成本敏感。
> **这构成「框架能端到端跑通并产出可解释结果」的工程证据，而非统计意义上的有效性证明。**

---

## 6. 真实数据：最有说服力的一项验证

「策略亏钱」不等于「框架有错」。为把两者分开，framebt 用**同一套代码只翻转排序方向**做对照实验：

| 排序模式 | 收益 | 说明 |
|---|---|---|
| momentum（买动量最高） | **−87.3%** | 顺势在本样本期巨亏 |
| reversal（买动量最低） | **+201.6%** | 跑赢等权买入持有 |
| random（随机排序） | −34.8% | 无信息基准（来自摩擦） |
| 等权买入持有 | +180.7% | |

不依赖框架的纯统计检验给出同样结论：60 日动量与未来 20 日收益相关系数 **−0.058**，
动量最高 20% 组未来收益 +1.25%，最低 20% 组 +3.14%。

→ 本样本期（2018–2026 A 股大盘股）60 日动量呈**反转**特征，因此顺势组合亏损、逆势组合盈利。
**框架忠实地把信号翻译成了盈亏**——它没有为了「看起来好看」而抹平这个残酷事实。
这正是可信回测该有的样子：它惩罚自我欺骗，而不是替你掩盖。

完整验证见 `reports/VALIDATION.md`：48,000 根真实日线、I1–I5 全通过、
账本外部审计偏差 `0.000000e+00`、成本可精确核验、A 股涨跌停/T+1 真实生效。

---

## 7. 与主流框架的差异对照

| 维度 | 事件驱动 | 向量化 | **framebt** |
|---|---|---|---|
| 时间推进 | 跳到下一事件时间戳 | 无显式推进 | **固定 Δt 主动推进** |
| 策略决策节奏 | 与事件节奏绑定 | 一次性 | **策略自主选择唤醒类型** |
| 防前视机制 | 隐藏未来数据 | 不适用 | **因果令牌 + 执行层延迟** |
| 数据可见性 | 框架预定义静态集合 | 全量矩阵 | **策略动态声明 + 因果过滤** |
| 空时间段 | 完全跳过 | 不适用 | **可配置跳过或静态帧** |
| 实盘代码复用 | 需适配层 | 不可用 | **回调接口可直接复用** |

> 与 backtrader / qlib / vnpy 等框架的定位不同：framebt 不追求「功能大而全」，
> 而是把**因果纪律**做成物理约束而非可选项，把**时间模型**做成可配置的底层节拍。
> 它是为「相信回测结果」这件事专门打磨的引擎。

---

## 8. 加入我们：这不是一个写完的工具，是一个正在生长的社区

framebt 最难得的品质，是它的**诚实**。

项目里有一整套「预登记 + 证伪」文化：`docs/ETF_TREND.md` 在跑出结果**之前**就写定判据，
主策略年化 9.10% 差 0.1pp 没过线，作者**按规定作废、绝不调参抢救**；
`docs/PATH_200K.md` 直言「你缺的从来不是引擎，是广度」；README 把**已知边界**逐条列出：
撮合按「每帧单 Bar + 开盘价」建模、做空未实现、验证池有存活偏差……

**这意味着加入 framebt 社区，你加入的不只是一个「声称自己很准」的框架，
而是一个「敢于告诉你哪里不准」的群体。** 我们奖励诚实，而非漂亮的曲线。

### 我们在找这样的人

- **想打磨因果引擎的人**：Phase 6 路线图上还有多时钟域、帧内事件队列、成交概率模型、异步任务确定性等开放问题。
- **想做多因子 / 自主发现的人**：符号回归、神经网络隐因子、因子生命周期都在持续演进。
- **想补全 A 股微观结构的人**：真实订单簿与排队位置模型、日内因子、做空机制。
- **想做好工程的人**：246 个测试、逐位复现、可核验成本——细节决定可信度。
- **想分享实证研究的人**：我们鼓励「预登记再跑数」的研究范式，把证伪的结果也当作贡献。

### 如何参与

1. **跑起来**：`pip install -e ".[data,dev]"` → `python examples/dual_ma_real.py`，先感受「可信回测」长什么样。
2. **读代码**：从 `engine.py`（帧生命周期）和 `data_bus.py`（因果令牌）入手，理解三层解耦。
3. **看测试**：`python -m pytest tests/test_invariants.py -v`，不变式是理解框架的钥匙。
4. **提想法**：在 GitHub 提 Issue / PR——从文档勘误、测试补全，到完整的 Phase 6 功能。
5. **做研究**：用 `tools/` 下的验证脚本跑你自己的策略，把「预登记 + 对照实验」的结果分享回来。

> **Repository**: https://github.com/LisonEvf/framebt.git
> **问题 & 讨论**: https://github.com/LisonEvf/framebt/issues
> **许可**: MIT

---

### 最后一句话

> 回测框架可以很多，但**让你敢于相信回测结果的框架**很少。
> 如果你也受够了「看起来很美、一上实盘就崩」的回测，
> 欢迎加入 framebt——我们一起把「可信」做成物理约束，而不是口头承诺。
