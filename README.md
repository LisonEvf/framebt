# framebt — 帧刷新量化回测框架

> 以 **固定逻辑步长 Δt 主动推进虚拟时间** 为骨架的确定性、因果性回测引擎。
> 策略代码写起来像在实盘中一样自由，回测结果却比传统框架更可信。

`framebt` 是设计文档《帧刷新量化回测框架 v1.0》的工程实现，**以 [opentdx](../opentdx) 数仓（Parquet + DuckDB）为数据源**。

---

## 1. 三条公理与三层解耦

**公理一：因果性优先于可见性。** 引擎不规定策略「能看什么变量」，只保证策略「不能因果地使用尚未发生的信息」。
**公理二：执行层才是防前视的主战场。** 前视最常见的来源不是「看到未来数据」，而是「看到当前数据并以当前价格成交」。因果约束施加在**执行层**（延迟管线），而非感知层。
**公理三：帧刷新是时间的底层节拍，策略的感知节奏由策略自主选择。**

由此得到三层解耦（本框架的核心贡献）：

| 解耦 | 传统事件驱动 | framebt |
|---|---|---|
| 时间推进 ↔ 事件到达 | 跳到下一事件时间戳 | 固定 Δt 主动推进，事件按真实时间戳注入 |
| 数据可见性 ↔ 因果性 | 框架预定义静态数据集 | 策略自由请求，引擎在访问层过滤未来 |
| 感知节奏 ↔ 帧节拍 | 策略被迫适配每个事件 | 多唤醒机制，策略注册它关心的类型 |

---

## 2. 架构

### 2.1 分层视图

```
┌─────────────────────────────────────────────────────┐
│  Strategy Host  策略宿主（唤醒分发 / 确定性 RNG）      │  core/host.py
├─────────────────────────────────────────────────────┤
│  Guard  风控中间件（盘前检查 / 增量风险指标）           │  guard.py
├─────────────────────────────────────────────────────┤
│  Execution  撮合（订单类型 / 延迟管线 / 条件单 / OCO）  │  execution/
├─────────────────────────────────────────────────────┤
│  Ledger  账本（多腿持仓 / T+1 结算 / 公司行动）         │  ledger.py
├─────────────────────────────────────────────────────┤
│  Rules  市场规则（涨跌停 / 停牌 / T+1 / 日历）          │  rules.py
├─────────────────────────────────────────────────────┤
│  Cost  成本（佣金 / 印花税 / 滑点 / 冲击）              │  cost.py
├─────────────────────────────────────────────────────┤
│  Data Bus  因果数据总线（令牌 / lag / 横截面 / 多周期）  │  data_bus.py
├─────────────────────────────────────────────────────┤
│  Engine Core  时钟（固定 Δt / 累加器 / 单时钟域）        │  engine.py
└─────────────────────────────────────────────────────┘
                     ▲
        DataSource  数据接入（opentdx Parquet）           data_sources/
```

### 2.2 帧生命周期（四阶段）

每一帧严格按四个阶段推进（`engine.py::Engine.run`）：

| 阶段 | 动作 |
|---|---|
| **Phase 1 市场数据注入** | 把落在 `[frame_start, frame_end)` 的 Bar 写入 Bus；区间内无数据时以前帧 BBO 生成 `is_static=True` 帧 |
| **Phase 2 唤醒分发** | `host.dispatch_frame` → 策略 `on_frame` → 返回订单/订单组，登记入在途队列 |
| **Phase 3 订单处理与撮合** | 风控+规则盘前拦截 → 条件单触发检查 → 延迟管线 → 价格-时间优先撮合 → 成本计入 → OCO 联动撤销 |
| **Phase 4 状态快照与绩效** | `ledger.mark_to_market` → 连续 MTM 权益曲线 |

**空闲区间折叠**：无数据且无在途订单时，直接跳到下一根 Bar，不生成无意义的空帧；仅当「数据耗尽但仍有在途订单」时才补静态帧让其以最后价撮合。

### 2.3 不变式（引擎硬约束，也是测试判据）

| 不变式 | 含义 | 测试 |
|---|---|---|
| **I1** | 任何数据请求返回的记录 `timestamp <= current_virtual_time` | `test_invariants.py::test_i1_*` |
| **I2** | 帧 t 生成的订单，最早成交于帧 t+delay（延迟管线） | `test_invariants.py::test_i2_*` |
| **I3** | 相同配置（数据/参数/种子/Δt 序列）逐位可复现 | `test_invariants.py::test_i3_*` |
| **I4** | 帧内多 Bar 按 `(timestamp, sequence_id)` 严格排序 | `test_invariants.py::test_i4_*` |
| **I5** | 策略无法访问/改写已过去帧与未来帧 | `test_invariants.py::test_i5_*` |

---

## 3. 安装与数据准备

核心引擎**零第三方依赖**（纯标准库），保证 I3 逐位复现与可移植性；只有读取 opentdx 数仓 Parquet 时才需要 `pandas + pyarrow`。

```bash
cd framebt
pip install -e .              # 仅核心引擎
pip install -e ".[data,dev]"  # + 数据源 + 测试
```

数据源默认指向 `../opentdx/db/data`（可用环境变量 `FRAMEBT_DATA_DIR` 或 `data_dir=` 参数覆盖）。

### 回填历史数据

opentdx 的 `db/etl.py` 以 `count=800` 单次拉取，但实测该值命中协议边界（返回 800 行却是错乱区间）；
`count<=700` 正常且 `start` 偏移分页可用。因此本仓库提供**分页回填工具**，
复用 opentdx 自有的 `db/persist.py` 写库（schema 与 merge 语义完全一致，不改动 opentdx 源码）：

```bash
python tools/backfill_history.py                  # 默认 20 只冒烟样例（非验证池，秒级）；全市场 5500+ 标的见 available_symbols()
python tools/backfill_history.py --max-bars 3000  # ≈12 年
python tools/backfill_history.py --symbols "SH 600519, SZ 000001"
```

回填后数仓含 **24 个 A 股大盘股 × 2000 根前复权日线（2018-05 ~ 2026-10）**，覆盖银行/白酒/医药/家电/
券商/地产/资源等行业，足以支撑横截面策略。

> ⚠️ **复权口径**：数仓内同一标的可能同时存在 `NONE/QFQ/HFQ` 多个文件。数据源**按行内
> `period`/`adjust` 列过滤**（默认 `DAILY + QFQ`），避免把不同复权口径的 Bar 混成一条序列。
> 用 `src.available_adjusts(symbol)` 查看某标的实际存在的口径。

---

## 4. 快速开始

### 4.1 用真实数仓数据回测

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

### 4.2 合成数据（测试友好）

```python
from framebt import run_backtest
from framebt.strategies import DualMAStrategy

bars = [{"timestamp": i, "open": o, "high": h, "low": l, "close": c, "volume": v}, ...]
res = run_backtest(bars, DualMAStrategy("S", short=5, long=20),
                   symbol="S", initial_cash=100_000.0, seed=1234)
```

### 4.3 装配完整中间件

```python
from framebt import (Engine, OpentdxParquetSource, DualMAStrategy,
                     StandardCost, StandardRisk, StandardAshareRules)

engine = Engine(
    src.load("SZ.000001"),
    DualMAStrategy("SZ.000001", 5, 20),
    initial_cash=1_000_000.0,
    settlement_cycle=1,                                  # T+1
    cost_model=StandardCost(commission=0.0003, min_commission=5.0,
                            stamp_tax=0.001, slippage="sqrt"),  # §8
    risk=StandardRisk(max_position_pct=0.95),            # §9
    rules=StandardAshareRules(limit=0.10),               # §7 A 股涨跌停
)
result = engine.run()
```

### 4.4 运行示例

```bash
python examples/dual_ma_real.py          # 真实数据双均线
python examples/multi_factor_universe.py # 横截面动量选股（§4.5）
python examples/conditional_stop_loss.py # 趋势 + 止损条件单（§5.2）
python examples/bracket_oco.py           # 入场 + OCO 止盈止损（§5.1）
```

---

## 5. 核心能力

### 5.1 因果数据总线（`data_bus.py`，§4）

所有数据访问**必须附带引擎签发的 `CausalToken`**；引擎强制过滤 `timestamp > token.issued_at - lag` 的记录。

```python
def on_frame(self, frame, token, bus, host):
    # 单序列 + 窗口
    closes = bus.get("close", "SH.600519", window=20, token=token)

    # lag 原语（§4.3）：强制排除当前帧，防同帧泄漏
    coef = bus.get("coint_coef", window=100, lag=1, token=token)

    # 横截面（§4.5）
    rows = bus.cross_section("momentum", universe, token=token, lag=1)
    top = sorted(rows, key=lambda s: rows[s].value, reverse=True)[:3]

    # 多时间框架（§4.4）：已收盘 bar vs 正在形成 bar
    closed, forming = bus.get_multi_tf("close", "SH.600519", token, bucket=5)
```

`closed_bars` 可安全用于信号；`forming_bar` 的 high/low 还会随帧变化，**默认不应用于触发信号**。

**令牌不可伪造（§4.1）**：`CausalToken` 是普通对象，策略完全可以自造一个 `issued_at` 更大的实例。
因此总线不只信任 token，而是把它与**引擎登记的权威虚拟时间**（`bus.set_now`）取较小者 ——
伪造 token 拿不到任何额外数据。I1 由总线自身保证，而不是依赖「恰好还没预加载未来数据」。
（负 `lag` 视为显式请求未来，直接抛 `CausalityViolation`。）

### 5.2 订单与撮合（`execution/`，§5）

| 订单类型 | 语义 |
|---|---|
| `MARKET` | 以当帧开盘价成交（因果最保守） |
| `LIMIT` | 限价挂单，未达价则 resting |
| `STOP` / `STOP_LIMIT` | 触发价达到后转市价/限价 |
| `TRAILING_STOP` | 跟踪止损，触发价随有利方向移动 |
| `IOC` / `FOK` | 立即成交或取消 / 全部成交或取消 |
| **条件单** | `Condition(trigger, level)`，由引擎帧间监控（§5.2） |
| **OCO / Bracket** | `OrderGroup.oco([...])`，任一腿成交自动撤销其余（§5.1） |

**延迟管线**是防前视的主战场：帧 t 生成的订单最早在帧 t+delay 成交（I2）。

**Bracket = 入场 + `on_fill` 返回 OCO 组**，无需专门订单类型：

```python
def on_fill(self, fill):
    if fill.side is OrderSide.BUY:
        tp = Order(SELL, fill.symbol, size=fill.size, order_type=LIMIT,
                   price=fill.price * 1.05)
        sl = Order(SELL, fill.symbol, size=fill.size, order_type=MARKET,
                   condition=Condition(Trigger.LAST_LE, fill.price * 0.97))
        return [OrderGroup.oco([tp, sl])]   # 任一成交，另一自动撤销
```

### 5.3 账本与规则（`ledger.py` / `rules.py`，§6 / §7）

- **多腿持仓**：金字塔加仓/分批止盈跟踪每笔成本与结算日，FIFO 计算已实现盈亏
- **T+1 结算**：买入腿在 `entry_frame + settlement_cycle` 前不可卖出
- **公司行动**：拆股 / 现金分红 / 退市现金结算
- **A 股规则**：涨跌停（默认 ±10%）、停牌、T+1、交易日历

### 5.4 成本与风控（`cost.py` / `guard.py`，§8 / §9）

- **成本**：佣金（含最低佣金）、印花税（仅卖出）、过户费、滑点（fixed / proportional / volume_based / sqrt 冲击）
- **风控中间件**：盘前检查（单笔上限、仓位上限、资金充足、最小保留现金），独立于策略与撮合。
  `StandardRisk(reserve_pending=True)` 会把**在途买单占用的资金**计入预留 —— 否则同一帧内
  多笔买单各自「资金充足」却合计透支账户。
- **增量风险**：逐帧最大回撤、历史模拟法 VaR、暴露度

### 5.5 策略宿主（`core/host.py` / `core/strategy.py`，§10）

多唤醒机制，策略只注册它关心的类型：

```python
on_frame / on_sub_step / on_event / on_condition / on_schedule / on_fill / on_lifecycle
```

默认唤醒集为 `[FRAME, FILL]`。**`FILL` 默认开启很重要**：`on_fill` 是挂出止损/止盈等
被动单的标准时机（成交后才知道真实成交价与数量），若默认关闭，这类策略会**静默失效**。
策略可覆写 `default_wakeups` 自定义，也可由引擎显式 `register(strategy, wakeups=[...])` 覆盖。

**在途订单查询**（配合 I2 延迟管线使用）：

```python
if host.has_pending(self.symbol):     # 已有未落地订单 → 不要重复下单
    return []
```

订单有强制延迟，本帧下的单不会立刻反映到持仓/现金上。策略若只看持仓做判断，
会在成交落地前**每帧重复下单**，导致仓位指数级增长。`host.pending_orders(symbol)` /
`host.has_pending(symbol)` 让「避免重复下单」可以被正确表达。

确定性 RNG 由宿主注入（`host.rng`），策略**禁止**使用 `random` / `numpy.random`，否则破坏 I3。

---

## 6. 自适应多因子系统（设计文档落地）

在 framebt 之上落地了《即时事件驱动 + 持续自适应的多因子系统》中**可由日频价量数据实现**的部分。
详细对照（含诚实的实现边界）见 **[docs/MULTIFACTOR.md](docs/MULTIFACTOR.md)**。

| 层 | 模块 | 内容 |
|---|---|---|
| 因子体系 | `factors/` | 57 个 WQ101/GTJA191 风格因子 · MAD→标准化→中性化→去冗余 · IC/ICIR/Newey-West t/分组单调性 |
| 权重优化 | `portfolio/` | 复合损失 `−ICIR+λ₁·Turnover+λ₂·‖w‖₁` · 在线梯度下降（折扣 β=0.95）· 市场状态门控 |
| 监控降权 | `monitor/` | IC/ICIR/FSC/拥挤度 · CUSUM 变点 + 卡尔曼平滑 · 三级预警 · **参数敏感性安全阀** |
| 自主发现 | `discovery/` | 符号回归（自实现遗传规划）· 神经网络隐因子（PyTorch 可选）· 正交化/VIF/影子运行入栈流水线 |
| 生命周期 | `factors/lifecycle.py` | 观察期(权重≤5%) → 提权 → 退休(ICIR+CUSUM 双条件) → 康复 |
| 策略执行 | `strategies/adaptive_multi_factor.py` | 因果在线学习：每帧用 `factor[t−1]` 与 `realized(t−1→t)` 更新权重并调仓 |

**核心原则落地**：因子函数只认 `Panel`，因此**同一份因子定义**同时驱动研究（整段面板）
与回测（`Panel.from_bus` 逐帧因果切片），不存在第二份实现。验证脚本的**因果审计**
抽样确认「预计算面板第 t 行 == 只用 ≤t 数据重算的第 t 行」，逐位一致。

### 真实数据验证结果（严格样本内/外切分）

```bash
python tools/validate_adaptive.py --train 800 --rebalance 10 --md reports/ADAPTIVE.md
```

| 项 | 值 |
|---|---|
| 数据 | 24 只 A 股 × 2000 根前复权日线（2018-05 ~ 2026-10） |
| 样本内选因子 | 57 候选 → 10 通过门槛 → 去冗余后 **8 个** |
| 因果审计 | ✓ 逐位一致（无前视） |
| **样本外总收益** | **+42.47%** |
| 基准（等权买入持有） | +16.71% |
| **超额** | **+25.76%** |
| 年化 / Sharpe / 最大回撤 | +7.15% / 0.37 / 31.53% |
| 成交 / 费用 / 年化单边换手 | 856 笔 / 12.3% 初始资金 / ~13.5× |

> ⚠️ **保留意见**：验证池仅 24 只且为存活至今的大盘股（存活偏差）；样本外仅一段 5 年历史；
> 换手偏高使结果对成本假设敏感。**这不构成统计意义上的有效性证明**，而是「框架能端到端
> 跑通并产出可解释结果」的工程证据。

---

## 7. 目录结构

```
framebt/
├── src/framebt/
│   ├── engine.py             # Engine：时间模型 + 四帧生命周期编排（§2/§3）
│   ├── data_bus.py           # DataBus / CausalToken 因果过滤 + 令牌钳制（§4）
│   ├── feed.py               # BarFeed（(timestamp, sequence_id) 全序，I4）
│   ├── data_sources/         # DataSource 抽象 + opentdx Parquet 读取（口径过滤）
│   ├── execution/            # MatchingEngine + OrderGroup/OCO（§5）
│   ├── ledger.py             # 多腿持仓 / T+1 / 公司行动（§6）
│   ├── rules.py              # A 股规则：涨跌停/停牌/日历（§7）
│   ├── cost.py               # 佣金/印花税/滑点/冲击（§8）
│   ├── guard.py              # 风控中间件 + 增量风险（§9）
│   ├── core/                 # events / market_frame / host / strategy / result / determinism
│   ├── factors/              # 多因子体系层：算子/面板/因子库/预处理/评估/生命周期
│   ├── portfolio/            # 权重优化层：在线 GD / 门控 / 因子合成与目标持仓
│   ├── monitor/              # 监控层：CUSUM+卡尔曼 / 健康看板 / 三级预警与降权
│   ├── discovery/            # 自主发现：符号回归 / 神经网络 / 入栈流水线
│   └── strategies/           # 自适应多因子 / 双均线 / 横截面动量 / 条件单 / Bracket
├── tools/
│   ├── backfill_history.py   # 分页回填 opentdx 数仓历史（复用其 persist 写库）
│   ├── validate_framework.py # 框架能力验证（输出 reports/VALIDATION.md）
│   └── validate_adaptive.py  # 多因子系统验证（输出 reports/ADAPTIVE.md）
├── examples/                 # 4 个可运行示例（真实数仓数据）
├── docs/MULTIFACTOR.md       # 多因子系统落地说明（含诚实的实现边界）
├── reports/                  # 验证报告（生成物）
└── tests/                    # 246 个测试：不变式 + 各模块 + 策略/因子回归
```

---

## 8. 测试与验证

```bash
python -m pytest                              # 246 passed
python -m pytest tests/test_invariants.py -v  # 只看不变式

python tools/validate_framework.py            # 框架能力验证（真实数据，逐项检验）
python tools/validate_framework.py --md reports/VALIDATION.md
```

### 8.1 单元测试

| 文件 | 覆盖 |
|---|---|
| `test_invariants.py` | I1–I5 全部不变式 |
| `test_data_bus.py` | 因果令牌 / 令牌钳制 / window / lag / 横截面 / 多时间框架 |
| `test_execution.py` | 延迟管线 / 各订单类型 / 条件单 / OCO / 撤单 / 组单 |
| `test_ledger_rules_cost.py` | 多腿 / T+1 / 公司行动 / 涨跌停 / 成本 / 风控 |
| `test_data_source.py` | opentdx Parquet 读取 + **复权口径过滤** |
| `test_strategies.py` | 策略回归：满仓、预热、OCO 不重复平仓、**`on_fill` 唤醒**、**重复下单防护** |
| `test_factors.py` | 算子（安全除法/滚动窗口）、Panel 因果性、57 因子有效性、预处理、IC/ICIR/单调性 |
| `test_portfolio.py` | 在线权重（学习/确定性/L1 收缩/换手惩罚）、门控、合成与缓冲带 |
| `test_monitor`（见 `test_changepoint.py`） | CUSUM / 卡尔曼 / 衰退概率 / 参数敏感性 |
| `test_adaptive_system.py` | **缺陷回归**：风控权益口径、预留自身、涨跌停生效、T+0 贯通 + 策略端到端 |
| `test_symbolic.py` | 符号回归：Spearman、表达式复算、受保护运算、确定性、发现能力 |
| `test_discovery_pipeline.py` | 正交化 / VIF / 增量 ICIR / 影子运行 / 五级入栈 + 神经网络隐因子 |

### 8.2 真实数据能力验证（`reports/VALIDATION.md`）

在 **24 标的 × 2000 日线（48,000 bars）** 上逐项检验：

| 检验项 | 结果 |
|---|---|
| 数据完整性（时间戳唯一/有序/OHLC 自洽） | ✓ 无问题 |
| **I1** 因果性：每帧可见 bar 数 = 已过帧数+1，单调增长 | ✓ |
| **I1** 前视抵抗：负 lag + 伪造未来 token 共 2000 次尝试，**0 次逃逸** | ✓ |
| **I2** 延迟管线：0 笔违反；`delay_frames=0` 构造即拒绝 | ✓ |
| **I3** 确定性：权益曲线与成交明细逐位一致 | ✓ |
| **A 股规则**：真实涨停日 236 / 跌停日 102；涨停买入、跌停卖出、T+1 同日卖出均被拒，次日放行 | ✓ |
| **账本外部审计**：由 FillEvent 明细独立重算权益，偏差 `0.000000e+00` | ✓ |
| **成本可核验**：固定股数下，含/不含成本的权益差 **恰等于** 累计费用 | ✓ |
| 吞吐量 | 单标的 91k bars/s；24 标的横截面 7.2k bars/s |

### 8.3 信号传导对照实验（最有说服力的一项）

「策略亏钱」不等于「框架有错」。为把两者分开，用**同一套代码只翻转排序方向**做对照：

| 排序模式 | 收益 | 说明 |
|---|---|---|
| momentum（买动量最高） | **−87.3%** | |
| reversal（买动量最低） | **+201.6%** | 跑赢等权买入持有 |
| random（随机排序） | −34.8% | 无信息基准 |
| 等权买入持有 | +180.7% | |

**不依赖框架**的纯统计检验给出同样结论：60 日动量与未来 20 日收益相关系数 **−0.058**，
动量最高 20% 组的未来收益 **+1.25%**，最低 20% 组 **+3.14%**。

→ 本样本期（2018–2026 A 股大盘股）60 日动量呈**反转**特征，因此顺势组合亏损、逆势组合盈利。
框架忠实地把信号翻译成了盈亏；随机组合的小幅亏损来自日频换手 + 次帧开盘成交的逆向选择，属预期摩擦。

---

## 9. 实现状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 0** 核心骨架 | Engine/固定 Δt、MarketFrame、因果令牌、市价+限价+延迟管线、账本、`on_frame` | ✅ 完成 |
| **Phase 1** 因果数据总线 | lag 原语、多时间框架（`get_multi_tf` + is_forming）、横截面、声明层提示 | ✅ 完成 |
| **Phase 2** 订单语义 | 条件单（Stop/StopLimit/TrailingStop）、OCO/Bracket、组单原子性、撤单、IOC/FOK | ✅ 完成 |
| **Phase 3** 账本与规则 | 多腿持仓、T+1 结算、公司行动、涨跌停/停牌/日历 | ✅ 完成 |
| **Phase 4** 成本与风控 | 佣金/印花税/滑点/冲击、风控中间件、增量风险 | ✅ 完成 |
| **Phase 5** 策略宿主 | 唤醒类型定义 + `on_frame`/`on_fill` 分发、确定性 RNG | 🟡 部分（事件注入层、异步推理未实现） |
| **Phase 6** 高级能力 | 成交概率模型、多时钟域、期权生命周期、帧内事件队列、混合数据总线 | ⬜ 路线图 |

> Phase 0–4 已在真实数据上逐项验证（见 §8.2 / [reports/VALIDATION.md](reports/VALIDATION.md)）：
> 48,000 根真实日线、24 标的、I1–I5 全通过、账本外部审计零偏差、成本可精确核验。
> §6 的自适应多因子系统另有一套独立的样本外验证（见 §6 / [reports/ADAPTIVE.md](reports/ADAPTIVE.md)）。

**已知边界**（诚实声明）：

- 撮合按「每帧单 Bar + 开盘价」建模，暂无真实订单簿与排队位置模型；`OrderBook` 为占位。
  日频换手策略因此会承受「次帧开盘成交」的逆向选择（对照实验中随机组合 −34.8% 即此项）。
- `prefetch_hints` 为**性能提示**，当前不影响正确性（未声明请求也会实时计算），符合「声明不构成白名单」。
- 做空仅支持账本层面的负持仓，**借券可用性/借券费/强平**未实现（Phase 6）。
- 多时间框架按「每 N 根基础 Bar」聚合，非按挂钟周期（日线数据下等价于 N 日线）。
- 帧内子步骤（`on_sub_step`）接口已定义，引擎尚未在帧内调用。
- 验证池为 **24 只大盘股**（存活偏差：均为 2018 年前上市且至今存续），
  结论不可外推到全市场；`tools/backfill_history.py --symbols` 可换池重跑。
- 成交假设为「订单全额成交」，无部分成交/成交量上限约束（Phase 6 的成交概率模型）。
- 多因子系统**未实现**的部分（DolphinDB 流批一体、L2 日内因子、财务 PIT 双时间戳、
  资讯事件 GMM 分类、实盘并行验证）在 [docs/MULTIFACTOR.md](docs/MULTIFACTOR.md) §6
  按「数据缺失 / 依赖缺失 / 明确未做」三类逐条列出。

---

## 10. 关键设计决策（对应设计文档 §15 开放问题）

| 开放问题 | 本实现的取舍 |
|---|---|
| **1. Decimal vs Float** | 全部用 `float`（双精度）。理由：I3 要求的「逐位复现」在**同机同解释器**下 float 完全满足（IEEE-754 确定性），而全 `Decimal` 会带来数量级性能损失。边界：**账本现金/数量用 float 累加**，若需分厘不差的对账，可在 `Ledger` 之上外挂 Decimal 影子账（不动引擎）。 |
| **2. 多时钟域因果令牌** | 未实现（Phase 6）。当前令牌只带 `issued_at / frame_id / sub_step`，**不带市场维度**；单时钟域下无歧义。跨市场时建议在令牌上加 `market` 字段而非新建令牌类型（保持「一个虚拟时间」的单一真相）。 |
| **3. 帧内事件队列与子步骤统一** | 未实现（Phase 6）。当前 `on_sub_step` 与 `EventQueue` 是两个独立接口。倾向**合并为单一「帧内时间轴」**：`sub_step` 就是帧内时间轴上的一个刻度，事件按 `(timestamp, sequence_id)` 插入同一时间轴。 |
| **4. 异步任务的确定性** | 未实现（Phase 5+）。设计上应让**完成顺序由虚拟时间决定而非真实并发**：`submit_async(fn, delay_ns)` 只登记「结果在帧 t+k 可用」，推理可在后台真并发跑，但**结果注入按虚拟时间排序**，从而与并发度无关。 |
| **5. 期权链内存管理** | 未实现（Phase 6）。建议按「标的 × 到期月」分片惰性加载 + LRU 淘汰，策略声明的关注范围作为**预取提示**（与 §4.2 同构，非白名单）。 |
| **6. 横截面缺失数据** | **引擎决定**：停牌/数据不足的标的返回 `is_valid=False` 且 `value=NaN`，**不填充**。理由：填充策略（前值/均值/剔除）会显著改变因子含义，应由**策略显式处理**（`is_valid` 就是给策略的开关）。引擎只保证「不静默伪造数据」。 |

---

## 11. 与现有框架的差异

| 维度 | 事件驱动 | 向量化 | **framebt** |
|---|---|---|---|
| 时间推进 | 跳到下一事件时间戳 | 无显式推进 | **固定 Δt 主动推进** |
| 策略决策节奏 | 与事件节奏绑定 | 一次性 | **策略自主选择唤醒类型** |
| 防前视机制 | 隐藏未来数据 | 不适用 | **因果令牌 + 执行层延迟** |
| 数据可见性 | 框架预定义静态集合 | 全量矩阵 | **策略动态声明 + 因果过滤** |
| 空时间段 | 完全跳过 | 不适用 | **可配置跳过或静态帧** |
| 处理预算模拟 | 无 | 无 | 预留（Phase 6） |
| 实盘代码复用 | 需适配层 | 不可用 | **回调接口可直接复用** |

---

## 12. 许可

MIT
