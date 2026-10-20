# framebt — 帧刷新量化回测框架

> 以 **固定逻辑步长 Δt 主动推进虚拟时间** 为骨架的确定性、因果性回测引擎。
> 策略代码写起来像在实盘中一样自由，回测结果却比传统框架更可信。

`framebt` 是设计文档《帧刷新量化回测框架 v1.0》的工程实现，**以 [opentdx](../opentdx) 数仓（Parquet + DuckDB）为数据源**。

> 文档地图：[`docs/INDEX.md`](docs/INDEX.md)（设计/预登记、验证报告、历史归档分开）；
> 验证报告索引：[`reports/INDEX.md`](reports/INDEX.md)。

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
| **I5** | 策略无法通过公开 API 改写已过去帧/篡改权威时间（私有属性不设防） | `test_invariants.py::test_i5_*` |

---

## 3. 安装与数据准备

核心引擎（Engine / DataBus / Ledger / Rules / Cost / Execution / Feed）**零第三方依赖**（纯标准库），保证 I3 逐位复现与可移植性；`import framebt` 本身也不触发任何第三方导入（报告 API 通过模块级 `__getattr__` 惰性加载）。读取 opentdx 数仓 Parquet 需要 `pandas + pyarrow`，生成 HTML 图表报告需要 `matplotlib`。

```bash
cd framebt
pip install -e .                 # 核心（零第三方依赖，import framebt 即可用）
pip install -e ".[data,dev]"     # + 数据源（pandas/pyarrow）+ 测试
pip install -e ".[report]"       # + 图表报告（matplotlib）
pip install -e ".[all]"          # 数据源 + 报告 + 测试
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

当前数仓为**全市场 5575 标的、DAILY/QFQ**（2026-10-19 报告口径：15,577,265 根 Bar）。
`backfill_history.py` 本身默认只回填 20 只冒烟样例；全市场请用 `--symbols` 或
`available_symbols()` 自行指定。早期版本的「24 只大盘股 × 2000 根」样本已被这次全市场
迁移替代，相关旧结论（含 +42.47% 样本外收益）已作废，见 §6。

> ⚠️ **复权口径**：数仓内同一标的可能同时存在 `NONE/QFQ/HFQ` 多个文件。数据源**按行内
> `period`/`adjust` 列过滤**（默认 `DAILY + QFQ`），避免把不同复权口径的 Bar 混成一条序列。
> 用 `src.available_adjusts(symbol)` 查看某标的实际存在的口径。
>
> ⚠️ **数据体检仍然必需**：数据源会丢弃 `open/high/low/close <= 0` 的 Bar，以及单日
> `|close/prev_close-1| > 30%` 的 Bar；但 30% 仍高于 A 股 20% 的板块上限，而且这种过滤会
> **静默丢掉大量早期历史**（如 `SH.601088` 原始约 2000 根，过滤后只剩 345 根）。
> 长历史横截面回测请先按 `reports/CAPABILITY.md` §3 的方法逐标的截断/清洗。

---

## 4. 快速开始

### 4.1 用真实数仓数据回测

```python
from framebt import Engine, Performance
from framebt.data_sources import OpentdxParquetSource
from framebt.strategies import DualMAStrategy

src  = OpentdxParquetSource()                # 默认读 opentdx/db/data，DAILY + QFQ
feed = src.load("SH.600519")                 # 约 2000 根前复权日线
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
python examples/demo_quickstart.py       # 无外部数据也可跑（合成兜底）+ HTML 报告
python examples/multi_factor_universe.py # 横截面动量选股（默认 200 只，--max-symbols 可调）
python examples/conditional_stop_loss.py # 趋势 + 止损条件单（§5.2）
python examples/bracket_oco.py           # 入场 + OCO 止盈止损（§5.1）
python examples/rotation_momentum.py     # 轮动 + 选股器/预算策略
python examples/rotation_trend_ma.py     # 轮动 + 趋势过滤
```

> `multi_factor_universe.py` 默认截取 200 只：全市场 5575 只的横截面会把这个示例跑成
> 小时级；要复现全市场规模请显式 `--max-symbols 5575`。

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
伪造 token 拿不到任何额外数据。引擎启动后 `bus.record / reset / set_now` 会启用**写保护**
（key 只在引擎实例内），策略调用直接抛 `CausalityViolation`；因此「改写过去帧」也被挡住
（I5 的公开 API 层）。负 `lag` 视为显式请求未来，直接抛 `CausalityViolation`。
边界：Python 无法阻止策略直接访问私有属性（如 `bus._series`），I5 保护的是公开写 API。

### 5.2 订单与撮合（`execution/`，§5）

| 订单类型 | 语义（成交价口径） |
|---|---|
| `MARKET` | 当帧开盘价；数量受当帧成交量约束 |
| `LIMIT` | 开盘更优则按开盘价，否则按**限价**成交；未触及则 resting；成交量按价格-时间优先 FIFO 分摊 |
| `STOP` | 帧内 `high/low` 穿越触发价才触发；开盘已越过则按开盘价（跳空），否则按触发价 |
| `STOP_LIMIT` | 同上触发；`Order` 目前只有一个 `price` 字段，语义为「限价 = 触发价」 |
| `TRAILING_STOP` | 用帧内 `high/low` 更新极值；反向穿越触发价成交（跳空按开盘，否则触发价） |
| `IOC` | 按当帧可成交量成交，余量立即取消 |
| `FOK` | 当帧可成交量不足则**整单取消** |
| **条件单** | `Condition(trigger, level)` 由引擎帧间监控；触发帧**下一帧开盘**成交（避免收盘触发/当帧开盘成交） |
| **OCO / Bracket** | `OrderGroup.oco([...])`，任一腿成交自动撤销其余（§5.1） |

> 仍未建模：逐笔盘口深度、跨价位成交量分配、STOP/TRAILING 的逐笔路径。
> `LIMIT` 不再按当日 low/high 成交（旧实现会「事后」拿到最低/最高价，偏乐观）。

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
- **公司行动**：拆股 / 现金分红 / **退市现金结算**（按 `ratio` 的每股对价计入现金）
- **A 股规则**：涨跌停（默认 ±10%）、停牌、T+1、交易日历
  - `trading_days` 传逻辑时间戳集合；引擎同时匹配 `frame.end_ns` 与 `end_ns-1`
  - **涨跌停是单一 `limit` 参数**，不自动识别创业板/科创板（20%）或北交所（30%）；
    全市场回测请按标的板块自行传入（或子类化 `StandardAshareRules`）

### 5.4 成本与风控（`cost.py` / `guard.py`，§8 / §9）

- **成本**：佣金（含最低佣金）、印花税（仅卖出）、过户费、滑点（fixed / proportional / volume_based / sqrt 冲击）
- **风控中间件**：盘前检查（单笔上限、仓位上限、资金充足、最小保留现金），独立于策略与撮合。
  `StandardRisk(reserve_pending=True)` 会把**在途买单占用的资金**计入预留 —— 否则同一帧内
  多笔买单各自「资金充足」却合计透支账户。
- **增量风险**：逐帧最大回撤、历史模拟法 VaR、暴露度

### 5.5 策略宿主（`core/host.py` / `core/strategy.py`，§10）

引擎**实际调度**的唤醒类型只有三类：

```python
on_frame / on_sub_step / on_fill
```

`on_event / on_condition / on_schedule / on_lifecycle` 是**保留接口**：已定义、可覆写，
但引擎当前没有事件队列/调度器/lifecycle 分发点，注册了也不会被调用。`StrategyHost.register`
会对这些类型发 `UserWarning`，避免「静默失效」。完整路线图见 `docs/PHASE6.md`。

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

多标的轮动尤其要**全局等待在途单**，否则上一帧未成交的买单会与本帧新买单叠加：

```python
if host.pending_orders():     # 有在途单 → 本帧不调仓，等它落地
    return []
```

> ⚠️ **默认 `risk=NoOpRisk()` 不拦超买**：如果策略绕过上面的等待、一帧内提交多笔各自
> 「资金充足」的买单，账本现金可以变负（等价无息杠杆，权益甚至可能为负）。实盘/严肃
> 回测请显式传 `StandardRisk(...)`；`reserve_pending=True`（默认）会把在途买单计入预留。

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

**当前报告（2026-10-17，500 只代表样本）**：

| 项 | 值 |
|---|---|
| 数据 | 500 只代表样本（全市场 5575，DAILY/QFQ） |
| 样本内选因子 | 57 候选 → **12 通过门槛** → 去冗余后 **10 个** |
| 因果审计 | ✓ 逐位一致（无前视） |
| **样本外总收益** | **−78.55%** |
| 基准（等权买入持有） | +845.57% |
| **超额** | **−924.12%** |
| 年化 / Sharpe / 最大回撤 | −5.90% / −0.10 / 97.39% |
| 成交 / 费用 / 年化单边换手 | 9740 笔 / 267.53% 初始资金 / 27.9× |

> ⚠️ **旧结果已作废**：2026-10 之前基于「24 只大盘股 × 2000 根」的样本外
> **+42.47% / 超额 +25.76% / Sharpe 0.37** 来自已被全市场迁移替代的数据池；当前代码与
> 数据无法复现该结果，**不得再引用**。当前 −78.55% 是策略层失败（换手 27.9×、费用
> 267.53% 初始资金），不是「框架正确性」的反证；它说明这套自适应系统在更大池子上
> **不成立**。验证池为存活标的（存活偏差），结论不可外推。
>
> **能说的**：因子定义/面板/在线学习/健康监控能端到端跑通，因果审计逐位通过；
> **不能说的**：这套多因子系统有正 alpha。

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
│   ├── report.py             # 图表化回测报告（惰性导入；需 matplotlib）
│   ├── live.py               # LiveFeed / LiveConnector / Transport（实盘复用地基）
│   ├── selection/            # 选股器 + 资金/凯利预算策略
│   ├── core/                 # events / market_frame / host / strategy / result / determinism
│   ├── factors/              # 多因子体系层：算子/面板/因子库/预处理/评估/生命周期
│   ├── portfolio/            # 权重优化层：在线 GD / 门控 / 因子合成与目标持仓
│   ├── monitor/              # 监控层：CUSUM+卡尔曼 / 健康看板 / 三级预警与降权
│   ├── discovery/            # 自主发现：符号回归 / 神经网络 / 入栈流水线
│   └── strategies/           # 自适应多因子 / 双均线 / 横截面动量 / 轮动 / ETF 趋势 / 条件单 / Bracket
├── tools/
│   ├── README.md             # 工具索引：验证/门禁、数据回填、研究脚本与外部依赖
│   ├── backfill_history.py   # 分页回填 opentdx 数仓历史（复用其 persist 写库）
│   ├── validate_framework.py # 框架能力验证（输出 reports/VALIDATION.md）
│   ├── validate_capability.py# 端到端验证：低波动组合逐笔对账（reports/CAPABILITY.md）
│   ├── validate_signal_transmission.py # 修正版信号对照（reports/SIGNAL_TRANSMISSION.md）
│   ├── gate_runner.py        # 「真实可用」门禁端到端（reports/USABILITY.md）
│   ├── mutate_gate.py        # 变异验证：15 个注入 bug（reports/MUTATION.md）
│   ├── replicate_external.py # 外部策略复刻：Faber/GEM/PP（reports/EXTERNAL_REPLICATION.md）
│   └── validate_adaptive.py  # 多因子系统验证（输出 reports/ADAPTIVE.md）
├── examples/                 # 7 个可运行示例（demo_quickstart 无外部数据时走合成数据）
├── docs/INDEX.md             # 文档地图：设计/预登记、验证报告、历史归档
├── docs/MULTIFACTOR.md       # 多因子系统落地说明（含诚实的实现边界）
├── docs/PHASE6.md            # 帧内执行 / 实盘 / 做空 / 双时钟域的设计与边界
├── docs/archive/             # 历史快照：审计、修复记录（不参与当前声明）
├── reports/INDEX.md          # 验证报告索引（每份报告回答什么问题）
├── reports/                  # 验证报告（Markdown；HTML/CSV 生成物不入库）
└── tests/                    # 429 个测试：不变式 + 各模块 + 策略/因子回归
```

---

## 8. 测试与验证

```bash
python -m pytest                              # 422 passed, 7 xfailed（共 429）
python -m pytest tests/test_invariants.py -v  # 只看不变式
python -m pytest tests/test_capability_gate.py # 「真实可用」门禁（秒级）

python tools/validate_framework.py            # 框架能力验证（真实数据，逐项检验）
python tools/validate_framework.py --md reports/VALIDATION.md
python tools/validate_capability.py --md reports/CAPABILITY.md   # 端到端逐笔对账（约 80s）
python tools/validate_signal_transmission.py  # 修正版信号对照（默认 100 只，秒级）
```

> Windows 中文控制台：验证脚本已内置 `sys.stdout.reconfigure(encoding="utf-8")`，
> 不再因 GBK 无法编码「− / ✓」而中断（旧版本需先设 `PYTHONIOENCODING=utf-8`）。

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
| `test_execution_backend.py` / `test_execution_book.py` | 后端契约、订单簿、FIFO 成交量分配、部分成交 |
| `test_live.py` | LiveFeed / LiveConnector / SimulatedTransport 与回测后端一致性 |
| `test_short_selling.py` | `SELL_SHORT`、可借量、保证金、T+1 与规则交互 |
| `test_sub_step.py` / `test_dual_clock.py` | `on_sub_step` 调度、`sub_steps` 双时钟域 |
| `test_report.py` / `test_rotation.py` | HTML 报告、轮动策略 + 选股器/预算策略 |
| `test_capability_gate.py` | 「真实可用」门禁：手算金标准、独立 oracle 随机重放、因果攻击、元形不变、撮合不变量、干净环境 |

### 8.2 真实数据能力验证（`reports/VALIDATION.md`，2026-10-19）

在 **5575 标的 / 15,577,265 根 DAILY/QFQ Bar** 上逐项检验：

| 检验项 | 结果 |
|---|---|
| 数据完整性（时间戳唯一/有序/OHLC 自洽） | ✓ 无问题 |
| **I1** 因果性：每帧可见 bar 数 = 已过帧数+1，单调增长 | ✓（1,406 帧样本） |
| **I1** 前视抵抗：负 lag + 伪造未来 token 共 **1,406** 次尝试，**0 次逃逸** | ✓ |
| **I2** 延迟管线：50 笔成交 / 0 笔违反；`delay_frames=0` 构造即拒绝 | ✓ |
| **I3** 确定性：权益曲线与成交明细逐位一致 | ✓ |
| **A 股规则**：真实涨停日 **374,858** / 跌停日 **239,168**；涨停买入、跌停卖出、T+1 同日卖出均被拒，次日放行 | ✓ |
| **账本外部审计**：由 FillEvent 明细独立重算权益，偏差 `0.000000e+00` | ✓ |
| **成本可核验**：固定股数下，含/不含成本的权益差 **恰等于** 累计费用 | ✓ |
| 吞吐量（报告口径） | 单标的 **36,125 bars/s**；横截面 **868 bars/s** |

> 吞吐量随机器/数据规模变化：本机 600519（2002 根）复测约 **8k bars/s**。
> 表格里的 36k/868 是 2026-10-19 全量报告的测量值，不应当作跨机器承诺。
>
> ⚠️ `reports/VALIDATION.md` 的 **6b–6d 节**由旧版脚本生成，存在策略重复下单/默认无风控
> 导致的负现金问题；6d 的「random 亏损」结论与表中 +4,818% 自相矛盾。6d 已由
> §8.3 的新脚本取代；6b–6c 的绝对收益数字只作历史记录，修复后的重跑尚未完成。

### 8.3 信号传导对照实验（修正版）

「策略亏钱」不等于「框架有错」。为把两者分开，用**同一套代码只翻转排序方向**做对照。
但旧版脚本有一个致命缺陷：`RankStrategy` 每帧按当前持仓下单、不查 `host.has_pending`，
在 I2 延迟下会重复提交买单；配合默认 `NoOpRisk` 使现金为负（隐含无息杠杆），
于是旧报告出现了 `−102%` / `+235,360%` 这种不可解释的数字。

修正版（[`tools/validate_signal_transmission.py`](tools/validate_signal_transmission.py)）：

- 买卖都检查在途单；有在途单时本帧不调仓；
- 由成交明细独立重算「期末现金 / 期间最低现金」，负现金直接判定该组无效；
- 结果写入 [`reports/SIGNAL_TRANSMISSION.md`](reports/SIGNAL_TRANSMISSION.md)。

**100 只降采样（2026-10-20 生成，`--max-symbols 100`）**：

| 排序模式 | 总收益 | 最大回撤 | 交易数 | 期末权益 | 最低现金 |
|---|---|---|---|---|---|
| momentum（买动量最高） | −72.03% | 81.37% | 627 | 279,662 | **1** |
| reversal（买动量最低） | +305.39% | 37.38% | 1185 | 4,053,858 | **0** |
| random（随机排序） | +17.65% | 37.96% | 3326 | 1,176,488 | **12** |
| 等权买入持有 | +2.37% | — | — | — | — |

三组均无负现金。**不依赖框架**的纯统计（2065 个样本）：60 日动量与未来 20 日收益
相关系数 **−0.065**，动量最高 20% 组未来收益 **+1.33%**，最低 20% 组 **+2.06%**。

→ 方向与纯统计一致（本样本期 60 日动量偏反转）。但这只是**机制合理性检查**：
100 只降采样、存活偏差、单一样本区间；绝对收益不得外推，也不构成策略能力证明。

### 8.4 端到端能力验证：低波动组合的逐笔对账（`reports/CAPABILITY.md`）

上面几项是「逐项」检验。这一项是**端到端**检验：拿一个**具体、非玩具、有经济含义**的策略
（60 日波动率最低的 10 只、月度调仓、20 万本金、100 股整手、T+1、±10% 涨跌停、含全部成本），
把**预期值先写进 `docs/PREREG_CAPABILITY.md`**，再跑，再对账。

```bash
python -u tools/validate_capability.py --md reports/CAPABILITY.md   # 约 80s
```

关键不在于策略赚不赚钱，而在于：**引擎的每一分钱能不能被另一套独立实现复算出来**。

| 检验 | 预期 | 实测 |
|---|---|---|
| 账本外部审计（FillEvent → 现金+权益） | 偏差 < 1e-9 | **0.000e+00** |
| 成本逐笔重算 vs 手算公式 | < 1e-9 元 | **7.1e-15 元** |
| 确定性 I3（两次运行） | 逐位相同 | **逐位相同** |
| **无成本引擎 vs 独立克隆**（纯 numpy 逐日循环，独立实现整手/延迟/T+1/涨跌停） | 终值 < 1e-9、0 笔不一致 | **0.000e+00、1122/1122 笔完全一致**（2026-10-20 重跑 1136/1136，多一个交易日） |
| 摩擦拖累 = 单边换手 × 单边费率 | 1.03% ± 0.35pp | **+1.04%** |
| 因果防御（伪造 token / 负 lag） | 逃逸 0 | **0 次逃逸** |

两个预登记预测**没通过**，都如实留在报告里（换手实际 9.66× 而非预测的 2–6×；
`LowVol − HighVol` 价差 18.15% 而非五分位口径的 6.46%）—— 归因证明差异来自
「取极端 10 只 ≠ 取 1/5 截面」，同一份无摩擦参考也给出 19.96%，**不是引擎的问题**。
（2026-10-20 重跑：9.81× / 18.81% / 无摩擦 20.66%——数据前进一个交易日带来小幅漂移，
结论不变。）

> ⚠️ **顺带的硬发现**：这份数仓的 `QFQ` 是「加法式」前复权，高分红标的的早期价位会被压到
> **负值**（`SH.601088` 原始 `min(close) = −8.56`、`SZ.000338` 起始 close = 0.02），越过零点后
> 跳回正价，制造 **+1650% / +900% / inf%** 的假涨跌幅。数据源现会丢弃 `≤0` 的 Bar 和单日
> `|收益| > 30%` 的 Bar，但 30% 仍高于 A 股 20% 的板块上限，且会**静默丢掉大量历史**
> （`SH.601088` 约 2000 根 → 345 根）。**不清洗直接回测会得到「高波动组年化 +37%」的假象**
> （见报告 §12）。任何用这份数仓做长历史回测的人必须先做数据体检。

### 8.5 「真实可用」门禁：独立 oracle + 变异验证（`reports/USABILITY.md`）

8.4 证明的是「一个策略 + 一段数据」可被独立复算。这一节证明的是**换一种用法仍然对，
而且测试本身有牙**。设计写在 `docs/GATE_USABILITY.md`（含判据与已登记边界，跑之前冻结）。

| 证据 | 做法 | 结果 |
|---|---|---|
| **G1 金标准** | 买入持有/往返/费用矩阵/退市/拆股/分红/限价/止损/跟踪止损/绩效，全部手算到分 | 逐项一致 |
| **G2 独立 oracle** | 3 标的 × 50 帧 × 6 组随机订单（MARKET/LIMIT/IOC/FOK）；oracle 不 import `ledger/cost/execution` | 逐帧权益、逐笔成本、期末现金/持仓偏差 **0.000e+00** |
| **G3 因果攻击** | 伪造 token、负 lag、策略写历史；**截断到前 80 帧**、**篡改未来 5 倍** | 逃逸 0；前 80 帧逐位相同 |
| **G4 元形不变** | 输入乱序、标的改名、拆单、价格×10、加无关标的、跨进程 | 逐位/按比例相同 |
| **G5 撮合不变量** | 成交价在 Bar 区间、单笔量约束、I2 延迟、T+1、拒单、部分成交 FIFO | 核心成立；多笔市价/STOP 的量约束缺口见 §5 B3/B4 |
| **G6 工程可用** | `python -S` 纯标准库 import、非法配置报错、空/单 Bar、GBK 控制台、吞吐 | 全过；~6k bars/s |
| **G7 真实数据** | 5 只真实标的 × 250 帧多标的 + 单标的 400 根，独立重放 | 现金/权益偏差 **0.000e+00** |
| **G8 变异有牙** | 向源码**临时副本**注入 15 个已知 bug（成交价、最低佣金、印花税、T+1、I2、成交量、退市、MTM…） | **15/15 全部被门禁杀死** |

```powershell
python -m pytest -q tests/test_capability_gate.py   # 门禁本体（秒级）
python -u tools/gate_runner.py --md reports/USABILITY.md   # 真实数据 + 工程项
python -u tools/mutate_gate.py --md reports/MUTATION.md    # 变异杀死率
```

> **边界不计通过**：`volume=0` 的无限成交、NaN 静默污染、同帧多笔市价单/STOP 不共享成交量、
> `StandardRisk` 不预留手续费、同帧大额预算单互相预留等，用 `xfail(strict=True)` 钉在门禁里，
> 并在报告 §5 单列；修好后会 XPASS 提醒更新判据。**没有「全部通过」这句话。**

### 8.6 外部策略复刻：Faber / GEM / 永久组合（`reports/EXTERNAL_REPLICATION.md`）

上面两节是「项目内部口径」。这一节把**有公开记录的规则**搬进来，做三层对账：

| 层 | 内容 | 结果 |
|---|---|---|
| **L1** | framebt vs 同一数据的独立逐日实现（自维护现金/持仓/费用） | 6 个策略 **逐笔成交、逐笔成本、逐帧权益全部 <1e-9 元**（最大 4.7e-10）；换 Yahoo 数据后仍 6/6 PASS |
| **L2** | 本地 QFQ vs Yahoo 调整后收盘（独立数据商） | 日收益相关 **0.997**、CAGR 差 **0.42pp** 通过；但 510300 最大回撤 **52.97% vs 45.45%（差 7.52pp）** → **L2 未过** |
| **L3** | 与 Faber / Antonacci / Browne 公开结果对照 | 风险方向 **4/4**；但「降回撤幅度」未复现：Faber/GEM 只比 BH 少 **1.61pp**（公开约 36–37pp） |

**结论**：引擎对「别人定的规则」执行得**逐笔准确**；但公开策略的**收益/回撤量级**不能直接移植到
A 股 2015-2026——2015 年 6–8 月的闪崩太快，月度趋势/动量信号来不及退出。L2 的差异来自
本地 QFQ 的**加法式前复权**（§8.4 已记录），不是引擎算术。详见
[reports/EXTERNAL_REPLICATION.md](reports/EXTERNAL_REPLICATION.md) 与预登记
[docs/REPLICATION.md](docs/REPLICATION.md)。

```powershell
python -u tools/replicate_external.py --md reports/EXTERNAL_REPLICATION.md
```

> 另发现一个 API 陷阱：引擎 token 的 `issued_at = frame_end = bar.timestamp + Δt`，
> 所以 `bus.get(..., lag=1)` 仍会包含当日 Bar；要「上月末信号」必须用 `lag=2`。
> 这不是数据问题，是帧边界上的 off-by-one，已在复刻工具中显式注释。

---

## 9. 实现状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 0** 核心骨架 | Engine/固定 Δt、MarketFrame、因果令牌、市价+限价+延迟管线、账本、`on_frame` | ✅ 完成 |
| **Phase 1** 因果数据总线 | lag 原语、多时间框架（`get_multi_tf` + is_forming）、横截面、声明层提示 | ✅ 完成 |
| **Phase 2** 订单语义 | 条件单（Stop/StopLimit/TrailingStop）、OCO/Bracket、组单原子性、撤单、IOC/FOK | ✅ 完成 |
| **Phase 3** 账本与规则 | 多腿持仓、T+1 结算、公司行动、涨跌停/停牌/日历 | ✅ 完成 |
| **Phase 4** 成本与风控 | 佣金/印花税/滑点/冲击、风控中间件、增量风险 | ✅ 完成 |
| **Phase 5** 策略宿主 | `on_frame` / `on_sub_step` / `on_fill` 分发、确定性 RNG | 🟡 部分（事件/条件/调度/lifecycle 为保留接口，引擎未调度） |
| **Phase 6** 高级能力 | 帧内子步进执行引擎（OrderBook + 价路径 crossing + FIFO 成交量分摊）、`sub_steps` 双时钟域、`SELL_SHORT` + 可借量/保证金、LiveConnector 实盘地基 | 🟡 部分（期权生命周期、帧内事件队列、多市场混合总线仍路线图） |

> Phase 0–4 已在真实数据上逐项验证（见 §8.2 / [reports/VALIDATION.md](reports/VALIDATION.md)）：
> 5575 标的 / 15,577,265 根真实日线、I1–I5 全通过、账本外部审计零偏差、成本可精确核验。
> §6 的自适应多因子系统另有独立验证（见 §6 / [reports/ADAPTIVE.md](reports/ADAPTIVE.md)）：
> 当前样本外为负，不能作为正 alpha 证据。

**已知边界**（诚实声明）：

- 撮合是**帧内子步进执行引擎**：价格-时间优先订单簿（`OrderBook`）+ 价路径 crossing
  （买限价单 `low ≤ 限价`、卖限价单 `high ≥ 限价`）+ FIFO 成交量分摊（部分成交可跨帧续接）。
  `LIMIT` 成交价是「限价或更优开盘价」，**不按当日 low/high**；`STOP/TRAILING` 用帧内
  high/low 穿越触发价成交。仍未建模逐笔盘口深度、跨价位成交量分配、STOP 的逐笔路径。
- `prefetch_hints` 为**性能提示**，当前不影响正确性（未声明请求也会实时计算），符合「声明不构成白名单」。
- **默认 `risk=NoOpRisk()` 不拦超买**：策略若绕过 `host.has_pending` 重复下单，现金可为负
  （隐含无息杠杆）。严肃回测请显式传 `StandardRisk(...)`。
- 条件单在触发帧的**下一帧开盘**成交（条件用收盘/BBO 判定）；`STOP_LIMIT` 目前只有一个
  `price` 字段，语义为「限价 = 触发价」。
- `IOC`/`FOK` 受当帧成交量约束（IOC 余量取消、FOK 不足整单取消）；`MARKET` 也受成交量约束。
- **I5 只保护公开写 API**：`bus.record / set_now / reset` 已加引擎写保护；Python 无法阻止
  策略直接访问私有属性（如 `bus._series`）。
- 做空：`SELL_SHORT` + 可借量（以 `float_shares` 代理）+ 保证金比例已实现；
  **借券费/强平**未实现。
- 交易日历 `trading_days` 按逻辑时间戳匹配（`frame.end_ns` 或 `end_ns-1`）；涨跌停是
  **单一 `limit`**，不自动区分创业板/科创板（20%）与北交所（30%）。
- 多时间框架按「每 N 根基础 Bar」聚合，非按挂钟周期（日线数据下等价于 N 日线）；
  `forming_bar` 返回最近一根基础 Bar，不是按 N 根做部分聚合的高周期 Bar。
- 验证池为 **5575 只全市场**（VALIDATION）/ **500 只代表样本**（ADAPTIVE），仍存在
  存活偏差（数据源不含退市股），结论不可外推；`tools/backfill_history.py --symbols` 可换池。
- 成交量分摊按「标的当日成交量」整帧摊派，未按价格-时间维度在帧内进一步拆分到各价位档。
- 多因子系统**未实现**的部分（DolphinDB 流批一体、L2 日内因子、财务 PIT 双时间戳、
  资讯事件 GMM 分类、实盘并行验证）在 [docs/MULTIFACTOR.md](docs/MULTIFACTOR.md) §6
  按「数据缺失 / 依赖缺失 / 明确未做」三类逐条列出。

> 2026-10-20 的修复与逐项验证记录见 [docs/archive/REMEDIATION.md](docs/archive/REMEDIATION.md)；
> 修复前的独立审计快照见 [docs/archive/AUDIT_README_CLAIMS.md](docs/archive/AUDIT_README_CLAIMS.md)。

---

## 10. 关键设计决策（对应设计文档 §15 开放问题）

| 开放问题 | 本实现的取舍 |
|---|---|
| **1. Decimal vs Float** | 全部用 `float`（双精度）。理由：I3 要求的「逐位复现」在**同机同解释器**下 float 完全满足（IEEE-754 确定性），而全 `Decimal` 会带来数量级性能损失。边界：**账本现金/数量用 float 累加**，若需分厘不差的对账，可在 `Ledger` 之上外挂 Decimal 影子账（不动引擎）。 |
| **2. 多时钟域因果令牌** | **双时钟域已实现**（`sub_steps`：日帧慢时钟 + 执行子步进快时钟）。令牌仍只带 `issued_at / frame_id / sub_step`，**不带市场维度**；多市场未实现。跨市场时建议在令牌上加 `market` 字段而非新建令牌类型（保持「一个虚拟时间」的单一真相）。 |
| **3. 帧内事件队列与子步骤统一** | **事件队列未实现**（Phase 6）；`on_sub_step` 已由引擎按 `sub_steps` 真正调度。当前 `on_sub_step` 与 `EventQueue` 仍是两个独立接口，倾向合并为单一「帧内时间轴」：`sub_step` 是时间轴上的刻度，事件按 `(timestamp, sequence_id)` 插入同一时间轴。 |
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
