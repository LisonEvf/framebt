# Phase 6 · 四轴打通（重新设计）

> ⚠️ **2026-10-20 更新**：本文是阶段设计/门禁快照，部分数字已过期——
> 测试数从 280 增至 **429**（422 passed + 7 xfailed 边界）；`on_sub_step` 已被引擎真正调度；吞吐量以
> `reports/VALIDATION.md`（36k / 868 bars/s）与本机复测（约 8k bars/s）为准，
> 本文的「84k / 7k」不再引用。`LIMIT / STOP / TRAILING` 的成交价语义也已修正
> （见 README §5.2 / §9）。

> 目标：把「回测 / 实盘同一套策略与引擎调度」从接口契约落到可运行实现，并在帧内执行、
> A 股融券做空、双时钟域三个维度补齐 Phase 6 路线图。
>
> **四轴**：① 数据/执行抽象地基 ② 帧内子步进执行引擎 ③ A 股融券做空 ④ 双时钟域调度。
> ⑤ 实盘 connector 是四轴的落地载体（数据轴 = LiveFeed，执行轴 = LiveConnector）。
>
> **每阶段门禁**：`validate_framework.py` 全绿 + `pytest` exit 0，再进下一阶段。

---

## 0. 现状基线

| 能力 | 现状 | 本阶段 |
|---|---|---|
| `MarketFeed`（数据抽象） | 已定义（`feed.py`） | 补 `LiveFeed` 实盘实现 |
| `ExecutionBackend`（撮合抽象） | 已定义（`backend.py`），`MatchingEngine` 实现 | 补 `LiveConnector` 实盘实现 |
| `OrderBook` 价格-时间优先 | 已实现（`book.py`） | 补**成交概率**模型 |
| 帧内价路径 crossing + FIFO 成交量分摊 | 已实现（`engine.py` LIMIT 分支） | 保持兼容 |
| `on_sub_step` 回调 | 已定义，引擎**未调用** | 双时钟域调度中真正调度 |
| 做空 | 仅账本负持仓，无借券/保证金 | `OrderSide.SELL_SHORT` + Ledger + margin + 可借数量 |
| 单时钟域 | 是 | 双时钟域（策略日帧 + 执行子步进） |

---

## 1. 四轴架构

```
                        ┌──────────────────────────────────────────┐
   策略代码              │  Engine（调度逻辑，与后端/数据源解耦）      │  engine.py
  FrameStrategy         │    · 帧生命周期四阶段                     │
        ▲               │    · 双时钟域：日帧(慢) + 子步进(快)       │  ← ④
        │  on_frame     │    · 帧内 on_sub_step 父调度同步          │
        │               └──────────────┬───────────────────────────┘
   ┌────┴────┐          ① 地基          │  ① 地基
   │ Market │  MarketFeed（抽象）       │  ExecutionBackend（抽象）  │
   │  Feed  │  ┌─────────────┐         │  ┌──────────────┐         │
   │        │  │ BarFeed 回测 │ LiveFeed│  │ MatchingEngine│LiveConn │
   │        │  └─────────────┘  实盘    │  └──────┬───────┘ 实盘     │
   └─────────┘                        ⑤ connector└───────⑤ connector
                        ② 帧内子步进执行引擎（OrderBook + 成交概率 + FIFO）
                        ③ A 股融券做空（SELL_SHORT + Ledger + margin + 可借）
```

---

## 2. ① 实盘复用地基（MarketFeed / ExecutionBackend）

**契约（已存在，本阶段加固）**：
- `MarketFeed.bars` → 按 `(timestamp, sequence_id)` 升序的有序 Bar。回测 = `BarFeed`，实盘 = `LiveFeed`。
- `ExecutionBackend`：`submit / cancel / process / has_in_flight / pending_orders / register_group`。
  回测 = `MatchingEngine`，实盘 = `LiveConnector`。
- `Engine(executor=...)` 已支持注入自定义后端；本阶段新增 `Engine(feed=...)` 接受任意 `MarketFeed`。

**本阶段新增**（在 `framebt/live.py`，`__init__` 导出 `LiveFeed / LiveConnector / Transport / SimulatedTransport`）：
- `LiveFeed(MarketFeed)`：增量喂入 `push(bar)`，按 `(timestamp, sequence_id)` 升序暴露 `bars`。
  未显式指定 `sequence_id` 的 Bar 按到达顺序分配；已显式指定则原样保留（尊重上游消息号）。
  与 `BarFeed` 同口径，引擎只读有序 Bar。
- `Transport(ABC)`：券商/CTP venue 抽象 —— `submit / cancel / get_fills(frame, frame_id) /
  get_state / available_to_borrow`。`get_fills` 的 `frame_id` 供 venue 执行延迟管线。
- `SimulatedTransport(Transport)`：确定性的「参考 venue」，成交语义与 `MatchingEngine` 对齐
  （延迟管线 + 帧内 crossing）；用于离线模拟与测试，与 `MatchingEngine` 独立实现。
- `LiveConnector(ExecutionBackend)`：封装券商订单流。`submit` 经 transport 报单并记
  `placement_frame`；`process` 轮询 `transport.get_fills` 转 `FillEvent`；OCO 联动在成交后
  撤销未成交的兄弟腿（先置 `state=CANCELLED` 再移出队列）。`delay` 自管（跳过 MatchingEngine 的
  delay_frames 校验）。
- 关键不变式：换后端/数据源，策略与 Engine 调度逻辑零改动。已扩 `test_executor_matches_matching_engine`
  覆盖 LiveConnector（`test_live.py`，10 例）。

**门禁**（已通过 ✅）：
- `pytest`：**422 passed + 7 xfailed（共 429）**（当前全量；早期门禁时为 280），exit 0。
- `validate_framework.py`：section 1–6 全绿（数据完整性 ✓ / I1 可见性单调 ✓ / I2 delay ✓ /
  I3 确定性 ✓ / A 股涨跌停+T+1 ✓ / 成本审计偏差 0 ✓；6b-6d 为旧版产物，6d 已作废，
  见 `reports/VALIDATION.md` 顶部与 `reports/SIGNAL_TRANSMISSION.md`）。
  section 7（吞吐量，仅信息）：单标的 DualMA ≈ **36k bars/s**（报告口径；本机复测约 8k），横截面动量 ≈ **868 bars/s**（全 5575 标
  O(n) 约 8 分钟，非阻塞门禁）。

---

## 3. ② 帧内子步进执行引擎（成交概率，N=1）

**现状**：`OrderBook`（价格-时间优先）+ 帧内 crossing + FIFO 成交量分摊已实现。
**本阶段补齐**：

### 3.1 成交概率模型
 resting 限价单的成交概率由「队列位置 + 价位深度 + 本帧成交量」决定，**确定性**（不引入 RNG，保 I3）：

```
fill_fraction(order) = min(1.0, frame_volume / level_clearing_volume)
                       * 1/(queue_position + 1)   # FIFO 加权：越靠后分得越少
```

- `level_clearing_volume` = 该价位的挂单总量（shares）。
- 模型只给**估计**（策略在 `on_sub_step` 可查询），实际成交仍走确定性 FIFO 分摊。
- 暴露：`MatchingEngine.estimated_fill_fraction(order) -> float`；`OrderBook` 提供
  `queue_position / depth` 供其计算。

### 3.2 双时钟域的前置（N=1）
 `on_sub_step(frame, step, total, token, bus)` 在**本帧撮合之后、推进虚拟时间之前**调用一次
 （`total=1`，即 N=1）。这是「执行时钟」对「策略日帧」的父调度同步：
 策略可在子步进中查询成交概率 / 在途订单，调整下一帧下单（不跨帧因果）。

**实现（已落地）**：
 - `MatchingEngine.estimated_fill_fraction(order) -> float`（`engine.py`）：确定性（无 RNG，
   保 I3），`min(1.0, frame_volume / level_shares) × 1/(queue_position+1)`，`level_shares =
   depth × order.size`；resting 数据缺失（无簿 / 非 resting / 市价 / 未 process）→ 0。
   进程内记 `_last_frame`（`process` 末尾同步），供子步进取本帧成交量。
 - `ExecutionBackend.estimated_fill_fraction` 默认返回 0（跨后端契约，实盘 `LiveConnector`
   未接入 venue 成交模型时「未知」）。
 - `Engine.run()` 在 Phase 3 撮合之后、Phase 4 快照之前插入
   `host.dispatch_sub_step(frame, 0, 1, token, self.bus)`；返回订单经 `_register_items`
   进延迟管线（I2）。`token` 由「仅注入帧时创建」改为「每帧恒创」（静态帧亦需权威时间）。

**完成记录（已通过 ✅）**：
 - `pytest`：292 passed（280 既有 + `tests/test_sub_step.py` 12 新），exit 0。
 - `validate_framework.py`：section 1–6 全绿（数据完整性 无✓ / I1 可见性单调✓/前视逃逸0✓ /
   I2 违反延迟0✓/delay=0拒绝✓ / I3 逐位一致✓ / A股 涨跌停被拒✓/T+1被拒✓放行✓ / 5a 偏差0✓ /
   5b 一致性✓ / 6b-6d 信号传导翻转一致✓）。`on_sub_step` 默认注册关闭（`default_wakeups` 不含
   SUB_STEP），6 类检查函数均未触碰，结果与既有一致——零回归。
 - 注：全 5575 标横截面动量 section 7 为 O(n) 信息项（约 7630s，非阻塞门禁）；6b-6d 扫描用
   `--max-symbols` 代表样本。

**门禁**：`test_sub_step.py`（概率模型 + 调度）；`validate_framework.py` 全绿（零回归）。

---

## 4. ③ A 股融券做空（SELL_SHORT + Ledger + margin + 可借数量）

### 4.1 订单
 `OrderSide.SELL_SHORT`（区别于 `SELL` 平多）。`Order.short = True` 标记。

### 4.2 Ledger
 - 空头持仓：`Position.net_size() < 0`。多空分开腿（`short_legs`）。
 - `unrealized_pnl(mark)`：空头 = `(entry - mark) * |size|`。
 - `margin_requirement()`：多头 0；空头 = 空头名义市值 × margin_ratio（默认 50%）。
 - `equity()` 扣保证金占用（权益 = 现金 + 多头市值 − 空头保证金占用... 见下）。

### 4.3 规则
 - T+1 仍适用。
 - **涨停不可开空**（`StandardAshareRules`：`SELL_SHORT` 在涨停价被拒）。
 - **可借数量**：做空数量 ≤ `rules.available_to_borrow(symbol)`（来自行情 float_shares / 可融券余额）。

### 4.4 成本
 `StandardCost` 加 `borrow_fee`（融券费，按空头名义市值/日）；`apply_lifecycle` 处理逼空。

**门禁**：`test_short_selling.py`；`validate_framework.py` 全绿（默认不开做空，零回归）。

---

## 5. ④ 双时钟域调度

- 引擎在帧内（撮合后）插入 `on_sub_step` 调用（`host.dispatch_sub_step`）。
- 父调度同步：日帧 `on_frame`（慢时钟，决策）→ 撮合（执行时钟）→ `on_sub_step`（快时钟，观测/微调）。
- 因果约束：子步进与主帧共享同一 `token`（`issued_at` 不变），不暴露未来。
- N=1 固定；N>1 多子步为后续（路线图）。

**门禁**：`test_sub_step.py`（调度顺序/因果）；`validate_framework.py` 全绿。

---

## 6. ⑤ 实盘 connector

- `LiveConnector` 收口券商/CTP：`transport` 抽象（`SimulatedTransport` 测试 / 真券商对接）。
- `LiveFeed` 收口实时行情：把 tick/逐笔流切成帧内 Bar。
- 实盘下双时钟域自然成立：日帧 = 收盘/定时唤醒，子步进 = 逐笔/分钟撮合。
- 不变式：`LiveConnector` 遵守 `ExecutionBackend` 契约，与 `MatchingEngine` 对模拟场景
  逐笔一致（扩写 `test_executor_matches_*`）。

**门禁**：`test_live_connector.py`；`validate_framework.py` 全绿。

---

## 7. 向后兼容硬约束（Phase 0–5 契约）

`MatchingEngine(delay_frames=1, fill_price_mode="open")`、`register`、
`process(frame, frame_id, pre_trade_check=...)`、`cancel`、`has_in_flight`、
`pending_count`、`OrderBook` 占位、`OrderGroup/Atomicity/LinkMode`、`run_backtest(...)` 签名。

任何新特性**默认关闭**（`on_sub_step` 不注册、做空不启用、成交概率不改变既有 FIFO 成交），
保证 I1–I5 与 270 单测零回归。

---

## 8. 阶段清单与门禁

| 阶段 | 内容 | 新增测试文件 | 门禁 |
|---|---|---|---|
| ① | LiveFeed / LiveConnector 地基 | `test_execution_backend.py` 扩写 | ✅ validate + pytest（429 项） |
| ② | 成交概率 + on_sub_step(N=1) | `test_sub_step.py`（12 例） | ✅ validate + pytest（292 passed） |
| ③ | SELL_SHORT / Ledger / margin / 借券 | `test_short_selling.py` | validate + pytest |
| ④ | 双时钟域调度 | `test_sub_step.py` 扩写 | validate + pytest |
| ⑤ | 实盘 connector | `test_live_connector.py` | validate + pytest |

