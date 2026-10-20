# framebt README 能力核验报告（独立审计）

> **2026-10-20 晚更新**：本报告是**修复前快照**。§6 的 P0/P1 项大部分已落实
> （惰性 report 导入、退市现金、I5 写保护、STOP/LIMIT/IOC/FOK 语义、条件单下一帧、
> 交易日历、UTF-8、6d 重复下单、文档数字更新），详见
> [`REMEDIATION.md`](REMEDIATION.md)。本文保留原始证据，便于对照修复前后。

> 审计对象：`framebt` 工作树（README + `reports/` + `src/` + `tools/`）
> 审计时间：2026-10-20 ｜ 环境：Windows / Python 3.12.10 / pandas 3.0.5 / pyarrow 25.0.1 / numpy 2.5.3
> 方法：跑完整测试、跑端到端能力验证、对 README 的每条硬声明做独立探针复算；不采信 README 自述，只采信可复现输出与代码路径。

---

## 0. 结论先行

**引擎是真的，且工程质量在同类个人项目里偏上；但 README 作为“能力说明书”已经不可信。**

- **真实成立**：因果令牌 + 延迟管线（I1/I2）在真实数据上可复现；I3 逐位确定性；账本/成本可被独立实现复算到机器精度；57 因子、CUSUM/卡尔曼、符号回归等模块确实存在；363 个测试全过。
- **明确失实/过期**：README 的旗舰业绩 **+42.47%** 已被仓库内最新的 `reports/ADAPTIVE.md` 推翻为 **−78.55%**；§8.2/§8.3 的 24 股验证数字整体过期（当前是 5575 股 / 1557 万根 Bar）；“零第三方依赖 / `pip install -e .`”在干净环境不成立；“退市现金结算”没实现；“I5 不可改写过去”没有被强制；“多唤醒”7 类里 4 类引擎从不调用。
- **定性**：**存在实质性失实宣传**。但证据更像「数据从 24 股迁移到全市场后，报告重跑了、README 没同步」加上「若干功能声明超前于实现」，**没有发现凭空编造**——旧数字在 git HEAD 的 24 股报告里能逐项对上。

**一句话**：这是一个“底子不错、文档失修、关键边界被夸大”的框架；不能按 README 字面采购或用于实盘决策。

---

## 1. 核验方式与规模

| 项 | 实际执行 | 结果 |
|---|---|---|
| 单元测试 | `python -m pytest -q`（PYTHONPATH=src） | **363 个用例全过**（exit 0；README 写 351） |
| 端到端能力验证 | `validate_capability.py`（全量，UTF-8） | **80.0s**，8/10 预登记预测通过，P1–P4 精确判据全部零偏差 |
| 真实数据 | `OpentdxParquetSource` | 5575 标的，DAILY/QFQ，报告口径 15,577,265 根 Bar |
| 独立探针 | 见 §2–§4；不调用被审计脚本的结论函数 | 复现/证伪逐条记录 |
| 打包 | `python -S -c "import framebt"`（无 site-packages） | **ModuleNotFoundError: matplotlib** |

> 说明：`validate_framework.py` 的全量跑在本机需要数小时（其自带报告 6c 横截面 17,942s），本次没有全量重跑；但 I1/I2/I3/规则/成本/账本这些核心项已用独立探针和 `validate_capability.py` 覆盖，足以判断声明真伪。

---

## 2. 属实的声明（给分项）

| README 声明 | 核验证据 | 判定 |
|---|---|---|
| I1 任何数据请求 `timestamp <= 当前虚拟时间` | 300 帧真实数据探针：可见 Bar 数 1..300 单调；`visible_ts <= issued_at` 全真 | ✅ |
| I1 伪造 token 拿不到额外数据 | 每帧伪造 `issued_at=1e9` 的 token：额外行数 **0**；再叠加公开的 `bus.set_now(1e9)`：仍 **0** | ✅（机制见 §4.5 的边界） |
| I2 帧 t 订单最早 t+delay 成交；`delay_frames=0` 拒绝 | 真实数据 400 帧 DualMA：27 笔成交、最早帧 33；`Engine(delay_frames=0)` 构造即 `ValueError` | ✅ |
| I3 相同配置逐位复现 | `validate_capability` P3：权益 True / 成交 True；`DetermRNG` 为自实现 mulberry32 | ✅ |
| 账本外部审计零偏差 | P1：逐帧权益重算最大相对/绝对偏差 **0.000e+00**（本次终值 303,593.039651） | ✅ |
| 成本可逐笔核验 | P2：max\|cost − 手算\| = **7.105e-15 元**；P4 无成本引擎 vs 独立 numpy 克隆 **1136/1136 笔一致、相对差 0.000e+00** | ✅ |
| 57 个 WQ/GTJA 风格因子 | `framebt.factors.library` 实测 `len(FACTOR_LIBRARY)=57`；公式确为 WQ101 子集改写 | ✅ |
| IC/ICIR/Newey-West t/分组单调性 | `factors/evaluate.py` 均有真实实现（`newey_west_tstat`、`monotonicity_score` 等） | ✅ |
| CUSUM 变点 + 卡尔曼平滑 | `monitor/changepoint.py` 实现 `cusum_detect` 与 `KalmanICFilter` | ✅ |
| A 股涨跌停 / T+1 机制 | 独立构造：涨停买入拒、跌停卖出拒、T+1 同日卖出拒、次日放行 | ✅（幅度口径见 §4.6） |
| 订单类型 / OCO / 部分成交 | MARKET/LIMIT/STOP/STOP_LIMIT/TRAILING/IOC/FOK 存在；`_allocate_volume` 有 FIFO 分摊与跨帧续接；示例可跑 | ✅（语义缺陷见 §4.2） |
| 图表化 HTML 报告 | `demo_quickstart.py` 2.3s 生成 `reports/demo_quickstart.html`，含权益/回撤/K 线 | ✅ |
| 363 个测试全过 | 实测 collect 363，exit 0 | ✅（README 的 351 反而少报） |

**特别肯定**：`reports/CAPABILITY.md` 是一份高质量验证——预登记 10 条预测、如实保留 2 条失败（P5b 换手 9.66×、P7 价差 18.15%），并把数据复权缺陷写进报告。这种“把失败也留下”的做法比绝大多数回测框架文档诚实。本次全量重跑（多一个交易日）得到 8/10、P1/P2/P4 仍为零偏差、P5a=1.05%、P5b=9.81×、P7=18.81%，与该报告的结论一致。

---

## 3. 失实 / 过期 / 夸大的声明

### S1（严重）旗舰业绩 +42.47% 已失效

| | README（第 294–298 行） | 仓库当前报告 `reports/ADAPTIVE.md`（第 48–60 行） |
|---|---|---|
| 样本外总收益 | **+42.47%** | **−78.55%** |
| 基准等权买入持有 | +16.71% | **+845.57%** |
| 超额 | **+25.76%** | **−924.12%** |
| Sharpe / 最大回撤 | 0.37 / 31.53% | **−0.10 / 97.39%** |
| 成交 / 费用 / 换手 | 856 笔 / 12.3% / ~13.5× | **9740 笔 / 267.53% / 27.9×** |
| 数据 | 24 只 × 2000 根 | **500 只代表样本**（全市场 5575） |

README 把 +42.47% 写成“真实数据验证结果”，但同一工作树里由同一脚本生成的报告是 −78.55%。这不是小数点漂移，是**符号反转**。旧数字确实来自 HEAD 的 24 股报告（`git show HEAD:reports/ADAPTIVE.md` 可逐项对上），所以是**过期**而非编造；但读者没有任何提示，属于实质性误导。

### S2（严重）§8.2 / §8.3 的验证数字整体过期

| 声明 | README | 当前 `reports/VALIDATION.md` |
|---|---|---|
| 验证池 | 24 标的 × 2000 日线 = 48,000 bars（第 370 行） | **5575 标的 / 15,577,265 bars**（第 3、12 行） |
| 涨停 / 跌停日 | 236 / 102（第 379 行） | **374,858 / 239,168**（第 39–40 行） |
| 单标的吞吐 | **91k bars/s**（第 382 行） | **36,125 bars/s**（第 111 行） |
| 横截面吞吐 | **7.2k bars/s** | **868 bars/s**（第 112 行） |
| momentum / reversal | **−87.3% / +201.6%**（第 390–391 行） | **−102.61% / +235,360.46%**（第 92–93 行） |
| random / 买入持有 | **−34.8% / +180.7%** | **+4,818.39% / +356.01%**（第 94–95 行） |
| 纯统计 60 日动量 | −0.058；最高 +1.25%、最低 +3.14% | −0.0020；**+3.78% / +9.22%**（第 99–101 行） |

README 第 398–399 行还说“随机组合的小幅亏损来自日频换手……属预期摩擦”，而当前报告里随机组合是 **+4,818.39%**。报告正文第 103–105 行同样写着“随机组合亏损”，与自己的数字表矛盾——因为这段结论是 `validate_framework.py:465-469` **硬编码**的，不随结果更新。README 引用这张表作为“最有说服力的一项”，但它的结论文字与数据已经对不上。

### S3（高）“核心引擎零第三方依赖 / `pip install -e .` 仅核心引擎”不成立

- `pyproject.toml:19` `dependencies = []`；可选依赖只有 `data=pandas,pyarrow`、`dev=pytest`，**没有 matplotlib**。
- 但 `src/framebt/__init__.py:62` 在包初始化时执行 `from .report import ...`，而 `report.py:33` 顶层 `import matplotlib`。
- 实测 `python -S -c "import framebt"`（不加载 site-packages）→ `ModuleNotFoundError: No module named 'matplotlib'`。
- 也就是说：干净环境 `pip install -e .` 后，连 `from framebt import Engine` 都 import 不了；“零依赖”只在“不 import 包”的意义上成立。
- `report.py:19` 的 docstring 还写“matplotlib（可选依赖 data）”，但 `data` 里并没有它。

**最低修复**：`__init__` 里把 `report` 改成惰性导入（或把 matplotlib 加进 `[project.optional-dependencies].report` 并在 README 说明）。

### S4（高）“公司行动：退市现金结算”没有实现

`ledger.py:143-147` 的 `delist` 分支只把腿清空、记 `realized_pnl`，**没有给现金入账**；`Ledger.apply_corporate_action`（`ledger.py:223-229`）只对 `dividend` 加现金。

独立复算：
```
买入 100 股 @10，现金 100,000 → 权益 100,000
apply_corporate_action(CorporateAction(kind="delist", ratio=10.0))
权益 → 99,000（现金仍 99,000）
```
正确语义应按 10 元/股现金结算给 100,000。`tests/test_ledger_rules_cost.py:126-132` 的用例名叫 `test_ledger_delist_cash_settlement`，但只断言 `net_size==0` 和 `realized_pnl==-200`，**没有断言现金**，所以测试把 bug 放过了。

### S5（高）I5“策略无法改写已过去帧”没有被强制

README 第 73 行把 I5 列为引擎硬约束；但 `DataBus.record`（`data_bus.py:73`）是公开、无时间校验的。策略在 `on_frame` 里可以直接写：
```python
bus.record("close", "S", 0, 9999.0)   # 改写第 0 帧
```
独立探针结果：第 2 帧再用 `bus.get` 读到的历史变成
`[(0,10.0),(1,11.0),(0,9999.0),(1,8888.0),(2,12.0)]` —— 过去被追加/污染。
`tests/test_invariants.py:162-175` 的 I5 用例只验证“可见历史长度单调”，**没有验证写保护**。所以 I5 目前是“约定”，不是“硬约束”。

### S6（中）“多唤醒机制”7 类里 4 类引擎从不调用

`host.py` 定义了 `dispatch_event / dispatch_condition / dispatch_schedule / dispatch_lifecycle`，但全仓库只有定义、**没有调用点**（`grep` 仅命中 `host.py` 自身）。真正被引擎分发的是：

- `FRAME`（`engine.py:205`）
- `SUB_STEP`（`engine.py:231-232`）
- `FILL`（`engine.py:217`）

`on_event / on_condition / on_schedule / on_lifecycle` 注册了也不会被调用。README 第 240–243 行把它们与 `on_frame` 并列展示，读起来像 7 类都可用；第 443 行 Phase 5 只笼统写“部分（事件注入层、异步推理未实现）”，没有说清是 4 类空接口。

### S7（中）“交易日历”是死参数

`StandardAshareRules.is_tradable` 的逻辑是：只有当 `frame.is_trading == False` 时才可能返回 False；但全仓库 `is_trading` 只在 `MarketFrame.__init__` 默认 `True`（`market_frame.py:66`），**没有任何地方传 False**。因此 `trading_days=` 永远不会拦住任何订单。README 第 228 行“涨跌停、停牌、T+1、交易日历”里的“交易日历”目前不生效。

### S8（中）IOC / FOK 不检查成交量，FOK 语义不成立

`execution/engine.py:461-477` 的 IOC/FOK 分支直接按 `size` 全额成交，完全跳过成交量约束；而 MARKET 分支（`:452-458`）会把 `size` 截到 `frame.volume`。

独立探针（当日 volume=100 股）：
```
MARKET 成交 100 股
IOC    成交 1,000,000 股
FOK    成交 1,000,000 股
```
FOK 的“全部成交或取消”至少应受可成交量约束（或明确声明为理想化）。README 第 205 行把它们列为已支持订单类型，但没有披露这个语义缺口。

### S9（中）STOP / TRAILING 的“收盘触发、开盘成交”是帧内前视

`execution/engine.py:508-527`：用 `frame.last`（收盘）判断触发，却用 `_tradable_price`（开盘）成交。两者在同一根 Bar 内，开盘在收盘之前。

独立探针：
```
SELL STOP @9；该帧 open=10.0, high=10.5, low=7.8, close=8.0
成交价 = 10.0（而不是 ≤9 或下一帧开盘）
```
等价于“用收盘才知道跌破止损，却回到开盘去卖”。BUY STOP 同理会在突破前的最低价买入。对止损/突破策略是**系统性有利偏差**；`conditional_stop_loss.py` 的 +180.60% 这类结果会被高估。README 只披露了 LIMIT 的“极值近似”，没有披露 STOP 的这个问题。

### S10（中）LIMIT 按当日极值成交，偏乐观（虽已披露）

`engine.py:529-551`：买限价单只要 `low <= limit`，就以 `open`（若已跌破）或 **`low`** 成交；卖限价单以 **`high`** 成交。

独立探针：
```
BUY LIMIT @10，open=12，low=9  → 成交 9.0
SELL LIMIT @10，open=8，high=11 → 成交 11.0
```
现实里限价 10 的买单在 12 开盘、跌到 9 的过程中应在 10 成交，拿不到 9；这是“事后知道最低点”的乐观假设。README 第 454–455 行确实写了“按帧内触发极值近似”，属于已披露的近似，但对限价单策略的收益影响可能很大。

### S11（中）README 的验证命令在中文 Windows 控制台直接崩

`validate_capability.py:1044` 的输出含 U+2212（−），`validate_framework.py` 的 `out()` 含 U+2713（✓）；默认 GBK stdout 无法编码：

```
python -u tools/validate_capability.py --quick --md reports/_audit_cap.md
→ UnicodeEncodeError: 'gbk' codec can't encode character '\u2212'
→ EXIT=1，报告文件未生成
```
加 `PYTHONIOENCODING=utf-8` 后全量 80s 正常。README 第 286–288、346–348 行给的命令没有这个前提；对中文 Windows 用户是“照抄即失败”。

### S12（低）过期/反向失实：README 少报了已实现的能力

- 第 459 行“`on_sub_step` 接口已定义，引擎尚未在帧内调用”——**已调用**：`engine.py:230-233` 每帧按 `sub_steps` 调度；独立探针 `sub_steps=3` × 5 帧得到 **15 次** `on_sub_step`。
- 第 444 行“多时钟域仍路线图”——`sub_steps` 双时钟域已实现，且有 `tests/test_dual_clock.py`。
- 第 457 行“做空仅支持账本层面的负持仓，借券可用性未实现”——`OrderSide.SELL_SHORT`、`available_to_borrow`、`margin_ratio`、`tests/test_short_selling.py`（22 例）都已存在（借券费/强平确实没有）。
- 第 332 行“5 个可运行示例”——实际 7 个；第 335/343 行“351 个测试”——实际 363。

### S13（低）吞吐量 91k bars/s 没有报告支持

- README 第 382 行：单标的 **91k bars/s**、横截面 **7.2k bars/s**。
- 仓库自带 `reports/VALIDATION.md`：**36,125 / 868 bars/s**。
- 本机独立复测（600519，2002 根，DualMA 5/20，预热后连跑 3 次）：**8,066–8,129 bars/s**。
- 91k 既高于自带报告 2.5×，也高于本机实测 11×。不同机器会有差异，但“91k”在任何仓库产物里都找不到依据。

---

## 4. 比文档错误更值得注意的实现问题

### 4.1 默认允许无限透支，验证报告里的异常收益就来自这里

`Engine` 默认 `risk=NoOpRisk()`；`Ledger.apply_fill` 只做 `cash -= notional + cost`，**不检查现金是否足够**。策略一帧内提交两笔各 100% 现金的预算单：

```
初始 1,000,000；两笔 BUY budget=1,000,000 @10
成交后 cash = -1,000,000；持仓 200,000 股；equity = 1,000,000
```

这等于 **2 倍无息杠杆**，且没有任何保证金/强平。README 第 234–236 行其实提醒过“在途买单要预留，否则会透支”，但默认引擎并不阻止；`reports/VALIDATION.md` 的 6d 恰好没有传 `risk`，于是：

- 50 标的复现：momentum 最终 **cash = −160,026**（权益仍为正）；
- 500 标的报告：momentum **−102.61%**（终值 −26,132），reversal **+235,360%**——负现金/杠杆会把信号方向放大成这种量级。

**这不是“框架忠实传导信号”的证据**，而是“策略重复下单 + 默认无风控”的合成结果。

### 4.2 6d 的对照实验本身有策略 bug

`validate_framework.py:392-409` 的 `RankStrategy` 每帧都按“当前持仓”决定买卖，**没有用 `host.has_pending`**；在 I2 延迟下，同一批买单会在成交前被反复提交。README 第 256–259 行专门警告过这个坑，但自家验证脚本没有遵守。配合 4.1，就得到不可解释的 −102% / +235,360%。

### 4.3 硬编码结论与数据脱节

`validate_framework.py:465-469` 无条件打印“翻转排序后显著反转 / 随机组合亏损”。当前 random 是 **+4,818%**，这句话与数据相反。报告不是“由数据生成结论”，而是“结论写死、数据填表”。

### 4.4 `validate_capability.py` 的核心判据是可信的

需要区分：CAPABILITY 的 P1/P2/P3/P4 是**精确判据**，本次全量复跑仍为 0 / 7.1e-15 / True / 1136 笔 0 不一致。这说明引擎的“算术与成交重放”是可靠的；不可靠的是**策略层语义与默认风控**。两者不能混为一谈。

### 4.5 I1 的边界：令牌防伪成立，但“权威时间”可被策略改写

`bus.set_now` 是公开方法。虽然在本引擎的增量注入下，策略即使 `set_now(1e9)` 也看不到未来（未来 Bar 还没注入，探针额外行数 0），但 README 第 192–194 行声称“I1 由总线自身保证，而不是依赖恰好还没预加载未来数据”。严格说：**总线自身并没有锁住 `set_now`**，当前安全性依赖“引擎增量注入 + 策略不主动篡改”两个外部条件；一旦换成预加载总线或复用 DataBus，这个假设就破了。

### 4.6 A 股涨跌停幅度是全局单一值，未按板块区分

`StandardAshareRules(limit=0.10)` 对所有标的一律 ±10%，但创业板/科创板是 ±20%、北交所 ±30%。`price_limit` 只用单一 `limit`，没有按代码前缀/板块切换。`reports/VALIDATION.md` 用 10% 去检验 `BJ.920000`（北交所）的涨跌停，口径本身是错的。README 写“默认 ±10%”算诚实，但“A 股规则”作为能力被高估了。

---

## 5. 关于“虚假宣传”的定性

**有失实宣传，但不是“凭空造假的杀猪盘”。**

支持“非故意”的证据：
1. README 的 +42.47%、−87.3%/+201.6%、24 股/48k bars 等数字，都能在 `git show HEAD:reports/*.md` 的 24 股报告里逐项对上——它们是**真实跑过**的旧结果。
2. `reports/CAPABILITY.md` 主动保留了两条失败预测，`reports/VALIDATION.md` 主动写了数据复权缺陷；这种自我暴露不像有意欺骗。
3. 引擎的 I1/I2/I3、账本/成本、57 因子等硬能力确实存在，且能被独立复算。

支持“实质失实”的证据：
1. 同一个工作树里，README 的旗舰业绩与 `reports/ADAPTIVE.md` **符号相反**；README 却没有任何“已过期”标注。
2. 6 项能力声明（零依赖、退市结算、I5、多唤醒、交易日历、IOC/FOK）与代码行为不符，其中退市结算是**功能性缺失**，不是措辞问题。
3. 验证脚本的结论文字硬编码，已经与自己的数字表矛盾；README 却把它称为“最有说服力的一项”。

所以更准确的表述是：**“工程实现 7.5/10，文档与宣传 3/10”**。如果按 README 字面评估，会显著高估其成熟度；如果按代码与 CAPABILITY 报告评估，它是一个有真实内核、但边界需要大量修补的研究型框架。

---

## 6. 修复优先级（如果要让 README 重新可信）

**P0（不修就是失实）**
1. 用 `reports/ADAPTIVE.md` 的当前数字替换 §6 的 +42.47% 表；明确写“24 股结果已作废”。
2. 用 `reports/VALIDATION.md` 的当前数字替换 §8.2/§8.3；删除/重写“随机组合小幅亏损”等已反向的结论。
3. `framebt/__init__.py` 惰性导入 `report`，或把 matplotlib 列入可选依赖并在 README 写清。
4. 修 `ledger` 的 delist 现金结算；补断言现金的测试。
5. 修 STOP/TRAILING 的帧内因果：收盘触发 → 下一帧开盘成交，或按触发价成交；LIMIT 至少改为按限价/更优价，而不是当日极值。
6. IOC/FOK 加成交量约束；FOK 做“可成交量不足则整单取消”。
7. 给 `DataBus.record/set_now` 加引擎私有授权（或至少在文档里把 I5 降级为“约定”）。

**P1（可信度）**
8. 所有验证脚本的输出改为 UTF-8 安全（`sys.stdout.reconfigure(encoding="utf-8")` 或去掉 ✓/−），并在 README 给一条 Windows 可复制的命令。
9. `RankStrategy` 加 `host.has_pending` 防重复下单；6d 加“现金不得为负”的断言；结论文字由数据生成，不硬编码。
10. `validate_framework.py` 的 `--max-symbols` 对核心检查也生效，或提供一个 5 分钟可跑完的 `--quick`。
11. 涨跌停按板块配置（至少允许 per-symbol limit 映射）。

**P2（文档）**
12. 把 `on_sub_step`、双时钟域、做空可用性从“未实现”改为“已实现/边界”；把 4 类空唤醒标为“未调度”。
13. 更新测试数（363）、示例数（7）、吞吐量（用当前报告/实测）。
14. `examples/multi_factor_universe.py` 默认限制在一个子集（如 200 只），否则它是“小时级示例”。

---

## 7. 附录：关键复现命令与原始数字

```powershell
# 1) 测试（363 个全过）
$env:PYTHONPATH="src"; python -m pytest -q

# 2) 端到端能力验证（必须 UTF-8，否则 GBK 控制台崩）
$env:PYTHONPATH="src"; $env:PYTHONIOENCODING="utf-8"
python -u tools/validate_capability.py
# 80.0s：P1=0.000e+00，P2=7.105e-15，P3=True，P4=1136/1136，
#        P5a=+1.05%，P5b=9.81×（FAIL），P7=+18.81%（FAIL），P8 逃逸 0，P9=+0.08%

# 3) 干净环境 import（证明 matplotlib 是硬依赖）
python -S -c "import sys; sys.path.insert(0,'src'); import framebt"

# 4) 退市结算
#    买入 100@10（现金 100,000）→ delist(ratio=10) 后权益 99,000，现金未增加

# 5) STOP 帧内因果
#    SELL STOP@9，帧 open=10/close=8 → 成交 10.0

# 6) IOC/FOK 成交量
#    volume=100 时 IOC/FOK 各成交 1,000,000 股；MARKET 只成交 100 股

# 7) I5 改写历史
#    策略内 bus.record("close","S",0,9999.0) → 后续 get 读到被追加的假记录

# 8) 默认透支
#    一帧两笔 budget=1,000,000 → cash = -1,000,000
```

**数据侧补充**：`SH.601088` 原始 QFQ `min(close) = -8.56`、`max|日收益| = inf`；经 `OpentdxParquetSource` 过滤后只剩 **345 根** Bar（原始约 2000 根），`max|日收益| = 29.8%`。这说明数据源现在的 `>30%` 过滤确实拦住了最脏的跳变，但代价是**静默丢掉大部分历史**，且 29.8% 仍高于 A 股 20% 的板块上限——README §8.4 仍写“只过滤 ≤0 的价格，拦不住”，已与当前代码不符。

---

## 8. 总评分（主观但按证据加权）

| 维度 | 分数 | 依据 |
|---|---|---|
| 因果性 / 防前视骨架 | 9.0 / 10 | I1/I2 真实数据可复现；令牌钳制有效 |
| 确定性 / 可复算 | 9.5 / 10 | I3、P1–P4、独立克隆 1136/1136 |
| 撮合真实性 | 5.0 / 10 | STOP 帧内前视、LIMIT 极值、IOC/FOK 无量约束、默认透支 |
| 账本 / 规则完整性 | 6.5 / 10 | 多腿/T+1/成本可靠；退市缺失、日历失效、板块幅度单一 |
| 多因子系统 | 6.0 / 10 | 模块真实、因果审计通过；当前 OOS −78.55%，换手/费用失控 |
| 工程与测试 | 8.0 / 10 | 363 测试全过、模块化清晰、报告可复现 |
| **README 可信度** | **3.0 / 10** | 旗舰数字符号反转、6 项能力失实、命令照抄即崩 |

> 结论：**可以把 framebt 当作“因果回测骨架 + 多因子研究脚手架”来用；不能把 README 当能力清单，也不能把 STOP/LIMIT/IOC/FOK 的回测结果当真实可成交结果。** 在修掉 §6 的 P0 项之前，任何基于该框架的收益结论都应打上“待复核”标签。
