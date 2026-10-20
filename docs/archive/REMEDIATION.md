# framebt 修复记录（2026-10-20）

> 对应审计：[`AUDIT_README_CLAIMS.md`](AUDIT_README_CLAIMS.md)。
> 目标不是把框架吹得更大，而是让 **README 的每条声明要么为真、要么明确标注边界**。
> 本文件记录改了什么、怎么验证、以及哪些仍没解决。

---

## 1. 代码修复（带回归验证）

| # | 审计发现 | 修复 | 验证 |
|---|---|---|---|
| 1 | `import framebt` 需要 matplotlib，但 `dependencies=[]` | `__init__.py` 用模块级 `__getattr__` 惰性导入 `report`；`pyproject` 新增 `report = ["matplotlib"]` / `all` extra | `python -S -c "import framebt"` 通过；`from framebt import generate_report` 仍可用 |
| 2 | 退市只清仓、不记现金 | `Ledger.apply_corporate_action` 先存 `qty_before`，`delist` 按 `ratio × qty` 入现金 | `test_ledger_delist_cash_settlement` 断言现金/权益，不再只查 `realized_pnl` |
| 3 | I5「不可改写过去」未强制 | `DataBus` 增加 `enable_write_guard(key)`；`record/set_now/reset` 必须带引擎私有 key | 新增 `test_i5_strategy_cannot_write_bus_history_or_clock`（3 次攻击全被拒）+ 引擎正常注入测试 |
| 4 | STOP 用收盘触发、当帧开盘成交（帧内前视） | 改为帧内 `high/low` 穿越触发：跳空按开盘、否则按触发价 | 新增 `test_stop_gap_fills_at_open_not_trigger`；更新 `test_stop_order_triggers_on_breakdown` |
| 5 | TRAILING 同样收盘触发/开盘成交 | 用 `high/low` 更新极值；先用「上一帧触发价」判跳空，再判帧内穿越 | 更新 `test_trailing_stop_trails_the_high`（成交 12.1×0.95） |
| 6 | LIMIT 按当日 low/high 成交（事后极值） | 开盘更优按开盘，否则按**限价** | 更新 `test_limit_order_fills_when_low_touches*`、`test_buy/sell_limit_*` |
| 7 | IOC/FOK 无视成交量 | IOC = min(size, volume) 后余量取消；FOK 可成交量不足整单取消 | 新增 `test_ioc_partial_fills_then_cancels_remainder`、`test_fok_cancels_when_volume_insufficient` |
| 8 | 条件单用收盘触发、当帧开盘成交 | 条件满足帧只「arm」，下一帧开盘才成交 | 更新 `test_conditional_order_fills_after_trigger`（帧 3 开盘成交） |
| 9 | `trading_days` 是死参数 | `is_tradable` 改为匹配 `frame.end_ns` 或 `end_ns-1` | 新增 `test_rules_trading_calendar_blocks_off_day` |
| 10 | 7 类唤醒里 4 类从不调用，静默失效 | 新增 `SCHEDULED_WAKEUPS`；`register` 对未调度类型发 `UserWarning` | 由 `host.register` 路径覆盖；README/USER_GUIDE 改为只列 3 类 |
| 11 | 验证脚本在 GBK 控制台崩 | `validate_capability.py` / `validate_framework.py` 在导入时 `sys.stdout.reconfigure(encoding="utf-8")` | 不设 `PYTHONIOENCODING` 跑 `validate_capability.py --quick --md ...` 成功写报告 |
| 12 | 6d 重复下单 + 默认无风控 → 负现金/杠杆 | `RankStrategy` 买卖都查在途单、有在途单则本帧不调仓；改用 `DetermRNG`；`universe` 与 `feed` 对齐；新增 `cash_path` 独立重算期末/最低现金 | 新脚本 100 只三组最低现金 0/1/12，无负现金 |
| 13 | `multi_factor_universe.py` 全市场跑成小时级 | 加 `--max-symbols`，默认 200 | 示例可分钟级跑完 |

**测试**：`369 passed`（原 363；新增/更新 6 个语义测试）。全量 `pytest` exit 0。

---

## 2. 文档修正

| 文件 | 修正 |
|---|---|
| `README.md` §3 | 零依赖改为「核心 import 不依赖第三方；Parquet 需 pandas/pyarrow；报告需 matplotlib」；安装命令加 `.[report]` / `.[all]`；删掉「24 股 × 2000 根」 |
| `README.md` §5.2 | 订单表补成交价口径：STOP/TRAILING 帧内 high/low、LIMIT 限价/开盘、IOC/FOK 成交量、条件单下一帧 |
| `README.md` §5.3 | 退市现金结算、交易日历、单一涨跌停参数边界 |
| `README.md` §5.5 | 只列 3 类已调度唤醒；补全局在途单等待与默认 NoOpRisk 透支警告 |
| `README.md` §6 | 用当前 ADAPTIVE 数字（−78.55% / 超额 −924.12%）；明确旧 +42.47% 已作废 |
| `README.md` §7 | 目录补 `live.py` / `selection/` / `rotation`；示例 7 个；测试 369 |
| `README.md` §8.2 | 用当前 VALIDATION 数字（5575 / 15.58M / 374,858 涨停 / 36k·868）；标注 6b–6d 已作废 |
| `README.md` §8.3 | 撤下旧 −87.3%/+201.6% 表，改为修正版 100 只对照 + 负现金闸门 |
| `README.md` §8.4 | 更新 QFQ 过滤说明（≤0 + >30%）；补 1136/1136 重跑；85s→80s |
| `README.md` §9 | 更新 Phase 5/6 状态；known boundaries 补 stop/limit/IOC/FOK、默认透支、I5 私有属性、板块涨跌停 |
| `README.md` §10 | 多时钟域改为「已实现 sub_steps」；事件队列仍未实现 |
| `USER_GUIDE.md` | 同步上述数字/语义；测试 369；订单表/唤醒表/写保护 |
| `docs/MULTIFACTOR.md` | 顶部加「+42.47% 已作废」声明；当前结果表改为 −78.55%；下一步改为 27.9× 换手 |
| `docs/PATH_200K.md` | 顶部加声明：24 股数字已作废，方法论批判仍成立 |
| `docs/PHASE6.md` | 顶部加门禁快照过期声明（280→369、84k→36k/8k、on_sub_step 已调度） |
| `reports/VALIDATION.md` | 顶部加 6b–6d 作废声明，指向 SIGNAL_TRANSMISSION.md |
| `reports/CAPABILITY.md` | 补注数据源现在也过滤 >30%，但 30% 仍高于板块上限 |
| `reports/SIGNAL_TRANSMISSION.md` | **新增**：修正版 100 只信号对照（含最低现金闸门） |

---

## 3. 修复后的验证结果

| 项 | 结果 |
|---|---|
| `pytest` | **369 passed**，exit 0 |
| `python -S -c "import framebt"` | 通过（核心确实 stdlib-only） |
| `validate_capability.py` 全量 | **8/10**；P1 = `0.000e+00`，P2 = `7.105e-15`，P3 逐位一致，P4 = 1136/1136 笔 0 不一致；P5a +1.05%；P5b 9.81×（FAIL，如实保留）；P7 +18.81%（FAIL）；P8 逃逸 0；P9 +0.08% |
| `validate_signal_transmission.py --max-symbols 100` | momentum −72.03% / reversal +305.39% / random +17.65% / 基准 +2.37%；最低现金 1/0/12，**无负现金** |
| 退市 | 100 股 @10（现金 -1000）+ delist@8 → 现金 **-200**（对价 800 已入账），权益 -200 |
| STOP | SELL STOP@9.5，open 8.8 → 成交 **8.8**（跳空按开盘，不按触发价） |
| LIMIT | BUY LIMIT@9.5，open 10 / low 9.2 → 成交 **9.5**（不再按 low 9.2） |
| IOC/FOK | volume=120：IOC 成交 120（余量取消），FOK 整单取消 |
| I5 | 策略 `bus.record / set_now / reset` 均抛 `CausalityViolation`；引擎自身注入正常 |

---

## 4. 仍未完成 / 明确保留的边界

1. **`VALIDATION.md` 的 6b–6c 未全量重跑**：全市场扫描是小时级；6d 已由新脚本取代。
   6b–6c 的数字保留但已在报告顶部标为历史记录。
2. **STOP/TRAILING 仍是日线近似**：用 high/low 假设「先更新有利极值、再检查不利极值」；
   未建模逐笔路径/盘口深度。LIMIT 也没建模跨价位成交量分配。
3. **`STOP_LIMIT` 只有一个 price 字段**，当前语义是「限价 = 触发价」；不是独立的
   stop price + limit price。
4. **默认 `NoOpRisk` 仍允许透支**：这是设计选择（方便纯账本测试），但 README/指南已明确
   警告；严肃回测必须显式传 `StandardRisk`。
5. **I5 只保护公开写 API**：Python 无法阻止 `bus._series` 这类私有属性访问。
6. **涨跌停单一 `limit`**：未内置板块识别（创业板/科创板 20%、北交所 30%）。
7. **存活偏差**：数据源不含退市股；5575/500 只样本仍不可外推。
8. **多因子系统当前 OOS 为负**（−78.55%）：这是策略层事实，不因文档修正而改变。

---

## 5. 复现命令

```powershell
# 核心零依赖
python -S -c "import sys; sys.path.insert(0,'src'); import framebt; print(framebt.__version__)"

# 全量测试（369）
$env:PYTHONPATH="src"; python -m pytest -q

# 端到端逐笔对账（约 80s；中文控制台不再崩）
$env:PYTHONPATH="src"; python -u tools/validate_capability.py

# 修正版信号对照（100 只，秒级）
$env:PYTHONPATH="src"; python -u tools/validate_signal_transmission.py --max-symbols 100

# 示例
$env:PYTHONPATH="src"; python examples/demo_quickstart.py
$env:PYTHONPATH="src"; python examples/multi_factor_universe.py --max-symbols 200
```
