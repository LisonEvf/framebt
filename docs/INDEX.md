# 文档地图

> 最后整理：2026-10-20。原则：**设计/预登记、当前验证报告、历史归档**分开；
> 生成物（HTML / CSV / 缓存）不入库。

## 入口

| 文件 | 用途 |
|---|---|
| [`../README.md`](../README.md) | 项目总览、能力清单、已知边界、快速开始 |
| [`../USER_GUIDE.md`](../USER_GUIDE.md) | 从安装到第一个回测、订单/规则/成本的实操说明 |

## 设计与预登记（先写判据，再跑数）

| 文件 | 内容 | 状态 |
|---|---|---|
| [`MULTIFACTOR.md`](MULTIFACTOR.md) | 自适应多因子系统落地说明与实现边界 | 当前 |
| [`ETF_TREND.md`](ETF_TREND.md) | 多资产 ETF 趋势跟踪预登记 + 结果（主策略未达标，等权多资产可用） | 当前 |
| [`PATH_200K.md`](PATH_200K.md) | 20 万本金路径的方法论批判（24 股时代数字已作废，批判仍成立） | 当前（顶部有声明） |
| [`PHASE6.md`](PHASE6.md) | 帧内执行 / 实盘 / 做空 / 双时钟域设计快照 | 设计快照（数字可能过期） |
| [`INTRADAY_T_TUTORIAL.md`](INTRADAY_T_TUTORIAL.md) | 扫单分时 T 策略技术手册；第 11 节收益结果已作废 | 教程（结果以 `reports/INTRADAY_SWEEP.md` 为准） |
| [`PREREG_CAPABILITY.md`](PREREG_CAPABILITY.md) | 低波动组合逐笔对账的预登记 | 已执行 → `reports/CAPABILITY.md` |
| [`GATE_USABILITY.md`](GATE_USABILITY.md) | 「真实可用」门禁预登记（G1–G8） | 已执行 → `reports/USABILITY.md` |
| [`REPLICATION.md`](REPLICATION.md) | 外部策略复刻预登记（Faber / GEM / PP） | 已执行 → `reports/EXTERNAL_REPLICATION.md` |

## 验证报告

统一入口：[`../reports/INDEX.md`](../reports/INDEX.md)。

| 报告 | 回答的问题 |
|---|---|
| [`CAPABILITY.md`](../reports/CAPABILITY.md) | 一个真实策略的账本/成本/成交能否被独立复算到 1e-9 |
| [`VALIDATION.md`](../reports/VALIDATION.md) | 全市场数据接入、I1–I5、规则、成本、吞吐的逐项扫描 |
| [`ADAPTIVE.md`](../reports/ADAPTIVE.md) | 自适应多因子系统的样本外结果（当前为负，如实记录） |
| [`SIGNAL_TRANSMISSION.md`](../reports/SIGNAL_TRANSMISSION.md) | 翻转排序方向后盈亏是否跟着反转（修正版） |
| [`USABILITY.md`](../reports/USABILITY.md) | 「真实可用」门禁：金标准 / oracle / 因果 / 元形 / 真实数据 |
| [`MUTATION.md`](../reports/MUTATION.md) | 门禁是否有牙：15 个注入 bug 的杀死记录 |
| [`EXTERNAL_REPLICATION.md`](../reports/EXTERNAL_REPLICATION.md) | 公开策略复刻的三层对账（独立实现 / Yahoo / 公开结果） |
| [`INTRADAY_SWEEP.md`](../reports/INTRADAY_SWEEP.md) | 扫单分时 T 修正版（负结果） |

## 历史归档

| 文件 | 说明 |
|---|---|
| [`archive/AUDIT_README_CLAIMS.md`](archive/AUDIT_README_CLAIMS.md) | 2026-10-20 修复前的独立审计快照（保留原始证据） |
| [`archive/REMEDIATION.md`](archive/REMEDIATION.md) | 对应的修复记录与未完成边界 |

归档文件是**历史快照**，不再更新；当前状态以 README、`reports/` 与代码为准。

## 建议阅读顺序

1. 新用户：`README.md` → `USER_GUIDE.md` → `examples/demo_quickstart.py`；
2. 想评估可信度：`reports/CAPABILITY.md` → `reports/USABILITY.md` → `reports/MUTATION.md`；
3. 想复刻外部策略：`docs/REPLICATION.md` → `reports/EXTERNAL_REPLICATION.md`；
4. 想知道坑在哪：README「Known boundaries」→ `reports/USABILITY.md` §5 → `docs/archive/REMEDIATION.md` §4。

## 维护约定

- **预登记先写**：任何「验证/能力」类工作先写 `docs/*PREREG*` / `docs/GATE_*` / `docs/REPLICATION*`，再跑数；
- **生成物不入库**：`reports/*.html`、`*.csv`、`tools/*_cache/`、`tools/report/` 已加入 `.gitignore`；
- **失败不删**：预登记里没通过的条目保留在报告里（例如 CAPABILITY 的 P5b/P7、REPLICATION 的 L2）；
- **历史不丢**：过期/过程性文档移入 `docs/archive/`，不直接删除。
