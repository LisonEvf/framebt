# 验证报告索引

> 所有报告都由脚本生成；**生成物**（HTML / CSV / 缓存）不入库，只保留可复现的 Markdown。
> 设计判据见对应 `docs/*.md`；没通过的条目保留在报告里，不删。

| 报告 | 类型 | 当前结论 | 复现 |
|---|---|---|---|
| [`CAPABILITY.md`](CAPABILITY.md) | 端到端逐笔对账 | P1–P4 精确判据零偏差；P5b/P7 未通过（如实保留） | `python -u tools/validate_capability.py --md reports/CAPABILITY.md` |
| [`VALIDATION.md`](VALIDATION.md) | 全市场逐项扫描 | 数据/I1–I5/规则/成本/账本/吞吐；6b–6d 已作废（顶部声明） | `python tools/validate_framework.py --md reports/VALIDATION.md` |
| [`ADAPTIVE.md`](ADAPTIVE.md) | 多因子系统 OOS | 当前样本外为负；不是框架正确性证明 | `python tools/validate_adaptive.py` |
| [`SIGNAL_TRANSMISSION.md`](SIGNAL_TRANSMISSION.md) | 信号传导对照 | 排序翻转 → 盈亏反转；含最低现金闸门 | `python tools/validate_signal_transmission.py --max-symbols 100` |
| [`USABILITY.md`](USABILITY.md) | 「真实可用」门禁 | L1/G1–G7 通过；7 条边界单列 | `python -u tools/gate_runner.py --md reports/USABILITY.md` |
| [`MUTATION.md`](MUTATION.md) | 门禁变异验证 | 15/15 注入 bug 被杀死；控制组 exit 0 | `python -u tools/mutate_gate.py --md reports/MUTATION.md` |
| [`EXTERNAL_REPLICATION.md`](EXTERNAL_REPLICATION.md) | 外部策略复刻 | L1 通过；L2 数据商 FAIL（QFQ vs Yahoo 回撤差）；L3 方向 4/4、幅度未复现 | `python -u tools/replicate_external.py --md reports/EXTERNAL_REPLICATION.md` |
| [`INTRADAY_SWEEP.md`](INTRADAY_SWEEP.md) | 日内 T 修正版 | 负结果（底仓+T −2.12%）；v1 已作废 | `python tools/sweep_intraday_v2.py --phase run`（需自建逐笔缓存） |

## 读报告时先看什么

1. **顶部声明**：是否有「作废/过期/样本外为负」；
2. **预登记**：判据是不是跑之前写的，失败项有没有保留；
3. **独立复算**：是引擎自己跟自己对，还是有第二套实现/外部数据；
4. **边界**：`xfail`/「未通过」/「未实现」比「全部通过」更值得看。

## 历史

修复前的审计与修复记录在 [`../docs/archive/`](../docs/archive/)，不在此目录。
