# tools

这些脚本是**研究/验证工具**，不随 `framebt` wheel 发布；运行前请设 `PYTHONPATH=src`。
生成物（HTML / CSV / 缓存 / `tools/report/`）已忽略，不入库；结果摘要保留在 `reports/`。

## 验证 / 门禁（推荐先跑）

| 脚本 | 用途 | 输出 |
|---|---|---|
| `validate_capability.py` | 端到端逐笔对账（低波动组合） | `reports/CAPABILITY.md` |
| `validate_framework.py` | 数据接入 / I1–I5 / 规则 / 成本 / 吞吐 | `reports/VALIDATION.md` |
| `gate_runner.py` | 「真实可用」门禁端到端 + 真实数据 | `reports/USABILITY.md` |
| `mutate_gate.py` | 门禁变异验证（15 个注入 bug） | `reports/MUTATION.md` |
| `replicate_external.py` | Faber / GEM / PP 外部策略复刻 | `reports/EXTERNAL_REPLICATION.md` |
| `validate_signal_transmission.py` | 排序翻转 → 盈亏反转 | `reports/SIGNAL_TRANSMISSION.md` |
| `validate_adaptive.py` | 多因子系统样本外验证 | `reports/ADAPTIVE.md` |

## 数据回填 / ETF 研究

| 脚本 | 依赖 | 说明 |
|---|---|---|
| `backfill_history.py` / `backfill_universe.py` | `opentdx`（可选） | 回填 Parquet 数仓 |
| `smoke_backfill.py` / `verify_backfill.py` | `opentdx` | 冒烟与校验 |
| `run_etf_trend.py` / `etf_robust.py` | `pandas` + 本地 `etf_cache/` | ETF 趋势与稳健性 |
| `etf_recon.py` | `opentdx` | 生成 `etf_universe.json` / `etf_candidates.json`（已忽略） |

## 研究脚本（可能需要额外数据/包）

| 脚本 | 说明 |
|---|---|
| `run_step1_investable.py` / `bench_oos_200k.py` | 20 万路径 / 样本外基准研究 |
| `ab_test_strategies.py` | 多标的策略 A/B |
| `sweep_intraday_v2.py` / `sweep_overlay_bt.py` | 扫单分时 T / fade 叠加；需要外部 `sweep_monitor`，用 `SWEEP_MONITOR_DIR` 指定路径 |
| `gen_report.py` | 报告生成辅助 |

## 约定

- 任何「验证/能力」工作先写 `docs/` 下的预登记，再跑；失败项保留在报告里。
- 不把本机绝对路径写进脚本：用 `Path(__file__)` / 环境变量（如 `FRAMEBT_DATA_DIR`、
  `SWEEP_MONITOR_DIR`）。
- 已知不可运行/已废弃的脚本直接删除，不留在主分支；需要时从 git 历史取。
