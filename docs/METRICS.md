# 绩效指标口径（Metrics）

> 代码：`src/framebt/metrics.py` ｜ 适配层：`Performance`（`core/result.py`）｜ 测试：`tests/test_metrics.py`

## 为什么单独定口径

2026-10-29 之前，指标散在三处且互相不一致：

| 位置 | 旧行为 | 问题 |
|---|---|---|
| `Performance.sharpe` | `年化收益 / 波动` | 不是标准 Sharpe；与外部记录对不上 |
| `Performance.total_return` | `末值 / 权益曲线首点 - 1` | 漏掉首帧盈亏（首帧若已成交，基准不是初始资金） |
| `Performance.volatility` | `std × √244` | 年化用 244、波动用 252 的脚本与报告混用 |
| `tools/*` | 各自实现 | 同一策略在不同脚本里 Sharpe 不同 |

现在所有入口（`Performance`、HTML 报告、复刻/审计工具）都调用 `framebt.metrics`，
**口径只改这一处**。

## 定义

| 指标 | 公式 | 说明 |
|---|---|---|
| 收益率 | `e_t / e_{t-1} - 1` | 简单收益（非对数） |
| 总收益 | `final / initial - 1` | `initial` = `BacktestResult.initial_cash`（含首帧盈亏） |
| 年化收益 CAGR | `(final / initial)^(1 / years) - 1`，`years = n_periods / ppy` | 复合年化 |
| 波动 | `std(r, ddof=1) × √ppy` | 年化 |
| Sharpe | `mean(r) / std(r) × √ppy` | 标准定义（`risk_free` 按年化传入，内部换算到单期） |
| Sortino | `mean(r) / downside_dev × √ppy`，`downside_dev = √mean(min(r,0)²)` | 只惩罚下行 |
| 最大回撤 | `max_t (peak_t - e_t) / peak_t` | 正数 |
| Calmar | `CAGR / max_drawdown` | 回撤为 0 时返回 0 |

`ppy`（`periods_per_year`）默认 **244**（A 股日线）。CAGR、波动、Sharpe 共用同一个 `ppy`——
这正是旧版最大的问题：年化按 244、波动按 252，Sharpe 自然对不上。

## 用法

```python
from framebt import Performance, summarize

# 1) 回测结果
perf = Performance(result)                    # 默认 ppy=244
print(perf.total_return, perf.sharpe, perf.sortino, perf.calmar)

# 2) 只要一组权益序列（研究/对账）
s = summarize(equity, initial=1_000_000, periods_per_year=252, n_periods=len(equity)-1)
```

`Performance.detailed()` 返回：`total_return / annualized_return / volatility / sharpe /
sharpe_cagr / sortino / max_drawdown / calmar / trades / frames / periods_per_year`。

## 迁移说明（2026-10-29）

- `Performance.sharpe`：`CAGR/vol` → **标准 `mean/std×√ppy`**。旧值保留为
  `Performance.sharpe_cagr`，仅用于和 2026-10-29 之前的报告对账。
- `total_return`：优先用 `BacktestResult.initial_cash`（Engine 自动写入）；手工构造
  `BacktestResult` 且未设 `initial_cash` 时，退回「权益曲线首点」。
- 新增 `sortino` / `calmar`；`volatility` 与 CAGR 统一用同一个 `ppy`。
- **旧报告里的 Sharpe / 波动不可直接与新报告比较**（`reports/` 下 2026-10-29 之前的文件）。

## 迁移示例（合成数据）

同一条权益曲线 `[100, 110, 99, 120]`、`periods_per_year=3`：

| 指标 | 旧口径 | 新口径 |
|---|---:|---:|
| 总收益 | 20% | 20% |
| 年化 | 20% | 20% |
| 波动 | 27.39% | 27.39% |
| Sharpe | `CAGR/vol` = 0.730 | `mean/std×√ppy` = 0.775 |

旧值可通过 `Performance.sharpe_cagr` 取到；新报告一律用标准 Sharpe。

## 边界

- `periods_per_year` 只影响年化类指标；总收益/最大回撤与它无关。
- 停牌/缺 Bar 的帧不会自动补收益；权益曲线只包含引擎实际推进的帧（静态帧也会计入）。
- `risk_free` 是**年化**值（如 0.02），内部按 `rf / ppy` 换算到单期。
