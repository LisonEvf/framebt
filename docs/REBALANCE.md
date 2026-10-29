# 目标权重调仓（Rebalance）

> 代码：`src/framebt/rebalance.py` ｜ 宿主入口：`host.rebalance_to(...)` ｜ 测试：`tests/test_rebalance.py`

## 为什么需要它

资产配置 / 轮动 / 多因子组合的日常动作是「按目标权重调仓」。在此之前每个策略都要手写：

1. 用当前收盘价算权益；
2. 目标权重 → 目标股数（A 股 100 股整手）；
3. 先卖后买（卖出所得同日可用于买入）；
4. 按可用现金缩放买入腿、留出现金缓冲；
5. 过滤「为 0.1% 权重差而交易」的碎单。

这五步在多个策略与复刻脚本里各写了一遍，且很容易漏掉「先卖后买」或「现金约束」。
现在收成一个纯函数 + 宿主方法。

## 用法

```python
from framebt import RebalanceConfig

class MonthlyAllocation(FrameStrategy):
    def on_frame(self, frame, token, bus, host):
        if not self._is_rebalance_day(frame):
            return []
        targets = {"SH.510300": 0.6, "SH.511010": 0.4}   # 权重和 ≤ 1
        return host.rebalance_to(targets, frame)

    # 需要更严格/更宽松时：
    # host.rebalance_to(targets, frame, config=RebalanceConfig(
    #     lot=100, min_trade_pct=0.005, cash_buffer=0.002, enforce_cash=True))
```

不需要宿主时也可以直接调函数：

```python
from framebt import rebalance_orders, target_shares
orders = rebalance_orders(targets, frame, host, prices={"SH.510300": 4.21})
shares = target_shares(targets, equity=1_000_000, prices={"SH.510300": 4.21})
```

## 语义（重要）

| 项 | 行为 |
|---|---|
| 决策价 | 本帧收盘（或显式 `prices=`） |
| 成交 | 帧 t 下单、帧 t+1 开盘（I2） |
| 未列出的持仓 | 目标 0 → **会被清掉**（不是「保持不动」） |
| 权重和 | 必须 ≤ 1；>1、负数、NaN 抛 `ValueError` |
| 整手 | `floor(w·equity/price/lot)×lot`，默认 100 |
| 碎单门槛 | `|delta|·price < equity·min_trade_pct` 不发单，默认 0.5% |
| 现金缓冲 | 买入可用现金 = `cash + 卖出所得 − cash_buffer·equity`，默认 0.2% |
| 现金缩放 | 买入需求 > 可用现金时按比例缩放（`enforce_cash=True`，默认） |
| 顺序 | 返回列表先 SELL 后 BUY；引擎按列表顺序入账 |
| 缺价 | 目标无价 → 跳过；**持仓无价**且未传 `prices=` → 抛错（不猜价格） |
| 方向 | 只支持多头权重；做空/杠杆不在本原语范围 |

## 与「复刻历史口径」的关系

默认参数偏「实盘友好」（缓冲 + 碎单门槛 + 现金缩放），会改变逐笔成交。复刻旧脚本时用：

```python
cfg = RebalanceConfig(lot=100, min_trade_pct=0.0, cash_buffer=0.0, enforce_cash=False)
```

`RebalanceConfig(lot=100, min_trade_pct=0.0, cash_buffer=0.0, enforce_cash=False)`。

## 已知边界

- 不处理 T+1：卖出腿仍由 `Ledger` / `MarketRules` 检查可卖量；当天买入的仓位当天不能卖。
- 不处理停牌/缺 Bar：目标标的缺价会跳过；持仓缺价会抛错（需要调用方给 `prices=`）。
- 不保证「卖出全部成交」：市价单受当日成交量约束，部分成交时余量会被丢弃；
  本帧的买入仍按计划执行，可能透支；`cash_buffer` 只能缓解。
- 不是调度器：何时调仓（月末/季度/阈值）仍需策略自己判断。
