# framebt 技术手册：扫单分时 T 策略实现

> ## ⛔ 本文第 11 节的收益结果已作废
>
> 本文的**策略逻辑讲解**（信号 → 开仓 → 平仓 → 成本 → 仓位 → 择时 → P&L 聚合）仍然有效，
> 但第 11 节的「底仓 +0.68% → +3.77%、1666 笔、72% 胜率、Sharpe 0.81」**不得引用**。
> 2026-10-20 复核发现三处致命缺陷（逐笔页序倒置未排序 / 大盘择时门控从未生效 / 高价股名义
> 放大 13 倍），修正后重跑用同一池、同一区间、同一信号参数得到的是**负结果**：
> 2990 笔、净 −42,155 元、底仓+T 累计 −2.12%。详见
> [`reports/INTRADAY_SWEEP.md`](../reports/INTRADAY_SWEEP.md) 与
> `tools/sweep_intraday_v2.py`。
>
> 本文是**技术使用手册**，不是概念介绍。目标：把「底仓 + 日内 T」这套策略从**信号因子计算 → 开仓 → 平仓 → 成本 → 仓位 → 择时 → P&L 聚合 → 因果性**每一层讲清楚，并给出带注释的实现代码。
>
> 配套源码：`tools/sweep_intraday_v2.py`（修正版；需要外部 `sweep_monitor`，
> 用 `SWEEP_MONITOR_DIR` 指定路径）、`sweep_monitor/sweep_monitor/engine/detection.py`（信号识别）。
> 旧 v1 `sweep_intraday_bt.py` 因逐笔页序/门控缺陷已从仓库移除（git 历史可查）。
> 配套结果：[`reports/INTRADAY_SWEEP.md`](../reports/INTRADAY_SWEEP.md)（现行）。

---

## 目录

- [1. 策略总览：两阶段架构与数据流](#1-策略总览两阶段架构与数据流)
- [2. 帧内信号因子：`detect_sweeps` 怎么算](#2-帧内信号感 detect_sweeps-怎么算)
- [3. 怎么算买入 / 卖出：开仓逻辑](#3-怎么算买入--卖出开仓逻辑)
- [4. 怎么算平仓：退出逻辑](#4-怎么算平仓退出逻辑)
- [5. 成本模型：`roundtrip_cost`](#5-成本模型roundtrip_cost)
- [6. 仓位计算：`target / N / 0.25`](#6-仓位计算target--n--025)
- [7. 大盘择时：`MARKET_GATE`](#7-大盘择时marketsgate)
- [8. 底仓 + 战术：P&L 叠加聚合](#8-底仓--战术p-l-叠加聚合)
- [9. 因果性：为什么这个结果可信](#9-因果性为什么这个结果可信)
- [10. 在 framebt 引擎里重写这个策略](#10-在-framebt-引擎里重写这个策略)
- [11. 参数表与调参建议](#11-参数表与调参建议)

---

## 1. 策略总览：两阶段架构与数据流

整个回测由**两个叠加的组合**构成：

```
组合收益 = 底仓收益（等权 buy&hold） + 战术收益（分时 T，日内）
```

```
┌──────────────────────────────────────────────────────────────────┐
│  阶段 A：底仓（Beta）                                              │
│  25 只大盘股等权买入持有，日收益 = 当日末笔价 / 昨末笔价 − 1，      │
│  等权平均 → base_ret[日]                                          │
├──────────────────────────────────────────────────────────────────┤
│  阶段 B：战术（Alpha，日内 T）                                     │
│  每标的当日逐笔 → 5min 滚动窗口 → 每 2 笔调 detect_sweeps          │
│  → 触发即开仓 → 回落/反弹 ≥0.4% 平仓 → 叠加到底仓上               │
└──────────────────────────────────────────────────────────────────┘
                        │
                        ▼
  comb[日] = base_ret[日] + Σ战术盈亏[日] / ACCOUNT
```

**逐日收益序列**（`run_day_intraday` 返回 `[(side, shares, entry, exit, pnl_net), ...]`）：

```python
base = np.array([base_ret[d] for d in days])                       # 底仓日收益
tact = np.array([per_day[d] / ACCOUNT for d in days])              # 战术日盈亏 → 收益率
comb = base + tact                                                 # 叠加
```

> 关键点：**战术仓每日净份额归零**（倒T 卖底仓当日买回、正T 买后当日卖底仓份额），所以战术盈亏**纯叠加**，不与底仓混仓。这是 `组合收益 = 底仓 + 战术` 成立的前提。

---

## 2. 帧内信号因子：`detect_sweeps` 怎么算

「帧内信号因子」= 在**当前滚动窗口**内识别出的扫单事件。这是整套策略的**唯一信号源**。

### 2.1 输入契约

`detect_sweeps(trades, ...)` 要求输入**按时间升序排列的 `Trade` 列表**，每个 `Trade` 必须有 5 个属性：

| 属性 | 含义 | 本例 `Tick` 的对应 |
|---|---|---|
| `.direction` | `"BUY"` / `"SELL"` / `"NEUTRAL"` | 成交方向（`bs_flag` 转换） |
| `.price` | 成交价 | |
| `.vol` | 成交量（股） | |
| `.amount` | 成交额（元） | |
| `.time` | `datetime` 时间戳 | |

> 中性笔（`bs_flag=2`，无法判定买卖方向）**必须过滤**：`detect_sweeps` 不处理 `NEUTRAL`，会触发 `SweepEvent` 校验失败。
> ```python
> ticks = [t for t in ticks if t.direction != "NEUTRAL"]
> ```

### 2.2 核心算法：按方向切 run，逐个判定扫单类型

```python
def detect_sweeps(trades, *, min_legs, min_levels, min_trend_steps,
                  volume_tolerance, max_burst_seconds=2.0, max_run_gap_seconds=600.0):
    # ① 按「同方向连续」切分最大 run；相邻两笔间隔 > max_run_gap（如午休 2.5h）即断开，
    #    避免把跨休市的伪连续当成一次扫单。
    i, n, events = 0, len(trades), []
    while i < n:
        direction = trades[i].direction
        j = i
        while j < n:
            if j > i and (trades[j].time - trades[j-1].time) > timedelta(seconds=max_run_gap_seconds):
                break
            if trades[j].direction != direction:
                break
            j += 1
        run = trades[i:j]
        i = j
        if len(run) < min_legs:
            continue

        # ② 对每个 run 按优先级判定扫单类型：multi_level > identical_volume > burst
        ok, levels = _is_multi_level(run, direction, min_levels, min_trend_steps)
        if ok:
            events.append(_make_event(run, "multi_level", levels)); continue
        if _has_identical_volume(run, min_legs, volume_tolerance):
            events.append(_make_event(run, "identical_volume", ...)); continue
        if _is_burst(run, min_legs, max_burst_seconds):
            events.append(_make_event(run, "burst", 1))
    return events
```

### 2.3 三种扫单形态

| 形态 | 判定条件 | 含义 |
|---|---|---|
| **multi_level** | 价格沿攻击方向单调，且跨 ≥ `min_levels` 档、进步步数 ≥ `min_trend_steps` | 单方向连续多档吃单（最可信） |
| **identical_volume** | ≥ `min_legs` 笔连续、成交量相同（容差 `volume_tolerance`） | 程序化拆单（如 TWAP） |
| **burst** | `min_legs` 笔在 ≤ `max_burst_seconds`（2s）内 | 高频扫单 |

- **攻击方向**：`BUY` → 价格上行（`+1`），`SELL` → 价格下行（`-1`）。
- **扫单强度**（可比较相对强弱）：

  ```
  intensity = legs × price_levels × amount(万元)
  ```

### 2.4 本例的「帧」= 滚动窗口

策略不把逐笔喂给一个固定 Δt 帧，而是用一个**定长 deque** 当滚动窗口：

```python
win = deque(maxlen=max(15, int(WIN_MIN * 4)))   # WIN_MIN=5 → maxlen=20 笔
for i, tk in enumerate(ticks):
    win.append(tk)
    if i % STEP != 0:          # STEP=2：每 2 笔才调一次 detect_sweeps（省算力）
        continue
    events = detect_sweeps(list(win), min_legs=..., min_levels=..., ...)
```

> 「帧内信号因子」在这里 = **最近 20 笔内的扫单事件集合**。`ev.direction` 即信号方向：`BUY` 扫 = 价格被拉高（过度反应）→ 触发倒T；`SELL` 扫 = 砸盘（超卖）→ 触发正T。

---

## 3. 怎么算买入 / 卖出：开仓逻辑

开仓由**扫单事件方向**决定。两种 T 的「买/卖」方向相反：

| 策略 | 触发扫单 | 动作 | 盈利来源 |
|---|---|---|---|
| **倒T（INV）** | `BUY` 扫（价格被拉高） | **先卖底仓**（高），回落买回 | 超买回归 |
| **正T（POS）** | `SELL` 扫（砸盘） | **先买**（低），反弹卖底仓 | 超卖反弹 |

```python
for ev in events:
    # ① 每日每方向最多 1 笔（traded_dir 去重）；② 已有持仓则跳过；③ 持仓超时长则不新开
    if open_pos is not None or (ev.direction in traded_dir):
        continue
    if (tk.t - t0) > timedelta(minutes=MAX_HOLD_MIN):
        continue

    if ev.direction == "BUY" and DO_INV:                 # 买扫 → 倒T（类空头）
        traded_dir.add("BUY")
        target = ACCOUNT / N * 0.25                      # 目标名义 = 账户 / 标的数 / 4
        shares = int(target / tk.price // 100 * 100)     # 取整到 100 股一手
        cap = int(target * 2 / tk.price // 100 * 100)    # 高价股一手即超目标 → 2× 封顶
        shares = max(100, min(shares, cap) if cap >= 100 else 100)
        open_pos = {"side": "INV", "shares": shares, "entry_price": tk.price}
    elif ev.direction == "SELL" and DO_POS:              # 卖扫 → 正T（类多头）
        traded_dir.add("SELL")
        target = ACCOUNT / N * 0.25
        shares = int(target / tk.price // 100 * 100)
        cap = int(target * 2 / tk.price // 100 * 100)
        shares = max(100, min(shares, cap) if cap >= 100 else 100)
        open_pos = {"side": "POS", "shares": shares, "entry_price": tk.price}
```

**要点**：
- **卖底仓 = 倒T 的「卖」**（散户无法做空，但卖出已持有的隔夜底仓等效类空头）。
- **买 = 正T 的「买」**（低接）。
- **每日每方向最多 1 笔**：`traded_dir` 集合保证一天内不会反复开同一个方向的仓。
- **当前价成交**：`entry_price = tk.price`（当前 tick 价）——这是可执行的因果价格，非未来价。

---

## 4. 怎么算平仓：退出逻辑

平仓有两条路径：**阈值触发** 或 **时间强平**（取先者）。

```python
if open_pos is not None:
    elapsed = (tk.t - t0).total_seconds() / 60.0
    ep, sh = open_pos["entry_price"], open_pos["shares"]
    exit_price = None

    # ① 阈值触发：倒T 跌 ≥TH_INV 买回；正T 涨 ≥TH_POS 卖
    if open_pos["side"] == "INV" and tk.price <= ep * (1 - TH_INV):
        exit_price = tk.price * (1 - TICK_SLIP)
    elif open_pos["side"] == "POS" and tk.price >= ep * (1 + TH_POS):
        exit_price = tk.price * (1 + TICK_SLIP)

    # ② 时间强平：持仓达 MAX_HOLD_MIN 分钟，未达阈值也强平
    if exit_price is None and elapsed >= MAX_HOLD_MIN:
        exit_price = tk.price * (1 - TICK_SLIP if open_pos["side"] == "INV" else 1 + TICK_SLIP)

    # ③ 成交：计算净盈亏
    if exit_price is not None:
        # 倒T 盈利 = (买入价−卖出价)×股；正T 盈利 = (卖出价−买入价)×股
        gross = (ep - exit_price) * sh if open_pos["side"] == "INV" else (exit_price - ep) * sh
        notional = max(ep, exit_price) * sh
        pnl = gross - roundtrip_cost(notional, notional)     # 减去双边成本
        trades_out.append((open_pos["side"], sh, ep, exit_price, pnl))
        open_pos = None                                      # 一日最多 1 笔
```

**参数**：`TH_INV = TH_POS = 0.004`（0.4% 回调即平仓）、`MAX_HOLD_MIN = 40`（分钟）、`TICK_SLIP = 0.0003`（滑点）。

---

## 5. 成本模型：`roundtrip_cost`

单笔往返成本 = 买入侧成本 + 卖出侧成本。A 股成本不对称：**印花税仅卖出收 0.1%**。

```python
def roundtrip_cost(buy_notion, sell_notion):
    buy_c  = max(5.0, buy_notion * 0.0003) + buy_notion * 0.00002   # 佣金(起5) + 过户费 0.002%
    sell_c = max(5.0, sell_notion * 0.0003) + sell_notion * 0.001   # 佣金(起5) + 印花税 0.1%
    return buy_c + sell_c
```

| 成本项 | 费率 | 说明 |
|---|---|---|
| 佣金 | 0.03%（最低 5 元） | 买卖双边 |
| 过户费 | 0.002% | 买卖双边 |
| 印花税 | **0.1%** | **仅卖出**（A 股特色，是最大单笔成本） |
| 滑点 | 0.03% | 成交偏移，倒T/正T 方向不同 |

> 单笔名义 1 万时，往返成本 ≈ 5 + 3 + 5 + 10 = **≈23 元（0.23%）**。阈值 0.4% 的盈利需先覆盖这 0.23% 才能转正——这是单笔净利薄的根源。

---

## 6. 仓位计算：`target / N / 0.25`

```python
target = ACCOUNT / N * 0.25          # 1,000,000 / 25 / 4 = 10,000 元/笔
shares = int(target / tk.price // 100 * 100)   # 取整到 100 股
cap  = int(target * 2 / tk.price // 100 * 100) # 2× 封顶
shares = max(100, min(shares, cap) if cap >= 100 else 100)
```

- **目标名义** = `账户 / 标的数 / 4`：保证 25 只各 25% 用仓时总敞口 ≤ 100%。
- **取整到 100 股**（A 股一手=100 股）。
- **2× 封顶**：高价股（如茅台 1200 元）一手即 12 万，远超 1 万目标 → 用 `target*2` 封顶，避免少数高价股把单笔名义放大到目标的数十倍。
- **下限 100 股**：保证每笔至少有 1 手。

---

## 7. 大盘择时：`MARKET_GATE`

用指数（`SH000001`）相对 MA20 的位置，决定**哪个方向允许开仓**：

```python
# 计算强市日：当日指数收盘 > 其 MA20
gate_days = set()
for i in range(19, len(dd)):
    if idx_close[dd[i]] > np.mean([idx_close[dd[j]] for j in range(i-19, i+1)]):
        gate_days.add(dd[i])

for (side, sh, ep, ex, pnl) in res:
    # 倒T(超买 fade) 仅在弱市(index<MA20)做：强市买盘进攻多为突破，fade 易被埋
    if side == "INV" and MARKET_GATE and (day in gate_days):
        continue
    # 正T(买 dip) 仅在强市做：下跌趋势不接飞刀
    if side == "POS" and MARKET_GATE and (day not in gate_days):
        continue
    out.append(...)
```

- **倒T（INV）**：弱市做。超买 fade 在弱市更可信（强市买盘多为真突破）。
- **正T（POS）**：强市做。下跌趋势接飞刀风险高。
- 本案例窗口（2026-06~10）全弱市 → 倒T 全触发、正T 被正确规避（0 笔）。

---

## 8. 底仓 + 战术：P&L 叠加聚合

```python
# 底仓：等权 buy&hold，日收益 = 当日末笔价 / 昨末笔价 − 1，等权平均
base_ret = {}
for s in SYMS:
    ds = sorted(closes[s])
    for i in range(1, len(ds)):
        d0, d1 = ds[i-1], ds[i]
        base_ret[d1] = base_ret.get(d1, 0.0) + closes[s][d1] / closes[s][d0] - 1
base_ret = {d: v / N for d, v in base_ret.items()}          # 等权平均

# 战术：按日聚合每笔盈亏
per_day = {}
for (day, s, side, sh, ep, ex, pnl) in all_trades:
    per_day[day] = per_day.get(day, 0.0) + pnl

# 叠加：组合日收益 = 底仓 + 战术/账户
base  = np.array([base_ret[d] for d in days])
tact  = np.array([per_day.get(d, 0.0) / ACCOUNT for d in days])
comb  = base + tact
```

**绩效统计**（`stats`）：

```python
cr   = float(np.prod(1 + r) - 1)                             # 累计收益（连乘）
ann  = cr / (len(r) / 252)                                   # 年化（252 交易日）
sh   = (r.mean() / (r.std() + 1e-12)) * np.sqrt(252)         # 夏普
dd   = float(np.min(np.prod(1 + r) - 1))                     # 最大回撤
hit  = float(np.mean(r[r != 0] > 0))                         # 胜率
```

---

## 9. 因果性：为什么这个结果可信

这套回测的因果性由**构造保证**，三条铁律：

| 原则 | 实现 | 防的是什么 |
|---|---|---|
| **tick 升序处理** | `for i, tk in enumerate(ticks)`，`win.append(tk)` 后检测 | 信号只用当前及之前成交 |
| **当前价成交** | `entry_price = tk.price`、`exit_price = tk.price` | 不用未来价、不以「收盘瞬间价」成交 |
| **窗口定长** | `deque(maxlen=20)` | 旧笔自然滑出，不偷看窗口外 |

> 这与 framebt 引擎的不变式 **I1（数据不过当前虚拟时间）**、**I2（订单有延迟）** 同源：前者是**具体策略的因果自守**，后者是**通用引擎的强制约束**。

---

## 10. 在 framebt 引擎里重写这个策略

`sweep_intraday_v2.py` 是**独立脚本**（直接用 `QuoteSource` 拉逐笔、自己循环）。若要在 **framebt 引擎**里表达同一套逻辑，核心是把它**映射到 `on_frame` 回调**。

> **粒度差异（重要）**：框架的 `bus.get(key, symbol, window=token=token)` 返回的是 **`list[float]` 标量因子序列**（如 `close`、`volume`），而扫单信号需要**带方向的逐笔成交**（`.direction/.vol/.amount`）。`Bar` 没有 `.direction`——逐笔在框架里属于 `Tick` 事件（高频 / 帧内事件队列，Phase 6）。因此把本策略搬进引擎，需先**把逐笔按自定义 key 注册进总线**（或喂 `Tick` 唤醒），再在回调里对窗口内逐笔跑 `detect_sweeps`。下面用注释标出这一步。

### 10.1 回调骨架（正确的接口）

```python
from framebt import FrameStrategy, Order, OrderSide, OrderType
from collections import deque

class SweepIntradayT(FrameStrategy):
    def __init__(self, symbol, *, win=20, step=2, th=0.004, max_hold_min=40,
                target_notional=10_000.0, do_inv=True, do_pos=True):
        super().__init__(symbol)
        self.win = deque(maxlen=win)          # 滚动窗口（帧内）
        self.step, self.th, self.max_hold = step, th, max_hold_min
        self.target = target_notional
        self.do_inv, self.do_pos = do_inv, do_pos
        self.open_pos = None
        self.traded_dir = set()
        self.t0 = None

    def on_frame(self, frame, token, bus, host):
        # ① 取窗口内逐笔（须已按自定义 key 注册；否则 bus.get 只给标量因子）
        trades = bus.get("trade", self.symbol, window=self.win.maxlen, token=token)
        #    token 是引擎签发的 CausalToken：无 token 直接抛 CausalityViolation（I1 的强制点）
        if not trades:
            return []
        self.win.extend(t for t in trades if t.direction != "NEUTRAL")   # 过滤中性笔
        if self.t0 is None:
            self.t0 = frame.time

        # ② 每 step 笔调一次 detect_sweeps（信号因子，同 §2）
        if (len(self.win) % self.step) != 0:
            return []
        events = detect_sweeps(list(self.win), min_legs=..., min_levels=..., ...)

        # ③ 开仓（同 §3）：每日每方向最多 1 笔
        if self.open_pos is None:
            for ev in events:
                if ev.direction in self.traded_dir:
                    continue
                elapsed_min = (frame.time - self.t0).total_seconds() / 60
                if elapsed_min > self.max_hold:
                    break
                if ev.direction == "BUY" and self.do_inv:       # 买扫 → 倒T（类空头）
                    self.traded_dir.add("BUY")
                    self._open("INV", ev, frame)
                elif ev.direction == "SELL" and self.do_pos:    # 卖扫 → 正T（类多头）
                    self.traded_dir.add("SELL")
                    self._open("POS", ev, frame)

        # ④ 平仓（同 §4）：阈值或时间强平
        if self.open_pos is not None:
            ep, sh = self.open_pos["entry_price"], self.open_pos["shares"]
            if self.open_pos["side"] == "INV" and frame.close <= ep * (1 - self.th):
                return [Order(OrderSide.BUY,  self.symbol, self._sh, order_type=OrderType.MARKET)]
            if self.open_pos["side"] == "POS" and frame.close >= ep * (1 + self.th):
                return [Order(OrderSide.SELL, self.symbol, self._sh, order_type=OrderType.MARKET)]
        return []
```

> `_open(...)`、`self._sh` 是 §3（开仓取整）/ §6（仓位计算）逻辑的**简写**，此处为骨架示意，非可运行完整实现。

### 10.2 映射关系

| sweep_intraday_v2.py | framebt 引擎 |
|---|---|
| `for tk in ticks` 循环 | `on_frame` 逐帧唤醒 |
| `deque(maxlen=20)` 滚动窗口 | `bus.get(..., window=20, token=token)` 因果数据 |
| `detect_sweeps(win)` | 帧内信号因子（同 §2） |
| `tk.price` 成交价 | `frame.close`（经延迟管线 I2，延迟一帧成交） |
| 手动 P&L 聚合 | `ledger` 账本 + `Performance` |
| 手动 `roundtrip_cost` | `StandardCost` 成本模型 |
| 手动 `MARKET_GATE` | 策略内大盘择时或横截面 |

> 引擎替你做了 §9 的因果性（I1/I2/I4）和 §5 的成本（`StandardCost`），你只需把 §2–§4 的信号与开平仓逻辑写进 `on_frame`。`Order` 经**延迟管线**成交（默认 `placement_frame = t+1`），即不变式 I2——帧 t 下单、帧 t+1 才成交，物理上不可能用当前价成交。

---

## 11. 参数表与调参建议

| 参数 | 默认 | 含义 | 调参方向 |
|---|---|---|---|
| `WIN_MIN` | 5 | 滚动窗口（分钟等价） | 窗口越大信号越平滑但越迟钝 |
| `STEP` | 2 | 每 2 笔调一次检测 | 越大越省算力 |
| `TH_INV`/`TH_POS` | 0.004 | 平仓阈值（0.4%） | 越大越难触发、单笔利润越高 |
| `MAX_HOLD_MIN` | 40 | 最长持仓（分钟） | 越大越倾向「过夜」，违背日内 T |
| `TICK_SLIP` | 0.0003 | 滑点（0.03%） | 贴近真实成交偏移 |
| `ACCOUNT / N / 0.25` | 10,000 元 | 单笔目标名义 | 用仓比例 |
| `MARKET_GATE` | True | 大盘择时 | 弱市做倒T、强市做正T |

**当前窗口结果**（2026-06-03 ~ 2026-10-16，25 只大盘股）：

> ⛔ **下表已作废**：该结果跑在「逐笔页序倒置」的数据上（见文首声明）。修正后同一池、同一区间、
> 同一信号参数的结果是 **倒T 1550 笔 + 正T 1440 笔、净 −42,155 元、底仓+T 累计 −2.12%**。

| 组合 | 累计 | 年化 | Sharpe | 最大回撤 |
|---|---|---|---|---|
| ~~底仓等权 BH~~ | +0.68% | +1.9% | 0.21 | +0.7% |
| ~~底仓 + 分时 T~~ | **~~+3.77%~~** | ~~+10.5%~~ | ~~0.81~~ | ~~+3.8%~~ |

- 倒T ~~1666 笔、72% 胜率、净 +30,845 元、22/25 标的盈利~~（作废）。
- 净增 ~~+3.09pp~~（作废）；修正后为 **−4.13pp**（战术腿单独）。

**诚实边界**：单窗口、参数未网格、92 交易日——这是**工程证据**而非统计有效性证明。
修正后重跑显示毛盈亏 +769 元、成本 42,924 元（成本是毛利的 56 倍），
即**这个信号在扣费前的毛 edge 约等于 0**；左尾（封涨停后被迫买回）才是亏损主因。
下一步若继续，应先给倒T 加硬止损、把 `min_levels` 换成 bp 归一化，再谈样本外。

---

### 运行

```bash
cd framebt
# 修正版：先建逐笔缓存（约 4 分钟），再跑全部变体
# 需要外部 sweep_monitor：set SWEEP_MONITOR_DIR=...  （旧 v1 已移除，git 历史可查）
python tools/sweep_intraday_v2.py --phase fetch
python tools/sweep_intraday_v2.py --phase run
# 输出：底仓 +2.09% / 底仓+分时T −2.12%；倒T 1550 笔 净−21,563元 正T 1440 笔 净−20,592元
# 生成：reports/INTRADAY_SWEEP.md（逐笔 CSV 为本地中间产物，*.csv 已忽略、不入库）
```
