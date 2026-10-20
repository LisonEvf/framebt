# framebt「真实可用」门禁报告

> 生成：2026-10-20 23:22:38 ｜ 用时 9.0s ｜ 预登记：[`docs/GATE_USABILITY.md`](../docs/GATE_USABILITY.md)

## 0. 结论

**核心能力真实可用（预登记门禁通过；7 条建模边界见 §5，不构成生产就绪）**

核心门禁 8/8 项通过；门禁本体见 `tests/test_capability_gate.py`（预登记 7 条 strict xfail 为已登记边界，不计通过）；变异结果见 [`MUTATION.md`](MUTATION.md)。

> 本报告只回答「框架算得对不对、挡不挡得住、可不可复现、改坏会不会被抓」，**不回答任何策略是否赚钱**。

## 1. 结果总表

| 项 | 判据 | 结果 | 证据 |
|---|---|---|---|
| G6a 纯标准库 import + 微型回测 | python -S 下 exit 0 | **PASS** | OK 1.0.0 1 1000.0 |
| G1-G5 pytest 门禁（金标准/oracle/因果/元形/不变量） | exit 0，除 7 条 strict xfail 外 0 failed | **PASS** | exit=0 |
| G6b GBK 控制台下 validate_capability.py --quick | exit 0 且写出报告 | **PASS** | exit=0；报告 16297B；用时 5.5s（GBK 控制台未崩） |
| G6c 3,000 帧吞吐 | > 2,000 bars/s 且 < 10s | **PASS** | 5,859 bars/s |
| G7a 真实数据体检（5 标的 × ≤400 根） | NaN/非正价/OHLC 矛盾/异常日收益/重复时间戳 均为 0 | **PASS** | SH.600519:0；SZ.000001:0；SH.601318:0；SH.601899:0；SH.600036:0（合计 0） |
| G7b 真实单标的端到端重放（SH.600519，400 根） | 逐帧权益 + 逐笔成本 1e-9 一致 | **PASS** | 2 笔成交；期末 92,858.20；收益 -7.14%；回撤 18.91%；现金偏差 +0.000e+00 |
| G7c 真实多标的端到端重放（5 标的 × 250 帧） | 逐帧权益 + 逐笔成本 1e-9 一致 | **PASS** | 5 笔成交；期末 999,099.79；收益 -0.09%；回撤 2.50%；现金偏差 +0.000e+00 |
| G8 变异杀死率（tools/mutate_gate.py） | 15/15 且控制组通过 | **PASS** | 见 reports/MUTATION.md |

## 2. 环境

- Python：3.12.10
- 平台：nt / win32
- 数据源：`C:\Users\Lison\Desktop\EvfWorkSpace\TRADE\opentdx\db\data`
- 标的：SH.600519, SZ.000001, SH.601318, SH.601899, SH.600036
- 命令：`python -u tools/gate_runner.py --md reports/USABILITY.md`；变异 `python -u tools/mutate_gate.py --md reports/MUTATION.md`

## 3. 真实数据体检

对每只标的最近 ≤400 根 QFQ 日线检查：NaN、非正价、OHLC 区间矛盾、|日收益|>31%、重复 `(timestamp, symbol)`。

| 标的 | NaN | 非正价 | OHLC 矛盾 | 异常日收益 | 重复 |
|---|---|---|---|---|---|
| SH.600519 | 0 | 0 | 0 | 0 | 0 |
| SZ.000001 | 0 | 0 | 0 | 0 | 0 |
| SH.601318 | 0 | 0 | 0 | 0 | 0 |
| SH.601899 | 0 | 0 | 0 | 0 | 0 |
| SH.600036 | 0 | 0 | 0 | 0 | 0 |

被剔除的候选（有效 Bar < 100，数据覆盖/复权问题）：`SZ.300750`（1 根）

## 4. 独立重放口径

- 成本：`max(名义×3bp, 5) + 卖 5bp 印花 + 0.2bp 过户 + 名义×5bp×(量/当帧量)`；
- 权益：`现金 + Σ(净持仓 × 当帧收盘)`；静态帧用最后收盘；
- 重放不使用 `Ledger` / `StandardCost` 的实现，只用 `FillEvent` 公开字段；
- 判定阈值：相对/绝对偏差 ≤ 1e-9。

## 5. 预登记边界（未修复，不计入 PASS）

门禁用 `xfail(strict=True)` 钉住 B1–B6；B7–B9 是代码审查确认的边界，未单独写门禁测试。修好后 strict xfail 会 XPASS 并提醒更新判据：

| # | 边界 | 影响 |
|---|---|---|
| B1 | `volume=0` 的 Bar 上市价单不受成交量约束 | 停牌/无成交日可无限成交 |
| B2 | NaN 行情/数量被静默接受 | 权益变 NaN，业绩不可读 |
| B3 | 同帧多笔市价单不共享当帧成交量 | 总成交可超 Bar 成交量 |
| B4 | STOP / TRAILING_STOP 不受成交量约束 | 止损单可超过当日成交量 |
| B5 | `StandardRisk` 不预留费用 | 满仓预算单现金可小幅为负 |
| B6 | 同帧多笔大额预算单互相预留 | 总量超现金时两笔都被拒（过度保守） |
| B7 | 涨跌停全局单一 `limit` | 创业板/科创板/北交所口径不准 |
| B8 | 数据结束后静态帧可成交尾部订单 | 尾部订单在「无市场」时成交 |
| B9 | `delta_t>1` 时 `frame.bars` 只留最后一根 | 只用 frame 访问会漏帧内 Bar |

## 6. 复现

```powershell
$env:PYTHONPATH="src"
python -m pytest -q tests/test_capability_gate.py
python -u tools/gate_runner.py --md reports/USABILITY.md
python -u tools/mutate_gate.py --md reports/MUTATION.md
```

**通过标准**（预登记 §7）：门禁 0 failed/0 error（strict xfail 除外）；本报告 G1–G7 全 PASS；变异 15/15；三条同时成立才可宣称「真实可用」。

