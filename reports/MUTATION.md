# 门禁变异验证（G8）

> 生成：2026-10-20 23:21:29 ｜ 命令：`python tools/mutate_gate.py` ｜ 控制组：exit 0

把已知 bug 注入 `src/framebt` 的**临时副本**，用 `tests/test_capability_gate.py` 去抓。只改副本，不动工作树。

**杀死率：15/15**（全部杀死）

| # | 注入的错误 | 结果 | 抓住它的测试 | 用时 |
|---|---|---|---|---|
| M01 | 成交价从开盘价改成收盘价 | **KILLED** | `tests/test_capability_gate.py::test_g1_buy_and_hold_exact_cash_cost_equity` | 0.81s |
| M02 | 佣金取消最低 5 元 | **KILLED** | `tests/test_capability_gate.py::test_g1_buy_and_hold_exact_cash_cost_equity` | 0.80s |
| M03 | 买入也收印花税 | **KILLED** | `tests/test_capability_gate.py::test_g1_buy_and_hold_exact_cash_cost_equity` | 0.77s |
| M04 | 卖出不计成本 | **KILLED** | `tests/test_capability_gate.py::test_g1_roundtrip_realized_pnl_and_cash` | 0.79s |
| M05 | 关闭涨停拒单 | **KILLED** | `tests/test_capability_gate.py::test_g5_limit_up_buy_rejected_and_limit_down_sell_rejected` | 0.85s |
| M06 | 关闭 T+1 可卖量检查 | **KILLED** | `tests/test_capability_gate.py::test_g2_fuzz_replay_cash_equity_cost_positions[0]` | 0.80s |
| M07 | 买入腿结算日改成当日（T+0） | **KILLED** | `tests/test_capability_gate.py::test_g5_ledger_settlement_cycle_is_really_t_plus_1` | 0.85s |
| M08 | 关闭 I2 延迟 | **KILLED** | `tests/test_capability_gate.py::test_g1_buy_and_hold_exact_cash_cost_equity` | 0.78s |
| M09 | 限价成交量分配忽略剩余量 | **KILLED** | `tests/test_capability_gate.py::test_g5_partial_fill_fifo_and_volume_allocation` | 0.85s |
| M10 | 伪造 token 直接采用其时间 | **KILLED** | `tests/test_capability_gate.py::test_g3_forged_token_cannot_read_future` | 0.82s |
| M11 | 关闭总线写保护 | **KILLED** | `tests/test_capability_gate.py::test_g3_write_guard_blocks_history_and_clock_tampering` | 0.82s |
| M12 | MTM 忽略持仓 | **KILLED** | `tests/test_capability_gate.py::test_g1_buy_and_hold_exact_cash_cost_equity` | 0.78s |
| M13 | 退市不结算现金 | **KILLED** | `tests/test_capability_gate.py::test_g1_delist_cash_settlement` | 0.78s |
| M14 | 限价单按当日最低价成交 | **KILLED** | `tests/test_capability_gate.py::test_g1_limit_fills_at_limit_not_daily_low` | 0.79s |
| M15 | 成本在现金中扣两次 | **KILLED** | `tests/test_capability_gate.py::test_g1_buy_and_hold_exact_cash_cost_equity` | 0.78s |

判据（预登记 §4）：控制组必须通过；每个变异体必须 exit≠0 且不是 import/collection error；杀死率 < 15/15 即 G8 失败。
