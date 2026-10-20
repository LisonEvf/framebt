"""变异验证（G8）：故意把引擎改坏，门禁必须能抓住。

做法：把 ``src/framebt`` 复制到临时目录 → 对**副本**做一个已知错误的文本替换 →
用 ``PYTHONPATH`` 指向副本跑 ``tests/test_capability_gate.py``。副本未打补丁时
（控制组）必须通过；打补丁后必须失败（被杀死）。

只改副本，**绝不修改工作树源码**。

用法：
    $env:PYTHONPATH="src"; python -u tools/mutate_gate.py
    $env:PYTHONPATH="src"; python -u tools/mutate_gate.py --md reports/MUTATION.md
    $env:PYTHONPATH="src"; python -u tools/mutate_gate.py --only M01
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "framebt"
GATE = "tests/test_capability_gate.py"

# 每条：(id, 相对文件, 原文, 替换, replace_all, 说明)
MUTATIONS = [
    ("M01", "execution/engine.py",
     "        if self.fill_price_mode == \"open\":\n            return frame.open(symbol)",
     "        if self.fill_price_mode == \"open\":\n            return frame.close(symbol)",
     False, "成交价从开盘价改成收盘价"),
    ("M02", "cost.py",
     "        commission = max(notional * self.commission, self.min_commission)",
     "        commission = notional * self.commission",
     False, "佣金取消最低 5 元"),
    ("M03", "cost.py",
     "        if fill.side is OrderSide.SELL and self.stamp_tax > 0:",
     "        if self.stamp_tax > 0:",
     False, "买入也收印花税"),
    ("M04", "ledger.py",
     "            self.cash += notional - fill.cost",
     "            self.cash += notional",
     False, "卖出不计成本"),
    ("M05", "rules.py",
     "            if upper is not None and frame.open(symbol) > upper:",
     "            if False and upper is not None and frame.open(symbol) > upper:",
     True, "关闭涨停拒单"),
    ("M06", "rules.py",
     "                if available + 1e-9 < order.size:",
     "                if False and available + 1e-9 < order.size:",
     False, "关闭 T+1 可卖量检查"),
    ("M07", "ledger.py",
     "settle_frame=fill.frame_id + self.settlement_cycle,",
     "settle_frame=fill.frame_id,",
     True, "买入腿结算日改成当日（T+0）"),
    ("M08", "execution/engine.py",
     "            if age < self.delay_frames:",
     "            if age < 0:",
     False, "关闭 I2 延迟"),
    ("M09", "execution/engine.py",
     "                fill_i = int(min(rem, remaining))",
     "                fill_i = int(rem)",
     False, "限价成交量分配忽略剩余量"),
    ("M10", "data_bus.py",
     "            return self._now_ns",
     "            return token.issued_at_ns",
     False, "伪造 token 直接采用其时间"),
    ("M11", "data_bus.py",
     "    def _check_write(self, key: Optional[object]) -> None:\n"
     "        if self._guard_key is not None and key is not self._guard_key:",
     "    def _check_write(self, key: Optional[object]) -> None:\n"
     "        if False:",
     False, "关闭总线写保护"),
    ("M12", "ledger.py",
     "                eq += pos.net_size() * mark_price[pos.symbol]",
     "                pass",
     False, "MTM 忽略持仓"),
    ("M13", "ledger.py",
     "            self.cash += action.ratio * qty_before",
     "            pass",
     False, "退市不结算现金"),
    ("M14", "execution/engine.py",
     "                fill_px = open_px if open_px <= limit_px else limit_px",
     "                fill_px = low_px",
     False, "限价单按当日最低价成交"),
    ("M15", "ledger.py",
     "            self.cash -= notional + fill.cost",
     "            self.cash -= notional + 2 * fill.cost",
     False, "成本在现金中扣两次"),
]


def _copy_package() -> tuple[Path, Path]:
    tmp = Path(tempfile.mkdtemp(prefix="framebt_mut_"))
    shutil.copytree(SRC, tmp / "framebt",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return tmp, tmp / "framebt"


def _apply(pkg: Path, rel: str, old: str, new: str, replace_all: bool) -> int:
    path = pkg / rel
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise RuntimeError(f"补丁锚点在 {rel} 中找不到：{old[:60]!r}")
    if count > 1 and not replace_all:
        raise RuntimeError(f"补丁锚点在 {rel} 中出现 {count} 次，需 replace_all 或更精确的锚点")
    path.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1),
                    encoding="utf-8")
    return count


def _env(tmp: Path) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(tmp)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _import_ok(tmp: Path) -> tuple[bool, str]:
    p = subprocess.run(
        [sys.executable, "-c", "import framebt; print(framebt.__file__)"],
        cwd=str(ROOT), env=_env(tmp), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )
    ok = p.returncode == 0 and str(tmp) in (p.stdout or "")
    return ok, ((p.stdout or "") + (p.stderr or "")).strip()


def _run_gate(tmp: Path) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", GATE, "-x", "-p", "no:cacheprovider",
         "-k", "not cross_process and not clean_stdlib"],
        cwd=str(ROOT), env=_env(tmp), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180,
    )
    return p.returncode, ((p.stdout or "") + (p.stderr or "")), time.perf_counter() - t0


def _failed_test(output: str | None) -> str:
    if not output:
        return ""
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            return line.split(" ", 1)[1].split(" - ")[0]
        if "XPASS(strict)" in line:
            return "XPASS(strict)"
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", default=None, help="写出 Markdown 报告")
    ap.add_argument("--only", default=None, help="只跑一个变异体（如 M01）")
    args = ap.parse_args()

    selected = [m for m in MUTATIONS if args.only is None or m[0] == args.only]
    if not selected:
        print(f"未知变异体 {args.only}")
        return 2

    print("== 控制组（未打补丁的副本）==")
    ctrl_tmp, _ = _copy_package()
    try:
        ok, detail = _import_ok(ctrl_tmp)
        if not ok:
            print("[FATAL] 控制组 import 失败：\n" + detail)
            return 2
        rc, out, dt = _run_gate(ctrl_tmp)
        print(f"  import OK；gate exit={rc}（{dt:.2f}s）")
        if rc != 0:
            print(out[-3000:])
            print("[FATAL] 控制组未通过，变异结果无意义")
            return 2
    finally:
        shutil.rmtree(ctrl_tmp, ignore_errors=True)

    rows = []
    for mid, rel, old, new, all_, note in selected:
        print(f"\n== {mid}  {note} ==")
        tmp, pkg = _copy_package()
        try:
            n = _apply(pkg, rel, old, new, all_)
            ok, detail = _import_ok(tmp)
            if not ok:
                print(f"  [INVALID] 打了补丁后 import 失败（变异无效）\n{detail[-800:]}")
                rows.append((mid, note, "INVALID", "", 0.0))
                continue
            rc, out, dt = _run_gate(tmp)
            failed = _failed_test(out)
            status = "KILLED" if rc != 0 else "SURVIVED"
            print(f"  {status}  exit={rc}  {dt:.2f}s  {failed}")
            rows.append((mid, note, status, failed, dt))
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {exc}")
            rows.append((mid, note, "ERROR", str(exc), 0.0))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    killed = sum(1 for r in rows if r[2] == "KILLED")
    total = len(rows)
    print(f"\n== 杀死率 {killed}/{total} ==")
    for mid, note, status, failed, dt in rows:
        print(f"  {mid}  {status:<9} {note:<28} {failed}")
    if killed != total:
        print("存在幸存/无效变异体，G8 失败")

    if args.md:
        path = Path(args.md)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# 门禁变异验证（G8）",
            "",
            f"> 生成：{time.strftime('%Y-%m-%d %H:%M:%S')} ｜ "
            f"命令：`python tools/mutate_gate.py` ｜ 控制组：exit 0",
            "",
            "把已知 bug 注入 `src/framebt` 的**临时副本**，用 "
            "`tests/test_capability_gate.py` 去抓。只改副本，不动工作树。",
            "",
            f"**杀死率：{killed}/{total}**" + ("（全部杀死）" if killed == total else
                                              "（存在幸存/无效，G8 失败）"),
            "",
            "| # | 注入的错误 | 结果 | 抓住它的测试 | 用时 |",
            "|---|---|---|---|---|",
        ]
        for mid, note, status, failed, dt in rows:
            lines.append(f"| {mid} | {note} | **{status}** | `{failed or '-'}` | {dt:.2f}s |")
        lines += [
            "",
            "判据（预登记 §4）：控制组必须通过；每个变异体必须 exit≠0 且不是 import/"
            "collection error；杀死率 < 15/15 即 G8 失败。",
        ]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"报告已写入 {path}")

    return 0 if killed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
