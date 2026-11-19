# -*- coding: utf-8 -*-
"""collect_e10 — 汇总 E10 域适配三线结果，生成对比表（evidence 镜像用）。

读取 runs/p1_sprint/adapt_*/summary.json 与 labelfree_<tier>/summary.json（基线），
输出 retention/verdict 对比表到 stdout 与 reports/p1_sprint_evidence/adaptation/E10_SUMMARY.md。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "p1_sprint"
OUT = REPO / "reports" / "p1_sprint_evidence" / "adaptation"


def load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> None:
    tier = sys.argv[1] if len(sys.argv) > 1 else "ntu120"
    base = load(RUNS / f"labelfree_{tier}" / "summary.json")
    rows = []
    if base:
        rows.append(("E9 frozen (published baseline)", base.get("retention"),
                     base.get("verdict"), base.get("full_ref"), base.get("linear_10pct"),
                     "frozen from-scratch tier pretext"))
    for label, path in [
        ("T1 TTA (10pct head, eval-BN)", RUNS / f"adapt_tta_{tier}" / "summary.json"),
        ("T2 LoRA same-domain", RUNS / f"adapt_eval_{tier}" / "summary_lora_samedomain.json"),
        ("T2b zero-shot NTU60→tier", RUNS / f"adapt_lora_xfer_{tier}" / "summary_zeroshot.json"),
        ("T2b LoRA cross-domain", RUNS / f"adapt_eval_{tier}" / "summary_lora_xfer.json"),
        ("T3 SupCon feedback", RUNS / f"adapt_supcon_{tier}" / "summary.json"),
        ("T4 ref5 (frozen + 5% semantic)", RUNS / f"adapt_t4_{tier}" / "summary_ref5.json"),
        ("T4 split5 (5% enc-CE + 5% semantic)", RUNS / f"adapt_t4_{tier}" / "summary_split5.json"),
        ("T4 same10 (10% enc-CE + 10% semantic)", RUNS / f"adapt_t4_{tier}" / "summary_same10.json"),
    ]:
        d = load(path)
        if d:
            rows.append((label, d.get("retention"), d.get("verdict"),
                         d.get("full_ref"), d.get("linear_10pct"),
                         d.get("protocol", "")[:80]))

    lines = [f"# E10 域适配三线汇总 — {tier}", "",
             "| 臂 | retention | verdict | full_ref | 10% linear | 备注 |",
             "|---|---|---|---|---|---|"]
    for label, ret, verdict, ref, lin, note in rows:
        lines.append(f"| {label} | {ret if ret is not None else '—'} "
                     f"| {verdict or '—'} | {ref if ref is not None else '—'} "
                     f"| {lin if lin is not None else '—'} | {note} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if len(rows) > 1:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"E10_SUMMARY_{tier}.md").write_text(text, encoding="utf-8")
        print(f"[done] {OUT / f'E10_SUMMARY_{tier}.md'}")


if __name__ == "__main__":
    main()
