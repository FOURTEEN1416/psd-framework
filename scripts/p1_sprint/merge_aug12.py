# -*- coding: utf-8 -*-
"""merge_aug12 — 把 aug12 三个分片（b/c/d）的 per_seed 合并进主 summary。

用法：python scripts/p1_sprint/merge_aug12.py
合并后重算 mean_b / mean_full_ref / retention / seeds，并按 85/90 判定带给 verdict。
原文件备份为 summary_aug12_lr2e2.json.bak。
"""
from __future__ import annotations

import json
from pathlib import Path

RUNS = Path(__file__).resolve().parents[2] / "runs" / "p1_sprint" / "adapt_t4_ntu120"
MAIN = RUNS / "summary_aug12_lr2e2.json"
PARTS = [RUNS / "summary_aug12_b.json", RUNS / "summary_aug12_c.json",
         RUNS / "summary_aug12_d.json"]


def verdict(ret: float) -> str:
    if ret >= 0.90:
        return "CONFIRMS"
    if ret >= 0.85:
        return "PARTIAL"
    return "FAILS"


def main() -> None:
    base = json.loads(MAIN.read_text(encoding="utf-8"))
    rows = {int(r["seed"]): r for r in base["per_seed"]}
    for p in PARTS:
        part = json.loads(p.read_text(encoding="utf-8"))
        for r in part["per_seed"]:
            rows[int(r["seed"])] = r
    ordered = [rows[k] for k in sorted(rows)]
    mean_b = sum(r["b_arm"] for r in ordered) / len(ordered)
    mean_ref = sum(r["full_ref"] for r in ordered) / len(ordered)
    ret = mean_b / mean_ref
    base["per_seed"] = ordered
    base["seeds"] = [int(r["seed"]) for r in ordered]
    base["mean_b"] = round(mean_b, 4)
    base["mean_full_ref"] = round(mean_ref, 4)
    base["retention"] = round(ret, 4)
    base["verdict"] = verdict(ret)
    base["protocol"] = base.get("protocol", "") + " | 10-seed completion 2026-11-20 (b/c/d shards merged)"
    MAIN.with_suffix(".json.bak").write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    MAIN.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"seeds={base['seeds']}\nmean_b={mean_b:.4f} mean_ref={mean_ref:.4f} "
          f"retention={ret*100:.2f}% verdict={base['verdict']}")


if __name__ == "__main__":
    main()
