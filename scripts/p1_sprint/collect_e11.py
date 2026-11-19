# -*- coding: utf-8 -*-
"""collect_e11 — 汇总 T4 扫参 + 预算曲线 + 跨 tier 铺开结果（E11 证据总表）。

读取 runs/p1_sprint/adapt_t4_<tier>/summary_*.json，输出：
  - NTU120 扫参表（含基线/同10对照）
  - 预算-分配率曲线（5/10/15% × 冻结/双用）
  - 跨 tier 铺开表
写入 reports/p1_sprint_evidence/adaptation/E11_SUMMARY.md
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "p1_sprint"
OUT = REPO / "reports" / "p1_sprint_evidence" / "adaptation"


def load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def row(label: str, d: dict | None, note: str = "") -> str:
    if not d:
        return f"| {label} | — | — | — | — | {note} |"
    mb = d.get("mean_b")
    mr = d.get("mean_full_ref") or d.get("full_ref")
    return (f"| {label} | {d.get('retention')} | {d.get('verdict')} "
            f"| {mb if mb is not None else (round(sum(d['b_arms'])/len(d['b_arms']),4) if d.get('b_arms') else '—')} "
            f"| {mr if mr is not None else '—'} | {note} |")


def main() -> None:
    t4 = RUNS / "adapt_t4_ntu120"
    base = load(RUNS / "labelfree_ntu120" / "summary.json")
    lines = ["# E11 — T4 扫参 + 预算曲线 + 跨 tier 铺开（NTU120 为主）", "",
             "## 1. NTU120 超参扫描（same10 双用臂，基线=冻结10%语义 74.96%）", "",
             "| 配置 | retention | verdict | mean_b | mean_ref | 备注 |",
             "|---|---|---|---|---|---|"]
    lines.append(row("base: lr1e-3 λ1.0 r8 ep100（T4 same10）", load(t4 / "summary_same10.json")))
    for tag, note in [("sweep_d0", "λ=0 无蒸馏锚"), ("sweep_lr1e2", "lr=1e-2"),
                      ("sweep_lr2e2", "lr=2e-2"), ("sweep_lr4e2", "lr=4e-2（语义臂校准崩溃）"),
                      ("sweep_ep200", "200 epochs"), ("sweep_lr1e2_r16", "lr=1e-2 + rank16"),
                      ("aug4_lr2e2", "lr=2e-2 + 锚点增广×4"), ("aug4_lr1e2", "lr=1e-2 + 锚点增广×4")]:
        lines.append(row(tag, load(t4 / f"summary_{tag}.json"), note))

    lines += ["", "## 2. 预算-分配率曲线（NTU120）", "",
              "| 臂 | retention | verdict | mean_b | mean_ref | 备注 |",
              "|---|---|---|---|---|---|"]
    lines.append(row("5% 冻结（全给语义）", load(t4 / "summary_ref5.json")))
    lines.append(row("5% 双用（编码器+语义）", load(t4 / "summary_split5.json")))
    lines.append(row("10% 冻结（已发布基线）", base))
    lines.append(row("10% 双用", load(t4 / "summary_same10.json")))
    lines.append(row("15% 冻结", load(t4 / "summary_ref15.json")))
    lines.append(row("15% 双用", load(t4 / "summary_dual15.json")))

    lines += ["", "## 3. 跨 tier 铺开（same10 双用 vs 冻结基线）", "",
              "| tier | 冻结基线 | 双用（最优配置） | Δ | 判定变化 |",
              "|---|---|---|---|---|"]
    for tier, base_path, dual_path, note in [
            ("ntu120", RUNS / "labelfree_ntu120" / "summary.json",
             RUNS / "adapt_t4_ntu120" / "summary_aug4_lr2e2.json", "lr2e-2+aug4"),
            ("panaf500", RUNS / "labelfree_panaf500" / "summary.json",
             RUNS / "adapt_t4_panaf500" / "summary_aug4.json", "lr1e-2+aug4"),
            ("ucf101", RUNS / "labelfree_ucf101" / "summary.json",
             RUNS / "adapt_t4_ucf101" / "summary_same10.json", "lr2e-2")]:
        b = load(base_path)
        s = load(dual_path)
        if b and s:
            delta = round(s["retention"] - b["retention"], 4)
            lines.append(f"| {tier}（{note}） | {b['retention']} ({b['verdict']}) "
                         f"| {s['retention']} ({s['verdict']}) | {delta:+.4f} "
                         f"| {b['verdict']}→{s['verdict']} |")
        else:
            lines.append(f"| {tier}（{note}） | {b['retention'] if b else '—'} "
                         f"| {s['retention'] if s else '—'} | — | — |")

    text = "\n".join(lines) + "\n"
    print(text)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "E11_SUMMARY.md").write_text(text, encoding="utf-8")
    print(f"[done] {OUT / 'E11_SUMMARY.md'}")


if __name__ == "__main__":
    main()
