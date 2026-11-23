# -*- coding: utf-8 -*-
"""audit_next_paper — 校验下一篇手稿中的每个数字都能在实测证据中找到。

对 paper_manuscript_v1.md 的 (数字, 证据文件集合) 清单逐项断言；
并复核 PanAf 配对统计（改善/持平/退化种子数）与摘要词数。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "next_paper" / "paper_manuscript_v1.md"
EV = REPO / "reports" / "p1_sprint_evidence"
RUNS = REPO / "runs" / "p1_sprint"

# (数字字符串, 允许出现的证据文件相对路径列表；百分数会自动尝试 /100 小数变体)
A = "adaptation"
CHECKS: list[tuple[str, list[str]]] = [
    # 四层冻结基线
    ("90.7", ["MANIFEST.md"]),
    ("82.5", ["MANIFEST.md", f"labelfree_tiers/panaf500_labelfree_summary.json"]),
    ("74.96", ["MANIFEST.md", "labelfree_tiers/ntu120_labelfree_summary.json"]),
    ("51.09", ["MANIFEST.md", "labelfree_tiers/ucf101_labelfree_summary.json"]),
    ("58.37", ["MANIFEST.md", "labelfree_tiers/ntu120_labelfree_summary.json"]),
    ("19.87", ["MANIFEST.md", "labelfree_tiers/ucf101_labelfree_summary.json"]),
    # E10 阴性套件
    ("74.08", ["adapt_tta_ntu120/summary.json"]),
    ("74.99", [f"{A}/ntu120/summary_lora_samedomain.json"]),
    ("69.77", [f"{A}/ntu120/summary_zeroshot.json"]),
    ("71.98", [f"{A}/ntu120/summary_lora_xfer.json"]),
    ("74.98", [f"{A}/ntu120/summary.json"]),
    ("59.66", [f"{A}/ntu120/summary_lora_xfer.json"]),
    ("45.27", [f"{A}/ntu120/summary_lora_xfer.json"]),
    # TENT 诊断（27.0 记录于 MANIFEST 16）
    ("27", ["MANIFEST.md"]),
    # 双用主结果
    ("85.16", [f"{A}/ntu120/summary_aug8_lr2e2.json", "E11_SUMMARY.md", "MANIFEST.md"]),
    ("92.21", [f"{A}/panaf500/summary_same10.json", "E11_SUMMARY.md", "MANIFEST.md"]),
    ("71.21", [f"{A}/ucf101/summary_same10.json", "E11_SUMMARY.md", "MANIFEST.md"]),
    ("10.20", ["MANIFEST.md"]),
    ("9.71", ["MANIFEST.md"]),
    ("20.12", ["MANIFEST.md"]),
    ("49.60", ["MANIFEST.md", f"{A}/ntu120/summary_aug8_lr2e2.json"]),
    ("43.75", ["MANIFEST.md", "labelfree_tiers/ntu120_labelfree_summary.json"]),
    # 预算曲线
    ("64.61", [f"{A}/ntu120/summary_ref5.json", "E11_SUMMARY.md"]),
    ("67.51", [f"{A}/ntu120/summary_split5.json", "E11_SUMMARY.md"]),
    ("79.04", [f"{A}/ntu120/summary_ref15.json", "E11_SUMMARY.md"]),
    ("83.60", [f"{A}/ntu120/summary_dual15.json", "E11_SUMMARY.md"]),
    ("2.90", ["MANIFEST.md"]),
    ("3.91", ["MANIFEST.md"]),
    ("4.56", ["MANIFEST.md"]),
    # 敏感性
    ("78.87", ["MANIFEST.md", f"{A}/ntu120/summary_same10.json"]),
    ("82.13", ["MANIFEST.md", f"{A}/ntu120/summary_sweep_lr1e2.json"]),
    ("82.73", ["MANIFEST.md", f"{A}/ntu120/summary_sweep_lr2e2.json"]),
    ("83.41", ["MANIFEST.md", f"{A}/ntu120/summary_aug4_lr2e2.json"]),
    ("85.16", [f"{A}/ntu120/summary_aug8_lr2e2.json"]),
    ("84.38", ["MANIFEST.md", f"{A}/ntu120/summary_aug12_lr2e2.json"]),
    ("79.05", ["MANIFEST.md", f"{A}/ntu120/summary_sweep_d0.json"]),
    ("81.49", ["MANIFEST.md", f"{A}/ntu120/summary_sweep_lr1e2_r16.json"]),
    ("78.88", ["MANIFEST.md", f"{A}/ntu120/summary_sweep_ep200.json"]),
    ("68.31", ["MANIFEST.md", f"{A}/ucf101/summary_aug4_lr1e2.json"]),
    # 方法/容量（run log）
    ("62,616", ["adapt_lora_ntu120_run.log"]),
    ("1,965,368", ["adapt_lora_ntu120_run.log"]),
    ("3.19", ["adapt_lora_ntu120_run.log"]),
    # CIL / ST-GCN（MANIFEST 13/14 + runs summary）
    ("68.6", ["MANIFEST.md", "cil_baseline_trans002/summary.json"]),
    ("72.9", ["cil_baseline_trans002/summary.json"]),
    ("36.17", ["supervised_stgcn_ntu120/summary.json"]),
    # 合成域 fullft / merge-rename / split-addition（CVIU LaTeX 稿 + runs summary）
    ("98.2", ["MANIFEST.md"]),
    ("76.8", ["sections/04-experiments.tex", "main.tex"]),
    ("76.5", ["merge_rename_upper/summary.json", "sections/04-experiments.tex"]),
    ("97.13", ["main.tex", "supplementary.tex"]),
    # SupCon 覆盖（run log）
    ("7,239", ["adapt_supcon_ntu120_run.log"]),
    # 120 类规模复现（Table 6 + Appendix C3）
    ("66.33", ["labelfree_tiers/ntu120_full_labelfree_summary.json", "MANIFEST.md"]),
    ("77.19", [f"{A}/ntu120_full/summary_full10.json", "MANIFEST.md"]),
    ("29.41", ["MANIFEST.md"]),
    ("33.06", [f"{A}/ntu120_full/summary_full10.json", "MANIFEST.md"]),
    ("44.33", ["labelfree_tiers/ntu120_full_labelfree_summary.json", "MANIFEST.md"]),
    ("42.83", [f"{A}/ntu120_full/summary_full10.json", "MANIFEST.md"]),
    ("10.84", ["next_paper_stats.json", "MANIFEST.md"]),
    ("3.65", ["next_paper_stats.json", "MANIFEST.md"]),
    ("9.34", ["next_paper_stats.json", "MANIFEST.md"]),
    ("12.68", ["next_paper_stats.json", "MANIFEST.md"]),
    ("2.99", ["next_paper_stats.json", "MANIFEST.md"]),
    ("4.40", ["next_paper_stats.json", "MANIFEST.md"]),
    ("12,000", ["MANIFEST.md"]),
    ("2,400", ["MANIFEST.md"]),
    ("1,200", ["MANIFEST.md"]),
    # 32ch 容量对照（Table 7 + Appendix C4/D）
    ("68.48", ["labelfree_tiers/ntu120_32ch_labelfree_summary.json", "MANIFEST.md"]),
    ("83.55", [f"{A}/ntu120_32ch/summary_aug8_lr1e2.json", "MANIFEST.md"]),
    ("56.85", ["labelfree_tiers/ntu120_32ch_labelfree_summary.json", "MANIFEST.md"]),
    ("61.28", [f"{A}/ntu120_32ch/summary_aug8_lr1e2.json", "MANIFEST.md"]),
    ("51.20", [f"{A}/ntu120_32ch/summary_aug8_lr1e2.json", "MANIFEST.md"]),
    ("38.93", ["MANIFEST.md"]),
    ("15.07", ["next_paper_stats.json", "MANIFEST.md"]),
    ("12.27", ["next_paper_stats.json", "MANIFEST.md"]),
    ("13.49", ["next_paper_stats.json", "MANIFEST.md"]),
    ("16.71", ["next_paper_stats.json", "MANIFEST.md"]),
    ("11.40", ["next_paper_stats.json", "MANIFEST.md"]),
    ("13.17", ["next_paper_stats.json", "MANIFEST.md"]),
    ("4.43", ["MANIFEST.md"]),
    ("84.20", ["MANIFEST.md"]),
    ("2,826,088", ["labelfree_tiers/ntu120_32ch_labelfree_summary.json", "MANIFEST.md"]),
    # 64ch 标准宽度对照（Table 7 第三行 + Appendix C5）
    ("74.78", ["labelfree_tiers/ntu120_64ch_labelfree_summary.json", "MANIFEST.md"]),
    ("59.68", ["labelfree_tiers/ntu120_64ch_labelfree_summary.json", "MANIFEST.md"]),
    ("6,364,200", ["labelfree_tiers/ntu120_64ch_labelfree_summary.json", "MANIFEST.md"]),
    ("83.62", [f"{A}/ntu120_64ch/summary_aug8_lr1e2.json", "MANIFEST.md"]),
    ("52.31", [f"{A}/ntu120_64ch/summary_aug8_lr1e2.json", "MANIFEST.md"]),
    ("62.56", [f"{A}/ntu120_64ch/summary_aug8_lr1e2.json", "MANIFEST.md"]),
    ("44.63", ["MANIFEST.md"]),
    ("8.84", ["next_paper_stats.json", "MANIFEST.md"]),
    ("7.98", ["next_paper_stats.json", "MANIFEST.md"]),
    ("9.61", ["next_paper_stats.json", "MANIFEST.md"]),
    ("7.68", ["next_paper_stats.json", "MANIFEST.md"]),
    ("7.06", ["next_paper_stats.json", "MANIFEST.md"]),
    ("8.32", ["next_paper_stats.json", "MANIFEST.md"]),
    ("8.16", ["next_paper_stats.json", "MANIFEST.md"]),
    ("2.88", ["MANIFEST.md"]),
    # class-addition taxonomy episode（Table 8 + Appendix C6）
    ("42.89", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("27.31", [f"{A}/ntu120/summary_taxadd_add60to120.json", "taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("32.51", [f"{A}/ntu120/summary_taxadd_add60to120.json", "taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("31.99", ["MANIFEST.md"]),
    ("34.84", ["MANIFEST.md"]),
    ("22.62", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("30.19", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("5.20", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("15.49", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("12.11", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("75.68", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("63.57", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("79.06", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("10.90", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("8.06", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("2.84", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("7.57", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("4.70", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("5.68", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("14.54", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("16.55", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("10.94", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("13.22", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("2.03", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("3.53", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("7.10", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("8.05", ["taxonomy_addition_stats.json", "MANIFEST.md"]),
    ("600", ["MANIFEST.md"]),
    # 64ch 预算扫描（Table 9）
    ("65.64", [f"{A}/ntu120_64ch/summary_ref5.json", "MANIFEST.md"]),
    ("70.57", [f"{A}/ntu120_64ch/summary_split5.json", "MANIFEST.md"]),
    ("78.64", [f"{A}/ntu120_64ch/summary_ref15.json", "MANIFEST.md"]),
    ("89.40", [f"{A}/ntu120_64ch/summary_dual15.json", "MANIFEST.md"]),
    ("4.93", [f"{A}/ntu120_64ch/budget_sweep_stats.json", "MANIFEST.md"]),
    ("10.76", [f"{A}/ntu120_64ch/budget_sweep_stats.json", "MANIFEST.md"]),
]


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    misses = []
    for num, files in CHECKS:
        variants = {num, num.replace(",", "")}
        try:
            variants.add(str(round(float(num) / 100, 4)))
        except ValueError:
            pass
        hits = []
        for f in files:
            for base in (EV, RUNS, REPO / "docs" / "paper" / "latex"):
                p = (base / f).resolve()
                if p.exists():
                    raw = p.read_bytes()
                    if any(v.encode() in raw for v in variants):
                        hits.append(f)
                    break
        if not hits:
            misses.append((num, files))
    if misses:
        print("MISSING EVIDENCE:")
        for num, files in misses:
            print(f"  {num}  <- {files}")
    else:
        print(f"ALL {len(CHECKS)} NUMBERS VERIFIED against evidence files.")

    # PanAf 配对统计复核
    base = json.loads((RUNS / "labelfree_panaf500" / "summary.json").read_text())
    t4 = json.loads((RUNS / "adapt_t4_panaf500" / "summary_aug4.json").read_text())
    pairs = list(zip(base["b_arms"], [r["b_arm"] for r in t4["per_seed"]]))
    up = sum(1 for a, b in pairs if b > a)
    down = sum(1 for a, b in pairs if b < a)
    tie = sum(1 for a, b in pairs if b == a)
    print(f"PanAf paired: up={up} down={down} tie={tie} (manuscript claims 5/3/2)")

    # 摘要词数
    m = re.search(r"## Abstract\n+(.*?)\n---", text, re.S)
    abstract = m.group(1)
    abstract_clean = re.sub(r"[`*#]", "", abstract)
    wc = len(abstract_clean.split())
    print(f"Abstract words: {wc} (limit 250: {'OK' if wc <= 250 else 'OVER'})")

    if misses or up != 5 or down != 2 or tie != 3 or wc > 250:
        raise SystemExit(1)
    print("AUDIT_OK")


if __name__ == "__main__":
    main()
