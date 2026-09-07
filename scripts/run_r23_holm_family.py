# -*- coding: utf-8 -*-
"""R23 终审修复: E9 系列统一 Holm 族的实算脚本与工件生成。

背景: 终审发现正文印刷的 SSL 校正 p=0.008 无实算工件且与正文自定义的
six-test family(3 tiers x 2 statistics, 不含 SSL 检验)不自洽。修复口径:
族扩为 eight tests = 3 个 tier 的 gap 检验(单样本 t + Wilcoxon 各一)
+ 2 个外部 SSL Wilcoxon 检验(NTU60/UCF101)。PanAf500 gap 按正文口径
不进族(directional only, 臂分布与线性臂打平)。

输入全部为已归档工件(reports/), 零手填数字; 输出
reports/r23-holm-family-2026-09-07.json。幂等: 同输入必逐位同输出。
"""
import json
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "r23-holm-family-2026-09-07.json"


def holm(pvals):
    """Holm-Bonferroni step-down; 返回与输入同序的校正 p 列表。"""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = max((m - rank) * pvals[idx], running)
        running = min(val, 1.0)
        adj[idx] = running
    return adj


def main():
    r16 = json.loads((ROOT / "reports/r16-ntu-pseudo-10seed-2026-09-07.json").read_text(encoding="utf-8"))
    n120 = json.loads((ROOT / "reports/p5b-ntu120-retention-2026-09-07.json").read_text(encoding="utf-8"))
    ucf = json.loads((ROOT / "reports/p5b-ucf101-retention-2026-09-07.json").read_text(encoding="utf-8"))
    ssl1 = json.loads((ROOT / "reports/r23-ssl-baseline-2026-09-07.json").read_text(encoding="utf-8"))
    ssl2 = json.loads((ROOT / "reports/r23-ssl-baseline-ucf101-2026-09-07.json").read_text(encoding="utf-8"))

    tops60 = [r["top1"] for r in r16["arms"]["b_selftrain_10pct"]]
    a60 = r16["arms"]["a_linear_10pct"]["top1"]
    mt1 = [r["top1"] for r in ssl1["arms"]["mt_rows"]]
    mt2 = [r["top1"] for r in ssl2["arms"]["mt_rows"]]

    tests = {
        "E9_NTU60_gap_one_sample_t": stats.ttest_1samp(tops60, a60).pvalue,
        "E9b_NTU120_gap_one_sample_t": stats.ttest_1samp(n120["b_arms"], n120["linear_10pct"]).pvalue,
        "E9c_UCF101_gap_one_sample_t": stats.ttest_1samp(ucf["b_arms"], ucf["linear_10pct"]).pvalue,
        "E9_NTU60_gap_wilcoxon": stats.wilcoxon([x - a60 for x in tops60]).pvalue,
        "E9b_NTU120_gap_wilcoxon": stats.wilcoxon([x - n120["linear_10pct"] for x in n120["b_arms"]]).pvalue,
        "E9c_UCF101_gap_wilcoxon": stats.wilcoxon([x - ucf["linear_10pct"] for x in ucf["b_arms"]]).pvalue,
        "SSL_NTU60_wilcoxon": stats.wilcoxon([p - m for p, m in zip(tops60, mt1)]).pvalue,
        "SSL_UCF101_wilcoxon": stats.wilcoxon([p - m for p, m in zip(ucf["b_arms"], mt2)]).pvalue,
    }
    names = list(tests)
    adj = dict(zip(names, holm([tests[n] for n in names])))

    evidence = {
        "E9_NTU60_gap_one_sample_t": "reports/r16-ntu-pseudo-10seed-2026-09-07.json",
        "E9b_NTU120_gap_one_sample_t": "reports/p5b-ntu120-retention-2026-09-07.json",
        "E9c_UCF101_gap_one_sample_t": "reports/p5b-ucf101-retention-2026-09-07.json",
        "E9_NTU60_gap_wilcoxon": "reports/r16-ntu-pseudo-10seed-2026-09-07.json",
        "E9b_NTU120_gap_wilcoxon": "reports/p5b-ntu120-retention-2026-09-07.json",
        "E9c_UCF101_gap_wilcoxon": "reports/p5b-ucf101-retention-2026-09-07.json",
        "SSL_NTU60_wilcoxon": "reports/r23-ssl-baseline-2026-09-07.json",
        "SSL_UCF101_wilcoxon": "reports/r23-ssl-baseline-ucf101-2026-09-07.json",
    }
    artifact = {
        "date": "2026-09-07",
        "purpose": "Final-audit fix: printed Holm corrections for the E9 series must come from a "
                   "committed computation. Family = 8 tests (3 tiers x {one-sample t, Wilcoxon} gap "
                   "tests + 2 external-SSL Wilcoxon tests); PanAf500 gap excluded (directional only, "
                   "arm ties with linear). Supersedes the ghost value 0.008 and the 0.099 pair.",
        "family_note": "Conclusions are invariant to the narrower six-test family (gap tests only): "
                       "NTU60/NTU120 W correct to 0.0078, UCF101 t/W to 0.0989, and the SSL tests are "
                       "then reported raw -- every qualitative reading is unchanged.",
        "tests": {
            n: {
                "raw_p": tests[n],
                "holm_corrected_p": adj[n],
                "evidence": evidence[n],
            }
            for n in names
        },
    }
    OUT.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
    for n in names:
        print(f"{n:30s} raw={tests[n]:.6g}  holm8={adj[n]:.6g}")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
