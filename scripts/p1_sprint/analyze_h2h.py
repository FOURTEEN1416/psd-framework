# -*- coding: utf-8 -*-
"""P1 头对头矩阵 — 统计分析与论文数字产出.

输入:
  runs/p1_sprint/h2h_matrix/summary.json          （本轮矩阵: 基线多种子/多子集/多预算 + PSD 新格）
  reports/r16-ntu-pseudo-10seed-2026-09-07.json   （E9 PSD 10 自训练种子, subset42, 已发表口径）
  runs/p1_sprint/published_baseline_ntu60/summary.json （已发表单次运行 i0/s42: 66.71 / 74.30）
  runs/p1_sprint/supervised_stgcn_ntu60/summary.json   （已发表单次运行 i0/s42: 66.52 / 83.46）
产出:
  runs/p1_sprint/h2h_matrix/analysis.json —— tab:h2h 回填所需的全部数字与检验结果。
诚实条款: 只做描述统计与标准检验（Welch t / Mann-Whitney U），不挑检验方法凑显著。
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "runs" / "p1_sprint" / "h2h_matrix" / "analysis.json"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    m = load(REPO / "runs/p1_sprint/h2h_matrix/summary.json")
    r16 = load(REPO / "reports/r16-ntu-pseudo-10seed-2026-09-07.json")
    pb = load(REPO / "runs/p1_sprint/published_baseline_ntu60/summary.json")
    sg = load(REPO / "runs/p1_sprint/supervised_stgcn_ntu60/summary.json")
    cells = m["cells"]

    psd10 = [r["top1"] for r in r16["arms"]["b_selftrain_10pct"]]  # 10 selftrain seeds, subset42
    aim_i0, stg_i0 = pb["budgets"]["10pct"]["top1"], sg["budgets"]["10pct"]["top1"]
    aim_full, stg_full = pb["budgets"]["full"]["top1"], sg["budgets"]["full"]["top1"]

    aim_seeds = [aim_i0] + [cells[f"aimclr_10_s42_i{s}"]["top1"] for s in (1, 2, 3, 4)
                            if f"aimclr_10_s42_i{s}" in cells]
    stg_seeds = [stg_i0] + [cells[f"stgcn_10_s42_i{s}"]["top1"] for s in (1, 2, 3, 4)
                            if f"stgcn_10_s42_i{s}" in cells]

    def welch(a, b):
        t, p = stats.ttest_ind(a, b, equal_var=False)
        u, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
        return {"n_a": len(a), "n_b": len(b),
                "mean_a": round(float(np.mean(a)), 4), "mean_b": round(float(np.mean(b)), 4),
                "std_a": round(float(np.std(a, ddof=1)), 4), "std_b": round(float(np.std(b, ddof=1)), 4),
                "diff_pp": round((float(np.mean(a)) - float(np.mean(b))) * 100, 2),
                "welch_t": round(float(t), 3), "welch_p": float(p),
                "mwu_p": float(pu)}

    tests = {
        "psd_vs_aimclr_10pct": welch(psd10, aim_seeds),
        "psd_vs_stgcn_10pct": welch(psd10, stg_seeds),
        "psd_vs_linear_head_10pct": welch(psd10, [0.6605]),  # 确定性线性头（单值，仅列差值）
    }

    # 子集重抽配对差（42=已发表格; 43/44=矩阵新格; PSD 取 3 selftrain seeds 均值）
    subsets = {}
    for sub in (42, 43, 44):
        e = {}
        if sub == 42:
            e["psd_mean"] = round(float(np.mean(psd10)), 4)
            e["aimclr"], e["stgcn"] = aim_i0, stg_i0
        else:
            k = f"psd_10_s{sub}"
            if k in cells:
                e["psd_mean"] = cells[k]["mean_top1"]
            e["aimclr"] = cells.get(f"aimclr_10_s{sub}_i0", {}).get("top1")
            e["stgcn"] = cells.get(f"stgcn_10_s{sub}_i0", {}).get("top1")
        if all(v is not None for v in e.values()):
            e["d_psd_aimclr_pp"] = round((e["psd_mean"] - e["aimclr"]) * 100, 2)
            e["d_psd_stgcn_pp"] = round((e["psd_mean"] - e["stgcn"]) * 100, 2)
        subsets[str(sub)] = e

    # 预算趋势
    budgets = {"0.05": {}, "0.10": {}, "0.20": {}, "full": {}}
    budgets["0.10"] = {"psd": round(float(np.mean(psd10)), 4), "psd_std": round(float(np.std(psd10, ddof=1)), 4),
                       "aimclr": round(float(np.mean(aim_seeds)), 4), "stgcn": round(float(np.mean(stg_seeds)), 4),
                       "linear_head_ref": 0.6605}
    budgets["full"] = {"aimclr": aim_full, "stgcn": stg_full,
                       "psd_linear_probe_ref": 0.7445,
                       "note": "PSD semantic pipeline not run at full budget (low-budget mechanism); frozen linear probe is the full-budget reference"}
    for frac in ("05", "20"):
        k_p, k_a, k_s = f"psd_f{frac}_s42", f"aimclr_f{frac}_s42_i0", f"stgcn_f{frac}_s42_i0"
        if k_p in cells:
            budgets[f"0.{frac[1] if frac != '05' else '0'}5" if frac == "05" else "0.20"] = {
                "psd": cells[k_p]["mean_top1"], "psd_std": cells[k_p]["std_top1"],
                "aimclr": cells.get(k_a, {}).get("top1"), "stgcn": cells.get(k_s, {}).get("top1")}

    out = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "matrix_mode": m.get("mode"),
        "psd_10seed_subset42": {"values": psd10, "mean": round(float(np.mean(psd10)), 4),
                                 "std": round(float(np.std(psd10, ddof=1)), 4)},
        "aimclr_5seed_subset42": {"values": aim_seeds, "mean": round(float(np.mean(aim_seeds)), 4),
                                   "std": round(float(np.std(aim_seeds, ddof=1)), 4)},
        "stgcn_5seed_subset42": {"values": stg_seeds, "mean": round(float(np.mean(stg_seeds)), 4),
                                  "std": round(float(np.std(stg_seeds, ddof=1)), 4)},
        "significance_tests": tests,
        "subset_robustness_10pct": subsets,
        "budget_curve": budgets,
        "failed_cells": m.get("failed", {}),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"\n[done] {OUT}")


if __name__ == "__main__":
    main()
