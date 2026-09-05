# -*- coding: utf-8 -*-
"""P3 种子预算曲线定位 — PSD-BUDGET-PREREG-001（描述性，无通过/失败判据）。

V0(对照)+V2(P1 最优门) × spc∈{2,3,4,6,8,12} × seeds 42-51，AK v1 warm 特征，
R16 修正协议（最终头=种子真标签∪池伪标签；oracle 停止关）。报告每格 mean±std
top-1/macro-F1/保留率(对 33.93% 全监督参照)/池大小/池 oracle 精度诊断。
膝点=V2 mean top-1≥24%(70% 保留) 的最小 spc（线性插值括住，不编造精确值）。
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p07_endtoend_ak import HEAD_CFG, KW, extract_features, load_dataset  # noqa: E402
from run_p15_align import run_one_align, ARMS  # noqa: E402

SEEDS10 = tuple(range(42, 52))
SPCS = (2, 3, 4, 6, 8, 12)
FULL_REF = 0.3393  # p07 supervised full-budget reference (unaffected by R16 protocol error)
OUT_DIR = REPO / "reports"


def main():
    t0 = time.time()
    data, kp, labels, splits = load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    n_train = int((splits == "train").sum())
    print(f"[p17] {len(data)} clips train={n_train} classes={class_names}")
    f_warm = extract_features(kp, "warm")

    grid = {}
    for arm in ("V0_control", "V2_consensus_quota"):
        for spc in SPCS:
            rows = [run_one_align(f_warm, labels_str, splits, class_names, ARMS[arm], seed, spc=spc)
                    for seed in SEEDS10]
            t = np.array([r["top1"] for r in rows])
            m = np.array([r["macro_f1"] for r in rows])
            grid[f"{arm}_spc{spc}"] = {
                "spc": spc, "label_frac_pct": round(100 * min(spc * len(class_names), n_train) / n_train, 1),
                "top1_mean": round(float(t.mean()), 4), "top1_std": round(float(t.std(ddof=1)), 4),
                "macro_f1_mean": round(float(m.mean()), 4),
                "retention_pct": round(100 * float(t.mean()) / FULL_REF, 1),
                "n_pool_mean": round(float(np.mean([r["n_pool"] for r in rows])), 1),
                "pool_prec_diag_mean": (round(float(np.mean([r["pool_oracle_precision_diag"] for r in rows
                                                              if r["pool_oracle_precision_diag"] is not None])), 4)
                                        if any(r["pool_oracle_precision_diag"] is not None for r in rows) else None),
                "per_seed_top1": [round(float(x), 4) for x in t],
            }
            g = grid[f"{arm}_spc{spc}"]
            print(f"[{arm} spc{spc}] top1={g['top1_mean']}±{g['top1_std']} ret={g['retention_pct']}% pool={g['n_pool_mean']}")

    # knee for V2: smallest spc with mean>=0.24, bracketed
    v2 = [(spc, grid[f"V2_consensus_quota_spc{spc}"]["top1_mean"]) for spc in SPCS]
    knee = None
    for i, (spc, val) in enumerate(v2):
        if val >= 0.24:
            if i == 0:
                knee = {"bracket": [None, spc], "first_spc_ge_24pct": spc}
            else:
                knee = {"bracket": [v2[i - 1][0], spc], "first_spc_ge_24pct": spc,
                        "interp_note": f"crosses 24% between spc{v2[i-1][0]} ({v2[i-1][1]:.3f}) and spc{spc} ({val:.3f})"}
            break
    if knee is None:
        knee = {"first_spc_ge_24pct": None, "note": "no spc up to 12 reaches 24% mean top-1"}

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-BUDGET-PREREG-001 (descriptive; no pass/fail)",
        "full_ref_top1": FULL_REF, "grid": grid, "v2_knee": knee,
        "claim_discipline": "exploratory; a <=20%-label knee would need a SEPARATE confirmatory pre-registration + user ruling before any positive canine claim",
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p17-budget-curve-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print("V2 knee:", json.dumps(knee, ensure_ascii=False))


if __name__ == "__main__":
    main()
