# -*- coding: utf-8 -*-
"""E9 NTU60 arm (b) 10-seed 扩容 — E9 系列惯例（先例: E9d PanAf500, E9b/c NTU120/UCF101）。

背景: R22 审计发现 E9 是四点中唯一仍停 3-seed 的保留率点（67.5%±0.15, GENERALIZES）,
而正文已把 "E9-series convention"（3→10 seeds 扩容）写成惯例且 PanAf 先例证明种子数是
结论变量（3-seed CONFIRMS → 10-seed PARTIAL）。同系列必须同口径——否则是审稿人必抓的
不一致。协议 PSD-NTU-PREREG-001 判据/臂/预算/10% 子集（seed 42 冻结）零改动, 仅
selftrain 随机性从 3 seeds (42/43/44) 扩到 10 seeds (42-51); (a)/(c) 纯监督臂重算入
工件保持自含（确定性, 数字应与 09-05 工件一致——不一致即红旗, 人工核查）。

用法:
    python scripts/run_r16_ntu_pseudo_10seed.py
产出:
    reports/r16-ntu-pseudo-10seed-<date>.json
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

from run_p07_endtoend_ak import evaluate  # noqa: E402
from run_p14_ntu_lowres import (  # noqa: E402
    CLASS_NAMES,
    HEAD_CFG,
    KW,
    NPZ,
    stratified_10pct,
    train_linear_head_scaled,
)
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

SEEDS_10 = tuple(range(42, 52))
KW_R16 = dict(KW)
KW_R16["precision_stop"] = False
OUT_DIR = REPO / "reports"


def main():
    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    labels_str = np.array([str(int(v)) for v in tr_lab])
    print(f"[ntu] train {tr_feat.shape} val {va_feat.shape}", flush=True)

    # (c) 全预算参照（纯监督; 确定性, 供与 09-05 工件一致性自检）
    clf_full = train_linear_head_scaled(tr_feat, tr_lab, np.ones(len(tr_lab), bool), 60)
    ev_full = evaluate(clf_full, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[c] full linear head: top1={ev_full['top1']}", flush=True)

    anchor = stratified_10pct(tr_lab, 42)
    print(f"[budget] 10% subset: {int(anchor.sum())} clips", flush=True)

    # (a) 纯线性头 @10%（纯监督, 确定性）
    clf_a = train_linear_head_scaled(tr_feat, tr_lab, anchor, 60)
    ev_a = evaluate(clf_a, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[a] linear@10%: top1={ev_a['top1']}", flush=True)

    # (b) R16 修正协议 @ 10 seeds: 种子真标签 ∪ 池伪标签, 无 oracle 停止
    universe = ~anchor
    rows_b = []
    for seed in SEEDS_10:
        r = run_selftrain(tr_feat, labels_str, anchor, run_seed=seed, class_names=CLASS_NAMES,
                          head_cfg=HEAD_CFG, pool_universe_mask=universe, **KW_R16)
        pool_idx = r["final_pool_idx"]
        pseudo = np.array([int(s) for s in r["final_pred_full"][pool_idx]])
        train_mask = anchor.copy()
        train_mask[pool_idx] = True
        y_train = tr_lab.copy()
        y_train[pool_idx] = pseudo
        clf_b = train_linear_head_scaled(tr_feat, y_train, train_mask, 60)
        ev_b = evaluate(clf_b, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
        rows_b.append({"seed": seed, "n_pool": int(len(pool_idx)), "rounds": len(r["rounds"]),
                       "stop": r["stop_reason"],
                       "pool_oracle_precision_diag": r["rounds"][-1].get("precision"), **ev_b})
        print(f"[b] seed{seed}: top1={ev_b['top1']} mf1={ev_b['macro_f1']} "
              f"pool={len(pool_idx)} stop={r['stop_reason']}", flush=True)

    tb = [x["top1"] for x in rows_b]
    retention = float(np.mean(tb)) / ev_full["top1"]
    if retention >= 0.90:
        verdict = "GENERALIZES (retention >= 90%)"
    elif retention >= 0.85:
        verdict = "PARTIAL (85-90%)"
    else:
        verdict = "ANIMAL_DOMAIN_SPECIFIC (<85%, reported as boundary)"

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-NTU-PREREG-001 arm (b), 10-seed expansion per E9-series convention "
                    "(precedents: E9d PanAf500, E9b/c NTU120/UCF101); R16-corrected head "
                    "= seeds(true) + pool(PSEUDO); precision-drop stopping disabled; "
                    "arms (a)/(c) recomputed (deterministic) for a self-contained artifact",
        "caliber_note": "Supersedes the 3-seed caliber (r16-ntu-pseudo-2026-09-05.json) as the "
                        "reported number if verdicts agree; any verdict flip follows the "
                        "pre-registered rule on the 10-seed caliber (E9d precedent).",
        "layer": "public_human_benchmark",
        "disclosures": [
            "HEAD_CFG device cpu->cuda for 36k pool (adaptation #1; other hyperparams identical to E7)",
            "final linear head uses StandardScaler + LR max_iter=1000 tol=1e-3 (adaptation #2)",
            "10% subset fixed at seed 42 per protocol; selftrain stochasticity over 10 seeds (42-51)",
            "pool_oracle_precision_diag is a released diagnostic computed AFTER training; never in the control path",
        ],
        "arms": {"c_full_linear": ev_full, "a_linear_10pct": ev_a, "b_selftrain_10pct": rows_b},
        "endpoints": {
            "retention_b_over_c": round(retention, 4),
            "verdict": verdict,
            "linear_only_retention_a_over_c": round(ev_a["top1"] / ev_full["top1"], 4),
            "pseudo_iteration_gain_pp": round((float(np.mean(tb)) - ev_a["top1"]) * 100, 2),
        },
        "consistency_check": {
            "note": "(a)/(c) are deterministic; mismatch vs the 09-05 artifact is a red flag requiring manual inspection",
            "ref_2026_09_05": {"c_full": 0.7445, "a_linear": 0.6605},
        },
        "config_echo": {"npz": str(NPZ), "weights": "runs/ntu_phaseB/joint_pretext/epoch300_model.pt",
                        "n_train": int(len(tr_lab)), "n_val": int(len(va_lab)),
                        "n_budget": int(anchor.sum()), "seeds": list(SEEDS_10)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r16-ntu-pseudo-10seed-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)", flush=True)
    print(json.dumps(result["endpoints"], ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
