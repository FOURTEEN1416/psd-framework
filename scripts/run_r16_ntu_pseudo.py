"""R16 协议修正重跑 — NTU E9 臂 (b) 最终头改用伪标签 + 无 oracle 停止。

背景同 run_r16_endtoend_pseudo.py（R16 CRITICAL F1/F2）: p14 的臂 (b) 把
run_selftrain 池片段用真实 NTU 标签重训 ScaledLR——最终分类器消费了
4009+34.4k ≈ 96% 训练标签，"99.5% 保留率 @10% 标注"与实现不符；且
precision-drop 停止消费了真标签精度（tex 却披露"NTU 无真值"）。
修正协议: 臂 (b) 最终头 = 种子真标签 ∪ 池伪标签; precision_stop=False。
臂 (a)/(c) 为纯监督参照，不涉伪标签，数字不变（重算以自含工件）。

用法:
    .venv/Scripts/python.exe scripts/run_r16_ntu_pseudo.py
产出:
    reports/r16-ntu-pseudo-<date>.json
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

from run_p07_endtoend_ak import evaluate  # int-label eval（与 p14 一致）  # noqa: E402
from run_p14_ntu_lowres import (  # noqa: E402
    BUDGET_FRAC,
    CLASS_NAMES,
    HEAD_CFG,
    KW,
    NPZ,
    SEEDS,
    stratified_10pct,
    train_linear_head_scaled,
)
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

KW_R16 = dict(KW)
KW_R16["precision_stop"] = False
OUT_DIR = REPO / "reports"


def main():
    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    labels_str = np.array([str(int(v)) for v in tr_lab])
    print(f"[ntu] train {tr_feat.shape} val {va_feat.shape}")

    # (c) 全预算参照（纯监督，不变）
    clf_full = train_linear_head_scaled(tr_feat, tr_lab, np.ones(len(tr_lab), bool), 60)
    ev_full = evaluate(clf_full, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[c] full linear head: top1={ev_full['top1']}")

    anchor = stratified_10pct(tr_lab, 42)
    print(f"[budget] 10% subset: {int(anchor.sum())} clips")

    # (a) 纯线性头 @10%（纯监督，不变）
    clf_a = train_linear_head_scaled(tr_feat, tr_lab, anchor, 60)
    ev_a = evaluate(clf_a, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[a] linear@10%: top1={ev_a['top1']}")

    # (b) R16 修正: 种子真标签 ∪ 池伪标签
    universe = ~anchor
    rows_b = []
    for seed in SEEDS:
        r = run_selftrain(tr_feat, labels_str, anchor, run_seed=seed, class_names=CLASS_NAMES,
                          head_cfg=HEAD_CFG, pool_universe_mask=universe, **KW_R16)
        pool_idx = r["final_pool_idx"]
        pseudo = np.array([int(s) for s in r["final_pred_full"][pool_idx]])
        train_mask = anchor.copy()
        train_mask[pool_idx] = True
        y_train = tr_lab.copy()
        y_train[pool_idx] = pseudo  # 池片段消费伪标签
        clf_b = train_linear_head_scaled(tr_feat, y_train, train_mask, 60)
        ev_b = evaluate(clf_b, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
        rows_b.append({"seed": seed, "n_pool": int(len(pool_idx)), "rounds": len(r["rounds"]),
                       "stop": r["stop_reason"],
                       "pool_oracle_precision_diag": r["rounds"][-1].get("precision"), **ev_b})
        print(f"[b] seed{seed}: top1={ev_b['top1']} mf1={ev_b['macro_f1']} pool={len(pool_idx)} stop={r['stop_reason']}")

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
        "protocol": "R16-corrected PSD-NTU-PREREG-001 arm (b): final head = seeds(true) + pool(PSEUDO); "
                    "precision-drop stopping disabled (no oracle control path); arms (a)/(c) unchanged supervised references",
        "correction_note": "Supersedes p14 arm (b) (true-label pool head + oracle-stopped iteration).",
        "layer": "public_human_benchmark",
        "disclosures": [
            "HEAD_CFG device cpu->cuda for 36k pool (adaptation #1; other hyperparams identical to E7)",
            "final linear head uses StandardScaler + LR max_iter=1000 tol=1e-3 (adaptation #2)",
            "10% subset fixed at seed 42 per protocol; selftrain stochasticity over 3 seeds",
            "pool_oracle_precision_diag is a released diagnostic computed AFTER training; it never enters the loop's control path (precision_stop=False)",
        ],
        "arms": {"c_full_linear": ev_full, "a_linear_10pct": ev_a, "b_selftrain_10pct": rows_b},
        "endpoints": {
            "retention_b_over_c": round(retention, 4),
            "verdict": verdict,
            "linear_only_retention_a_over_c": round(ev_a["top1"] / ev_full["top1"], 4),
            "pseudo_iteration_gain_pp": round((float(np.mean(tb)) - ev_a["top1"]) * 100, 2),
        },
        "config_echo": {"npz": str(NPZ), "weights": "runs/ntu_phaseB/joint_pretext/epoch300_model.pt",
                        "n_train": int(len(tr_lab)), "n_val": int(len(va_lab)),
                        "n_budget": int(anchor.sum()), "seeds": list(SEEDS)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r16-ntu-pseudo-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps(result["endpoints"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
