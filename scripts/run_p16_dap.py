# -*- coding: utf-8 -*-
"""P2 APTv2 域自适应预训练骨干 — PSD-DAP-PREREG-001 执行（ADR 0008）。

流程（判据跑前冻结，见协议 §3-4）:
  0. 保真自检: 原 Y_CKPT 特征全监督 LR 参照（应≈33.93%）。
  1. DAP: Y_CKPT 起点，503 条 APTv2 canidae (15→30 重采样,24,3) 无标签 clip 两视图 InfoNCE 续训。
  2. 遗忘自检: DAP 特征全监督参照 <30% → 标 catastrophic_forgetting，下游 delta 按混杂解读。
  3. DAP 特征 + P1 最优 V2 门，spc2 × seeds 42-51，端到端 val top-1。
  4. 判据: RESCUE≥20%&胜≥8/10 vs P1-V2(14.11%); PARTIAL 15-20; NULL<15。
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p07_endtoend_ak import HEAD_CFG, KW, Y_CKPT, extract_features, load_dataset  # noqa: E402
from run_p15_align import run_one_align, ARMS  # noqa: E402
from run_r16_endtoend_pseudo import _pick_seeds_str, evaluate_str  # noqa: E402
from psd.training.stgcnbc_feature_extractor import STGCNBCFeatureExtractor, center_keypoints  # noqa: E402

POOL = REPO / "runs/data_campaign/unified/real_expansion_pool_v1.pkl"
SEEDS10 = tuple(range(42, 52))
P1V2_BASELINE = 0.1411  # P1 V2 mean top-1 (reports/p15-label-alignment-2026-09-05.json)
OUT_DIR = REPO / "reports"


def load_aptv2_clips():
    """503 条 APTv2 pretrain_geometric (15,24,3) 原始像素 → harmonize(bbox归一+死掩码,
    round2 同款) → 重采样 T=30 → center。GT 无关。"""
    import torch
    from public_real_round2_lib import harmonize_aptv2_keypoints
    entries = pickle.load(open(POOL, "rb"))
    entries = entries.get("entries", entries) if isinstance(entries, dict) else entries
    apt = [e for e in entries
           if e.get("usage_scope") == "pretrain_geometric" and "aptv2" in str(e.get("source_channel"))]
    arrs = []
    for e in sorted(apt, key=lambda x: str(x.get("sample_id", x.get("sequence_id", "")))):
        kp = np.asarray(e["keypoints"], dtype=np.float32)  # (15,24,3) raw pixel
        kp = harmonize_aptv2_keypoints(kp)                  # bbox [0,1] + deadmask
        if kp.shape[0] != 30:
            t = torch.from_numpy(kp.transpose(1, 2, 0).reshape(24 * 3, -1)[np.newaxis])  # (1,72,T)
            t = torch.nn.functional.interpolate(t.float(), size=30, mode="linear", align_corners=False)
            kp = t.numpy()[0].reshape(24, 3, 30).transpose(2, 0, 1).astype(np.float32)  # (30,24,3)
        arrs.append(kp)
    X = np.stack(arrs)
    X = np.stack([center_keypoints(X[i]) for i in range(len(X))])
    return X.astype(np.float32)


def full_ref(emb, labels_str, splits, class_names):
    from sklearn.linear_model import LogisticRegression
    tm = (splits == "train")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb[tm], labels_str[tm])
    return evaluate_str(clf, emb, labels_str, splits == "val", class_names)["top1"]


def main():
    import torch
    t0 = time.time()
    data, kp, labels, splits = load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})

    # 0. fidelity self-check on original Y_CKPT
    f_orig = extract_features(kp, "warm")
    ref_orig = full_ref(f_orig, labels_str, splits, class_names)
    print(f"[p16] fidelity original Y_CKPT full-ref = {ref_orig}")

    # 1. DAP
    clips = load_aptv2_clips()
    print(f"[p16] APTv2 DAP clips {clips.shape}")
    from psd.training.dap_infonce import load_backbone, train_dap
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_backbone(Y_CKPT, num_classes=22, device=device)
    curve = train_dap(clips, model, epochs=60, batch=32, lr=1e-4, temperature=0.1,
                      seed=42, device=device, log_every=15)

    # re-extract AK features with DAP backbone
    ex = STGCNBCFeatureExtractor(model, device=device)
    f_dap = []
    for i in range(0, len(kp), 32):
        f_dap.append(ex.extract(kp[i:i + 32]))
    f_dap = np.vstack(f_dap)

    # 2. forgetting check
    ref_dap = full_ref(f_dap, labels_str, splits, class_names)
    forgetting = ref_dap < 0.30
    print(f"[p16] DAP full-ref = {ref_dap} (forgetting={forgetting})")

    # 3. V2 spc2 x 10 seeds on DAP features
    rows = [run_one_align(f_dap, labels_str, splits, class_names, ARMS["V2_consensus_quota"], s, spc=2)
            for s in SEEDS10]
    t = np.array([r["top1"] for r in rows])
    mean = float(t.mean())
    # paired wins vs P1-V2 per-seed baseline (load p15)
    p15 = json.load(open(OUT_DIR / "p15-label-alignment-2026-09-05.json", encoding="utf-8"))
    p1v2 = np.array([r["top1"] for r in p15["runs"]["V2_consensus_quota"]])
    wins = int((t > p1v2).sum())
    try:
        from scipy.stats import wilcoxon
        p = float(wilcoxon(t, p1v2, zero_method="zsplit")[1])
    except Exception:
        p = None

    if mean >= 0.20 and wins >= 8:
        verdict = "RESCUE"
    elif mean >= 0.15:
        verdict = "PARTIAL"
    else:
        verdict = "NULL"

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-DAP-PREREG-001",
        "fidelity": {"original_full_ref": ref_orig, "dap_full_ref": ref_dap,
                     "catastrophic_forgetting": forgetting},
        "dap_loss_curve": curve,
        "v2_dap": {"top1_mean": round(mean, 4), "top1_std": round(float(t.std(ddof=1)), 4),
                   "per_seed_top1": [round(float(x), 4) for x in t],
                   "macro_f1_mean": round(float(np.mean([r["macro_f1"] for r in rows])), 4),
                   "n_pool_mean": round(float(np.mean([r["n_pool"] for r in rows])), 1)},
        "vs_P1V2": {"p1v2_baseline": P1V2_BASELINE, "mean_delta_pp": round((mean - P1V2_BASELINE) * 100, 2),
                    "paired_wins": wins, "wilcoxon_p": round(p, 4) if p else None},
        "decision": {"verdict": verdict,
                     "rule": "RESCUE mean>=20%&wins>=8/10; PARTIAL 15-20%; NULL <15%"},
        "config_echo": {"clips": int(clips.shape[0]), "epochs": 60, "lr": 1e-4, "temperature": 0.1,
                        "seed": 42, "downstream_seeds": list(SEEDS10), "device": device},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p16-dap-aptv2-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps({"fidelity": result["fidelity"], "v2_dap_mean": result["v2_dap"]["top1_mean"],
                      "vs_P1V2": result["vs_P1V2"], "decision": result["decision"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
