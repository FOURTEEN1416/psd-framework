"""P1.2 AK v2 复现/鲁棒层实验 — 预注册协议 PSD-AKV2-PREREG-001 §3 终点执行。

在 v2 多段数据集(352 clips, 8 类有样本)上重跑 E7/E8/P0.8 三臂 × 10 seeds,
执行 EP2(复现)与 EP3(天花板假设检验): v2 full-supervision vs v1 33.93%。
判据冻结: ≥+3.0pp=数据瓶颈成立 / ±3.0pp=任务内禀天花板 / ≤-3.0pp=v2 退化+漏斗分析。

口径披露: v2 为 8 类空间(bark/sit 段未过 0.80 一致性门, sit 仅 2 样本保留在池中);
spc2 在 v2 = 16/256 ≈ 6% 标注比例(绝对预算与 v1 同为 2 片段/类)。
v1 数字不替换, v2 为并列复现层。

用法:
    .venv/Scripts/python.exe scripts/run_p12_ak_v2_replicate.py
产出:
    reports/p12-akv2-replication-<date>.json / .md
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p07_endtoend_ak import extract_features, run_one  # noqa: E402
from run_p08_aimclr_arm import build_aimclr_views, extract_aimclr_features  # noqa: E402
from run_p10_seedexpansion import paired_stats  # noqa: E402

PKL_V2 = REPO / "runs" / "public_real_dataset" / "full12v2_T30.pkl"
SEEDS10 = tuple(range(42, 52))
V1_CEILING = 0.3393  # p07 full-supervision top-1 (deterministic)
OUT_DIR = REPO / "reports"


def load_v2():
    with open(PKL_V2, "rb") as f:
        data = pickle.load(f)
    kp = np.stack([np.asarray(d["keypoints"], dtype=np.float32) for d in data])
    labels = np.array([int(d["label"]) for d in data])
    splits = np.array([d["split"] for d in data])
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    return kp, labels, labels_str, splits, class_names


def main():
    t0 = time.time()
    kp, labels, labels_str, splits, class_names = load_v2()
    print(f"[v2] {len(kp)} clips | train={int((splits=='train').sum())} val={int((splits=='val').sum())} | classes={class_names}")

    print("[feat] warm...")
    f_warm = extract_features(kp, "warm")
    print("[feat] aimclr...")
    f_aimclr = extract_aimclr_features(build_aimclr_views(kp))

    results = {}
    for arm, emb in (("warm", f_warm), ("aimclr", f_aimclr)):
        for spc in (2, 4, -1):
            rows = []
            for seed in SEEDS10:
                r = run_one(emb, labels, labels_str, splits, class_names, arm, spc, seed)
                rows.append(r)
                print(f"[{arm} spc{spc}] seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')}")
            results[f"{arm}_spc{spc}"] = rows
    rows = []
    for seed in SEEDS10:
        torch.manual_seed(seed)
        f_scr = extract_features(kp, "scratch")
        r = run_one(f_scr, labels, labels_str, splits, class_names, "scratch", 2, seed)
        rows.append(r)
        print(f"[scratch spc2] seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')}")
    results["scratch_spc2"] = rows

    def top1(key):
        return np.array([r["top1"] for r in results[key] if "top1" in r])

    def mf1(key):
        return np.array([r["macro_f1"] for r in results[key] if "macro_f1" in r])

    summary = {}
    for k, rows_ in results.items():
        t = [r["top1"] for r in rows_ if "top1" in r]
        m = [r["macro_f1"] for r in rows_ if "macro_f1" in r]
        summary[k] = {"n": len(t), "top1_mean": round(float(np.mean(t)), 4), "top1_std": round(float(np.std(t, ddof=1)), 4),
                      "macro_f1_mean": round(float(np.mean(m)), 4), "macro_f1_std": round(float(np.std(m, ddof=1)), 4)}

    tests = {}
    for spc in (2, 4):
        for metric, fn in (("top1", top1), ("macro_f1", mf1)):
            wa, aa = fn(f"warm_spc{spc}"), fn(f"aimclr_spc{spc}")
            n = min(len(wa), len(aa))
            tests[f"warm_vs_aimclr_spc{spc}_{metric}"] = paired_stats(wa[:n], aa[:n])
    for metric, fn in (("top1", top1), ("macro_f1", mf1)):
        w, s = fn("warm_spc2"), fn("scratch_spc2")
        n = min(len(w), len(s))
        tests[f"warm_vs_scratch_spc2_{metric}"] = paired_stats(w[:n], s[:n])

    # EP3 天花板检验（判据冻结于协议 §4）
    full_mean = summary["warm_spc-1"]["top1_mean"]
    delta_pp = (full_mean - V1_CEILING) * 100
    if delta_pp >= 3.0:
        verdict = "DATA_BOTTLENECK_CONFIRMED (v2 full >= v1 + 3pp)"
    elif delta_pp <= -3.0:
        verdict = "V2_DEGRADATION (funnel analysis required)"
    else:
        verdict = "TASK_INTRINSIC_CEILING (within +/-3pp)"

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-AKV2-PREREG-001 EP2/EP3",
        "layer": "public_real_v2",
        "disclosures": [
            "v2 is an 8-class space (bark absent; sit n=2) after the 0.80 frame-consistency gate",
            "spc2 on v2 = 16/256 train clips ~ 6% label fraction (absolute budget 2 clips/class identical to v1)",
            "v1 numbers are NOT replaced; v2 is a parallel replication tier",
            "Y_CKPT warm-start saw v1 train videos; v2 adds new segments of the same videos to training pool (inherent to task-pretraining, val videos disjoint)",
        ],
        "config_echo": {"pkl": str(PKL_V2), "seeds": list(SEEDS10), "classes": class_names,
                        "n_train": int((splits == "train").sum()), "n_val": int((splits == "val").sum())},
        "summary": summary, "paired_tests": tests,
        "ep3_ceiling_test": {"v2_full_top1": full_mean, "v1_ceiling": V1_CEILING,
                             "delta_pp": round(delta_pp, 2), "verdict": verdict},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p12-akv2-replication-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps({"summary": summary, "ep3": result["ep3_ceiling_test"]}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
