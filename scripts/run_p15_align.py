# -*- coding: utf-8 -*-
"""P1 标签对齐伪标签修正 — PSD-ALIGN-PREREG-001 主终点执行（ADR 0007）。

四臂（唯一差异=门控/分配机制，其余逐字同 R16 修正协议）:
  V0 对照   = R16 原样（standing 门，AK 类空间恒惰性）
  V1 全类锚点共识门 = 头路 top-1 == 原型路 top-1 方可入池
  V2 = V1 + 种子先验配额（每伪类按 κ 降序 top-K_c）
  V3 原型路主导 = 分配恒走原型路，头只训练不分配
主终点: AK v1 spc2（18/141≈13%）× seeds 42-51，端到端 val top-1 均值。
判据（跑前冻结）: ALIGNS=best mean≥20.0% 且对 V0 配对胜≥8/10; PARTIAL=15-20% 或胜 5-7; NULL=<15%。
全部 GT 无关: 最终头=种子真标签∪池伪标签; precision_stop=False。

用法:
    .venv/Scripts/python.exe scripts/run_p15_align.py
产出:
    reports/p15-label-alignment-<date>.json
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
from run_r16_endtoend_pseudo import _pick_seeds_str, evaluate_str  # noqa: E402
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

SEEDS10 = tuple(range(42, 52))
OUT_DIR = REPO / "reports"

ARMS = {
    "V0_control": dict(standing_mode="consensus", gate_mode="standing"),
    "V1_consensus_all": dict(standing_mode="none", gate_mode="consensus_all"),
    "V2_consensus_quota": dict(standing_mode="none", gate_mode="consensus_all_quota"),
    "V3_proto_primary": dict(standing_mode="none", gate_mode="proto_primary"),
}


def run_one_align(emb, labels_str, splits, class_names, arm_kwargs, seed):
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    anchor_mask = _pick_seeds_str(labels_str, splits, 2, rng)
    universe = (splits == "train") & ~anchor_mask
    kw = dict(KW)
    kw["precision_stop"] = False
    kw.update(arm_kwargs)
    r = run_selftrain(emb, labels_str, anchor_mask, run_seed=seed, class_names=class_names,
                      head_cfg=HEAD_CFG, pool_universe_mask=universe, **kw)
    pool_idx = r["final_pool_idx"]
    train_mask = anchor_mask.copy()
    train_mask[pool_idx] = True
    y_train = np.array([None] * len(labels_str), dtype=object)
    y_train[anchor_mask] = labels_str[anchor_mask]
    y_train[pool_idx] = r["final_pred_full"][pool_idx]
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(emb[train_mask], y_train[train_mask])
    ev = evaluate_str(clf, emb, labels_str, splits == "val", class_names)
    return {"seed": seed, "n_pool": int(len(pool_idx)), "rounds": len(r["rounds"]),
            "stop": r["stop_reason"],
            "pool_oracle_precision_diag": r["rounds"][-1].get("precision"),
            "top1": ev["top1"], "macro_f1": ev["macro_f1"]}


def main():
    t0 = time.time()
    data, kp, labels, splits = load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    print(f"[p15] {len(data)} clips train={int((splits=='train').sum())} classes={class_names}")
    print("[p15] warm features...")
    f_warm = extract_features(kp, "warm")

    results = {}
    for arm, kwargs in ARMS.items():
        rows = []
        for seed in SEEDS10:
            r = run_one_align(f_warm, labels_str, splits, class_names, kwargs, seed)
            rows.append(r)
            print(f"[{arm}] seed{seed}: top1={r['top1']} mf1={r['macro_f1']} pool={r['n_pool']} prec_diag={r['pool_oracle_precision_diag']}")
        results[arm] = rows

    summary = {}
    for arm, rows in results.items():
        t = np.array([r["top1"] for r in rows])
        m = np.array([r["macro_f1"] for r in rows])
        summary[arm] = {"n": len(t), "top1_mean": round(float(t.mean()), 4),
                        "top1_std": round(float(t.std(ddof=1)), 4),
                        "macro_f1_mean": round(float(m.mean()), 4),
                        "n_pool_mean": round(float(np.mean([r["n_pool"] for r in rows])), 1),
                        "pool_prec_diag_mean": round(float(np.mean([r["pool_oracle_precision_diag"] for r in rows])), 4)}

    # 配对 vs V0（同 seed）
    v0 = np.array([r["top1"] for r in results["V0_control"]])
    tests = {}
    for arm in ARMS:
        if arm == "V0_control":
            continue
        a = np.array([r["top1"] for r in results[arm]])
        d = a - v0
        wins = int((d > 0).sum()); losses = int((d < 0).sum())
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(a, v0, zero_method="zsplit")[1])
        except Exception:
            p = None
        tests[arm] = {"mean_delta_pp": round(float(d.mean()) * 100, 2),
                      "wins_vs_V0": wins, "losses_vs_V0": losses, "wilcoxon_p": round(p, 4) if p is not None else None}

    # 冻结判据
    best_arm = max((a for a in ARMS if a != "V0_control"), key=lambda a: summary[a]["top1_mean"])
    best_mean = summary[best_arm]["top1_mean"] * 100
    best_wins = tests[best_arm]["wins_vs_V0"]
    if best_mean >= 20.0 and best_wins >= 8:
        verdict = "ALIGNS"
    elif best_mean >= 15.0 or 5 <= best_wins <= 7:
        verdict = "PARTIAL"
    else:
        verdict = "NULL"

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-ALIGN-PREREG-001 primary endpoint (AK v1 spc2, 10 seeds, warm features)",
        "arms": ARMS, "summary": summary, "paired_vs_V0": tests,
        "decision": {"best_arm": best_arm, "best_mean_top1_pct": round(best_mean, 2),
                     "wins_vs_V0": best_wins, "verdict": verdict,
                     "rule": "ALIGNS: mean>=20% & wins>=8/10; PARTIAL: 15-20% or wins 5-7; NULL: <15%"},
        "config_echo": {"seeds": list(SEEDS10), "classes": class_names,
                        "n_train": int((splits == "train").sum()), "n_val": int((splits == "val").sum())},
        "runs": results,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p15-label-alignment-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps({"summary": summary, "tests": tests, "decision": result["decision"]},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
