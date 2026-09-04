"""P1.0 种子扩容 — E7/E8/P0.8 关键对比 n=3 → n=10 + 配对显著性检验。

任务背景: PR 审稿风险"n=3 未达显著"。特征提取确定性(骨干固定), 协议随机性
来自 seed(锚点选择/伪标签迭代), 故扩容只需重跑 run_one 层——分钟级。
AimCLR 特征亦确定性(骨干固定), 同法扩容。scratch 臂每 seed 独立随机初始化
(诚实口径: 随机骨干本身是随机变量)。

统计: 同 seed 配对差(warm-aimclr / warm-scratch / aimclr-scratch),
Wilcoxon signed-rank(n=10 最小双侧 p=0.002)+ 符号计数。措辞纪律:
p<0.05 才可写 significant; 否则维持"方向一致未达显著"。

用法:
    .venv/Scripts/python.exe scripts/run_p10_seedexpansion.py
产出:
    reports/p10-seedexpansion-<date>.json / .md
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

from run_p07_endtoend_ak import (  # noqa: E402
    extract_features,
    load_dataset,
    run_one,
)
from run_p08_aimclr_arm import build_aimclr_views, extract_aimclr_features  # noqa: E402

SEEDS10 = tuple(range(42, 52))
OUT_DIR = REPO / "reports"


def paired_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """配对 Wilcoxon + 符号计数。scipy 缺失时退化为精确符号检验。"""
    d = a - b
    wins = int((d > 0).sum()); losses = int((d < 0).sum())
    out = {"mean_delta_pp": round(float(d.mean()) * 100, 2),
           "std_delta_pp": round(float(d.std(ddof=1)) * 100, 2),
           "wins": wins, "losses": losses, "ties": int(len(d) - wins - losses)}
    try:
        from scipy.stats import wilcoxon
        stat, p = wilcoxon(a, b, zero_method="zsplit")
        out["wilcoxon_p"] = round(float(p), 4)
    except Exception:
        from math import comb
        n = wins + losses
        k = min(wins, losses)
        p = sum(comb(n, i) for i in range(k + 1)) / (2 ** n) * 2
        out["sign_test_p"] = round(min(1.0, p), 4)
    return out


def main():
    t0 = time.time()
    data, kp, labels, splits = load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})

    print("[feat] warm...")
    f_warm = extract_features(kp, "warm")
    print("[feat] aimclr...")
    f_aimclr = extract_aimclr_features(build_aimclr_views(kp))

    results = {}
    for arm, emb in (("warm", f_warm), ("aimclr", f_aimclr)):
        for spc in (2, 4):
            rows = []
            for seed in SEEDS10:
                r = run_one(emb, labels, labels_str, splits, class_names, arm, spc, seed)
                rows.append(r)
                print(f"[{arm} spc{spc}] seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')}")
            results[f"{arm}_spc{spc}"] = rows

    # scratch 臂: 每 seed 独立随机骨干(诚实口径), 仅 spc2(与 p07 主对比同档)
    import torch
    rows = []
    for seed in SEEDS10:
        torch.manual_seed(seed)  # 随机骨干初始化可复现
        f_scr = extract_features(kp, "scratch")  # build_stgcn_bc 用当前 torch RNG=每 seed 新初始化
        r = run_one(f_scr, labels, labels_str, splits, class_names, "scratch", 2, seed)
        rows.append(r)
        print(f"[scratch spc2] seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')}")
    results["scratch_spc2"] = rows

    def top1(key):
        return np.array([r["top1"] for r in results[key] if "top1" in r])

    def mf1(key):
        return np.array([r["macro_f1"] for r in results[key] if "macro_f1" in r])

    tests = {}
    for spc in (2, 4):
        for metric, fn in (("top1", top1), ("macro_f1", mf1)):
            wa = fn(f"warm_spc{spc}"); aa = fn(f"aimclr_spc{spc}")
            n = min(len(wa), len(aa))
            tests[f"warm_vs_aimclr_spc{spc}_{metric}"] = paired_stats(wa[:n], aa[:n])
    for metric, fn in (("top1", top1), ("macro_f1", mf1)):
        w = fn("warm_spc2"); s = fn("scratch_spc2")
        n = min(len(w), len(s))
        tests[f"warm_vs_scratch_spc2_{metric}"] = paired_stats(w[:n], s[:n])

    summary = {}
    for k, rows in results.items():
        t = [r["top1"] for r in rows if "top1" in r]
        m = [r["macro_f1"] for r in rows if "macro_f1" in r]
        summary[k] = {"n": len(t), "top1_mean": round(float(np.mean(t)), 4), "top1_std": round(float(np.std(t, ddof=1)), 4),
                      "macro_f1_mean": round(float(np.mean(m)), 4), "macro_f1_std": round(float(np.std(m, ddof=1)), 4)}

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real",
        "protocol": "p07/p08 end-to-end arms at n=10 seeds (42..51); scratch re-inits backbone per seed; paired tests on same-seed deltas",
        "summary": summary, "paired_tests": tests,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p10-seedexpansion-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps({"summary": summary, "tests": tests}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
