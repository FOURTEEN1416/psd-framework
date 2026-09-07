# -*- coding: utf-8 -*-
"""Post-hoc 诊断（非预注册终点）——NTU60 10% 子集重抽方差（E9 披露缺口量化）。

背景: E9/§4.4 披露 "the ±0.24 spread covers self-training seeds but not subset
resampling"——本诊断量化后者: 用 4 个替代子集种子(43-46)重抽 10% 分层子集,
重训 (a) linear 头（与 E9 同配置 ScaledLR），报告跨子集 spread。
确定性参照 (c) 无需重跑（求解器无随机性）。
产出: reports/r23-subset-resample-<date>.json
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

from run_p14_ntu_lowres import (  # noqa: E402
    CLASS_NAMES,
    NPZ,
    stratified_10pct,
    train_linear_head_scaled,
)
from run_p07_endtoend_ak import evaluate  # noqa: E402

SUBSET_SEEDS = (43, 44, 45, 46)   # 42 号=原 E9 子集，不重抽
OUT_DIR = REPO / "reports"


def main():
    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    print(f"[data] train {tr_feat.shape} val {va_feat.shape}", flush=True)

    # 原 seed-42 子集的 (a) 臂（E9 报告值 66.05%）重跑一次作锚点自检
    anchor42 = stratified_10pct(tr_lab, 42)
    clf42 = train_linear_head_scaled(tr_feat, tr_lab, anchor42, 60)
    ev42 = evaluate(clf42, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[a] seed42 (E9 subset): top1={ev42['top1']} (expect 0.6605)", flush=True)

    rows = [{"subset_seed": 42, "top1": round(float(ev42["top1"]), 4), "role": "original"}]
    for s in SUBSET_SEEDS:
        a = stratified_10pct(tr_lab, s)
        clf = train_linear_head_scaled(tr_feat, tr_lab, a, 60)
        ev = evaluate(clf, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
        rows.append({"subset_seed": s, "top1": round(float(ev["top1"]), 4), "role": "resample"})
        print(f"[a] subset_seed{s}: top1={ev['top1']}", flush=True)

    resample = [r["top1"] for r in rows if r["role"] == "resample"]
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "purpose": "Post-hoc robustness diagnostic (NOT a pre-registered endpoint): quantifies the "
                   "subset-resampling variance component disclosed in E9/§4.4 ('spreads cover "
                   "self-training seeds but not subset resampling').",
        "layer": "public_human_benchmark",
        "rows": rows,
        "stats": {
            "original_seed42": rows[0]["top1"],
            "resample_mean": round(float(np.mean(resample)), 4),
            "resample_std_ddof1": round(float(np.std(resample, ddof=1)), 4),
            "n_resamples": len(resample),
        },
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r23-subset-resample-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}", flush=True)
    print(json.dumps(result["stats"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
