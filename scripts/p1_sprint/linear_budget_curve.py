# -*- coding: utf-8 -*-
"""P1 头对头补充 — PSD 线性头（冻结特征）在 5%/10%/20% 预算的确定性读数.

目的: 补全 tab:h2h 预算列中 PSD linear head 行（E9 arm-a 同实现, ScaledLR,
无随机性 → 单值）。10% 处必须复现已报告的 66.05%（一致性断言）。
复用: run_p14_ntu_lowres 的 ScaledLR/train_linear_head_scaled + evaluate。
产出: runs/p1_sprint/h2h_matrix/linear_budget_curve.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_p14_ntu_lowres as p14  # noqa: E402
from run_p07_endtoend_ak import evaluate  # noqa: E402

OUT = REPO / "runs" / "p1_sprint" / "h2h_matrix" / "linear_budget_curve.json"


def stratified(labels: np.ndarray, seed: int, frac: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in range(60):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * frac)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    return mask


def main() -> None:
    z = np.load(p14.NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    rows = []
    for frac in (0.05, 0.10, 0.20):
        anchor = stratified(tr_lab, 42, frac)
        clf = p14.train_linear_head_scaled(tr_feat, tr_lab, anchor, 60)
        ev = evaluate(clf, va_feat, va_lab, np.ones(len(va_lab), bool), p14.CLASS_NAMES)
        rows.append({"frac": frac, "n_anchor": int(anchor.sum()),
                     "top1": round(ev["top1"], 4), "macro_f1": round(ev["macro_f1"], 4)})
        print(f"[linear] frac={frac} n={int(anchor.sum())} top1={ev['top1'] * 100:.2f}%")
    ref = [r for r in rows if r["frac"] == 0.10][0]
    assert abs(ref["top1"] - 0.6605) < 0.002, f"10% 线性头未复现 66.05%: {ref['top1']}"
    out = {"date": datetime.now().isoformat(timespec="seconds"),
           "method": "PSD linear head on frozen 256-d features (E9 arm-a; ScaledLR; deterministic)",
           "rows": rows,
           "consistency": {"reported_e9_linear_10pct": 0.6605, "reproduced": ref["top1"]}}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
