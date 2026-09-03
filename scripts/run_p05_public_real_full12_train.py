# -*- coding: utf-8 -*-
"""AK 公开真实层 12 类全类 两臂对比训练驱动（R1 补强, 预注册 BOARD 09-03）.

协议（与 Q3c 逐字一致, 唯一差异变量 = backbone 权重来源）:
  - warm 臂:  load Y best.pt backbone + 双层冻结 + head 重训（= Q3c 协议原样）
  - scratch 臂: 随机初始化 backbone + 同样冻结 + head 重训
  - 两臂 × 3 seeds(42/43/44), epochs=60 早停 15, batch 16, AMP
指标: overall best_val_acc + macro-F1 + per-class P/R/F1（类不平衡如实强制同框）。
产出: reports/p05-public-real-full12-twoarm-<日期>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from psd.models.stgcn_bc import STGCNBC  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402
from psd.data.ak_mapping import MAPPED_PSD_CLASSES  # noqa: E402
from run_c1_decouple import freeze_backbone, load_y_backbone  # noqa: E402

SEEDS = (42, 43, 44)


def macro_prf(y_true, y_pred, n_classes: int):
    """宏平均 P/R/F1（零除类计 0，如实）."""
    per_f1 = []
    per_p, per_r = [], []
    for c in range(n_classes):
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_p.append(prec); per_r.append(rec); per_f1.append(f1)
    return float(np.mean(per_p)), float(np.mean(per_r)), float(np.mean(per_f1))


def collect_preds(model, val, device):
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for s in val:
            x = torch.from_numpy(np.asarray(s["keypoints"], dtype=np.float32)).unsqueeze(0)
            logits, _ = model(x.to(device))
            yp.append(int(logits.argmax(dim=1).item()))
            yt.append(int(s["label"]))
    return yt, yp


def per_class_acc(y_true, y_pred, names):
    hits = {}
    for t, p in zip(y_true, y_pred):
        hits.setdefault(t, []).append(t == p)
    return {names[c]: round(float(np.mean(v)), 4) for c, v in sorted(hits.items())}


def run_one(arm: str, seed: int, train, val, n_classes: int, args, device: str):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = STGCNBC(in_channels=3, num_classes=n_classes)
    if arm == "warm":
        info = load_y_backbone(model, REPO / args.init)
        print(f"  [init] warm seed={seed} missing(head)={len(info.missing)} unexpected={len(info.unexpected)}")
    else:
        print(f"  [init] scratch seed={seed} (random backbone)")
    freeze_backbone(model)
    tc = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
                     seed=seed, device=device, output_dir=str(REPO / f"{args.output_dir}/{arm}_s{seed}"),
                     use_amp=True)
    trainer = STGCNBCTrainer(model, train, val, config=tc)
    fit = trainer.fit()
    yt, yp = collect_preds(model, val, tc.device if tc.device != "auto"
                           else ("cuda" if torch.cuda.is_available() else "cpu"))
    mp, mr, mf1 = macro_prf(yt, yp, n_classes)
    return {
        "arm": arm, "seed": seed, "best_val_acc": fit["best_val_acc"],
        "macro_P": round(mp, 4), "macro_R": round(mr, 4), "macro_F1": round(mf1, 4),
        "per_class_val_acc": per_class_acc(yt, yp, MAPPED_PSD_CLASSES),
        "best_epoch": fit.get("best_epoch"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="runs/public_real_dataset/full12_T30.pkl")
    ap.add_argument("--init", default="runs/p05_stgcn_bc_full/best.pt")
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--output-dir", default="runs/public_real_finetune_full12")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    with open(REPO / args.pkl, "rb") as f:
        data = pickle.load(f)
    train = [d for d in data if d.get("split") == "train"]
    val = [d for d in data if d.get("split") == "val"]
    dist = Counter(d["psd_class"] for d in data)
    print(f"[data] train={len(train)} val={len(val)} dist={dist}")
    assert max(d["label"] for d in data) < args.num_classes, "label 越界——pkl 与 num-classes 不符"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    for arm in ("warm", "scratch"):
        for seed in SEEDS:
            t0 = time.time()
            r = run_one(arm, seed, train, val, args.num_classes, args, device)
            r["wall_clock_sec"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"  [done] {arm} s{seed}: acc={r['best_val_acc']:.4f} macroF1={r['macro_F1']:.4f} ({r['wall_clock_sec']}s)")

    def agg(arm, key):
        vals = [r[key] for r in runs if r["arm"] == arm]
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4),
                "values": vals}

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real",
        "protocol": "AK full12(9 classes with samples) frozen-backbone head-retrain two-arm; "
                    "warm=Y best.pt backbone, scratch=random backbone, both frozen (single variable = backbone weights)",
        "classes_in_space": MAPPED_PSD_CLASSES,
        "config_echo": {"pkl": args.pkl, "num_classes": args.num_classes, "epochs": args.epochs,
                        "batch_size": args.batch_size, "patience": args.patience,
                        "seeds": list(SEEDS), "n_train": len(train), "n_val": len(val),
                        "class_dist": dist},
        "runs": runs,
        "agg": {arm: {"best_val_acc": agg(arm, "best_val_acc"),
                       "macro_F1": agg(arm, "macro_F1")} for arm in ("warm", "scratch")},
    }
    out = REPO / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    for arm in ("warm", "scratch"):
        a = result["agg"][arm]
        print(f"[done] {arm}: acc={a['best_val_acc']['mean']}±{a['best_val_acc']['std']} "
              f"macroF1={a['macro_F1']['mean']}±{a['macro_F1']['std']}")


if __name__ == "__main__":
    main()
