# -*- coding: utf-8 -*-
"""AK 12 类 端到端正面对照实验（R1 补强第二波, 用户裁决 2026-09-03）.

四臂 × 3 seeds(42/43/44) + 三个 trivial 基线, 同一 pkl/协议/预算:
  A1 warm+frozen   (已有 twoarm JSON, 此处复跑纳入统一口径)
  A2 scratch+frozen(已有)
  B1 warm+full     : 预训练 backbone, 全参数微调（不冻结）
  B2 scratch+full  : 随机 backbone, 全参数训练（= 传统从头蛮训基线）
  C1 majority      : 验证集最频类恒预测
  C2 random        : 均匀随机预测
  C3 flatten+LR    : (T,24,3) 展平 + LogisticRegression（sklearn, 证明/证伪时序模型必要性）

科学问题: "预训练管线优于从头蛮训"从推断变成实测; 冻结 vs 全参在两种初始化下的表现。
产出: reports/p05-public-real-full12-endtoend-<日期>.json
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
from run_p05_public_real_full12_train import collect_preds, macro_prf, per_class_acc  # noqa: E402

SEEDS = (42, 43, 44)


def run_nn(arm: str, seed: int, train, val, n_classes: int, args, device: str, freeze: bool):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = STGCNBC(in_channels=3, num_classes=n_classes)
    if arm.startswith("warm"):
        info = load_y_backbone(model, REPO / args.init)
        _ = len(info.missing)
    if freeze:
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
        "arm": arm, "seed": seed,
        "best_val_acc": fit["best_val_acc"], "macro_F1": round(mf1, 4),
        "macro_P": round(mp, 4), "macro_R": round(mr, 4),
        "per_class_val_acc": per_class_acc(yt, yp, MAPPED_PSD_CLASSES),
        "best_epoch": fit.get("best_epoch"),
        "frozen": freeze,
    }


def run_trivial(kind: str, seed: int, train, val, n_classes: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    ytr = [int(s["label"]) for s in train]
    yt = [int(s["label"]) for s in val]
    rng = np.random.default_rng(seed)
    if kind == "majority":
        maj = Counter(ytr).most_common(1)[0][0]
        yp = [maj] * len(yt)
    elif kind == "random":
        yp = [int(rng.integers(0, n_classes)) for _ in yt]
    elif kind == "flatten_lr":
        from sklearn.linear_model import LogisticRegression
        Xtr = np.stack([np.asarray(s["keypoints"], dtype=np.float64).ravel() for s in train])
        Xva = np.stack([np.asarray(s["keypoints"], dtype=np.float64).ravel() for s in val])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=seed)
        clf.fit(Xtr, ytr)
        yp = list(clf.predict(Xva))
    else:
        raise ValueError(kind)
    acc = float(np.mean([t == p for t, p in zip(yt, yp)]))
    mp, mr, mf1 = macro_prf(yt, yp, n_classes)
    return {"arm": kind, "seed": seed, "best_val_acc": round(acc, 4),
            "macro_F1": round(mf1, 4), "macro_P": round(mp, 4), "macro_R": round(mr, 4)}


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
    ap.add_argument("--arms", default="warm_full,scratch_full,majority,random,flatten_lr",
                    help="逗号分隔; warm_frozen/scratch_frozen 已在 twoarm JSON, 默认不复跑")
    args = ap.parse_args()

    with open(REPO / args.pkl, "rb") as f:
        data = pickle.load(f)
    train = [d for d in data if d.get("split") == "train"]
    val = [d for d in data if d.get("split") == "val"]
    dist = Counter(d["psd_class"] for d in data)
    print(f"[data] train={len(train)} val={len(val)} dist={dist}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = []
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for arm in arms:
        for seed in SEEDS:
            t0 = time.time()
            if arm in ("warm_full", "scratch_full"):
                r = run_nn(arm, seed, train, val, args.num_classes, args, device,
                           freeze=False)
            elif arm == "warm_frozen":
                r = run_nn(arm, seed, train, val, args.num_classes, args, device, freeze=True)
            else:
                r = run_trivial(arm, seed, train, val, args.num_classes)
            r["wall_clock_sec"] = round(time.time() - t0, 1)
            runs.append(r)
            print(f"  [done] {arm} s{seed}: acc={r['best_val_acc']:.4f} macroF1={r['macro_F1']:.4f} ({r['wall_clock_sec']}s)")

    def agg(arm, key):
        vals = [r[key] for r in runs if r["arm"] == arm]
        if not vals:
            return None
        return {"mean": round(float(np.mean(vals)), 4), "std": round(float(np.std(vals)), 4), "values": vals}

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real",
        "protocol": "AK full12 end-to-end comparison: frozen two-arm (see twoarm JSON) + full-finetune two-arm + trivial baselines; "
                    "identical pkl/budget/seeds; single variable per pair as labeled",
        "classes_in_space": MAPPED_PSD_CLASSES,
        "config_echo": {"pkl": args.pkl, "num_classes": args.num_classes, "epochs": args.epochs,
                        "batch_size": args.batch_size, "patience": args.patience,
                        "seeds": list(SEEDS), "n_train": len(train), "n_val": len(val),
                        "class_dist": dist},
        "runs": runs,
        "agg": {a: {"best_val_acc": agg(a, "best_val_acc"), "macro_F1": agg(a, "macro_F1")} for a in arms},
    }
    out = REPO / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    for a in arms:
        g = result["agg"][a]
        if g:
            print(f"[done] {a}: acc={g['best_val_acc']['mean']}±{g['best_val_acc']['std']} "
                  f"macroF1={g['macro_F1']['mean']}±{g['macro_F1']['std']}")


if __name__ == "__main__":
    main()
