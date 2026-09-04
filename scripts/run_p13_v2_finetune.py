"""P1.3 v2 端到端微调臂 — 直接检验"绝对精度天花板"是否训练深度受限。

背景: p12 已证 v2 冻结骨干+线性头全监督 37.50%（数据瓶颈成立）。本臂回答审稿人
下一个自然问题: "如果端到端微调骨干，天花板还能多高？"——若显著抬升，则 ~34-38%
既非方法缺陷也非仅数据量，而是探针协议下界；论文获得诚实的"可达上界"数字。
低预算对照 finetune_spc2（16 片段端到端）预期过拟合坍缩——反衬 warm-start 设计价值。

臂: finetune_full（256 train 全量端到端 50ep）/ finetune_spc2（每类2段端到端）
    × 3 seeds；评估 val 96 clips top-1 + macro-F1（8 类有样本口径）。
参照: 冻结头 full 37.50% / warm spc2 33.23%（p12）。

用法:
    .venv/Scripts/python.exe scripts/run_p13_v2_finetune.py
产出:
    reports/p13-v2-finetune-<date>.json / .md
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
import torch.nn as nn

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from psd.models.stgcn_bc import build_stgcn_bc  # noqa: E402

PKL_V2 = REPO / "runs" / "public_real_dataset" / "full12v2_T30.pkl"
SEEDS = (42, 43, 44)
EPOCHS = 50
LR = 1e-3
WD = 1e-4
BATCH = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = REPO / "reports"


def load_v2():
    with open(PKL_V2, "rb") as f:
        data = pickle.load(f)
    kp = np.stack([np.asarray(d["keypoints"], dtype=np.float32) for d in data])
    labels = np.array([int(d["label"]) for d in data])
    splits = np.array([d["split"] for d in data])
    classes = sorted({str(d["psd_class"]) for d in data})
    return kp, labels, splits, classes


def evaluate(model, kp, labels, mask, n_cls=12):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(kp[mask]), 64):
            x = torch.from_numpy(kp[mask][i:i + 64]).to(DEVICE)
            logits, _ = model(x)
            preds.append(logits.argmax(1).cpu().numpy())
    pred = np.concatenate(preds)
    y = labels[mask]
    top1 = float((pred == y).mean())
    present = sorted(set(y.tolist()))
    f1s = []
    for c in present:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"top1": round(top1, 4), "macro_f1": round(float(np.mean(f1s)), 4)}


def finetune(kp, labels, splits, train_mask, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_stgcn_bc(in_channels=3, num_classes=12).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()
    idx = np.where(train_mask)[0]
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(idx)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            x = torch.from_numpy(kp[b]).to(DEVICE)
            y = torch.from_numpy(labels[b]).to(DEVICE)
            logits, _ = model(x)
            loss = ce(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    val_mask = splits == "val"
    return evaluate(model, kp, labels, val_mask)


def main():
    t0 = time.time()
    kp, labels, splits, classes = load_v2()
    train_mask = splits == "train"
    print(f"[v2] {len(kp)} clips train={train_mask.sum()} val={(~train_mask & (splits=='val')).sum()} device={DEVICE}")

    runs = []
    for seed in SEEDS:
        r = finetune(kp, labels, splits, train_mask, seed)
        runs.append({"arm": "finetune_full", "seed": seed, **r})
        print(f"[full] seed{seed}: top1={r['top1']} macroF1={r['macro_f1']}")
    # spc2: 每类 2 段（train 内），不足取全部
    rng = np.random.default_rng(42)
    anchor = np.zeros(len(kp), dtype=bool)
    for c in sorted(set(labels[train_mask].tolist())):
        ci = np.where(train_mask & (labels == c))[0]
        anchor[rng.choice(ci, size=min(2, len(ci)), replace=False)] = True
    for seed in SEEDS:
        r = finetune(kp, labels, splits, anchor, seed)
        runs.append({"arm": "finetune_spc2", "seed": seed, **r})
        print(f"[spc2] seed{seed}: top1={r['top1']} macroF1={r['macro_f1']}")

    agg = {}
    for arm in ("finetune_full", "finetune_spc2"):
        sel = [r for r in runs if r["arm"] == arm]
        t = [r["top1"] for r in sel]; m = [r["macro_f1"] for r in sel]
        agg[arm] = {"top1_mean": round(float(np.mean(t)), 4), "top1_std": round(float(np.std(t, ddof=1)), 4),
                    "macro_f1_mean": round(float(np.mean(m)), 4), "macro_f1_std": round(float(np.std(m, ddof=1)), 4)}

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real_v2",
        "protocol": "end-to-end fine-tune ST-GCN+BC (12-slot head, 8 classes present) on v2; 50ep lr1e-3 wd1e-4 batch32; eval val96",
        "reference_frozen_head": {"full_top1": 0.3750, "warm_spc2_top1": 0.3323},
        "classes": classes, "config_echo": {"pkl": str(PKL_V2), "seeds": list(SEEDS), "epochs": EPOCHS},
        "runs": runs, "agg": agg, "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p13-v2-finetune-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps(agg, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
