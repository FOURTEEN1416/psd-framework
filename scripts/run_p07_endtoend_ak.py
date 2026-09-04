# -*- coding: utf-8 -*-
"""E7/E8 — 完整 PSD 管线端到端评估（AK full12 公开真实层，冲击 PR 预注册实验）.

预注册（BOARD 09-04 fb1a060）:
  E7: 端到端管线 = Y/scratch backbone 特征 → 种子(spc2≈13%) → run_selftrain(锚点+聚类+伪标签迭代)
      → seeds∪pool 训练线性头 → val top-1 + macro-F1 + per-class。臂 {warm,scratch} × seeds{42,43,44}。
  E8: 标注效率曲线 = budget {spc2, spc4, full} × warm × seeds{42,43,44}（全监督对照）。
防泄漏铁律: anchor_mask 与 pool_universe_mask 仅 train split；val(56) 全程隔离，truth 仅评估。
口径: 公开真实层（AK 自提取 full12，9 类有样本）；线性头为 Ω 最小可行实现（与 method §3.3.3 一致）。
用法: python scripts/run_p07_endtoend_ak.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from psd.training.tcl_selftrain import run_selftrain  # noqa: E402
from psd.training.stgcnbc_feature_extractor import STGCNBCFeatureExtractor  # noqa: E402

SEEDS = (42, 43, 44)
PKL = REPO / "runs/public_real_dataset/full12_T30.pkl"
Y_CKPT = REPO / "runs/p05_stgcn_bc_full/best.pt"
OUT = REPO / "reports/p07-endtoend-ak-full12-2026-09-04.json"

# P0.4 主配置参数（复用，防漂移）
HEAD_CFG = {"hidden_dim": 64, "epochs": 150, "lr": 0.001, "weight_decay": 0.0001,
            "batch_size": 128, "device": "cpu"}
KW = dict(calib_method="softmax_temperature", calib_target=0.10,
          tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
          tau_select={"rule": "quantile", "target_coverage": 0.35},
          alpha=1.0, standing_mode="consensus", subcluster_k=2, subcluster_min_share=0.05,
          max_iters=6, converge_change_rate=0.01, precision_drop_patience=2)


def load_dataset():
    with open(PKL, "rb") as f:
        data = pickle.load(f)
    kp = np.stack([np.asarray(d["keypoints"], dtype=np.float32) for d in data])
    labels = np.array([int(d["label"]) for d in data])
    splits = np.array([d["split"] for d in data])
    return data, kp, labels, splits


def extract_features(kp, arm: str):
    """warm=Y backbone / scratch=随机初始化 → (N,D) penultimate 特征。"""
    if arm == "warm":
        ex = STGCNBCFeatureExtractor.from_checkpoint(Y_CKPT, device="cpu", num_classes=22)
    else:
        from psd.models.stgcn_bc import build_stgcn_bc
        m = build_stgcn_bc(in_channels=3, num_classes=22)  # 随机初始化
        ex = STGCNBCFeatureExtractor(m, device="cpu")
    feats = []
    B = 32
    for i in range(0, len(kp), B):
        feats.append(ex.extract(kp[i:i + B]))
    return np.vstack(feats)


def pick_seeds(labels, splits, spc: int, rng: np.random.Generator):
    """train split 内每类 spc 个 clip 作种子（full=全部 train）。返回 anchor_mask。"""
    n = len(labels)
    mask = np.zeros(n, dtype=bool)
    train_idx = np.where(splits == "train")[0]
    by_cls = defaultdict(list)
    for i in train_idx:
        by_cls[int(labels[i])].append(i)
    for c, idxs in by_cls.items():
        if spc >= 0:
            chosen = rng.choice(idxs, size=min(spc, len(idxs)), replace=False)
        else:
            chosen = idxs
        mask[chosen] = True
    return mask


def train_linear_head(emb, labels, train_mask, n_cls: int):
    """seeds∪pool 训练线性头（Ω 最小可行实现）。"""
    from sklearn.linear_model import LogisticRegression
    X = emb[train_mask]
    y = labels[train_mask]
    keep = np.unique(y)
    if len(keep) < 2:
        return None
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(X, y)
    return clf


def evaluate(clf, emb, labels, val_mask, class_names):
    X = emb[val_mask]
    y = labels[val_mask]
    pred = clf.predict(X)
    top1 = float(np.mean(pred == y))
    # macro-F1 + per-class
    per_cls = {}
    f1s = []
    for c in range(len(class_names)):
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1)
        if np.sum(y == c) > 0:
            per_cls[class_names[c]] = {"support": int(np.sum(y == c)), "acc": round(float(np.mean(pred[y == c] == c)), 4)}
    return {"top1": round(top1, 4), "macro_f1": round(float(np.mean(f1s)), 4), "per_class": per_cls}


def run_one(emb, labels, labels_str, splits, class_names, arm, spc, seed, smoke=False):
    rng = np.random.default_rng(seed)
    # full 监督臂: 全部 train 作种子, 无伪标签迭代(池宇宙为空)
    if spc < 0:
        train_mask = (splits == "train")
        clf = train_linear_head(emb, labels, train_mask, len(class_names))
        ev = evaluate(clf, emb, labels, splits == "val", class_names)
        return {"arm": arm, "spc": -1, "seed": seed, "budget": "full",
                "n_seeds": int(train_mask.sum()), "n_pool": 0,
                "tau_star": None, "rounds": 0, "stop": "full-supervision(no pseudo-label)", **ev}
    anchor_mask = pick_seeds(labels, splits, spc, rng)
    # 防泄漏: 池宇宙 = train 非种子（val 永不入池）
    universe = (splits == "train") & ~anchor_mask
    r = run_selftrain(emb, labels_str, anchor_mask, run_seed=seed, class_names=class_names,
                      head_cfg=HEAD_CFG, pool_universe_mask=universe, **KW)
    # 端到端头: seeds ∪ 最终池
    train_mask = anchor_mask.copy()
    train_mask[r["final_pool_idx"]] = True
    clf = train_linear_head(emb, labels, train_mask, len(class_names))
    if clf is None:
        return {"arm": arm, "spc": spc, "seed": seed, "error": "种子类别不足"}
    val_mask = splits == "val"
    ev = evaluate(clf, emb, labels, val_mask, class_names)
    return {"arm": arm, "spc": spc, "seed": seed,
            "n_seeds": int(anchor_mask.sum()), "n_pool": int(len(r["final_pool_idx"])),
            "tau_star": round(float(r["tau_operating"]), 4),
            "rounds": len(r["rounds"]), "stop": r["stop_reason"], **ev}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data, kp, labels, splits = load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    print(f"[data] {len(data)} clips | train={int((splits=='train').sum())} val={int((splits=='val').sum())} | classes={class_names}")

    t0 = time.time()
    # 特征提取（两臂各一次，跨 seed/budget 复用）
    feats = {}
    for arm in ("warm", "scratch"):
        print(f"[feat] extracting {arm} backbone features...")
        feats[arm] = extract_features(kp, arm)
        print(f"  {arm}: {feats[arm].shape}")

    runs = []
    # E7: {warm,scratch} × spc2 × 3seeds
    for arm in ("warm", "scratch"):
        for seed in SEEDS:
            r = run_one(feats[arm], labels, labels_str, splits, class_names, arm, 2, seed, args.smoke)
            runs.append(r)
            print(f"[E7] {arm} seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')} (seeds={r.get('n_seeds')} pool={r.get('n_pool')})")
    # E8: warm × {spc2,spc4,full} × 3seeds
    for spc in (2, 4, -1):
        for seed in SEEDS:
            r = run_one(feats["warm"], labels, labels_str, splits, class_names, "warm", spc, seed, args.smoke)
            r["budget"] = "full" if spc < 0 else f"spc{spc}"
            runs.append(r)
            print(f"[E8] warm {r['budget']} seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')}")

    def agg(arm, budget):
        sel = [r for r in runs if r["arm"] == arm and r.get("spc") == budget and "top1" in r]
        if not sel:
            return None
        t = [r["top1"] for r in sel]; m = [r["macro_f1"] for r in sel]
        return {"n": len(sel), "top1_mean": round(float(np.mean(t)), 4), "top1_std": round(float(np.std(t)), 4),
                "macro_f1_mean": round(float(np.mean(m)), 4), "macro_f1_std": round(float(np.std(m)), 4)}

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real",
        "protocol": "E7 end-to-end PSD pipeline (Y/scratch backbone -> seeds spc2 -> run_selftrain anchor-cluster-pseudo-label -> seeds+pool linear head -> val); E8 budget curve spc2/spc4/full",
        "leakage_guard": "anchor_mask & pool_universe restricted to train split; val(56) isolated; truth eval-only",
        "config_echo": {"pkl": str(PKL), "y_ckpt": str(Y_CKPT), "seeds": list(SEEDS),
                        "n_clips": len(data), "classes": class_names,
                        "n_train": int((splits == "train").sum()), "n_val": int((splits == "val").sum())},
        "runs": runs,
        "agg": {f"{a}_spc{b}": agg(a, b) for a in ("warm", "scratch") for b in (2, 4, -1)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {OUT}  ({result['wall_clock_sec']}s)")
    for k, v in result["agg"].items():
        if v:
            print(f"  {k}: top1={v['top1_mean']}±{v['top1_std']} macroF1={v['macro_f1_mean']}±{v['macro_f1_std']}")


if __name__ == "__main__":
    main()
