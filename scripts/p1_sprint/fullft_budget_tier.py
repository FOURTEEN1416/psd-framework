# -*- coding: utf-8 -*-
"""fullft_budget_tier — matched-budget 全量微调基线（NTU120 10% 锚点）。

目的：回答审稿人最可能提出的问题——“anchor recycling 的增益是不是 just fine-tuning？”
协议与 t4_budget_split_tier.py 的 same10 臂逐项对齐：
  · 每个 seed 用同一分层 10% 锚点（build_anchor 与 run_arms 同 RNG 同序）；
  · 从同一冻结 pretext ckpt 初始化，但**全参数解冻**（不是 LoRA）；
  · 同一增广族（Feeder_single shear 0.5 / temporal pad 6，可 V 视图）；
  · 训练后重导特征，算该 seed 的 full-label ref，并用同一锚点跑 (b) 语义臂；
  · retention_seed = b_arm / full_ref，汇总 retention = mean_b / mean_full_ref。

与 recycling 的差异只在“可训练参数集合”：full FT = encoder 全部参数 + 新头；
recycling = LoRA 分支 + 新头（BN running stats 冻结）。BN 口径在本脚本中同样冻结，
以保持特征导出口径一致（与 t4 相同）。

用法：
    .venv/Scripts/python.exe scripts/p1_sprint/fullft_budget_tier.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/fullft_budget_tier.py \
        --seeds 10 --epochs 50 --lr 1e-3 --tag full10
产出：
    runs/p1_sprint/adapt_t4_ntu120/summary_fullft_<tag>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from adapt_common import REPO, adapt_out_dir, load_frozen_model
from labelfree_tier_replication import TIERS, dump_features, ScaledLR, extract_backbone

BUDGET_FRAC = 0.10


def build_anchor(labels_str: np.ndarray, seed: int, budget_frac: float) -> np.ndarray:
    """与 run_arms / t4 逐式一致的分层锚点构造。"""
    anchor = np.zeros(len(labels_str), dtype=bool)
    rng = np.random.default_rng(seed)
    for c in sorted(set(labels_str.tolist())):
        ci = np.where(labels_str == c)[0]
        k = max(1, int(round(len(ci) * budget_frac)))
        anchor[rng.choice(ci, size=k, replace=False)] = True
    return anchor


def run_semantic_arm(tr_feat, va_feat, tr_lab, va_lab, anchor, seed, budget_frac):
    """与 run_arms 的 (b) 臂逐式一致的单种子语义评测（t4 同款）。"""
    from run_p5b_generic_retention import KW
    from psd.training.tcl_selftrain import run_selftrain

    labels_str = np.array([str(l) for l in tr_lab])
    full_labels_str = np.array([str(l) for l in np.concatenate([tr_lab, va_lab])])
    class_names = [str(c) for c in sorted(set(tr_lab.tolist()))]
    universe = np.ones(len(tr_lab), dtype=bool)
    universe[anchor] = False
    r = run_selftrain(tr_feat, labels_str, anchor, run_seed=seed,
                      class_names=class_names,
                      head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                "weight_decay": 1e-4, "batch_size": 128, "device": "cpu"},
                      pool_universe_mask=universe, **KW)
    tm2 = anchor.copy()
    tm2[r["final_pool_idx"]] = True
    y2 = np.array([None] * len(labels_str), dtype=object)
    y2[anchor] = labels_str[anchor]
    y2[r["final_pool_idx"]] = r["final_pred_full"][r["final_pool_idx"]]
    va_lab_str = full_labels_str[len(tr_lab):]
    acc_b = float(np.mean(ScaledLR().fit(tr_feat[tm2], y2[tm2]).predict(va_feat) == va_lab_str))
    return acc_b, len(r["final_pool_idx"]), r["stop_reason"]


def finetune_full(model, X_t: torch.Tensor, anchor_idx: np.ndarray, y_anchor: np.ndarray,
                  n_cls: int, device: str, epochs: int, batch: int, lr: float,
                  aug_views: int, feeder_args: dict | None):
    """全参数微调 encoder_q + 新线性头；BN running stats 冻结。"""
    head = torch.nn.Linear(256, n_cls).to(device)
    torch.nn.init.xavier_uniform_(head.weight)
    params = list(model.encoder_q.parameters()) + list(head.parameters())
    for p in model.encoder_q.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    model.eval()  # BN running stats 冻结（eval 口径）
    x_all = X_t[anchor_idx].float().to(device)
    y_all = torch.from_numpy(y_anchor.astype(np.int64)).to(device)
    n = len(x_all)
    eff_batch = min(batch, n)
    rng = np.random.default_rng(42)
    losses, accs = [], []
    for ep in range(1, epochs + 1):
        ep_l = []
        if aug_views > 1:
            from feeder.ntu_feeder import Feeder_single
            feeder = Feeder_single(feeder_args["data_path"], feeder_args["label_path"],
                                   shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True)
            steps = max(1, (n * aug_views) // batch)
            for _ in range(steps):
                pick = rng.choice(n, size=batch, replace=True)
                xs = torch.stack([
                    torch.from_numpy(np.asarray(feeder[int(anchor_idx[p])][0])).float()
                    for p in pick]).to(device)
                ys = torch.from_numpy(y_anchor[pick].astype(np.int64)).to(device)
                logits = head(extract_backbone(model.encoder_q, xs))
                loss = F.cross_entropy(logits, ys)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_l.append(float(loss.item()))
        else:
            perm = rng.permutation(n)
            for i in range(0, n - n % eff_batch, eff_batch):
                idx = perm[i:i + eff_batch]
                x, y = x_all[idx], y_all[idx]
                logits = head(extract_backbone(model.encoder_q, x))
                loss = F.cross_entropy(logits, y)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_l.append(float(loss.item()))
        sched.step()
        with torch.no_grad():
            acc = float((head(extract_backbone(model.encoder_q, x_all)).argmax(1)
                         == y_all).float().mean())
        losses.append(round(float(np.mean(ep_l)), 4) if ep_l else None)
        accs.append(round(acc, 4))
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  [fullft] epoch {ep}/{epochs} loss={losses[-1]} anchor_acc={acc:.3f}",
                  flush=True)
    return model, head, losses, accs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="ntu120")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug-views", type=int, default=1)
    ap.add_argument("--tag", default="fullft")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed-list", type=int, nargs="+", default=None)
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.tag = 1, 2, "smoke"

    tier = args.tier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(args.seed_list) if args.seed_list else list(range(42, 42 + args.seeds))
    n_cls = TIERS[tier]
    fd = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
    import pickle
    pkl = pickle.load(open(fd / "train_label.pkl", "rb"))
    pkl_labels = np.asarray(pkl[1]).reshape(-1)
    X = np.array(np.load(fd / "train_position.npy", mmap_mode="r"))
    X_t = torch.from_numpy(X)
    out_path = adapt_out_dir("t4", tier) / f"summary_fullft_{args.tag}.json"
    done_rows = []
    if out_path.exists():
        try:
            done_rows = json.loads(out_path.read_text(encoding="utf-8")).get("per_seed", [])
        except Exception:
            done_rows = []
    done_seeds = {int(r["seed"]) for r in done_rows}
    seeds = [s for s in seeds if s not in done_seeds]
    rows = list(done_rows)
    t0 = time.time()
    labels_checked = False
    for seed in seeds:
        print(f"[fullft] seed {seed}", flush=True)
        model, ckpt = load_frozen_model(tier, device)
        if not labels_checked:
            store0 = dump_features(model, tier, device)
            assert np.array_equal(np.asarray(store0["train_label"]), pkl_labels), \
                "pkl 标签序与 dump_features 不一致——锚集构造将失真"
            labels_checked = True
        labels_str0 = np.array([str(l) for l in pkl_labels])
        anchor = build_anchor(labels_str0, seed, BUDGET_FRAC)
        anchor_idx = np.where(anchor)[0]
        y_anchor = pkl_labels[anchor]
        model, head, losses, accs = finetune_full(
            model, X_t, anchor_idx, y_anchor, n_cls, device,
            args.epochs, args.batch, args.lr, args.aug_views,
            {"data_path": str(fd / "train_position.npy"),
             "label_path": str(fd / "train_label.pkl")})
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        store = dump_features(model, tier, device)
        tr_feat, tr_lab = store["train_feat"], store["train_label"]
        va_feat, va_lab = store["val_feat"], store["val_label"]
        ref_j = float(np.mean(ScaledLR().fit(tr_feat, tr_lab).predict(va_feat) == va_lab))
        labels_str = np.array([str(l) for l in tr_lab])
        anchor2 = build_anchor(labels_str, seed, BUDGET_FRAC)
        acc_b, pool, stop = run_semantic_arm(tr_feat, va_feat, tr_lab, va_lab,
                                             anchor2, seed, BUDGET_FRAC)
        rows.append({"seed": seed, "full_ref": round(ref_j, 4), "b_arm": round(acc_b, 4),
                     "retention_seed": round(acc_b / ref_j, 4) if ref_j > 0 else 0.0,
                     "pool": pool, "stop": stop,
                     "ft_anchor_acc": accs[-1], "ft_loss_tail": losses[-3:]})
        print(f"[fullft] seed{seed}: ref={ref_j:.4f} b={acc_b:.4f} "
              f"ret={rows[-1]['retention_seed']:.4f} pool={pool}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "experiment": "e10_t4_budget_split", "arm": "fullft", "tier": tier,
            "budget_frac": BUDGET_FRAC, "per_seed": rows,
            "mean_b": round(float(np.mean([r["b_arm"] for r in rows])), 4),
            "mean_full_ref": round(float(np.mean([r["full_ref"] for r in rows])), 4),
            "retention": round(float(np.mean([r["b_arm"] for r in rows]))
                               / float(np.mean([r["full_ref"] for r in rows])), 4),
            "partial": len(rows) < len(seeds), "seeds": [r["seed"] for r in rows],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = sorted(rows, key=lambda r: r["seed"])
    mean_b = round(float(np.mean([r["b_arm"] for r in rows])), 4)
    mean_ref = round(float(np.mean([r["full_ref"] for r in rows])), 4)
    ret = round(mean_b / mean_ref, 4) if mean_ref > 0 else 0.0
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "doneMarker": f"P1_ADAPT_T4_FULLFT_{tier.upper()}_DONE",
        "experiment": "e10_t4_budget_split", "arm": "fullft", "tier": tier,
        "mode": "smoke" if args.smoke else "full",
        "budget_frac": BUDGET_FRAC,
        "protocol": "per-seed full fine-tune of encoder_q on THIS seed's anchors "
                    "(all params trainable, BN running stats frozen), then re-dump "
                    "features, full-label linear ref, and the SAME-anchor (b) semantic arm",
        "date": datetime.now().isoformat(timespec="seconds"),
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "aug_views": args.aug_views, "rank": None, "distill_lambda": 0.0},
        "per_seed": rows, "mean_b": mean_b, "mean_full_ref": mean_ref,
        "retention": ret, "verdict": verdict, "seeds": [r["seed"] for r in rows],
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = adapt_out_dir("t4", tier) / f"summary_fullft_{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps({"mean_b": mean_b, "mean_full_ref": mean_ref,
                      "retention": ret, "verdict": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
