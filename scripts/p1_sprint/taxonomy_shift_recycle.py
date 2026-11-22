# -*- coding: utf-8 -*-
"""taxonomy_shift_recycle — NTU120 60→49 merge episode 下的 anchor recycling。

目的：把标题里的 “evolving taxonomies” 变成本文自己的实验，而不是引用前作。
episode：60 类按 11 对相邻类合并为 49 类（与 TRANS-002 同规模），Y′ 标签固定。
四臂（同 seed、同 10% 锚点、同三臂尺）：
  remap-only   : 旧 60 类头在锚点上训练，按合并映射把概率求和成 49 类——零新标注；
  semantic-only: 冻结特征 + 10% Y′ 锚点跑 PSD 语义层；
  recycling    : 同一 Y′ 锚点先 LoRA 微调编码器，再跑 PSD 语义层；
  full-label   : Y′ 全标注线性参照（每个 seed 的 denominator）。

用法：
    .venv/Scripts/python.exe scripts/p1_sprint/taxonomy_shift_recycle.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/taxonomy_shift_recycle.py --seeds 10 --tag merge60to49
产出：
    runs/p1_sprint/adapt_t4_ntu120/summary_taxshift_<tag>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from adapt_common import REPO, adapt_out_dir, cosine_distill, inject_lora, load_frozen_model, lora_parameters
from labelfree_tier_replication import TIERS, dump_features, ScaledLR
from t4_budget_split_tier import build_anchor, run_semantic_arm

BUDGET_FRAC = 0.10
N_OLD, N_NEW = 60, 49


def merge_map() -> np.ndarray:
    """60→49：前 11 对相邻类各合并为一类，其余 38 类顺延。"""
    m = np.zeros(N_OLD, dtype=np.int64)
    for i in range(11):
        m[2 * i] = i
        m[2 * i + 1] = i
    for old in range(22, 60):
        m[old] = old - 11
    assert len(set(m.tolist())) == N_NEW
    return m


def remap(labels: np.ndarray, m: np.ndarray) -> np.ndarray:
    return m[np.asarray(labels, dtype=np.int64)]


def train_lora_on_labels(model, teacher, X_t, anchor_idx, y_anchor, n_cls, device,
                         epochs, batch, lr, aug_views, feeder_args):
    """小改自 t4.finetune_lora_ce：只保留 CE+蒸馏，标签已是 Y′。"""
    head = torch.nn.Linear(256, n_cls).to(device)
    torch.nn.init.xavier_uniform_(head.weight)
    params = lora_parameters(model.encoder_q) + list(head.parameters())
    for p in model.encoder_q.parameters():
        p.requires_grad = id(p) in {id(x) for x in lora_parameters(model.encoder_q)}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    model.eval()
    x_all = X_t[anchor_idx].float().to(device)
    y_all = torch.from_numpy(y_anchor.astype(np.int64)).to(device)
    n = len(x_all)
    eff = min(batch, n)
    rng = np.random.default_rng(42)
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
                from labelfree_tier_replication import extract_backbone
                f_s = extract_backbone(model.encoder_q, xs)
                with torch.no_grad():
                    f_t = extract_backbone(teacher, xs)
                loss = F.cross_entropy(head(f_s), ys) + cosine_distill(f_s, f_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_l.append(float(loss.item()))
        else:
            from labelfree_tier_replication import extract_backbone
            perm = rng.permutation(n)
            for i in range(0, n - n % eff, eff):
                idx = perm[i:i + eff]
                f_s = extract_backbone(model.encoder_q, x_all[idx])
                with torch.no_grad():
                    f_t = extract_backbone(teacher, x_all[idx])
                loss = F.cross_entropy(head(f_s), y_all[idx]) + cosine_distill(f_s, f_t)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_l.append(float(loss.item()))
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  [tax-lora] epoch {ep}/{epochs} loss={np.mean(ep_l) if ep_l else None:.4f}",
                  flush=True)
    return model


def remap_accuracy(model, tr_feat, tr_lab_old, va_feat, va_lab_new, anchor, m):
    """旧 60 类头在锚点上训练；按 m 聚合概率到 49 类。"""
    head = ScaledLR().fit(tr_feat[anchor], tr_lab_old[anchor])
    prob = head.clf.predict_proba(head.sc.transform(va_feat))  # (N, 60)
    agg = np.zeros((len(va_feat), N_NEW), dtype=np.float64)
    for old in range(N_OLD):
        agg[:, m[old]] += prob[:, old]
    return float(np.mean(agg.argmax(1) == va_lab_new))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--aug-views", type=int, default=8)
    ap.add_argument("--tag", default="merge60to49")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.tag = 1, 2, "smoke"

    tier = "ntu120"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(42, 42 + args.seeds))
    fd = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
    pkl = pickle.load(open(fd / "train_label.pkl", "rb"))
    tr_lab_old = np.asarray(pkl[1]).reshape(-1)
    X = np.array(np.load(fd / "train_position.npy", mmap_mode="r"))
    X_t = torch.from_numpy(X)
    m = merge_map()
    tr_lab_new = remap(tr_lab_old, m)

    # frozen features once
    model, ckpt = load_frozen_model(tier, device)
    store = dump_features(model, tier, device)
    tr_feat, va_feat = store["train_feat"], store["val_feat"]
    assert np.array_equal(np.asarray(store["train_label"]), tr_lab_old), \
        "feeder 标签序与 pkl 不一致——锚集构造会失真"
    va_lab_new = remap(store["val_label"], m)
    full_ref = float(np.mean(ScaledLR().fit(tr_feat, tr_lab_new).predict(va_feat) == va_lab_new))
    print(f"[taxshift] frozen full ref (49 cls) = {full_ref:.4f}", flush=True)

    rows = []
    t0 = time.time()
    for seed in seeds:
        print(f"[taxshift] seed {seed}", flush=True)
        anchor = build_anchor(np.array([str(x) for x in tr_lab_new]), seed, BUDGET_FRAC)
        acc_sem, pool_sem, stop_sem = run_semantic_arm(
            tr_feat, va_feat, tr_lab_new, va_lab_new, anchor, seed, BUDGET_FRAC)
        acc_remap = remap_accuracy(model, tr_feat, tr_lab_old, va_feat, va_lab_new, anchor, m)
        # recycling: fresh frozen model + LoRA on Y′ anchors
        m2, _ = load_frozen_model(tier, device)
        teacher = copy.deepcopy(m2.encoder_q)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        inject_lora(m2.encoder_q, r=8)
        m2 = m2.to(device)
        anchor_idx = np.where(anchor)[0]
        m2 = train_lora_on_labels(m2, teacher, X_t, anchor_idx, tr_lab_new[anchor], N_NEW,
                                  device, args.epochs, args.batch, args.lr, args.aug_views,
                                  {"data_path": str(fd / "train_position.npy"),
                                   "label_path": str(fd / "train_label.pkl")})
        m2.eval()
        for p in m2.parameters():
            p.requires_grad_(False)
        st2 = dump_features(m2, tier, device)
        tr2, va2 = st2["train_feat"], st2["val_feat"]
        va2_lab = remap(st2["val_label"], m)
        ref2 = float(np.mean(ScaledLR().fit(tr2, tr_lab_new).predict(va2) == va2_lab))
        acc_rec, pool_rec, stop_rec = run_semantic_arm(
            tr2, va2, tr_lab_new, va2_lab, anchor, seed, BUDGET_FRAC)
        rows.append({
            "seed": seed,
            "remap_only": round(acc_remap, 4),
            "semantic_only": round(acc_sem, 4),
            "recycling": round(acc_rec, 4),
            "full_ref_frozen": round(full_ref, 4),
            "full_ref_recycled": round(ref2, 4),
            "retention_semantic": round(acc_sem / full_ref, 4),
            "retention_recycling": round(acc_rec / ref2, 4) if ref2 > 0 else 0.0,
            "pool_semantic": pool_sem, "pool_recycling": pool_rec,
            "stop_semantic": stop_sem, "stop_recycling": stop_rec,
        })
        print(f"[taxshift] seed{seed}: remap={acc_remap:.4f} sem={acc_sem:.4f} "
              f"rec={acc_rec:.4f} ref={full_ref:.4f}/{ref2:.4f}", flush=True)
        out_path = adapt_out_dir("t4", tier) / f"summary_taxshift_{args.tag}.json"
        out_path.write_text(json.dumps({
            "experiment": "e13_taxonomy_shift", "tier": tier, "budget_frac": BUDGET_FRAC,
            "merge": "11 adjacent pairs 60->49", "per_seed": rows,
            "mean_remap": round(float(np.mean([r["remap_only"] for r in rows])), 4),
            "mean_semantic": round(float(np.mean([r["semantic_only"] for r in rows])), 4),
            "mean_recycling": round(float(np.mean([r["recycling"] for r in rows])), 4),
            "mean_retention_semantic": round(float(np.mean([r["retention_semantic"] for r in rows])), 4),
            "mean_retention_recycling": round(float(np.mean([r["retention_recycling"] for r in rows])), 4),
            "partial": len(rows) < len(seeds), "seeds": [r["seed"] for r in rows],
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "doneMarker": "P1_TAXSHIFT_NTU120_MERGE60TO49_DONE",
        "experiment": "e13_taxonomy_shift", "tier": tier, "budget_frac": BUDGET_FRAC,
        "merge": "11 adjacent pairs 60->49 (TRANS-002-scale merge episode)",
        "date": datetime.now().isoformat(timespec="seconds"),
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "aug_views": args.aug_views, "rank": 8, "distill_lambda": 1.0},
        "per_seed": rows,
        "means": {
            "remap_only": round(float(np.mean([r["remap_only"] for r in rows])), 4),
            "semantic_only": round(float(np.mean([r["semantic_only"] for r in rows])), 4),
            "recycling": round(float(np.mean([r["recycling"] for r in rows])), 4),
            "retention_semantic": round(float(np.mean([r["retention_semantic"] for r in rows])), 4),
            "retention_recycling": round(float(np.mean([r["retention_recycling"] for r in rows])), 4),
        },
        "seeds": [r["seed"] for r in rows],
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = adapt_out_dir("t4", tier) / f"summary_taxshift_{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result["means"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
