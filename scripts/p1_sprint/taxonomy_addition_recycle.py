# -*- coding: utf-8 -*-
"""taxonomy_addition_recycle — NTU120 60→120 class-addition episode 下的 anchor recycling。

目的：补全 “evolving taxonomies” 的另一方向——taxonomy 扩张（merge=收缩，addition=扩张）。
设定：120 类 balanced feeder（12,000/2,400，每类 100/20）为唯一数据源；
  旧类 = label 0–59，新类 = label 60–119；
  编码器 = 60 类 tier 的无标签 pretext（从未见过新类 clip，符合“新类后到”）。
阶段：
  phase1（初始 60 类部署）  : 仅旧类 10% 锚点（600）跑 PSD 语义层（60 类问题）；
  phase2（taxonomy 长到 120）: 旧锚点保留 + 新类 10% 锚点（600）→ 共 1,200；
  臂：semantic-only（冻结特征）vs recycling（同锚点先 LoRA-CE+蒸馏再语义层）。
指标：120 类总体 accuracy/retention（各自 full-label 线性参照为分母）、
  旧类 accuracy（phase1 vs phase2 = 遗忘）、新类 accuracy。

用法：
    .venv/Scripts/python.exe scripts/p1_sprint/taxonomy_addition_recycle.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/taxonomy_addition_recycle.py --seeds 10 --tag add60to120
产出：
    runs/p1_sprint/adapt_t4_ntu120/summary_taxadd_<tag>.json（逐种子断点续跑）
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

from adapt_common import REPO, adapt_out_dir, load_frozen_model
from labelfree_tier_replication import dump_features, ScaledLR
from t4_budget_split_tier import build_anchor, run_semantic_arm
from taxonomy_shift_recycle import train_lora_on_labels

BUDGET_FRAC = 0.10
N_OLD, N_NEW = 60, 120
FEEDER_TIER = "ntu120_full"          # 120 类 balanced feeder（唯一数据源）
DONE_MARKER = "P1_TAXADD_NTU120_ADD60TO120_DONE"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-2)
    ap.add_argument("--aug-views", type=int, default=8)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--tag", default="add60to120")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="显式种子列表（断点续跑单种子用）")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.epochs, args.tag = 1, 2, "smoke"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = (list(args.seed_list) if args.seed_list
             else list(range(42, 42 + args.seeds)))

    fd = REPO / "runs" / "p1_sprint" / f"labelfree_{FEEDER_TIER}" / "feeder"
    pkl = pickle.load(open(fd / "train_label.pkl", "rb"))
    tr_lab = np.asarray(pkl[1]).reshape(-1).astype(np.int64)
    va_pkl = pickle.load(open(fd / "val_label.pkl", "rb"))
    va_lab = np.asarray(va_pkl[1]).reshape(-1).astype(np.int64)
    assert len(set(tr_lab.tolist())) == N_NEW, "feeder 必须是 120 类"

    old_tr = tr_lab < N_OLD
    new_tr = ~old_tr
    old_va = va_lab < N_OLD
    new_va = ~old_va

    out_path = adapt_out_dir("t4", "ntu120") / f"summary_taxadd_{args.tag}.json"
    done_rows: dict = {}
    if out_path.exists() and not args.smoke:
        try:
            for r in json.loads(out_path.read_text(encoding="utf-8")).get("per_seed", []):
                if r.get("acc_all_recycling") is not None:
                    done_rows[int(r["seed"])] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            done_rows = {}
    seeds = [s for s in seeds if s not in done_rows]
    if done_rows:
        print(f"[resume] {len(done_rows)} seeds already done: {sorted(done_rows)}", flush=True)

    # 冻结特征一次（16ch headline 编码器，60 类 tier pretext）
    model, ckpt = load_frozen_model("ntu120", device)
    store = dump_features(model, FEEDER_TIER, device)
    tr_feat, va_feat = store["train_feat"], store["val_feat"]
    assert np.array_equal(np.asarray(store["train_label"]), tr_lab), "标签序失真"
    full_ref_frozen = float(np.mean(
        ScaledLR().fit(tr_feat, tr_lab).predict(va_feat) == va_lab))
    old_ref_frozen = float(np.mean(
        ScaledLR().fit(tr_feat[old_tr], tr_lab[old_tr]).predict(va_feat[old_va]) == va_lab[old_va]))
    print(f"[taxadd] frozen full ref (120 cls) = {full_ref_frozen:.4f}; "
          f"old-60 ref = {old_ref_frozen:.4f}", flush=True)

    rows = list(done_rows.values())
    t0 = time.time()
    for seed in seeds:
        print(f"[taxadd] seed {seed}", flush=True)
        # --- 锚点：phase1 旧类 10%（部署已花掉）；phase2 追加新类 10% ---
        lab_str_old = np.array([str(x) for x in tr_lab[old_tr]])
        a_old_sub = build_anchor(lab_str_old, seed, BUDGET_FRAC)
        lab_str_new = np.array([str(x) for x in tr_lab[new_tr]])
        a_new_sub = build_anchor(lab_str_new, seed, BUDGET_FRAC)
        old_idx_full = np.where(old_tr)[0]
        new_idx_full = np.where(new_tr)[0]
        anchor_full = np.zeros(len(tr_lab), dtype=bool)
        anchor_full[old_idx_full[a_old_sub]] = True
        anchor_full[new_idx_full[a_new_sub]] = True
        n_old_anchors = int(a_old_sub.sum())
        n_new_anchors = int(a_new_sub.sum())

        # --- phase1：初始 60 类部署（冻结特征、仅旧类） ---
        acc_old_p1, pool_p1, stop_p1, ok_p1 = run_semantic_arm(
            tr_feat[old_tr], va_feat[old_va], tr_lab[old_tr], va_lab[old_va],
            a_old_sub, seed, BUDGET_FRAC, return_pred=True)

        # --- phase2 semantic-only（冻结特征、120 类） ---
        acc_all_sem, pool_sem, stop_sem, ok_sem = run_semantic_arm(
            tr_feat, va_feat, tr_lab, va_lab, anchor_full, seed, BUDGET_FRAC,
            return_pred=True)

        # --- phase2 recycling（同 1,200 锚点先 LoRA 再语义层） ---
        m2, _ = load_frozen_model("ntu120", device)
        teacher = copy.deepcopy(m2.encoder_q)
        for p in teacher.parameters():
            p.requires_grad_(False)
        teacher.eval()
        from adapt_common import inject_lora
        inject_lora(m2.encoder_q, r=args.rank)
        m2 = m2.to(device)
        X = np.array(np.load(fd / "train_position.npy", mmap_mode="r"))
        X_t = torch.from_numpy(X)
        m2 = train_lora_on_labels(m2, teacher, X_t,
                                  np.where(anchor_full)[0], tr_lab[anchor_full],
                                  N_NEW, device, args.epochs, args.batch,
                                  args.lr, args.aug_views,
                                  {"data_path": str(fd / "train_position.npy"),
                                   "label_path": str(fd / "train_label.pkl")})
        m2.eval()
        for p in m2.parameters():
            p.requires_grad_(False)
        st2 = dump_features(m2, FEEDER_TIER, device)
        tr2, va2 = st2["train_feat"], st2["val_feat"]
        full_ref_rec = float(np.mean(
            ScaledLR().fit(tr2, tr_lab).predict(va2) == va_lab))
        acc_all_rec, pool_rec, stop_rec, ok_rec = run_semantic_arm(
            tr2, va2, tr_lab, va_lab, anchor_full, seed, BUDGET_FRAC,
            return_pred=True)

        # --- 旧类/新类分解（遗忘 = phase1 旧类 acc − phase2 旧类 acc） ---
        old_sem = float(np.mean(ok_sem[old_va])); new_sem = float(np.mean(ok_sem[new_va]))
        old_rec = float(np.mean(ok_rec[old_va])); new_rec = float(np.mean(ok_rec[new_va]))

        rows.append({
            "seed": seed,
            "acc_old_phase1": round(acc_old_p1, 4),
            "acc_old_semantic": round(old_sem, 4),
            "acc_new_semantic": round(new_sem, 4),
            "acc_all_semantic": round(acc_all_sem, 4),
            "acc_old_recycling": round(old_rec, 4),
            "acc_new_recycling": round(new_rec, 4),
            "acc_all_recycling": round(acc_all_rec, 4),
            "forgetting_semantic_pp": round((acc_old_p1 - old_sem) * 100, 2),
            "forgetting_recycling_pp": round((acc_old_p1 - old_rec) * 100, 2),
            "retention_semantic": round(acc_all_sem / full_ref_frozen, 4) if full_ref_frozen else 0.0,
            "retention_recycling": round(acc_all_rec / full_ref_rec, 4) if full_ref_rec else 0.0,
            "full_ref_frozen": round(full_ref_frozen, 4),
            "full_ref_recycled": round(full_ref_rec, 4),
            "old_ref_frozen": round(old_ref_frozen, 4),
            "n_old_anchors": n_old_anchors, "n_new_anchors": n_new_anchors,
            "pool_phase1": pool_p1, "pool_semantic": pool_sem, "pool_recycling": pool_rec,
            "stop_phase1": stop_p1, "stop_semantic": stop_sem, "stop_recycling": stop_rec,
        })
        print(f"[taxadd] seed{seed}: p1_old={acc_old_p1:.4f} "
              f"sem(all/old/new)={acc_all_sem:.4f}/{old_sem:.4f}/{new_sem:.4f} "
              f"rec(all/old/new)={acc_all_rec:.4f}/{old_rec:.4f}/{new_rec:.4f} "
              f"ref={full_ref_frozen:.4f}/{full_ref_rec:.4f}", flush=True)
        all_rows = sorted(rows, key=lambda r: r["seed"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "experiment": "e14_taxonomy_addition", "tier": FEEDER_TIER,
            "budget_frac": BUDGET_FRAC, "addition": "old 60 -> +60 new (120 total)",
            "per_seed": all_rows,
            "mean_phase1_old": round(float(np.mean([r["acc_old_phase1"] for r in all_rows])), 4),
            "mean_semantic": round(float(np.mean([r["acc_all_semantic"] for r in all_rows])), 4),
            "mean_recycling": round(float(np.mean([r["acc_all_recycling"] for r in all_rows])), 4),
            "mean_retention_semantic": round(float(np.mean([r["retention_semantic"] for r in all_rows])), 4),
            "mean_retention_recycling": round(float(np.mean([r["retention_recycling"] for r in all_rows])), 4),
            "partial": len(all_rows) < 10, "seeds": [r["seed"] for r in all_rows],
            "wall_clock_sec": round(time.time() - t0, 1),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    result = {
        "doneMarker": DONE_MARKER,
        "experiment": "e14_taxonomy_addition", "tier": FEEDER_TIER,
        "budget_frac": BUDGET_FRAC, "addition": "old 60 -> +60 new (120 total)",
        "protocol": "phase1 = initial 60-class deployment (old-class 10% anchors only); "
                    "phase2 = taxonomy grows to 120 (+10% anchors of the 60 new classes); "
                    "arms: semantic-only (frozen) vs recycling (LoRA-CE+distill on the same 1,200 anchors)",
        "date": datetime.now().isoformat(timespec="seconds"),
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "rank": args.rank, "aug_views": args.aug_views},
        "per_seed": sorted(rows, key=lambda r: r["seed"]),
        "mean_phase1_old": round(float(np.mean([r["acc_old_phase1"] for r in rows])), 4),
        "mean_semantic": round(float(np.mean([r["acc_all_semantic"] for r in rows])), 4),
        "mean_recycling": round(float(np.mean([r["acc_all_recycling"] for r in rows])), 4),
        "mean_retention_semantic": round(float(np.mean([r["retention_semantic"] for r in rows])), 4),
        "mean_retention_recycling": round(float(np.mean([r["retention_recycling"] for r in rows])), 4),
        "mean_forgetting_semantic_pp": round(float(np.mean([r["forgetting_semantic_pp"] for r in rows])), 2),
        "mean_forgetting_recycling_pp": round(float(np.mean([r["forgetting_recycling_pp"] for r in rows])), 2),
        "mean_acc_new_semantic": round(float(np.mean([r["acc_new_semantic"] for r in rows])), 4),
        "mean_acc_new_recycling": round(float(np.mean([r["acc_new_recycling"] for r in rows])), 4),
        "seeds": sorted(r["seed"] for r in rows),
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out_path}")
    print(json.dumps({"mean_semantic": result["mean_semantic"],
                      "mean_recycling": result["mean_recycling"],
                      "mean_retention_semantic": result["mean_retention_semantic"],
                      "mean_retention_recycling": result["mean_retention_recycling"]},
                     indent=1))


if __name__ == "__main__":
    main()
