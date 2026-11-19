# -*- coding: utf-8 -*-
"""E10-T3 — 伪标签回灌闭环: PSD 语义层的伪标签反过来继续训练物理层。

动机: 现有 PSD 是单向管线（物理层→语义层，自训练只更新分类头）。本线把方向
反过来: 语义层产出的**置信度过滤伪标签**（锚点真标签 + 池内伪标签）作为
免费监督，以监督对比损失（SupCon）回灌适配物理层（LoRA 低秩 + 蒸馏锚防漂移），
然后回到统一尺重测 E9 三臂。这把 PSD 从"解耦管线"升级为"闭环系统"——
直接回答"既然跨域失败，框架如何自我修复"。

协议:
  1) 冻结模型 → run_arms(seed=42) 取 (b) 臂伪标签工件（anchor 真标签 +
     final_pool_idx 的伪标签；run_selftrain 内部已做置信度过滤/门控）；
  2) encoder_q/k 注入 LoRA（B=0），teacher=原 encoder_q；
  3) 以伪标签为监督做 SupCon 适配（batch 256, 单视图无增广）+ λ_d 蒸馏锚；
  4) 适配后统一尺 run_arms ×seeds → 新保留率/判据；
  5) --rounds 2: 用新特征重跑 (b) 臂取新伪标签再适配一轮（默认 1 轮）。

用法:
    .venv/Scripts/python.exe -u scripts/p1_sprint/supcon_feedback_tier.py --tier ntu120
产出:
    runs/p1_sprint/adapt_supcon_<tier>/summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime

import numpy as np
import torch

from adapt_common import (REPO, adapt_out_dir, copy_params_q_to_k, cosine_distill,
                          count_params, extract_backbone, inject_lora,
                          lora_parameters, load_frozen_model, supcon_loss)


def pseudo_labels_from_arms(model, tier: str, device: str, psd_seed: int) -> dict:
    """跑一次 (b) 臂（指定种子）拿伪标签工件 + 基线读数（统一尺内取件）。"""
    from labelfree_tier_replication import run_arms
    arms = run_arms(model, tier, (psd_seed,), device)
    lb = arms["last_b"]
    return {"full_ref": arms["full_ref"], "linear_10pct": arms["linear_10pct"],
            "b_arm_seed_acc": arms["b_arms"][0], "pseudo": lb}


def supcon_adapt(model, teacher, X: np.ndarray, y_sem: np.ndarray, args, device: str,
                 out_dir, loss_hist: list) -> None:
    """LoRA 上的 SupCon 适配。y_sem: object 数组，None=不参与（未入选池样本）。"""
    from labelfree_tier_replication import TIERS
    n_cls = len([c for c in sorted(set(str(v) for v in y_sem if v is not None))])
    class_to_idx = {c: i for i, c in enumerate(sorted(set(str(v) for v in y_sem if v is not None)))}
    lab = np.array([-1 if v is None else class_to_idx[str(v)] for v in y_sem], dtype=np.int64)
    keep = np.where(lab >= 0)[0]
    print(f"[supcon] supervised samples: {len(keep)}/{len(lab)} classes={n_cls}", flush=True)

    params = lora_parameters(model.encoder_q)
    trainable_ids = {id(p) for p in params}
    for p in model.encoder_q.parameters():
        p.requires_grad = id(p) in trainable_ids
    tr, tt = count_params(model)
    print(f"[supcon] trainable={tr:,}/{tt:,} ({100*tr/tt:.2f}%)", flush=True)

    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    X_t = torch.from_numpy(np.array(X))  # (N,3,30,V,1)；np.array 强制可写副本（memmap 只读）
    n = len(keep)
    model.train()
    rng = np.random.default_rng(args.seed)
    nan_batches = 0
    for epoch in range(1, args.epochs + 1):
        perm = rng.permutation(keep)
        ep_losses = []
        for i in range(0, n - n % args.batch, args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 4:
                continue
            x = X_t[idx].float().to(device)
            y = torch.from_numpy(lab[idx]).to(device)
            f_s = extract_backbone(model.encoder_q, x)
            with torch.no_grad():
                f_t = extract_backbone(teacher, x)
            loss = supcon_loss(f_s, y, temperature=args.temperature)
            if args.distill > 0:
                loss = loss + args.distill * cosine_distill(f_s, f_t)
            if not torch.isfinite(loss):
                nan_batches += 1
                continue
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            opt.step()
            ep_losses.append(float(loss.item()))
        loss_hist.append(round(float(np.mean(ep_losses)), 4) if ep_losses else None)
        print(f"[supcon] epoch {epoch}/{args.epochs} loss={loss_hist[-1]} "
              f"nan_skipped={nan_batches}", flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), out_dir / f"supcon_r{args.round}_epoch{epoch}_model.pt")
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=["ntu120", "ucf101", "panaf500"])
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--psd-seed", type=int, default=42, help="取伪标签所用的 (b) 臂种子")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42, help="适配训练的采样种子")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tier = args.tier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.rounds, args.epochs, args.save_every = 1, 1, 1
    seeds = tuple(range(42, 42 + (1 if args.smoke else args.seeds)))
    t0 = time.time()

    model, ckpt = load_frozen_model(tier, device)
    out_dir = adapt_out_dir("supcon", tier)
    loss_hist = []
    rounds_meta = []

    for rd in range(1, args.rounds + 1):
        args.round = rd
        # 1) 当前模型的 (b) 臂伪标签
        pl = pseudo_labels_from_arms(model, tier, device, args.psd_seed)
        y_sem = pl["pseudo"]["y2"]
        print(f"[round {rd}] pseudo labeled={int(np.sum([v is not None for v in y_sem]))} "
              f"b_acc={pl['b_arm_seed_acc']:.4f}", flush=True)

        # 2) teacher 只在第一轮取（原冻结编码器）；后续轮继续用同一锚
        if rd == 1:
            teacher = copy.deepcopy(model.encoder_q)
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
            inject_lora(model.encoder_q, r=args.rank, alpha=args.alpha)
            inject_lora(model.encoder_k, r=args.rank, alpha=args.alpha)
            model = model.to(device)  # 新注入的 LoRA 模块回迁 device
            copy_params_q_to_k(model)

        # 3) SupCon 适配
        from labelfree_tier_replication import dump_features
        store = dump_features(model, tier, device)  # 仅取对齐的 X；y2 索引与 train 行一致
        X = None
        from feeder.ntu_feeder import Feeder_single  # noqa: E402
        feeder_dir = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
        X = np.load(feeder_dir / "train_position.npy", mmap_mode="r")
        supcon_adapt(model, teacher, np.asarray(X), y_sem, args, device, out_dir, loss_hist)

        # 4) 统一尺重测
        from labelfree_tier_replication import run_arms
        arms = run_arms(model, tier, seeds if rd == args.rounds else (args.psd_seed,), device)
        rounds_meta.append({
            "round": rd,
            "pseudo_labeled": int(np.sum([v is not None for v in y_sem])),
            "full_ref": round(arms["full_ref"], 4),
            "linear_10pct": round(arms["linear_10pct"], 4),
            "b_arms": [round(a, 4) for a in arms["b_arms"]],
            "retention": round(arms["retention"], 4),
            "verdict": arms["verdict"],
        })
        print(f"[round {rd}] retention={arms['retention']:.4f} verdict={arms['verdict']}", flush=True)

    result = {
        "doneMarker": f"P1_ADAPT_SUPCON_{tier.upper()}_DONE",
        "experiment": "e10_t3_supcon_pseudo_feedback",
        "tier": tier, "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD pseudo-label feedback loop: confidence-gated pseudo-labels from the "
                    "(b) semantic pipeline retrain the physical encoder via supervised "
                    "contrastive loss (LoRA low-rank, cosine distillation anchor to the frozen "
                    "pretext encoder); E9 three arms re-measured with the unified ruler",
        "ckpt": str(ckpt.relative_to(REPO)),
        "train": {"rounds": args.rounds, "epochs_per_round": args.epochs, "batch": args.batch,
                  "lr": args.lr, "temperature": args.temperature,
                  "distill_lambda": args.distill, "psd_seed": args.psd_seed,
                  "loss_curve_tail": [x for x in loss_hist[-5:] if x is not None]},
        "rounds": rounds_meta,
    }
    final = rounds_meta[-1]
    result.update({"full_ref": final["full_ref"], "linear_10pct": final["linear_10pct"],
                   "b_arms": final["b_arms"], "retention": final["retention"],
                   "verdict": final["verdict"], "seeds": list(seeds)})
    result["wall_clock_sec"] = round(time.time() - t0, 1)

    out = out_dir / "summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps({k: final[k] for k in ("round", "full_ref", "linear_10pct",
                                            "retention", "verdict")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
