# -*- coding: utf-8 -*-
"""E10-T2 — 参数高效继续预训练: 物理层只在低秩子空间演化（LoRA）+ 源锚蒸馏。

动机: 冻结物理层在弱参照域封顶。本线让编码器在**目标域全量无标签数据**上
继续 AimCLR 对比预训练，但可训练参数被约束在 LoRA 低秩分支（<5% 参数），
并以冻结原模型为蒸馏锚（特征余弦）防灾难性遗忘。低秩约束是 PSD
"物理层缓慢演化"哲学的数学化: 适配不推翻预训练几何，只做低秩修正。

协议:
  1) 加载已发布 pretext ckpt；encoder_q/k 注入 LoRA（B=0 → 起点=冻结模型）；
     teacher = 原始 encoder_q 深拷贝（全冻结）；
  2) Feeder_triple 三视图（与 pretext 同参: shear 0.5, padding 6），仅 LoRA 可训练，
     SGD 继续预训练 epochs 轮；损失 = AimCLR(CE + DDM) + λ_d·(1-cos(f_s,f_t))；
  3) 适配后全模型冻结 → run_arms 三臂 ×seeds（统一尺）→ 新保留率/判据。

用法:
    .venv/Scripts/python.exe -u scripts/p1_sprint/lora_continue_tier.py --tier ntu120
产出:
    runs/p1_sprint/adapt_lora_<tier>/summary.json (+ 每 save_every 轮 ckpt)
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from adapt_common import (REPO, adapt_out_dir, copy_params_q_to_k, cosine_distill,
                          count_params, extract_backbone, inject_lora,
                          lora_parameters, load_frozen_model)


def aimclr_pretext_loss(model, data1, data2, data3):
    """与 external/AimCLR processor/pretrain_aimclr.train 逐式一致（epoch≤mining 分支）。"""
    logits, labels, logits_e, logits_ed, labels_ddm = model(data1, data2, data3)
    model.update_ptr(logits.size(0))
    loss1 = F.cross_entropy(logits, labels)
    loss2 = -torch.mean(torch.sum(torch.log(logits_e) * labels_ddm, dim=1))
    loss3 = -torch.mean(torch.sum(torch.log(logits_ed) * labels_ddm, dim=1))
    return loss1 + (loss2 + loss3) / 2.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=["ntu120", "ucf101", "panaf500"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--distill", type=float, default=1.0, help="蒸馏锚权重 λ_d")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--source-ckpt", default=None,
                    help="跨域模式: 源域编码器 ckpt（如 NTU60 phaseB epoch300）。"
                         "缺省=该 tier 自身 pretext（同域继续预训练）")
    ap.add_argument("--zero-shot", action="store_true",
                    help="只做源编码器零样本迁移评测（run_arms），不训练")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tier = args.tier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.epochs = 2
        args.save_every = 2
    seeds = tuple(range(42, 42 + (1 if args.smoke else args.seeds)))
    t0 = time.time()
    torch.manual_seed(42)

    model, ckpt = load_frozen_model(tier, device, ckpt_path=args.source_ckpt)
    n_params_total = sum(p.numel() for p in model.parameters())
    mode = "cross_domain" if args.source_ckpt else "same_domain"

    if args.zero_shot:
        from labelfree_tier_replication import run_arms
        arms = run_arms(model, tier, tuple(range(42, 42 + (1 if args.smoke else args.seeds))),
                        device)
        result = {
            "doneMarker": f"P1_ADAPT_LORAZS_{tier.upper()}_DONE",
            "experiment": "e10_t2b_cross_domain_zero_shot",
            "tier": tier, "mode": mode, "zero_shot": True,
            "date": datetime.now().isoformat(timespec="seconds"),
            "ckpt": str(ckpt.relative_to(REPO)),
            "full_ref": round(arms["full_ref"], 4),
            "linear_10pct": round(arms["linear_10pct"], 4),
            "b_arms": [round(a, 4) for a in arms["b_arms"]],
            "retention": round(arms["retention"], 4),
            "verdict": arms["verdict"],
            "seeds": list(range(42, 42 + (1 if args.smoke else args.seeds))),
            "wall_clock_sec": round(time.time() - t0, 1),
        }
        out = adapt_out_dir("lora_xfer", tier) / "summary_zeroshot.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[done] {out}")
        print(json.dumps({k: result[k] for k in ("full_ref", "linear_10pct",
                                                 "retention", "verdict")},
                         ensure_ascii=False, indent=1))
        return

    # teacher = 原始冻结 encoder_q（注入前深拷贝）
    teacher = copy.deepcopy(model.encoder_q).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # LoRA 注入 q/k，k 全量同步后冻结（注入的新模块在 CPU，需整体回迁 device）
    n_lora_q = inject_lora(model.encoder_q, r=args.rank, alpha=args.alpha)
    n_lora_k = inject_lora(model.encoder_k, r=args.rank, alpha=args.alpha)
    assert n_lora_q == n_lora_k, "q/k LoRA 注入数不一致"
    model = model.to(device)
    copy_params_q_to_k(model)
    params = lora_parameters(model.encoder_q)
    trainable_ids = {id(p) for p in params}
    for p in model.encoder_q.parameters():
        p.requires_grad = id(p) in trainable_ids
    tr, tt = count_params(model)
    print(f"[lora] injected {n_lora_q} convs per encoder; trainable={tr:,}/{tt:,} "
          f"({100*tr/tt:.2f}%)", flush=True)

    # 数据: 与 pretext 同一 Feeder_triple 三视图
    from feeder.ntu_feeder import Feeder_triple  # noqa: E402  (external/AimCLR)
    feeder_dir = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
    feeder = Feeder_triple(str(feeder_dir / "train_position.npy"),
                           str(feeder_dir / "train_label.pkl"),
                           shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=args.batch, shuffle=True,
                                         drop_last=True, num_workers=0)

    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    out_dir = adapt_out_dir("lora", tier)
    loss_hist = []
    nan_batches = 0
    model.train()
    for epoch in range(1, args.epochs + 1):
        if epoch == int(args.epochs * 0.67):
            for g in opt.param_groups:
                g["lr"] = args.lr * 0.1
        ep_losses = []
        for (data1, data2, data3), _label in loader:
            data1 = data1.float().to(device, non_blocking=True)
            data2 = data2.float().to(device, non_blocking=True)
            data3 = data3.float().to(device, non_blocking=True)
            loss = aimclr_pretext_loss(model, data1, data2, data3)
            if args.distill > 0:
                with torch.no_grad():
                    f_t = extract_backbone(teacher, data2)
                f_s = extract_backbone(model.encoder_q, data2)
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
        print(f"[lora] epoch {epoch}/{args.epochs} loss={loss_hist[-1]} "
              f"nan_skipped={nan_batches}", flush=True)
        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), out_dir / f"lora_epoch{epoch}_model.pt")
    model.eval()

    # 适配后统一尺三臂
    arms = None
    if not args.smoke:
        from labelfree_tier_replication import run_arms
        arms = run_arms(model, tier, seeds, device)

    result = {
        "doneMarker": f"P1_ADAPT_LORA_{tier.upper()}_DONE",
        "experiment": "e10_t2_lora_continue_pretrain",
        "tier": tier, "mode": mode, "zero_shot": False,
        "mode_note": ("cross_domain: strong-reference source encoder adapted to the weak "
                      "tier with LoRA + distillation anchor" if mode == "cross_domain"
                      else "same_domain: the tier's own from-scratch pretext continued"),
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "LoRA low-rank continued AimCLR pretraining on ALL unlabeled target-domain "
                    "train data (Feeder_triple, shear 0.5 / pad 6), plus cosine feature "
                    "distillation to the frozen pretext encoder (anti-forgetting anchor); "
                    "only LoRA branches trainable (<5% params); then E9 three arms re-measured "
                    "with the unified ruler",
        "ckpt": str(ckpt.relative_to(REPO)),
        "lora": {"rank": args.rank, "alpha": args.alpha, "convs_per_encoder": n_lora_q,
                 "trainable_params": int(tr), "total_params": int(tt),
                 "trainable_pct": round(100 * tr / tt, 3)},
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "optimizer": "SGD(momentum=0.9, wd=1e-4)", "distill_lambda": args.distill,
                  "loss_curve_tail": loss_hist[-5:]},
    }
    if arms:
        result.update({
            "full_ref": round(arms["full_ref"], 4),
            "linear_10pct": round(arms["linear_10pct"], 4),
            "b_arms": [round(a, 4) for a in arms["b_arms"]],
            "retention": round(arms["retention"], 4),
            "verdict": arms["verdict"],
            "seeds": list(seeds),
        })
    result["wall_clock_sec"] = round(time.time() - t0, 1)

    out_dir = adapt_out_dir("lora_xfer" if mode == "cross_domain" else "lora", tier)
    out = out_dir / "summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    show = {"trainable_pct": result["lora"]["trainable_pct"], "loss_tail": loss_hist[-3:]}
    if arms:
        show.update({"full_ref": result["full_ref"], "linear_10pct": result["linear_10pct"],
                     "retention": result["retention"], "verdict": result["verdict"]})
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
