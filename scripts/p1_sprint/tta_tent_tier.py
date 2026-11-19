# -*- coding: utf-8 -*-
"""E10-T1 — 测试时适配（TTA, TENT 风格）: 让冻结物理层在推理时"动起来"。

动机: E9 四层协议显示冻结物理层在弱参照域封顶（NTU120 retention 74.96% FAILS）。
本脚本检验最便宜的"动起来": 不碰训练集、不加标注预算，仅在**无标签测试流**上
以预测熵最小化更新 BatchNorm 仿射参数（TENT, ICLR'21），再回到与已发布数字
同一把尺（run_arms 三臂）重测。若 FAILS 域被显著拉动，说明瓶颈在特征校准
而非标签量——这是三线中最便宜的因果证据。

协议:
  1) 加载已发布 pretext ckpt，冻结校验（trainable=0）；
  2) 冻结特征上拟线性头: full（(c) 臂口径）与 10%（(a) 臂口径, seed 42）；
  3) TENT: val 流逐批（shuffle=False 确定序），模型 train() 态（BN 用批统计），
     仅 BN affine requires_grad，Adam lr=1e-3 每批一步，预测取本批前向输出；
  4) 在线 TTA 准确率（固定头）+ 适配后重导特征 → run_arms 三臂 ×seeds
     （(b) 臂即 "PSD+TTA"）。

用法:
    .venv/Scripts/python.exe -u scripts/p1_sprint/tta_tent_tier.py --tier ntu120
产出:
    runs/p1_sprint/adapt_tta_<tier>/summary.json
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import numpy as np
import torch

from adapt_common import (REPO, backbone_logits, adapt_out_dir, count_params,
                          load_frozen_model, torch_linear_head)

BATCH = 128


def tent_adapt(model, enc, lin, mean, std, tier: str, device: str,
               lr: float, steps: int, scope: str = "all", bn_mode: str = "train"):
    """在无标签 val 流上做 TENT；返回 (在线预测, 在线标签, 适配参数量)。

    scope: all=全部 BN 仿射 | data_bn=仅输入 BN（最小侵入）。
    bn_mode: train=BN 用批统计（TENT 原版口径） | eval=BN 用冻结 running stats，
             仅仿射可训练（与线性头拟合时的特征口径一致）。
    """
    from feeder.ntu_feeder import Feeder_single  # noqa: E402  (external/AimCLR)

    bns = [m for m in enc.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if scope == "data_bn":
        bns = [enc.data_bn]
    for p in model.parameters():
        p.requires_grad_(False)
    for m in bns:
        for p in m.parameters():
            p.requires_grad_(True)
    n_adapt = sum(p.numel() for m in bns for p in m.parameters())
    opt = torch.optim.Adam([p for m in bns for p in m.parameters()], lr=lr)
    print(f"[tent] adapting {len(bns)} BN layers, {n_adapt} affine params, lr={lr}", flush=True)

    feeder_dir = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
    feeder = Feeder_single(str(feeder_dir / "val_position.npy"),
                           str(feeder_dir / "val_label.pkl"),
                           shear_amplitude=-1, temperal_padding_ratio=-1, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=BATCH, shuffle=False, num_workers=0)

    if bn_mode == "train":
        model.train()  # BN 用批统计（TENT 原版口径）；仅 BN 仿射可训练
    else:
        model.eval()  # running stats 冻结，仅仿射更新
    preds, ys = [], []
    for data, label in loader:
        x = data.float().to(device)
        y = np.asarray(label).reshape(-1)
        for _ in range(steps):
            logits = backbone_logits(enc, lin, mean, std, x)
            ent = -(torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1)).sum(1).mean()
            loss = -ent  # 最小化熵
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            logits = backbone_logits(enc, lin, mean, std, x)
        preds.append(logits.argmax(dim=1).cpu().numpy())
        ys.append(y)
    model.eval()
    return np.concatenate(preds), np.concatenate(ys), n_adapt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=["ntu120", "ucf101", "panaf500"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=1, help="每批梯度步数")
    ap.add_argument("--head", choices=["10pct", "full"], default="10pct",
                    help="TENT 熵所用的分类器（10pct=预算诚实口径）")
    ap.add_argument("--scope", choices=["all", "data_bn"], default="all",
                    help="BN 仿射适配范围")
    ap.add_argument("--bn-mode", choices=["train", "eval"], default="train",
                    help="适配期 BN 统计口径")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    tier = args.tier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = tuple(range(42, 42 + (1 if args.smoke else args.seeds)))
    t0 = time.time()

    model, ckpt = load_frozen_model(tier, device)
    enc = model.encoder_q

    # 冻结特征 + 两个线性头（full / 10%），先复算基线读数（应与已发布一致）
    from labelfree_tier_replication import ScaledLR, dump_features
    store = dump_features(model, tier, device)
    tr_feat, tr_lab = store["train_feat"], store["train_label"]
    va_feat, va_lab = store["val_feat"], store["val_label"]
    ref_acc = float(np.mean(ScaledLR().fit(tr_feat, tr_lab).predict(va_feat) == va_lab))
    rng = np.random.default_rng(42)
    mask_a = np.zeros(len(tr_lab), dtype=bool)
    for c in sorted(set(tr_lab.tolist())):
        ci = np.where(tr_lab == c)[0]
        k = max(1, int(round(len(ci) * 0.10)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    lin10_acc = float(np.mean(ScaledLR().fit(tr_feat[mask_a], tr_lab[mask_a]).predict(va_feat) == va_lab))
    print(f"[pre-tta] full ref={ref_acc:.4f} 10% linear={lin10_acc:.4f}", flush=True)

    lin, mean, std = torch_linear_head(tr_feat[mask_a] if args.head == "10pct" else tr_feat,
                                       tr_lab[mask_a] if args.head == "10pct" else tr_lab,
                                       device)

    # TENT 适配 + 在线预测
    preds, ys, n_adapt = tent_adapt(model, enc, lin, mean, std, tier, device,
                                    args.lr, args.steps, scope=args.scope,
                                    bn_mode=args.bn_mode)
    online_acc = float(np.mean(preds == ys))
    n_unique = int(len(np.unique(preds)))
    print(f"[tent] online acc={online_acc:.4f} unique_preds={n_unique}/{preds.max()+1}", flush=True)

    # 适配后固定头复测（TENT 标准口径: 头不动，特征已校准）
    store2 = dump_features(model, tier, device)
    va_feat2 = store2["val_feat"]
    post_fixed_acc = float(np.mean(
        lin((torch.from_numpy(va_feat2).to(device) - mean) / std).argmax(1).cpu().numpy() == va_lab))

    # 适配后重导特征 → E9 三臂统一尺（(b) 臂即 PSD+TTA）
    arms = None
    if not args.smoke:
        from labelfree_tier_replication import run_arms
        arms = run_arms(model, tier, seeds, device)

    result = {
        "doneMarker": f"P1_ADAPT_TTA_{tier.upper()}_DONE",
        "experiment": "e10_t1_tta_tent",
        "tier": tier, "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "TENT test-time entropy minimization over BatchNorm affine params "
                    "on the unlabeled val stream; classifier fixed at the 10% linear head "
                    f"({args.head}); then E9 three arms re-measured with the unified ruler",
        "ckpt": str(ckpt.relative_to(REPO)),
        "tent": {"scope": "data_bn_only" if args.scope == "data_bn" else "all_batchnorm_affine",
                 "lr": args.lr, "steps_per_batch": args.steps,
                 "batch": BATCH, "head": args.head, "adapted_params": int(n_adapt),
                 "order": "val stream, shuffle=False (deterministic)"},
        "pre_tta": {"full_ref": round(ref_acc, 4), "linear_10pct": round(lin10_acc, 4)},
        "post_tta_online_acc_fixed_head": round(online_acc, 4),
        "post_tta_rehead_val_acc": round(post_fixed_acc, 4),
        "unique_predicted_classes": n_unique,
        "collapse_warning": bool(n_unique <= 2),
    }
    if arms:
        result.update({
            "full_ref": round(arms["full_ref"], 4),
            "linear_10pct": round(arms["linear_10pct"], 4),
            "b_arms": [round(a, 4) for a in arms["b_arms"]],
            "retention": round(arms["retention"], 4),
            "verdict": arms["verdict"],
            "seeds": list(seeds),
            "note": "post-TTA features re-fed to the SAME run_arms ruler; "
                    "linear_10pct here re-fits the head on adapted features",
        })
    result["wall_clock_sec"] = round(time.time() - t0, 1)

    out = adapt_out_dir("tta", tier) / "summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    show = {k: result[k] for k in ("pre_tta", "post_tta_online_acc_fixed_head",
                                   "post_tta_rehead_val_acc") if k in result}
    if arms:
        show.update({"full_ref": result["full_ref"], "linear_10pct": result["linear_10pct"],
                     "retention": result["retention"], "verdict": result["verdict"]})
    print(json.dumps(show, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
