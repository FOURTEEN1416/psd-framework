# -*- coding: utf-8 -*-
"""peft_budget_tier — BitFit / Adapter 参数高效基线（NTU120 10% 锚点）。

与 fullft_budget_tier.py 同协议，只换可训练参数集合：
  · bitfit : 冻结 encoder 全部权重，只训练所有 bias（Conv/BN/Linear）+ 新头；
  · adapter: 每个 Conv2d 旁挂 bottleneck adapter（1x1 降维→ReLU→1x1 升维，
             第二层零初始化，残差），只训练 adapter + 新头。
BN running stats 冻结（eval 口径）与 recycling/fullft 一致。

用法：
    .venv/Scripts/python.exe scripts/p1_sprint/peft_budget_tier.py --mode bitfit --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/peft_budget_tier.py --mode adapter --smoke
产出：
    runs/p1_sprint/adapt_t4_ntu120/summary_peft_<mode>_<tag>.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adapt_common import REPO, adapt_out_dir, load_frozen_model
from fullft_budget_tier import BUDGET_FRAC, build_anchor, run_semantic_arm
from labelfree_tier_replication import TIERS, dump_features, ScaledLR, extract_backbone


class Conv2dAdapter(nn.Module):
    """Conv2d + bottleneck adapter 残差分支；up 零初始化 → 初始恒等。"""

    def __init__(self, conv: nn.Conv2d, bottleneck: int = 16):
        super().__init__()
        self.conv = conv
        self.down = nn.Conv2d(conv.out_channels, bottleneck, 1, bias=False)
        self.up = nn.Conv2d(bottleneck, conv.out_channels, 1, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        y = self.conv(x)
        return y + self.up(F.relu(self.down(y)))


def inject_adapter(module: nn.Module, bottleneck: int = 16) -> int:
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, Conv2dAdapter(child, bottleneck))
            n += 1
        else:
            n += inject_adapter(child, bottleneck)
    return n


def adapter_parameters(module: nn.Module) -> list:
    return [p for m in module.modules() if isinstance(m, Conv2dAdapter)
            for p in list(m.down.parameters()) + list(m.up.parameters())]


def configure_trainable(model, mode: str, bottleneck: int) -> list:
    """冻结全部，返回要训练的参数列表（含新头由调用方追加）。"""
    for p in model.parameters():
        p.requires_grad = False
    if mode == "bitfit":
        for m in model.modules():
            for name, p in m.named_parameters(recurse=False):
                if name.endswith("bias"):
                    p.requires_grad = True
        return [p for p in model.parameters() if p.requires_grad]
    if mode == "adapter":
        inject_adapter(model.encoder_q, bottleneck)
        return adapter_parameters(model.encoder_q)
    raise ValueError(mode)


def finetune_peft(model, X_t, anchor_idx, y_anchor, n_cls, device, epochs, batch, lr,
                  aug_views, feeder_args, mode: str, bottleneck: int):
    params = configure_trainable(model, mode, bottleneck)
    model = model.to(device)  # 新注入的 adapter 模块随模型搬到 GPU
    head = nn.Linear(256, n_cls).to(device)
    nn.init.xavier_uniform_(head.weight)
    params = params + list(head.parameters())
    print(f"[{mode}] trainable params={sum(p.numel() for p in params):,}", flush=True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    model.eval()
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
                loss = F.cross_entropy(head(extract_backbone(model.encoder_q, xs)), ys)
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
                loss = F.cross_entropy(
                    head(extract_backbone(model.encoder_q, x_all[idx])), y_all[idx])
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
            print(f"  [{mode}] epoch {ep}/{epochs} loss={losses[-1]} anchor_acc={acc:.3f}",
                  flush=True)
    return model, head, losses, accs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="ntu120")
    ap.add_argument("--mode", required=True, choices=["bitfit", "adapter"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--aug-views", type=int, default=1)
    ap.add_argument("--bottleneck", type=int, default=16)
    ap.add_argument("--tag", default="peft")
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
    out_path = adapt_out_dir("t4", tier) / f"summary_peft_{args.mode}_{args.tag}.json"
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
        print(f"[peft:{args.mode}] seed {seed}", flush=True)
        model, ckpt = load_frozen_model(tier, device)
        if not labels_checked:
            store0 = dump_features(model, tier, device)
            assert np.array_equal(np.asarray(store0["train_label"]), pkl_labels), \
                "pkl 标签序与 dump_features 不一致"
            labels_checked = True
        labels_str0 = np.array([str(l) for l in pkl_labels])
        anchor = build_anchor(labels_str0, seed, BUDGET_FRAC)
        anchor_idx = np.where(anchor)[0]
        y_anchor = pkl_labels[anchor]
        model, head, losses, accs = finetune_peft(
            model, X_t, anchor_idx, y_anchor, n_cls, device,
            args.epochs, args.batch, args.lr, args.aug_views,
            {"data_path": str(fd / "train_position.npy"),
             "label_path": str(fd / "train_label.pkl")},
            args.mode, args.bottleneck)
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
        print(f"[peft:{args.mode}] seed{seed}: ref={ref_j:.4f} b={acc_b:.4f} "
              f"ret={rows[-1]['retention_seed']:.4f} pool={pool}", flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "experiment": "e10_t4_budget_split", "arm": f"peft_{args.mode}", "tier": tier,
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
        "doneMarker": f"P1_ADAPT_T4_PEFT_{args.mode.upper()}_{tier.upper()}_DONE",
        "experiment": "e10_t4_budget_split", "arm": f"peft_{args.mode}", "tier": tier,
        "mode": "smoke" if args.smoke else "full",
        "budget_frac": BUDGET_FRAC,
        "protocol": f"per-seed {args.mode} adaptation on THIS seed's anchors "
                    "(frozen backbone, BN running stats frozen), then re-dump features, "
                    "full-label linear ref, and the SAME-anchor (b) semantic arm",
        "date": datetime.now().isoformat(timespec="seconds"),
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "aug_views": args.aug_views, "bottleneck": args.bottleneck},
        "per_seed": rows, "mean_b": mean_b, "mean_full_ref": mean_ref,
        "retention": ret, "verdict": verdict, "seeds": [r["seed"] for r in rows],
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = adapt_out_dir("t4", tier) / f"summary_peft_{args.mode}_{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps({"mean_b": mean_b, "mean_full_ref": mean_ref,
                      "retention": ret, "verdict": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
