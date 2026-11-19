# -*- coding: utf-8 -*-
"""tta_grid_probe — T1 内部选型探针（不产结论数字，只选不塌缩的 TENT 配置）。

一次特征导出，多组 (head, lr, scope) 各自从 ckpt 复位模型跑 TENT，
以在线固定头准确率+预测类数判断塌缩。选定配置后再用 tta_tent_tier.py 正式跑。
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from adapt_common import load_frozen_model, torch_linear_head
from labelfree_tier_replication import dump_features
from tta_tent_tier import tent_adapt

GRID = [
    ("10pct", 1e-3, "all", "train"),
    ("full", 1e-3, "all", "train"),
    ("10pct", 1e-4, "all", "eval"),
    ("10pct", 1e-5, "all", "eval"),
    ("full", 1e-4, "all", "eval"),
    ("full", 1e-5, "all", "eval"),
    ("10pct", 1e-4, "data_bn", "eval"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="ntu120")
    args = ap.parse_args()
    tier = args.tier
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load_frozen_model(tier, device)
    store = dump_features(model, tier, device)
    tr_feat, tr_lab = store["train_feat"], store["train_label"]
    va_feat, va_lab = store["val_feat"], store["val_label"]
    rng = np.random.default_rng(42)
    mask_a = np.zeros(len(tr_lab), dtype=bool)
    for c in sorted(set(tr_lab.tolist())):
        ci = np.where(tr_lab == c)[0]
        k = max(1, int(round(len(ci) * 0.10)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True

    # 诊断基线: train 态（批统计）不更新时的固定头准确率 —— 分离"BN 口径差"与"适配损伤"
    from adapt_common import REPO, backbone_logits as bl
    from feeder.ntu_feeder import Feeder_single  # noqa: E402
    lin0, mean0, std0 = torch_linear_head(tr_feat[mask_a], tr_lab[mask_a], device)
    enc0 = model.encoder_q
    model.train()
    accs = []
    fd = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
    fe = Feeder_single(str(fd / "val_position.npy"), str(fd / "val_label.pkl"),
                       shear_amplitude=-1, temperal_padding_ratio=-1, mmap=True)
    ld = torch.utils.data.DataLoader(fe, batch_size=128, shuffle=False, num_workers=0)
    for data, label in ld:
        with torch.no_grad():
            lg = bl(enc0, lin0, mean0, std0, data.float().to(device))
        accs.append(lg.argmax(1).cpu().numpy() == np.asarray(label).reshape(-1))
    model.eval()
    print(f"{'no-adapt train-mode fixed-head acc':40s} {np.mean(np.concatenate(accs)):.4f}",
          flush=True)

    print(f"{'head':6} {'lr':8} {'scope':8} {'bn':6} {'online_acc':>10} {'uniq':>5}", flush=True)
    for head, lr, scope, bn in GRID:
        model, _ = load_frozen_model(tier, device)  # 每组从 ckpt 复位
        enc = model.encoder_q
        feat, lab = (tr_feat[mask_a], tr_lab[mask_a]) if head == "10pct" else (tr_feat, tr_lab)
        lin, mean, std = torch_linear_head(feat, lab, device)
        preds, ys, n = tent_adapt(model, enc, lin, mean, std, tier, device, lr, 1,
                                  scope=scope, bn_mode=bn)
        acc = float(np.mean(preds == ys))
        uniq = int(len(np.unique(preds)))
        print(f"{head:6} {lr:<8} {scope:8} {bn:6} {acc:>10.4f} {uniq:>5}", flush=True)


if __name__ == "__main__":
    main()
