# -*- coding: utf-8 -*-
"""P1 大修 — 监督 ST-GCN 基线扩充到第二基准：NTU120（E9b 12,000-clip/60-class 子集口径）.

回应评估工作流确认的 "head-to-head confined to one benchmark" 弱点：把头对头
扩到 NTU120。与 NTU60 头对头同协议：
  - 数据：runs/p1_sprint/labelfree_ntu120/feeder/（E9b 同一 12k 子集导出，
    25 槽 NTU 图，train 7,337 / val 4,663，60 类）
  - 预算：10% 分层标注（protocol-frozen seed-42 draw，每类 k=max(1,round(10%·n_c))，
    与 E9b 10% 线性头 44.18% 同一抽选口径）
  - 模型：net.st_gcn.Model（external/AimCLR，与 AimCLR SSL 骨干同类）端到端 CE，
    hidden 16（受控容量，同 NTU60 头对头口径）
  - 配方：SGD lr0.1 step[50,70] wd1e-4 momentum0.9(nesterov off)、80ep、batch128、
    eval every 5 报 best top-1；5 个初始化种子（42-46，同 NTU60 基线惯例）
  - 对照臂（已在稿内）：AimCLR linear 44.18%（同特征线性头，确定性单次）、
    PSD label-free 管线 43.75%±1.16（十种子）
产出:
    runs/p1_sprint/supervised_stgcn_ntu120/summary.json（冒烟为 summary_smoke.json）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
AIMCLR_ROOT = REPO_ROOT / "external" / "AimCLR"
FEEDER_DIR = "runs/p1_sprint/labelfree_ntu120/feeder"
RUN_DIR = REPO_ROOT / "runs" / "p1_sprint" / "supervised_stgcn_ntu120"
N_CLASS = 60
BUDGET_FRAC = 0.10
BUDGET_SEED = 42
SEEDS = (42, 43, 44, 45, 46)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def stratified_mask(labels: np.ndarray, seed: int, frac: float = BUDGET_FRAC) -> np.ndarray:
    """E9 系逐行同式分层抽选：每类 k=max(1, round(frac*n_c))，default_rng(seed)。"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in sorted(set(labels.tolist())):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * frac)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    return mask


def make_loader(data_rel: str, label_rel: str, batch: int, shuffle: bool, drop_last: bool):
    import torch  # noqa: F401
    from feeder.ntu_feeder import Feeder_single  # noqa: E402  (external/AimCLR)

    ds = Feeder_single(data_path=data_rel, label_path=label_rel,
                       shear_amplitude=-1, temperal_padding_ratio=-1, mmap=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle,
                                       pin_memory=True, num_workers=0,
                                       drop_last=drop_last)


def adjust_lr(base_lr: float, epoch1: int, steps: list) -> float:
    import numpy as np
    return base_lr * (0.1 ** np.sum(epoch1 > np.array(steps)))


def evaluate(model, loader, device) -> float:
    import torch

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for data, label in loader:
            data = data.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            out = model(data)
            correct += int((out.argmax(dim=1) == label).sum().item())
            total += int(label.numel())
    return correct / max(1, total)


def train_budget(tag: str, train: dict, val: dict, n_train: int, args, work_root: str) -> dict:
    import torch
    import torch.nn as nn
    from net.st_gcn import Model  # noqa: E402  (external/AimCLR 官方 ST-GCN)
    from processor.processor import init_seed  # noqa: E402

    init_seed(args.seed)
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"

    work_dir = (REPO_ROOT / work_root) / tag
    work_dir.mkdir(parents=True, exist_ok=True)
    train_loader = make_loader(train["data"], train["label"], args.batch_size, True, True)
    val_loader = make_loader(val["data"], val["label"], args.batch_size, False, False)

    model = Model(in_channels=3, hidden_channels=args.hidden_channels, hidden_dim=256,
                  num_class=N_CLASS, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                  edge_importance_weighting=True, dropout=0.5).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                nesterov=False, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()

    best_top1, best_epoch = 0.0, -1
    t_budget = time.time()
    for epoch in range(1, args.epochs + 1):
        lr = adjust_lr(args.lr, epoch, args.steps)
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        t_ep = time.time()
        loss_sum, n_batches = 0.0, 0
        for data, label in train_loader:
            data = data.float().to(device, non_blocking=True)
            label = label.long().to(device, non_blocking=True)
            out = model(data)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            n_batches += 1
        msg = (f"[stgcn:{tag}] epoch {epoch}/{args.epochs} lr={lr:.4f} "
               f"loss={loss_sum / max(1, n_batches):.4f} ({time.time() - t_ep:.1f}s)")
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            top1 = evaluate(model, val_loader, device)
            if top1 >= best_top1:
                best_top1, best_epoch = top1, epoch
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_top1": top1, "budget": tag, "n_train": n_train},
                           work_dir / "best_model.pt")
            msg += f" val_top1={top1 * 100:.2f}% best={best_top1 * 100:.2f}%"
        print(msg, flush=True)

    print(f"[stgcn:{tag}] best top1 = {best_top1 * 100:.2f}% (epoch {best_epoch}, "
          f"{time.time() - t_budget:.1f}s, params={n_params / 1e6:.2f}M)")
    return {"top1": round(best_top1, 4), "best_epoch": int(best_epoch),
            "n_train": int(n_train), "n_params": int(n_params),
            "work_dir": str(work_dir.relative_to(REPO_ROOT)).replace("\\", "/")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="缩样管线验证（数字无意义）")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 70])
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden-channels", type=int, default=16,
                    help="16=与 SSL 骨干同容量受控变体（同 NTU60 头对头口径）")
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not AIMCLR_ROOT.exists():
        raise SystemExit(f"[supervised-stgcn] external/AimCLR 缺失: {AIMCLR_ROOT}")

    t0 = time.time()
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))

    import pickle
    with open(REPO_ROOT / FEEDER_DIR / "train_label.pkl", "rb") as f:
        _, tr_lab = pickle.load(f)
    with open(REPO_ROOT / FEEDER_DIR / "val_label.pkl", "rb") as f:
        _, va_lab = pickle.load(f)
    tr_lab = np.array(tr_lab)
    classes = sorted(set(tr_lab.tolist()))
    remap = {c: i for i, c in enumerate(classes)}
    tr_lab_c = np.array([remap[int(l)] for l in tr_lab])
    va_lab_c = np.array([remap[int(l)] for l in va_lab])
    assert len(classes) == N_CLASS

    # 10% 分层子集（protocol-frozen seed-42 draw）落盘为独立 label pkl，Feeder 直接消费
    mask10 = stratified_mask(tr_lab_c, BUDGET_SEED)
    sub_dir = REPO_ROOT / FEEDER_DIR / "budget10"
    sub_dir.mkdir(parents=True, exist_ok=True)
    np.save(sub_dir / "train10_mask.npy", mask10)
    with open(sub_dir / "train_label_10pct.pkl", "wb") as f:
        pickle.dump(([f"train10_{i:06d}" for i in range(int(mask10.sum()))],
                     [int(x) for x in tr_lab_c[mask10]]), f)

    if args.smoke:
        args.epochs, args.batch_size, args.eval_interval = 2, 64, 1
        seeds = (42,)
        work_root = "runs/p1_sprint/supervised_stgcn_ntu120/smoke"
        tr10_data = FEEDER_DIR + "/train_position.npy"  # 冒烟不抽子集，仅验证管线
    else:
        seeds = tuple(args.seeds)
        work_root = "runs/p1_sprint/supervised_stgcn_ntu120"
        tr10_data = FEEDER_DIR + "/budget10/train_position_10pct.npy"
        # 落盘 10% 子集原始序列（Feeder 消费）
        full = np.load(REPO_ROOT / FEEDER_DIR / "train_position.npy", mmap_mode="r")
        np.save(sub_dir / "train_position_10pct.npy",
                np.ascontiguousarray(full[np.where(mask10)[0]]))
        del full

    val = {"data": FEEDER_DIR + "/val_position.npy",
           "label": FEEDER_DIR + "/budget10/val_label_remap.pkl"}
    with open(REPO_ROOT / val["label"], "wb") as f:
        pickle.dump(([f"val_{i:06d}" for i in range(len(va_lab_c))],
                     [int(x) for x in va_lab_c]), f)

    os.chdir(REPO_ROOT)  # 相对数据路径按 repo root 解析
    per_seed = {}
    for s in seeds:
        args.seed = s
        per_seed[s] = train_budget(f"10pct_s{s}",
                                   {"data": tr10_data,
                                    "label": FEEDER_DIR + "/budget10/train_label_10pct.pkl"},
                                   val, int(mask10.sum()), args, work_root)

    tops = [v["top1"] for v in per_seed.values()]
    mean = float(np.mean(tops))
    std = float(np.std(tops, ddof=1)) if len(tops) > 1 else 0.0
    result = {
        "doneMarker": f"P1_SPRINT_SUPERVISED_STGCN_NTU120_{'SMOKE_OK' if args.smoke else 'DONE'}",
        "experiment": "p1_sprint_supervised_stgcn_ntu120",
        "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "method": "Supervised ST-GCN (end-to-end, cross-entropy) on NTU120 "
                  "(E9b 12,000-clip/60-class subset caliber)",
        "implementation": "in-repo/official-code ST-GCN (external/AimCLR/net/st_gcn.py); "
                          "same controlled 16-hidden-channel capacity as the NTU60 head-to-head",
        "budgets": {"10pct": {"per_seed": per_seed,
                              "mean": round(mean, 4), "std": round(std, 4),
                              "seeds": list(seeds)}},
        "budget_definition": {
            "frac": BUDGET_FRAC, "seed": BUDGET_SEED,
            "rule": "stratified per-class k=max(1, round(0.10*n_c)) over the 7,337 "
                    "train clips - same formula as the E9b 10% linear head (44.18%)",
        },
        "recipe": {"epochs": args.epochs, "batch": args.batch_size, "lr": args.lr,
                   "steps": args.steps, "weight_decay": args.wd, "momentum": 0.9,
                   "nesterov": False, "optimizer": "SGD", "loss": "CrossEntropyLoss",
                   "hidden_channels": args.hidden_channels, "hidden_dim": 256,
                   "dropout": 0.5, "graph": "ntu-rgb+d/spatial",
                   "edge_importance_weighting": True, "augmentation": "none",
                   "num_worker": 0, "device": [args.device]},
        "config_echo": {"feeder": FEEDER_DIR, "n_train_full": int(len(tr_lab_c)),
                        "n_train_10pct": int(mask10.sum()), "n_val": int(len(va_lab_c)),
                        "classes": N_CLASS},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = RUN_DIR / ("summary_smoke.json" if args.smoke else "summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {out}")
    print(json.dumps({"10pct_mean": round(mean, 4), "10pct_std": round(std, 4),
                      "seeds": list(seeds)}, indent=1))


if __name__ == "__main__":
    main()
