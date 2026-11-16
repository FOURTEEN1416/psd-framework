# -*- coding: utf-8 -*-
"""P1 头对头 — 监督 ST-GCN 基线：NTU60 xsub，10% 与全量标注预算.

目的: 冷审 L13「零发表方法对比」的第二个基线——监督 ST-GCN（经典监督参考）
在与论文 E9 完全相同的协议下训练评测（同数据 data/ntu60_frame50/xsub、
同 40,091/16,487 划分、同 10% 分层标注预算 seed42），top-1 与论文数字同表对比。

复用决策（按任务书优先级）:
  - pyskl 未安装（2026-11-16 实测 ModuleNotFoundError）；
  - 仓内/官方代码 ST-GCN 可复用 → external/AimCLR/net/st_gcn.py 的
    net.st_gcn.Model（即 AimCLR 管线所用同一骨干类）端到端监督训练，
    feeder 用官方 Feeder_single（与线性评测同一数据管线、无增强口径）。
配方: SGD lr0.1 step[50,70] wd1e-4 momentum0.9(nesterov off)、80ep、batch128、
CE loss、eval every 5 报 best top-1（ST-GCN 风格官方配方 + 与 AimCLR 基线相同
harness 口径）。运行口径 hidden_channels=16 = 与 AimCLR SSL 骨干同容量的受控
对比（同一 net.st_gcn.Model 类）；标准 64ch 宽度在本机 8GB GPU 上 batch128/64
均实测 CUDA OOM/CUBLAS 失败（2026-11-16），如需 64ch 变体以 --hidden-channels 64
另行发射并如实记录显存口径。
10% 预算 = E9 逐行同式分层抽选（default_rng(42)，每类 k=max(1,round(10%·n_c))），
数据文件与 published_baseline_ntu60.py 共享（runs/p1_sprint/ntu60_budget_data/）。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/supervised_stgcn_ntu60.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/supervised_stgcn_ntu60.py
产出:
    runs/p1_sprint/supervised_stgcn_ntu60/summary.json（全量；冒烟为 summary_smoke.json）
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
XSUB = REPO_ROOT / "data" / "ntu60_frame50" / "xsub"
RUN_DIR = REPO_ROOT / "runs" / "p1_sprint" / "supervised_stgcn_ntu60"
BUDGET_DIR = REPO_ROOT / "runs" / "p1_sprint" / "ntu60_budget_data"
N_CLASS = 60
BUDGET_FRAC = 0.10
BUDGET_SEED = 42

PAPER_ANCHORS = {
    "e9_linear_head_10pct": 0.6605,
    "e9_psd_semantic_10pct": 0.6753,
    "e9_linear_head_full_ref": 0.7445,
    "e9_linear_retention_10pct": 0.887,
    "source": "reports/p14-ntu-lowres-2026-09-04.json (identical NTU60 xsub frame50 split)",
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from published_baseline_ntu60 import build_10pct_budget, build_smoke_data  # noqa: E402


def make_loader(data_rel: str, label_rel: str, batch: int, shuffle: bool, drop_last: bool):
    import torch  # noqa: F401  (feeder 模块内部依赖 torch)
    from feeder.ntu_feeder import Feeder_single  # noqa: E402  (external/AimCLR)

    ds = Feeder_single(data_path=data_rel, label_path=label_rel,
                       shear_amplitude=-1, temperal_padding_ratio=-1, mmap=True)
    return torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=shuffle,
                                       pin_memory=True, num_workers=0,
                                       drop_last=drop_last)


def adjust_lr(base_lr: float, epoch1: int, steps: list[int]) -> float:
    """官方 Processor.adjust_lr 同式: lr = base * 0.1^(已越过的 step 数)。"""
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

    work_dir = (REPO_ROOT / work_root) / tag  # 绝对化，避免 chdir 下 relative_to 失败
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
    ap.add_argument("--epochs", type=int, default=80, help="监督训练轮数（ST-GCN 风格 80）")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 70], help="lr 0.1x 衰减 epoch")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--hidden-channels", type=int, default=64,
                    help="64=标准 ST-GCN 宽度；16=与 SSL 骨干同容量受控变体")
    ap.add_argument("--eval-interval", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0, help="本仓锚定种子（同线性评测 init_seed(0)）")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not AIMCLR_ROOT.exists():
        raise SystemExit(f"[supervised-stgcn] external/AimCLR 缺失: {AIMCLR_ROOT}")
    for rel in ("train_position.npy", "train_label.pkl", "val_position.npy", "val_label.pkl"):
        if not (XSUB / rel).exists():
            raise SystemExit(f"[supervised-stgcn] NTU60 数据缺失: {XSUB / rel}")

    t0 = time.time()
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))

    if args.smoke:
        data = build_smoke_data()
        args.epochs, args.batch_size, args.eval_interval = 2, 64, 1
        budgets = {"full": {"train": data["full"], "n_train": 512},
                   "10pct": {"train": data["10pct"], "n_train": 512}}
        val = data["val"]
        work_root = "runs/p1_sprint/supervised_stgcn_ntu60/smoke"
    else:
        b10 = build_10pct_budget()
        budgets = {"full": {"train": {"data": "data/ntu60_frame50/xsub/train_position.npy",
                                      "label": "data/ntu60_frame50/xsub/train_label.pkl"},
                            "n_train": 40091},
                   "10pct": {"train": {"data": b10["data"], "label": b10["label"]},
                             "n_train": b10["n"]}}
        val = {"data": "data/ntu60_frame50/xsub/val_position.npy",
               "label": "data/ntu60_frame50/xsub/val_label.pkl"}
        work_root = "runs/p1_sprint/supervised_stgcn_ntu60"

    os.chdir(REPO_ROOT)  # 相对数据路径按 repo root 解析
    results = {}
    for tag in ("full", "10pct"):
        b = budgets[tag]
        results[tag] = train_budget(tag, b["train"], val, b["n_train"], args, work_root)

    smoke = args.smoke
    result = {
        "doneMarker": f"P1_SPRINT_SUPERVISED_STGCN_NTU60_{'SMOKE_OK' if smoke else 'DONE'}",
        "experiment": "p1_sprint_supervised_stgcn_ntu60",
        "mode": "smoke" if smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "method": "Supervised ST-GCN (end-to-end, cross-entropy) on NTU60 xsub",
        "implementation": "in-repo/official-code ST-GCN (external/AimCLR/net/st_gcn.py, "
                          "the backbone class of the AimCLR pipeline); pyskl not installed "
                          "(verified 2026-11-16) so reuse path = in-repo ST-GCN per task order",
        "faithful_reimplementation": False,
        "budgets": results,
        "budget_definition": {
            "frac": BUDGET_FRAC, "seed": BUDGET_SEED,
            "rule": "stratified per-class k=max(1, round(0.10*n_c)) over the 40,091 "
                    "train labels - identical formula/seed to E9 (paper 66.05/67.53 arms)",
        },
        "recipe": {"epochs": args.epochs, "batch": args.batch_size, "lr": args.lr,
                   "steps": args.steps, "weight_decay": args.wd, "momentum": 0.9,
                   "nesterov": False, "optimizer": "SGD", "loss": "CrossEntropyLoss",
                   "hidden_channels": args.hidden_channels, "hidden_dim": 256,
                   "dropout": 0.5, "graph": "ntu-rgb+d/spatial",
                   "edge_importance_weighting": True, "augmentation": "none "
                   "(same no-aug input pipeline as the official linear-eval protocol)",
                   "init_seed": args.seed, "num_worker": 0, "device": [args.device]},
        "paper_reference": PAPER_ANCHORS,
        "config_echo": {"data": "data/ntu60_frame50/xsub", "n_val": 16487},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    suffix = "_smoke" if smoke else ""
    out = RUN_DIR / f"summary{suffix}.json"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
