# -*- coding: utf-8 -*-
"""P1 头对头 — 已发表方法基线：AimCLR (AAAI 2022) 线性评测，NTU60 xsub，10% 与全量预算.

目的: 拆除冷审 L13「全文零发表方法对比」——把已发表方法放到与论文 E9 完全
相同的协议下（同数据 data/ntu60_frame50/xsub、同 40,091/16,487 划分、同 10%
分层标注预算 seed42），产出可与论文数字同表对比的 top-1：
  论文侧参照（reports/p14-ntu-lowres-2026-09-04）: 线性头 10%=66.05% /
  全量=74.45% / retention 88.7%；PSD 语义管线 10%=67.53%（retention 90.7%）。

实现（复用优先级按任务书）:
  - official AimCLR 代码存在于 external/AimCLR → 直接复用其 LE_Processor
    （linear eval 官方协议: 仅训 fc 头，SGD lr3.0 step[80]，100ep，batch128，
    eval every 5，报 best top-1）， faithful_reimplementation=false。
  - encoder 权重: 官方 released checkpoint（GDrive 14rayAgGWAHFL-JdeCJQOwP4cJ-fHglJS）
    本机不可达（2026-09-07 FAIL-TO-RUN reports/r23-extbase-anchor-2026-09-07.json；
    2026-11-16 复测仍超时）。故默认权重 = 仓内按官方配方训练的 pretext
    （runs/ntu_phaseB/joint_pretext/epoch300_model.pt，300ep，Feeder_triple
    shear0.5/tpr6，SGD lr0.1 step[250]，queue 32768，mining@150——与官方
    pretext yaml 逐超参一致，仅路径/设备/num_worker 适配 + 已披露的坍缩修复
    初始化跳过，见 psd/training/p01_processor.py 与 reports/p01-aimclr-2026-08-23.md）。
    保真锚: 该 encoder 全量线性评测 74.45% vs 官方 README joint 74.34%（Δ0.15pp）。
  - 10% 预算 = E9 逐行同式分层抽选（default_rng(42)，每类 k=max(1,round(10%·n_c))），
    与论文 66.05/67.53 臂同一子集定义。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/published_baseline_ntu60.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/published_baseline_ntu60.py
产出:
    runs/p1_sprint/published_baseline_ntu60/summary.json（全量；冒烟为 summary_smoke.json）
    runs/p1_sprint/ntu60_budget_data/（两脚本共享的 10% 预算数据文件）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AIMCLR_ROOT = REPO_ROOT / "external" / "AimCLR"
XSUB = REPO_ROOT / "data" / "ntu60_frame50" / "xsub"
RUN_DIR = REPO_ROOT / "runs" / "p1_sprint" / "published_baseline_ntu60"
BUDGET_DIR = REPO_ROOT / "runs" / "p1_sprint" / "ntu60_budget_data"
DEFAULT_WEIGHTS = "runs/ntu_phaseB/joint_pretext/epoch300_model.pt"

BUDGET_FRAC = 0.10
BUDGET_SEED = 42
N_CLASS = 60

# 论文侧同表参照（来自已归档实验产物，非本次运行产出）
PAPER_ANCHORS = {
    "e9_linear_head_10pct": 0.6605,
    "e9_psd_semantic_10pct": 0.6753,
    "e9_linear_head_full_ref": 0.7445,
    "e9_linear_retention_10pct": 0.887,
    "source": "reports/p14-ntu-lowres-2026-09-04.json (identical NTU60 xsub frame50 split)",
}


def stratified_10pct(labels: np.ndarray, seed: int) -> np.ndarray:
    """与 run_p14_ntu_lowres.py 逐行一致（E9 协议冻结式）。返回 bool mask。"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in range(N_CLASS):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    return mask


def _idx_fingerprint(idx: np.ndarray) -> str:
    return hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()[:16]


def build_10pct_budget() -> dict:
    """生成（或复用）E9 同定义的 10% 预算数据对（position.npy + label.pkl）。

    两个头对头脚本共享 BUDGET_DIR；meta.json 记录抽选指纹防止口径漂移。
    """
    with open(XSUB / "train_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    labels_arr = np.asarray(labels, dtype=np.int64)
    mask = stratified_10pct(labels_arr, BUDGET_SEED)
    idx = np.where(mask)[0]

    meta_path = BUDGET_DIR / "meta.json"
    pos_out = BUDGET_DIR / "train_10pct_position.npy"
    lab_out = BUDGET_DIR / "train_10pct_label.pkl"
    meta = {
        "seed": BUDGET_SEED,
        "budget_frac": BUDGET_FRAC,
        "n_train": int(len(labels_arr)),
        "n_subset": int(len(idx)),
        "index_sha256_16": _idx_fingerprint(idx),
        "source_label": "data/ntu60_frame50/xsub/train_label.pkl",
        "formula": "per-class k=max(1, round(0.10*n_c)), default_rng(42), E9-identical",
    }
    if pos_out.exists() and lab_out.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("index_sha256_16") == meta["index_sha256_16"]:
            print(f"[budget] reuse 10pct files: n={meta['n_subset']} fp={meta['index_sha256_16']}")
            return {"data": "runs/p1_sprint/ntu60_budget_data/train_10pct_position.npy",
                    "label": "runs/p1_sprint/ntu60_budget_data/train_10pct_label.pkl",
                    "n": meta["n_subset"]}

    BUDGET_DIR.mkdir(parents=True, exist_ok=True)
    full_pos = np.load(XSUB / "train_position.npy", mmap_mode="r")
    np.save(pos_out, np.asarray(full_pos[idx]))
    with open(lab_out, "wb") as f:
        pickle.dump(([names[i] for i in idx], [int(labels_arr[i]) for i in idx]), f)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[budget] built 10pct files: n={meta['n_subset']}/{meta['n_train']} fp={meta['index_sha256_16']}")
    return {"data": "runs/p1_sprint/ntu60_budget_data/train_10pct_position.npy",
            "label": "runs/p1_sprint/ntu60_budget_data/train_10pct_label.pkl",
            "n": meta["n_subset"]}


def build_smoke_data() -> dict:
    """冒烟切片（数字无意义，仅验管线）: 全量前512 / 10%子集前512 / val 前1024。"""
    sm = BUDGET_DIR / "smoke_data"
    sm.mkdir(parents=True, exist_ok=True)
    out = {}
    with open(XSUB / "train_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    n_tr = 512
    np.save(sm / "train_full_position.npy", np.asarray(np.load(XSUB / "train_position.npy", mmap_mode="r")[:n_tr]))
    with open(sm / "train_full_label.pkl", "wb") as f:
        pickle.dump((list(names[:n_tr]), [int(v) for v in labels[:n_tr]]), f)
    out["full"] = {"data": "runs/p1_sprint/ntu60_budget_data/smoke_data/train_full_position.npy",
                   "label": "runs/p1_sprint/ntu60_budget_data/smoke_data/train_full_label.pkl"}

    b10 = build_10pct_budget()
    with open(REPO_ROOT / b10["label"], "rb") as f:
        names10, labels10 = pickle.load(f)
    n10 = min(512, len(names10))
    np.save(sm / "train_10pct_position.npy",
            np.asarray(np.load(REPO_ROOT / b10["data"], mmap_mode="r")[:n10]))
    with open(sm / "train_10pct_label.pkl", "wb") as f:
        pickle.dump((list(names10[:n10]), [int(v) for v in labels10[:n10]]), f)
    out["10pct"] = {"data": "runs/p1_sprint/ntu60_budget_data/smoke_data/train_10pct_position.npy",
                    "label": "runs/p1_sprint/ntu60_budget_data/smoke_data/train_10pct_label.pkl"}

    with open(XSUB / "val_label.pkl", "rb") as f:
        vnames, vlabels = pickle.load(f)
    n_va = 1024
    np.save(sm / "val_position.npy", np.asarray(np.load(XSUB / "val_position.npy", mmap_mode="r")[:n_va]))
    with open(sm / "val_label.pkl", "wb") as f:
        pickle.dump((list(vnames[:n_va]), [int(v) for v in vlabels[:n_va]]), f)
    out["val"] = {"data": "runs/p1_sprint/ntu60_budget_data/smoke_data/val_position.npy",
                  "label": "runs/p1_sprint/ntu60_budget_data/smoke_data/val_label.pkl"}
    return out


def linear_cfg(work_dir: str, weights: str, train: dict, val: dict,
               epochs: int, batch: int, eval_interval: int, device: int) -> dict:
    """官方 linear_eval yaml 逐键镜像（configs/ntu60_phaseb_lineareval_xsub_joint.yaml 同源），
    仅 work_dir/weights/数据路径/epochs/eval_interval/batch 按本次运行声明。"""
    return {
        "work_dir": work_dir,
        "weights": weights,
        "ignore_weights": ["encoder_q.fc", "encoder_k", "queue"],
        "train_feeder": "feeder.ntu_feeder.Feeder_single",
        "train_feeder_args": {"data_path": train["data"], "label_path": train["label"],
                              "shear_amplitude": -1, "temperal_padding_ratio": -1, "mmap": True},
        "test_feeder": "feeder.ntu_feeder.Feeder_single",
        "test_feeder_args": {"data_path": val["data"], "label_path": val["label"],
                             "shear_amplitude": -1, "temperal_padding_ratio": -1, "mmap": True},
        "model": "net.aimclr.AimCLR",
        "model_args": {"base_encoder": "net.st_gcn.Model", "pretrain": False,
                       "in_channels": 3, "hidden_channels": 16, "hidden_dim": 256,
                       "num_class": N_CLASS, "dropout": 0.5,
                       "graph_args": {"layout": "ntu-rgb+d", "strategy": "spatial"},
                       "edge_importance_weighting": True},
        "nesterov": False, "weight_decay": 0.0, "base_lr": 3.0,
        "optimizer": "SGD", "step": [80],
        "device": [device], "batch_size": batch, "test_batch_size": batch,
        "num_epoch": epochs, "stream": "joint", "num_worker": 0,
        "save_interval": -1, "eval_interval": eval_interval, "print_log": True,
    }


def run_linear_eval(tag: str, cfg: dict) -> float:
    """官方 LE_Processor 直连（同 run_ntu_lineareval.py 链路），返回 best top-1 (0-1)。"""
    work_dir = REPO_ROOT / cfg["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    from processor.processor import init_seed  # noqa: E402  (external/AimCLR)
    init_seed(0)  # 本仓锚定种子（官方未全局定种，已声明的本地差异，同 run_ntu_lineareval.py）
    from processor.linear_evaluation import LE_Processor  # noqa: E402

    os.chdir(REPO_ROOT)  # 相对路径（weights/data/label）统一按 repo root 解析
    proc = LE_Processor(["--config", str(cfg_path)])
    proc.start()
    best = float(getattr(proc, "best_result", 0.0))
    print(f"[le:{tag}] best top1 = {best:.2f}%")
    return best / 100.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="缩样管线验证（数字无意义）")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS,
                    help="AimCLR pretext checkpoint（默认=仓内官方配方 pretext；"
                         "若官方 released checkpoint 日后可得可覆盖）")
    ap.add_argument("--epochs", type=int, default=100, help="官方线性评测 100ep")
    ap.add_argument("--batch-size", type=int, default=128, help="官方 batch 128")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    if not AIMCLR_ROOT.exists():
        raise SystemExit(f"[published-baseline] external/AimCLR 缺失: {AIMCLR_ROOT}")
    for rel in ("train_position.npy", "train_label.pkl", "val_position.npy", "val_label.pkl"):
        if not (XSUB / rel).exists():
            raise SystemExit(f"[published-baseline] NTU60 数据缺失: {XSUB / rel}")
    if not (REPO_ROOT / args.weights).exists():
        raise SystemExit(f"[published-baseline] weights 不存在: {args.weights}")

    t0 = time.time()
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))

    if args.smoke:
        data = build_smoke_data()
        epochs, batch, eval_interval = 2, 64, 1
        budgets = {
            "full": {"train": data["full"], "n_train": 512},
            "10pct": {"train": data["10pct"], "n_train": 512},
        }
        val = data["val"]
        work_root = "runs/p1_sprint/published_baseline_ntu60/smoke"
    else:
        b10 = build_10pct_budget()
        epochs, batch, eval_interval = args.epochs, args.batch_size, 5
        budgets = {
            "full": {"train": {"data": "data/ntu60_frame50/xsub/train_position.npy",
                               "label": "data/ntu60_frame50/xsub/train_label.pkl"},
                     "n_train": 40091},
            "10pct": {"train": {"data": b10["data"], "label": b10["label"]}, "n_train": b10["n"]},
        }
        val = {"data": "data/ntu60_frame50/xsub/val_position.npy",
               "label": "data/ntu60_frame50/xsub/val_label.pkl"}
        work_root = "runs/p1_sprint/published_baseline_ntu60"

    results = {}
    for tag in ("full", "10pct"):
        b = budgets[tag]
        cfg = linear_cfg(work_dir=f"{work_root}/{tag}", weights=args.weights,
                         train=b["train"], val=val, epochs=epochs, batch=batch,
                         eval_interval=eval_interval, device=args.device)
        top1 = run_linear_eval(tag, cfg)
        results[tag] = {"top1": round(top1, 4), "n_train": b["n_train"],
                        "work_dir": f"{work_root}/{tag}"}

    smoke = args.smoke
    result = {
        "doneMarker": f"P1_SPRINT_PUBLISHED_AIMCLR_NTU60_{'SMOKE_OK' if smoke else 'DONE'}",
        "experiment": "p1_sprint_published_baseline_ntu60",
        "mode": "smoke" if smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "method": "AimCLR (AAAI 2022) linear evaluation on NTU60 xsub (joint stream)",
        "implementation": "official AimCLR code (external/AimCLR LE_Processor); "
                          "linear protocol = official (fc-only, SGD lr3.0 step[80], "
                          "100ep, batch128, eval@5, best top-1)",
        "faithful_reimplementation": False,
        "weights": args.weights,
        "weights_provenance": {
            "default": "in-house pretext trained with the OFFICIAL AimCLR recipe on the "
                       "identical NTU60 xsub train split (40,091 clips, 300ep, Feeder_triple "
                       "shear0.5/tpr6, SGD lr0.1 step[250], queue32768, mining@150)",
            "deviation_from_official": "weights_init skip (documented collapse fix, "
                                       "psd/training/p01_processor.py, reports/p01-aimclr-2026-08-23.md); "
                                       "paths/device/num_worker adaptations",
            "official_released_checkpoint": "UNREACHABLE (GDrive FAIL-TO-RUN 2026-09-07, "
                                            "reports/r23-extbase-anchor-2026-09-07.json; re-verified 2026-11-16)",
            "fidelity_anchor": "full-budget linear 74.45% (E9 (c)) vs official README joint 74.34% (d0.15pp)",
        },
        "budgets": results,
        "budget_definition": {
            "frac": BUDGET_FRAC, "seed": BUDGET_SEED,
            "rule": "stratified per-class k=max(1, round(0.10*n_c)) over the 40,091 "
                    "train labels - identical formula/seed to E9 (paper 66.05/67.53 arms)",
        },
        "paper_reference": PAPER_ANCHORS,
        "config_echo": {"epochs": epochs, "batch": batch, "device": [args.device],
                        "num_worker": 0, "init_seed": 0, "data": "data/ntu60_frame50/xsub",
                        "n_val": 16487},
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
