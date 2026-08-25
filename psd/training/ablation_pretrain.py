"""tab3「−自监督预训练」消融编排 — W31（NEXT-BATCH-plan.md W31 节）.

职责:
    - make_w12_split: 逐行复刻 W12 口径切分（scripts/run_p05_full.py L117-129），
      保证消融两臂与 W12 主实验同数据同切分；
    - train_one_arm: 单臂训练（复用 STGCNBCTrainer + psd.models.aimclr_finetune），
      受控种子保证同 seed 两臂头初始化逐位相等、DataLoader 洗牌序列一致；
    - run_ablation: arms × seeds 编排 + 聚合。

公平性协议（唯一差异变量 = encoder 初始权重）:
    同 seed 下两臂共享同一份 train/val 样本列表对象、相同训练超参、
    相同洗牌序列；分类/边界头逐位相等（psd/models 测试断言）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psd.data.stgcn_bc_dataset import make_synthetic_dataset  # noqa: E402
from psd.models.aimclr_finetune import build_arm_model  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402


def make_w12_split(
    samples_per_class: int = 100,
    T: int = 30,
    data_seed: int = 42,
    val_split: float = 0.2,
) -> Tuple[List[Dict], List[Dict]]:
    """W12 口径合成数据生成 + 确定性切分（与 run_p05_full.py 完全一致）.

    口径: samples_per_class × 22 类, noise_std 默认 0.05,
    np.random.default_rng(data_seed).permutation 后前 val_n 为验证集。
    """
    samples = make_synthetic_dataset(
        samples_per_class=samples_per_class, T=T, seed=data_seed
    )
    total = len(samples)
    rng = np.random.default_rng(data_seed)
    indices = rng.permutation(total)
    val_n = int(total * val_split)
    train_samples = [samples[i] for i in indices[val_n:]]
    val_samples = [samples[i] for i in indices[:val_n]]
    return train_samples, val_samples


def _build_train_config(cfg: Dict, output_dir: str) -> TrainConfig:
    """从扁平化 cfg 构造 TrainConfig（只取已知字段）。"""
    fields = {
        "lr": cfg.get("lr", 1e-3),
        "weight_decay": cfg.get("weight_decay", 1e-4),
        "epochs": cfg.get("epochs", 50),
        "batch_size": cfg.get("batch_size", 32),
        "warmup_epochs": cfg.get("warmup_epochs", 5),
        "device": cfg.get("device", "auto"),
        "early_stopping": cfg.get("early_stopping", False),
        "patience": cfg.get("patience", 15),
        "output_dir": output_dir,
    }
    return TrainConfig(**fields)


def train_one_arm(
    arm: str,
    seed: int,
    cfg: Dict,
    train_samples: List[Dict],
    val_samples: List[Dict],
    output_dir: str,
    pretrained_ckpt: Optional[str] = None,
) -> Dict:
    """单臂训练（scratch|warm），返回合规摘要.

    公平性关键: torch.manual_seed(seed) 在 trainer 构建前置位——
    DataLoader 洗牌与模型构建消费同一 RNG 流，同 seed 两臂批序一致。
    """
    if arm not in ("scratch", "warm"):
        raise ValueError(f"未知消融臂: {arm!r}（可选 scratch|warm）")

    tc = _build_train_config(cfg, output_dir)
    torch.manual_seed(seed)

    model = build_arm_model(
        arm,
        seed,
        pretrained_ckpt=pretrained_ckpt if arm == "warm" else None,
        num_classes=int(cfg.get("num_classes", 22)),
        boundary_weight=float(cfg.get("boundary_weight", 0.3)),
    )
    trainer = STGCNBCTrainer(model, train_samples, val_samples, config=tc)
    summary = trainer.fit()

    best_ckpt = Path(tc.output_dir) / "best.pt"
    return {
        "arm": arm,
        "seed": seed,
        "best_val_acc": float(summary["best_val_acc"]),
        "best_epoch": int(summary["best_epoch"]),
        "epochs_trained": int(summary["total_epochs_trained"]),
        "final_train_acc": float(summary["final_train_acc"]),
        "device": str(summary["device"]),
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        "output_dir": str(tc.output_dir),
        "ckpt_path": str(best_ckpt) if best_ckpt.exists() else "N/A",
    }


def resolve_ckpt_path(raw: str) -> Path:
    """相对路径按仓库根解析（配置内路径统一仓库根锚定）。"""
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def run_ablation(cfg: Dict) -> Dict:
    """arms × seeds 全量编排并聚合.

    Args:
        cfg: configs/ablation_pretrain.yaml 反序列化后的字典
    Returns:
        {"runs": [...], "aggregate": {...}, "data": {...}}
    """
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    abl_cfg = cfg["ablation"]

    arms = list(abl_cfg["arms"])
    seeds = [int(s) for s in abl_cfg["seeds"]]
    ckpt_raw = cfg["pretrained"]["ckpt"]
    flat_cfg = {
        "lr": train_cfg.get("lr", 1e-3),
        "weight_decay": train_cfg.get("weight_decay", 1e-4),
        "epochs": train_cfg.get("epochs", 50),
        "batch_size": train_cfg.get("batch_size", 32),
        "warmup_epochs": train_cfg.get("warmup_epochs", 5),
        "device": train_cfg.get("device", "auto"),
        "early_stopping": train_cfg.get("early_stopping", False),
        "patience": train_cfg.get("patience", 15),
        "num_classes": cfg.get("model", {}).get("num_classes", 22),
        "boundary_weight": cfg.get("model", {}).get("boundary_weight", 0.3),
    }

    train_samples, val_samples = make_w12_split(
        samples_per_class=int(data_cfg.get("samples_per_class", 100)),
        T=int(data_cfg.get("T", 30)),
        data_seed=int(data_cfg.get("seed", 42)),
        val_split=float(data_cfg.get("val_split", 0.2)),
    )

    runs: List[Dict] = []
    for arm in arms:
        for seed in seeds:
            out_dir = str(
                REPO_ROOT / train_cfg["output_dir"] / f"{arm}_seed{seed}"
            )
            print(f"[w31] 训练臂={arm} seed={seed} → {out_dir}")
            s = train_one_arm(
                arm=arm,
                seed=seed,
                cfg=flat_cfg,
                train_samples=train_samples,
                val_samples=val_samples,
                output_dir=out_dir,
                pretrained_ckpt=str(resolve_ckpt_path(ckpt_raw)),
            )
            print(f"[w31]   best_val_acc={s['best_val_acc']:.4f} "
                  f"@ epoch {s['best_epoch']} ({s['epochs_trained']}ep)")
            runs.append(s)

    aggregate: Dict[str, Dict] = {}
    for arm in arms:
        accs = [r["best_val_acc"] for r in runs if r["arm"] == arm]
        arr = np.asarray(accs, dtype=np.float64)
        aggregate[arm] = {
            "best_val_acc_mean": round(float(arr.mean()), 6),
            "best_val_acc_std": round(float(arr.std(ddof=0)), 6),
            "n_seeds": len(accs),
            "per_seed": accs,
        }

    return {
        "runs": runs,
        "aggregate": aggregate,
        "data": {
            "samples_per_class": int(data_cfg.get("samples_per_class", 100)),
            "T": int(data_cfg.get("T", 30)),
            "data_seed": int(data_cfg.get("seed", 42)),
            "val_split": float(data_cfg.get("val_split", 0.2)),
            "train_size": len(train_samples),
            "val_size": len(val_samples),
            "caliber": "synthetic",
        },
    }


__all__ = [
    "make_w12_split",
    "train_one_arm",
    "run_ablation",
    "resolve_ckpt_path",
]
