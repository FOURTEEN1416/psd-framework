"""P1 冲刺实验 1 — 冻结骨干 vs 全量微调对照（回应「收益是否只来自预训练」）.

问题（PR-D-26-13140 审稿人质疑）: PSD 框架在下游识别上的增益是否只是
「AimCLR 预训练表征本身好」，与 PSD 的语义对齐/伪标签机制无关？
本脚本提供受控对照:
  臂 A(frozen):   加载 P0.1 AimCLR 预训练 encoder_q，完全冻结，仅训练新分类头
                  （Linear(hidden_dim→num_classes)，官方 linear-eval 口径的本仓适配）；
  臂 B(fullft):   同一 encoder 架构 + 同一 ckpt 初始化 + 同一新头，整体解冻全量微调；
  臂 C(scratch):  同架构随机初始化 + 新头，全量微调（= 预训练消融的 -SSL 臂）。
三臂同数据（合成 K9 22 类，psd/data/synth_stgcn 单一真源）、同预算（epochs/batch/lr
逐项一致）、同 seed。若 frozen ≈ fullft，说明 PSD 增益主要来自预训练表征而非框架
机制——该结论直接回应审稿人质疑并写进 limitation。

复用声明（不重复造轮子）:
  - encoder 构建与 ckpt 加载: psd.models.aimclr_finetune（W31 消融两臂同款）；
  - 训练循环契约: psd.training.train_stgcn_bc.STGCNBCTrainer（消费
    forward→(cls_logits, boundary_logits) + compute_loss 的任意模型）；
  - 数据: psd.data.synth_stgcn.make_synthetic_dataset。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/frozen_backbone_contrast.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/frozen_backbone_contrast.py \
        --epochs 100 --samples-per-class 50 --seed 42
产出:
    runs/p1_sprint/frozen_backbone_contrast/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from torch.nn.parameter import Parameter  # noqa: E402

from psd.data.synth_stgcn import make_synthetic_dataset  # noqa: E402
from psd.models.aimclr_finetune import AimCLRFinetune, load_pretrained_encoder  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402

PRETRAINED_CKPT = REPO_ROOT / "runs" / "ntu_phaseB" / "joint_pretext" / "epoch120_model.pt"
OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "frozen_backbone_contrast"


class FrozenHeadOnly(torch.nn.Module):
    """AimCLRFinetune 的冻结包装: encoder 全部参数 requires_grad=False。

    保留 forward/compute_loss 契约（STGCNBCTrainer 消费）。
    """

    def __init__(self, inner: AimCLRFinetune):
        super().__init__()
        self.inner = inner
        for p in self.inner.encoder.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.inner(x)

    def compute_loss(self, cls_logits, boundary_logits, cls_labels, boundary_labels=None):
        return self.inner.compute_loss(cls_logits, boundary_logits, cls_labels, boundary_labels)

    @property
    def num_classes(self):
        return self.inner.num_classes


def count_trainable(model: torch.nn.Module) -> tuple[int, int]:
    trainables = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainables, total


def build_arm(arm: str, seed: int, num_classes: int) -> torch.nn.Module:
    """受控构建: 同 seed 同构建顺序，唯一差异变量 = encoder 初始化与可训练性。"""
    torch.manual_seed(seed)
    model = AimCLRFinetune(num_classes=num_classes)
    if arm in ("frozen", "fullft"):
        n_loaded = load_pretrained_encoder(model, PRETRAINED_CKPT)
        assert n_loaded > 0, "预训练权重加载失败"
    if arm == "frozen":
        return FrozenHeadOnly(model)
    return model


def evaluate_top1(model: torch.nn.Module, samples: list[dict], device: str,
                  batch_size: int = 64) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            x = torch.stack([torch.as_tensor(s["keypoints"], dtype=torch.float32)
                             for s in chunk]).to(device)
            cls_logits, _ = model(x)
            pred = cls_logits.argmax(dim=-1).cpu()
            y = torch.tensor([int(s["label"]) for s in chunk])
            correct += int((pred == y).sum())
            total += len(chunk)
    return correct / max(total, 1)


def run_arm(arm: str, seed: int, cfg: TrainConfig, train_samples: list[dict],
            val_samples: list[dict]) -> dict:
    t0 = time.time()
    model = build_arm(arm, seed, num_classes=22)
    n_trainable, n_total = count_trainable(model)
    print(f"[{arm}] trainable={n_trainable:,} / total={n_total:,} params")

    trainer = STGCNBCTrainer(model, train_samples, val_samples, cfg)
    summary = trainer.fit()

    # 全量微调后重新计数（frozen 应不变；fullft/scratch 应=total）
    n_trainable_after, _ = count_trainable(trainer.model)
    val_acc = evaluate_top1(trainer.model, val_samples, str(trainer.device))
    return {
        "arm": arm,
        "seed": seed,
        "best_val_acc": round(float(summary["best_val_acc"]), 4),
        "best_epoch": int(summary["best_epoch"]),
        "final_val_acc_recomputed": round(val_acc, 4),
        "epochs_run": int(summary["total_epochs_trained"]),
        "trainable_params": int(n_trainable),
        "total_params": int(n_total),
        "trainable_params_after_fit": int(n_trainable_after),
        "frozen_verified": (n_trainable_after == n_trainable) if arm == "frozen" else None,
        "duration_sec": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="最小步数管线验证（不产出结论数字）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--samples-per-class", type=int, default=50)
    ap.add_argument("--val-samples-per-class", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arms", default="frozen,fullft,scratch",
                    help="逗号分隔：frozen,fullft,scratch 的任意子集")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        assert a in ("frozen", "fullft", "scratch"), f"未知臂: {a}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        epochs, spc, vspc = 2, 4, 2
        print("[smoke] 冒烟模式: epochs=2, samples_per_class=4（仅验证管线通，数字无意义）")
    else:
        epochs, spc, vspc = args.epochs, args.samples_per_class, args.val_samples_per_class

    train_samples = make_synthetic_dataset(samples_per_class=spc, seed=args.seed)
    val_samples = make_synthetic_dataset(samples_per_class=vspc, seed=args.seed + 1000)
    print(f"[data] train={len(train_samples)} val={len(val_samples)} device={device}")

    cfg = TrainConfig(
        epochs=epochs, batch_size=args.batch_size, lr=args.lr,
        val_interval=1, save_interval=max(epochs, 1), early_stopping=False,
        use_amp=(device == "cuda"), device=device,
        output_dir=str(OUT_DIR / "trainer_artifacts"),
        seed=args.seed,
    )

    rows = []
    for arm in arms:
        row = run_arm(arm, args.seed, cfg, train_samples, val_samples)
        rows.append(row)
        print(f"[{arm}] done: best_val_acc={row['best_val_acc']} "
              f"({row['duration_sec']}s)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "frozen_backbone_contrast",
        "mode": "smoke" if args.smoke else "full",
        "pretrained_ckpt": str(PRETRAINED_CKPT.relative_to(REPO_ROOT)),
        "arms": rows,
        "config": {"epochs": epochs, "samples_per_class": spc,
                   "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed},
    }
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
