"""C1 解耦成本实验脚本。

用法:
    python scripts/run_c1_decouple.py --n-per-class 100 --checkpoint runs/p05_stgcn_bc_full/best.pt
    
功能：
    1. 加载 Y(22类) 预训练 checkpoint
    2. 冻结 backbone，仅重训 head（语义层）
    3. 在 Y'(21类) 标签下训练，使用 subsampled 每类 n 样本
    4. 记录 best_val_acc, total_time_sec, epochs_run 等指标
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

# repo imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from psd.data.synth_stgcn import make_synthetic_dataset
from psd.models.stgcn_bc import STGCNBC, build_stgcn_bc
from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig
from psd.training.stgcn_loss import STGCNBCLoss

# Y/Y' label mapping (from run_p05_full.py)
Y_LABEL_NAMES = list(ALL_BEHAVIORS_22)
Y_NUM_CLASSES = 22

Y_PRIME_LABEL_NAMES = [
    "sit", "down", "locomotion",  # 0,1,2  (locomotion = stand+track)
    "heel", "sit_up", "stay", "bark", "bite",
    "alert_sit", "alert_down", "apprehend", "escort", "obstacle",
    "recall", "watch", "guard", "release", "retrieve",
    "jump", "scale", "search_blind",
]
Y_PRIME_NUM_CLASSES = 21

_Y_TO_YP_MAP: dict = {}
for name in Y_LABEL_NAMES:
    if name in ("stand", "track"):
        _Y_TO_YP_MAP[name] = 2  # -> locomotion
    else:
        _Y_TO_YP_MAP[name] = Y_PRIME_LABEL_NAMES.index(name)


def _map_samples_to_yprime(samples):
    """将 Y(22类) 样本映射为 Y'(21类) 标签."""
    mapped = []
    for s in samples:
        new_s = dict(s)
        orig_name = s.get("label_name", "")
        if orig_name in _Y_TO_YP_MAP:
            new_s["label"] = _Y_TO_YP_MAP[orig_name]
            new_s["label_name"] = Y_PRIME_LABEL_NAMES[new_s["label"]]
        mapped.append(new_s)
    return mapped


def main() -> None:
    ap = argparse.ArgumentParser(description="C1 解耦成本实验")
    ap.add_argument("--n-per-class", type=int, required=True,
                    help="每类样本数（如 50 或 100）")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Y checkpoint best.pt 路径（默认: runs/p05_stgcn_bc_full/best.pt）")
    ap.add_argument("--epochs", type=int, default=50,
                    help="训练轮数（默认: 50，匹配 W12 等预算}")
    ap.add_argument("--patience", type=int, default=15,
                    help="早停耐心值（默认: 15）")
    ap.add_argument("--output-json", type=str, default=None,
                    help="输出结果 JSON 路径（默认: 控制台打印）")
    args = ap.parse_args()

    checkpoint_path = args.checkpoint or str(
        Path(__file__).resolve().parents[1] / "runs" / "p05_stgcn_bc_full" / "best.pt"
    )
    n_per_class = args.n_per_class
    total_epochs = args.epochs
    patience = args.patience

    print(f"[c1] ===== C1 解耦实验 =====")
    print(f"[c1] n_per_class={n_per_class}, epochs={total_epochs}, patience={patience}")
    print(f"[c1] checkpoint={checkpoint_path}")

    # -------------------------------------------------
    # 1. 生成 Y' 标签合成数据
    # -------------------------------------------------
    print("[c1] 生成合成数据 (samples_per_class={})...".format(n_per_class))
    all_samples = make_synthetic_dataset(samples_per_class=n_per_class, T=30, seed=42)
    # map to Y'
    samples = _map_samples_to_yprime(all_samples)
    total = len(samples)
    rng = np.random.default_rng(42)
    indices = rng.permutation(total)
    val_n = int(total * 0.2)
    train_samples = [samples[i] for i in indices[val_n:]]
    val_samples = [samples[i] for i in indices[:val_n]]
    print(f"[data] train={len(train_samples)} val={len(val_samples)} total={total}")

    # -------------------------------------------------
    # 2. 构建模型：加载 Y checkpoint，冻结 backbone
    # -------------------------------------------------
    print("[c1] 构建 STGCNBC 模型 (num_classes=21 Y'体系)...")
    model = build_stgcn_bc(in_channels=3, num_classes=Y_PRIME_NUM_CLASSES)
    # load checkpoint: strip head keys so only backbone loads; head stays random-initialized
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        # Remove head parameters so that strict=False only loads backbone;
        # head params will remain randomly initialized.
        ckpt_sd = {k: v for k, v in ckpt["model_state_dict"].items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(ckpt_sd, strict=False)
        # with strict=False, shape-mismatch errors for head are suppressed;
        # remaining keys are backbone-only and should match shape.
        print(f"[c1] Checkpoint loaded: missing={len(missing)} keys, unexpected={len(unexpected)} keys (head keys stripped, strict=False)")
        # overwrite ckpt dict for trainer (best_val_acc etc)
        best_val_from_ckpt = ckpt.get("val_acc", 0.0)
        print(f"[c1] CKPT contained val_acc={best_val_from_ckpt:.4f}")
    else:
        print(f"[c1] WARNING: checkpoint not found at {checkpoint_path}; proceeding with random init.")

    # freeze backbone: all params whose name starts with 'backbone.'
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            param.requires_grad = False
    frozen_cnt = sum(1 for p in model.parameters() if not p.requires_grad)
    trainable_cnt = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"[c1] Frozen backbone params: {frozen_cnt:,} | Trainable params: {trainable_cnt:,}")
    # should be ~all backbone + head.conv_boundary; head.fc_cls is trainable

    # -------------------------------------------------
    # 3. 训练配置
    # -------------------------------------------------
    tc = TrainConfig(
        lr=0.001,
        weight_decay=0.0001,
        epochs=total_epochs,
        batch_size=32,
        num_workers=0,
        val_interval=1,
        save_interval=5,
        lr_scheduler="cosine",
        warmup_epochs=5,
        early_stopping=True,
        patience=patience,
        use_amp=True,  # will be False on CPU
        device="auto",
        grad_clip=1.0,
        output_dir=f"runs/c1_decouple_n{n_per_class}",
    )
    # ensure output dir exists
    Path(tc.output_dir).mkdir(parents=True, exist_ok=True)

    # build trainer
    trainer = STGCNBCTrainer(model, train_samples, val_samples, config=tc)

    # -------------------------------------------------
    # 4. 训练
    # -------------------------------------------------
    t0 = time.time()
    summary = trainer.fit()
    total_sec = time.time() - t0

    # -------------------------------------------------
    # 5. 结果整理与输出
    # -------------------------------------------------
    # complement summary with extra fields
    summary["ckpt_path"] = str(Path(tc.output_dir) / "best.pt") if Path(tc.output_dir / "best.pt").exists() else "N/A"
    summary["epochs_run"] = summary.pop("total_epochs_trained")
    summary["n_per_class"] = n_per_class
    summary["total_samples"] = len(train_samples) + len(val_samples)
    summary["train_size"] = len(train_samples)
    summary["val_size"] = len(val_samples)
    summary["total_time_sec"] = round(total_sec, 1)
    summary["trainable_params"] = trainable_cnt
    summary["frozen_params"] = frozen_cnt
    summary["device"] = str(summary["device"])

    # checks
    random_baseline = 1.0 / Y_PRIME_NUM_CLASSES  # 1/21 ≈ 0.0476
    convergence_threshold = random_baseline * 3
    summary["loss_monotonic_trend"] = all(
        h["train_loss"] for h in trainer.history
    )  # placeholder; real check via history
    # simple val_acc check
    summary["val_acc_significant"] = summary["best_val_acc"] >= convergence_threshold
    summary["no_nan"] = not np.isnan(summary["best_val_acc"])
    summary["all_pass"] = summary["val_acc_significant"] and summary["no_nan"]

    # print result
    taxonomy_label = "Y' 粗粒度(21类)"
    print("\n" + "=" * 64)
    print(f"P0.5 {taxonomy_label} 训练结果 (解耦)")
    print("=" * 64)
    print(f"  samples_per_class : {n_per_class}")
    print(f"  taxonomy          : Y' 粗粒度(21类)")
    print(f"  num_classes       : {Y_PRIME_NUM_CLASSES}")
    print(f"  total_samples     : {total}")
    print(f"  epochs_run        : {summary['epochs_run']}")
    print(f"  best_val_acc      : {summary['best_val_acc']:.4f}")
    print(f"  best_epoch        : {summary['best_epoch']}")
    print(f"  final_train_acc   : {summary['final_train_acc']:.4f}")
    print(f"  final_val_acc     : {summary['final_val_acc']:.4f}")
    print(f"  device            : {summary['device']}")
    print(f"  trainable_params  : {summary['trainable_cnt']:,}")
    print(f"  frozen_params     : {summary['frozen_params']:,}")
    print(f"  ckpt_path         : {summary['ckpt_path']}")
    print(f"  total_time_sec    : {summary['total_time_sec']}s")
    print(f"  val_acc_significant: {'✓' if summary['val_acc_significant'] else '✗'} (≥{convergence_threshold:.4f})")
    print(f"  no_nan: {'✓' if summary['no_nan'] else '✗'}")
    print(f"[overall] {'✓ 全部通过' if summary['all_pass'] else '✗ 有未通过项'}")
    print(f"[time] 总耗时 {summary['total_time_sec']:.1f}s")

    # write result JSON if requested
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[output] 结果 JSON -> {out_path}")


if __name__ == "__main__":
    main()