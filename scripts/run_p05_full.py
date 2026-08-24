"""P0.5 ST-GCN+BC 完整训练 + E6 双贴合实验（W12 窗口）.

用法:
    # Y 细粒度（22 类）主实验
    python scripts/run_p05_full.py --config configs/p05_stgcn_bc_full.yaml

    # Y′ 粗粒度（21 类）消融
    python scripts/run_p05_full.py --config configs/p05_e6_taxonomy.yaml

    # 消融：覆盖 samples_per_class
    python scripts/run_p05_full.py --config configs/p05_stgcn_bc_full.yaml --samples-per-class 20
    python scripts/run_p05_full.py --config configs/p05_stgcn_bc_full.yaml --samples-per-class 50
    python scripts/run_p05_full.py --config configs/p05_stgcn_bc_full.yaml --samples-per-class 100
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.stgcn_bc_dataset import (  # noqa: E402
    make_synthetic_dataset,
)
from psd.models.stgcn_bc_model import build_stgcn_bc  # noqa: E402
from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402


# ============================================================================
# E6 双贴合分类体系映射
# ============================================================================

# Y: 原始 22 类（index 0-21）
Y_LABEL_NAMES = list(ALL_BEHAVIORS_22)
Y_NUM_CLASSES = 22

# Y': stand(2)+track(8) → locomotion，共 21 类
Y_PRIME_LABEL_NAMES = [
    "sit", "down", "locomotion",  # 0,1,2  (locomotion = stand+track)
    "heel", "sit_up", "stay", "bark", "bite",
    "alert_sit", "alert_down", "apprehend", "escort", "obstacle",
    "recall", "watch", "guard", "release", "retrieve",
    "jump", "scale", "search_blind",
]
Y_PRIME_NUM_CLASSES = 21

# 原始 22 类 index → Y' 类 index 映射
# 0=sit,1=down,2=stand,3=heel,4=sit_up,5=stay,6=bark,7=bite,8=track,...
_Y_TO_YP_MAP: dict = {}
_y_idx = 0
for name in ALL_BEHAVIORS_22:
    if name in ("stand", "track"):
        _Y_TO_YP_MAP[name] = 2  # → locomotion
    else:
        _Y_TO_YP_MAP[name] = Y_PRIME_LABEL_NAMES.index(name)


def _map_samples_to_yprime(samples):
    """将 Y(22类) 样本映射为 Y'(21类) 标签."""
    mapped = []
    for s in samples:
        new_s = dict(s)
        orig_name = s["label_name"]
        if orig_name in _Y_TO_YP_MAP:
            new_s["label"] = _Y_TO_YP_MAP[orig_name]
            new_s["label_name"] = Y_PRIME_LABEL_NAMES[new_s["label"]]
        mapped.append(new_s)
    return mapped


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p05_stgcn_bc_full.yaml")
    ap.add_argument("--samples-per-class", type=int, default=None,
                    help="覆盖 config 的 samples_per_class（用于消融）")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]
    eval_cfg = cfg.get("evaluation", {})

    # 判断 Y/Y' 变体
    taxonomy_variant = data_cfg.get("taxonomy_variant", "Y")
    if taxonomy_variant == "Y_prime":
        num_classes = Y_PRIME_NUM_CLASSES
        label_names = Y_PRIME_LABEL_NAMES
        print("[e6] 使用 Y' 粗粒度分类体系（21 类：stand+track→locomotion）")
    else:
        num_classes = Y_LABEL_NAMES.__len__()  # 22
        num_classes = len(Y_LABEL_NAMES)
        label_names = Y_LABEL_NAMES
        print(f"[e6] 使用 Y 细粒度分类体系（{num_classes} 类）")

    # 确定 samples_per_class
    n_spc = args.samples_per_class or data_cfg.get("samples_per_class", 100)
    T = data_cfg.get("T", 30)
    seed = data_cfg.get("seed", 42)

    print(f"[data] samples_per_class={n_spc}, T={T}, seed={seed}")
    print(f"[data] 生成合成数据...")

    samples = make_synthetic_dataset(samples_per_class=n_spc, T=T, seed=seed)
    total = len(samples)

    # Y' 映射
    if taxonomy_variant == "Y_prime":
        samples = _map_samples_to_yprime(samples)

    # 切分 train/val (8:2)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total)
    val_n = int(total * data_cfg.get("val_split", 0.2))
    train_samples = [samples[i] for i in indices[val_n:]]
    val_samples = [samples[i] for i in indices[:val_n]]
    print(f"[split] train={len(train_samples)} val={len(val_samples)}")

    # 构建模型
    model = build_stgcn_bc(
        in_channels=model_cfg["in_channels"],
        num_classes=num_classes,
        base_channels=model_cfg.get("base_channels", 64),
        num_stages=model_cfg.get("num_stages", 10),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={total_params:,}, num_classes={num_classes}")

    # 训练配置
    tc = TrainConfig(
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        epochs=train_cfg.get("epochs", 50),
        batch_size=train_cfg.get("batch_size", 32),
        use_amp=train_cfg.get("use_amp", True),
        device=train_cfg.get("device", "auto"),
        early_stopping=train_cfg.get("early_stopping", True),
        patience=train_cfg.get("patience", 15),
        output_dir=str(REPO_ROOT / train_cfg["output_dir"]),
    )
    trainer = STGCNBCTrainer(model, train_samples, val_samples, config=tc)
    summary = trainer.fit()

    # 打印结果
    taxonomy_label = "Y' 粗粒度(21类)" if taxonomy_variant == "Y_prime" else "Y 细粒度(22类)"
    print("\n" + "=" * 64)
    print(f"P0.5 {taxonomy_label} 训练结果")
    print("=" * 64)
    print(f"  samples_per_class : {n_spc}")
    print(f"  taxonomy          : {taxonomy_variant}")
    print(f"  num_classes       : {num_classes}")
    print(f"  total_samples     : {total}")
    print(f"  epochs_run        : {summary['total_epochs_trained']}")
    print(f"  best_val_acc      : {summary['best_val_acc']:.4f}")
    print(f"  best_epoch        : {summary['best_epoch']}")
    print(f"  final_train_acc   : {summary['final_train_acc']:.4f}")
    print(f"  final_val_acc     : {summary['final_val_acc']:.4f}")
    print(f"  device            : {summary['device']}")
    print(f"  ckpt_path         : {summary.get('ckpt_path', 'N/A')}")

    # 判据检查
    random_baseline = eval_cfg.get("random_baseline", 1.0 / num_classes)
    convergence_threshold = eval_cfg.get("convergence_threshold", random_baseline * 3)
    checks = {}

    # 1. loss 单调下降趋势
    history_path = Path(tc.output_dir) / "history.json"
    if history_path.exists():
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
        if hist:
            losses = [h["train_loss"] for h in hist]
            first_third = np.mean(losses[:len(losses)//3])
            last_third = np.mean(losses[-(len(losses)//3):])
            loss_decreasing = last_third < first_third
            checks["loss_monotonic_trend"] = loss_decreasing
            print(f"\n[check] loss 趋势: first_third={first_third:.4f} last_third={last_third:.4f} → {'✓' if loss_decreasing else '✗'}")

    # 2. val_acc 显著超随机基线
    acc_ok = summary["best_val_acc"] >= convergence_threshold
    checks["val_acc_significant"] = acc_ok
    print(f"[check] val_acc={summary['best_val_acc']:.4f} ≥ {convergence_threshold:.4f} (随机基线×3) → {'✓' if acc_ok else '✗'}")

    # 3. 无 NaN
    no_nan = not np.isnan(summary["best_val_acc"])
    checks["no_nan"] = no_nan
    print(f"[check] 无 NaN: {'✓' if no_nan else '✗'}")

    all_pass = all(checks.values())
    print(f"\n[overall] {'✓ 全部通过' if all_pass else '✗ 有未通过项'}")
    print(f"[time] 总耗时 {time.time()-t0:.1f}s")

    # 写结果 JSON
    date_str = datetime.now().strftime("%Y-%m-%d")
    suffix = "Yprime" if taxonomy_variant == "Y_prime" else "Y"
    out_json = REPO_ROOT / "reports" / f"p05-stgcnbc-synthetic-{n_spc}perclass-{suffix}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "synthetic",
        "config": cfg,
        "experiment": {
            "samples_per_class": n_spc,
            "taxonomy": taxonomy_variant,
            "num_classes": num_classes,
            "total_samples": total,
        "train_size": len(train_samples),
        "val_size": len(val_samples),
        },
        "summary": {k: (float(v) if hasattr(v, "item") else v) for k, v in summary.items()},
        "checks": {k: bool(v) for k, v in checks.items()},
        "all_pass": bool(all_pass),
        "total_time_sec": round(time.time() - t0, 1),
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[output] 结果 JSON -> {out_json}")


if __name__ == "__main__":
    main()
