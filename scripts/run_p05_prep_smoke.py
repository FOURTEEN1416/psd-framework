"""P0.5 合成层冒烟训练主入口（W11 窗口）.

执行链:
  Step 3: 加载合成数据 → 构建 ST-GCN+BC 模型 → 短训 → 检查收敛判据

用法:
    python scripts/run_p05_prep_smoke.py --config configs/p05_stgcn_bc.yaml --smoke
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
    STGCNBCDataset,
    make_synthetic_dataset,
)
from psd.models.stgcn_bc_model import build_stgcn_bc  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p05_stgcn_bc.yaml")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式（缩短 epoch 数）")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    print("=" * 64)
    print("P0.5 ST-GCN+BC 合成层冒烟训练（W11 窗口）")
    print("=" * 64)

    # GPU 错峰检查
    if torch.cuda.is_available():
        mem_used = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[gpu] 当前已用 {mem_used:.2f} GB / {mem_total:.2f} GB")
        if mem_used > 4.0:
            print("[gpu] ⚠️ GPU 占用较高，可能与其他窗口冲突")
    else:
        print("[gpu] CPU 模式运行")

    # 加载/生成合成数据
    synth_path = REPO_ROOT / cfg["data"]["synthetic_path"]
    if synth_path.exists():
        import pickle
        with open(synth_path, "rb") as f:
            samples = pickle.load(f)
        print(f"[data] 加载合成集: {len(samples)} 样本 ({synth_path})")
    else:
        print(f"[data] 生成合成集（路径不存在: {synth_path}）")
        samples = make_synthetic_dataset(
            samples_per_class=cfg["data"]["samples_per_class"],
            T=cfg["data"]["T"],
            seed=cfg["data"]["seed"],
        )
        from psd.data.stgcn_bc_dataset import save_synthetic_dataset
        save_synthetic_dataset(samples, str(synth_path))
        print(f"[data] 已保存: {len(samples)} 样本")

    # 切分 train/val
    n = len(samples)
    split = cfg["data"]["val_split"]
    rng = np.random.default_rng(cfg["data"]["seed"])
    indices = rng.permutation(n)
    val_n = int(n * split)
    train_idx, val_idx = indices[val_n:], indices[:val_n]
    train_ds = STGCNBCDataset(
        samples=[samples[i] for i in train_idx],
        T=cfg["data"]["T"],
        augment=True,
    )
    val_ds = STGCNBCDataset(
        samples=[samples[i] for i in val_idx],
        T=cfg["data"]["T"],
        augment=False,
    )
    print(f"[split] train={len(train_ds)} val={len(val_ds)}")

    # 构建模型
    model_cfg = cfg["model"]
    model = build_stgcn_bc(
        in_channels=model_cfg["in_channels"],
        num_classes=model_cfg["num_classes"],
        base_channels=model_cfg["base_channels"],
        num_stages=model_cfg["num_stages"],
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={total_params:,}")

    # 训练
    train_cfg = cfg["train"]
    epochs = 5 if args.smoke else train_cfg.get("epochs", 30)
    tc = TrainConfig(
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
        epochs=epochs,
        batch_size=train_cfg["batch_size"],
        use_amp=train_cfg.get("use_amp", True),
        device=train_cfg.get("device", "auto"),
        early_stopping=train_cfg.get("early_stopping", False),
        patience=999,
        output_dir=str(REPO_ROOT / train_cfg["output_dir"]),
    )
    trainer = STGCNBCTrainer(model, train_ds, val_ds, config=tc)
    summary = trainer.fit()

    # 打印结果
    print("\n" + "=" * 64)
    print("冒烟结果")
    print("=" * 64)
    print(f"  epochs_run: {summary['total_epochs_trained']}")
    print(f"  best_val_acc: {summary['best_val_acc']:.4f}")
    print(f"  final_train_acc: {summary['final_train_acc']:.4f}")
    print(f"  final_val_acc: {summary['final_val_acc']:.4f}")
    print(f"  device: {summary['device']}")
    print(f"  ckpt_path: {summary.get('ckpt_path', 'N/A')}")

    # 判据检查
    criteria = cfg.get("smoke_criteria", {})
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
    else:
        checks["loss_monotonic_trend"] = False
        print("\n[check] loss 趋势: ✗（无 history.json）")

    # 2. val_acc ≥ 10%
    random_baseline = 0.045
    acc_threshold = criteria.get("val_acc_vs_random_baseline", 0.10)
    acc_ok = summary["best_val_acc"] >= acc_threshold
    checks["val_acc_ge_10pct"] = acc_ok
    print(f"[check] val_acc={summary['best_val_acc']:.4f} ≥ {acc_threshold} → {'✓' if acc_ok else '✗'}")

    # 3. 无 NaN
    no_nan = not np.isnan(summary["best_val_acc"])
    checks["no_nan"] = no_nan
    print(f"[check] 无 NaN: {'✓' if no_nan else '✗'}")

    all_pass = all(checks.values())
    print(f"\n[overall] {'✓ 全部通过' if all_pass else '✗ 有未通过项'}")
    print(f"[time] 总耗时 {time.time()-t0:.1f}s")

    # 写结果 JSON
    out_json = REPO_ROOT / "reports" / "p05-smoke-result.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "synthetic",
        "config": cfg,
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
