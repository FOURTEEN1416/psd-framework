"""P0.3 Phase B 自训语义桥 — W13-C1 方案 B（用户裁决：B）。

背景：runs/p05_stgcn_bc_full/ 正被并行窗口实时覆写（2026-08-24 13:20 实测
last.pt mtime 与检查时刻相差 15s），其 best.pt 内容与内嵌 val_acc 元数据矛盾
（声称 96.36%，实测 9-13%）——外部产物不可作为稳定依赖。经用户裁决改为本窗
口独立训练语义桥，配方与 W12 主运行完全一致：

  数据   make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
  切分   np.random.default_rng(42).permutation(2200)，val=前 440（同 W12 协议）
  超参   configs/p05_stgcn_bc_full.yaml train 段（lr 1e-3 / AdamW / cosine /
         warmup5 / batch32 / AMP / epochs50 / 不早停 / patience15）
  架构   build_stgcn_bc(in_channels=3, num_classes=22) 默认 64/10/mstcn

差异声明（报告须披露）：torch 种子本窗口固定为 42（W12 未固定，单次方差
±0.5pp）；输出目录归 W13 领地（runs/p03_phaseb_bridge/），零触碰 *p05* 文件。

产出：best.pt + bridge_meta.json（含 sha256 锁定 + val 子集复核精度）。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.synth_stgcn import ALL_BEHAVIORS_22, make_synthetic_dataset  # noqa: E402
from psd.models.stgcn_bc import build_stgcn_bc  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "runs" / "p03_phaseb_bridge"
SEED = 42


def main() -> None:
    t0 = time.time()
    print("=" * 64)
    print("P0.3 Phase B 自训语义桥（W13-C1 方案 B，配方=W12 主运行）")
    print("=" * 64)

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    samples = make_synthetic_dataset(samples_per_class=100, T=30,
                                     noise_std=0.05, seed=SEED)
    total = len(samples)
    indices = rng.permutation(total)
    val_n = int(total * 0.2)
    val_samples = [samples[i] for i in indices[:val_n]]
    train_samples = [samples[i] for i in indices[val_n:]]
    print(f"[data] train={len(train_samples)} val={len(val_samples)}（seed={SEED} 随机切分）")

    tc = TrainConfig(
        lr=1e-3, weight_decay=1e-4, epochs=50, batch_size=32,
        use_amp=True, device="auto", early_stopping=False, patience=15,
        output_dir=str(OUTPUT_DIR),
    )
    model = build_stgcn_bc(in_channels=3, num_classes=len(ALL_BEHAVIORS_22))
    trainer = STGCNBCTrainer(model, train_samples, val_samples, config=tc)
    summary = trainer.fit()
    print(f"[train] {json.dumps(summary, ensure_ascii=False)}")

    # ---- 复核：加载落盘 best.pt 重评 val 子集（防「元数据≠权重」再现）
    best_path = OUTPUT_DIR / "best.pt"
    ck = torch.load(best_path, map_location="cpu", weights_only=False)
    recheck_model = build_stgcn_bc(in_channels=3, num_classes=len(ALL_BEHAVIORS_22))
    recheck_model.load_state_dict(ck["model_state_dict"], strict=True)
    recheck_model = recheck_model.to(trainer.device).eval()
    correct = n = 0
    with torch.no_grad():
        for i in range(0, len(val_samples), 128):
            batch = [val_samples[j] for j in range(i, min(i + 128, len(val_samples)))]
            x = torch.stack([torch.from_numpy(np.asarray(
                s["keypoints"], dtype=np.float32)) for s in batch]).to(trainer.device)
            y = torch.tensor([int(s["label"]) for s in batch]).to(trainer.device)
            logits, _ = recheck_model(x)
            correct += int((logits.argmax(1) == y).sum())
            n += len(batch)
    recheck_val_acc = correct / max(n, 1)
    gap_ok = abs(recheck_val_acc - float(ck.get("best_val_acc", 0.0))) < 0.02
    print(f"[recheck] 落盘权重 val 复核={recheck_val_acc:.4f} "
          f"(ckpt 内嵌 {float(ck.get('best_val_acc', 0)):.4f}) → "
          f"{'一致' if gap_ok else '不一致！拒绝交付'}")

    # ---- 哈希锁定
    sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
    meta = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "recipe": {
            "generator": "make_synthetic_dataset(spc=100,T=30,noise_std=0.05,seed=42)",
            "split": f"rng({SEED}).permutation, val=first 440 (W12 protocol)",
            "hyperparams": {"lr": 1e-3, "weight_decay": 1e-4, "epochs": 50,
                            "batch_size": 32, "use_amp": True,
                            "early_stopping": False},
            "arch": "build_stgcn_bc defaults (64ch/10stages/mstcn)",
        },
        "summary": {k: v for k, v in summary.items()},
        "recheck_val_acc": round(float(recheck_val_acc), 4),
        "metadata_consistent": bool(gap_ok),
        "sha256_best_pt": sha,
        "ckpt_path": str(best_path.relative_to(REPO_ROOT)),
    }
    with open(OUTPUT_DIR / "bridge_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[lock] sha256={sha[:16]}… | meta -> bridge_meta.json")
    print(f"\n[done] 总耗时 {time.time()-t0:.1f}s | "
          f"{'✅ 可交付' if gap_ok and recheck_val_acc >= 0.9 else '⚠️ 未达预期，需人工复核'}")


if __name__ == "__main__":
    main()
