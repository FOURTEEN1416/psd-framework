"""Q3c — AK 公开真实层 4 类子集 ST-GCN+BC 微调（backbone 冻结，语义层重训）.

消费 Q3b 产物 runs/public_real_dataset/partialclass4_T30.pkl
（list[dict]: keypoints(T,24,3)/label(0-3)/split/video_id/psd_class），
init 自 runs/p05_stgcn_bc_full/best.pt 的 Y 预训练 backbone（W19 同款双层冻结），
仅重训 4 类新 head → 论文 tab2 公开真实层数字。

用法:
    python scripts/run_p05_public_real_finetune.py \
        --pkl runs/public_real_dataset/partialclass4_T30.pkl \
        --init runs/p05_stgcn_bc_full/best.pt \
        --output-json reports/p05-public-real-partialclass-result-<日期>.json

口径: 公开真实层（AGENTS.md 硬规则 3），禁止与合成层混排。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from psd.models.stgcn_bc import STGCNBC  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402
from run_c1_decouple import freeze_backbone, load_y_backbone  # noqa: E402  W19 双层冻结复用


def load_dataset(pkl_path: str | Path) -> tuple[List[dict], List[dict], Dict[str, int]]:
    """pkl → (train_samples, val_samples, 类分布). split 字段权威，缺失即报错。"""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    if not data:
        raise ValueError(f"{pkl_path} 为空——Q3b 提点产物不可用")
    train = [d for d in data if d.get("split") == "train"]
    val = [d for d in data if d.get("split") == "val"]
    if not train or not val:
        raise ValueError(f"split 不完整: train={len(train)} val={len(val)}")
    dist = Counter(d["psd_class"] for d in data)
    return train, val, dict(dist)


def per_class_val_acc(model, val_samples: List[dict], device: str) -> Dict[str, float]:
    """逐类验证精度（predict 接口，CPU/GPU 均可）。"""
    model.eval()
    hits: Dict[str, List[bool]] = {}
    with torch.no_grad():
        for s in val_samples:
            x = torch.from_numpy(np.asarray(s["keypoints"], dtype=np.float32)).unsqueeze(0)
            cls_logits, _ = model(x.to(device))
            pred = int(cls_logits.argmax(dim=1).item())
            name = str(s["psd_class"])
            hits.setdefault(name, []).append(pred == int(s["label"]))
    return {k: round(float(np.mean(v)), 4) for k, v in sorted(hits.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pkl", default="runs/public_real_dataset/partialclass4_T30.pkl")
    ap.add_argument("--init", default="runs/p05_stgcn_bc_full/best.pt")
    ap.add_argument("--num-classes", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--output-dir", default="runs/public_real_finetune")
    ap.add_argument("--output-json", required=True)
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train, val, dist = load_dataset(REPO / args.pkl)
    print(f"[data] train={len(train)} val={len(val)} dist={dist}")

    model = STGCNBC(in_channels=3, num_classes=args.num_classes)
    info = load_y_backbone(model, REPO / args.init)   # 完整模型入口：自动剥离 head.*，head 保持全新
    print(f"[init] backbone←{args.init} missing(head 新建)={len(info.missing)} unexpected={len(info.unexpected)}")
    freeze_backbone(model)   # W19 双层冻结（requires_grad + BN eval 补丁）
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"[freeze] 可训 {trainable:,} / 冻结 {frozen:,}")

    tc = TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, patience=args.patience,
        seed=args.seed, device=args.device,
        output_dir=str(REPO / args.output_dir), use_amp=True,
    )
    trainer = STGCNBCTrainer(model, train, val, config=tc)
    fit_res = trainer.fit()

    per_class = per_class_val_acc(model, val, tc.device if tc.device != "auto"
                                  else ("cuda" if torch.cuda.is_available() else "cpu"))
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real",
        "protocol": "AK partialclass4 frozen-backbone head-retrain (warm-init from Y best.pt)",
        "config_echo": {"pkl": args.pkl, "init": args.init, "num_classes": args.num_classes,
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "seed": args.seed, "n_train": len(train), "n_val": len(val),
                        "class_dist": dist},
        "summary": {**fit_res,
                    "per_class_val_acc": per_class,
                    "wall_clock_sec": round(time.time() - t0, 1)},
    }
    out = REPO / args.output_json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}")
    print(f"[done] best_val_acc={fit_res['best_val_acc']:.4f} per_class={per_class}")


if __name__ == "__main__":
    main()
