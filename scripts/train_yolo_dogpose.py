# -*- coding: utf-8 -*-
"""dog-pose 微调训练入口（W20-C 路线, GPU 空闲后执行）.

产出 24 点犬类姿态权重(K9Graph 同拓扑), 供 run_p05_public_real_pipeline.py 提点.

前置: NTU Phase B 结束(nvidia-smi 显存回落 <2GB)方可启动——GPU 排队纪律.
用法: python scripts/train_yolo_dogpose.py [--epochs 50] [--model yolo11s-pose.pt]
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11s-pose.pt", help="迁移起点(COCO pose 迁移)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data="dog-pose.yaml",          # D:\Desktop\datasets\dog-pose (已预下载)
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        project=str(REPO / "runs" / "public_real_yolo_dogpose"),
        name="train",
        exist_ok=True,
    )
    best = REPO / "runs" / "public_real_yolo_dogpose" / "train" / "weights" / "best.pt"
    print(f"\n[done] best weights: {best}")
    print("下一步: python scripts/run_p05_public_real_pipeline.py --stage extract "
          f"--weights {best}")


if __name__ == "__main__":
    main()
