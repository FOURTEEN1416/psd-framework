# -*- coding: utf-8 -*-
"""P7: YOLO11x-pose 灵长类姿态微调（AP-10K ape 子集, 17kpt）。

DLC snapshot-60000(TF) 无法在 DLC 3.0.1(pytorch) 加载且无转换器 → 改走与 P2'
同栈的 ultralytics 微调。yolo11x-pose.pt(COCO) 本身 kpt_shape=[17,3] 与 AP-10K
一致, 零拓扑重塑。batch=2/workers=2 为 15.7GB RAM 机器纪律(p18 教训第 7 条)。

用法:
    .venv/Scripts/python.exe -u scripts/p7_finetune_ape_pose.py            # 正式 100ep
    .venv/Scripts/python.exe -u scripts/p7_finetune_ape_pose.py --smoke    # 2ep 烟测
产出: k9-training-system/runs/pose/runs/p7_ape_pose/(weights/best.pt)
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data" / "ap10k" / "ape-pose" / "ape-pose.yaml"
CKPT = REPO / "yolo11x-pose.pt"
PROJECT = r"D:\Desktop\k9-training-system\runs\pose\runs"


def train(epochs: int, name: str):
    from ultralytics import YOLO

    model = YOLO(str(CKPT))
    model.train(
        data=str(DATA), epochs=epochs, batch=2, workers=2, imgsz=640,
        project=PROJECT, name=name, exist_ok=True,
        optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=False, patience=30, seed=0, deterministic=True,
        device=0,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        train(epochs=2, name="p7_ape_smoke")
    else:
        train(epochs=100, name="p7_ape_pose")
