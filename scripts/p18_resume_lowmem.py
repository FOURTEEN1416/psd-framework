# -*- coding: utf-8 -*-
"""P2' YOLO11x-pose 低内存续训 — resume=True 因 RAM 枯竭(bad allocation)不可用时降级方案。

原训练 15.7GB 总内存机器上 workers=8 + val 峰值 → epoch12 val 阶段 std::bad_alloc。
本脚本加载 last.pt 权重(epoch 11)重开训练: workers=2 + batch=2 压 RAM;
AdamW 自适应优化器下 optimizer 状态重置影响有限; 超参对齐原 args.yaml。
产出目录 x_pose_dog-3 (与 -2 的真 resume 历史隔离)。
"""
from ultralytics import YOLO

CKPT = r"D:\Desktop\k9-training-system\runs\pose\runs\p18_yolox_finetune\x_pose_dog-2\weights\last.pt"
DATA = r"D:\Desktop\datasets\dog-pose\dog-pose.yaml"
PROJECT = r"D:\Desktop\k9-training-system\runs\pose\runs\p18_yolox_finetune"

def train():
    model = YOLO(CKPT)
    model.train(
        data=DATA, epochs=89, batch=2, workers=2, imgsz=640,
        project=PROJECT, name="x_pose_dog-3", exist_ok=True,
        optimizer="AdamW", lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3.0, cos_lr=False, patience=20, seed=0, deterministic=True,
        device=0,
    )


if __name__ == "__main__":
    train()
