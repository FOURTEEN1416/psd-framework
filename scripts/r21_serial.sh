#!/bin/bash
# R21 serial chain: P2′ → extract+E7 → P5-B NTU120 (no Python wrapper, no self-kill)
cd /d/Desktop/psd-framework
PY="D:/Desktop/psd-framework/.venv/Scripts/python.exe"
export PYTHONUNBUFFERED=1

echo "=== STEP 1: P2′ YOLO11x-pose fine-tune ==="
$PY -c "
from ultralytics import YOLO
model = YOLO('yolo11x-pose.pt')
model.train(data='D:/Desktop/datasets/dog-pose/dog-pose.yaml',
    epochs=100, imgsz=640, batch=4, device=0,
    project='runs/p18_yolox_finetune', name='x_pose_dog',
    patience=20, save=True, amp=True, lr0=0.001, lrf=0.01, optimizer='AdamW')
print('FINE-TUNE COMPLETE')
"
if [ $? -ne 0 ]; then echo "P2′ fine-tune FAILED"; exit 1; fi
echo "=== P2′ fine-tune DONE ==="

echo "=== STEP 2: P2′ extract + E7 rerun ==="
$PY scripts/run_p18_superanimal_extract.py
if [ $? -ne 0 ]; then echo "P2′ extract FAILED"; exit 1; fi
echo "=== P2′ extract+E7 DONE ==="

echo "=== STEP 3: P5-B NTU120 retention ==="
$PY scripts/run_p19_ntu120_retention.py --pretext --max 12000
if [ $? -ne 0 ]; then echo "P5-B FAILED"; exit 1; fi
echo "=== P5-B DONE ==="

echo "=== ALL SERIAL PIPELINE COMPLETE ==="
