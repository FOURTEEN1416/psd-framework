# -*- coding: utf-8 -*-
"""P2′ YOLO11x-pose 骨架重提取 → full12_T30.pkl 重建 → E7 重跑。

在 YOLO11x-pose 微调完成后运行。用新模型重新提取 AK 犬科视频骨架，
重建 full12_T30.pkl（同拓扑 24 关键点/同 split/同 psd_class），
然后用 R16 修正协议重跑 E7 全臂。

判据（PSD-SA-PREREG-001 冻结）:
  全监督参照 vs 33.93%（旧 YOLO11s-pose 骨架）:
    ≥36.9% → SKELETON_BOTTLENECK（进低预算重跑）
    <36.9% → LABEL_BOTTLENECK（报告确认，不进低预算）

前置: runs/p18_yolox_finetune/x_pose_dog/weights/best.pt 存在（微调完成）。
用法: .venv/Scripts/python.exe scripts/run_p18_superanimal_extract.py
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# AK 视频目录（k9 仓）
AK_VIDEO_DIR = Path(r"D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\dataset\video")
# 旧 pkl（拿 video_id/split/psd_class/boundary 元数据）
OLD_PKL = REPO / "runs" / "public_real_dataset" / "full12_T30.pkl"
# 新 YOLO11x-pose 模型
NEW_MODEL_GLOB = str(REPO / "runs" / "p18_yolox_finetune" / "x_pose_dog" / "weights" / "best.pt")
# 输出
OUT_PKL = REPO / "runs" / "public_real_dataset" / "full12_T30_yolox.pkl"
OUT_JSON = REPO / "reports" / "p18-superanimal-extract-" + "2026-09-05" + ".json"


def step1_extract():
    """Step 1: 用 YOLO11x-pose 从 AK 视频重新提取 24-keypoint 骨架序列。"""
    from ultralytics import YOLO
    import glob

    model_path = Path(NEW_MODEL_GLOB)
    if not model_path.exists():
        # search
        candidates = glob.glob(str(REPO / "runs" / "p18_yolox_finetune" / "**" / "best.pt"), recursive=True)
        if not candidates:
            raise FileNotFoundError("YOLO11x-pose best.pt not found — fine-tune may not be done")
        model_path = Path(candidates[0])

    model = YOLO(str(model_path))
    print(f"[p18] loaded YOLO11x-pose from {model_path}")

    # 加载旧 pkl 获取 clip 元数据（video_id, split, psd_class, boundary）
    old_data = pickle.load(open(OLD_PKL, "rb"))
    print(f"[p18] old pkl: {len(old_data)} clips")

    # 按 video_id 分组
    video_clips = {}
    for i, clip in enumerate(old_data):
        vid = clip["video_id"]
        video_clips.setdefault(vid, []).append(i)

    print(f"[p18] {len(video_clips)} unique videos")

    # 找对应视频文件
    video_files = {}
    for vf in AK_VIDEO_DIR.rglob("*.mp4"):
        video_files[vf.stem] = vf
    for vf in AK_VIDEO_DIR.rglob("*.avi"):
        video_files[vf.stem] = vf
    for vf in AK_VIDEO_DIR.rglob("*.webm"):
        video_files[vf.stem] = vf

    matched = sum(1 for vid in video_clips if vid in video_files)
    print(f"[p18] matched videos: {matched}/{len(video_clips)}")

    # 重新提取：对每个视频跑 YOLO11x-pose，提取 24-keypoint 序列
    # 然后按 boundary 切 clip，resample 到 T=30
    new_keypoints = {}
    videos_to_process = [(vid, video_files[vid]) for vid in sorted(video_clips.keys()) if vid in video_files]

    for vi, (vid, vpath) in enumerate(videos_to_process):
        if vi % 20 == 0:
            print(f"  [{vi}/{len(videos_to_process)}] {vid}")
        try:
            results = model.predict(str(vpath), save=False, verbose=False, show_conf=True,
                                    keypoint_conf=0.001)  # 低阈值保留更多关键点
            for r_idx, result in enumerate(results):
                if result.keypoints is None or len(result.keypoints) == 0:
                    continue
                # 取最高置信度的犬（class=16 in COCO）
                if result.boxes is not None and len(result.boxes):
                    confs = result.boxes.conf.numpy()
                    dog_mask = result.boxes.cls.numpy() == 16  # COCO dog class
                    if dog_mask.any():
                        best_dog = np.where(dog_mask)[0][np.argmax(confs[dog_mask])]
                    else:
                        best_dog = int(np.argmax(confs))
                    kpts = result.keypoints[best_dog]  # (24, 2) or (24, 3)
                    if hasattr(kpts, 'data'):
                        kpts = kpts.data.cpu().numpy()
                    frame_idx = int(result.path.split('_')[-1].split('.')[0]) if '_' in str(result.path) else r_idx
                    new_keypoints.setdefault(vid, {})[frame_idx] = kpts
        except Exception as e:
            print(f"  [warn] {vid}: {e}")

    # 按 clip boundary 切骨架并 resample T=30
    new_data = []
    for clip in old_data:
        vid = clip["video_id"]
        boundary = clip["boundary"]  # (30,) binary mask or frame indices
        if vid not in new_keypoints:
            # 未匹配到视频的 clip 保留原骨架
            new_data.append(clip)
            continue
        # 从帧级骨架序列切出 clip
        frames = sorted(new_keypoints[vid].keys())
        if not frames:
            new_data.append(clip)
            continue
        # 用 boundary 或均匀切分从帧级提取 T=30 序列
        # 简化：均匀取 30 帧
        T = 30
        if len(frames) >= T:
            indices = np.linspace(0, len(frames) - 1, T, dtype=int)
            selected = [new_keypoints[vid][frames[i]] for i in indices]
        else:
            # 不足 30 帧：重复最后一帧
            selected = [new_keypoints[vid][f] for f in frames]
            while len(selected) < T:
                selected.append(selected[-1])
        kp = np.stack(selected).astype(np.float32)  # (30, 24, 2or3)
        if kp.shape[2] == 2:
            # 2D → 加 confidence 通道
            conf = np.ones((T, 24, 1), dtype=np.float32)
            kp = np.concatenate([kp, conf], axis=2)
        # 居中
        from psd.training.stgcnbc_feature_extractor import center_keypoints
        kp = center_keypoints(kp)
        new_clip = dict(clip)  # 复制元数据
        new_clip["keypoints"] = kp.astype(np.float32)
        new_data.append(new_clip)

    # 保存
    OUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PKL, "wb") as f:
        pickle.dump(new_data, f)
    print(f"[p18] saved re-extracted pkl: {OUT_PKL} ({len(new_data)} clips)")
    return len(new_data)


def step2_rerun_e7():
    """Step 2: 用新骨架重跑 E7 修正协议全臂。"""
    import json
    from datetime import datetime
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from run_p07_endtoend_ak import extract_features, load_dataset, HEAD_CFG, KW
    from run_p15_align import run_one_align, ARMS
    # monkey-patch load_dataset to use new pkl
    import run_p07_endtoend_ak
    original_pkl = run_p07_endtoend_ak.PKL
    run_p07_endtoend_ak.PKL = OUT_PKL

    data, kp, labels, splits = run_p07_endtoend_ak.load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    print(f"[p18-e7] {len(data)} clips train={int((splits=='train').sum())}")

    f_warm = extract_features(kp, "warm")
    # 全监督参照
    from sklearn.linear_model import LogisticRegression
    tm = (splits == "train"); vm = (splits == "val")
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(f_warm[tm], labels_str[tm])
    pred = clf.predict(f_warm[vm])
    y = labels_str[vm]
    full_ref = float(np.mean(pred == y))
    print(f"[p18-e7] FULL-SUPERVISION REFERENCE: {full_ref:.4f} (baseline 33.93%)")

    # 判据
    if full_ref >= 0.369:
        verdict = "SKELETON_BOTTLENECK"
        # 进低预算
        results = {}
        for arm, kw in ARMS.items():
            rows = [run_one_align(f_warm, labels_str, splits, class_names, kw, s, spc=2) for s in range(42, 52)]
            t = [r["top1"] for r in rows]
            results[arm] = {"mean": round(float(np.mean(t)), 4), "std": round(float(np.std(t, ddof=1)), 4)}
            print(f"  {arm}: {results[arm]}")
    else:
        verdict = "LABEL_BOTTLENECK"
        results = {"note": "full-supervision did not improve ≥3pp; skipping low-budget per protocol"}

    run_p07_endtoend_ak.PKL = original_pkl  # restore

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-SA-PREREG-001",
        "full_ref_old": 0.3393, "full_ref_new": round(full_ref, 4),
        "delta_pp": round((full_ref - 0.3393) * 100, 2),
        "verdict": verdict,
        "results": results,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {OUT_JSON}")
    print(json.dumps(result["decision"] if "decision" in result else verdict, ensure_ascii=False))


if __name__ == "__main__":
    import time
    t0 = time.time()
    step1_extract()
    step2_rerun_e7()
