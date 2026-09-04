"""P1.1 AK v2 多段扩容数据集构建 — 预注册协议 PSD-AKV2-PREREG-001 逐字实现。

v1 每视频仅 1 clip（seen.add(vid)），犬科视频中位 80 帧的标注素材被浪费；
v2 按协议切 K=min(4,max(1,n//40)) 个连续段，段标签=池类帧覆盖率最高者且门 ≥0.80，
段内均匀抽 30 帧走同一 YOLO(dog-pose best.pt)+assemble_clip+质量门。
split 仍按视频（AK 官方 train/val 成员），v1 数字不替换，v2 为复现/鲁棒层。

用法:
    .venv/Scripts/python.exe scripts/run_p11_ak_v2_build.py            # 全量
    .venv/Scripts/python.exe scripts/run_p11_ak_v2_build.py --limit 20 # 冒烟
产出:
    runs/public_real_dataset/full12v2_T30.pkl + full12v2_manifest.json
    + full12v2_extract_quality.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from psd.data.ak_mapping import MAPPED_PSD_CLASSES, PSD_PARTIAL_CLASS_TO_IDX, map_ak_index
from psd.data.ak_pose_extract import CLIP_LEN_T, assemble_clip, pick_best_instance, uniform_frame_indices
from run_p05_public_real_full12 import (
    CACHE_DIR,
    CANINE_SPECIES,
    META,
    MEAN_VIS_THRESHOLD,
    TRAIN_CSV,
    VAL_CSV,
    VIDEO_DIR,
    VIDEO_TAR,
    extract_missing_videos_from_tar,
    parse_species,
)

POOL = set(MAPPED_PSD_CLASSES)
CONSISTENCY_GATE = 0.80
MIN_SEG_FRAMES = 40
MAX_SEGS = 4
EXPECTED_KPTS = 24

OUT_DIR = REPO / "runs" / "public_real_dataset"
MANIFEST_JSON = OUT_DIR / "full12v2_manifest.json"
QUALITY_JSON = OUT_DIR / "full12v2_extract_quality.json"
DATASET_PKL = OUT_DIR / "full12v2_T30.pkl"


def scan_frame_labels() -> dict:
    """逐帧标签 {video_id: {frame_idx0: set(PSD类名)}}，仅犬科。"""
    meta = pd.read_excel(META)
    canine_ids = set(meta.loc[meta["list_animal"].apply(
        lambda c: bool(parse_species(c) & CANINE_SPECIES)), "video_id"].astype(str))
    out: dict = {}
    for csv in (TRAIN_CSV, VAL_CSV):
        for chunk in pd.read_csv(csv, sep=" ", chunksize=1_000_000):
            chunk = chunk[chunk["original_vido_id"].astype(str).isin(canine_ids)]
            for _, row in chunk.iterrows():
                vid = str(row["original_vido_id"])
                fi = int(row["frame_id"]) - 1
                mapped = set()
                for tok in str(row["labels"]).split(","):
                    tok = tok.strip()
                    if not tok:
                        continue
                    try:
                        cls = map_ak_index(int(tok))
                    except ValueError:
                        continue
                    if cls in POOL:
                        mapped.add(cls)
                out.setdefault(vid, {})[fi] = mapped
    return out


def plan_segments(frame_labels: dict) -> list:
    """协议 §2: K 段连续切分 + 段标签覆盖率门。返回 [{video_id, split, seg, start, end, psd_class}]。"""
    # split 成员表（与 v1 一致: AK 官方 train/val csv）
    tr_ids = set()
    va_ids = set()
    for csv, sink in ((TRAIN_CSV, tr_ids), (VAL_CSV, va_ids)):
        for chunk in pd.read_csv(csv, sep=" ", usecols=["original_vido_id"], chunksize=1_000_000):
            sink.update(chunk["original_vido_id"].astype(str))
    samples = []
    for vid, fmap in frame_labels.items():
        n = max(fmap.keys()) + 1
        split = "train" if vid in tr_ids else "val"
        k = min(MAX_SEGS, max(1, n // MIN_SEG_FRAMES))
        bounds = np.linspace(0, n, k + 1).round().astype(int)
        for si in range(k):
            s, e = int(bounds[si]), int(bounds[si + 1])
            if e - s < MIN_SEG_FRAMES:
                continue
            cov = Counter()
            for fi in range(s, e):
                for cls in fmap.get(fi, set()):
                    cov[cls] += 1
            if not cov:
                continue
            cls, cnt = cov.most_common(1)[0]
            if cnt / (e - s) < CONSISTENCY_GATE:
                continue
            samples.append({"video_id": vid, "split": split, "seg": si, "start": s, "end": e,
                            "psd_class": cls, "class_idx": PSD_PARTIAL_CLASS_TO_IDX[cls],
                            "seg_coverage": round(cnt / (e - s), 3)})
    return samples


def extract_segment(model, video_path: str, start: int, end: int, t: int = CLIP_LEN_T):
    """段内均匀抽 t 帧提点——与 v1 extract_one_video 同构，仅帧索引窗口改为 [start,end)。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": "open_failed"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None, {"error": "zero_frames"}
    end = min(end, n)
    if end - start < 2:
        cap.release()
        return None, {"error": "seg_beyond_video"}
    idxs = [start + i for i in uniform_frame_indices(end - start, t)]
    want = set(idxs)
    frames, stats = [], {"no_detect": 0, "low_conf": 0}
    for fi in range(max(idxs) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if fi not in want:
            continue
        res = model(frame, verbose=False)[0]
        if res.keypoints is None or res.keypoints.xyn is None or res.keypoints.xyn.shape[1] != EXPECTED_KPTS:
            frames.append(None)
            stats["no_detect"] += 1
            continue
        inst_list = []
        if len(res.boxes) > 0:
            xyn = res.keypoints.xyn.cpu().numpy()
            kconf = res.keypoints.conf.cpu().numpy()
            bconf = res.boxes.conf.cpu().numpy()
            for i in range(len(bconf)):
                vis = np.clip(np.nan_to_num(kconf[i], nan=0.0), 0, 1)
                kp = np.concatenate([xyn[i], vis[:, None]], axis=1).astype(np.float32)
                inst_list.append({"kp": kp, "score": float(bconf[i])})
        best = pick_best_instance(inst_list)
        if best is None:
            frames.append(None)
            stats["no_detect"] += 1
        elif best[0, 0] == -1 or float(best[:, 2].mean()) < MEAN_VIS_THRESHOLD:
            frames.append(None)
            stats["low_conf"] += 1
        else:
            frames.append(best)
    cap.release()
    while len(frames) < t:
        frames.append(frames[-1] if frames else None)
    return frames, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--weights", default="runs/public_real_yolo_dogpose/train/weights/best.pt")
    args = ap.parse_args()

    print("[scan] per-frame canine labels...")
    fmap = scan_frame_labels()
    print(f"  videos with frames: {len(fmap)}")
    samples = plan_segments(fmap)
    dist = Counter(s["psd_class"] for s in samples)
    print(f"[plan] gated segments: {len(samples)} | classes: {dict(dist)}")

    # 视频路径解析（本地 → 缓存 → tar 补抽），与 v1 同构
    local_mp4 = {p.stem for p in VIDEO_DIR.glob("*.mp4")}
    needed = {s["video_id"] for s in samples if s["video_id"] not in local_mp4}
    got = extract_missing_videos_from_tar(VIDEO_TAR, needed, CACHE_DIR) if needed else {}
    for s in samples:
        s["video_path"] = str(VIDEO_DIR / f"{s['video_id']}.mp4") if s["video_id"] in local_mp4 \
            else str(got.get(s["video_id"], ""))
    samples = [s for s in samples if s["video_path"] and Path(s["video_path"]).exists()]
    if args.limit:
        samples = samples[:args.limit]
    print(f"[paths] resolvable samples: {len(samples)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps(
        {"protocol": "PSD-AKV2-PREREG-001", "clip_t": CLIP_LEN_T,
         "gates": {"consistency": CONSISTENCY_GATE, "min_seg_frames": MIN_SEG_FRAMES, "max_segs": MAX_SEGS},
         "samples": samples}, ensure_ascii=False, indent=2), encoding="utf-8")

    from ultralytics import YOLO
    model = YOLO(str(REPO / args.weights) if not Path(args.weights).is_absolute() else args.weights)
    data, quality = [], []
    for i, s in enumerate(samples):
        frames, st = extract_segment(model, s["video_path"], s["start"], s["end"])
        clip = assemble_clip(frames, s["class_idx"]) if frames else None
        rec = {**{k: s[k] for k in ("video_id", "split", "seg", "psd_class", "seg_coverage")},
               **({"status": "ok", "n_interpolated": clip["n_interpolated"]} if clip else
                  {"status": "fail", **st, **({"error": "assemble_none"} if clip is None and frames else {})})}
        quality.append(rec)
        if clip is not None:
            data.append({"keypoints": clip["keypoints"], "label": clip["label"],
                         "boundary": clip["boundary"], "video_id": s["video_id"],
                         "seg": s["seg"], "split": s["split"], "psd_class": s["psd_class"]})
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(samples)}] ok={len(data)}")

    import pickle
    with open(DATASET_PKL, "wb") as f:
        pickle.dump(data, f)
    QUALITY_JSON.write_text(json.dumps(
        {"weights": args.weights, "n_samples": len(data), "protocol": "full12v2",
         "distribution": dict(Counter(d["psd_class"] for d in data)),
         "splits": dict(Counter(d["split"] for d in data)),
         "quality": quality}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {DATASET_PKL}: {len(data)} clips | train={sum(1 for d in data if d['split']=='train')} "
          f"val={sum(1 for d in data if d['split']=='val')}")
    print("  classes:", dict(Counter(d["psd_class"] for d in data)))


if __name__ == "__main__":
    main()
