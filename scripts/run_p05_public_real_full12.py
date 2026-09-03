# -*- coding: utf-8 -*-
"""AK 犬科 12 类全类扩展管线（R1 补强, 用户裁决 A' 2026-09-03）.

背景: Q3c 部分类协议(4 类门禁) → 44.90% 类不平衡主导, R1 行为级证据不足.
本管线放开至 MAPPED_PSD_CLASSES 全部 12 类(关门禁), 其余协议与 Q3c 逐字一致:
  - 犬科 caliber-A 名单 / 本地 mp4 + tar 补抽 / YOLO11s-pose(dog-pose) 提点 / T=30 组装
  - 样本判定 R2(first-mapped-hit), allowed = 全部 12 类映射池
  - 类不平衡如实报告(per-class 强制同框), macro-F1 为补充主指标

预注册(BOARD 09-03): 两臂 scratch vs warm-init × 3 seeds(42/43/44);
  旧 4 类结果保留不动, 本管线为新增独立实验。

用法:
  python scripts/run_p05_public_real_full12.py --stage manifest
  python scripts/run_p05_public_real_full12.py --stage extract --weights <best.pt>
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from psd.data.ak_mapping import MAPPED_PSD_CLASSES, PSD_PARTIAL_CLASS_TO_IDX  # noqa: E402
from psd.data.ak_pose_extract import (  # noqa: E402
    CLIP_LEN_T,
    assemble_clip,
    extract_missing_videos_from_tar,
    first_mapped_label,
    pick_best_instance,
    uniform_frame_indices,
)

META = Path(r"D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\AR_metadata.xlsx")
TRAIN_CSV = Path(r"D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\annotation\train.csv")
VAL_CSV = Path(r"D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\annotation\val.csv")
VIDEO_DIR = Path(r"D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\dataset\video")
VIDEO_TAR = VIDEO_DIR.parent / "video.tar.gz"

CACHE_DIR = REPO / "runs" / "public_real_video_cache"
OUT_DIR = REPO / "runs" / "public_real_dataset"
MANIFEST_JSON = OUT_DIR / "full12_manifest.json"
EXTRACT_QUALITY_JSON = OUT_DIR / "full12_extract_quality.json"
DATASET_PKL = OUT_DIR / "full12_T30.pkl"

CANINE_SPECIES = {
    "African Painted Dog", "Coyote", "Desert Fox", "Dholes", "Dingo Dog",
    "Dog", "Fox", "Jackal", "Wild Dog", "Wolf",
}
EXPECTED_KPTS = 24
CONF_THRESHOLD = 0.30
MEAN_VIS_THRESHOLD = 0.20


def parse_species(cell) -> set:
    try:
        return {str(x).strip() for x in eval(str(cell))}
    except Exception:
        return set()


def scan_video_labels(csv_path: Path) -> dict:
    """视频级保序标签序列 {video_id: [ak_index_str,...]}（与 Q3c 逐字一致）."""
    seq: dict[str, list] = {}
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().strip().split(" ")
        iv, il = header.index("original_vido_id"), header.index("labels")
        for line in f:
            p = line.rstrip("\n").split(" ")
            if len(p) <= max(iv, il):
                continue
            vid = p[iv]
            seq.setdefault(vid, [])
            for t in p[il].split(","):
                t = t.strip()
                if t and t not in seq[vid]:
                    seq[vid].append(t)
    return seq


def select_samples_full12(video_labels_by_split, canine_ids, local_mp4_ids):
    """与 ak_pose_extract.select_samples 同构, allowed 放开为全部 12 类映射池."""
    samples: list[dict] = []
    seen = set()
    for split in ("train", "val"):
        labels_map = video_labels_by_split.get(split, {})
        for vid, seq in labels_map.items():
            if vid not in canine_ids or vid in seen:
                continue
            cls = first_mapped_label(seq, allowed=set(MAPPED_PSD_CLASSES))
            if cls not in PSD_PARTIAL_CLASS_TO_IDX:
                continue
            seen.add(vid)
            samples.append(
                {
                    "video_id": vid,
                    "split": split,
                    "psd_class": cls,
                    "class_idx": PSD_PARTIAL_CLASS_TO_IDX[cls],
                    "source": "mp4" if vid in local_mp4_ids else "tar",
                }
            )
    return samples


def build_manifest():
    meta = pd.read_excel(META)
    mask = meta["list_animal"].apply(lambda c: bool(parse_species(c) & CANINE_SPECIES))
    canine_ids = set(meta.loc[mask, "video_id"].astype(str))

    tr_seq, va_seq = scan_video_labels(TRAIN_CSV), scan_video_labels(VAL_CSV)
    local_mp4 = {p.stem for p in VIDEO_DIR.glob("*.mp4")}

    samples = select_samples_full12(
        video_labels_by_split={"train": tr_seq, "val": va_seq},
        canine_ids=canine_ids,
        local_mp4_ids=local_mp4,
    )
    info = {
        "canine_total": len(canine_ids),
        "local_mp4_total": len(local_mp4),
        "samples": Counter(s["split"] for s in samples),
        "classes": Counter(s["psd_class"] for s in samples),
    }
    return samples, info


def stage_manifest() -> list[dict]:
    samples, info = build_manifest()
    print(f"[manifest] 犬科 {info['canine_total']} 视频 | 本地mp4 {info['local_mp4_total']}")
    print(f"[manifest] 样本: {dict(info['samples'])} | 类别: {dict(info['classes'])}")

    needed = [s["video_id"] for s in samples if s["source"] == "tar"]
    print(f"[tar] 需补抽 {len(needed)} 个视频 ...")
    got = extract_missing_videos_from_tar(VIDEO_TAR, needed, CACHE_DIR)
    miss = sorted(set(needed) - set(got))
    print(f"[tar] 补抽成功 {len(got)} / 失败 {len(miss)} {miss[:8]}")

    for s in samples:
        if s["source"] == "tar" and s["video_id"] in got:
            s["video_path"] = str(got[s["video_id"]])
        else:
            s["video_path"] = str(VIDEO_DIR / f"{s['video_id']}.mp4")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_JSON.write_text(
        json.dumps({"protocol": "full12_all_mapped_classes", "classes": MAPPED_PSD_CLASSES,
                    "clip_t": CLIP_LEN_T,
                    "build_info": {k: (dict(v) if isinstance(v, Counter) else v)
                                   for k, v in info.items()},
                    "missing_after_tar": miss,
                    "samples": samples}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[manifest] 清单落盘 {MANIFEST_JSON}")
    return [s for s in samples if Path(s["video_path"]).exists()]


def extract_one_video(model, video_path: str, t: int = CLIP_LEN_T):
    """单视频抽帧+提点（与 Q3c extract_one_video 逐字一致）."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {"error": "open_failed"}
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release()
        return None, {"error": "zero_frames"}
    idxs = uniform_frame_indices(n, t)
    want = set(idxs)
    frames, stats = [], {"no_detect": 0, "low_conf": 0}
    for fi in range(max(idxs) + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if fi not in want:
            continue
        res = model(frame, verbose=False)[0]
        if res.keypoints is None or res.keypoints.xyn is None:
            frames.append(None)
            stats["no_detect"] += 1
            continue
        if res.keypoints.xyn.shape[1] != EXPECTED_KPTS:
            raise RuntimeError(
                f"权重拓扑不匹配: 输出 {res.keypoints.xyn.shape[1]} 点, 需要 {EXPECTED_KPTS} 点"
            )
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


def stage_extract(weights: str, limit: int | None = None) -> None:
    from ultralytics import YOLO

    samples_all, _ = build_manifest()
    needed = [s["video_id"] for s in samples_all if s["source"] == "tar"]
    got = extract_missing_videos_from_tar(VIDEO_TAR, needed, CACHE_DIR)
    for s in samples_all:
        s["video_path"] = (str(got[s["video_id"]]) if s["source"] == "tar" and s["video_id"] in got
                           else str(VIDEO_DIR / f"{s['video_id']}.mp4"))
    samples = [s for s in samples_all if Path(s["video_path"]).exists()]
    if limit:
        samples = samples[:limit]

    model = YOLO(weights)
    dataset, quality = [], []
    for i, s in enumerate(samples, 1):
        frames, st = extract_one_video(model, s["video_path"])
        clip = assemble_clip(frames, s["class_idx"]) if frames else None
        if clip is None:
            quality.append({**s, "status": "all_missing"})
            print(f"  [{i}/{len(samples)}] {s['video_id']} ALL-MISSING")
            continue
        dataset.append({
            "keypoints": clip["keypoints"],
            "label": clip["label"],
            "boundary": clip["boundary"],
            "video_id": s["video_id"], "split": s["split"], "psd_class": s["psd_class"],
        })
        quality.append({**s, "status": "ok", "n_interpolated": clip["n_interpolated"],
                        **st})
        if i % 10 == 0 or i == len(samples):
            print(f"  [{i}/{len(samples)}] 完成, 累计样本 {len(dataset)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import pickle

    with open(DATASET_PKL, "wb") as f:
        pickle.dump(dataset, f)
    dist = Counter((d["split"], d["psd_class"]) for d in dataset)
    EXTRACT_QUALITY_JSON.write_text(json.dumps({
        "weights": weights, "n_samples": len(dataset), "protocol": "full12",
        "classes": MAPPED_PSD_CLASSES,
        "distribution": {f"{a}/{b}": c for (a, b), c in sorted(dist.items())},
        "quality": quality}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] 样本 {len(dataset)} → {DATASET_PKL}")
    print(f"[done] 质量统计 → {EXTRACT_QUALITY_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["manifest", "extract"], required=True)
    ap.add_argument("--weights", default="yolo11s-pose.pt", help="YOLO pose 权重路径")
    ap.add_argument("--limit", type=int, default=None, help="冒烟模式: 只处理前 N 个")
    args = ap.parse_args()
    if args.stage == "manifest":
        stage_manifest()
    else:
        stage_extract(args.weights, args.limit)


if __name__ == "__main__":
    main()
