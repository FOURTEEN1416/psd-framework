"""MammalNet 家犬子集 → AimCLR NTU 兼容预训练池（P1 数据扩容接线）.

裁决口径（2026-10-20 对抗验收后的接线方案）:
  - 用途限定: **无标签自监督预训练池**（AimCLR Feeder_triple 的 train_data.npy 扩容源），
    绝不进 K9-Bench-v0 验收门（标签域与 22 类工作犬考核零覆盖，见
    k9 仓 reports/p1-data-expansion-adversarial-acceptance-2026-10-20.md 发现#1）。
  - 技术栈: MammalNet mp4 → YOLO-pose（APTv2 犬科微调权重，17 点）
    → APT2_TO_K9 映射（build_unified_pool 权威表）→ K9Graph 24 点
    → to_ntu_view（interpet4d 契约）→ (N,3,T=64,25,1) train_data.npy + train_label.pkl。
  - 提点率参差是野生数据真实难度（冒烟实测 3/3、1/3、0/3 均有），
    失败 clip 计入 manifest 并跳过，不伪造。

产出（gitignore 内 data/processed/，不入库）:
- data/processed/p01_mammalnet/train_data.npy  : (N,3,T=64,25,1) float32
- data/processed/p01_mammalnet/train_label.pkl : ([name...],[breed_label...]) kNN probe 代理
- reports/p1-mammalnet-aimclr-pool-<日期>.json : 落盘统计（files/检测率/各 split 计数）

用法:
    & "D:\\Desktop\\PR论文\\psd-framework\\.venv\\Scripts\\python.exe" `
        scripts/export_mammalnet_pool.py `
        --data-root "D:/Desktop/PR论文/datasets/mammalnet_canine" `
        --pose-weights "D:/Desktop/PR论文/k9-training-system/runs/pose/runs/aptv2_finetune/yolo_canidae_pose/weights/best.pt" `
        --out-dir data/processed/p01_mammalnet --max-clips 0   # 0=全量

GPU 可用则用 GPU 提点；--max-clips N 冒烟用。CPU-only 路径不做。
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# build_unified_pool 的映射表是纯函数（无 IO），直接 import 复用
from scripts.build_unified_pool import APTV2_KPT_NAMES, map_aptv2_keypoints  # noqa: E402
from psd.data.interpet4d import (  # noqa: E402
    resample_to_fixed_t,
    to_ntu_view,
)

# 默认 MammalNet 家犬子集根（外层 datasets/，不入 git）
DEFAULT_DATA_ROOT = Path(r"D:/Desktop/PR论文/datasets/mammalnet_canine")
DEFAULT_POSE_WEIGHTS = Path(
    r"D:/Desktop/PR论文/k9-training-system/runs/pose/runs/aptv2_finetune/"
    r"yolo_canidae_pose/weights/best.pt"
)
TARGET_T = 64  # 与 configs/p01_aimclr.yaml / export_interpet4d 口径一致


def _extract_clip_kpts(
    model, video_path: Path, target_t: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """对单个 mp4 抽等距 target_t 帧，逐帧提点，返回 (T,17,3) 或 None（全程无检出）。

    返回的 17 点序为 APTv2 官方序（YOLO-pose APTv2 微调权重实测 kpt_shape=[17,3]）。
    """
    import cv2  # 延迟导入（IO 层）

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idxs = np.linspace(0, total - 1, target_t).astype(int)

    frames_kpts: List[Optional[np.ndarray]] = []
    frames_vis: List[Optional[np.ndarray]] = []
    hit = 0
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            frames_kpts.append(None)
            frames_vis.append(None)
            continue
        r = model.predict(frame, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            frames_kpts.append(None)
            frames_vis.append(None)
            continue
        xy = r.keypoints.xy[0].cpu().numpy()  # (17,2)
        conf = r.keypoints.conf
        vis = conf[0].cpu().numpy() if conf is not None else np.ones(17, dtype=np.float32)
        if xy.shape != (17, 2):
            frames_kpts.append(None)
            frames_vis.append(None)
            continue
        frames_kpts.append(xy.astype(np.float32))
        frames_vis.append(vis.astype(np.float32))
        hit += 1
    cap.release()

    if hit == 0:
        return None

    # 缺失帧用最近邻填充（保持时序等距）；首尾缺失则钳到最近有效帧
    valid = [i for i, k in enumerate(frames_kpts) if k is not None]
    filled_k = np.zeros((target_t, 17, 2), dtype=np.float32)
    filled_v = np.zeros((target_t, 17), dtype=np.float32)
    for i in range(target_t):
        if frames_kpts[i] is not None:
            filled_k[i] = frames_kpts[i]
            filled_v[i] = frames_vis[i]
        else:
            nearest = min(valid, key=lambda j: abs(j - i))
            filled_k[i] = frames_kpts[nearest]
            filled_v[i] = np.zeros(17, dtype=np.float32)  # 填充帧置信度归零
        kp17 = np.stack(
            [filled_k[..., 0], filled_k[..., 1], filled_v], axis=-1
        )  # (T,17,3) [x,y,vis]
    return kp17, filled_v

def _subset_key(name: str, subset: Dict[str, dict]) -> str:
    """把 clip 名解析为 subset_annotations 的键。

    MammalNet 的 60 个长视频被切成 `_1`/`_2` 分片（如 `_cjcT3Hwoi0_1`），
    subset_annotations 只登记基名（`_cjcT3Hwoi0`）。对抗验收实测 90/716 命中失败
    全部由该后缀造成（去后缀 90/90 命中）——此处统一剥离。
    """
    if name in subset:
        return name
    base = re.sub(r"_\d+$", "", name)
    return base if base in subset else name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    ap.add_argument("--pose-weights", default=str(DEFAULT_POSE_WEIGHTS))
    ap.add_argument("--out-dir", default="data/processed/p01_mammalnet")
    ap.add_argument("--target-t", type=int, default=TARGET_T)
    ap.add_argument("--max-clips", type=int, default=0, help="0=全量；>0 冒烟")
    ap.add_argument("--conf-threshold", type=float, default=0.5)
    args = ap.parse_args()

    src = Path(args.data_root)
    pose_w = Path(args.pose_weights)
    if not src.exists():
        raise SystemExit(f"[export] 数据根不存在: {src}")
    if not pose_w.exists():
        raise SystemExit(f"[export] 姿态权重不存在: {pose_w}")

    # 读 manifest 得 breed 标签（kNN probe 代理标签）
    sub_ann = src / "subset_annotations.json"
    if not sub_ann.exists():
        raise SystemExit(f"[export] 缺 subset_annotations.json: {sub_ann}")
    with open(sub_ann, "r", encoding="utf-8") as f:
        subset: Dict[str, dict] = json.load(f)

    videos = sorted(src.glob("videos/*/*.mp4"))
    if not videos:
        raise SystemExit(f"[export] 未找到 mp4: {src}/videos/*/*.mp4")
    if args.max_clips > 0:
        videos = videos[: args.max_clips]
    print(f"[export] clips={len(videos)} target_t={args.target_t}")

    from ultralytics import YOLO  # 延迟导入（IO/模型层）

    model = YOLO(str(pose_w))

    views: List[np.ndarray] = []
    names: List[str] = []
    skipped_no_detect: List[str] = []
    for i, vp in enumerate(videos, 1):
        stem = vp.stem
        got = _extract_clip_kpts(model, vp, args.target_t)
        if got is None:
            skipped_no_detect.append(stem)
            continue
        kp17, vis = got
        kp24 = map_aptv2_keypoints(kp17)  # (T,24,3)，第三维是 vis(0-1)，不可映射槽位为 NaN
        # 对抗验收修复（2026-10-20）：to_ntu_view 的第三维是 **z 坐标**，不是置信度。
        # 直接把 vis 当 z 送进去会被 _normalize_sequence 一并质心化/尺度化，
        # 产出 ch2∈[0, 0.005] 的残值——与 InterPet4D 参照池的 ch2∈[-0.78, 0.83] 语义冲突，
        # 两批数据混合预训练时模型会学到两套互斥的 z 信号（实测对照见验收报告）。
        # 正确做法（同仓 ak_pose_extract 的 2D 口径）：2D 源无真实 z，第三维置 0，
        # 置信度只作为 to_ntu_view 的 weight 用于低置信置零，不进张量。
        vis = kp24[..., 2].astype(np.float32)
        kp24_xyz = kp24.copy()
        kp24_xyz[..., 2] = 0.0  # 2D 源：z 诚实置 0（非伪造 3D）
        kp24_xyz = np.nan_to_num(kp24_xyz, nan=0.0)
        view = to_ntu_view(kp24_xyz, weight=vis, conf_threshold=args.conf_threshold)
        views.append(view)
        names.append(stem)
        if i % 50 == 0 or i == len(videos):
            print(f"[export] {i}/{len(videos)} 检出clip={len(views)}")

    if not views:
        raise SystemExit("[export] 无任何 clip 检出骨架，产出为空——检查权重/数据")

    data = np.stack(views).astype(np.float32)  # (N,3,T,25,1)

    # 代理标签：品种（subset_annotations 的 taxnomy[0].keyword），与 InterPet4D 的 dog ID 同一角色
    # 注意：分片 clip（xxx_1/xxx_2）须剥离后缀再查（_subset_key），否则误落 unknown
    breeds = sorted(
        {
            (subset.get(_subset_key(n, subset), {}).get("taxnomy") or [{}])[0].get("keyword", "unknown")
            for n in names
        }
    )
    breed_to_idx = {b: i for i, b in enumerate(breeds)}
    labels = [
        breed_to_idx[
            (subset.get(_subset_key(n, subset), {}).get("taxnomy") or [{}])[0].get("keyword", "unknown")
        ]
        for n in names
    ]

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_data.npy", data)
    with open(out_dir / "train_label.pkl", "wb") as f:
        pickle.dump((list(names), list(labels)), f)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(src),
        "pose_weights": str(pose_w),
        "clips_seen": len(videos),
        "clips_with_pose": len(views),
        "clips_skipped_no_detection": skipped_no_detect,
        "detection_rate": round(len(views) / max(1, len(videos)), 4),
        "train_data_shape": list(data.shape),
        "target_t": args.target_t,
        "num_proxy_classes": len(breeds),
        "breeds": breeds,
        "usage_scope": "pretrain_pool_only（无标签自监督；不进 K9-Bench-v0 验收门）",
    }
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    rpath = REPO_ROOT / "reports" / f"p1-mammalnet-aimclr-pool-{datetime.now():%Y-%m-%d}.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[export] train_data.npy {tuple(data.shape)} -> {out_dir}")
    print(f"[export] train_label.pkl ({len(names)} names, {len(breeds)} proxy breeds)")
    print(f"[export] 检出率 {report['detection_rate']:.1%}，跳过 {len(skipped_no_detect)} clip")
    print(f"[export] 报告 -> {rpath}")


if __name__ == "__main__":
    main()
