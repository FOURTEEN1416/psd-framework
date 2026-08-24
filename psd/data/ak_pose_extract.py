# -*- coding: utf-8 -*-
"""AK 犬科 mp4 自提取姿态管线（W20-C 路线，用户裁决 2026-08-24）.

裁决口径:
  - 骨架路线 C: 本地犬科 mp4 抽帧 → YOLO11-pose(dog-pose 微调权重, 24 点) 提点
    → 组装 (T=30, 24, 3) 时序骨架样本 → ST-GCN+BC 微调公开真实层数字
  - 宽松门禁 4 类: jump/stay/track/watch（train+val 合计 ≥10 视频）
  - 样本判定 R2(first-mapped-hit): 视频标签序列中首个属于部分类协议的动作

拓扑事实: dog-pose.yaml 的 24 关键点与 K9Graph(assets-map §2) 逐名逐序一致
  —— 提点输出零投影损失，backbone 冻结微调的解耦实验设计完整保留。

分层设计: 本模块上半部为纯逻辑（TDD 覆盖），下半部为 IO/模型层
  （集成冒烟覆盖，不进单测——依赖真实视频与权重）。
"""
from __future__ import annotations

import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

#: 宽松门禁可训练 4 类（保持 assets-map §1 相对顺序 → 编号 0..3）
GATE4_CLASSES: List[str] = ["stay", "track", "watch", "jump"]

#: 4 类训练编号
GATE4_CLASS_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(GATE4_CLASSES)}

#: ST-GCN 输入时序长度（对齐合成层协议 T=30）
CLIP_LEN_T: int = 30

__all__ = [
    "GATE4_CLASSES",
    "GATE4_CLASS_TO_IDX",
    "first_mapped_label",
    "select_samples",
    "uniform_frame_indices",
    "pick_best_instance",
    "assemble_clip",
    "extract_missing_videos_from_tar",
]


def first_mapped_label(label_seq: List[str], allowed: Optional[set] = None) -> Optional[str]:
    """R2 规则: 返回标签序列中首个命中协议池的 PSD 类名.

    Args:
        label_seq: AK 动作 index 字符串列表（保序），如 ``["5", "67", "3"]``。
        allowed: 限定协议池（如宽松门禁 4 类）; None = 全部 12 类映射池。

    Returns:
        PSD 类名；无命中返回 None。
    """
    from psd.data.ak_mapping import map_ak_index  # 局部导入避免循环依赖

    for token in label_seq:
        token = str(token).strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError:
            continue
        cls = map_ak_index(idx)
        if cls is not None and (allowed is None or cls in allowed):
            return cls
    return None


def select_samples(
    video_labels_by_split: Dict[str, Dict[str, List[str]]],
    split_of: Dict[str, str],
    canine_ids: set,
    local_mp4_ids: set,
) -> List[dict]:
    """构建训练样本清单（纯逻辑，IO 由调用方完成）.

    Args:
        video_labels_by_split: {split: {video_id: [ak_index_str, ...]}}，
            来自 train.csv / val.csv 的视频级保序标签序列。
        split_of: video_id -> 'train' | 'val'（AR_metadata type 列）。
        canine_ids: 犬科视频集合（W2 口径A）。
        local_mp4_ids: 已解压本地 mp4 的 video_id 集合。

    Returns:
        样本字典列表: video_id / split / psd_class / class_idx / source('mp4'|'tar')。
    """
    samples: List[dict] = []
    seen = set()
    for split in ("train", "val"):
        labels_map = video_labels_by_split.get(split, {})
        for vid, seq in labels_map.items():
            if vid not in canine_ids or vid in seen:
                continue
            cls = first_mapped_label(seq, allowed=set(GATE4_CLASSES))
            if cls not in GATE4_CLASS_TO_IDX:
                continue
            seen.add(vid)
            samples.append(
                {
                    "video_id": vid,
                    "split": split,
                    "psd_class": cls,
                    "class_idx": GATE4_CLASS_TO_IDX[cls],
                    "source": "mp4" if vid in local_mp4_ids else "tar",
                }
            )
    return samples


def uniform_frame_indices(n_frames: int, t: int = CLIP_LEN_T) -> List[int]:
    """从 n_frames 帧中均匀抽 t 帧; 不足 t 帧时循环补齐.

    Raises:
        ValueError: n_frames <= 0。
    """
    if n_frames <= 0:
        raise ValueError(f"n_frames 必须为正, 得到 {n_frames}")
    if n_frames >= t:
        pos = np.linspace(0, n_frames - 1, t).round().astype(int)
        return pos.tolist()
    # 短视频循环补齐（披露策略: 低资源场景保留全部动态）
    base = list(range(n_frames))
    out: List[int] = []
    while len(out) < t:
        out.extend(base)
    return out[:t]


def pick_best_instance(instances: List[dict]) -> Optional[np.ndarray]:
    """多检测实例取最高置信度.

    Args:
        instances: [{"kp": (24,3) ndarray, "score": float}, ...]。

    Returns:
        最高分的 (24,3) ndarray; 空列表返回 None。
    """
    if not instances:
        return None
    best = max(instances, key=lambda d: d["score"])
    return np.asarray(best["kp"], dtype=np.float32)


def assemble_clip(
    frames: List[Optional[np.ndarray]],
    label_idx: int,
    t: int = CLIP_LEN_T,
) -> Optional[dict]:
    """组装时序样本; 缺检帧线性插值, 首尾缺检最近邻复制, 全缺返回 None.

    Args:
        frames: 每帧 (24,3) 或 None（未检出），长度应为 t。
        label_idx: 门禁 4 类编号 0..3。

    Returns:
        {"keypoints": (t,24,3) f32, "label": int, "boundary": (t,) f32,
         "n_interpolated": int}; 全缺检返回 None。
    """
    valid_positions = [i for i, f in enumerate(frames) if f is not None]
    if not valid_positions:
        return None

    nan_frame = np.full((frames[valid_positions[0]].shape), np.nan, dtype=np.float32)
    kp_stack = np.stack(
        [f.astype(np.float32) if f is not None else nan_frame for f in frames]
    )

    # 首尾最近邻复制
    first_v, last_v = valid_positions[0], valid_positions[-1]
    kp_stack[:first_v] = kp_stack[first_v]
    kp_stack[last_v + 1 :] = kp_stack[last_v]

    # 中间 NaN 线性插值
    for i in range(first_v, last_v + 1):
        if not np.isnan(kp_stack[i]).any():
            continue
        prev_v = max(j for j in valid_positions if j < i) if any(j < i for j in valid_positions) else last_v
        next_v = min(j for j in valid_positions if j > i) if any(j > i for j in valid_positions) else prev_v
        span = next_v - prev_v
        w = (i - prev_v) / span if span else 0.0
        kp_stack[i] = (1 - w) * kp_stack[prev_v] + w * kp_stack[next_v]

    assert not np.isnan(kp_stack).any(), "插值后仍存在 NaN"
    return {
        "keypoints": kp_stack.astype(np.float32),
        "label": int(label_idx),
        "boundary": np.zeros(t, dtype=np.float32),
        "n_interpolated": sum(1 for i in range(len(frames)) if frames[i] is None),
    }


def extract_missing_videos_from_tar(
    tar_path: str | Path,
    needed_ids: Iterable[str],
    dest_dir: str | Path,
) -> Dict[str, Path]:
    """从 AK video.tar.gz 流式补抽缺失 mp4 到 dest_dir（K9 数据只读合规）.

    Args:
        tar_path: ``dataset/video.tar.gz`` 路径。
        needed_ids: 缺失的 video_id 集合。
        dest_dir: 解压目的地（本仓 runs/public_real_* 缓存）。

    Returns:
        {video_id: 解压后 mp4 路径}；tar 中不存在的 id 不出现在结果中。
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(needed_ids)
    got: Dict[str, Path] = {}
    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            name = m.name
            stem = Path(name).stem
            if m.isfile() and stem in wanted and stem not in got and name.endswith(".mp4"):
                target = dest / f"{stem}.mp4"
                with tf.extractfile(m) as src, open(target, "wb") as dst:
                    dst.write(src.read())
                got[stem] = target
            if len(got) == len(wanted):
                break
    return got
