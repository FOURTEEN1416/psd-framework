"""ST-GCN+BC 数据集 + 合成数据生成器.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/dataset.py`（只读参考）

数据源优先级:
    1. 真实数据（pyskl pickle）— 主路径
    2. InterPet4D kp_world 3D（无行为标签）— 辅助
    3. 合成数据（基于行为模板 + 时序噪声）— baseline 验证
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from psd.models.stgcn_bc_constants import NUM_KEYPOINTS
from psd.models.stgcn_bc_labels import ALL_BEHAVIORS_22, BEHAVIOR_TO_IDX, NUM_BEHAVIORS


# ============================================================================
# 22 类行为的"姿态模板"（用于合成数据生成）
# ============================================================================
# 每个模板是 (24, 3) 的关键点偏置，模拟该行为的典型姿态
# 偏置单位：相对犬体尺度的百分比（-0.3 ~ +0.3）

_BASE_POSE_3D: np.ndarray = np.array([
    # 前左肢 (0-2): paw → knee → elbow
    [-0.10,  0.10, 0.00],   # 0  front_left_paw
    [-0.10,  0.10, 0.30],   # 1  front_left_knee
    [-0.10,  0.10, 0.50],   # 2  front_left_elbow
    # 后左肢 (3-5): paw → knee → elbow
    [-0.10, -0.20, 0.00],   # 3  rear_left_paw
    [-0.10, -0.20, 0.30],   # 4  rear_left_knee
    [-0.10, -0.20, 0.50],   # 5  rear_left_elbow
    # 前右肢 (6-8): paw → knee → elbow
    [ 0.10,  0.10, 0.00],   # 6  front_right_paw
    [ 0.10,  0.10, 0.30],   # 7  front_right_knee
    [ 0.10,  0.10, 0.50],   # 8  front_right_elbow
    # 后右肢 (9-11): paw → knee → elbow
    [ 0.10, -0.20, 0.00],   # 9  rear_right_paw
    [ 0.10, -0.20, 0.30],   # 10 rear_right_knee
    [ 0.10, -0.20, 0.50],   # 11 rear_right_elbow
    # 尾部 (12-13)
    [ 0.00, -0.30, 0.55],   # 12 tail_start
    [ 0.00, -0.50, 0.50],   # 13 tail_end
    # 耳基 (14-15)
    [-0.05,  0.08, 0.78],   # 14 left_ear_base
    [ 0.05,  0.08, 0.78],   # 15 right_ear_base
    # 头部 (16-17)
    [ 0.00,  0.15, 0.80],   # 16 nose
    [ 0.00,  0.12, 0.75],   # 17 chin
    # 耳尖 (18-19)
    [-0.05,  0.10, 0.82],   # 18 left_ear_tip
    [ 0.05,  0.10, 0.82],   # 19 right_ear_tip
    # 眼睛 (20-21)
    [-0.03,  0.10, 0.80],   # 20 left_eye
    [ 0.03,  0.10, 0.80],   # 21 right_eye
    # 躯干根 (22-23)
    [ 0.00,  0.00, 0.60],   # 22 withers
    [ 0.00,  0.05, 0.70],   # 23 throat
], dtype=np.float32)


def _behavior_pose_template(behavior: str) -> np.ndarray:
    """返回指定行为的姿态模板 (24, 3)."""
    pose = _BASE_POSE_3D.copy()
    behavior_lower = behavior.lower()
    if behavior_lower == "sit":
        pose[12:24, 2] -= 0.2
        pose[16:24, 2] -= 0.3
    elif behavior_lower == "down":
        pose[:, 2] -= 0.5
    elif behavior_lower == "stand":
        pass
    elif behavior_lower == "heel":
        pose[0:5, 0] += 0.1
    elif behavior_lower == "sit_up":
        pose[8:14, 2] += 0.2
        pose[12:24, 2] -= 0.2
    elif behavior_lower == "stay":
        pass
    elif behavior_lower == "bark":
        pose[0:5, 2] += 0.15
    elif behavior_lower == "bite":
        pose[0:5, 0] += 0.2
        pose[0:5, 2] += 0.05
    elif behavior_lower == "track":
        pose[0:5, 2] -= 0.2
        pose[0:5, 0] += 0.1
    elif behavior_lower in ("alert_sit", "alert_down"):
        if "down" in behavior_lower:
            pose[:, 2] -= 0.5
        pose[0:5, 0] += 0.15
    elif behavior_lower == "search_blind":
        pose[0:5, 0] += 0.1
    elif behavior_lower == "apprehend":
        pose[:, 0] += 0.15
        pose[0:5, 0] += 0.25
    elif behavior_lower == "escort":
        pose[0:5, 0] += 0.1
    elif behavior_lower == "obstacle":
        pose[8:14, 2] += 0.3
    elif behavior_lower == "recall":
        pose[:, 0] += 0.2
    elif behavior_lower == "watch":
        pose[0:5, 2] += 0.1
    elif behavior_lower == "guard":
        pose[0:5, 2] += 0.05
    elif behavior_lower == "release":
        pose[0:5, 2] += 0.1
        pose[0:5, 0] -= 0.05
    elif behavior_lower == "retrieve":
        pose[0:5, 2] -= 0.15
    elif behavior_lower == "jump":
        pose[:, 2] += 0.3
    elif behavior_lower == "scale":
        pose[8:14, 2] += 0.4
        pose[:, 2] += 0.1
    return pose.astype(np.float32)


def _generate_clip(
    behavior: str,
    T: int = 30,
    noise_std: float = 0.05,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """生成单个合成 clip."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    template = _behavior_pose_template(behavior)
    t = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    sin_phase = rng.uniform(0, 2 * np.pi)
    sin_amp = 0.05
    sin_wave = sin_amp * np.sin(2 * np.pi * t + sin_phase)

    keypoints = np.zeros((T, NUM_KEYPOINTS, 3), dtype=np.float32)
    for f in range(T):
        keypoints[f] = template + sin_wave[f] * 0.1
        keypoints[f, :, 0] += rng.normal(0, noise_std, NUM_KEYPOINTS)
        keypoints[f, :, 1] += rng.normal(0, noise_std, NUM_KEYPOINTS)
        keypoints[f, :, 2] += rng.normal(0, noise_std, NUM_KEYPOINTS)

    boundary_labels = np.zeros(T, dtype=np.float32)
    boundary_labels[:2] = 1.0
    boundary_labels[-2:] = 1.0

    return keypoints, boundary_labels


def make_synthetic_dataset(
    samples_per_class: int = 10,
    T: int = 30,
    noise_std: float = 0.05,
    seed: int = 42,
) -> List[Dict]:
    """生成合成数据集（22 类 × N 样本/类）."""
    rng = np.random.default_rng(seed)
    samples: List[Dict] = []
    for behavior in ALL_BEHAVIORS_22:
        label_idx = BEHAVIOR_TO_IDX[behavior]
        for i in range(samples_per_class):
            clip_seed = int(rng.integers(0, 2**31 - 1))
            kpt, boundary = _generate_clip(
                behavior, T=T, noise_std=noise_std, seed=clip_seed
            )
            samples.append({
                "keypoints": kpt,
                "label": label_idx,
                "label_name": behavior,
                "boundary": boundary,
                "frame_dir": f"syn_{behavior}_{i:03d}",
            })
    return samples


def save_synthetic_dataset(samples: List[Dict], output_path: str) -> None:
    """保存合成数据集为 pickle."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)


def load_pyskl_pickle(pkl_path: str) -> List[Dict]:
    """加载 pyskl 格式 pickle."""
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)
    samples: List[Dict] = []
    for ann in raw:
        kpt = ann["keypoint"]
        if kpt.ndim != 4:
            raise ValueError(f"keypoint 应为 4D (M,T,V,C), 实际 {kpt.shape}")
        M, T, V, C = kpt.shape
        if M != 1:
            raise ValueError(f"当前仅支持单犬 M=1, 实际 M={M}")
        if V != NUM_KEYPOINTS:
            raise ValueError(f"关键点数应为 {NUM_KEYPOINTS}, 实际 {V}")
        kpt_single = kpt[0]
        if C == 2:
            score = ann.get("keypoint_score")
            if score is None:
                score = np.ones((T, V), dtype=np.float32)
            else:
                score = score[0]
            kpt_3c = np.concatenate(
                [kpt_single, score[..., np.newaxis]], axis=-1
            )
        else:
            kpt_3c = kpt_single
        samples.append({
            "keypoints": kpt_3c.astype(np.float32),
            "label": int(ann.get("label", -1)),
            "label_name": ann.get("label_name", ""),
            "boundary": ann.get("boundary", np.zeros(T, dtype=np.float32)),
            "frame_dir": ann.get("frame_dir", "unknown"),
        })
    return samples


# ============================================================================
# PyTorch Dataset
# ============================================================================

class STGCNBCDataset(Dataset):
    """ST-GCN+BC 训练数据集."""

    def __init__(
        self,
        samples: Optional[List[Dict]] = None,
        pkl_path: Optional[str] = None,
        T: int = 30,
        augment: bool = False,
        normalize: bool = True,
    ):
        if samples is not None:
            self.samples = samples
        elif pkl_path is not None:
            self.samples = load_pyskl_pickle(pkl_path)
        else:
            raise ValueError("必须提供 samples 或 pkl_path")
        self.T = T
        self.augment = augment
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        kpt = sample["keypoints"].copy()
        label = sample["label"]
        boundary = sample["boundary"].copy()
        T_orig = kpt.shape[0]

        if T_orig >= self.T:
            if self.augment:
                start = np.random.randint(0, T_orig - self.T + 1)
            else:
                start = (T_orig - self.T) // 2
            kpt = kpt[start:start + self.T]
            boundary = boundary[start:start + self.T]
        else:
            pad_len = self.T - T_orig
            kpt = np.concatenate(
                [kpt, np.tile(kpt[-1:], (pad_len, 1, 1))], axis=0
            )
            boundary = np.concatenate(
                [boundary, np.zeros(pad_len, dtype=np.float32)]
            )

        if self.augment:
            if np.random.rand() < 0.5:
                kpt[..., 0] = -kpt[..., 0]
            scale = 1.0 + np.random.uniform(-0.1, 0.1)
            kpt[..., :3] = kpt[..., :3] * scale

        if self.normalize:
            kpt = self._normalize(kpt)

        return {
            "keypoints": torch.from_numpy(kpt).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "boundary": torch.from_numpy(boundary).float(),
            "frame_dir": sample.get("frame_dir", ""),
        }

    @staticmethod
    def _normalize(kpt: np.ndarray) -> np.ndarray:
        """按 withers 中心 + 体长尺度归一化."""
        center = kpt[:, 22:23, :].mean(axis=0, keepdims=True)
        kpt = kpt - center
        bone_ref = kpt[:, 22, :2] - kpt[:, 12, :2]
        bone_len = np.linalg.norm(bone_ref, axis=-1).mean()
        if bone_len < 1e-6:
            bone_len = 1.0
        kpt[..., :2] = kpt[..., :2] / bone_len
        return kpt.astype(np.float32)


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """批处理 collate 函数."""
    keypoints = torch.stack([b["keypoints"] for b in batch])
    labels = torch.stack([b["label"] for b in batch])
    boundaries = torch.stack([b["boundary"] for b in batch])
    frame_dirs = [b["frame_dir"] for b in batch]
    return {
        "keypoints": keypoints,
        "labels": labels,
        "boundaries": boundaries,
        "frame_dirs": frame_dirs,
    }


__all__ = [
    "STGCNBCDataset",
    "make_synthetic_dataset",
    "save_synthetic_dataset",
    "load_pyskl_pickle",
    "collate_fn",
    "ALL_BEHAVIORS_22",
    "BEHAVIOR_TO_IDX",
    "NUM_BEHAVIORS",
]
