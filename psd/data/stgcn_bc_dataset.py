"""ST-GCN+BC 数据集模块（W11 移植 + W12 扩量测试兼容层）.

导出: make_synthetic_dataset, ALL_BEHAVIORS_22, NUM_CLASSES, NUM_JOINTS,
      save_synthetic_dataset, STGCNBCDataset, collate_fn
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

# 从实现模块重新导出（单一真相：psd.data.synth_stgcn）
from psd.data.synth_stgcn import (  # noqa: F401
    ALL_BEHAVIORS_22,
    NUM_CLASSES,
    NUM_JOINTS,
    _generate_clip,
    make_synthetic_dataset,
    save_synthetic_dataset,
)


class STGCNBCDataset(Dataset):
    """PyTorch Dataset for ST-GCN+BC samples.

    Args:
        samples: List[Dict] from make_synthetic_dataset
        T: 帧数（用于断言，非必须与 samples 一致）
        augment: 是否做数据增强（当前版本不支持，保留参数便于 W12 扩展）
    """

    def __init__(self, samples: List[Dict], T: int = 30, augment: bool = False) -> None:
        self.samples = samples
        self.T = T
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        kpt = np.asarray(s["keypoints"], dtype=np.float32)  # (T, 24, 3)
        return {
            "keypoints": torch.from_numpy(kpt),
            "label": torch.tensor(s["label"], dtype=torch.long),
            "boundary": torch.from_numpy(np.asarray(s["boundary"], dtype=np.float32)),
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """DataLoader collate function."""
    kpts = torch.stack([b["keypoints"] for b in batch])       # (B, T, 24, 3)
    labels = torch.stack([b["label"] for b in batch])         # (B,)
    bounds = torch.stack([b["boundary"] for b in batch])      # (B, T)
    return {"keypoints": kpts, "labels": labels, "boundaries": bounds}


def load_pyskl_pickle(pkl_path: str) -> List[Dict]:
    """加载 pyskl 格式 pickle（当前为 stub，真实数据加载由 W12 实现）.

    Args:
        pkl_path: .pkl 文件路径

    Returns:
        List[Dict]，每个 Dict 含 keypoints (T,V,C), label, frame_dir 等
    """
    import pickle
    from pathlib import Path
    path = Path(pkl_path)
    if not path.exists():
        raise FileNotFoundError(f"Pickle 不存在: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


__all__ = [
    "ALL_BEHAVIORS_22",
    "NUM_CLASSES",
    "NUM_JOINTS",
    "make_synthetic_dataset",
    "save_synthetic_dataset",
    "load_pyskl_pickle",
    "STGCNBCDataset",
    "collate_fn",
]
