"""ST-GCN+BC BC 头（Boundary-Classification Head）.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/bc_head.py`（只读参考）
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BCHead(nn.Module):
    """Boundary-Classification Head（自研）."""

    def __init__(self, in_channels: int, num_classes: int = 22, boundary_kernel: int = 5, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.boundary_kernel = boundary_kernel
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc_cls = nn.Linear(in_channels, num_classes)
        pad = (boundary_kernel - 1) // 2
        self.conv_boundary = nn.Conv1d(in_channels, 1, kernel_size=boundary_kernel, padding=pad)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入 (B, M, C, T', V) → (cls_logits (B, num_classes), boundary_logits (B, T'))."""
        if x.ndim != 5:
            raise ValueError(f"输入应为 (B, M, C, T', V), 实际 {x.shape}")
        B, M, C, T_p, V = x.shape
        feat = x.mean(dim=(1, 3, 4))  # (B, C)
        feat = self.dropout(feat)
        cls_logits = self.fc_cls(feat)  # (B, num_classes)
        feat_t = x.mean(dim=4).mean(dim=1)  # (B, C, T')
        boundary_logits = self.conv_boundary(feat_t).squeeze(1)  # (B, T')
        return cls_logits, boundary_logits


def generate_boundary_labels(total_frames: int, segment_starts: list, segment_ends: list, sigma_ratio: float = 0.05) -> torch.Tensor:
    """生成帧级边界软标签（高斯环绕 start/end）."""
    if len(segment_starts) != len(segment_ends):
        raise ValueError("segment_starts 与 segment_ends 长度不一致")
    sigma = max(1.0, total_frames * sigma_ratio)
    t = np.arange(total_frames, dtype=np.float32)
    labels = np.zeros(total_frames, dtype=np.float32)
    for s, e in zip(segment_starts, segment_ends):
        if s >= 0 and s < total_frames:
            labels = np.maximum(labels, np.exp(-((t - s) ** 2) / (2 * sigma ** 2)))
        if e >= 0 and e < total_frames:
            labels = np.maximum(labels, np.exp(-((t - e) ** 2) / (2 * sigma ** 2)))
    return torch.from_numpy(labels)


__all__ = ["BCHead", "generate_boundary_labels"]
