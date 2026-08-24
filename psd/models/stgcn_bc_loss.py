"""ST-GCN+BC 联合损失: L = L_cls + 0.3 · L_boundary.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/loss.py`（只读参考）
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class STGCNBCLoss(nn.Module):
    """ST-GCN+BC 联合损失.

    L = L_cls + boundary_weight · L_boundary
    L_cls: CrossEntropyLoss（22 类分类）
    L_boundary: BCEWithLogitsLoss（边界检测，支持软标签）
    """

    def __init__(
        self,
        boundary_weight: float = 0.3,
        label_smoothing: float = 0.0,
        boundary_pos_weight: Optional[float] = None,
    ):
        super().__init__()
        self.boundary_weight = boundary_weight
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        if boundary_pos_weight is not None:
            pos_weight = torch.tensor([boundary_pos_weight])
            self.boundary_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.boundary_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        l_cls = self.cls_loss(cls_logits, cls_labels)
        if boundary_labels is None:
            return {
                "total": l_cls,
                "cls": l_cls,
                "boundary": torch.tensor(0.0, device=l_cls.device),
            }
        T_pred = boundary_logits.size(-1)
        T_label = boundary_labels.size(-1)
        if T_pred != T_label:
            boundary_labels = F.interpolate(
                boundary_labels.unsqueeze(1), size=T_pred, mode="linear", align_corners=False
            ).squeeze(1)
        l_boundary = self.boundary_loss(boundary_logits, boundary_labels)
        total = l_cls + self.boundary_weight * l_boundary
        return {"total": total, "cls": l_cls, "boundary": l_boundary}


__all__ = ["STGCNBCLoss"]
