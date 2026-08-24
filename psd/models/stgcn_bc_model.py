"""ST-GCN+BC 整体模型: ST-GCN backbone + BC Head + 联合损失.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/model.py`（只读参考）
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from psd.models.stgcn_bc_bc_head import BCHead
from psd.models.stgcn_bc_k9_graph import K9Graph
from psd.models.stgcn_bc_labels import NUM_BEHAVIORS
from psd.models.stgcn_bc_loss import STGCNBCLoss
from psd.models.stgcn_bc_stgcn import STGCN


class STGCNBC(nn.Module):
    """ST-GCN+BC 整体模型.

    Args:
        in_channels: 输入通道数（3D=3, 2D+conf=3）
        num_classes: 类别数（默认 22）
        base_channels: ST-GCN 基础通道（默认 64）
        num_stages: ST-GCN block 数量（默认 10）
        tcn_type: 'mstcn' (ST-GCN++) 或 'unit_tcn' (原始 ST-GCN)
        adaptive: 邻接矩阵自适应模式
        boundary_weight: 边界损失权重（默认 0.3）
        dropout: 分类头 dropout
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = NUM_BEHAVIORS,
        base_channels: int = 64,
        num_stages: int = 10,
        tcn_type: str = "mstcn",
        adaptive: str = "importance",
        boundary_weight: float = 0.3,
        dropout: float = 0.0,
        graph: Optional[K9Graph] = None,
    ):
        super().__init__()
        self.backbone = STGCN(
            graph=graph,
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_stages,
            tcn_type=tcn_type,
            adaptive=adaptive,
        )
        self.head = BCHead(
            in_channels=self.backbone.out_channels,
            num_classes=num_classes,
            dropout=dropout,
        )
        self.loss_fn = STGCNBCLoss(boundary_weight=boundary_weight)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """输入 (B, T, V, C) → (cls_logits (B, num_classes), boundary_logits (B, T'))."""
        feat = self.backbone(x)
        cls_logits, boundary_logits = self.head(feat)
        return cls_logits, boundary_logits

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.loss_fn(cls_logits, boundary_logits, cls_labels, boundary_labels)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """推理接口."""
        cls_logits, boundary_logits = self.forward(x)
        cls_probs = torch.softmax(cls_logits, dim=-1)
        cls_pred = cls_probs.argmax(dim=-1)
        boundary_probs = torch.sigmoid(boundary_logits)
        boundary_pred = boundary_probs > 0.5
        return {
            "cls_probs": cls_probs,
            "cls_pred": cls_pred,
            "boundary_probs": boundary_probs,
            "boundary_pred": boundary_pred,
        }


def build_stgcn_bc(
    in_channels: int = 3,
    num_classes: int = NUM_BEHAVIORS,
    base_channels: int = 64,
    num_stages: int = 10,
    **kwargs,
) -> STGCNBC:
    """便捷构建函数."""
    return STGCNBC(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        num_stages=num_stages,
        **kwargs,
    )


__all__ = ["STGCNBC", "build_stgcn_bc"]
