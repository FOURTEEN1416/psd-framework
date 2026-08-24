"""ST-GCN+BC 损失函数测试.

TDD 前置：确认 loss 计算正确、边界对齐插值有效。
Owner: W11 窗口
"""
import pytest
import torch


def test_loss_cls_only():
    from psd.models.stgcn_bc_loss import STGCNBCLoss
    loss_fn = STGCNBCLoss(boundary_weight=0.3)
    B, C = 4, 22
    cls_logits = torch.randn(B, C)
    cls_labels = torch.randint(0, C, (B,))
    result = loss_fn(cls_logits, torch.zeros(B, 8), cls_labels, None)
    assert "total" in result
    assert "cls" in result
    assert "boundary" in result
    assert torch.isfinite(result["total"])
    assert result["boundary"].item() == 0.0


def test_loss_with_boundary():
    from psd.models.stgcn_bc_loss import STGCNBCLoss
    loss_fn = STGCNBCLoss(boundary_weight=0.3)
    B, C = 4, 22
    cls_logits = torch.randn(B, C)
    cls_labels = torch.randint(0, C, (B,))
    boundary_logits = torch.randn(B, 8)
    boundary_labels = torch.rand(B, 8)  # 软标签 [0,1]
    result = loss_fn(cls_logits, boundary_logits, cls_labels, boundary_labels)
    assert torch.isfinite(result["total"])
    assert result["total"] == result["cls"] + 0.3 * result["boundary"]


def test_loss_time_alignment():
    """boundary_logits 比 labels 短时，loss 做插值对齐."""
    from psd.models.stgcn_bc_loss import STGCNBCLoss
    loss_fn = STGCNBCLoss(boundary_weight=0.3)
    B, C = 2, 22
    cls_logits = torch.randn(B, C)
    cls_labels = torch.randint(0, C, (B,))
    boundary_logits = torch.randn(B, 4)   # 较短
    boundary_labels = torch.rand(B, 8)    # 较长
    result = loss_fn(cls_logits, boundary_logits, cls_labels, boundary_labels)
    assert torch.isfinite(result["total"])
