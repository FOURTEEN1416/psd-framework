"""ST-GCN+BC 模型组件测试.

TDD 前置：确认前向输出形状、损失有限、predict 正常。
Owner: W11 窗口
"""
import pytest
import torch


def test_stgcn_backbone_output_shape():
    """STGCN backbone 输出形状正确."""
    from psd.models.stgcn_bc_stgcn import STGCN
    model = STGCN(in_channels=3, base_channels=64, num_stages=10)
    x = torch.randn(2, 30, 24, 3)
    feat = model(x)
    # 输出 (B, M=1, C_out, T', V)
    assert feat.ndim == 5
    assert feat.shape[0] == 2
    assert feat.shape[1] == 1  # M=1
    assert feat.shape[3] < 30  # 时间下采样


def test_bc_head_forward():
    """BCHead 输入 (B, M, C, T', V) → (cls, boundary)."""
    from psd.models.stgcn_bc_bc_head import BCHead
    head = BCHead(in_channels=128, num_classes=22)
    x = torch.randn(4, 1, 128, 8, 24)
    cls_logits, boundary_logits = head(x)
    assert cls_logits.shape == (4, 22)
    assert boundary_logits.shape == (4, 8)


def test_loss_finite_and_decreasing():
    """联合损失有限且随训练可下降."""
    from psd.models.stgcn_bc_model import build_stgcn_bc
    model = build_stgcn_bc(num_classes=22)
    B, T, V, C = 8, 30, 24, 3
    x = torch.randn(B, T, V, C)
    labels = torch.randint(0, 22, (B,))
    # 用全 0 边界标签（合成数据中每段只有首尾 2 帧=1，中间 0）
    T_prime = 8
    boundaries = torch.zeros(B, T_prime)
    cls_logits, boundary_logits = model(x)
    loss_dict = model.compute_loss(cls_logits, boundary_logits, labels, boundaries)
    assert torch.isfinite(loss_dict["total"])
    assert loss_dict["cls"].item() > 0  # CrossEntropy 初始不为 0
    assert loss_dict["boundary"].item() >= 0


def test_build_stgcn_bc_factory():
    from psd.models.stgcn_bc_model import build_stgcn_bc
    model = build_stgcn_bc(num_classes=22, base_channels=32, num_stages=5)
    x = torch.randn(2, 30, 24, 3)
    cls_logits, boundary_logits = model(x)
    assert cls_logits.shape == (2, 22)


def test_num_classes_parameter():
    """num_classes 参数生效."""
    from psd.models.stgcn_bc_model import build_stgcn_bc
    model = build_stgcn_bc(num_classes=10)
    x = torch.randn(2, 30, 24, 3)
    cls_logits, _ = model(x)
    assert cls_logits.shape[1] == 10
