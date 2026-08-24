"""ST-GCN+BC 训练器测试.

TDD 前置：确认模型前向输出形状、损失有限可下降、训练一轮不炸。
Owner: W11 窗口
"""
import pytest
import torch
import numpy as np


def test_model_forward_shape():
    """前向输出形状 (B, 22) 和 (B, T')."""
    from psd.models.stgcn_bc_model import build_stgcn_bc
    model = build_stgcn_bc(num_classes=22)
    B, T, V, C = 4, 30, 24, 3
    x = torch.randn(B, T, V, C)
    cls_logits, boundary_logits = model(x)
    assert cls_logits.shape == (B, 22)
    assert boundary_logits.shape[0] == B
    assert boundary_logits.shape[-1] < T  # 时间维度因下采样缩短


def test_model_loss_finite():
    """损失有限且非 NaN."""
    from psd.models.stgcn_bc_model import build_stgcn_bc
    model = build_stgcn_bc(num_classes=22)
    B, T, V, C = 4, 30, 24, 3
    x = torch.randn(B, T, V, C)
    labels = torch.randint(0, 22, (B,))
    boundaries = torch.zeros(B, 8)  # T' ≈ 8 after downsampling
    cls_logits, boundary_logits = model(x)
    loss_dict = model.compute_loss(cls_logits, boundary_logits, labels, boundaries)
    assert torch.isfinite(loss_dict["total"]).all()
    assert loss_dict["total"].item() > 0


def test_model_predict():
    """推理接口返回正确形状."""
    from psd.models.stgcn_bc_model import build_stgcn_bc
    model = build_stgcn_bc(num_classes=22)
    model.eval()
    x = torch.randn(2, 30, 24, 3)
    result = model.predict(x)
    assert result["cls_probs"].shape == (2, 22)
    assert result["cls_pred"].shape == (2,)
    assert result["boundary_probs"].shape[0] == 2


def test_trainer_one_epoch_cpu():
    """CPU 小批量端到端一轮不炸."""
    from psd.models.stgcn_bc_model import build_stgcn_bc
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset, STGCNBCDataset
    from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig
    synth = make_synthetic_dataset(samples_per_class=8, T=30, seed=42)
    train_ds = STGCNBCDataset(samples=synth[:16], T=30, augment=False)
    val_ds = STGCNBCDataset(samples=synth[16:24], T=30, augment=False)
    model = build_stgcn_bc(num_classes=22)
    config = TrainConfig(
        epochs=1, batch_size=4, device="cpu", use_amp=False,
        early_stopping=False, output_dir="/tmp/w11_smoke_test",
    )
    trainer = STGCNBCTrainer(model, train_ds, val_ds, config=config)
    summary = trainer.fit()
    assert summary["total_epochs_trained"] == 1
    assert isinstance(summary["best_val_acc"], float)
    assert not np.isnan(summary["best_val_acc"])


def test_train_function_signature():
    """train(config_path) -> dict 接口存在."""
    from psd.training.train_stgcn_bc import train
    assert callable(train)
