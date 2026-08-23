"""P0.4 Ω 轻量分类头测试（W10 窗口，先测后码）。

覆盖: 线性头/MLP 可分性收敛、predict_proba 概率归一、同种子确定性。
Φ 冻结原则: 头只吃预计算 embedding，本模块不触碰任何骨干前向。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.training.tcl_head import TorchHead


def _blob_data(n_per_class: int = 40, dim: int = 16, n_classes: int = 3, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=4.0, size=(n_classes, dim))
    X = np.vstack([centers[c] + rng.normal(size=(n_per_class, dim)) for c in range(n_classes)])
    y = np.repeat(np.arange(n_classes), n_per_class)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


HEAD_CFG = {"epochs": 60, "lr": 0.01, "weight_decay": 0.0, "batch_size": 32, "device": "cpu"}


@pytest.mark.parametrize("hidden_dim", [0, 8])
def test_head_fits_separable_blobs(hidden_dim):
    X, y = _blob_data()
    head = TorchHead(dim_in=X.shape[1], n_classes=3, hidden_dim=hidden_dim,
                     seed=42, **HEAD_CFG).fit(X, y)
    acc = (head.predict(X) == y).mean()
    assert acc > 0.9, f"可分数据上训练精度不足: hidden={hidden_dim} acc={acc:.3f}"


def test_predict_proba_normalized_and_shaped():
    X, y = _blob_data()
    head = TorchHead(dim_in=X.shape[1], n_classes=3, hidden_dim=8,
                     seed=42, **HEAD_CFG).fit(X, y)
    probs = head.predict_proba(X)
    assert probs.shape == (len(X), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_same_seed_deterministic():
    X, y = _blob_data()
    p1 = TorchHead(dim_in=X.shape[1], n_classes=3, hidden_dim=8,
                   seed=7, **HEAD_CFG).fit(X, y).predict_proba(X)
    p2 = TorchHead(dim_in=X.shape[1], n_classes=3, hidden_dim=8,
                   seed=7, **HEAD_CFG).fit(X, y).predict_proba(X)
    assert np.allclose(p1, p2, atol=1e-6)


def test_subset_of_classes_allowed():
    """池迭代中某类可能暂时无入池样本——头须允许类别子集训练。"""
    X, y = _blob_data()
    keep = y != 2  # 只用 0/1 两类训练
    head = TorchHead(dim_in=X.shape[1], n_classes=3, hidden_dim=8,
                     seed=42, **HEAD_CFG).fit(X[keep], y[keep])
    probs = head.predict_proba(X[keep])
    assert probs.shape == (int(keep.sum()), 3)
    # 未参与训练的列恒为 0 概率（argmax 不可能命中）
    assert probs[:, 2].max() == 0.0
