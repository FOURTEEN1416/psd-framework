"""ST-GCN+BC 数据集 + 合成数据生成器测试.

TDD 前置：确认 make_synthetic_dataset 形状/类别/种子确定性。
Owner: W11 窗口
"""
import pytest
import numpy as np
import torch


def test_make_synthetic_dataset_basic():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    samples = make_synthetic_dataset(samples_per_class=3, T=30, seed=42)
    assert len(samples) == 22 * 3  # 22 classes × 3 samples


def test_make_synthetic_dataset_shapes():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    samples = make_synthetic_dataset(samples_per_class=5, T=30, seed=42)
    for s in samples:
        assert s["keypoints"].shape == (30, 24, 3)
        assert s["boundary"].shape == (30,)
        assert isinstance(s["label"], int)
        assert 0 <= s["label"] < 22
        assert isinstance(s["label_name"], str)
        assert isinstance(s["frame_dir"], str)


def test_make_synthetic_dataset_all_22_classes():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset, ALL_BEHAVIORS_22
    samples = make_synthetic_dataset(samples_per_class=2, T=30, seed=42)
    names = {s["label_name"] for s in samples}
    assert names == set(ALL_BEHAVIORS_22)


def test_make_synthetic_dataset_deterministic():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    s1 = make_synthetic_dataset(samples_per_class=2, T=30, seed=42)
    s2 = make_synthetic_dataset(samples_per_class=2, T=30, seed=42)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a["keypoints"], b["keypoints"])
        assert a["label"] == b["label"]


def test_make_synthetic_dataset_different_seed():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    s1 = make_synthetic_dataset(samples_per_class=2, T=30, seed=42)
    s2 = make_synthetic_dataset(samples_per_class=2, T=30, seed=99)
    # At least one sample should differ
    diff = False
    for a, b in zip(s1, s2):
        if not np.allclose(a["keypoints"], b["keypoints"]):
            diff = True
            break
    assert diff, "Different seeds should produce different data"


def test_save_and_load_synthetic_dataset(tmp_path):
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset, save_synthetic_dataset, load_pyskl_pickle
    import pickle, pathlib
    samples = make_synthetic_dataset(samples_per_class=2, T=30, seed=42)
    out = tmp_path / "syn_test.pkl"
    save_synthetic_dataset(samples, str(out))
    loaded = pickle.loads(out.read_bytes())
    assert len(loaded) == len(samples)
    assert loaded[0]["keypoints"].shape == (30, 24, 3)


def test_stgcnbc_dataset_from_samples():
    from psd.data.stgcn_bc_dataset import STGCNBCDataset, make_synthetic_dataset
    samples = make_synthetic_dataset(samples_per_class=4, T=30, seed=42)
    ds = STGCNBCDataset(samples=samples, T=30, augment=False)
    assert len(ds) == 88  # 22 × 4
    item = ds[0]
    assert isinstance(item["keypoints"], torch.Tensor)
    assert item["keypoints"].shape == (30, 24, 3)
    assert isinstance(item["label"], torch.Tensor)
    assert item["label"].item() == samples[0]["label"]


def test_stgcnbc_dataset_random_seed():
    """同一 seed 两次生成样本顺序一致."""
    from psd.data.stgcn_bc_dataset import STGCNBCDataset, make_synthetic_dataset
    s1 = make_synthetic_dataset(samples_per_class=3, T=30, seed=7)
    s2 = make_synthetic_dataset(samples_per_class=3, T=30, seed=7)
    ds1 = STGCNBCDataset(samples=s1, T=30)
    ds2 = STGCNBCDataset(samples=s2, T=30)
    for i in range(len(ds1)):
        np.testing.assert_array_equal(ds1[i]["keypoints"].numpy(), ds2[i]["keypoints"].numpy())
