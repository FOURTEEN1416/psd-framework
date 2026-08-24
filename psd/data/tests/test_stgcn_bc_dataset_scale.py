"""ST-GCN+BC 数据集扩量测试（n=100/类）。

TDD 前置：验证 100 样本/类时 shape、类别完整性、种子确定性。
Owner: W12 窗口
"""
import pytest
import numpy as np
import torch


def test_make_synthetic_dataset_100_per_class():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    samples = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    assert len(samples) == 22 * 100  # 2200 样本


def test_make_synthetic_dataset_100_shapes():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    samples = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    for s in samples:
        assert s["keypoints"].shape == (30, 24, 3), f"unexpected shape: {s['keypoints'].shape}"
        assert s["boundary"].shape == (30,)
        assert isinstance(s["label"], int)
        assert 0 <= s["label"] < 22
        assert isinstance(s["label_name"], str)
        assert isinstance(s["frame_dir"], str)


def test_make_synthetic_dataset_100_all_classes():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset, ALL_BEHAVIORS_22
    samples = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    names = {s["label_name"] for s in samples}
    assert names == set(ALL_BEHAVIORS_22)
    # 每类恰好 100 个
    from collections import Counter
    dist = Counter(s["label_name"] for s in samples)
    for cls in ALL_BEHAVIORS_22:
        assert dist[cls] == 100, f"{cls} has {dist[cls]} samples, expected 100"


def test_make_synthetic_dataset_100_deterministic():
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset
    s1 = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    s2 = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    for a, b in zip(s1, s2):
        np.testing.assert_array_equal(a["keypoints"], b["keypoints"])
        assert a["label"] == b["label"]
        assert a["label_name"] == b["label_name"]


def test_stgcnbc_dataset_from_100_samples():
    from psd.data.stgcn_bc_dataset import STGCNBCDataset, make_synthetic_dataset
    samples = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    ds = STGCNBCDataset(samples=samples, T=30, augment=False)
    assert len(ds) == 2200
    item = ds[0]
    assert isinstance(item["keypoints"], torch.Tensor)
    assert item["keypoints"].shape == (30, 24, 3)


def test_save_and_load_100_samples(tmp_path):
    from psd.data.stgcn_bc_dataset import make_synthetic_dataset, save_synthetic_dataset
    import pickle
    samples = make_synthetic_dataset(samples_per_class=100, T=30, seed=42)
    out = tmp_path / "syn_100_test.pkl"
    save_synthetic_dataset(samples, str(out))
    loaded = pickle.loads(out.read_bytes())
    assert len(loaded) == 2200
    assert loaded[0]["keypoints"].shape == (30, 24, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
