"""P0.1 InterPet4D 加载器行为契约测试（T2 严格 TDD）。

数据源：InterPet4D smal_npy/*.npz（K9 盘点路径，实测 (T,24,3) 骨架）。
契约对应 psd/data/interpet4d.py 的公开函数。
"""
from pathlib import Path

import numpy as np
import pytest

from psd.data.interpet4d import (
    build_label_index,
    is_valid_clip,
    load_clip,
    parse_clip_id,
    resample_to_fixed_t,
    to_ntu_view,
)

DATA_ROOT = Path(r"D:\Desktop\k9-training-system\data\interpet4d\smal_npy")

REAL_CLIP = DATA_ROOT / "interpet_dog01_p01_take01_ego_001.npz"


def test_parse_clip_id_dog():
    assert parse_clip_id("interpet_dog01_p01_take01_ego_001.npz") == "dog01"


def test_parse_clip_id_two_digits():
    assert parse_clip_id("interpet_dog12_p23_take10_ego_002") == "dog12"


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="InterPet4D 数据未挂载")
def test_load_clip_shapes():
    clip = load_clip(REAL_CLIP)
    kp = clip["kp_world"]
    w = clip["kp_weight"]
    assert kp.shape[1:] == (24, 3)
    assert kp.dtype == np.float32
    assert w.shape[1] == 24
    assert kp.shape[0] == w.shape[0] > 0
    assert "frame_idx" in clip


@pytest.mark.skipif(not REAL_CLIP.exists(), reason="InterPet4D 数据未挂载")
def test_resample_fixed_length():
    clip = load_clip(REAL_CLIP)
    t_orig = clip["kp_world"].shape[0]
    out = resample_to_fixed_t(clip["kp_world"], target_t=64)
    assert out.shape == (64, 24, 3)
    # 均匀重采样必须保留时间端点
    if t_orig >= 2:
        assert np.allclose(out[0], clip["kp_world"][0])
        assert np.allclose(out[-1], clip["kp_world"][-1])


def test_low_confidence_joints_zeroed():
    """kp_weight < conf_threshold 的关节数值必须置零（NTU 惯例）。"""
    t = 5
    rng = np.random.default_rng(0)
    kp = rng.normal(size=(t, 24, 3)).astype(np.float32)
    w = np.ones((t, 24), dtype=np.float32)
    w[:, 7] = 0.1  # 关节 7 低置信
    view = to_ntu_view(kp, weight=w, conf_threshold=0.5)
    # NTU 视图 (C,T,V,M)：低置信关节在所有帧的坐标应为 0
    assert view[:, :, 7, 0].max() == 0.0
    # 高置信关节不受影响（至少有非零值）
    assert np.abs(view[:, :, 0, 0]).max() > 0


def test_ntu_view_shape_and_dead_joint():
    t = 10
    rng = np.random.default_rng(1)
    kp = rng.normal(size=(t, 24, 3)).astype(np.float32)
    view = to_ntu_view(kp, weight=np.ones((t, 24), dtype=np.float32))
    assert view.shape == (3, t, 25, 1)
    assert view.dtype == np.float32
    # 第 25 槽位为死关节（恒等映射 0-23，槽 24 恒零）
    assert np.all(view[:, :, 24, :] == 0)


def test_ntu_view_normalized_finite():
    t = 30
    rng = np.random.default_rng(2)
    kp = (rng.normal(size=(t, 24, 3)) * 100).astype(np.float32)  # 米制大坐标
    w = np.ones((t, 24), dtype=np.float32)
    view = to_ntu_view(kp, weight=w, normalize=True)
    assert np.isfinite(view).all()
    # 去中心后幅度应有界（尺度归一化到 O(1)）
    assert np.abs(view).max() < 100.0


def test_is_valid_clip_rejects_nan():
    """全帧 NaN 的 clip（SMAL 拟合失败）必须被识别为无效。

    实测案例：interpet_dog06_p14_take01_ego_001 kp_world 100% NaN。
    """
    good = np.zeros((10, 24, 3), dtype=np.float32)
    assert is_valid_clip(good)
    bad = np.full((10, 24, 3), np.nan, dtype=np.float32)
    assert not is_valid_clip(bad)
    partial = good.copy()
    partial[0, 0, 0] = np.nan
    assert not is_valid_clip(partial)  # 任何非有限值都不可入训练


def test_build_label_index_thirteen_dogs():
    names = [f"interpet_dog{i:02d}_p01_take01_ego_00{i}" for i in range(13)]
    sample_names, labels, num_class = build_label_index(names)
    assert len(sample_names) == len(labels) == 13
    assert num_class == 13
    assert sorted(set(labels)) == list(range(13))
    assert labels[0] == build_label_index(names)[1][0]
