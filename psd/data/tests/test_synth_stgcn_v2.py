# -*- coding: utf-8 -*-
"""W28/C4 TDD — synth_stgcn_v2 测试.

RED 先行纪律: 每个测试先看失败(非 error), 再最小实现.
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.data import synth_stgcn_v2 as v2


# ---------------------------------------------------------------------------
# 统计层
# ---------------------------------------------------------------------------

def _layout_right_angle_at_elbow() -> np.ndarray:
    """关节 7(肘) 处三点成直角的布局: 5(肩)->7(肘)->9(腕).

    其余关节放在无害位置(单位方格内), 保证全序列可计算.
    """
    T = 4
    kpts = np.full((T, 17, 2), 0.5, dtype=np.float64)
    kpts[:, 5, :] = (0.4, 0.3)   # 左肩
    kpts[:, 7, :] = (0.5, 0.3)   # 左肘 (顶点)
    kpts[:, 9, :] = (0.5, 0.4)   # 左腕 -> 肘处内角 90°
    return kpts


class TestJointAngleSeries:
    def test_shape_and_range(self):
        rng = np.random.default_rng(0)
        kpts = rng.uniform(0, 1, size=(6, 17, 2))
        angles = v2.joint_angle_series(kpts)
        assert angles.shape == (6, 17)
        assert np.all(np.isfinite(angles))
        assert np.all(angles >= -1e-12) and np.all(angles <= np.pi + 1e-12)

    def test_known_right_angle(self):
        kpts = _layout_right_angle_at_elbow()
        angles = v2.joint_angle_series(kpts)
        assert angles[0, 7] == pytest.approx(np.pi / 2, abs=1e-9)

    def test_known_straight_angle(self):
        kpts = _layout_right_angle_at_elbow()
        kpts[:, 9, :] = (0.6, 0.3)  # 与肩共线 -> 平角 π
        angles = v2.joint_angle_series(kpts)
        assert angles[0, 7] == pytest.approx(np.pi, abs=1e-9)

    def test_static_sequence_constant_angles(self):
        kpts = _layout_right_angle_at_elbow()
        angles = v2.joint_angle_series(kpts)
        assert np.allclose(angles, angles[0:1])


class TestSpeedSeries:
    def test_shape_and_uniform_motion(self):
        T, V = 8, 17
        t = np.arange(T, dtype=np.float64)
        kpts = np.zeros((T, V, 2))
        kpts[..., 0] = 0.01 * t[:, None]  # 匀速 x 方向平移
        spd = v2.speed_series(kpts)
        assert spd.shape == (T - 1, V)
        assert np.allclose(spd, 0.01)

    def test_zero_speed_for_static(self):
        kpts = np.zeros((5, 17, 2))
        spd = v2.speed_series(kpts)
        assert np.all(spd == 0.0)


class TestKsDistance:
    def test_identical_is_zero(self):
        a = np.array([1.0, 2.0, 3.0])
        assert v2.ks_distance(a, a.copy()) == pytest.approx(0.0)

    def test_disjoint_is_one(self):
        a = np.arange(10.0)
        b = np.arange(10.0, 20.0)
        assert v2.ks_distance(a, b) == pytest.approx(1.0)

    def test_hand_computed_overlap(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert v2.ks_distance(a, b) == pytest.approx(1.0 / 3.0)


class TestHistL1:
    def test_identical_is_zero(self):
        a = np.array([0.1, 0.2, 0.3, 0.4])
        assert v2.hist_l1_distance(a, a.copy(), bins=10) == pytest.approx(0.0)

    def test_disjoint_is_two(self):
        a = np.array([0.0, 0.1, 0.2, 0.3])       # 全部落在低端 bin
        b = np.array([100.0, 100.1, 100.2, 100.3])  # 高端 bin
        assert v2.hist_l1_distance(a, b, bins=20) == pytest.approx(2.0)


class TestFidelityMetrics:
    def test_structure_and_self_identity(self):
        rng = np.random.default_rng(1)
        ang = rng.uniform(0, np.pi, size=(30, 17))
        spd = rng.uniform(0, 0.05, size=(29, 17))
        m = v2.fidelity_metrics(ang, ang.copy(), spd, spd.copy())
        assert set(m) == {"ks_per_joint", "ks_mean",
                          "vel_hist_per_joint", "vel_hist_mean"}
        assert len(m["ks_per_joint"]) == 17
        assert len(m["vel_hist_per_joint"]) == 17
        assert m["ks_mean"] == pytest.approx(0.0)
        assert m["vel_hist_mean"] == pytest.approx(0.0)
