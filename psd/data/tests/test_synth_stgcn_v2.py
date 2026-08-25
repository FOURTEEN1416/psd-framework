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


# ---------------------------------------------------------------------------
# 拟合与生成 (RED-2)
# ---------------------------------------------------------------------------

V = 17


def _make_ar1_reference(t_len: int = 400, seed: int = 7):
    """已知参数的 AR(1) 参考序列.

    Returns:
        kpts (1, T, V, 3), mu (V, 2), sigma_pos (V,), phi float.
    """
    rng = np.random.default_rng(seed)
    mu = np.stack(
        [np.linspace(0.35, 0.65, V), 0.3 + 0.03 * np.sin(np.arange(V))],
        axis=1,
    )
    sigma_pos = 0.02 + 0.001 * np.arange(V)
    phi = 0.8
    innov = (sigma_pos * np.sqrt(1.0 - phi**2))[:, None]
    p = np.zeros((t_len, V, 2))
    p[0] = mu + sigma_pos[:, None] * rng.standard_normal((V, 2))
    for t in range(1, t_len):
        p[t] = mu + phi * (p[t - 1] - mu) + innov * rng.standard_normal((V, 2))
    kpts = np.concatenate([p, np.ones((t_len, V, 1))], axis=-1)
    return kpts[None].astype(np.float64), mu, sigma_pos, phi


class TestFitFromReference:
    def test_param_contract(self):
        kpts, _, _, _ = _make_ar1_reference()
        prm = v2.fit_from_reference(kpts)
        for key in ("topology", "v", "clip_t", "mu", "sigma_pos",
                    "phi", "innov", "conf_pools"):
            assert key in prm, key
        assert prm["topology"] == "coco17"
        assert prm["v"] == V
        assert prm["clip_t"] == 400
        assert prm["mu"].shape == (V, 2)
        assert prm["sigma_pos"].shape == (V, 2)
        assert prm["phi"].shape == (V, 2)
        assert prm["innov"].shape == (V, 2)
        assert np.all(np.isfinite(prm["mu"]))
        assert np.all(prm["phi"] >= 0.0) and np.all(prm["phi"] <= 0.99)
        assert np.all(prm["innov"] > 0.0)
        # conf 池: 逐关节收集, 供生成通道 bootstrap
        assert len(prm["conf_pools"]) == V
        assert all(len(pool) == 400 for pool in prm["conf_pools"])

    def test_closed_form_recovers_ground_truth(self):
        kpts, mu, sigma_pos, phi = _make_ar1_reference(t_len=4000)
        prm = v2.fit_from_reference(kpts)
        assert np.allclose(prm["mu"], mu, atol=5e-3)
        assert np.allclose(prm["sigma_pos"], sigma_pos[:, None], rtol=0.15)
        assert np.abs(prm["phi"].mean() - phi) < 0.15


class TestMakeSyntheticDatasetV2:
    def test_output_contract(self):
        kpts, _, _, _ = _make_ar1_reference(t_len=60)
        prm = v2.fit_from_reference(kpts)
        samples = v2.make_synthetic_dataset_v2(
            prm, samples_per_class=3, classes=["watch", "track"], seed=11,
        )
        assert isinstance(samples, list) and len(samples) == 6
        s0 = samples[0]
        for key in ("keypoints", "label", "label_name", "boundary",
                    "frame_dir"):
            assert key in s0, key
        t_clip = int(prm["clip_t"])
        assert s0["keypoints"].shape == (t_clip, V, 3)
        assert np.all(np.isfinite(s0["keypoints"]))
        b = s0["boundary"]
        assert b.shape == (t_clip,)
        assert np.all(b[:2] == 1.0) and np.all(b[-2:] == 1.0)
        assert np.all(b[2:-2] == 0.0)
        assert s0["frame_dir"].startswith("synv2_")
        assert s0["label_name"] in ("watch", "track")
        assert s0.get("generator") == "synth_stgcn_v2"

    def test_reproducible_and_seed_sensitive(self):
        kpts, _, _, _ = _make_ar1_reference(t_len=60)
        prm = v2.fit_from_reference(kpts)
        a = v2.make_synthetic_dataset_v2(prm, samples_per_class=2, seed=5)
        b = v2.make_synthetic_dataset_v2(prm, samples_per_class=2, seed=5)
        c = v2.make_synthetic_dataset_v2(prm, samples_per_class=2, seed=6)
        assert np.array_equal(a[0]["keypoints"], b[0]["keypoints"])
        assert not np.array_equal(a[0]["keypoints"], c[0]["keypoints"])

    def test_generated_conf_bootstrapped_from_pool(self):
        kpts, _, _, _ = _make_ar1_reference(t_len=60)
        prm = v2.fit_from_reference(kpts)
        samples = v2.make_synthetic_dataset_v2(prm, samples_per_class=1,
                                               seed=3)
        pool = set(np.asarray(prm["conf_pools"][0]).tolist())
        gen_conf = samples[0]["keypoints"][..., 2]
        assert set(np.unique(gen_conf).tolist()).issubset(pool)


class TestV1StyleBaseline17j:
    def test_output_contract_and_independence_from_legacy(self):
        samples = v2.make_v1style_baseline_17j(samples_per_class=2,
                                               classes=["stay"], T=30,
                                               seed=9)
        assert len(samples) == 2
        s0 = samples[0]
        assert s0["keypoints"].shape == (30, V, 3)
        assert s0.get("generator") == "synth_v1style_17j"
        assert s0["frame_dir"].startswith("v1style17_")


class TestLegacySynthStgcnUntouched:
    """保护契约: psd/data/synth_stgcn.py 行为冻结, 任何漂移立即红."""

    LEGACY_MD5_SEED42_N1_T30 = "0d67fcaf225f6b899d2d3489350f69cf"

    def test_legacy_dataset_byte_fingerprint_frozen(self):
        import hashlib

        from psd.data.synth_stgcn import make_synthetic_dataset
        legacy = make_synthetic_dataset(samples_per_class=1, T=30, seed=42)
        k = legacy[0]["keypoints"]
        assert hashlib.md5(k.tobytes()).hexdigest() == self.LEGACY_MD5_SEED42_N1_T30
        assert k.shape == (30, 24, 3) and k.dtype == np.float32
        assert legacy[0]["frame_dir"] == "syn_sit_000"

    def test_legacy_module_exports_intact(self):
        from psd.data import synth_stgcn as legacy
        for name in ("ALL_BEHAVIORS_22", "NUM_CLASSES", "NUM_JOINTS",
                     "make_synthetic_dataset", "save_synthetic_dataset"):
            assert hasattr(legacy, name)


class TestFidelityAdvantage:
    """科学内核: 分布拟合使 v2 速度谱显著优于 v1 方法论基线."""

    def test_v2_speed_hist_beats_v1style(self):
        kpts, _, _, _ = _make_ar1_reference(t_len=120)
        prm = v2.fit_from_reference(kpts)
        v2_samples = v2.make_synthetic_dataset_v2(prm, samples_per_class=4,
                                                  classes=["watch"], seed=11)
        base_samples = v2.make_v1style_baseline_17j(
            samples_per_class=4, classes=["watch"], T=int(prm["clip_t"]),
            seed=11,
        )
        ref_k = kpts[0][..., :2]
        ref_ang = v2.joint_angle_series(ref_k)
        ref_spd = v2.speed_series(ref_k)

        v2_k = np.concatenate([s["keypoints"][..., :2] for s in v2_samples])
        base_k = np.concatenate(
            [s["keypoints"][..., :2] for s in base_samples]
        )
        m_v2 = v2.fidelity_metrics(ref_ang, v2.joint_angle_series(v2_k),
                                   ref_spd, v2.speed_series(v2_k))
        m_base = v2.fidelity_metrics(ref_ang,
                                     v2.joint_angle_series(base_k),
                                     ref_spd, v2.speed_series(base_k))
        assert m_v2["vel_hist_mean"] < m_base["vel_hist_mean"]
        assert m_v2["vel_hist_mean"] < 0.9  # 宽松绝对上限, 防双退化假阳性
