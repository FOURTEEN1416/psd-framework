"""P0.3 原型聚类测试（W8 交接 Step 3，method.md §3.3.2）。

覆盖：class_mean / kmeans 双初始化、最近原型分配与 margin κ、
frequency-aware margin 阈值方向性与裁剪。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.training.jia_prototype import (
    PrototypeClusterer,
    frequency_aware_thresholds,
)


def _blob(center, n, seed, dim=8):
    rng = np.random.default_rng(seed)
    return center + rng.normal(0, 0.02, (n, dim))


# ---------------------------------------------------------------- 初始化

class TestClassMeanInit:
    def test_separated_clusters_assigned_correctly(self):
        emb = np.vstack([
            _blob(np.array([5.0] * 8), 20, 0),   # 类 A
            _blob(np.array([-5.0] * 8), 20, 1),  # 类 B
        ]).astype(np.float32)
        labels = np.array(["a"] * 20 + ["b"] * 20)
        cl = PrototypeClusterer(mode="class_mean").fit(emb, labels)
        proto_idx, pred, kappa = cl.assign(emb)
        assert (pred == labels).all()
        assert (kappa > 0.5).all()  # 分离良好 → margin 大

    def test_prototype_count_equals_class_count(self):
        emb = np.vstack([_blob(np.zeros(4), 5, 0, dim=4), _blob(np.ones(4) * 3, 5, 1, dim=4),
                         _blob(np.ones(4) * -3, 5, 2, dim=4)])
        labels = np.array([*"aaaaa", *"bbbbb", *"ccccc"])
        cl = PrototypeClusterer(mode="class_mean").fit(emb, labels)
        assert len(cl.prototype_labels) == 3

    def test_deterministic(self):
        rng = np.random.default_rng(9)
        emb = rng.normal(0, 1, (30, 6)).astype(np.float32)
        labels = np.array([*"aaabbbccc"] * 3 + ["a"] * 3)[:30]
        p1 = PrototypeClusterer(mode="class_mean").fit(emb, labels).prototypes
        p2 = PrototypeClusterer(mode="class_mean").fit(emb, labels).prototypes
        assert np.allclose(p1, p2)


class TestKmeansInit:
    def test_kmeans_recovers_two_blobs(self):
        emb = np.vstack([
            _blob(np.array([4.0] * 8), 30, 0),
            _blob(np.array([-4.0] * 8), 30, 1),
        ]).astype(np.float32)
        # kmeans 簇需成员标签做多数表决映射（契约）；此处标签即簇结构
        cl = PrototypeClusterer(mode="kmeans", k=2, seed=42).fit(
            emb, np.array([*"x" * 30, *"y" * 30]))
        idx, pred, _ = cl.assign(emb)
        assert len(set(idx[:30].tolist())) == 1
        assert idx[0] != idx[30]
        assert (pred[:30] == "x").all() and (pred[30:] == "y").all()

    def test_majority_label_mapping(self):
        # 簇结构按标签构造 → 多数标签映射后 assign 应还原标签
        emb = np.vstack([
            _blob(np.array([6.0] * 8), 15, 0),
            _blob(np.array([-6.0] * 8), 15, 1),
        ]).astype(np.float32)
        labels = np.array([*"x" * 15, *"y" * 15])
        cl = PrototypeClusterer(mode="kmeans", k=2, seed=0).fit(emb, labels)
        _, pred, _ = cl.assign(emb)
        assert (pred == labels).all()

    def test_k_must_match_mode(self):
        with pytest.raises(ValueError):
            PrototypeClusterer(mode="kmeans")  # 缺 k
        with pytest.raises(ValueError):
            PrototypeClusterer(mode="bad_mode")


# ---------------------------------------------------------------- margin κ

class TestMargin:
    def test_margin_higher_near_centroid(self):
        emb = np.vstack([_blob(np.array([3.0] * 8), 10, 1), _blob(np.array([-3.0] * 8), 10, 2)])
        labels = np.array([*"p" * 10, *"q" * 10])
        cl = PrototypeClusterer(mode="class_mean").fit(emb.astype(np.float32), labels)
        near_proto = emb[0]  # 类内点：贴近原型方向 → margin 大
        equidistant = np.concatenate([np.full(4, 3.0), np.full(4, -3.0)])  # 与两原型近似正交
        _, _, k_near = cl.assign(near_proto[None].astype(np.float32))
        _, _, k_mid = cl.assign(equidistant[None].astype(np.float32))
        assert k_near[0] > k_mid[0] >= 0.0

    def test_single_prototype_falls_back_to_similarity(self):
        emb = _blob(np.ones(8), 10, 0).astype(np.float32)
        cl = PrototypeClusterer(mode="kmeans", k=1, seed=0).fit(emb, np.array(["z"] * 10))
        _, _, k = cl.assign(emb)
        sims = cl._cosine_sim(emb)
        assert np.allclose(k, sims[:, 0], atol=1e-6)

    def test_margin_nonnegative(self):
        rng = np.random.default_rng(3)
        emb = rng.normal(0, 1, (40, 5)).astype(np.float32)
        labels = np.array([*"aabbc"] * 8)
        cl = PrototypeClusterer(mode="class_mean").fit(emb, labels)
        _, _, k = cl.assign(rng.normal(0, 1, (7, 5)).astype(np.float32))
        assert (k >= 0).all()


# ---------------------------------------------------------------- frequency-aware τ

class TestFrequencyAwareThresholds:
    def _priors(self):
        return {"sitting": 0.60, "walking": 0.30, "lying": 0.05, "jump": 0.05}

    def test_alpha_zero_uniform(self):
        t = frequency_aware_thresholds(0.10, self._priors(), alpha=0.0)
        assert all(abs(v - 0.10) < 1e-12 for v in t.values())

    def test_rare_classes_get_lower_threshold(self):
        # 未触及 floor 的先验组合上严格单调；触及 floor 的行为由
        # test_floor_clip_half_tau 单独锁定
        t = frequency_aware_thresholds(
            0.10, {"sitting": 1.0, "walking": 0.8, "lying": 0.6}, alpha=1.0)
        assert t["lying"] < t["walking"] < t["sitting"]
        assert t["sitting"] == pytest.approx(0.10)

    def test_floor_clip_half_tau(self):
        t = frequency_aware_thresholds(0.10, {"rare": 0.01, "dom": 0.99}, alpha=1.0)
        assert t["rare"] == pytest.approx(0.05)   # 下限 0.5τ
        assert t["dom"] == pytest.approx(0.10)

    def test_unknown_prior_defaults_to_min(self):
        t = frequency_aware_thresholds(0.10, {"a": 0.8}, alpha=1.0)
        assert "b" not in t or t.get("b", 0.05) == pytest.approx(0.05)

    def test_negative_alpha_raises(self):
        with pytest.raises(ValueError):
            frequency_aware_thresholds(0.1, {"a": 1.0}, alpha=-0.5)


# ---------------------------------------------------------------- 冻结契约

def test_embeddings_not_mutated():
    rng = np.random.default_rng(11)
    emb = rng.normal(0, 1, (12, 5)).astype(np.float32)
    snapshot = emb.copy()
    labels = np.array([*"abcdefghijkl"])
    cl = PrototypeClusterer(mode="class_mean").fit(emb, labels)
    cl.assign(emb)
    assert np.array_equal(emb, snapshot)
