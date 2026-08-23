"""P0.3 评估指标测试（W8 交接 Step 4，对齐 experiment-skeleton E3 + 统计协议）。"""
from __future__ import annotations

import numpy as np
import pytest

from psd.training.jia_metrics import (
    inject_label_noise,
    majority_class_baseline,
    nmi,
    purity,
    random_assignment_purity,
)


class TestPurity:
    def test_perfect(self):
        pred = np.array([0, 0, 1, 1])
        true = np.array(["a", "a", "b", "b"])
        assert purity(pred, true) == pytest.approx(1.0)

    def test_known_mixed_case(self):
        # 簇0={a,a,b}→多数a计2；簇1={b,b}→2；纯度=4/5
        pred = np.array([0, 0, 0, 1, 1])
        true = np.array(["a", "a", "b", "b", "b"])
        assert purity(pred, true) == pytest.approx(0.8)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            purity(np.array([0, 1]), np.array(["a"]))


class TestNMI:
    def test_perfect_clustering(self):
        true = np.array(["a", "a", "b", "b", "c", "c"])
        assert nmi(true.copy(), true) == pytest.approx(1.0)

    def test_symmetry(self):
        rng = np.random.default_rng(5)
        a = rng.integers(0, 3, 100)
        b = rng.integers(0, 4, 100)
        assert nmi(a, b) == pytest.approx(nmi(b, a))

    def test_independent_near_zero(self):
        # 大样本独立离散变量 NMI ≈ 0
        rng = np.random.default_rng(7)
        a = rng.integers(0, 2, 20000)
        b = rng.integers(0, 2, 20000)
        assert nmi(a, b) < 0.01

    def test_hand_computed_value(self):
        # U=V 时恒为 1（含非平凡划分）
        u = np.array([0, 0, 1])
        assert nmi(u, u) == pytest.approx(1.0)
        v = np.array([5, 5, 9])  # 标签值不同但划分相同
        assert nmi(u, v) == pytest.approx(1.0)


class TestBaselines:
    def test_random_assignment_purity_formula(self):
        priors = {"a": 0.5, "b": 0.5}
        assert random_assignment_purity(priors) == pytest.approx(0.5)
        priors = {"a": 0.7, "b": 0.2, "c": 0.1}
        assert random_assignment_purity(priors) == pytest.approx(0.49 + 0.04 + 0.01)

    def test_majority_baseline(self):
        labels = np.array([*"aaaab"])
        assert majority_class_baseline(labels) == pytest.approx(0.8)

    def test_priors_from_labels_consistent(self):
        labels = np.array([*"aaab"])
        from psd.training.jia_metrics import label_priors
        p = label_priors(labels)
        assert random_assignment_purity(p) == pytest.approx(0.625)


class TestNoiseInjection:
    def test_zero_rate_identity(self):
        labels = np.array([*"aabbcc"])
        out = inject_label_noise(labels, rate=0.0, seed=42)
        assert (out == labels).all()

    def test_full_rate_all_changed(self):
        labels = np.array([*"aabbcc"])
        out = inject_label_noise(labels, rate=1.0, seed=42)
        assert (out != labels).all()
        assert set(out.tolist()) <= set("abc")  # 仍在类别体系内（子集）

    def test_partial_rate_count(self):
        labels = np.array(["a"] * 50 + ["b"] * 50)
        out = inject_label_noise(labels, rate=0.2, seed=42)
        assert (out != labels).sum() == 20  # 恰 20%

    def test_deterministic_and_label_set_preserved(self):
        labels = np.array([*"aabbc" * 10])
        o1 = inject_label_noise(labels, 0.3, seed=1)
        o2 = inject_label_noise(labels, 0.3, seed=1)
        assert (o1 == o2).all()
        assert set(o1.tolist()) <= set(labels.tolist())

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            inject_label_noise(np.array(["a"]), rate=-0.1)
        with pytest.raises(ValueError):
            inject_label_noise(np.array(["a"]), rate=1.5)
