"""Synthetic ST-GCN+BC dataset tests — W11 TDD pre-code.

断言：
  A. 生成器产出张量形状正确 (T, 24, 3)
  B. 标签域 ⊆ 22 类清单（引用 assets-map §1）
  C. 随机种子确定性（同 seed 产出相同）
  D. 边界标签结构正确（首尾各 2 帧 = 1）
  E. 帧长分布合理（T 在 [20, 60] 区间）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# 权威 22 类清单（来自 assets-map §1，单一真相）
ALL_BEHAVIORS_22 = [
    "sit", "down", "stand", "heel", "sit_up", "stay", "bark", "bite",
    "track", "alert_sit", "alert_down", "apprehend", "escort", "obstacle",
    "recall", "watch", "guard", "release", "retrieve", "jump", "scale",
    "search_blind",
]
NUM_CLASSES = len(ALL_BEHAVIORS_22)  # 22
NUM_JOINTS = 24


@pytest.fixture
def gen():
    """延迟导入避免环境依赖（K9 仓未装时仍可跑冒烟测试）。"""
    try:
        from psd.data.synth_stgcn import make_synthetic_dataset  # noqa: F811
        return make_synthetic_dataset
    except ImportError as e:
        pytest.skip(f"synth_stgcn not yet implemented: {e}")


class TestSynthStgcnDataset:
    """合成数据生成器断言 suite."""

    def test_output_shape(self, gen):
        """A: 产出张量形状 (T, 24, 3)."""
        samples = gen(samples_per_class=2, T=30, noise_std=0.05, seed=0)
        assert len(samples) == NUM_CLASSES * 2
        for s in samples:
            kpt = s["keypoints"]
            assert kpt.shape == (30, NUM_JOINTS, 3), f"bad shape: {kpt.shape}"

    def test_label_domain(self, gen):
        """B: 标签域 ⊆ 22 类清单."""
        samples = gen(samples_per_class=5, T=30, seed=0)
        label_names = {s["label_name"] for s in samples}
        assert label_names.issubset(set(ALL_BEHAVIORS_22)), \
            f"unexpected labels: {label_names - set(ALL_BEHAVIORS_22)}"

    def test_deterministic_seed(self, gen):
        """C: 同 seed 产出相同."""
        s1 = gen(samples_per_class=3, T=30, seed=42)
        s2 = gen(samples_per_class=3, T=30, seed=42)
        for a, b in zip(s1, s2):
            assert np.array_equal(a["keypoints"], b["keypoints"])
            assert a["label"] == b["label"]
            assert a["label_name"] == b["label_name"]

    def test_boundary_labels(self, gen):
        """D: 首尾各 2 帧标记为 1."""
        samples = gen(samples_per_class=1, T=30, seed=0)
        b = samples[0]["boundary"]
        assert b.shape == (30,)
        assert b[0] == 1.0 and b[1] == 1.0, "首帧边界标签错误"
        assert b[28] == 1.0 and b[29] == 1.0, "尾帧边界标签错误"
        # 中间帧应全 0（合成 clip 无内部边界）
        assert np.all(b[2:28] == 0.0), "中间帧应有零边界标签"

    def test_frame_length_range(self, gen):
        """E: T 在 [20, 60] 区间."""
        for T in [20, 30, 60]:
            samples = gen(samples_per_class=1, T=T, seed=0)
            assert samples[0]["keypoints"].shape[0] == T

    def test_class_balance(self, gen):
        """每类样本数相等."""
        n_per_class = 7
        samples = gen(samples_per_class=n_per_class, T=30, seed=0)
        from collections import Counter
        counts = Counter(s["label_name"] for s in samples)
        assert all(c == n_per_class for c in counts.values()), \
            f"类别不平衡: {dict(counts)}"

    def test_keypoint_values_reasonable(self, gen):
        """关键点值在合理范围内（非 NaN/Inf，量级 O(1)）."""
        samples = gen(samples_per_class=1, T=30, noise_std=0.05, seed=0)
        kpt = samples[0]["keypoints"]
        assert not np.any(np.isnan(kpt)), "存在 NaN"
        assert not np.any(np.isinf(kpt)), "存在 Inf"
        assert np.abs(kpt).max() < 10.0, f"量级异常: max={np.abs(kpt).max()}"
