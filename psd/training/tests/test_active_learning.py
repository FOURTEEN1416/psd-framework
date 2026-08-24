"""P0.5 主动学习效率实验单元测试 — W14 窗口.

覆盖: 熵/随机采样器、增量模拟运行器、真实池打分。
口径: 合成层（主曲线）/ 公开真实层（池打分仅排序清单）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from psd.training.active_learning import (
    EntropySelector,
    RandomSelector,
    entropy_scores,
)


# ---------------------------------------------------------------------------
# Task 1: 熵打分与采样器
# ---------------------------------------------------------------------------

class TestEntropyScores:
    def test_uniform_max_onehot_zero(self):
        """均匀分布熵 = log(C)（最大），one-hot 熵 = 0（最小）。"""
        c = 22
        uniform = np.full((5, c), 1.0 / c, dtype=np.float64)
        onehot = np.zeros((3, c), dtype=np.float64)
        onehot[:, 0] = 1.0

        s_uniform = entropy_scores(uniform)
        s_onehot = entropy_scores(onehot)

        assert s_uniform.shape == (5,)
        assert np.allclose(s_uniform, math.log(c), atol=1e-6)
        assert s_onehot.shape == (3,)
        assert np.allclose(s_onehot, 0.0, atol=1e-6)

    def test_monotonic_with_confidence(self):
        """置信度越高熵越低：p=[0.9,0.1] < p=[0.6,0.4]。"""
        probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        s = entropy_scores(probs)
        assert s[0] < s[1]


class TestRandomSelector:
    def test_deterministic_given_rng(self):
        """同 rng 状态两次选择结果一致（可复现）。"""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        sel_a = RandomSelector().select(pool_size=50, k=10, rng=rng_a)
        sel_b = RandomSelector().select(pool_size=50, k=10, rng=rng_b)
        assert sel_a == sel_b

    def test_excludes_labeled_and_within_budget(self):
        """选择不含已标注 id，数量正确，范围合法。"""
        labeled = {0, 1, 2}
        # RandomSelector.select 接受 exclude 集合
        sel = RandomSelector().select(pool_size=20, k=7, rng=np.random.default_rng(0), exclude=labeled)
        assert len(sel) == 7
        assert len(set(sel)) == 7  # 无重复
        assert not (set(sel) & labeled)
        assert all(0 <= i < 20 for i in sel)


class TestEntropySelector:
    def test_picks_highest_entropy_first(self):
        """选中的是熵最大的样本（排除已标注后）。"""
        scores = np.array([0.1, 2.5, 0.3, 9.9, 1.0])
        labeled = [2]
        sel = EntropySelector().select(scores=scores, exclude=set(labeled), k=2)
        assert len(sel) == 2
        # 排除 idx2 后熵 top-2 是 idx3(9.9), idx1(2.5)，且按分降序返回
        assert sel[0] == 3
        assert sel[1] == 1

    def test_k_capped_by_available(self):
        """k 超过剩余可用量时截断到可用量。"""
        scores = np.array([0.1, 2.5, 0.3])
        sel = EntropySelector().select(scores=scores, exclude={0}, k=10)
        assert sorted(sel) == [1, 2]
