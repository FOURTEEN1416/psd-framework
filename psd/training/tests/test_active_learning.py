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
    ALSimulationRunner,
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

# ---------------------------------------------------------------------------
# Task 3: 增量式 AL 模拟运行器
# ---------------------------------------------------------------------------

def _tiny_pool_and_val():
    """tiny 合成池/验证集（隔离 seed，避免 W12 seed42 记忆）。"""
    from psd.data.synth_stgcn import make_synthetic_dataset
    pool = make_synthetic_dataset(samples_per_class=2, T=10, noise_std=0.05, seed=20261)   # 44
    val = make_synthetic_dataset(samples_per_class=1, T=10, noise_std=0.05, seed=20262)    # 22
    return pool, val


class TestALSimulationRunner:
    def _make_runner(self, budgets=(4, 8)):
        from psd.models.stgcn_bc import build_stgcn_bc
        from psd.training.train_stgcn_bc import TrainConfig
        pool, val = _tiny_pool_and_val()
        model_fn = lambda: build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2)
        cfg = TrainConfig(
            epochs=2, batch_size=16, use_amp=False, device="cpu",
            warmup_epochs=0, early_stopping=False, save_interval=1000,
            output_dir="runs/_tmp_al_test",
        )
        return ALSimulationRunner(
            build_model=model_fn, pool_samples=pool, val_samples=val,
            train_config=cfg, budgets=budgets,
        )

    def test_incremental_rounds_nested_selections(self):
        """轨迹产出各预算点指标；labeled 集合嵌套增长。"""
        r = self._make_runner()
        result = r.run_trajectory(strategy="entropy", seed=42)
        assert set(result.keys()) == {4, 8}
        assert all(isinstance(v, float) and math.isfinite(v) for v in result.values())
        sel4 = set(r.selected_at(4))
        sel8 = set(r.selected_at(8))
        assert sel4 <= sel8 and len(sel8) == 8

    def test_paired_initial_core_identical_across_strategies(self):
        """同 seed 两臂初始核逐 id 相等（配对设计核心保证）。"""
        r1 = self._make_runner()
        r2 = self._make_runner()
        core_a = r1._initial_core(seed=42)
        core_b = r2._initial_core(seed=42)
        assert core_a == core_b

    def test_small_batch_no_empty_loader(self):
        """n_train < batch_size 时训练不产生空 loader（drop_last 规避）。"""
        r = self._make_runner(budgets=(3,))
        acc = r._train_stage(sample_ids=list(range(3)), init_seed=7)
        assert isinstance(acc, float) and math.isfinite(acc)

    def test_unknown_strategy_raises(self):
        r = self._make_runner()
        with pytest.raises(ValueError):
            r.run_trajectory(strategy="mc_dropout", seed=1)
