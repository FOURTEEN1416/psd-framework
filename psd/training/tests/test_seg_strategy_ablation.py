"""tab3 −无监督分割第三臂实验测试（W38 窗口，TDD 先行）.

覆盖 W34 入册设计的必测点:
  1. uniform_cut_segments 等段数均匀切分：无缝覆盖、段长差 ≤1、边界非法输入报错
  2. fixed_grid_segments 固定网格平铺（stride=window 变体）：整除/有余数/窗口超长
  3. evaluate_criterion 预注册判据操作化：
     均匀窗显著劣于 SMQ（方向门+幅度门）且 ≥ 随机 null 才构成边界增益消融

测试策略: 纯函数零依赖——不加载 checkpoint、不读数据盘，
SMQ 推理与蒙特卡洛 null 复用 psd.training.segment_iou 既有数学（本窗不重复实现）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import seg_strategy_ablation as ssa  # noqa: E402


# ---------------------------------------------------------------- uniform cut

@pytest.mark.parametrize("total,n", [(10, 1), (10, 3), (10, 10), (7, 3),
                                     (100, 7), (3258, 55), (2410, 46)])
def test_uniform_cut_full_coverage_contiguous(total: int, n: int) -> None:
    segs = ssa.uniform_cut_segments(total, n)
    assert len(segs) == n
    assert segs[0][0] == 0
    assert segs[-1][1] == total
    for (s1, e1), (s2, _) in zip(segs, segs[1:]):
        assert e1 == s2, "段间必须无缝衔接"


@pytest.mark.parametrize("total,n", [(10, 3), (7, 3), (100, 7), (3258, 55)])
def test_uniform_cut_lengths_differ_by_at_most_one(total: int, n: int) -> None:
    lens = [e - s for s, e in ssa.uniform_cut_segments(total, n)]
    assert max(lens) - min(lens) <= 1


def test_uniform_cut_single_segment() -> None:
    assert ssa.uniform_cut_segments(42, 1) == [(0, 42)]


def test_uniform_cut_even_split_exact() -> None:
    assert ssa.uniform_cut_segments(12, 4) == [(0, 3), (3, 6), (6, 9), (9, 12)]


@pytest.mark.parametrize("total,n", [(10, 0), (10, -2), (10, 11)])
def test_uniform_cut_invalid_n_raises(total: int, n: int) -> None:
    with pytest.raises(ValueError):
        ssa.uniform_cut_segments(total, n)


def test_uniform_cut_is_deterministic() -> None:
    assert ssa.uniform_cut_segments(97, 13) == ssa.uniform_cut_segments(97, 13)


# ---------------------------------------------------------------- fixed grid

def test_fixed_grid_exact_multiple() -> None:
    assert ssa.fixed_grid_segments(8, 4) == [(0, 4), (4, 8)]


def test_fixed_grid_with_remainder_tail() -> None:
    assert ssa.fixed_grid_segments(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_fixed_grid_window_larger_than_total() -> None:
    assert ssa.fixed_grid_segments(5, 16) == [(0, 5)]


@pytest.mark.parametrize("total,window", [(3258, 16), (2410, 16), (37, 16)])
def test_fixed_grid_full_coverage(total: int, window: int) -> None:
    segs = ssa.fixed_grid_segments(total, window)
    assert segs[0][0] == 0 and segs[-1][1] == total
    for (s1, e1), (s2, _) in zip(segs, segs[1:]):
        assert e1 == s2


@pytest.mark.parametrize("total,window", [(10, 0), (10, -4)])
def test_fixed_grid_invalid_window_raises(total: int, window: int) -> None:
    with pytest.raises(ValueError):
        ssa.fixed_grid_segments(total, window)


# ---------------------------------------------------------------- 判据评估器

def test_criterion_established_when_uniform_between() -> None:
    """均匀窗显著劣于 SMQ 且 ≥ 随机 null → 构成边界增益消融。"""
    smq = [0.50, 0.48, 0.52, 0.51]
    uni = [0.36, 0.35, 0.38, 0.35]
    null = [0.30, 0.31, 0.29, 0.32]
    c = ssa.evaluate_criterion(smq, uni, null)
    assert c["uniform_worse_than_smq"] is True
    assert c["uniform_ge_random_null"] is True
    assert c["boundary_gain_ablation_established"] is True


def test_criterion_fails_when_uniform_close_to_smq() -> None:
    smq = [0.50, 0.48, 0.52, 0.51]
    uni = [0.49, 0.47, 0.53, 0.52]  # 与 SMQ 在噪声量级内
    null = [0.30, 0.31, 0.29, 0.32]
    c = ssa.evaluate_criterion(smq, uni, null)
    assert c["uniform_worse_than_smq"] is False
    assert c["boundary_gain_ablation_established"] is False


def test_criterion_fails_when_uniform_below_null() -> None:
    smq = [0.50, 0.48, 0.52, 0.51]
    uni = [0.25, 0.24, 0.26, 0.25]  # 比随机还差
    null = [0.30, 0.31, 0.29, 0.32]
    c = ssa.evaluate_criterion(smq, uni, null)
    assert c["uniform_ge_random_null"] is False
    assert c["boundary_gain_ablation_established"] is False


def test_criterion_direction_gate_requires_supermajority() -> None:
    """仅 2/4 episode 方向占优 → 方向门不过，即使均值差大也不算显著劣于。"""
    smq = [0.40, 0.60, 0.40, 0.60]
    uni = [0.20, 0.80, 0.20, 0.80]  # 2 负 2 正，均值打平且方向分裂
    null = [0.30, 0.30, 0.30, 0.30]
    c = ssa.evaluate_criterion(smq, uni, null)
    assert c["direction_gate_pass"] is False
    assert c["uniform_worse_than_smq"] is False


def test_criterion_reports_gate_details() -> None:
    smq = [0.50, 0.48, 0.52, 0.51]
    uni = [0.36, 0.35, 0.38, 0.35]
    null = [0.30, 0.31, 0.29, 0.32]
    c = ssa.evaluate_criterion(smq, uni, null)
    assert c["n_episodes"] == 4
    assert c["direction_wins"] == 4
    assert c["mean_gap_smq_minus_uniform"] == pytest.approx(0.5025 - 0.36, abs=1e-6)


# ---------------------------------------------------------------- 指标封装

def test_evaluate_arm_returns_standard_metric_keys() -> None:
    gt = [(0, 100), (100, 200)]
    pred = [(0, 90), (90, 200)]
    m = ssa.evaluate_arm(pred, gt, gt_bounds=[100], tol=16)
    assert set(m) == {"mean_matched_iou", "seg_precision@0.5",
                      "seg_recall@0.5", "boundary_f1"}
    assert 0.0 <= m["mean_matched_iou"] <= 1.0
