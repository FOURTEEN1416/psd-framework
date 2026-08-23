"""P0.2 契约测试：分割段提取 + IoU 匹配数学（RED→GREEN）。"""
import numpy as np
import pytest

from psd.training.segment_iou import (
    boundary_f1,
    indices_to_runs,
    match_segments,
    random_baseline_mean_iou,
    runs_to_segments,
    segmentation_from_indices,
)


class TestIndicesToRuns:
    def test_basic(self):
        runs = indices_to_runs([0, 0, 1, 1, 1, 2])
        assert runs == [(0, 0, 2), (1, 2, 5), (2, 5, 6)]

    def test_empty(self):
        assert indices_to_runs([]) == []


class TestRunsToSegments:
    def test_merge_adjacent_same_code(self):
        # 相邻 patch 同码 → 合并为一段
        idx = [7, 7, 7, 7, 7, 7]
        segs = segmentation_from_indices(idx, min_len=1)
        assert segs == [(0, 6)]

    def test_min_len_filter(self):
        runs = [(0, 0, 4), (1, 4, 5), (2, 5, 10)]
        segs = runs_to_segments(runs, min_len=2)
        assert segs == [(0, 4), (5, 10)]


class TestMatchSegments:
    def test_perfect(self):
        gt = [(0, 10), (10, 20)]
        res = match_segments(gt, gt)
        assert res["mean_matched_iou"] == pytest.approx(1.0)

    def test_known_partial_value(self):
        pred = [(0, 10), (10, 20)]
        gt = [(0, 15), (15, 20)]
        res = match_segments(pred, gt)
        # 最优匹配：(0,10)&(0,15) IoU=10/15；(10,20)&(15,20) 交 5 并 10 → 5/10
        assert res["mean_matched_iou"] == pytest.approx((10 / 15 + 5 / 10) / 2)

    def test_count_mismatch_no_crash(self):
        pred = [(0, 30)]
        gt = [(0, 10), (10, 20), (20, 30)]
        res = match_segments(pred, gt)
        assert 0.0 < res["mean_matched_iou"] <= 1.0


class TestBoundaryF1:
    def test_exact(self):
        pred = [(0, 50), (50, 100), (100, 150)]
        gt_bounds = [50, 100]
        p, r, f1 = boundary_f1(pred, gt_bounds, tol=5)
        assert (p, r, f1) == (1.0, 1.0, 1.0)

    def test_tolerance(self):
        pred = [(0, 52), (52, 104), (104, 150)]
        gt_bounds = [50, 100]
        _, _, f1 = boundary_f1(pred, gt_bounds, tol=5)
        assert f1 == 1.0  # 52/104 距 GT 边界 2/4 ≤ tol
        _, _, f1_far = boundary_f1(pred, gt_bounds, tol=1)
        assert f1_far == 0.0


class TestRandomBaseline:
    def test_below_perfect_and_stable(self):
        gt = [(0, 40), (40, 80), (80, 120)]
        base = random_baseline_mean_iou(gt, total_len=120, n_random_segs=3,
                                        n_sims=50, seed=0)
        assert 0.0 <= base < 0.6  # 随机切割不应接近完美
