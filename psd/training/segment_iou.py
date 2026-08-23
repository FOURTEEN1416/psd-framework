"""P0.2 分割评估数学 owner：motion word 索引 → 分割段 → IoU 匹配。

协议（拼接式 episode，公开真实层口径）：
- 输入：SMQ VQ 层输出的逐帧 code 序列 indices（长度=T）
- 分割段：相邻帧同码合并为 run（VQ patch 网格使 run 长度必为 patch_size 整数倍），
  过滤短于 min_len 的段后作为预测分割
- 匹配：预测段 × GT 段 IoU 矩阵 + Hungarian 最大匹配（scipy）
- 随机基线：等段数随机切割的同协议期望 IoU（对照下界，类比 W3 kNN 随机基线纪律）
"""
from __future__ import annotations

import random

import numpy as np
from scipy.optimize import linear_sum_assignment


def indices_to_runs(indices: list[int] | np.ndarray) -> list[tuple[int, int, int]]:
    """逐帧 code → [(code, start, end)] 半开区间 runs。"""
    runs: list[tuple[int, int, int]] = []
    n = len(indices)
    if n == 0:
        return runs
    start = 0
    for i in range(1, n + 1):
        if i == n or indices[i] != indices[start]:
            runs.append((int(indices[start]), start, i))
            start = i
    return runs


def runs_to_segments(runs: list[tuple[int, int, int]], min_len: int = 1) -> list[tuple[int, int]]:
    """过滤短于 min_len 的 run，返回 [(start, end)]。"""
    return [(s, e) for _, s, e in runs if e - s >= min_len]


def segmentation_from_indices(
    indices: list[int] | np.ndarray, min_len: int = 1
) -> list[tuple[int, int]]:
    """code 序列 → 预测分割段。"""
    return runs_to_segments(indices_to_runs(indices), min_len=min_len)


def _iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match_segments(pred: list[tuple[int, int]], gt: list[tuple[int, int]]) -> dict:
    """Hungarian 最大总 IoU 匹配。

    返回 mean_matched_iou（配对均值）、pairs、seg_precision/recall@0.5。
    """
    if not pred or not gt:
        return {"mean_matched_iou": 0.0, "pairs": [],
                "seg_precision": 0.0, "seg_recall": 0.0}
    mat = np.array([[_iou(p, g) for g in gt] for p in pred])
    row, col = linear_sum_assignment(-mat)
    pairs = [(int(r), int(c), float(mat[r, c])) for r, c in zip(row, col)]
    matched_ious = [iou for _, _, iou in pairs]

    pred_hit = sum(1 for r, _, _ in pairs if mat[r].max() > 0.5)
    gt_hit = sum(1 for _, c, _ in pairs if mat[:, c].max() > 0.5)
    return {
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "pairs": pairs,
        "seg_precision": pred_hit / len(pred),
        "seg_recall": gt_hit / len(gt),
    }


def boundary_f1(
    pred_segs: list[tuple[int, int]], gt_bounds: list[int], tol: int
) -> tuple[float, float, float]:
    """内部边界 F1：预测段起点 vs GT 边界，最近邻容差匹配（不含 0 起点）。"""
    pred_bounds = sorted({s for s, _ in pred_segs if s > 0})
    tp = 0
    used = set()
    for g in gt_bounds:
        best, best_i = None, None
        for i, pb in enumerate(pred_bounds):
            if i in used:
                continue
            d = abs(pb - g)
            if d <= tol and (best is None or d < best):
                best, best_i = d, i
        if best_i is not None:
            used.add(best_i)
            tp += 1
    precision = tp / len(pred_bounds) if pred_bounds else 0.0
    recall = tp / len(gt_bounds) if gt_bounds else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def random_baseline_mean_iou(
    gt: list[tuple[int, int]],
    total_len: int,
    n_random_segs: int,
    n_sims: int = 200,
    seed: int = 0,
) -> float:
    """等段数随机切割的同协议期望 mean_matched_iou（蒙特卡洛对照下界）。"""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_sims):
        cuts = sorted(rng.sample(range(1, total_len), n_random_segs - 1))
        bounds = [0] + cuts + [total_len]
        pred = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        vals.append(match_segments(pred, gt)["mean_matched_iou"])
    return float(np.mean(vals))
