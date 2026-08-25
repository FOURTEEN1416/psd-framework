# -*- coding: utf-8 -*-
"""W28/C4 — 合成保真度 v2: 分布统计 + 保真度指标 + 参数化拟合生成器.

任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §2-C4
领地: 本文件新增, 禁止修改 psd/data/synth_stgcn.py 行为.

设计要点:
    - 参考分布源参数化 (--reference-pkl), 当前 Q3b 产物为 n=1 冒烟残留
      (30 帧 × 17 关节 COCO 归一化坐标); 待全量产物落盘后零成本替换.
    - 逐关节角: 每关节由 (parent, self, child) 三点内角定义, 值域 [0, π],
      无方向角周期性问题; 端点/叶关节用祖父/孙关节补齐三元组.
    - 速度谱: 逐关节帧间位移幅值 ||Δp||.
    - 保真度指标: 逐关节 KS 距离 (两样本 D 统计量) + 速度直方图 L1 差.
    - 拟合: 逐关节逐坐标 AR(1) 平滑模型, 闭式解
          phi = 1 - Var(v) / (2 * sigma_pos^2),  s^2 = sigma_pos^2 (1 - phi^2)
      使生成序列的位置边缘方差与帧间速度方差同时匹配实测.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "COCO17_PARENT",
    "joint_angle_series",
    "speed_series",
    "ks_distance",
    "hist_l1_distance",
    "fidelity_metrics",
]

# ---------------------------------------------------------------------------
# 拓扑: COCO-17 (AK 公开真实层提点所用). parent 映射, 根关节 parent=None.
# 语义: 0 nose; 1/2 左右眼; 3/4 左右耳; 5/6 左右肩; 7/8 左右肘; 9/10 左右腕;
#       11/12 左右髋; 13/14 左右膝; 15/16 左右踝.
COCO17_PARENT: List[Optional[int]] = [
    5,     # 0 nose <- 左肩 (头部锚定)
    0, 0,  # 1/2 eye <- nose
    1, 2,  # 3/4 ear <- eye
    11,    # 5 左肩 <- 左髋
    12,    # 6 右肩 <- 右髋
    5, 6,  # 7/8 elbow <- shoulder
    7, 8,  # 9/10 wrist <- elbow
    5,     # 11 左髋 <- 左肩 (躯干闭环, 与 5 互为父子的环路在三点角中无害)
    6,     # 12 右髋 <- 右肩
    11, 12,  # 13/14 knee <- hip
    13, 14,  # 15/16 ankle <- knee
]


def _angle_triplets(v: int) -> tuple[int, int, int]:
    """关节 i 的 (a, b, c) 三元组: b 处由 a-b-c 构成的内角.

    规则: b=i; a=COCO17_PARENT[i]; c=首个以 i 为 parent 的关节.
    叶关节(耳/腕/踝)无子关节 → 三元组取 (祖父, 叶, 父), 在叶处观察
    肢体折叠程度: 伸直时内角→0, 弯曲时增大; 与中段关节的
    "三点张角"互补, 运动学意义明确且全覆盖 17 关节.
    """
    parent = COCO17_PARENT[v]
    if parent is None:
        # 根关节(nose): 由首个子关节与孙关节支撑 (eye-nose-ear)
        children = [k for k, p in enumerate(COCO17_PARENT) if p == v]
        if not children:
            raise ValueError(f"根关节 {v} 无法构造三点角")
        grandchild = next(
            (g for g, p in enumerate(COCO17_PARENT) if p == children[0]), None
        )
        if grandchild is None:
            raise ValueError(f"根关节 {v} 缺少孙关节")
        return (children[0], v, grandchild)
    children = [k for k, p in enumerate(COCO17_PARENT) if p == v]
    if children:
        return (parent, v, children[0])
    # 叶关节: 祖父-父-叶 三点角, 拐点观察位置在叶处
    grandparent = COCO17_PARENT[parent]
    if grandparent is None:
        raise ValueError(f"叶关节 {v} 的父为根且无子链支撑, 无法构造三点角")
    return (grandparent, v, parent)


def _angle_triplet_table(v_count: int) -> List[tuple[int, int, int]]:
    return [_angle_triplets(v) for v in range(v_count)]


def joint_angle_series(keypoints: np.ndarray) -> np.ndarray:
    """逐关节三点内角时间序列.

    Args:
        keypoints: (T, V, C>=2) xy 归一化坐标.

    Returns:
        (T, V) float64, 每列为一关节的内角弧度, 值域 [0, π].
        零长度骨骼(两关节投影重合)退化保护: 该处内角记 π/2.
    """
    kpts = np.asarray(keypoints, dtype=np.float64)
    T, V = kpts.shape[0], kpts.shape[1]
    out = np.full((T, V), np.pi / 2, dtype=np.float64)
    for v in range(V):
        a, b, c = _angle_triplets(v)
        ba = kpts[:, a, :2] - kpts[:, b, :2]
        bc = kpts[:, c, :2] - kpts[:, b, :2]
        denom = (
            np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
        )
        cosang = np.where(
            denom > 1e-12,
            (ba * bc).sum(-1) / np.maximum(denom, 1e-12),
            0.0,
        )
        out[:, v] = np.arccos(np.clip(cosang, -1.0, 1.0))
    return out


def speed_series(keypoints: np.ndarray) -> np.ndarray:
    """逐关节帧间速度幅值序列.

    Args:
        keypoints: (T, V, C>=2).

    Returns:
        (T-1, V) float64, s[t, j] = ||p[t+1, j] - p[t, j]|| (xy 分量).
    """
    kpts = np.asarray(keypoints, dtype=np.float64)[..., :2]
    return np.linalg.norm(np.diff(kpts, axis=0), axis=-1)


def ks_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """两样本 KS D 统计量: D = sup_x |F_a(x) - F_b(x)|."""
    a = np.sort(np.asarray(sample_a, dtype=np.float64).ravel())
    b = np.sort(np.asarray(sample_b, dtype=np.float64).ravel())
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def hist_l1_distance(
    sample_a: np.ndarray,
    sample_b: np.ndarray,
    bins: int = 20,
) -> float:
    """两样本直方图 L1 差: 共享 bin 边界, 归一化为频率后 sum|h_a - h_b|.

    两样本均为同一常数时分布恒等, 返回 0.
    """
    a = np.asarray(sample_a, dtype=np.float64).ravel()
    b = np.asarray(sample_b, dtype=np.float64).ravel()
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if not hi > lo:
        return 0.0
    edges = np.linspace(lo, hi, bins + 1)
    h_a = np.histogram(a, bins=edges)[0] / len(a)
    h_b = np.histogram(b, bins=edges)[0] / len(b)
    return float(np.abs(h_a - h_b).sum())


# ---------------------------------------------------------------------------
# 拟合与生成 (RED-2)
# ---------------------------------------------------------------------------

def fit_from_reference(reference_kpts: np.ndarray) -> Dict[str, object]:
    """从实测关键点序列拟合 AR(1) 生成参数.

    对每关节每坐标 x: mu / sigma_pos / Var(v) 三统计量 → 闭式解
        phi   = 1 - Var(v) / (2 sigma_pos^2)   (clamp 到 [0, 0.99])
        innov = sigma_pos * sqrt(1 - phi^2)
    使生成序列的位置边缘方差与帧间速度方差同时匹配实测.

    Args:
        reference_kpts: (N, T, V, C>=2); C==3 时第 3 维视为可见度,
            逐关节收集 bootstrap 池用于生成 conf 通道.

    Returns:
        params dict (见测试契约).
    """
    raise NotImplementedError("W28 RED: 待实现")


def make_synthetic_dataset_v2(
    params: Dict[str, object],
    samples_per_class: int = 10,
    classes: Optional[List[str]] = None,
    seed: int = 42,
) -> List[Dict]:
    """按拟合参数生成合成数据集 (接口与 v1 兼容, 字段含 generator 标识).

    时序模型: p_t = mu + phi * (p_{t-1} - mu) + innov * eps,
    首帧 p_0 ~ N(mu, sigma_pos^2).
    """
    raise NotImplementedError("W28 RED: 待实现")


def make_v1style_baseline_17j(
    samples_per_class: int = 10,
    classes: Optional[List[str]] = None,
    T: int = 30,
    noise_std: float = 0.05,
    seed: int = 42,
) -> List[Dict]:
    """对照基线: v1 方法论(姿态模板 + sin 波 + i.i.d. 高斯噪声)在
    17 关节归一化域的忠实移植.

    注意: 独立实现, 不 import/复用/修改 psd/data/synth_stgcn.py 的任何行为;
    存在目的仅为保真度对比中隔离"分布拟合"单一变量.
    """
    raise NotImplementedError("W28 RED: 待实现")


def fidelity_metrics(ref_angles: np.ndarray, syn_angles: np.ndarray,
                     ref_speed: np.ndarray, syn_speed: np.ndarray,
                     bins: int = 20) -> Dict[str, object]:
    """保真度指标汇总.

    Returns:
        {
          "ks_per_joint": list[float],        # 逐关节角度 KS
          "ks_mean": float,
          "vel_hist_per_joint": list[float],  # 逐关节速度直方图 L1 差
          "vel_hist_mean": float,
        }
    """
    ref_angles = np.asarray(ref_angles, dtype=np.float64)
    syn_angles = np.asarray(syn_angles, dtype=np.float64)
    ref_speed = np.asarray(ref_speed, dtype=np.float64)
    syn_speed = np.asarray(syn_speed, dtype=np.float64)
    v_count = ref_angles.shape[1]
    ks_per_joint = [
        ks_distance(ref_angles[:, j], syn_angles[:, j]) for j in range(v_count)
    ]
    vel_per_joint = [
        hist_l1_distance(ref_speed[:, j], syn_speed[:, j], bins=bins)
        for j in range(v_count)
    ]
    return {
        "ks_per_joint": ks_per_joint,
        "ks_mean": float(np.mean(ks_per_joint)),
        "vel_hist_per_joint": vel_per_joint,
        "vel_hist_mean": float(np.mean(vel_per_joint)),
    }
