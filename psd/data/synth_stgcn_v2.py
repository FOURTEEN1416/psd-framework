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
    "fit_from_reference",
    "make_synthetic_dataset_v2",
    "make_v1style_baseline_17j",
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
        params dict:
            topology: "coco17"
            v / clip_t: 关节数 / 参考 clip 帧数
            mu, sigma_pos: (V, 2) 位置均值与边缘 std
            phi, innov: (V, 2) AR(1) 系数与新息 std
            conf_pools: list[V] of np.ndarray 或 None
    """
    kpts = np.asarray(reference_kpts, dtype=np.float64)
    if kpts.ndim == 3:
        kpts = kpts[None]
    n, t_len, v_count, c_dim = kpts.shape
    xy = kpts[..., :2]

    pos = xy.reshape(-1, v_count, 2)                     # 样本内池化位置
    vel = np.diff(xy, axis=1).reshape(-1, v_count, 2)    # 仅样本内帧间差分
    mu = pos.mean(axis=0)
    var_pos = pos.var(axis=0)
    var_vel = vel.var(axis=0)

    eps = 1e-12
    phi = np.where(
        var_pos > eps, 1.0 - var_vel / (2.0 * var_pos + eps), 0.0
    )
    phi = np.clip(phi, 0.0, 0.99)
    innov = np.sqrt(np.maximum(var_pos * (1.0 - phi**2), 0.0))

    conf_pools = None
    if c_dim >= 3:
        # 通道布局: x(0), y(1), 可见度/置信度(2)
        conf_pools = [kpts[:, :, j, 2].ravel() for j in range(v_count)]

    return {
        "topology": "coco17",
        "v": v_count,
        "clip_t": t_len,
        "mu": mu,
        "sigma_pos": np.sqrt(var_pos),
        "phi": phi,
        "innov": innov,
        "conf_pools": conf_pools,
    }


def _default_classes() -> List[str]:
    """默认类清单: 只读引用旧模块常量 (不触碰其任何行为)."""
    from psd.data.synth_stgcn import ALL_BEHAVIORS_22

    return list(ALL_BEHAVIORS_22)


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
    rng = np.random.default_rng(seed)
    class_names = classes if classes is not None else _default_classes()
    v_count: int = int(params["v"])
    t_clip: int = int(params["clip_t"])
    mu = np.asarray(params["mu"], dtype=np.float64)
    sigma_pos = np.asarray(params["sigma_pos"], dtype=np.float64)
    phi = np.asarray(params["phi"], dtype=np.float64)
    innov = np.asarray(params["innov"], dtype=np.float64)
    pools = params.get("conf_pools")

    samples: List[Dict] = []
    for ci, cname in enumerate(class_names):
        for i in range(samples_per_class):
            p = np.empty((t_clip, v_count, 2), dtype=np.float64)
            p[0] = mu + sigma_pos * rng.standard_normal((v_count, 2))
            for t in range(1, t_clip):
                p[t] = (
                    mu
                    + phi * (p[t - 1] - mu)
                    + innov * rng.standard_normal((v_count, 2))
                )
            if pools is not None:
                conf = np.stack(
                    [
                        rng.choice(np.asarray(pools[j]), size=t_clip)
                        for j in range(v_count)
                    ],
                    axis=1,
                )
            else:
                conf = np.ones((t_clip, v_count))
            kpts = np.concatenate(
                [p, conf[..., None]], axis=-1
            ).astype(np.float32)

            boundary = np.zeros(t_clip, dtype=np.float32)
            boundary[:2] = 1.0
            boundary[-2:] = 1.0

            samples.append({
                "keypoints": kpts,
                "label": ci,
                "label_name": cname,
                "boundary": boundary,
                "frame_dir": f"synv2_{cname}_{i:03d}",
                "generator": "synth_stgcn_v2",
            })
    return samples


# 17 关节归一化域四足站姿模板 (侧视, 头朝上/前肢上端为肩)
_V17_STAND_NORM = np.array([
    [0.50, 0.62],   # 0 nose
    [0.46, 0.66], [0.54, 0.66],   # 1/2 eyes
    [0.44, 0.68], [0.56, 0.68],   # 3/4 ears
    [0.42, 0.45], [0.58, 0.45],   # 5/6 shoulders
    [0.40, 0.55], [0.60, 0.55],   # 7/8 elbows
    [0.38, 0.65], [0.62, 0.65],   # 9/10 wrists
    [0.44, 0.28], [0.56, 0.28],   # 11/12 hips
    [0.43, 0.15], [0.57, 0.15],   # 13/14 knees
    [0.42, 0.04], [0.58, 0.04],   # 15/16 ankles
], dtype=np.float64)


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
    参数字面值沿用 v1 (noise_std=0.05, sin 摆幅 0.05*0.1).
    """
    rng = np.random.default_rng(seed)
    class_names = classes if classes is not None else _default_classes()
    sin_amp = 0.005  # v1: sin_amp(0.05) × 全局系数(0.1), 归一化域同量级
    v_count = len(_V17_STAND_NORM)

    samples: List[Dict] = []
    for ci, cname in enumerate(class_names):
        for i in range(samples_per_class):
            phase = rng.uniform(0, 2 * np.pi)
            t_axis = np.arange(T, dtype=np.float64) / max(T - 1, 1)
            wave = sin_amp * np.sin(2 * np.pi * t_axis + phase)

            kpts = np.zeros((T, v_count, 3), dtype=np.float32)
            kpts[..., :2] = _V17_STAND_NORM[None, :, :] + wave[:, None, None]
            for c in range(2):
                kpts[..., c] += rng.normal(
                    0.0, noise_std, size=(T, v_count)
                ).astype(np.float32)
            kpts[..., 2] = 1.0

            boundary = np.zeros(T, dtype=np.float32)
            boundary[:2] = 1.0
            boundary[-2:] = 1.0

            samples.append({
                "keypoints": kpts,
                "label": ci,
                "label_name": cname,
                "boundary": boundary,
                "frame_dir": f"v1style17_{cname}_{i:03d}",
                "generator": "synth_v1style_17j",
            })
    return samples


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


# ---------------------------------------------------------------------------
# 实验入口 (GREEN-3): 当次运行证据 JSON
# ---------------------------------------------------------------------------

import argparse  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

# gate4 类清单: 与 Q3b 参考 (partialclass4 manifest) 同口径
GATE4_CLASSES = ["stay", "track", "watch", "jump"]


def _load_reference_pkl(path: str) -> np.ndarray:
    """加载 Q3b 风格参考 pkl → (N, T, V, C), 非 coco17 fail-fast."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    clips = [np.asarray(d["keypoints"], dtype=np.float64) for d in data]
    kpts = np.stack(clips)
    if kpts.shape[2] != 17:
        raise ValueError(
            f"参考拓扑为 {kpts.shape[2]} 关节, 本模块仅支持 coco17(17 关节); "
            f"24 关节(dog-pose/K9Graph)参考需另行适配拓扑映射"
        )
    return kpts


def _stack_joint_series(kpts_xy: np.ndarray, fn) -> np.ndarray:
    """对每条 clip 独立计算序列后沿时间轴拼接 → (ΣT_i, V)."""
    return np.concatenate([fn(kpts_xy[i]) for i in range(len(kpts_xy))],
                          axis=0)


def run_fidelity_experiment(
    reference_pkl: str,
    output_json: str,
    samples_per_class: int = 8,
    classes: Optional[List[str]] = None,
    seed: int = 42,
    bins: int = 20,
) -> Dict[str, object]:
    """C4 主实验: 拟合实测分布 → v1style/v2 双路生成 → 保真度对比证据.

    三层口径: 本实验属【合成层自证】(syn 数据 vs 公开真实层参考的分布
    距离), 不产生任何行为识别精度数字, 不得与训练指标混排.
    """
    kpts = _load_reference_pkl(reference_pkl)
    params = fit_from_reference(kpts)
    class_names = classes if classes is not None else list(GATE4_CLASSES)

    v2_samples = make_synthetic_dataset_v2(
        params, samples_per_class=samples_per_class,
        classes=class_names, seed=seed,
    )
    base_samples = make_v1style_baseline_17j(
        samples_per_class=samples_per_class, classes=class_names,
        T=int(params["clip_t"]), seed=seed,
    )

    ref_xy = kpts[..., :2]
    ref_ang = _stack_joint_series(ref_xy, joint_angle_series)
    ref_spd = _stack_joint_series(ref_xy, speed_series)

    syn_xy = np.concatenate(
        [s["keypoints"][..., :2].astype(np.float64) for s in v2_samples]
    )
    base_xy = np.concatenate(
        [s["keypoints"][..., :2].astype(np.float64) for s in base_samples]
    )
    m_syn = fidelity_metrics(ref_ang, joint_angle_series(syn_xy),
                             ref_spd, speed_series(syn_xy), bins=bins)
    m_base = fidelity_metrics(ref_ang, joint_angle_series(base_xy),
                              ref_spd, speed_series(base_xy), bins=bins)

    try:
        import subprocess

        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
    except Exception:
        git_sha = None

    evidence = {
        "meta": {
            "experiment": "w28-c4-synth-fidelity",
            "reference_path": str(reference_pkl),
            "n_ref_clips": int(kpts.shape[0]),
            "ref_shape": [int(x) for x in kpts.shape],
            "reference_topology": str(params["topology"]),
            "samples_per_class": samples_per_class,
            "classes": class_names,
            "seed": seed,
            "bins": bins,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_sha": git_sha,
            "caliber_note": (
                "合成层自证: syn_v2/v1style 生成数据 vs 公开真实层(AK 提点)"
                "参考分布的距离; 无识别精度数字; 与训练指标严禁混报"
            ),
        },
        "reference_stats": {
            "mu_per_joint": params["mu"].tolist(),
            "sigma_pos_per_joint": params["sigma_pos"].tolist(),
            "phi_per_joint": params["phi"].tolist(),
            "innov_per_joint": params["innov"].tolist(),
            "phi_mean": float(np.mean(params["phi"])),
            "sigma_pos_mean": float(np.mean(params["sigma_pos"])),
        },
        "v1style": {
            "ks_per_joint": m_base["ks_per_joint"],
            "ks_mean": m_base["ks_mean"],
            "vel_hist_per_joint": m_base["vel_hist_per_joint"],
            "vel_hist_mean": m_base["vel_hist_mean"],
        },
        "synv2": {
            "ks_per_joint": m_syn["ks_per_joint"],
            "ks_mean": m_syn["ks_mean"],
            "vel_hist_per_joint": m_syn["vel_hist_per_joint"],
            "vel_hist_mean": m_syn["vel_hist_mean"],
        },
        "verdict": {
            "synv2_vel_hist_mean": m_syn["vel_hist_mean"],
            "v1style_vel_hist_mean": m_base["vel_hist_mean"],
            "synv2_wins_velocity": bool(
                m_syn["vel_hist_mean"] < m_base["vel_hist_mean"]
            ),
            "synv2_ks_mean": m_syn["ks_mean"],
            "v1style_ks_mean": m_base["ks_mean"],
            "synv2_wins_angle_ks": bool(
                m_syn["ks_mean"] < m_base["ks_mean"]
            ),
            "known_limitation": (
                "角度边缘未显式建模(v3 迭代项); KS 绝对值受限于当前参考"
                "样本量(n_ref 见 meta.n_ref_clips), 仅支持同管线相对比较"
            ),
        },
    }

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    return evidence


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="W28/C4 合成保真度实验: syn_v2 vs v1style 分布距离证据",
    )
    ap.add_argument("--reference-pkl", required=True,
                    help="Q3b 风格参考 pkl (list[dict], keypoints (T,17,C))")
    ap.add_argument("--output-json", required=True,
                    help="证据 JSON 输出路径")
    ap.add_argument("--samples-per-class", type=int, default=8)
    ap.add_argument("--classes", default=None,
                    help="逗号分隔类名; 缺省用 gate4 口径")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args(argv)

    classes = args.classes.split(",") if args.classes else None
    ev = run_fidelity_experiment(
        args.reference_pkl, args.output_json,
        samples_per_class=args.samples_per_class,
        classes=classes, seed=args.seed, bins=args.bins,
    )
    v = ev["verdict"]
    print(f"[w28-c4] syn_v2 vel_hist={v['synv2_vel_hist_mean']:.4f} "
          f"vs v1style {v['v1style_vel_hist_mean']:.4f} "
          f"(wins_velocity={v['synv2_wins_velocity']})")
    print(f"[w28-c4] angle KS mean: syn_v2 {v['synv2_ks_mean']:.4f} "
          f"vs v1style {v['v1style_ks_mean']:.4f} "
          f"(wins_angle={v['synv2_wins_angle_ks']})")
    print(f"[w28-c4] evidence -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
