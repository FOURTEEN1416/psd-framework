# -*- coding: utf-8 -*-
"""W40 数据飞轮 round2 — 核心库（预注册配置: configs/public_real_round2.yaml）.

三件套语义（全部无监督、标签零接触，沿 W30 deferred 判例）:
  1. APTv2 调和变换: 序列级 bbox 归一化 + NaN 槽位补零 + 目标域死掩码调和
  2. DogSet 运动学先验门禁: 无量纲速度比统计层消费，禁跨拓扑张量混批
  3. AdaBN 域适应: backbone 权重冻结，仅 BN running statistics 累积真实域几何

Owner: W40（wt/W40）。消费统一池产物 runs/data_campaign/unified/real_expansion_pool_v1.pkl
（只读），禁触组装器与 AK 原始资产。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.nn.modules.batchnorm import _BatchNorm

# ---------------------------------------------------------------------------
# 预注册常量（与 configs/public_real_round2.yaml 一一对应）
# ---------------------------------------------------------------------------

# W30 映射表诚实缺失槽位（尾尖/耳base×2/下巴/耳tip×2/喉部）
APTV2_NAN_SLOTS = {13, 14, 15, 17, 18, 19, 23}
# 目标域（AK/w35 YOLO 提点链）死关节约定（双眼/withers/throat）
TARGET_DEAD_JOINTS = {20, 21, 22, 23}

TRANSFORM_APTV2 = "bbox_norm+nanslot_fill+deadmask_harmonize"
TRANSFORM_PASSTHROUGH = "passthrough"

GATE_FACTOR_DEFAULT = 3.0
GATE_DEGENERATE_FRACTION = 0.20


# ---------------------------------------------------------------------------
# 1. 调和变换
# ---------------------------------------------------------------------------

def harmonize_aptv2_keypoints(kp: np.ndarray) -> np.ndarray:
    """APTv2 原始像素域条目 → 目标域约定调和数组 (T,24,3) f32。

    步骤（顺序即语义，勿重排）:
      i.   不可见点（vis=0 或坐标非有限）整体置零——目标域中不可见=不存在；
      ii.  全部残留 NaN/Inf 补零（覆盖 7 个映射缺失槽位）;
      iii. 序列级联合包围盒把可见点 xy 仿射到 [0,1]——逐帧归一会注入假尺度
           动态，沿 W30"宁缺毋滥"判例拒绝;
      iv.  死关节 {20,21,22,23} 置零（AK/w35 提点链约定；APTv2 原生双眼/
           withers 值被覆盖——一致性优先于信息保留，报告显式披露此代价）。
    ch3 保持 vis01 二值原值（伪造连续 conf 被预注册禁止）。
    """
    kp = np.asarray(kp, dtype=np.float32).copy()
    if kp.ndim != 3 or kp.shape[1] != 24 or kp.shape[2] != 3:
        raise ValueError(f"期望 (T,24,3)，实际 {kp.shape}")
    finite = np.isfinite(kp[..., 0]) & np.isfinite(kp[..., 1])
    visible = finite & (kp[..., 2] > 0)

    kp[~np.isfinite(kp)] = 0.0          # ii. NaN/Inf 补零（含 7 缺失槽位）
    kp[~visible] = 0.0                  # i.  不可见点整体置零

    if visible.any():                   # iii. 序列级 bbox 归一化
        xs, ys = kp[..., 0][visible], kp[..., 1][visible]
        xmin, xmax = float(xs.min()), float(xs.max())
        ymin, ymax = float(ys.min()), float(ys.max())
        sx = (xmax - xmin) if (xmax - xmin) > 1e-9 else 1.0
        sy = (ymax - ymin) if (ymax - ymin) > 1e-9 else 1.0
        kp[..., 0] = (kp[..., 0] - xmin) / sx
        kp[..., 1] = (kp[..., 1] - ymin) / sy

    dead = sorted(TARGET_DEAD_JOINTS)   # iv. 目标域死掩码调和
    kp[:, dead, :] = 0.0

    out = np.clip(kp, 0.0, None).astype(np.float32)
    if not np.isfinite(out).all():
        raise AssertionError("调和后仍存在非有限值——实现缺陷")
    return out


def prepare_w35_keypoints(kp: np.ndarray) -> np.ndarray:
    """video_c1_w35 条目直通（image_norm_xy_conf01_deadmasked 与目标域同约定）。"""
    return np.asarray(kp, dtype=np.float32).copy()


# ---------------------------------------------------------------------------
# 2. DogSet 运动学先验（无量纲速度比，统计层消费）
# ---------------------------------------------------------------------------

def kinematic_ratio(kp: np.ndarray, fps: float, dims: int = 3,
                    max_scale_frames: int = 500) -> float:
    """无量纲速度比 = 全关节 RMS 速度(按秒归一) / 体尺度代理。

    体尺度代理 = 逐帧全关节两两距离中位数的跨帧中位数（拓扑无关，
    无需关节名映射）；速度只计入前 ``dims`` 个坐标维（clips 的 ch3 是
    conf 不是空间坐标，必须 dims=2）。
    """
    pts = np.asarray(kp, dtype=np.float32)[..., :dims]
    T = pts.shape[0]
    if T < 2 or fps <= 0:
        return 0.0

    # 体尺度: 帧间均匀抽样至多 max_scale_frames 帧控本
    frame_idx = np.unique(np.linspace(0, T - 1, min(T, max_scale_frames)).astype(int))
    scales: List[float] = []
    for t in frame_idx:
        P = pts[t]
        n = P.shape[0]
        if n < 2:
            continue
        d2 = ((P[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        iu = np.triu_indices(n, 1)
        med = float(np.median(np.sqrt(d2[iu])))
        if med > 0:
            scales.append(med)
    scale = float(np.median(scales)) if scales else 0.0
    if scale <= 1e-9:
        return 0.0

    v = np.diff(pts, axis=0)                       # (T-1, V, dims)
    speeds = np.sqrt((v ** 2).sum(-1))             # 每步位移
    rms_speed_per_sec = float(np.sqrt((speeds ** 2).mean()) * fps)
    return rms_speed_per_sec / scale


def kinematic_gate_thresholds(ratios: List[float], lo_q: float = 0.5,
                              hi_q: float = 99.5,
                              factor: float = GATE_FACTOR_DEFAULT) -> Dict[str, float]:
    """DogSet 参考分布 → 宽容带阈值 [p0.5, p99.5]（硬排除再乘 factor）。"""
    arr = np.asarray(list(ratios), dtype=np.float64)
    return {
        "lo": float(np.percentile(arr, lo_q)),
        "hi": float(np.percentile(arr, hi_q)),
        "factor": float(factor),
    }


def apply_kinematic_gate(samples: Dict[str, float],
                         thresholds: Dict[str, float],
                         degenerate_fraction: float = GATE_DEGENERATE_FRACTION,
                         ) -> Tuple[List[str], Dict]:
    """物理离群硬排除；排除率超过 ``degenerate_fraction`` 触发退化保护（纯报告模式，全保留）。

    默认 0.20 按预注册配置；小样本单元测试可显式放宽以单独检验排除逻辑。
    """
    factor = float(thresholds.get("factor", GATE_FACTOR_DEFAULT))
    band_lo = float(thresholds["lo"]) / factor
    band_hi = float(thresholds["hi"]) * factor
    excluded = sorted(k for k, r in samples.items() if r > band_hi or r < band_lo)
    frac = len(excluded) / max(len(samples), 1)
    report_only = frac > degenerate_fraction
    kept = sorted(samples) if report_only else [k for k in sorted(samples)
                                                if k not in set(excluded)]
    report = {
        "band": {"lo_inclusive": band_lo, "hi_inclusive": band_hi},
        "excluded": [] if report_only else excluded,
        "excluded_fraction_raw": round(frac, 4),
        "report_only": report_only,
        "n_input": len(samples),
        "n_kept": len(kept),
    }
    return kept, report


# ---------------------------------------------------------------------------
# 3. 适应集组装
# ---------------------------------------------------------------------------

def _w35_src_fps(fps_field) -> float:
    """w35 的 fps_or_sampling 形如 {'src_fps': 29.97, 'strategy': 'uniform_T30'}。"""
    if isinstance(fps_field, dict):
        return float(fps_field.get("src_fps", 30.0))
    try:
        v = float(fps_field)
        return v if v > 0 else 30.0
    except (TypeError, ValueError):
        return 30.0


def build_adaptation_set(
    pool_entries: List[dict],
    gate_ref: Optional[Dict[str, float]],
    fps_assumptions: Optional[Dict[str, float]],
) -> Tuple[List[np.ndarray], List[dict], Dict]:
    """从池条目构造 Phase A 适应集（仅 usage_scope=='pretrain_geometric' 槽位）。

    Returns:
        arrays: 调和后 (T,24,3) 数组列表（sample_id 升序确定性排序）
        metas:  每样本溯源 {sample_id, source_channel, transform, kinematic_ratio, T}
        report: 计数/门禁/fps 假设回显
    """
    fps_assumptions = dict(fps_assumptions or {})
    selected = sorted(
        (e for e in pool_entries if e.get("usage_scope") == "pretrain_geometric"),
        key=lambda e: str(e["sample_id"]),
    )

    arrays: List[np.ndarray] = []
    metas: List[dict] = []
    ratios: Dict[str, float] = {}

    for e in selected:
        sid = str(e["sample_id"])
        src = str(e["source_channel"])
        kp = np.asarray(e["keypoints"], dtype=np.float32)
        if src == "aptv2_c2_w26":
            arr = harmonize_aptv2_keypoints(kp)
            transform = TRANSFORM_APTV2
            fps = float(fps_assumptions.get(src, 15.0))   # 名义值假设，报告显式
            ratio = kinematic_ratio(arr, fps=fps, dims=2)
        elif src == "video_c1_w35":
            arr = prepare_w35_keypoints(kp)
            transform = TRANSFORM_PASSTHROUGH
            fps = _w35_src_fps(e.get("fps_or_sampling"))
            ratio = kinematic_ratio(arr, fps=fps, dims=2)
        else:
            raise ValueError(
                f"pretrain_geometric 槽位出现未登记来源 {src}（sample_id={sid}）——"
                "池 schema 演进需先更新预注册配置")
        arrays.append(arr)
        metas.append({
            "sample_id": sid,
            "source_channel": src,
            "transform": transform,
            "kinematic_ratio": round(float(ratio), 6),
            "T": int(arr.shape[0]),
        })
        ratios[sid] = float(ratio)

    counts: Dict[str, int] = {}
    for m in metas:
        counts[m["source_channel"]] = counts.get(m["source_channel"], 0) + 1

    if gate_ref is not None:
        _, gate_report = apply_kinematic_gate(ratios, gate_ref)
        if not gate_report["report_only"] and gate_report["excluded"]:
            drop = set(gate_report["excluded"])
            keep_idx = [i for i, m in enumerate(metas) if m["sample_id"] not in drop]
            arrays = [arrays[i] for i in keep_idx]
            metas = [metas[i] for i in keep_idx]
            gate_report["n_dropped_from_set"] = len(drop)
        else:
            gate_report["n_dropped_from_set"] = 0
    else:
        gate_report = {"status": "skipped_no_reference", "excluded": [],
                       "report_only": False, "n_dropped_from_set": 0}

    report = {
        "counts": counts,
        "n_total": len(arrays),
        "slot": "usage_scope==pretrain_geometric",
        "gate": gate_report,
        "fps_assumptions_echo": fps_assumptions,
    }
    return arrays, metas, report


# ---------------------------------------------------------------------------
# 4. AdaBN 域适应
# ---------------------------------------------------------------------------

def adabn_adapt(model: torch.nn.Module,
                arrays: List[np.ndarray],
                batch_size: int = 16,
                seed: int = 42,
                passes: int = 1,
                device: str = "cpu") -> Dict:
    """AdaBN 式无监督域适应（预注册: momentum=None 单遍累积平均，fp32 前向）。

    只动 backbone 内 BatchNorm 的 running statistics；全部权重参数与 head
    零接触。变长 T 分桶独立批处理（30/15 不可同批堆叠）。结束后 BN 恢复 eval。
    """
    model = model.to(device)
    model.eval()
    bns = [m for m in model.backbone.modules() if isinstance(m, _BatchNorm)]
    before = [(bn.running_mean.detach().clone(), bn.running_var.detach().clone())
              for bn in bns]

    for bn in bns:                      # AdaBN 三连: 重置 → 累积模式 → train
        bn.reset_running_stats()
        bn.momentum = None              # None = 累积移动平均（单遍即全量统计）
        bn.train()

    buckets: Dict[int, List[np.ndarray]] = {}
    for a in arrays:
        buckets.setdefault(int(np.asarray(a).shape[0]), []).append(np.asarray(a))

    gen = torch.Generator().manual_seed(seed)
    n_forward = 0
    with torch.no_grad():
        for _ in range(passes):
            for t_len in sorted(buckets):
                pool_t = buckets[t_len]
                order = torch.randperm(len(pool_t), generator=gen).tolist()
                for i in range(0, len(order), batch_size):
                    chunk = [pool_t[j] for j in order[i:i + batch_size]]
                    x = torch.from_numpy(np.stack(chunk).astype(np.float32)).to(device)
                    model(x)
                    n_forward += int(x.shape[0])

    for bn in bns:                      # 冻结回去
        bn.eval()

    per_bn_moved: List[bool] = []
    for bn, (bm, bv) in zip(bns, before):
        moved = not (torch.equal(bn.running_mean.detach(), bm)
                     and torch.equal(bn.running_var.detach(), bv))
        per_bn_moved.append(bool(moved))

    return {
        "n_forward_samples": n_forward,
        "buckets": {k: len(v) for k, v in sorted(buckets.items())},
        "n_bn_modules": len(bns),
        "per_bn_moved": per_bn_moved,
        "bn_momentum": None,
        "passes": passes,
        "batch_size": batch_size,
        "shuffle_seed": seed,
    }


# ---------------------------------------------------------------------------
# 5. 协议一致性
# ---------------------------------------------------------------------------

def make_train_config(echo: Dict):
    """round1 protocol_echo → TrainConfig（Phase B 与基线逐字段一致）。"""
    from psd.training.train_stgcn_bc import TrainConfig
    return TrainConfig(
        epochs=int(echo["epochs"]),
        batch_size=int(echo["batch_size"]),
        patience=int(echo["patience"]),
        seed=int(echo["seed"]),
        use_amp=bool(echo["use_amp"]),
    )
