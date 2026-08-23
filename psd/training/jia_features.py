"""P0.3 段特征池化 — W8 窗口 owner（Phase A Step 2）。

把变长种子段（帧区间切片）经 P0.1 冻结骨干 Φ 转为定长 embedding。

设计要点：
- 复用 W3 加载器口径（只读 import）：resample_to_fixed_t(T=64) +
  to_ntu_view(normalize=True, conf_threshold=0.5)，与 P0.1/P0.2 导出完全一致；
- 编码器注入式（encode_fn: (B,3,T,V,M) -> (B,D)）：本模块不 import torch，
  GPU 前向由 scripts 层装配真实 AimCLR backbone——保证核心池化逻辑可 TDD；
- Φ 全程冻结（method.md §3.3.2 设计决策），本模块无任何参数更新路径。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from psd.data.interpet4d import resample_to_fixed_t, to_ntu_view


def build_segment_view(
    kp_seg: np.ndarray,
    weight_seg: np.ndarray,
    target_t: int = 64,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """段切片 (L,24,3)+(L,24) → NTU 兼容视图 (3,target_t,25,1)。"""
    kp64 = resample_to_fixed_t(kp_seg, target_t=target_t)
    w64 = resample_to_fixed_t(weight_seg, target_t=target_t)
    return to_ntu_view(kp64, weight=w64, conf_threshold=conf_threshold, normalize=True)


def extract_segment_embeddings(
    segments: list[dict],
    load_clip_fn,
    encode_fn,
    batch_size: int = 128,
    target_t: int = 64,
    conf_threshold: float = 0.5,
) -> np.ndarray:
    """段列表 → embedding 矩阵 (N, D)，顺序与输入一致。

    参数：
        segments: 段记录列表（至少含 clip_id/start_frame/end_frame）
        load_clip_fn: clip_id -> {"kp_world","kp_weight",...}（每 clip 只加载一次）
        encode_fn: (B,3,T,V,M) -> (B,D) 的冻结编码器前向
    异常：
        编码输出含非有限值时抛 ValueError（防 NaN 静默污染聚类）。
    """
    if not segments:
        return np.zeros((0, 0), dtype=np.float32)

    # 每 clip 只加载一次（进程内缓存）；缺 clip 立即 KeyError
    cache: dict[str, dict] = {}
    for seg in segments:
        cid = seg["clip_id"]
        if cid not in cache:
            data = load_clip_fn(cid)
            if data is None:
                raise KeyError(f"clip 未找到: {cid}")
            cache[cid] = data

    views = np.stack([
        build_segment_view(
            cache[s["clip_id"]]["kp_world"][s["start_frame"]: s["end_frame"] + 1],
            cache[s["clip_id"]]["kp_weight"][s["start_frame"]: s["end_frame"] + 1],
            target_t=target_t,
            conf_threshold=conf_threshold,
        )
        for s in segments
    ])

    outs = []
    for i in range(0, len(views), batch_size):
        outs.append(np.asarray(encode_fn(views[i: i + batch_size]), dtype=np.float32))
    emb = np.vstack(outs)
    if not np.isfinite(emb).all():
        raise ValueError("编码器产出非有限值——拒绝进入聚类（NaN 防线）")
    return emb


def group_indices_by_clip(segments: list[dict]) -> dict[str, list[int]]:
    """辅助：clip_id -> 段索引列表（报告统计用）。"""
    out: defaultdict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(segments):
        out[s["clip_id"]].append(i)
    return dict(out)
