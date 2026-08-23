"""SMQ 输入适配层 — P0.2 数据 owner。

职责：InterPet4D smal_npy → SMQ 官方管线可消费的 (C, T, V, M) 骨架视图。
- 复用 interpet4d.py 的加载/重采样/归一化逻辑（单一 truth，不重复实现）
- SMQ 使用 SMAL 原生 V=24（无需 AimCLR NTU 视图的死关节槽位 25）
- episode 构造：多 clip 拼接为「拼接式 episode」，源 clip 区间即 GT 段
  （口径披露：InterPet4D 无行为标注，该协议度量的是无监督边界恢复能力，
   非行为识别精度；指标层 = 公开真实层）

适配逻辑全部在本模块，禁止修改 external/SMQ 内部实现。
"""
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from psd.data.interpet4d import (
    _normalize_sequence,
    is_valid_clip,
    load_clip,
    parse_clip_id,
    resample_to_fixed_t,
)

SMAL_NUM_JOINTS = 24


def clip_to_smq_view(
    kp: np.ndarray,
    weight: np.ndarray,
    target_t: int | None = 128,
    conf_threshold: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """(T,24,3)+置信度 → SMQ 视图 (3, T, 24, 1) float32。

    target_t=None 时保留原生帧长（E-D 协议：全长序列保留运动细节，
    匹配 SMQ 论文的长未修剪序列设定）。
    流程与 W3 的 to_ntu_view 对齐：重采样(可选) → 质心去中心+尺度归一 →
    低置信置零；V=24 原生关节数（SMQ 以 V 为构造参数，无死槽位需求）。
    """
    if target_t is not None:
        kp = resample_to_fixed_t(np.asarray(kp, dtype=np.float64), target_t=target_t)
        weight = resample_to_fixed_t(np.asarray(weight, dtype=np.float64), target_t=target_t)
    else:
        kp = np.asarray(kp, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
    kp = kp.astype(np.float32, copy=True)
    assert kp.ndim == 3 and kp.shape[1] == SMAL_NUM_JOINTS and kp.shape[2] == 3
    assert weight.shape == kp.shape[:2]

    if normalize:
        kp = _normalize_sequence(kp, weight)
    kp[weight < conf_threshold] = 0.0

    view = np.transpose(kp, (2, 0, 1))[:, :, :, None]  # (3,T,24,1)
    return np.ascontiguousarray(view, dtype=np.float32)


def export_smq_features(data_root: str | Path, out_dir: str | Path,
                        target_t: int | None = None) -> dict:
    """smal_npy/*.npz → features/<stem>.npy（跳过无效 clip）。

    target_t=None 保留原生帧长。返回 {"features_dir", "names", "skipped"}。
    """
    src = Path(data_root)
    feats = Path(out_dir)
    feats.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    skipped: list[str] = []
    for path in sorted(src.glob("*.npz")):
        clip = load_clip(path)
        if not is_valid_clip(clip["kp_world"]):
            skipped.append(path.stem)
            continue
        view = clip_to_smq_view(clip["kp_world"], clip["kp_weight"], target_t=target_t)
        np.save(feats / f"{path.stem}.npy", view)
        names.append(path.stem)
    return {"features_dir": str(feats), "names": names, "skipped": skipped}


def select_eval_clips(names: list[str], total: int, seed: int = 42) -> list[str]:
    """确定性抽选 eval clips：按 dog 轮转（每轮洗牌后的 dog 序），不重复。

    与训练集隔离由调用方负责（train 目录排除本函数返回的 clips）。
    """
    by_dog: dict[str, list[str]] = defaultdict(list)
    for n in sorted(names):
        by_dog[parse_clip_id(n + ".npz")].append(n)
    rng = random.Random(seed)
    dogs = sorted(by_dog)
    rng.shuffle(dogs)
    picked: list[str] = []
    cursor = {d: 0 for d in dogs}
    while len(picked) < total:
        progressed = False
        for d in dogs:
            if len(picked) >= total:
                break
            pool = by_dog[d]
            i = cursor[d]
            while i < len(pool) and pool[i] in set(picked):
                i += 1
            if i < len(pool):
                picked.append(pool[i])
                cursor[d] = i + 1
                progressed = True
        if not progressed:
            raise ValueError("clip 池不足以选出互异 eval clips")
    return picked


def group_into_episodes(picked: list[str], clips_per_episode: int) -> list[list[str]]:
    """顺序切分为 episode 组；同组内 dog 互斥由上游轮转保证并在此校验。"""
    groups = [picked[i:i + clips_per_episode]
              for i in range(0, len(picked), clips_per_episode)]
    for g in groups:
        dogs = [parse_clip_id(n + ".npz") for n in g]
        if len(set(dogs)) != len(dogs):
            raise ValueError(f"episode 内 dog 重复: {g}")
    return groups


def rotate_by_dog(names: list[str], seed: int = 42) -> list[str]:
    """确定性 dog 轮转排序（训练 episode 分组用）。

    洗牌后的 dog 序轮转取 clip，尽量让相邻 clip 来自不同 dog。
    """
    by_dog: dict[str, list[str]] = defaultdict(list)
    for n in sorted(names):
        by_dog[parse_clip_id(n + ".npz")].append(n)
    rng = random.Random(seed)
    dogs = sorted(by_dog)
    rng.shuffle(dogs)
    out: list[str] = []
    cursor = {d: 0 for d in dogs}
    while len(out) < len(names):
        progressed = False
        for d in dogs:
            if cursor[d] < len(by_dog[d]):
                out.append(by_dog[d][cursor[d]])
                cursor[d] += 1
                progressed = True
        if not progressed:
            break
    return out


def chunk_names(names: list[str], size: int) -> list[list[str]]:
    """顺序等宽切分（无约束；训练 episode 用，尾部允许不足/重复 dog）。"""
    return [names[i:i + size] for i in range(0, len(names), size)]


def build_episode(features_dir: str | Path, clip_names: list[str]) -> dict:
    """按序拼接 clip 视图为 episode（允许不等长；边界取累计帧和）。

    返回 {"data": (3, T_total, 24, 1), "segments": [{name,start,end}...]}。
    """
    feats = Path(features_dir)
    views, segments = [], []
    pos = 0
    for name in clip_names:
        v = np.load(feats / f"{name}.npy")
        views.append(v.astype(np.float32))
        segments.append({"name": name, "start": pos, "end": pos + v.shape[1]})
        pos += v.shape[1]
    return {"data": np.concatenate(views, axis=1), "segments": segments}
