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


def _normalize_framewise(kp: np.ndarray) -> np.ndarray:
    """逐帧归一化（E-H，对齐官方 LARa hip-centered 思路）：
    每帧独立去有效关节质心 + 中位数半径缩放。

    与序列级归一的区别：保留跨帧的姿态变化信息（序列级统计会把慢漂移
    连同运动信号一起抹平），只去除全局平移/尺度。
    """
    out = np.empty_like(kp)
    for t in range(kp.shape[0]):
        frame = kp[t]
        valid = np.isfinite(frame).all(axis=-1)
        if valid.sum() == 0:
            out[t] = frame
            continue
        centroid = frame[valid].mean(axis=0, keepdims=True)
        centered = frame - centroid
        radii = np.linalg.norm(centered[valid], axis=-1)
        scale = float(np.median(radii))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        out[t] = centered / scale
    return out.astype(np.float32)


def clip_to_smq_view(
    kp: np.ndarray,
    weight: np.ndarray,
    target_t: int | None = 128,
    conf_threshold: float = 0.5,
    normalize: bool | str = True,
) -> np.ndarray:
    """(T,24,3)+置信度 → SMQ 视图 (3, T, 24, 1) float32。

    normalize: False=不归一 | True/'sequence'=序列级质心+尺度（W3 口径）
             | 'frame'=逐帧质心+尺度（E-H：保留跨帧姿态变化）
    target_t=None 保留原生帧长。低置信关节置零在归一之后统一执行。
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

    if normalize is True or normalize == "sequence":
        kp = _normalize_sequence(kp, weight)
    elif normalize == "frame":
        kp = _normalize_framewise(kp)
    kp[weight < conf_threshold] = 0.0

    view = np.transpose(kp, (2, 0, 1))[:, :, :, None]  # (3,T,24,1)
    return np.ascontiguousarray(view, dtype=np.float32)


def export_smq_features(data_root: str | Path, out_dir: str | Path,
                        target_t: int | None = None,
                        normalize: bool | str = True) -> dict:
    """smal_npy/*.npz → features/<stem>.npy（跳过无效 clip）。

    target_t=None 保留原生帧长；normalize 见 clip_to_smq_view。
    返回 {"features_dir", "names", "skipped"}。
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
        view = clip_to_smq_view(clip["kp_world"], clip["kp_weight"],
                                target_t=target_t, normalize=normalize)
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


# ---------------------------------------------------------------- 种子伪 GT（双口径第二协议）

def load_seed_segments(
    seeds_dir: str | Path,
    name: str,
    *,
    min_conf: float = 0.8,
    min_duration_s: float = 0.5,
    fps: float = 30.0,
) -> list[dict]:
    """读取单 clip 的规则种子段，按 W6 报告 §8 消费规则过滤。

    规则：置信度 ≥min_conf 且最短持续 ≥min_duration_s。
    口径标注：「公开真实层-物理先验伪标签」（与拼接协议并列汇报，不择优单报）。
    只读消费 data/seeds 生成物与 rule_seeds.py 引擎输出（W6 领地，禁改）。
    """
    d = np.load(Path(seeds_dir) / f"{name}.npz", allow_pickle=True)
    min_frames = max(1, int(round(min_duration_s * fps)))
    out: list[dict] = []
    for s in d["segments"]:
        conf = float(s["conf"])
        dur = int(s["end"]) - int(s["start"])
        if conf >= min_conf and dur >= min_frames:
            out.append({"start": int(s["start"]), "end": int(s["end"]),
                        "label": str(s["label"]), "conf": conf})
    return out


def build_seed_gt_episode(
    seeds_dir: str | Path,
    features_dir: str | Path,
    clip_names: list[str],
    *,
    min_conf: float = 0.8,
    min_duration_s: float = 0.5,
) -> list[dict]:
    """各 clip 种子伪 GT → 拼接 episode 坐标系（偏移 = 特征视图实际帧长累计）。

    帧数以 features_all 导出视图为准（与模型输入严格同源），保证坐标零错位。
    """
    feats = Path(features_dir)
    segs: list[dict] = []
    pos = 0
    for n in clip_names:
        t = int(np.load(feats / f"{n}.npy", mmap_mode="r").shape[1])
        for s in load_seed_segments(seeds_dir, n, min_conf=min_conf,
                                    min_duration_s=min_duration_s):
            segs.append({"start": s["start"] + pos, "end": s["end"] + pos,
                         "label": s["label"], "conf": s["conf"]})
        pos += t
    return segs
