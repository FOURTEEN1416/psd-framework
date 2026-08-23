"""P0.3 种子消费适配器 — W8 窗口 owner（Phase A Step 1）。

职责：把 W6 落盘的规则种子 NPZ（data/seeds/rule_seeds/*.npz）转成统一的
「段记录列表」，供下游 Φ 特征抽取与原型聚类消费。

接口契约（W8 交接 §2）：下游一切消费以「段列表 → Φ 特征 → 聚类」为接口；
种子来源细节只存在于本模块——未来 SMQ 提案可另写适配器产出同构段记录，
聚类代码零改动。

消费规则（W6 报告 §8 移交建议，强制执行）：
- 置信度 ≥ conf_min（默认 0.8）
- 段时长 ≥ min_duration_s（默认 0.5s，按 nominal_fps=30 换算帧数）
- unknown 标签防御性剔除（W6 引擎中 unknown 段置信度恒 0，正常被置信门
  自然挡掉；显式剔除使契约不依赖该实现细节）

口径标注：公开真实层 - 物理先验伪标签（InterPet4D v1），禁止与合成层混报。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

FPS_DEFAULT = 30.0  # 与 configs/rule_seeds.yaml nominal_fps 一致


# ---------------------------------------------------------------- 加载解析

def load_seed_segments(seeds_dir: str | Path) -> list[dict]:
    """读取目录下全部种子 NPZ，返回按 (clip_id, start_frame) 排序的段记录列表。

    每条记录字段：clip_id / start_frame / end_frame / label / confidence /
    rule_ids(list[str]，复合规则按 '|' 拆分)。
    """
    root = Path(seeds_dir)
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"种子目录无 .npz 文件: {root}")

    segs: list[dict] = []
    for path in files:
        with np.load(path, allow_pickle=False) as npz:
            arr = npz["segments"]
        for row in arr:
            rules_raw = str(row["rules"])
            segs.append({
                "clip_id": path.stem,
                "start_frame": int(row["start"]),
                "end_frame": int(row["end"]),
                "label": str(row["label"]),
                "confidence": float(row["conf"]),
                "rule_ids": [r for r in rules_raw.split("|") if r],
            })
    segs.sort(key=lambda s: (s["clip_id"], s["start_frame"]))
    return segs


# ---------------------------------------------------------------- 过滤与统计

def segment_duration_s(seg: dict, fps: float = FPS_DEFAULT) -> float:
    """段时长（秒）= 帧数（含端点）/ fps。"""
    n_frames = seg["end_frame"] - seg["start_frame"] + 1
    return n_frames / max(float(fps), 1e-6)


def filter_segments(
    segments: list[dict],
    conf_min: float = 0.8,
    min_duration_s: float = 0.5,
    fps: float = FPS_DEFAULT,
    drop_unknown: bool = True,
) -> list[dict]:
    """W6 移交消费规则：置信度门 + 最短时长门（+unknown 防御性剔除）。

    阈值均为闭区间边界（≥ 判定），与 W6 报告「≥0.8」「≥0.5s」口径一致。
    """
    kept = []
    for s in segments:
        if s["confidence"] < conf_min:
            continue
        if segment_duration_s(s, fps=fps) < min_duration_s:
            continue
        if drop_unknown and s["label"] == "unknown":
            continue
        kept.append(s)
    return kept


def label_stats(segments: list[dict]) -> dict[str, int]:
    """类别计数统计（保序：首现顺序）。"""
    out: dict[str, int] = {}
    for s in segments:
        out[s["label"]] = out.get(s["label"], 0) + 1
    return out


# ---------------------------------------------------------------- clip 级切分

def split_clips(
    clip_ids: list[str],
    eval_ratio: float = 0.3,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """clip 级不相交切分：(anchor_clips, eval_clips)。

    为什么 clip 级而非段级：同一 clip 内相邻段高度相关（同犬同场景连续
    运动），段级切分会让原型见到评估段的"近邻副本"，纯度虚高。评估侧
    固定 seed 使全部扫描配置可比。
    """
    if not clip_ids:
        raise ValueError("clip_ids 为空")
    rng = np.random.default_rng(seed)
    ids = np.asarray(sorted(clip_ids))
    perm = rng.permutation(len(ids))
    n_eval = int(round(eval_ratio * len(ids)))
    n_eval = min(max(n_eval, 1), len(ids) - 1)  # 两侧至少各 1
    eval_idx = np.sort(perm[:n_eval])
    anchor_idx = np.sort(perm[n_eval:])
    eval_set = set(ids[eval_idx].tolist())
    anchor_set = set(ids[anchor_idx].tolist())
    # 保持输入排序稳定输出
    eval_clips = [c for c in sorted(clip_ids) if c in eval_set]
    anchor_clips = [c for c in sorted(clip_ids) if c in anchor_set]
    return anchor_clips, eval_clips


# ---------------------------------------------------------------- 分层采样

def sample_anchor_segments(
    segments: list[dict],
    ratio: float,
    seed: int = 42,
) -> list[dict]:
    """按类别分层抽样 ratio 比例的锚点段（稀有类 ceil 保底 ≥1 条）。

    直接操纵 |S|（method.md §3.3.2 敏感性轴）；分层保证每个出现过的类
    在锚点侧有代表，避免 class_mean 初始化时整类缺席。
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio 必须在 [0,1]: {ratio}")
    if ratio == 0.0 or not segments:
        return []
    rng = np.random.default_rng(seed)

    by_label: dict[str, list[int]] = {}
    for i, s in enumerate(segments):
        by_label.setdefault(s["label"], []).append(i)

    chosen: list[int] = []
    for label in sorted(by_label):
        idx = np.asarray(by_label[label])
        k = int(math.ceil(ratio * len(idx)))
        k = min(max(k, 1 if ratio > 0 else 0), len(idx))
        pick = rng.choice(idx, size=k, replace=False)
        chosen.extend(pick.tolist())

    chosen = sorted(set(chosen))
    return [segments[i] for i in chosen]
