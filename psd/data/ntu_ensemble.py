"""NTU 三流融合（3s ensemble）— external/AimCLR/ensemble_ntu_cs.py 的本仓移植。

官方逻辑（保真目标）:
    r = joint*0.6 + bone*0.6 + motion*0.4   （三流 logits 加权求和）
    top1 = argmax(r)；top5 = label ∈ argsort(r)[-5:]

本仓移植差异（已在测试与报告同步声明）:
    1. 三流分数按 val_label.pkl 的 sample_name 顺序显式对齐——官方隐式依赖
       三个 test_result.pkl 插入顺序一致，本移植改为按键对齐 + fail-fast；
    2. 输出结构化 dict 供 JSON 归档，替代官方纯 print。

白名单归属: psd/data/*ntu*（AGENTS.md §4 / W9 任务书 §4）。
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

# 官方 ensemble_ntu_cs.py: alpha = [0.6, 0.6, 0.4]  # joint, bone, motion
DEFAULT_ALPHA = {"joint": 0.6, "bone": 0.6, "motion": 0.4}
STREAM_ORDER = ("joint", "bone", "motion")


def load_score_pkl(path: str | Path) -> dict:
    """读取 LE_Processor 落盘的 test_result.pkl（sample_name -> 分数向量）。"""
    path = Path(path)
    with open(path, "rb") as f:
        result = pickle.load(f)
    if not isinstance(result, dict):
        raise TypeError(f"{path} 应为 dict(sample_name->ndarray)，实得 {type(result)}")
    return result


def load_labels(label_pkl: str | Path) -> tuple[list, list]:
    """读取 val_label.pkl，返回 (sample_names, labels)。"""
    with open(Path(label_pkl), "rb") as f:
        names, labels = pickle.load(f)
    return list(names), list(labels)


def fuse_scores(
    score_dicts: Mapping[str, dict],
    alpha: Mapping[str, float] | None = None,
) -> tuple[list, np.ndarray]:
    """按官方 alpha 加权融合三流分数。

    行序跟随 joint 字典插入序（与官方 r1 迭代序一致）；各流键集合不一致、
    或分数向量维度不一致时 raise ValueError（fail-fast，不静默对齐）。
    返回 (sample_names, fused[N,C])。
    """
    alpha = DEFAULT_ALPHA if alpha is None else dict(alpha)
    missing_streams = [s for s in STREAM_ORDER if s not in score_dicts]
    if missing_streams:
        raise ValueError(f"缺少流的分数: {missing_streams}")

    joint = score_dicts["joint"]
    names = list(joint.keys())

    for stream in ("bone", "motion"):
        d = score_dicts[stream]
        missing = [n for n in names if n not in d]
        extra = [n for n in d if n not in joint]
        if missing or extra:
            raise ValueError(
                f"{stream}: 分数键与 joint 不一致（缺失 {missing[:5]}，多余 {extra[:5]}）"
            )

    rows = []
    for name in names:
        vecs = {}
        for stream in STREAM_ORDER:
            vecs[stream] = np.asarray(score_dicts[stream][name], dtype=np.float64)
        dims = {v.shape for v in vecs.values()}
        if len(dims) != 1:
            raise ValueError(f"{name}: 三流分数维度不一致 {dims}")
        rows.append(sum(alpha[s] * vecs[s] for s in STREAM_ORDER))

    fused = np.stack(rows) if rows else np.zeros((0, 0))
    return names, fused


def topk_accuracy(scores: np.ndarray, labels: Sequence[int], k: int) -> float:
    """官方语义: rank=argsort(r)；hit = label ∈ rank[-k:]。"""
    scores = np.asarray(scores)
    rank = np.argsort(scores, axis=1)
    hits = [int(labels[i]) in rank[i, -k:] for i in range(len(labels))]
    return sum(hits) / len(hits)


def run_ensemble(
    stream_paths: Mapping[str, str | Path],
    label_pkl: str | Path,
    alpha: Mapping[str, float] | None = None,
) -> dict:
    """端到端三流融合：读分 → 按 label 序对齐 → 加权融合 → top1/top5。"""
    alpha = DEFAULT_ALPHA if alpha is None else dict(alpha)

    names, labels = load_labels(label_pkl)

    aligned: dict[str, dict] = {}
    for stream in STREAM_ORDER:
        d = load_score_pkl(stream_paths[stream])
        missing = [n for n in names if n not in d]
        if missing:
            raise ValueError(f"{stream}: 分数缺样本 {missing[:5]}（label 共 {len(names)}）")
        aligned[stream] = {n: d[n] for n in names}

    _, fused = fuse_scores(aligned, alpha=alpha)

    return {
        "top1": topk_accuracy(fused, labels, k=1),
        "top5": topk_accuracy(fused, labels, k=5),
        "n": len(labels),
        "alpha": {s: alpha[s] for s in STREAM_ORDER},
        "stream_paths": {s: str(stream_paths[s]) for s in STREAM_ORDER},
    }


def collect_stream_result(work_dir: str | Path) -> dict:
    """从线性评估 work_dir 收集单流成绩（log.txt 由官方 IO.print_log 落盘）。

    官方每个 eval 点打印两行: "\\tTop1: xx.xx%" 与 "\\tBest Top1: yy.yy%"。
    """
    work_dir = Path(work_dir)
    info: dict = {
        "work_dir": str(work_dir),
        "best_top1": None,
        "last_top1": None,
    }
    log = work_dir / "log.txt"
    if not log.exists():
        return info

    best_values: list[float] = []
    cur_values: list[float] = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("Best Top1:"):
            best_values.append(float(styled_float(stripped)))
        elif stripped.startswith("Top1:"):
            cur_values.append(float(styled_float(stripped)))

    if best_values:
        info["best_top1"] = best_values[-1]
    if cur_values:
        info["last_top1"] = cur_values[-1]
    return info


def styled_float(line: str) -> str:
    """从 'Best Top1: 74.01%' 形态行提取数字串。"""
    payload = line.split(":", 1)[1].strip().rstrip("%")
    return payload.strip()
