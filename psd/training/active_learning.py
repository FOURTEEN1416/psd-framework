"""P0.5 主动学习效率实验核心 — W14 窗口.

职责:
    1. 不确定性采样器（熵）与随机采样器（对照）
    2. 增量式 AL 模拟运行器（配对设计 + 冷启动重训，复用 STGCNBCTrainer）
    3. P0.4 真实池熵打分（best.pt 迁移代理排序，公开真实层口径）

口径声明:
    - 主曲线 = 合成层（新种子池 + 固定 GT 验证集，AGENTS.md 三层铁律）
    - 真实池打分仅产出排序清单与分布统计，禁止作为 acc 数字汇报

选型理由（熵 vs 边际 vs MC-dropout）:
    - MC-dropout: 需每池样本多次随机前向；GPU 被 NTU 长训占用、CPU 算力紧
      （0.92s/step 实测），成本 ×N_forward 不可接受；且 STGCNBC 默认 dropout=0
    - 边际(top1-top2): 只看前两名差距，忽略其余类别概率质量；
      22 类中语义相近类多（sit/stay/alert_sit），尾部质量含信息
    - 熵: 单次确定性前向的信息论标准量，对全分布敏感，
      AL 文献标准基线（Settles 2009 综述）→ 选定
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# 采样器
# ---------------------------------------------------------------------------

def entropy_scores(probs: np.ndarray) -> np.ndarray:
    """softmax 概率 → 每样本 Shannon 熵.

    Args:
        probs: (B, C) 每行和为 1 的概率矩阵

    Returns:
        (B,) float64 熵值
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError(f"probs 应为 (B, C)，收到 shape={probs.shape}")
    return -(probs * np.log(probs + _EPS)).sum(axis=-1)


class RandomSelector:
    """随机采样（对照臂）。"""

    def select(
        self,
        pool_size: int,
        k: int,
        rng: np.random.Generator,
        exclude: Optional[Set[int]] = None,
    ) -> List[int]:
        exclude = exclude or set()
        candidates = [i for i in range(pool_size) if i not in exclude]
        k = min(k, len(candidates))
        idx = rng.permutation(len(candidates))[:k]
        return sorted(candidates[j] for j in idx)


class EntropySelector:
    """熵不确定性采样（实验臂）：按熵降序取前 k。"""

    def select(
        self,
        scores: np.ndarray,
        exclude: Optional[Set[int]] = None,
        k: int = 10,
    ) -> List[int]:
        scores = np.asarray(scores, dtype=np.float64)
        exclude = exclude or set()
        order = np.argsort(-scores, kind="stable")
        picked: List[int] = []
        for i in order:
            if int(i) in exclude:
                continue
            picked.append(int(i))
            if len(picked) >= k:
                break
        return picked
