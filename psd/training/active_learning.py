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
from dataclasses import replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig

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


# ---------------------------------------------------------------------------
# 前向概率批量推理（运行器与真实池打分共用）
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_probs(
    model,
    samples: Sequence[Dict],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """样本列表 → (N, C) softmax 概率矩阵.

    Args:
        model: STGCNBC（函数内强制 eval 并搬到指定设备）
        samples: 含 "keypoints" (T,24,3) 的 dict 列表
        batch_size: 批大小
        device: 推理设备
    """
    model = model.to(device).eval()
    out: List[np.ndarray] = []
    for s in range(0, len(samples), batch_size):
        chunk = samples[s : s + batch_size]
        kpts = torch.stack([
            x["keypoints"].float() if torch.is_tensor(x["keypoints"])
            else torch.from_numpy(np.asarray(x["keypoints"], dtype=np.float32))
            for x in chunk
        ]).to(device)
        probs = model(kpts)[0].softmax(dim=-1)
        out.append(probs.cpu().numpy())
    if not out:
        raise ValueError("samples 为空，无法推理")
    return np.concatenate(out, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# 增量式 AL 模拟运行器
# ---------------------------------------------------------------------------

class ALSimulationRunner:
    """配对增量式 AL 效率模拟.

    协议（W14 预注册）:
        - 同 seed 两臂共享同一随机初始核（budgets[0]），增量由各臂策略选择
        - 打分模型 = 上一累计预算训得的模型（greedy AL 标准做法）
        - 每预算点冷启动重训（固定 torch 初始化种子），消除累积漂移混淆
        - 曲线点 = best_val_acc（与仓内 P0.5 口径一致）
    """

    def __init__(
        self,
        build_model: Callable[[], "torch.nn.Module"],
        pool_samples: List[Dict],
        val_samples: List[Dict],
        train_config: TrainConfig,
        budgets: Sequence[int],
        device: str = "cpu",
    ):
        self.build_model = build_model
        self.pool = pool_samples
        self.val = val_samples
        self.template_cfg = train_config
        self.budgets = sorted(int(b) for b in budgets)
        self.device = device
        self._selections: Dict[int, List[int]] = {}
        if self.budgets and self.budgets[0] > len(pool_samples):
            raise ValueError("最小预算超过池容量")

    # -- 公开接口 -----------------------------------------------------------

    def run_trajectory(self, strategy: str, seed: int) -> Dict[int, float]:
        """跑一条轨迹（strategy ∈ {entropy, random}），返回 {budget: best_val_acc}."""
        if strategy not in {"entropy", "random"}:
            raise ValueError(f"未知策略: {strategy}")
        labeled: Set[int] = set()
        metrics: Dict[int, float] = {}
        prev_model = None

        for i, b in enumerate(self.budgets):
            if i == 0:
                new_ids = RandomSelector().select(
                    pool_size=len(self.pool), k=b,
                    rng=np.random.default_rng(seed),
                )
            else:
                delta = b - self.budgets[i - 1]
                if strategy == "random":
                    new_ids = RandomSelector().select(
                        pool_size=len(self.pool), k=delta,
                        rng=np.random.default_rng(1000 + seed * 100 + b),
                        exclude=labeled,
                    )
                else:  # entropy：以上一阶段模型对剩余池打分
                    remaining = [j for j in range(len(self.pool)) if j not in labeled]
                    probs = predict_probs(prev_model, [self.pool[j] for j in remaining], device=self.device)
                    scores = entropy_scores(probs)
                    new_ids = EntropySelector().select(scores=scores, exclude=labeled, k=delta)

            labeled |= set(new_ids)
            ordered = sorted(labeled)
            self._selections[b] = ordered
            acc, prev_model = self._fit_stage(
                ordered, init_seed=seed * 1000 + b, subdir=f"s{seed}_{strategy}_b{b}",
            )
            metrics[b] = acc
            logger.info("[AL] seed=%s strategy=%s budget=%s val_acc=%.4f", seed, strategy, b, acc)

        return metrics

    def selected_at(self, budget: int) -> List[int]:
        return list(self._selections[int(budget)])

    def _initial_core(self, seed: int) -> List[int]:
        """初始随机核心（两臂共享，配对设计保证）。"""
        return RandomSelector().select(
            pool_size=len(self.pool), k=self.budgets[0],
            rng=np.random.default_rng(seed),
        )

    # -- 内部 ---------------------------------------------------------------

    def _fit_stage(
        self,
        sample_ids: List[int],
        init_seed: int,
        subdir: Optional[str] = None,
    ) -> Tuple[float, "torch.nn.Module"]:
        """给定标注 id 冷启动重训，返回 (best_val_acc, model)。"""
        torch.manual_seed(init_seed)
        samples = [self.pool[i] for i in sample_ids]

        cfg = replace(
            self.template_cfg,
            batch_size=min(self.template_cfg.batch_size, len(samples)),
            output_dir=str(Path(self.template_cfg.output_dir) / subdir) if subdir else self.template_cfg.output_dir,
        )
        model = self.build_model()
        trainer = STGCNBCTrainer(model, samples, self.val, config=cfg)
        summary = trainer.fit()
        return float(summary["best_val_acc"]), model

    def _train_stage(
        self,
        sample_ids: List[int],
        init_seed: int,
        subdir: Optional[str] = None,
    ) -> float:
        """仅返回精度的训练阶段便捷接口。"""
        acc, _ = self._fit_stage(sample_ids, init_seed=init_seed, subdir=subdir)
        return acc
