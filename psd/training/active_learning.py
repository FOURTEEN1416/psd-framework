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

from psd.data.interpet4d import resample_to_fixed_t
from psd.data.smq_input import _normalize_framewise
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig

logger = logging.getLogger(__name__)

_EPS = 1e-12

# 归一化复用 P0.2 owner 实现（smq_input），本模块仅重导出别名，不重复实现
_framewise_normalize = _normalize_framewise


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
    return softmax_np(predict_logits(model, samples, batch_size=batch_size, device=device))


@torch.no_grad()
def predict_logits(
    model,
    samples: Sequence[Dict],
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """样本列表 → (N, C) 原始 logits（饱和诊断用）。"""
    model = model.to(device).eval()
    out: List[np.ndarray] = []
    for s in range(0, len(samples), batch_size):
        chunk = samples[s : s + batch_size]
        kpts = torch.stack([
            x["keypoints"].float() if torch.is_tensor(x["keypoints"])
            else torch.from_numpy(np.asarray(x["keypoints"], dtype=np.float32))
            for x in chunk
        ]).to(device)
        out.append(model(kpts)[0].cpu().numpy())
    if not out:
        raise ValueError("samples 为空，无法推理")
    return np.concatenate(out, axis=0).astype(np.float64)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """数值稳定 softmax（沿最后一维）。"""
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


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
        init_from_ckpt: Optional[Dict] = None,
    ):
        """warm-start 扩展（W23 预注册协议）:

        Args:
            init_from_ckpt: 可选 state_dict；提供时**每个预算点**的模型在 trainer.fit()
                前加载该权重（从同一外部先验起步，差异只来自各自标注集）；
                None 时保持 W14 冷启动行为（随机初始化）。优化器状态不随 ckpt 加载
                （全量训练收敛态动量对小样本微调是噪声源），AdamW 全新构建。
        """
        self.build_model = build_model
        self.pool = pool_samples
        self.val = val_samples
        self.template_cfg = train_config
        self.budgets = sorted(int(b) for b in budgets)
        self.device = device
        self.init_from_ckpt = init_from_ckpt
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
        """给定标注 id 训练一阶段，返回 (best_val_acc, model)。

        warm-start（init_from_ckpt 提供时）: 加载先验权重后再 fit（W23 协议）;
        冷启动（默认）: 固定初始化种子随机起步（W14 协议）。
        """
        torch.manual_seed(init_seed)
        samples = [self.pool[i] for i in sample_ids]

        cfg = replace(
            self.template_cfg,
            batch_size=min(self.template_cfg.batch_size, len(samples)),
            output_dir=str(Path(self.template_cfg.output_dir) / subdir) if subdir else self.template_cfg.output_dir,
        )
        model = self.build_model()
        if self.init_from_ckpt is not None:
            model.load_state_dict(self.init_from_ckpt)
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


# ---------------------------------------------------------------------------
# P0.4 真实池熵打分（best.pt 迁移代理排序；公开真实层口径）
# ---------------------------------------------------------------------------

def clip_segment_to_stgcn_input(
    kp_world: np.ndarray,
    start: int,
    end: int,
    T: int = 30,
) -> np.ndarray:
    """原始 clip 骨架切片 → ST-GCN 输入 (T,24,3).

    链路: 切 [start,end) → 逐帧归一化（复用 P0.2 owner 实现）→ 重采样 T 帧。
    NaN 段显式抛错（由上层决定跳过并计数，禁止静默）。
    """
    seg = np.asarray(kp_world, dtype=np.float32)[int(start) : int(end)]
    if seg.shape[0] == 0:
        raise ValueError(f"空片段 [{start},{end})")
    if not np.isfinite(seg).all():
        raise ValueError(f"片段 [{start},{end}) 含非有限值")
    seg = _framewise_normalize(seg)
    return resample_to_fixed_t(seg, target_t=T)


def make_clip_loader(smal_npy_dir: str | Path) -> Callable[[str], Dict]:
    """生产用 loader 工厂：按 {smal_npy_dir}/{clip_id}.npz 约定解析。"""
    from psd.data.interpet4d import load_clip

    root = Path(smal_npy_dir)

    def _load(clip_id: str) -> Dict:
        return load_clip(root / f"{clip_id}.npz")

    return _load


def score_real_pool(
    pool_entries: List[Dict],
    load_clip_fn: Callable[[str], Dict],
    model,
    budgets: Sequence[int],
    T: int = 30,
    device: str = "cpu",
    batch_size: int = 32,
) -> Dict:
    """对 P0.4 移交池做熵打分，产出排序清单与分布统计。

    口径声明: 合成域迁移模型的不确定性代理排序，供人工标注优先级参考；
    不产生也不得作为精度数字汇报（公开真实层无 22 类行为 GT）。
    NaN 片段跳过并计数（n_skipped 显式登记，不静默剔除）。
    """
    feats: List[np.ndarray] = []
    metas: List[Dict] = []
    n_skipped = 0
    for e in pool_entries:
        try:
            kp = load_clip_fn(e["clip_id"])["kp_world"]
            x = clip_segment_to_stgcn_input(kp, e["start_frame"], e["end_frame"], T=T)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            n_skipped += 1
            logger.warning("[pool] 跳过 %s: %s", e.get("clip_id"), exc)
            continue
        feats.append(x)
        metas.append(e)

    logits = predict_logits(
        model, [{"keypoints": f} for f in feats],
        batch_size=batch_size, device=device,
    )
    probs = softmax_np(logits)
    scores = entropy_scores(probs)

    # 饱和诊断（负结果显式登记，禁止静默）：top1-top2 边际与熵退化标志
    top2 = np.sort(logits, axis=-1)[:, ::-1]
    margins = top2[:, 0] - top2[:, 1]
    margin_stats = {
        "mean": float(margins.mean()),
        "min": float(margins.min()),
        "max": float(margins.max()),
    }
    entropy_degenerate = bool(scores.max() < 1e-3)

    order = np.argsort(-scores, kind="stable")
    ranked_ids = [str(metas[i].get("clip_id", i)) for i in order]
    ranked_records = [
        {
            "clip_id": str(metas[i].get("clip_id", i)),
            "entropy": float(scores[i]),
            "pseudo_label": metas[i].get("pseudo_label"),
            "kappa_margin": metas[i].get("kappa_margin"),
        }
        for i in order
    ]
    topk = {str(int(b)): ranked_ids[: min(int(b), len(ranked_ids))] for b in budgets}
    stats = {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "q25": float(np.percentile(scores, 25)),
        "q50": float(np.percentile(scores, 50)),
        "q75": float(np.percentile(scores, 75)),
    }
    return {
        "n_scored": len(feats),
        "n_skipped": n_skipped,
        "entropy_stats": stats,
        "logit_margin_stats": margin_stats,
        "entropy_degenerate": entropy_degenerate,
        "ranked_ids": ranked_ids,
        "ranked_records": ranked_records,
        "topk": topk,
    }
