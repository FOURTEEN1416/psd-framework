"""ST-GCN+BC penultimate 特征抽取器 — W13-C1 窗口 owner（方案 B 语义桥 Φ'）。

依据 dev-docs/handovers/W13-C1-phaseb-fix.md：
- W12 已训 checkpoint（runs/p05_stgcn_bc_full/best.pt，合成层 val_acc 96.4%）
  的分类头前一层特征作为新语义桥特征空间；
- penultimate 定义 = BCHead.forward 中 fc_cls 之前的全局池化向量：
    feat = head_input.mean(dim=(1, 3, 4))   # (B, M, C, T', V) -> (B, C)
  实现用 forward hook 捕获头输入后复现该池化，不复制模型内部代码；
- 真实段输入必须先 center_keypoints（模型在原点中心模板上训练、无归一化层，
  真实世界坐标含任意平移——不居中则跨域分布错位）；
- 特征对齐（fit/apply）为纯 numpy：μ/σ 在参照侧拟合，目标侧套用，
  方案 B 用 mean-only（任务书 §二.B），方案 A 消融加除 std（§二.A）。

边界合规：checkpoint 只读加载；不改 jia_phaseB_mapper 映射逻辑。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 输入居中

def center_keypoints(kp: np.ndarray) -> np.ndarray:
    """逐段有限关节质心居中 + 非有限值填充为质心。

    尺度保持米制不动（与 ST-GCN+BC 训练分布一致），只消除平移分量。
    整帧全 NaN 时该帧填 0（质心不可估的退化情形兜底）。
    """
    kp = np.asarray(kp, dtype=np.float64)
    finite = np.isfinite(kp)                                          # (T,V,3)
    coords = np.where(finite, kp, 0.0)
    cnt = np.maximum(finite.sum(axis=(0, 1)), 1).astype(np.float64)   # 每轴有限计数
    centroid = coords.sum(axis=(0, 1)) / cnt                          # (3,)
    centered = kp - centroid
    return np.where(np.isfinite(centered), centered, 0.0)


# ---------------------------------------------------------------- 特征对齐

def fit_feature_alignment(ref_emb: np.ndarray) -> dict:
    """在参照分布上拟合逐维均值与标准差（纯 numpy）。

    返回 {"mean": (D,), "std": (D,)}；std 带 1e-8 下限防除零。
    """
    emb = np.asarray(ref_emb, dtype=np.float64)
    if emb.ndim != 2 or len(emb) == 0:
        raise ValueError(f"ref_emb 必须是非空二维矩阵: {emb.shape}")
    if not np.isfinite(emb).all():
        raise ValueError("ref_emb 含非有限值——拒绝拟合（NaN 防线）")
    return {
        "mean": emb.mean(axis=0),
        "std": np.maximum(emb.std(axis=0), 1e-8),
    }


def apply_feature_alignment(
    emb: np.ndarray, stats: dict, use_std: bool = False,
) -> np.ndarray:
    """按拟合统计量对齐特征并 L2 归一（行级）。

    use_std=False：只减均值（方案 B 口径）；True：z-score（方案 A 消融口径）。
    """
    emb = np.asarray(emb, dtype=np.float64)
    if not np.isfinite(emb).all():
        raise ValueError("emb 含非有限值——拒绝对齐（NaN 防线）")
    out = emb - stats["mean"]
    if use_std:
        out = out / stats["std"]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------- 抽取器

class STGCNBCFeatureExtractor:
    """ST-GCN+BC penultimate 特征批量抽取器（torch 依赖封装在本类内）。"""

    def __init__(self, model, device: str = "cpu"):
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.device = device
        self.feature_dim = int(model.backbone.out_channels)
        self._head_input: list = []
        model.head.register_forward_pre_hook(self._capture_head_input)

    @classmethod
    def from_checkpoint(cls, ckpt_path: str | Path, device: str = "cpu",
                        num_classes: int = 22, **arch_kwargs):
        """从 trainer 格式 checkpoint 构建（model_state_dict 严格加载）。

        arch_kwargs 透传 build_stgcn_bc（base_channels/num_stages/tcn_type 等）——
        trainer checkpoint 不内嵌架构信息，加载方必须与训练时架构一致。
        只读消费：不训练、不落盘、不修改权重文件。
        """
        import torch

        from psd.models.stgcn_bc import build_stgcn_bc

        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"checkpoint 不存在: {path}")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck.get("model_state_dict", ck)
        model = build_stgcn_bc(in_channels=3, num_classes=num_classes, **arch_kwargs)
        model.load_state_dict(state, strict=True)
        return cls(model, device=device)

    def _capture_head_input(self, module, args, kwargs=None):
        self._head_input.append(args[0])

    def extract(self, keypoints_batch: np.ndarray) -> np.ndarray:
        """(B,T,24,3) 关键点 → (B,D) penultimate 特征（内部先居中）。

        T 以模型卷积结构自适应；训练口径 T=30，调用方应先 resample 到 30。
        """
        import torch

        kp = np.asarray(keypoints_batch, dtype=np.float32)
        if kp.ndim != 4 or kp.shape[1:] != (kp.shape[1], 24, 3):
            raise ValueError(f"期望 (B,T,24,3): 实际 {kp.shape}")
        # 居中必须在转 tensor 前（numpy 域做 NaN 兜底）；回 float32 与模型权重同型
        kp_centered = np.stack([center_keypoints(kp[i]) for i in range(len(kp))])
        x = torch.from_numpy(np.ascontiguousarray(kp_centered, dtype=np.float32)).to(self.device)

        self._head_input.clear()
        with torch.no_grad():
            self.model(x)
        if not self._head_input:
            raise RuntimeError("hook 未捕获头输入——模型结构可能已变化")
        head_in = self._head_input[-1]
        feats = head_in.mean(dim=(1, 3, 4))     # 与 BCHead.forward 池化完全一致
        out = feats.cpu().numpy().astype(np.float64)
        self._head_input.clear()
        if not np.isfinite(out).all():
            raise ValueError("抽取特征含非有限值——拒绝输出（NaN 防线）")
        return out
