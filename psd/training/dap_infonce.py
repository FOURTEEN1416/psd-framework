# -*- coding: utf-8 -*-
"""P2 域自适应预训练（DAP）— PSD-DAP-PREREG-001 机制实现（ADR 0008）。

在 Y_CKPT（ST-GCN+BC，合成 22 类监督）之上，用 APTv2 canidae 无标签骨架 clip 做
两视图 InfoNCE 续训（表征级域自适应，比 L9 的 AdaBN 二阶统计是更强干预）。
GT 无关：仅消费 clip 的几何，不碰任何标签。增广=时间裁剪±25% + 关节抖动 + 尺度。
model.forward 返回 (logits, features)，对比损失作用于 features（penultimate）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load_backbone(ckpt_path, num_classes: int = 22, device: str = "cpu"):
    """从 trainer checkpoint 构建可训练 ST-GCN+BC（返回 model，train 模式由调用方设）。"""
    import torch
    from psd.models.stgcn_bc import build_stgcn_bc

    ck = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    state = ck.get("model_state_dict", ck)
    model = build_stgcn_bc(in_channels=3, num_classes=num_classes)
    model.load_state_dict(state, strict=True)
    return model.to(device)


def _augment(kp, rng):
    """kp (B,T,24,3) → 一个增广视图。时间随机裁剪回 T + 关节抖动 + 全局尺度。"""
    import torch
    B, T, V, C = kp.shape
    # 时间裁剪 ±25%（保持输出 T：裁一段再线性重采样回 T）
    keep = int(round(T * float(rng.uniform(0.75, 1.0))))
    start = int(rng.integers(0, T - keep + 1))
    seg = kp[:, start:start + keep]  # (B,keep,V,C)
    if keep != T:
        Bk, Vk, Ck = seg.shape[0], seg.shape[2], seg.shape[3]
        flat = seg.reshape(Bk, keep, Vk * Ck).permute(0, 2, 1)  # (B, V*C, keep) 3D
        flat = torch.nn.functional.interpolate(flat.float(), size=T, mode="linear", align_corners=False)
        kp = flat.permute(0, 2, 1).reshape(B, T, Vk, Ck)
    else:
        kp = seg
    # 关节抖动（仅 x,y,z 前三通道，σ=0.02）+ 全局尺度 0.9-1.1
    kp = kp + torch.randn_like(kp) * 0.02
    kp = kp * float(rng.uniform(0.9, 1.1))
    return kp


def info_nce(f1, f2, temperature: float = 0.1):
    """对称 InfoNCE（同 batch 内负样本）。f1,f2 (B,D) L2 归一后。"""
    import torch
    z1 = torch.nn.functional.normalize(f1, dim=1)
    z2 = torch.nn.functional.normalize(f2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, labels)
                  + torch.nn.functional.cross_entropy(logits.t(), labels))


def train_dap(clips: np.ndarray, model, *, epochs: int = 60, batch: int = 32,
              lr: float = 1e-4, temperature: float = 0.1, seed: int = 42,
              device: str = "cuda", log_every: int = 10):
    """在 clips (N,30,24,3) 上两视图 InfoNCE 续训 model（就地更新权重）。返回 loss 曲线。"""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x_all = torch.from_numpy(np.ascontiguousarray(clips, dtype=np.float32)).to(device)
    n = x_all.shape[0]
    curve = []
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0; nb = 0
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            if idx.numel() < 2:
                continue
            base = x_all[idx]
            a1 = _augment(base, rng); a2 = _augment(base, rng)
            _, f1 = model(a1); _, f2 = model(a2)
            loss = info_nce(f1, f2, temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.item()); nb += 1
        avg = tot / max(nb, 1)
        curve.append(round(avg, 4))
        if (ep + 1) % log_every == 0:
            print(f"  [dap] epoch {ep+1}/{epochs} infoNCE={avg:.4f}")
    model.eval()
    return curve
