# -*- coding: utf-8 -*-
"""adapt_common — E10 域适配三线共享件（"让冻结编码器动起来"）。

三线（用户裁决: 三线并行推进）:
  T1 TTA       tta_tent_tier.py       测试时熵最小化，只动 BN 仿射参数（零训练成本）
  T2 LoRA-SSL  lora_continue_tier.py  目标域无标签数据上低秩继续预训练 + 源锚蒸馏
  T3 SupCon    supcon_feedback_tier.py PSD 伪标签回灌物理层（监督对比 + 蒸馏锚）

统一尺: 三线适配后的模型一律回到 labelfree_tier_replication.run_arms()
（与已发布 E9 数字同一实现）重测 (c)/(a)/(b) 三臂，判据同 E9。

PSD 哲学延伸: 物理层不是永久冻结，而是"只在低秩子空间中演化"——
LoRA 分支 B=0 初始化保证适配起点与冻结模型逐 bit 等价。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "external" / "AimCLR"))
sys.path.insert(0, str(REPO / "external" / "AimCLR" / "torchlight"))

from labelfree_tier_replication import (  # noqa: E402  统一尺复用
    BUDGET_FRAC, SEED_MAP, ScaledLR, TIERS, build_model, dump_features,
    extract_backbone, latest_ckpt, run_arms,
)


class Conv2dLoRA(nn.Module):
    """Conv2d 的并行低秩分支: y = conv(x) + scale * B(A(x))。

    A: 1x1 降秩 (cin→r)；B: 与原卷积同 kernel/stride/padding 升秩 (r→cout)。
    B 零初始化 → 注入瞬间与原模型逐 bit 等价（冻结起点的安全保证）。
    """

    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.conv = conv
        kh, kw = conv.kernel_size
        self.lora_A = nn.Conv2d(conv.in_channels, r, kernel_size=1, bias=False)
        self.lora_B = nn.Conv2d(r, conv.out_channels, kernel_size=(kh, kw),
                                stride=conv.stride, padding=conv.padding,
                                dilation=conv.dilation, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = alpha / r

    def forward(self, x):
        return self.conv(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(module: nn.Module, r: int = 8, alpha: float = 16.0) -> int:
    """递归把 module 下所有 nn.Conv2d 替换为 Conv2dLoRA（原地 setattr）。

    返回注入数量。遍历顺序确定 → 对 encoder_q / encoder_k 分别调用时
    参数表顺序一致，AimCLR 的 momentum zip 对齐得以保持。
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            setattr(module, name, Conv2dLoRA(child, r=r, alpha=alpha))
            n += 1
        else:
            n += inject_lora(child, r=r, alpha=alpha)
    return n


def lora_parameters(module: nn.Module) -> list:
    return [p for m in module.modules() if isinstance(m, Conv2dLoRA)
            for p in (m.lora_A.parameters(), m.lora_B.parameters()) for p in p]


def count_params(model: nn.Module) -> tuple:
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    return tr, tt


def copy_params_q_to_k(model) -> None:
    """encoder_q → encoder_k 全量复制（含 LoRA），k 侧全部冻结。"""
    for pq, pk in zip(model.encoder_q.parameters(), model.encoder_k.parameters()):
        pk.data.copy_(pq.data)
        pk.requires_grad = False


def load_frozen_model(tier: str, device: str, ckpt_path=None):
    """加载冻结 encoder（默认该 tier 已发布 pretext ckpt；可指定外部源 ckpt，
    如 NTU60 phaseB 强参照编码器 → 跨域适配）。trainable=0 校验。"""
    n_cls = TIERS[tier]
    ckpt = (Path(ckpt_path).resolve() if ckpt_path else
            latest_ckpt(REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "pretext"))
    model = build_model(n_cls, "cpu")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tr, tt = count_params(model)
    assert tr == 0, "冻结失败"
    print(f"[encoder] ckpt={ckpt} frozen verified: trainable=0/{tt}", flush=True)
    return model, ckpt


def torch_linear_head(feat: np.ndarray, labels: np.ndarray, device: str):
    """ScaledLR(StandardScaler+LogisticRegression) → torch Linear 权重（TENT 用）。"""
    head = ScaledLR().fit(feat, labels)
    W = head.clf.coef_.astype(np.float32)          # (C, 256)
    b = head.clf.intercept_.astype(np.float32)     # (C,)
    mean = head.sc.mean_.astype(np.float32)       # (256,)
    std = head.sc.scale_.astype(np.float32)
    lin = nn.Linear(W.shape[1], W.shape[0], bias=True)
    lin.weight.data = torch.from_numpy(W)
    lin.bias.data = torch.from_numpy(b)
    return lin.to(device), torch.from_numpy(mean).to(device), torch.from_numpy(std).to(device)


def backbone_logits(enc, lin, mean, std, x):
    feat = extract_backbone(enc, x)
    return lin((feat - mean) / std)


def supcon_loss(feat: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1):
    """监督对比损失（SupCon）；labels<0 的样本不参与。批内同类=正样本。

    注意: 对角线置 -inf 后不能用 (logp*pos).sum —— -inf*0=NaN 会污染整行；
    用 torch.where(pos, logp, 0) 安全掩码。
    """
    mask = labels >= 0
    if mask.sum() < 2:
        return feat.new_zeros(())
    feat = F.normalize(feat[mask], dim=1)
    lab = labels[mask]
    sim = feat @ feat.T / temperature
    n = sim.size(0)
    eye = torch.eye(n, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, float("-inf"))
    pos = (lab.unsqueeze(0) == lab.unsqueeze(1)) & ~eye
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    logp_pos = torch.where(pos, logp, torch.zeros_like(logp))
    pos_cnt = pos.sum(1)
    ok = pos_cnt > 0
    if int(ok.sum()) == 0:
        return feat.new_zeros(())
    loss = -logp_pos.sum(1)[ok] / pos_cnt[ok]
    return loss.mean()


def cosine_distill(student_feat: torch.Tensor, teacher_feat: torch.Tensor):
    """特征余弦蒸馏锚: 1 - cos(s, t)，防适配漂移（源域知识锚）。"""
    s = F.normalize(student_feat, dim=1)
    t = F.normalize(teacher_feat, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def adapt_out_dir(line: str, tier: str) -> Path:
    d = REPO / "runs" / "p1_sprint" / f"adapt_{line}_{tier}"
    d.mkdir(parents=True, exist_ok=True)
    return d
