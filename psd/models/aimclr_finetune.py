"""AimCLR encoder 微调适配器 — W31 tab3「−自监督预训练」消融两臂模型.

设计依据（dev-docs/handovers/NEXT-BATCH-plan.md W31 节）:
    两臂 {随机初始化, 加载 P0.1 AimCLR 预训练 backbone} 在同数据同预算下对照。
    P0.1 预训练权重（NTU 25 点 / hidden 16 通道 / 256 维输出）与 STGCNBC 骨干
    （K9Graph 24 点 / MSTCN / base 64）形状不相容，直接移植不可能——故两臂
    统一采用 P0.1 同款 st_gcn encoder 架构 + 新分类/边界头，唯一差异变量 =
    encoder 初始权重。

边界合规（AGENTS.md 硬规则 4/5）:
    - external/AimCLR 只读 import（sys.path 注入），不改其内部任何文件；
    - 输入 (B, T, 24, 3) 经「槽 24 零填充」升到 NTU 25 点，复用 P0.1
      reports/p01-aimclr §3 的恒等映射惯例；
    - forward/compute_loss 契约与 STGCNBC 一致，供 STGCNBCTrainer 直接消费。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from psd.training.stgcn_loss import STGCNBCLoss

_AIMCLR_ROOT = Path(__file__).resolve().parents[2] / "external" / "AimCLR"

# 与 configs/p01_aimclr.yaml model_args 对齐 + net/aimclr.py 预训练态改写：
#   ① L34——pretrain=True 时 base_encoder 收 num_class=feature_dim(128)，
#      配置里的 num_class=12 仅非预训练模式生效；
#   ② L44-48——mlp=True 暴力替换 fc → Sequential(Linear(256,256), ReLU, Linear(256,128))。
# 实测 epoch120_model.pt 键集（fc.0/fc.2, 输出 128）与上述一致；本适配器不消费
# 该原生头（分类语义由 cls_head 承担），保留仅为 state_dict strict 同构。
P01_ENCODER_ARGS: Dict = {
    "in_channels": 3,
    "hidden_channels": 16,
    "hidden_dim": 256,
    "num_class": 128,
    "dropout": 0.5,
    "mlp": True,
    "graph_args": {"layout": "ntu-rgb+d", "strategy": "spatial"},
    "edge_importance_weighting": True,
}

NUM_JOINTS_K9 = 24   # 合成数据关节点数（K9Graph 口径）
NUM_JOINTS_NTU = 25  # P0.1 encoder 拓扑（槽 24 为死关节零填充）


def _import_stgcn_model():
    """从 external/AimCLR 只读导入 net.st_gcn.Model（eval_aimclr.py 同款路径注入）。"""
    if not _AIMCLR_ROOT.exists():
        raise ImportError(
            f"external/AimCLR 缺失: {_AIMCLR_ROOT} —— "
            "请从主检出复制（gitignore 资产不随 worktree 走）"
        )
    if str(_AIMCLR_ROOT) not in sys.path:
        sys.path.insert(0, str(_AIMCLR_ROOT))
    from net.st_gcn import Model  # noqa: PLC0415  # noqa: E402

    return Model


def encode_temporal_map(encoder, x: torch.Tensor) -> torch.Tensor:
    """(B,T,V=24,C=3) 关键点 → encoder 时间保持特征图 (B, hidden_dim, T', V_ntu).

    内部流程与 net/st_gcn.py Model.forward 及 eval_aimclr.extract_backbone
    完全一致（data_bn → st_gcn_networks×edge_importance），仅在全局池化前
    停下以保留时间分辨率供边界头消费。
    """
    b, t, v, c = x.shape
    assert v == NUM_JOINTS_K9 and c == 3, f"期望 (B,T,{NUM_JOINTS_K9},3), 实际 {tuple(x.shape)}"

    device = encoder.A.device
    # 槽 24 零填充: 24 → NTU 25 点（P0.1 恒等映射惯例）
    pad = torch.zeros(b, t, 1, c, dtype=x.dtype, device=device)
    x25 = torch.cat([x.to(device), pad], dim=2)              # (B,T,25,3)
    xm = x25.unsqueeze(1)                                    # (B,M=1,T,V,C)
    n, m, _, vv, cc = xm.shape
    xt = xm.permute(0, 1, 3, 4, 2).contiguous()              # (N,M,V,C,T)
    xt = xt.view(n * m, vv * cc, t)
    xt = encoder.data_bn(xt)
    xt = xt.view(n, m, vv, cc, t)
    xt = xt.permute(0, 1, 3, 4, 2).contiguous().view(n * m, cc, t, vv)

    for gcn, importance in zip(encoder.st_gcn_networks, encoder.edge_importance):
        xt, _ = gcn(xt, encoder.A * importance)

    feat = xt.reshape((n, m) + xt.shape[1:])                 # (N,M,C,T',V')
    return feat[:, 0]                                        # M=1 → (B,C,T',V')


def _apply_mlp_replacement(encoder) -> None:
    """复刻 net/aimclr.py L44-48: mlp=True 时 fc → Sequential(Linear, ReLU, fc).

    P0.1 预训练 ckpt 的 encoder_q 即此结构（fc.0/fc.2 键），不复刻则 strict
    加载必然 Missing/Unexpected key 失败。
    """
    dim_mlp = encoder.fc.weight.shape[1]
    encoder.fc = nn.Sequential(
        nn.Linear(dim_mlp, dim_mlp),
        nn.ReLU(),
        encoder.fc,
    )


def _build_encoder(encoder_args: Optional[Dict]):
    """按 P0.1 参数构建 st_gcn.Model encoder（含 mlp 替换态）。"""
    args = dict(P01_ENCODER_ARGS)
    if encoder_args:
        args.update(encoder_args)
    use_mlp = bool(args.pop("mlp", True))
    model_cls = _import_stgcn_model()
    encoder = model_cls(**args)
    if use_mlp:
        _apply_mlp_replacement(encoder)
    return encoder, int(args["hidden_dim"])


class AimCLRFinetune(nn.Module):
    """AimCLR encoder + 分类/边界双头的微调模型（STGCNBCTrainer 兼容契约）.

    forward((B,T,24,3)) → (cls_logits (B,num_classes), boundary_logits (B,T'))
    compute_loss 与 STGCNBC 相同（STGCNBCLoss，时间维自动插值对齐）。
    """

    def __init__(
        self,
        num_classes: int = 22,
        boundary_weight: float = 0.3,
        encoder_args: Optional[Dict] = None,
    ):
        super().__init__()
        self.encoder, hidden_dim = _build_encoder(encoder_args)
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.bnd_head = nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1)
        self.loss_fn = STGCNBCLoss(boundary_weight=boundary_weight)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat_map = encode_temporal_map(self.encoder, x)       # (B,D,T',V')
        pooled = feat_map.mean(dim=(2, 3))                    # (B,D)
        cls_logits = self.cls_head(pooled)                    # (B,num_classes)
        bnd_in = feat_map.mean(dim=3)                         # (B,D,T')
        bnd_logits = self.bnd_head(bnd_in).squeeze(1)         # (B,T')
        return cls_logits, bnd_logits

    def compute_loss(
        self,
        cls_logits: torch.Tensor,
        boundary_logits: torch.Tensor,
        cls_labels: torch.Tensor,
        boundary_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.loss_fn(cls_logits, boundary_logits, cls_labels, boundary_labels)


def load_pretrained_encoder(model: AimCLRFinetune, ckpt_path: str | Path) -> int:
    """把 P0.1 AimCLR ckpt 的 encoder_q.* 权重 strict 加载进 model.encoder.

    Returns:
        加载的张量条数（>0 校验防前缀剥离失败静默空载）。
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"P0.1 预训练 ckpt 不存在: {path}")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    prefix = "encoder_q."
    stripped = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not stripped:
        raise ValueError(f"{path} 中无 encoder_q.* 键——非 P0.1 AimCLR 格式")
    model.encoder.load_state_dict(stripped, strict=True)
    return len(stripped)


def build_arm_model(
    arm: str,
    seed: int,
    pretrained_ckpt: Optional[str | Path] = None,
    num_classes: int = 22,
    boundary_weight: float = 0.3,
) -> AimCLRFinetune:
    """按消融臂构建模型（受控初始化公平性核心）.

    公平性协议: torch.manual_seed(seed) 后按固定顺序构建（encoder→heads），
    同 seed 下两臂的 cls/bnd 头逐位相等；warm 臂随后仅覆盖 encoder 权重。

    Args:
        arm: "scratch"（随机初始化）| "warm"（加载 P0.1 预训练 encoder）
        seed: 本臂训练种子（同时控制初始化与 DataLoader 洗牌）
        pretrained_ckpt: warm 臂必填；scratch 臂必须为 None 或忽略
    """
    if arm not in ("scratch", "warm"):
        raise ValueError(f"未知消融臂: {arm!r}（可选 scratch|warm）")
    torch.manual_seed(seed)
    model = AimCLRFinetune(num_classes=num_classes, boundary_weight=boundary_weight)
    if arm == "warm":
        if pretrained_ckpt is None:
            raise ValueError("warm 臂必须提供 pretrained_ckpt（P0.1 epoch120_model.pt）")
        load_pretrained_encoder(model, pretrained_ckpt)
    return model


__all__ = [
    "AimCLRFinetune",
    "build_arm_model",
    "load_pretrained_encoder",
    "encode_temporal_map",
    "P01_ENCODER_ARGS",
]
