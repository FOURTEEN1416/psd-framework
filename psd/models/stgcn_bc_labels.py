"""ST-GCN+BC 22 类行为标签映射.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/labels.py`（只读参考）
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from psd.models.stgcn_bc_constants import (
    ALL_BEHAVIORS_22, NUM_BEHAVIORS_22,
    P0_BEHAVIORS, P1_BEHAVIORS, P2_BEHAVIORS,
    BEHAVIOR_NAMES_CN, BEHAVIOR_SUBJECTS,
)


# 行为名称 → 索引
BEHAVIOR_TO_IDX: Dict[str, int] = {
    name: idx for idx, name in enumerate(ALL_BEHAVIORS_22)
}

# 索引 → 行为名称
IDX_TO_BEHAVIOR: Dict[int, str] = {
    idx: name for name, idx in BEHAVIOR_TO_IDX.items()
}

# 总类别数
NUM_BEHAVIORS: int = NUM_BEHAVIORS_22  # 22

# 按层级分组
P0_IDX: List[int] = [BEHAVIOR_TO_IDX[b] for b in P0_BEHAVIORS]
P1_IDX: List[int] = [BEHAVIOR_TO_IDX[b] for b in P1_BEHAVIORS]
P2_IDX: List[int] = [BEHAVIOR_TO_IDX[b] for b in P2_BEHAVIORS]

# 层级标签
LAYER_LABELS: List[str] = (
    ["P0"] * len(P0_BEHAVIORS)
    + ["P1"] * len(P1_BEHAVIORS)
    + ["P2"] * len(P2_BEHAVIORS)
)

# FCI-IGP 阶段映射
FCI_IGP_STAGE: Dict[str, str] = {
    "sit": "B", "down": "B", "stand": "B", "heel": "B",
    "sit_up": "B", "stay": "B", "bark": "B", "bite": "C",
    "track": "A", "alert_sit": "A", "alert_down": "A",
    "apprehend": "C", "escort": "C", "obstacle": "B",
    "recall": "B", "watch": "C",
    "guard": "C", "release": "C", "retrieve": "B",
    "jump": "B", "scale": "B", "search_blind": "A",
}


def get_behavior_idx(name: str) -> int:
    return BEHAVIOR_TO_IDX[name]


def get_behavior_name(idx: int) -> str:
    return IDX_TO_BEHAVIOR[idx]


def get_behavior_cn(name: str) -> str:
    return BEHAVIOR_NAMES_CN.get(name, name)


def get_layer(idx: int) -> str:
    return LAYER_LABELS[idx]


def get_fci_igp_stage(name: str) -> str:
    return FCI_IGP_STAGE.get(name, "B")


def get_all_labels() -> List[Tuple[int, str, str, str, str]]:
    return [
        (idx, name, get_behavior_cn(name), get_layer(idx), get_fci_igp_stage(name))
        for idx, name in sorted(IDX_TO_BEHAVIOR.items())
    ]


def labels_summary() -> str:
    lines = [
        f"ST-GCN+BC 22-class label mapping",
        f"  Total: {NUM_BEHAVIORS} classes",
        f"  P0 (basic):    {len(P0_IDX)} classes, idx {P0_IDX[0]}-{P0_IDX[-1]}",
        f"  P1 (training): {len(P1_IDX)} classes, idx {P1_IDX[0]}-{P1_IDX[-1]}",
        f"  P2 (advanced): {len(P2_IDX)} classes, idx {P2_IDX[0]}-{P2_IDX[-1]}",
    ]
    return "\n".join(lines)


__all__ = [
    "BEHAVIOR_TO_IDX", "IDX_TO_BEHAVIOR", "NUM_BEHAVIORS",
    "P0_IDX", "P1_IDX", "P2_IDX", "LAYER_LABELS", "FCI_IGP_STAGE",
    "get_behavior_idx", "get_behavior_name", "get_behavior_cn",
    "get_layer", "get_fci_igp_stage", "get_all_labels", "labels_summary",
]
