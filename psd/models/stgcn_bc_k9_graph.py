"""ST-GCN+BC K9Graph — 24 节点犬类骨架拓扑（pyskl 兼容）.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/k9_graph.py`（只读参考）
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from psd.models.stgcn_bc_constants import (
    NUM_KEYPOINTS,
    WITHERS, THROAT, NOSE, CHIN,
    LEFT_EAR_BASE, RIGHT_EAR_BASE,
    LEFT_EAR_TIP, RIGHT_EAR_TIP,
    LEFT_EYE, RIGHT_EYE,
    TAIL_START, TAIL_END,
    FRONT_LEFT_PAW, FRONT_LEFT_KNEE, FRONT_LEFT_ELBOW,
    FRONT_RIGHT_PAW, FRONT_RIGHT_KNEE, FRONT_RIGHT_ELBOW,
    REAR_LEFT_PAW, REAR_LEFT_KNEE, REAR_LEFT_ELBOW,
    REAR_RIGHT_PAW, REAR_RIGHT_KNEE, REAR_RIGHT_ELBOW,
)


class K9Graph:
    """犬类 24 节点骨架拓扑（pyskl 兼容）."""

    NODE_NAMES: List[str] = [
        "front_left_paw", "front_left_knee", "front_left_elbow",
        "rear_left_paw", "rear_left_knee", "rear_left_elbow",
        "front_right_paw", "front_right_knee", "front_right_elbow",
        "rear_right_paw", "rear_right_knee", "rear_right_elbow",
        "tail_start", "tail_end",
        "left_ear_base", "right_ear_base",
        "nose", "chin",
        "left_ear_tip", "right_ear_tip",
        "left_eye", "right_eye",
        "withers", "throat",
    ]

    RAW_OUTWARD_EDGES: List[Tuple[int, int]] = [
        (WITHERS, THROAT),
        (THROAT, NOSE),
        (NOSE, CHIN),
        (NOSE, LEFT_EAR_BASE),
        (NOSE, RIGHT_EAR_BASE),
        (LEFT_EAR_BASE, LEFT_EAR_TIP),
        (RIGHT_EAR_BASE, RIGHT_EAR_TIP),
        (NOSE, LEFT_EYE),
        (NOSE, RIGHT_EYE),
        (WITHERS, TAIL_START),
        (TAIL_START, TAIL_END),
        (WITHERS, FRONT_LEFT_ELBOW),
        (FRONT_LEFT_ELBOW, FRONT_LEFT_KNEE),
        (FRONT_LEFT_KNEE, FRONT_LEFT_PAW),
        (WITHERS, FRONT_RIGHT_ELBOW),
        (FRONT_RIGHT_ELBOW, FRONT_RIGHT_KNEE),
        (FRONT_RIGHT_KNEE, FRONT_RIGHT_PAW),
        (WITHERS, REAR_LEFT_ELBOW),
        (REAR_LEFT_ELBOW, REAR_LEFT_KNEE),
        (REAR_LEFT_KNEE, REAR_LEFT_PAW),
        (WITHERS, REAR_RIGHT_ELBOW),
        (REAR_RIGHT_ELBOW, REAR_RIGHT_KNEE),
        (REAR_RIGHT_KNEE, REAR_RIGHT_PAW),
    ]

    def __init__(self) -> None:
        self.num_nodes: int = NUM_KEYPOINTS
        self.root: int = WITHERS
        self.outward: List[Tuple[int, int]] = list(self.RAW_OUTWARD_EDGES)
        self.inward: List[Tuple[int, int]] = [(v, u) for (u, v) in self.outward]
        self.parent: np.ndarray = np.full(self.num_nodes, -1, dtype=np.int64)
        for (p, c) in self.outward:
            self.parent[c] = p
        self.adjacency: np.ndarray = self._build_adjacency()
        self.partition_labeling: List[Tuple[int, int]] = self._build_partition()

    def _build_adjacency(self) -> np.ndarray:
        adj = np.zeros((self.num_nodes, self.num_nodes), dtype=np.int8)
        for (u, v) in self.outward:
            adj[u, v] = 1
            adj[v, u] = 1
        np.fill_diagonal(adj, 1)
        return adj

    def _build_partition(self) -> List[Tuple[int, int]]:
        labeling: List[Tuple[int, int]] = []
        for v in range(self.num_nodes):
            if v == self.root:
                labeling.append((v, 0))
            else:
                is_inward = any(v == neighbor for (neighbor, _) in self.inward if neighbor == v)
                is_outward = any(v == neighbor for (_, neighbor) in self.outward if neighbor == v)
                if is_inward:
                    labeling.append((v, 1))
                elif is_outward:
                    labeling.append((v, 2))
                else:
                    labeling.append((v, 0))
        return labeling

    def get_bones(self) -> List[Tuple[int, int]]:
        return [(c, p) for (p, c) in self.outward]

    def summary(self) -> str:
        return (
            f"K9Graph (24-node canine skeleton topology)\n"
            f"  Root: {self.NODE_NAMES[self.root]} (idx={self.root})\n"
            f"  Nodes: {self.num_nodes}\n"
            f"  Edges: {len(self.outward)} (directed outward)"
        )

    def __repr__(self) -> str:
        return (
            f"K9Graph(num_nodes={self.num_nodes}, root={self.NODE_NAMES[self.root]}, "
            f"edges={len(self.outward)})"
        )
