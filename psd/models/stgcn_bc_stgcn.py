"""ST-GCN+BC ST-GCN 主干网络.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/stgcn_bc/stgcn.py`（只读参考）
"""
from __future__ import annotations

import copy as cp
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from psd.models.stgcn_bc_k9_graph import K9Graph


def build_spatial_adjacency(graph: K9Graph) -> torch.Tensor:
    """构建 ST-GCN 空间分区邻接矩阵 (3, V, V)."""
    V = graph.num_nodes
    self_link = [(i, i) for i in range(V)]
    inward = graph.inward
    outward = graph.outward

    def edge2mat(link: List[Tuple[int, int]]) -> np.ndarray:
        A = np.zeros((V, V), dtype=np.float32)
        for i, j in link:
            A[j, i] = 1.0
        return A

    def normalize_digraph(A: np.ndarray) -> np.ndarray:
        Dl = np.sum(A, axis=0)
        Dn = np.zeros_like(A)
        for i in range(V):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i] ** (-1)
        return A @ Dn

    I_mat = edge2mat(self_link)
    In_mat = normalize_digraph(edge2mat(inward))
    Out_mat = normalize_digraph(edge2mat(outward))
    A = np.stack([I_mat, In_mat, Out_mat], axis=0)
    return torch.from_numpy(A).float()


class UnitGCN(nn.Module):
    """空间图卷积."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        adaptive: str = "importance",
        conv_pos: str = "pre",
        with_res: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_subsets = A.size(0)
        self.adaptive = adaptive
        self.conv_pos = conv_pos
        self.with_res = with_res
        assert adaptive in [None, "init", "offset", "importance"]
        assert conv_pos in ["pre", "post"]
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)
        if self.adaptive == "init":
            self.A = nn.Parameter(A.clone())
        else:
            self.register_buffer("A", A)
        if self.adaptive in ["offset", "importance"]:
            self.PA = nn.Parameter(A.clone())
            if self.adaptive == "offset":
                nn.init.uniform_(self.PA, -1e-6, 1e-6)
            elif self.adaptive == "importance":
                nn.init.constant_(self.PA, 1.0)
        if self.conv_pos == "pre":
            self.conv = nn.Conv2d(in_channels, out_channels * self.num_subsets, 1)
        else:
            self.conv = nn.Conv2d(self.num_subsets * in_channels, out_channels, 1)
        if self.with_res:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.down = nn.Identity()

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        n, c, t, v = x.shape
        res = self.down(x) if self.with_res else 0
        A_switch = {None: self.A, "init": self.A}
        if hasattr(self, "PA"):
            A_switch.update({
                "offset": self.A + self.PA,
                "importance": self.A * self.PA,
            })
        A_eff = A_switch[self.adaptive]
        if self.conv_pos == "pre":
            x = self.conv(x)
            x = x.view(n, self.num_subsets, -1, t, v)
            x = torch.einsum("nkctv,kvw->nctw", x, A_eff).contiguous()
        else:
            x = torch.einsum("nctv,kvw->nkctw", x, A_eff).contiguous()
            x = x.view(n, -1, t, v)
            x = self.conv(x)
        return self.act(self.bn(x) + res)


class UnitTCN(nn.Module):
    """时间卷积: kernel_size=9, 在 T 维度做 1D 卷积."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0), stride=(stride, 1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.drop(self.bn(self.conv(x))))


class MSTCN(nn.Module):
    """Multi-Scale Temporal Conv (对齐 pyskl ST-GCN++ mstcn)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        stride: int = 1,
        ms_cfg: Optional[List] = None,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        if ms_cfg is None:
            ms_cfg = [(3, 1), (3, 2), (3, 3), (3, 4), ("max", 3), "1x1"]
        self.ms_cfg = ms_cfg
        num_branches = len(ms_cfg)
        self.num_branches = num_branches
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = nn.ReLU(inplace=True)
        if mid_channels is None:
            mid_channels = out_channels // num_branches
            rem_mid_channels = out_channels - mid_channels * (num_branches - 1)
        else:
            rem_mid_channels = int(mid_channels)
        self.mid_channels = mid_channels
        self.rem_mid_channels = rem_mid_channels
        branches: List[nn.Module] = []
        for i, cfg in enumerate(ms_cfg):
            branch_c = rem_mid_channels if i == 0 else mid_channels
            if cfg == "1x1":
                branches.append(nn.Conv2d(in_channels, branch_c, kernel_size=1, stride=(stride, 1)))
                continue
            assert isinstance(cfg, tuple)
            if cfg[0] == "max":
                branches.append(nn.Sequential(
                    nn.Conv2d(in_channels, branch_c, kernel_size=1),
                    nn.BatchNorm2d(branch_c), self.act,
                    nn.MaxPool2d(kernel_size=(cfg[1], 1), stride=(stride, 1), padding=(1, 0)),
                ))
                continue
            k, d = cfg
            pad = (k + (k - 1) * (d - 1) - 1) // 2
            branches.append(nn.Sequential(
                nn.Conv2d(in_channels, branch_c, kernel_size=1),
                nn.BatchNorm2d(branch_c), self.act,
                nn.Conv2d(branch_c, branch_c, kernel_size=(k, 1), padding=(pad, 0), stride=(stride, 1), dilation=(d, 1)),
            ))
        self.branches = nn.ModuleList(branches)
        tin_channels = mid_channels * (num_branches - 1) + rem_mid_channels
        self.transform = nn.Sequential(
            nn.BatchNorm2d(tin_channels), self.act,
            nn.Conv2d(tin_channels, out_channels, kernel_size=1),
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.drop = nn.Dropout(dropout, inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        branch_outs = [branch(x) for branch in self.branches]
        feat = torch.cat(branch_outs, dim=1)
        feat = self.transform(feat)
        out = self.bn(feat)
        out = self.drop(out)
        return self.act(out + res)


class STGCNBlock(nn.Module):
    """ST-GCN Block: unit_gcn → unit_tcn/mstcn → ReLU (+ residual)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        A: torch.Tensor,
        stride: int = 1,
        residual: bool = True,
        tcn_type: str = "mstcn",
        adaptive: str = "importance",
        gcn_kwargs: Optional[dict] = None,
        tcn_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        gcn_kwargs = gcn_kwargs or {}
        tcn_kwargs = tcn_kwargs or {}
        self.gcn = UnitGCN(in_channels, out_channels, A.clone(), adaptive=adaptive, **gcn_kwargs)
        if tcn_type == "unit_tcn":
            self.tcn = UnitTCN(out_channels, out_channels, 9, stride=stride, **tcn_kwargs)
        elif tcn_type == "mstcn":
            self.tcn = MSTCN(out_channels, out_channels, stride=stride, **tcn_kwargs)
        else:
            raise ValueError(f"Unknown tcn_type: {tcn_type}")
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = UnitTCN(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x: torch.Tensor, A: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = self.residual(x)
        x = self.tcn(self.gcn(x, A)) + res
        return self.relu(x)


class STGCN(nn.Module):
    """ST-GCN 主干网络（PyTorch 原生，pyskl ST-GCN++ 兼容架构）."""

    EPS = 1e-4

    def __init__(
        self,
        graph: Optional[K9Graph] = None,
        in_channels: int = 3,
        base_channels: int = 64,
        data_bn_type: str = "VC",
        ch_ratio: int = 2,
        num_person: int = 1,
        num_stages: int = 10,
        inflate_stages: Optional[List[int]] = None,
        down_stages: Optional[List[int]] = None,
        tcn_type: str = "mstcn",
        adaptive: str = "importance",
    ):
        super().__init__()
        graph = graph or K9Graph()
        A = build_spatial_adjacency(graph)
        self.register_buffer("A_buffer", A)
        self.data_bn_type = data_bn_type
        if data_bn_type == "MVC":
            self.data_bn = nn.BatchNorm1d(num_person * in_channels * A.size(1))
        elif data_bn_type == "VC":
            self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        else:
            self.data_bn = nn.Identity()
        inflate_stages = inflate_stages or [5, 8]
        down_stages = down_stages or [5, 8]
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.ch_ratio = ch_ratio
        self.inflate_stages = inflate_stages
        self.down_stages = down_stages
        modules: List[nn.Module] = []
        if self.in_channels != self.base_channels:
            modules.append(STGCNBlock(in_channels, base_channels, A.clone(), stride=1, residual=False, tcn_type=tcn_type, adaptive=adaptive))
        inflate_times = 0
        cur_channels = base_channels
        for i in range(2, num_stages + 1):
            stride = 1 + (i in down_stages)
            in_c = cur_channels
            if i in inflate_stages:
                inflate_times += 1
            out_c = int(self.base_channels * self.ch_ratio ** inflate_times + self.EPS)
            cur_channels = out_c
            modules.append(STGCNBlock(in_c, out_c, A.clone(), stride=stride, tcn_type=tcn_type, adaptive=adaptive))
        if self.in_channels == self.base_channels:
            num_stages -= 1
        self.num_stages = num_stages
        self.gcn = nn.ModuleList(modules)
        self.out_channels = cur_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播. 输入 (B, T, V, C) 或 (B, M, T, V, C)."""
        if x.ndim == 4:
            x = x.unsqueeze(1)
        N, M, T, V, C = x.size()
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        if self.data_bn_type == "MVC":
            x = self.data_bn(x.view(N, M * V * C, T))
        elif self.data_bn_type == "VC":
            x = self.data_bn(x.view(N * M, V * C, T))
        else:
            x = x.view(N * M, V * C, T)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        for i in range(self.num_stages):
            x = self.gcn[i](x)
        x = x.reshape((N, M) + x.shape[1:])
        return x


__all__ = [
    "build_spatial_adjacency", "UnitGCN", "UnitTCN", "MSTCN",
    "STGCNBlock", "STGCN",
]
