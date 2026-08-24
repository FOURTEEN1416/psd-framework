"""ST-GCN+BC 模型模块（W11 移植 + W12 测试兼容层）.

导出: STGCNBC, build_stgcn_bc
"""
from psd.models.stgcn_bc import STGCNBC, build_stgcn_bc  # noqa: F401

__all__ = ["STGCNBC", "build_stgcn_bc"]
