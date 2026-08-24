"""ST-GCN+BC 常量定义（W12 兼容层）.

从 psd.data.synth_stgcn 重新导出，避免循环依赖。
"""
from psd.data.synth_stgcn import ALL_BEHAVIORS_22, NUM_CLASSES, NUM_JOINTS  # noqa: F401

__all__ = ["ALL_BEHAVIORS_22", "NUM_CLASSES", "NUM_JOINTS"]
