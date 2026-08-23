"""P0.1 训练处理器 — external/AimCLR 官方 AimCLR_Processor 的本仓适配子类。

唯一差异：跳过官方 PT_Processor.load_model() 中的 weights_init
（Conv/Linear 全量 N(0, 0.02) 小方差初始化）。

依据（2026-08-23 诊断实验链，详见 reports/p01-aimclr-2026-08-23.md）：
- 该初始化使前向信号逐层衰减，表征落入 InfoNCE 无法逃逸的 cone；
- 无论真实/随机数据、增强开关、SGD/Adam，loss 均平台于 2*log(queue_size)；
- 移除该初始化后同一循环 300 步内特征余弦相似度 1.00 → 0.14，loss 正常下降。
其余预训练逻辑（Feeder_triple 增强、MoCo 队列、DDM 损失、NN mining）
保持官方实现原样，不在本仓复制第二份。
"""
import sys
from pathlib import Path

_AIMCLR_ROOT = Path(__file__).resolve().parents[2] / "external" / "AimCLR"
for _p in (str(_AIMCLR_ROOT), str(_AIMCLR_ROOT / "torchlight")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch.nn as nn  # noqa: E402
from processor.pretrain_aimclr import AimCLR_Processor  # noqa: E402


class P01AimCLRProcessor(AimCLR_Processor):
    def load_model(self):
        # 与官方 PT_Processor.load_model 一致，仅省略 self.model.apply(weights_init)
        self.model = self.io.load_model(self.arg.model, **(self.arg.model_args))
        self.loss = nn.CrossEntropyLoss()
        self.re_criterion = nn.L1Loss(reduction="none")
