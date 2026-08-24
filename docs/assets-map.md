# Assets Map — 跨仓代码移植唯一映射 truth

> **Owner**: `docs/assets-map.md`（PSD-Framework，此文件）
> **版本**: v1.0 · 2026-08-24 · W11 编制
> **对齐**: AGENTS.md 规则#6（跨仓 import 禁，复用走显式移植）

## 1. 声明

本文档是 **PSD-Framework 与 k9-training-system 之间代码复用的唯一权威映射**。
所有跨仓代码移植必须在此表登记；未登记的路径禁止出现在本仓 `psd/` 中。

**22 类清单权威附录**（从 K9 源码提取，禁止在 DATA_LOCATIONS.md 或配置文件另抄一份）：

```python
ALL_BEHAVIORS_22 = (
    # P0 基础 8 类
    "sit", "down", "stand", "heel",
    "sit_up", "stay", "bark", "bite",
    # P1 训练 8 类
    "track", "alert_sit", "alert_down", "apprehend",
    "escort", "obstacle", "recall", "watch",
    # P2 高级 6 类
    "guard", "release", "retrieve", "jump", "scale", "search_blind",
)
NUM_BEHAVIORS_22 = 22
```

## 2. 文件迁移映射表（13 项）

| # | 源文件 (K9 相对路径) | 目标路径 (本仓) | 职责 | 移植方式 | 测试锚点 |
|---|---------------------|----------------|------|---------|---------|
| 1 | `backend/ml/behavior/constants.py` | `psd/models/stgcn_bc_constants.py` | 24 关键点索引 + 22 类行为常量（P0/P1/P2 分组） | 重实现（纯常量，无依赖） | `psd/models/tests/test_stgcn_bc_constants.py` |
| 2 | `backend/ml/behavior/stgcn_bc/k9_graph.py` | `psd/models/stgcn_bc_k9_graph.py` | K9Graph 24 节点骨架拓扑，pyskl 兼容接口 | 重实现（纯图结构，无 ML 依赖） | `psd/models/tests/test_stgcn_bc_k9_graph.py` |
| 3 | `backend/ml/behavior/stgcn_bc/labels.py` | `psd/models/stgcn_bc_labels.py` | 22 类行为标签映射（idx↔name、层级分组） | 重实现（引用 constants 的 NUM_KEYPOINTS/ALL_BEHAVIORS_22） | `psd/models/tests/test_stgcn_bc_labels.py` |
| 4 | `backend/ml/behavior/stgcn_bc/stgcn.py` | `psd/models/stgcn_bc_stgcn.py` | ST-GCN 主干（PyTorch 原生自研，pyskl 兼容架构） | 复制 + 适配 import（K9Graph→本地） | `psd/models/tests/test_stgcn_bc_model.py` |
| 5 | `backend/ml/behavior/stgcn_bc/bc_head.py` | `psd/models/stgcn_bc_bc_head.py` | BC 头（边界分类联合优化） | 复制（无 K9 依赖） | `psd/models/tests/test_stgcn_bc_model.py` |
| 6 | `backend/ml/behavior/stgcn_bc/loss.py` | `psd/models/stgcn_bc_loss.py` | 联合损失 L = L_cls + 0.3·L_boundary | 复制（无 K9 依赖） | `psd/models/tests/test_stgcn_bc_loss.py` |
| 7 | `backend/ml/behavior/stgcn_bc/model.py` | `psd/models/stgcn_bc_model.py` | STGCNBC 整体模型 | 复制 + 适配 import | `psd/models/tests/test_stgcn_bc_model.py` |
| 8 | `backend/ml/behavior/stgcn_bc/dataset.py` | `psd/data/stgcn_bc_dataset.py` | STGCNBCDataset + 合成数据生成器（make_synthetic_dataset） | 复制 + 适配 import（常量路径变更） | `psd/data/tests/test_stgcn_bc_dataset.py` |
| 9 | `backend/ml/behavior/stgcn_bc/trainer.py` | `psd/training/train_stgcn_bc.py` | STGCNBCTrainer + TrainConfig（含 train(config)->dict 接口） | 复制 + 适配 import + 封装为 train() 函数 | `psd/training/tests/test_stgcn_bc_trainer.py` |
| 10 | `backend/ml/behavior/stgcn_bc/inference.py` | `psd/models/stgcn_bc_inference.py` | STGCNBCInferer（推理器，W12 可能消费） | 复制 + 适配 import（暂不写测试，仅落盘） | — |
| 11 | `backend/ml/behavior/stgcn_bc/export_onnx.py` | `psd/models/stgcn_bc_export_onnx.py` | ONNX 导出（W12 可选） | 复制 + 适配 import（暂不写测试，仅落盘） | — |
| 12 | `external/AimCLR/net/st_gcn.py` | `psd/models/stgcn_bc_reference.py` | 官方 AimCLR ST-GCN（仅参考，不激活） | 复制引用（仅存档对照） | — |

> 注：第 10-12 行（inference/export_onnx/reference）在本窗口仅落盘，未实现完整内容（W12 按需补全）。实际已落盘实现为 #1-#9，共 9 个源文件 + 7 个测试文件。

## 3. 22 类清单附录（权威来源，DATA_LOCATIONS.md 引用此段）

| idx | 英文标签 | 中文 | 层级 | FCI-IGP 阶段 |
|-----|---------|------|------|------------|
| 0 | sit | 坐 | P0 | B |
| 1 | down | 卧 | P0 | B |
| 2 | stand | 立 | P0 | B |
| 3 | heel | 随行 | P0 | B |
| 4 | sit_up | 坐立 | P0 | B |
| 5 | stay | 停留 | P0 | B |
| 6 | bark | 叫 | P0 | B |
| 7 | bite | 咬 | P0 | C |
| 8 | track | 追踪 | P1 | A |
| 9 | alert_sit | 示警坐 | P1 | A |
| 10 | alert_down | 示警卧 | P1 | A |
| 11 | apprehend | 扑咬 | P1 | C |
| 12 | escort | 押解 | P1 | C |
| 13 | obstacle | 障碍穿越 | P1 | B |
| 14 | recall | 返回 | P1 | B |
| 15 | watch | 警戒 | P1 | C |
| 16 | guard | 守卫 | P2 | C |
| 17 | release | 放口 | P2 | C |
| 18 | retrieve | 衔取 | P2 | B |
| 19 | jump | 跳跃 | P2 | B |
| 20 | scale | 攀登 | P2 | B |
| 21 | search_blind | 搜索盲区 | P2 | A |

## 4. K9 仓只读引用（不可触碰其内部）

| K9 仓目录 | 用途 | 说明 |
|-----------|------|------|
| `D:\Desktop\k9-training-system\backend\ml\behavior\` | 只读遍历（Step 0 已读完） | 全程只读，禁止任何写操作 |
| `D:\Desktop\k9-training-system\backend\ml\behavior\stgcn_bc\` | 只读引用（实现蓝本） | K9 仓只读，代码重现在本仓独立包内 |

## 5. 更新规则

- 新增跨仓移植时，必须先在此表登记新行，再写实现代码
- 本表更新须随同具体移植 commit 提交
- 22 类清单以附录为准，任何配置文件/测试只引用附录行号，禁止另建同类表

---

*编制: W11 窗口（歆歆）2026-08-24*
