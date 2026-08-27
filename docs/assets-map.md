# Assets Map — PSYD-Framework × K9-Training-System 跨仓复用映射

> **Owner**: `docs/assets-map.md`（本仓跨仓代码复用的唯一映射 truth）
> **依据**: AGENTS.md 规则 #6（跨仓只允许文档指针；代码复用走本 map 显式移植）
> **编制日期**: 2026-08-24
> **版本**: v0.1（W11 阻断解除版）

---

## 0. 所有者声明

本文档是 **PSD-Framework 仓内跨仓代码复用的唯一权威映射 truth**。

- AGENTS.md 规则 #6 禁止跨仓 import；所有复用必须通过本 map 显式登记后移植
- HANDOVER §6 资产指针与本 map 冲突时以本 map 为准
- ADR-0002 裁决②引用的"assets-map"即本文档
- 任何新增复用关系必须先在本 map 登记，再实施移植

**禁止重复 truth**：22 类标签、关键点索引等常量仅在 K9 仓权威定义，本 map 登记引用关系而非另抄一份。

---

## 1. 22 类行为标签清单（权威来源 = K9 仓 `constants.py`）

> 全仓第一份 22 类权威记录。`DATA_LOCATIONS.md` 合成层小节、`configs/p05_*.yaml` 均引用此清单，**禁止另抄一份产生重复 truth**。

| idx | 英文名 | 中文名 | P 层级 | FCI-IGP 阶段 | 来源文件 |
|-----|--------|--------|--------|-------------|---------|
| 0 | sit | 坐 | P0 | B | K9 `backend/ml/behavior/constants.py` |
| 1 | down | 卧 | P0 | B | 同上 |
| 2 | stand | 立 | P0 | B | 同上 |
| 3 | heel | 随行 | P0 | B | 同上 |
| 4 | sit_up | 坐立 | P0 | B | 同上 |
| 5 | stay | 停留 | P0 | B | 同上 |
| 6 | bark | 叫 | P0 | B | 同上 |
| 7 | bite | 咬 | P0 | C | 同上 |
| 8 | track | 追踪 | P1 | A | 同上 |
| 9 | alert_sit | 示警坐 | P1 | A | 同上 |
| 10 | alert_down | 示警卧 | P1 | A | 同上 |
| 11 | apprehend | 扑咬 | P1 | C | 同上 |
| 12 | escort | 押解 | P1 | C | 同上 |
| 13 | obstacle | 障碍穿越 | P1 | B | 同上 |
| 14 | recall | 返回 | P1 | B | 同上 |
| 15 | watch | 警戒 | P1 | C | 同上 |
| 16 | guard | 守卫 | P2 | C | 同上 |
| 17 | release | 放口 | P2 | C | 同上 |
| 18 | retrieve | 衔取 | P2 | B | 同上 |
| 19 | jump | 跳跃 | P2 | B | 同上 |
| 20 | scale | 攀登 | P2 | B | 同上 |
| 21 | search_blind | 搜索盲区 | P2 | A | 同上 |

**层级统计**：P0=8 / P1=8 / P2=6，总 22 类。
**科目映射**：A(追踪)=3 / B(服从)=12 / C(护卫)=7。

---

## 2. 24 关键点索引（权威来源 = K9 仓 `constants.py`）

| idx | 英文名 | 中文名 | 分组 |
|-----|--------|--------|------|
| 0 | front_left_paw | 前左爪 | 前左肢 |
| 1 | front_left_knee | 前左膝 | 前左肢 |
| 2 | front_left_elbow | 前左肘 | 前左肢 |
| 3 | rear_left_paw | 后左爪 | 后左肢 |
| 4 | rear_left_knee | 后左膝 | 后左肢 |
| 5 | rear_left_elbow | 后左肘 | 后左肢 |
| 6 | front_right_paw | 前右爪 | 前右肢 |
| 7 | front_right_knee | 前右膝 | 前右肢 |
| 8 | front_right_elbow | 前右肘 | 前右肢 |
| 9 | rear_right_paw | 后右爪 | 后右肢 |
| 10 | rear_right_knee | 后右膝 | 后右肢 |
| 11 | rear_right_elbow | 后右肘 | 后右肢 |
| 12 | tail_start | 尾根 | 尾部 |
| 13 | tail_end | 尾尖 | 尾部 |
| 14 | left_ear_base | 左耳根 | 头部 |
| 15 | right_ear_base | 右耳根 | 头部 |
| 16 | nose | 鼻 | 头部 |
| 17 | chin | 下巴 | 头部 |
| 18 | left_ear_tip | 左耳尖 | 头部 |
| 19 | right_ear_tip | 右耳尖 | 头部 |
| 20 | left_eye | 左眼 | 头部 |
| 21 | right_eye | 右眼 | 头部 |
| 22 | withers | 鬐甲 | 躯干 |
| 23 | throat | 喉咙 | 躯干 |

---

## 3. 移植映射表

每行五列：**源文件(K9 相对路径)** → **目标路径(本仓)** → **职责** → **移植方式** → **测试锚点(ps tests 文件)**。

### 3.1 核心模型栈（ST-GCN+BC 主干）

| 源文件 | 目标路径 | 职责 | 移植方式 | 测试锚点 |
|--------|---------|------|---------|---------|
| `backend/ml/behavior/stgcn_bc/k9_graph.py` | `psd/models/stgcn_k9_graph.py` | 24 节点犬类骨架拓扑（pyskl 兼容） | **复制+命名空间适配**：改 import 路径，保留所有公开 API（K9Graph 类 + NODE_NAMES + RAW_OUTWARD_EDGES） | `tests/test_stgcn_k9_graph.py` |
| `backend/ml/behavior/stgcn_bc/stgcn.py` | `psd/models/stgcn_backbone.py` | ST-GCN 主干（Spatial GCN + Temporal Conv + Residual） | **复制+适配**：改 K9Graph import 为本仓路径；输入适配 (B,T,V,C) ↔ (B,C,T,V) 转置 | `tests/test_stgcn_backbone.py` |
| `backend/ml/behavior/stgcn_bc/bc_head.py` | `psd/models/stgcn_bc_head.py` | BC 头（分类头 + 边界检测头联合） | **复制+适配**：无 K9 仓专有依赖，直接移植 | `tests/test_stgcn_bc_head.py` |
| `backend/ml/behavior/stgcn_bc/loss.py` | `psd/training/stgcn_loss.py` | 联合损失 L = L_cls + 0.3·L_boundary | **复制**：纯 PyTorch，零依赖 | `tests/test_stgcn_loss.py` |
| `backend/ml/behavior/stgcn_bc/model.py` | `psd/models/stgcn_bc.py` | 整体模型（backbone + BCHead + Loss 集成） | **复制+适配**：改 imports 指向本仓模块；`build_stgcn_bc()` 便捷函数保留 | `tests/test_stgcn_bc_model.py` |

### 3.2 数据与合成管线

| 源文件 | 目标路径 | 职责 | 移植方式 | 测试锚点 |
|--------|---------|------|---------|---------|
| `backend/ml/behavior/stgcn_bc/dataset.py`（仅 `_generate_clip` / `make_synthetic_dataset` / `save_synthetic_dataset`） | `psd/data/synth_stgcn.py` | 合成数据生成器（22 类 × N 样本 + 姿态模板 + 时序噪声） | **选择性复制**：仅移植合成生成函数；保留 `STGCNBCDataset` 真实数据类但不激活（真实数据为 W12 消费） | `tests/test_synth_stgcn.py` |
| `backend/ml/behavior/stgcn_bc/data_adapter.py` | `psd/data/stgcn_data_adapter.py` | YOLO26-pose ↔ pyskl 格式适配 | **评估后决定**：仅当 W12 需要真实数据加载时才移植；本窗口可跳过 | —（暂不移植） |

### 3.3 训练与评估

| 源文件 | 目标路径 | 职责 | 移植方式 | 测试锚点 |
|--------|---------|------|---------|---------|
| `backend/ml/behavior/stgcn_bc/trainer.py` | `psd/training/train_stgcn_bc.py` | 训练器（AMP + cosine LR + early stopping + checkpoint） | **复制+适配**：改 imports；输出 dict 接口对齐 W11 规范（`train(config_path) -> {best_val_acc, epochs_run, ckpt_path}`） | `tests/test_train_stgcn_bc.py` |
| `backend/ml/behavior/stgcn_bc/__init__.py` | `psd/models/__init__.py`（增量） | 本仓模型模块导出 | **增量编辑**：append `from .stgcn_bc import STGCNBC, build_stgcn_bc` 等 | 无需单独测试（import smoke） |

### 3.4 常量引用（不复制，仅指针）

| 源文件（K9） | 引用方式 | 说明 |
|-------------|---------|------|
| `backend/ml/behavior/constants.py` | **只读引用**（通过本 map 指针） | 22 类定义 + 关键点索引的权威来源；本仓不复制，由 `psd/data/stgcn_labels.py` 的 docstring 引用本 map §1/§2 |
| `backend/ml/behavior/stgcn_bc/labels.py` | **只读引用** | 标签映射辅助函数；本仓 `psd/data/stgcn_labels.py` 实现最小接口（`NUM_BEHAVIORS` + `ALL_BEHAVIORS_22` + `BEHAVIOR_TO_IDX`） |

---

## 4. 移植纪律

1. **K9 仓全程只读**：不修改任何 `.py` 文件；`.venv`/`__pycache__` 不同步
2. **单一真相**：22 类清单、关键点索引在本 map §1/§2 登记；`DATA_LOCATIONS.md` 合成层小节与 p05 config 均引用此处，禁止另抄一份
3. **测试先行**（TDD）：每个新模块先写 `tests/test_*.py` 断言形状/域/确定性，后实现
4. **接口收敛**：本仓训练入口统一为 `train(config_path: str) -> dict`；checkpoint 存 `runs/p05_stgcn_bc/models/`
5. **gitignore 合规**：`data/synthetic/` 大文件不入仓，目录内放 `_manifest.json` 登记样本数/类别分布/生成命令/种子

---

## 5. K9-Training-System → PSD-Framework Pointer Map (Audit v1.0, 2026-08-24)

> 此板块为本次审计临时登记，注记 k9 仓资产的移植指向与状态。
> 仅作文档指针，不修改任何 `.py` 文件。符合 AGENTS.md §9 “跨仓只允许文档指针；代码复用走本 map 显式移植”。

| K9 资产 | PSD 指向路径 | 移植状态 | 负责人 | 备注 |
|---------|-------------|---------|--------|------|
| stgcn_bc/ 主干（model.py, stgcn.py, bc_head.py, loss.py） | psd/models/stgcn_*.py / psd/data/synth_stgcn.py / psd/training/train_stgcn_bc.py | ✅ 已移植（W11 2026-08-24 commit `580460f`；2026-08-27 接手核查：全部落点文件在仓且模块直连 import 通过，P0.5 合成层 97.27% 即在其上） | — | K9 仓保持只读 |
| constants.py（22类行为 + 24关键点索引） | psd/data/stgcn_labels.py（docstring 引用 §1/§2） | ✅ 已登录（§1/§2） | — | 全仓第一份 22 类权威记录；所有 YAML / config 均引用此处；禁止另抄一份产生重复 truth |
| FCI-IGP 评分卡 YAML | psd/scoring/fci_igp.yaml | ⏳ 待移植（W5：YAML 权重 + Schema 扩展） | — | 结构兼容（7维权重 + 5级评级 + DQ 硬约束）；直接移植，无 celery 依赖 |
| API `/api/llm/explain` | psd/api/llm_explain.py | ⏳ 待移植（W6：产品线保留 k9，研究线移交 psd） | — | LLM 双阶段见 ADR 0011 v1.1；实验室阶段 Agnes API，实用期本地 Llama 3.2 1B；代码环境变量解耦 |
| 22类 / 24关键点 导入辅助函数 | psd/models/__init__.py（增量编辑） | ⏳ 待移植（W7：append 导出语句；注：本仓 psd/models 现为 namespace package 无 __init__.py，子模块直连导入可用，此导出为可选收敛项非阻塞） | — | `from .stgcn_bc import STGCNBC, build_stgcn_bc` 等；import smoke test 即可验证 |
| 合成数据生成函数 | psd/data/synth_stgcn.py | ✅ 已移植（W11 随主干进仓；2026-08-27 接手核实在仓） | — | 仅移植合成函数；真实数据类 `STGCNBCDataset` 保留原路径但不激活（W12 消费） |

---

## 6. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-24 | W11 阻断解除：首次完整映射；登记 22 类权威清单（§1）+ 24 关键点索引（§2）+ 11 项移植任务（§3）+ K9→PSD 指针审计（§5） |
