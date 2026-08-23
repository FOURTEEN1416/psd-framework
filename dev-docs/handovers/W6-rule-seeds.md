# W6 交接文档 — 规则引擎粗标（物理层先验种子）

> **你是 W6 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.2 → `dev-docs/stage-plan.md` §1 技术路线图第一环 → `research/RESEARCH_DATA_BLOCKADE_SOLUTION.md`。

---

## 1. 任务目标（一句话）

实现技术路线图第一环：基于**物理层先验**（运动学规则）对无标注骨架序列生成高置信伪标签种子，为 P0.3 姚青迁移提供锚点来源之一（与 20-50 人工黄金样本并列）。

## 2. 开工前置确认项（第一个动作就是它）

**目标行为类别体系待定，两条路径：**

| 路径 | 数据域 | 类别体系 | 状态 |
|------|--------|---------|------|
| **b【推荐起步】** | InterPet4D smal_npy（路径实锤） | 该数据集原生标注类别 | ✅ 可立即开工 |
| a | K9 合成数据（22 类体系） | stage-plan 的 22 类 | ⏳ 合成数据磁盘路径未在本仓 DATA_LOCATIONS 登记——需用户/ K9 truth 提供口径后启用 |

**先按路径 b 打通管线**，路径 a 作为第二迭代；若你判断可直接拿到合成数据路径，在回复中向用户确认后再做。

## 3. 执行链

1. **规则设计**（物理层先验，写进 `configs/rule_seeds.yaml` 可调参数）：
   - 关节速度阈值 → 静止类 vs 运动类初分
   - 躯干/四肢角度区间 → 姿态类（坐/卧/立型）
   - 质心高度变化率 → 起卧/跳跃型事件
   - 规则触发输出 = 类别 + 置信度分数 + 命中规则 ID
2. **实现**：`psd/data/rule_seeds.py`（纯函数规则引擎，输入 W3 加载器的数组，输出去种子集 NPZ+JSON）
3. **入口脚本**：`scripts/make_rule_seeds.py --config configs/rule_seeds.yaml`
4. **质检**：随机抽 ≥30 个种子样本输出统计表（类别分布/置信度分布/规则命中占比）；若抽到带原生标注的序列可对照算一致率——一致率是核心质量指标
5. **归档**：`reports/rule-seeds-<日期>.md`

## 4. 边界（与 W4/W5 并行互斥）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `psd/data/rule_seeds.py`、`scripts/make_rule_seeds.py`、`configs/rule_seeds.yaml`、`reports/rule-seeds-*`、`data/seeds/**`（生成物，已 ignore）、`dev-docs/stage-plan.md` 不改（本任务非子阶段，产出供 P0.3 用） |
| ❌ 禁触 | 一切 `*smq*` 文件（W4 领地）、`psd/models/`、`psd/training/`、`docs/paper/**`（W5 领地）、`external/**`、`docs/DATA_LOCATIONS.md`、`dev-docs/research/**`、`dev-docs/project-brief.md` |
| 环境 | 复用 `.venv` 只读 import（numpy/pandas 已装）；**如需新装包，先检查是否已存在，确需安装时在回复中说明并接受 pip 锁重试** |

## 5. 完成标准与 Git

- [ ] 种子生成一条命令可复现
- [ ] 质检表落盘 reports/（含一致率或置信度分布证据）
- [ ] TDD：规则函数至少 5 个单测（速度阈值边界/角度归一化等），`pytest psd/data/test_rule_seeds.py` 通过
- [ ] 提交：`feat: 规则引擎粗标——物理层先验种子生成器（P0.3 锚点备料）`；遇 `index.lock` 重试；禁 push

## 6. 升级路径

- 规则覆盖率过低（高置信种子 < 序列总数 10%）→ 记录数据后上报，不自行放宽阈值凑数
- 发现 InterPet4D 原生类别体系与预期不符 → 如实记录结构，停在该步等用户裁决
