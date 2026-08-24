# W20 任务书 — AK 公开真实层部分类协议（P0.5 第三层口径）

> 窗口: W20（全新独立窗口，不依赖任何旧窗口会话）
> 日期: 2026-08-24 | 编制: 协调者歆歆
> 前序状态: W16 窗口的序数映射**已被证伪并清除**（毒文件已删，commit `8fabca2`）；本窗口从零重建
> ⚠️ 必读: `dev-docs/rescue-plan-2026-08-24.md` §0——含 df_action.xlsx 证伪记录与权威语义映射表

## 1. 任务目标（一句话）

在 Animal Kingdom 犬科公开真实数据上，用**语义正确的部分类协议**（12±2 个可诚实映射的 PSD 类）产出公开真实层的微调评估数字，补全论文 tab2 三层口径的中间列。

## 2. 为什么是"部分类"而不是 22 类（科学事实，不可协商）

`df_action.xlsx`（140 动作真值表，位于 K9 仓 `data\animal_kingdom\action_recognition\annotation\`）证明：
- AK 是野生动物行为体系（含 Flying/Molting/Hatching 等），与 PSD-22 工作犬训练体系仅部分重叠
- **可诚实映射**: sit/down/stand/stay/bark/bite/watch/apprehend/retrieve/scale/jump/track ≈ **12 类**
- **零覆盖（K9 特有）**: heel/sit_up/alert_sit/alert_down/escort/recall/guard/release/search_blind + obstacle(与 scale 同源) ≈ **10 类**
- 任何把这 10 类硬塞给 AK 数据的标签都是捏造——审稿人对照 AK 论文一查即穿

**协议先例**: P0.1 的 dog-ID 代理探针已在论文中建立了「子集口径 + 显式披露」的合法先例（`reports/p01-aimclr-2026-08-23.md` §2），本实验沿用同一披露范式。

## 3. 执行步骤

### Step 1 — 权威映射模块
- 新建 `psd/data/ak_mapping.py`
- 映射表写成显式常量 dict：`AK_ACTION_INDEX -> PSD_CLASS_NAME`，**每一项附一行语义理由**（如 `3: "bark",  # Barking 直译对应`）
- 同步常量 `EXCLUDED_CLASSES: list[str]`（10 个零覆盖类）+ `MAPPING_STRENGTH`（strong/medium/weak 三档，来自 rescue-plan §0 表）
- 多标签视频取主标签规则 = train.csv 标签列表第一项（与 W16 曾用的规则一致，但这次基于正确语义）

### Step 2 — TDD
- 测试文件: `psd/data/tests/test_ak_mapping.py`
- 必测点:
  - spot check: Barking(3)→bark、Sitting(108)→sit、Attacking(1)→apprehend、Jumping(15)→jump
  - excluded 断言: heel/alert_sit/... 全部不在映射值域
  - 反向证伪回归: index 1→sit / 2→down / 5→stand 的旧序数断言必须**抛错或不存在**
  - 多标签取首规则

### Step 3 — 犬科样本量统计（训练前门禁）
- 数据源: K9 仓 `data\animal_kingdom\action_recognition\annotation\train.csv` + `val.csv`（空格分隔宽表，列 `original_vido_id video_id frame_id path labels type`）
- 物种过滤口径: 以 `docs/DATA_LOCATIONS.md` 登记的犬科 329 视频清单为准（若该清单无机器可读形式，从 pose_estimation 注释或既有盘点报告溯源；溯源不到时在报告中登记并采用保守替代口径）
- 统计各 PSD 类的视频数分布 → 写入报告
- **门禁**: 任一类 <10 视频 → 该类标记 untrainable，从训练集剔除并在报告披露；最终可训练类数如实汇报（可能 <12）

### Step 4 — 微调评估（GPU 排队纪律）
- 配置: `configs/p05_public_real_partialclass.yaml`（num_classes=实际可训练类数，注明子集清单与版本）
- 初始化: `runs/p05_stgcn_bc_full/best.pt`（Y 预训练权重；head 层因类数不同需新建——这正是解耦设计的用例之一，可在报告顺带记录）
- 数据格式: 对齐 `psd/data/stgcn_bc_dataset.py` 消费接口；AK 若无骨架关键点则须先核实——**⚠️ 开工前置检查：AK action_recognition 注释是否含关键点坐标？若无骨架数据，本实验退化为「视频级 RGB 路线不可行」结论或改接 pose_estimation 目录的关键点管线，两种情况都先写清现状再动手**
- GPU 被 NTU Phase B 长训独占期间：只做 Step 1-3 + 配置 + TDD；训练段等空闲或夜间错峰
- 结果 JSON: `reports/p05-public-real-partialclass-<日期>.json`

### Step 5 — 报告
- `reports/p05-public-real-partialclass-<日期>.md`: 映射表全文、excluded 披露、样本量分布、精度、三层口径声明（本数字属公开真实层，禁止与合成层 97.3% 混排对比）
- 与 P0.1 同款披露措辞风格

### Step 6 — 回写
- commit: `feat(p05): AK 公开真实层部分类评估——<N>类协议 val_acc <数字>`（中文 Conventional Commits）

## 4. 领地边界

**可写**: `psd/data/ak_mapping.py`、`psd/data/tests/test_ak_mapping.py`、`configs/p05_public_real*`、`reports/p05-public-real-*`、`runs/public_real_*`
**禁触**: `docs/paper/**`、`dev-docs/decisions/**`、`dev-docs/rescue-plan-2026-08-24.md`（只读引用）、`dev-docs/HANDOVER.md`（协调者回写）、K9 仓全部文件（**只读**，绝不修改）、`*ntu*` 文件

## 5. 完成标准

- [ ] 映射模块每项有语义理由，TDD 含反向证伪回归
- [ ] 犬科样本量分布统计落盘，untrainable 类如实剔除
- [ ] 骨架数据可用性前置检查有明确结论（这是开工第一天必须回答的问题）
- [ ] 精度数字有当次运行 JSON 证据，报告含三层口径声明
- [ ] 全仓 pytest 回归不劣于基线

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 独立窗口版任务书建册（W16 证伪后从零重建） |
