# 动捕犬类数据集调研（MOCAP_DATASETS）

> Owner: dev-docs/research/MOCAP_DATASETS.md（C3 路动捕犬调研唯一 truth）
> 任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §2-C3
> 执行窗口: W27（wt/W27，2026-08-25）
> 调研纪律: GitHub-First（AGENTS.md 硬规则1），全程 GitHub 工具链检索发现；官方主源仅做事实核验

## 0. 用途定位声明（硬要求，禁止挪用）

**动捕犬数据在本仓的定位 = 真实运动学先验（kinematic prior），不是行为分类主粮。**

- 可用方向：合成管线保真度拟合（关节角边缘分布/帧间速度谱 → C4 syn_v2 参数）、规则种子物理先验校准（psd/data/rule_seeds.py 的速度/朝向阈值）、warm-start 微调的辅助序列池
- 不可用方向：行为类别监督信号（DogSet 只含无标注运动片段，无 sit/stay/recall 等工作犬语义）；论文中不得作为"真实 K9 行为数据"引用，只能作为 motion prior 引用
- 三层口径归属：公开真实层（辅助资产），汇报时单列，禁止与合成层或真实 K9 层混报

## 1. 调研方法

五条检索轴（用户指定）× GitHub 工具链：

| 检索轴 | 工具调用 | 结果 |
|--------|---------|------|
| CMU mocap animal/dog | repo 检索 ×2 + 镜像仓验证 + 官方 subjects 页核验 | **负结果**（见 §7.1） |
| DogMo | repo 检索 | 命中 BIT-PIE/DogMo 官方项目页 |
| canine motion capture | repo 检索 | 仅学生毕设仓，无数据集 |
| quadruped mocap BVH | repo 检索 + **代码级检索 `"JOINT Tail" filename:.bvh`** | **命中关键线索**：hlcdyy/pan-motion-retargeting 与 Digital-Humans-23/motion-matching 内含犬类 BVH |
| awesome 动捕清单 | repo 检索 ×2 | 均为人形向，无动物专节 |

> 方法论注记：repo 关键词检索对"数据集藏在方法仓库内"的情形命中率低；**代码级检索骨架特征 token**（如四足特有的 `JOINT Tail`）是本轮突破点，后续数据类调研建议保留此轴。

## 2. 候选对比总表（2026-08-25 逐仓验证）

| 数据集 | 物种真实性 | 格式 | V(关节数) | 规模 | fps | 许可 | 获取方式 | 判定 |
|--------|-----------|------|-----------|------|-----|------|---------|------|
| **MANN DogSet**（Zhang & Starke et al., SIGGRAPH 2018） | ✅ 真实犬光学动捕 | BVH | 21 | **51 文件 / 147,541 帧 / ≈41 分钟** | 60 | 研究/教育专用；禁商用禁再分发；University of Edinburgh IP | 作者官网直链 zip（33MB）；GitHub 有部分镜像（Digital-Humans-23/motion-matching 仅 30 条子集） | 🥇 **Top1，已获取并转换** |
| **DogMo**（BIT-PIE，arXiv:2510.24117） | ✅ 真实犬多视角 RGB-D | SMAL 网格拟合（非骨架 BVH） | SMAL 拓扑 | 大规模（官网口径，未取数） | — | CC BY-SA 4.0（项目页声明） | pie-lab.cn/DogMo 门禁下载 | 备选：格式不合 BVH→(T,V,3) 管线，留作未来 mesh 先验扩展 |
| **DeformingThings4D**（ICCV 2021, rabbityl/, 363★） | ❌ 合成动画 | `.anime`（网格顶点偏移）+人形才有 FBX | 无骨架拓扑 | 1972 动画（动物 1772 条/88,137 帧） | — | DT4D ToU（研究用）+代码 CC NC | Google Form 门禁 | 否：网格无骨架，需额外 fitting；门禁摩擦高 |
| **CMU Graphics Lab mocap**（una-dinosauria/cmu-mocap 镜像，188★） | ❌ 全人类科目 | BVH/ASF-AMC | 人形 ~30 | 144 subjects | 120 | 无限制（研究+商用均可） | GitHub 直链 | **否：库内零犬类数据**（§7.1） |
| Barkour（google-deepmind/barkour_robot 等） | 部分（视频基准） | 视频/机器人 RL | — | — | — | 各异 | GitHub | 不适用：机器人敏捷基准，非动捕 |
| Keenan-Laas/Markerless-Motion-Capture-of-Canines | — | 代码仓 | — | — | — | — | GitHub | 否：本科毕设，无可用数据集 |

## 3. Top1 详卡：MANN DogSet

- **来源链**: 论文 *Mode-Adaptive Neural Networks for Quadruped Motion Control*（ACM TOG 37(4), SIGGRAPH 2018, He Zhang⁺/Sebastian Starke⁺/Taku Komura/Jun Saito）→ 官方托管 `https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip`（33MB，51 条 BVH）→ 经 hlcdyy/pan-motion-retargeting（TVCG 2023 PAN 论文）README 交叉确认其权威性
- **拓扑**: 单一统一骨架，21 可动关节 + End Sites；脊柱沿 X 轴水平（四足 rest 型位）；厘米制；60fps（Frame Time 0.0166667s）
- **命名陷阱（防语义事故核心，详见 §4）**: 沿用人形 MotionBuilder 式命名——`LeftArm/RightArm` 是**前腿**上段，`LeftHand/RightHand` 是**前爪**，`UpLeg/Leg/Foot` 是**后腿**三段，`Tail/Tail1` 为尾椎两节
- **规模实测**（本窗当次解压扫描）: 51 文件全部同一拓扑同一 fps；帧数 min=155 / max=13,399 / 总计 147,541（≈2459s）
- **许可**: 数据目录内 LICENSE 原文要点——"only for research or education purposes…not freely available for commercial use or redistribution…the intellectual property belongs to the University of Edinburgh"；学术仓内部使用合规；**禁止再分发原始 BVH**（external/ 已 gitignore，转换产物 pkl 同属派生数据，随 runs/ 本地留存不提交 git）
- **引用义务**: SIGGRAPH 2018 论文 + AI4Animation 仓库

## 4. 关节映射表（MANN BVH 名 → 本仓规范语义名）

> ⚠️ 本表是防语义事故的硬性契约。转换脚本 `scripts/bvh_dogset_to_sequence.py` 中 `JOINT_MAPPING` 为同名 owner；下游消费 pkl 时以 payload 内 `joint_order_canonical` 为准，禁止自行按索引猜语义。
> 事故案例存档：调研初期"CMU subject16=犬"的记忆错误被 OFFSET 形态学比对证伪（§7.1）——任何"看起来像"的隐式映射都不可信。

| 规范序 | 规范名 | MANN BVH 原始名 | 语义 | 备注 |
|-------|--------|----------------|------|------|
| 0 | hips | Hips | 骨盆根 | 6 通道（含平移） |
| 1 | spine_a | Spine | 脊柱前段（髋→胸） | |
| 2 | spine_b | Spine1 | 脊柱胸段 | |
| 3 | neck | Neck | 颈基 | |
| 4 | head | Head | 头 | |
| 5 | tail_a | Tail | 尾椎近端 | 犬特有，人形无 |
| 6 | tail_b | Tail1 | 尾椎远端 | |
| 7 | fl_scapula | LeftShoulder | 左前腿肩胛 | **前腿！** |
| 8 | fl_upper | LeftArm | 左前腿上段（肱骨） | **不是手臂** |
| 9 | fl_lower | LeftForeArm | 左前腿下段（桡尺骨） | |
| 10 | fl_paw | LeftHand | 左前爪 | **不是手** |
| 11 | fr_scapula | RightShoulder | 右前腿肩胛 | |
| 12 | fr_upper | RightArm | 右前腿上段 | |
| 13 | fr_lower | RightForeArm | 右前腿下段 | |
| 14 | fr_paw | RightHand | 右前爪 | |
| 15 | hl_femur | LeftUpLeg | 左后腿股骨 | 后腿仅 3 关节 |
| 16 | hl_tibia | LeftLeg | 左后腿胫骨 | |
| 17 | hl_paw | LeftFoot | 左后爪 | |
| 18 | hr_femur | RightUpLeg | 右后腿股骨 | |
| 19 | hr_tibia | RightLeg | 右后腿胫骨 | |
| 20 | hr_paw | RightFoot | 右后爪 | |

形态学备注：前腿 4 关节（肩胛+肱+桡+爪）/ 后腿 3 关节（股+胫+爪）不对称是 MANN 原始建模决定，如实保留；若下游需对称拓扑须显式重映射并在决策记录留痕。

## 5. 转换管线（已交付）

- **脚本**: `scripts/bvh_dogset_to_sequence.py`（唯一 owner）
  - 通用 BVH 解析（逐关节 CHANNELS 轴序自适应；本库为 Z-X-Y 序，勿套用 CMU 的 Z-Y-X 假设）
  - 正向运动学 FK → 全局关节位置 (T,V=21,3) float32，厘米制原坐标
  - 按名映射（名字集合强校验，不符即拒绝转换）
  - 自测模式 `--self-test`：内置合成 BVH 已知答案验证解析/FK/映射全链路
- **产物（本地，gitignore）**: `runs/data_campaign/mocap/sequences/*.pkl`（51 个，格式 B：keypoints/topology_name/V/fps_or_sampling/source/split/joint_order_canonical）+ `runs/data_campaign/mocap/manifest.jsonl`
- **原始包**: `external/dogset-mann-siggraph2018/raw/`（51 BVH + LICENSE，gitignore）
- **质检语义**: NaN 与骨长变异系数（阈值 5%）为硬失败；单关节帧间峰值速度 >1500cm/s 打 `qc_flag=suspect_glitch`（实测 45/51 条带标记，成因见 §6）

## 6. 数据质量已知问题（诚实披露）

1. **孤立尖峰毛刺（主要问题）**: 全库热帧分析（速度>1200cm/s 阈）：孤立段 ≤2 帧 = **1420 个** vs 持续段 ≥4 帧 = 32 个；峰值集中于前后爪（fl_paw 29 文件 / fr_paw 10 / head 8）。典型如 D1_013 头部单帧瞬移 93cm——判定为源动捕标记遮挡/交换毛刺，非 FK 错误（FK 错误会表现为全关节持续性系统偏差；且骨长 CV=0 恒刚体）。**处置：数据如实保留 + manifest 打标，不做静默清洗；下游规则种子/伪标签环节应加去尖峰滤波（中值或速度截断），该决策留给消费侧窗口。**
2. **无行为标签**: 全部序列无动作语义标注，只有文件名会话编号（D1_XXX_KAN01/02_YYY，KAN01/KAN02 含义官方未注明，不作臆测）。
3. **单只犬泛化边界**: 来源为少量个体（论文采集），品种/体态多样性有限——作先验可用，作分布代表不可。

## 7. 检索负结果与证伪记录

### 7.1 "CMU mocp 有犬类数据"证伪（重要）
- 流传说法："CMU subject 16 = dog"。本窗实测：una-dinosauria/cmu-mocap 镜像 `data/016`（58 条 jump/walk/run BVH）骨架 OFFSET 与人形 subject 001 几乎同比例（腿~16 臂~12 竖直脊柱），且含 LThumb/LeftHandIndex1 等手指链——**是人形**。
- 官方主源核验（mocap.cs.cmu.edu/subjects.php）：144 个科目全部人类；subject 16 描述"run, jump, walk"，无物种标注；28-32/54/55 号仅为"人类模仿动物行为"。
- 结论：**CMU 图形实验室动捕库不含任何真实动物科目**。"CMU mocap animal"轴关闭。

### 7.2 其他空轴
- Truebones（商业动物 BVH 商店）：GitHub 无官方仓库，授权模型不明，放弃。
- "quadruped mocap"/"animal animation BVH"/"awesome animal pose" repo 检索：零有效命中（截至 2026-08-25）。

## 8. 后续建议（均未验证，供排期参考）

1. C4 syn_v2 拟合时消费本产物：逐关节角边缘分布 + 速度谱对齐 DogSet 实测（消费前先做去尖峰滤波，方案待定）
2. 若需大规模犬类运动统计，可申请 DogMo（CC BY-SA，SMAL 网格域）作补充，需先解决 SMAL→骨架提取
3. manifest 的 split 字段当前统一 `unsplit`；进入 warm-start 微调池前再定切分协议

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-25 | W27 初版：五轴调研 + Top1 DogSet 获取转换 + 映射表 + CMU 负结果证伪 |
