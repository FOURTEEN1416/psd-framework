# W20 报告 — AK 公开真实层部分类协议：骨架可用性前置检查 + 映射重建 + 犬科样本量门禁

> 窗口: W20 | 日期: 2026-08-24 | 执行: 歆歆（sliver-vibe-coding，接管项目路线）
> 任务书: `dev-docs/handovers/W20-p05-public-real-partialclass.md` v1.0
> 前序事实: W16 序数映射已被证伪清除（commit `8fabca2`）；本窗口从零重建
> 数据: 全程只读 K9 仓，零修改 ✓

---

## 0. 一页结论

| # | 问题 | 结论 |
|---|------|------|
| 1 | **骨架数据可用性（第一天必答）** | **本地不可用**。AK action_recognition 注释无关键点；pose_estimation tar 经全量扫描为纯帧图（28,197 jpg / 34,358 条目，标注文件命中 0）；官方 MPII 格式关键点标注（23 点）随下载包分发、本地未持有。且 PE 子集 ≈4.6 帧/视频的静态稀疏标注本身构不成 ST-GCN 所需 T=30 时序序列 |
| 2 | **12±2 类映射协议是否成立** | 映射表成立（15 个 AK index → 12 PSD 类，TDD 31 绿），但发现并修正 rescue-plan §0 两处 index 笔误（73→74、15→67） |
| 3 | **犬科样本量门禁** | **远比预期严峻**：主标签规则下 12 类仅 **3 类可训练**（stay/track/watch）；down/stand/scale 在 AK 犬科集上零覆盖 |
| 4 | **训练段状态** | 冻结待裁决——骨架路线是总闸门（§4 三方案对比） |

---

## 1. 骨架数据可用性前置检查（任务书 Step 4 开工前置）

三源证据链（全部当次运行实测，2026-08-24）：

### 1.1 action_recognition 注释目录
```
annotation/  = df_action.xlsx (12KB) + train.csv (217MB) + val.csv (54MB) + AR_metadata.xlsx(2.2MB)
               → 仅帧级动作标签宽表与元数据，无任何关键点坐标文件
```

### 1.2 pose_estimation/dataset.tar.gz 全量流式扫描
```
总条目 34,358 = .jpg 28,197 + 目录条目 6,161；唯一视频目录 6,160
非图片关键词命中（annot/json/csv/pkl/npy/mat/xml/txt）= 0
→ 包内纯帧图，官方标注 JSON 不在其中
```

### 1.3 官方分发渠道核验（GitHub-First，sutdcv/Animal-Kingdom 官方仓 README）
- 标注为 **MPII 格式 JSON**：`joints`(23 关键点, 640×360) + `joints_vis` + `animal` 物种字段，随数据集下载包（表单申请）分发
- 本地 K9 仓未持有该 JSON → 需另行申请下载
- **结构性障碍**：PE 子集 28,197 帧 ÷ 6,160 视频 ≈ **4.6 帧/视频**，是图像级姿态数据集；即使拿到全部标注也无法构成时序骨架行为序列

### 1.4 可用素材盘点（影响 §4 方案对比）
| 素材 | 数量 | 说明 |
|------|------|------|
| PE 帧图 ∩ 犬科精确 329 集 | 93 视频 | tar 内有抽帧图，但稀疏且无标注 |
| 本地 mp4 ∩ 犬科 | **211 视频（100% 为犬科）** | action_recognition/dataset/video/ 下完整视频，自提取路线素材 |

---

## 2. 权威映射重建（Step 1+2 完成，TDD 31 绿）

### 2.1 对 rescue-plan §0 的两处修正（重要披露）

df_action.xlsx 有 `S/N` 与 `index` 双编号列。逐行 pandas 核验发现 §0 表两处括号数字取自 S/N 列：

| 动作 | §0 记载 | xlsx index 列真值 | 若照抄的后果 |
|------|---------|------------------|--------------|
| Lying on its side | (73) | **74**（S/N=73） | index 73 = **Licking 舔毛** → 舔毛污染 down 类 |
| Jumping | (15) | **67**（S/N=15） | index 15 = **Chirping 鸣叫** → 鸟鸣训练成 jump |

其余 13 项经核验与 index 列一致。任务书 Step 2 的 spot check「Jumping(15)」已按真值修正为 67。本差异属上游文档笔误，rescue-plan 为本窗口禁触文件，登记于此由协调者收编。

### 2.2 映射协议 v2（15 index → 12 类）

完整映射表见 `psd/data/ak_mapping.py`（每项附语义理由与 count）。要点：
- 强对应 6 类直译：sit(108)/down(70,74)/stand(116)/bark(3)/bite(8)/jump(67)
- 中对应 5 类语义近似：stay(68)/watch(2,102)/apprehend(1)/retrieve(13)/scale(16)
- 弱对应 1 类须显著披露：track(45 Exploring, 14 Chasing)
- 排除 10 类显式登记（K9 特有零覆盖 + obstacle 与 scale 同源合并）
- 反向证伪回归固化进测试：旧序数断言 1→sit / 2→down / 5→stand 及两个 S/N 陷阱（73、15）永久性断言禁止复活

---

## 3. 犬科样本量统计与 untrainable 门禁（Step 3 完成）

口径复现：W2 盘点口径A（`reports/data-inventory-2026-08-23.md` §2.2）——AR_metadata.xlsx `list_animal` ∩ 犬科精确 10 物种 = **329 视频（train 231 / val 98）**，CSV 犬科帧行 24,865+9,907=34,772 ✓（与 W2 实测逐位吻合）。

主标签规则（labels 第一项）下各 PSD 类视频数：

| PSD 类 | 强度 | train | val | 门禁(<10) | 宽松对照(train/val)* |
|--------|------|-------|-----|-----------|---------------------|
| stay | 中 | 18 | 8 | ✅ | 34/8 |
| track | 弱 | 25 | 12 | ✅ | 29/17 |
| watch | 中 | 40 | 9 | ✅ | 63/16 |
| jump | 强 | 8 | 5 | ❌ | 23/10 |
| sit | 强 | 6 | 1 | ❌ | 6/1 |
| bark | 强 | 7 | 0 | ❌ | 9/0 |
| bite | 强 | 2 | 1 | ❌ | 6/4 |
| apprehend | 中 | 0 | 2 | ❌ | 6/5 |
| retrieve | 中 | 2 | 1 | ❌ | 6/2 |
| down / stand / scale | — | **0** | **0** | ❌ 真零覆盖 | 0/0 |

\* 宽松对照 = 视频任一标签命中映射表（上限估计）。即使采用该口径也仅 4 类过线。

**门禁结论**：主标签口径可训练类数 = **3**（远低于任务书预期的 12±2 下限）。强对应类在犬科子集上反而几乎全军覆没——df_action 全表中 Sitting(count=69)/Standing(126) 等标注段几乎不落在犬科视频上。

### 3.1 门禁规则解释分歧披露（复核会话补充，2026-08-24 晚）

任务书「任一类 <10 视频」存在两种合理解释，导致可训练类数 3 vs 4 的分歧：

| 解释 | 规则 | 可训练类 |
|------|------|---------|
| 严格（本报告采用） | train ≥10 **且** val ≥1 | stay / track / watch（3 类） |
| 宽松 | train + val 合计 ≥10 | stay / track / watch / **jump**(8+5=13)（4 类） |

本报告采用严格口径的理由：train<10 难支撑微调学习；val=0 则无法评估。若裁决采用宽松口径，jump 入列、可训练类数改为 4——该分歧交用户/协调者裁定，两种口径的完整分布均已在上表与 JSON 中。

证据 JSON：`p05-public-real-partialclass-stats-2026-08-24.json`（同目录）。
复现方式：统计脚本按领地纪律未入本窗口白名单，逻辑与常量以本报告 + JSON 字段完整记录（canine_reproduction.method / distribution / gate）。

---

## 4. 公开真实层微调评估路线对比（~~冻结~~ → 用户已裁决 C 路线）

> **裁决记录（2026-08-24 晚, 用户拍板）**：
> ① 骨架路线 = **C 自提取**；② 门禁口径 = **宽松 4 类**（jump 入列）。
> 样本判定随之统一为 R2(first-mapped-hit)——若维持绝对主标签规则, jump train 将为 0 个可用视频, 违背裁决意图。

| | 方案 B：登记结构性不可行 | 方案 C：mp4 自提取骨架管线（✅ 已裁决采纳） | 方案 A：申请官方 PE 标注 |
|---|---|---|---|
| 做法 | tab2 中间列改报 P0.1 kNN 探针数字（20.89%，dog-ID 代理先例）+ 本报告样本量/零覆盖披露 | 211 个犬科 mp4 → 抽帧 → 动物姿态模型（如 ViTPose/RTMPose-AK 版）提 23 点 → 时序序列喂 ST-GCN 微调（3 类子集） | 表单申请官方 MPII JSON 后评估 |
| 得到什么 | 诚实的负结果+披露，tab2 完整闭环，今天收尾 | 真实微调数字（3±1 类子集），新增可复用提点管线资产 | 合成层拓扑可比的 23 点静态标注 |
| 代价/风险 | 公开真实层无微调数字，论文该列弱化 | 新增重依赖（mmpose 等）、跨窗口工作量、23 点 vs 合成层 24 点拓扑需重设计 graph、3 类子集数字单薄易被审稿人质疑 cherry-pick | 标注仍是 ~4.6 帧/视频静态帧，构不成时序序列，大概率同样走不通 |
| 时间 | 0（已完成 95%） | 约 2-3 个窗口（环境+提点+对齐+训练） | 申请周期不可控 + 同样撞时序障碍 |

### 4.1 C 路线选型定案（GitHub-First 调研后）

| 维度 | ✅ YOLO11-pose + dog-pose 微调 | DeepLabCut SuperAnimal | AP-10K HRNet |
|------|------------------------------|------------------------|--------------|
| 拓扑 | **24 点原生=K9Graph 零投影**（dog-pose.yaml 与 assets-map §2 逐名逐序一致, 含 withers/throat） | 泛四足拓扑需投影 | 23 点通用动物需投影 |
| 依赖 | pip ultralytics 纯 torch ✓ | DLC 重依赖(GUI/wx) | mmcv Windows 编译地狱 |
| 域匹配 | 犬类专用(StanfordExtra) | 泛四足 | 泛动物 |
| 权重 | COCO 迁移自训(~1h GPU)+数据已预下载(8476 图) | DLC 服务器下载 | Google Drive 不稳定 |

反方质疑与缓解：COCO→dog 迁移效果存疑 → 训练后目检+置信度过滤兜底；AK 野生视角域差 → 失败帧率如实披露；AGPL-3 学术使用合规。

### 4.2 C 路线第一阶段完成证据（2026-08-24 晚, commit `76a29b2`）

| 项 | 结果 |
|----|------|
| 选型调研 | GitHub 工具链三路对比定案 ultralytics（社区无现成狗姿态权重, 0 命中） |
| 管线 TDD | `psd/data/ak_pose_extract.py` + 测试 **19 绿**（R2 规则/抽帧/多实例/插值组装/tar 补抽; TDD 红灯当场抓获测试自身误用 index 15=Chirping 当 Jumping 的笔误） |
| 样本清单(R2/4类) | **172 视频 = train 123 / val 49**（stay 27 / track 46 / watch 72 / jump 27） |
| video.tar.gz 补抽 | 缺失 77 个视频流式补抽 **77/77 零失败**, 缓存 runs/public_real_video_cache/ |
| 端到端冒烟 | COCO 权重 CPU 冒烟通过; 并暴露 17 点人体拓扑污染风险 → 已加 **24 点防呆 fail-fast** |
| dog-pose 数据 | 已预下载解压 D:\Desktop\datasets\dog-pose（8476 图=6773+1703 ✓） |
| 全仓回归 | pytest **288 passed** |

### 4.2.1 复核会话补强（v2.1, 2026-08-24 深夜）

独立复验全部 PASS，两项缺陷当场修复：

| 复核项 | 结果 |
|--------|------|
| 拓扑零投影程序化验证 | ✅ GitHub 官方 yaml 24 点 vs assets-map §2：逐名 diff=[]、索引映射恒等（C0-C3）——声明从目测升级为铁证 |
| 清单数字独立重算 | ✅ 换实现路径重算 train=123(watch57/track29/stay19/jump18)、val=49 逐格一致 |
| R2 口径差异定量 | R2a(首中12类池,预估113)=R2b(首中4类池,实际123) train 差 10 个——apprehend 先行视频在 R2b 下顺延命中 4 类；与宽松门禁裁决精神一致，正式口径为 R2b |
| tar 缓存完整性 | ✅ 77 文件全在、无零字节、大小两极+中位抽样均可读; 最短视频仅 12 帧(<T=30)将启用循环补齐策略(设计内) |
| 🔧 缺陷① manifest 归档 | stage_extract 曾覆盖 manifest JSON 为 quality 版(3行), 首次归档内容与 commit 描述不符 → 已改分文件写入(extract_quality.json), 重跑 manifest 重建完整版并重新归档(172 样本) |
| 🔧 缺陷② 死参数 | select_samples 的 split_of 参数从未使用 → 已删除(签名简化, split 归属由 CSV 键决定), 测试同步更新 19 绿 |

### 4.3 待 GPU 接力队列（排队纪律: NTU PID 35208 结束且显存 <2GB 后串行执行）

```
① python scripts/train_yolo_dogpose.py --epochs 50 --batch 16     # ~1h, 产出 24 点犬类权重
② python scripts/run_p05_public_real_pipeline.py --stage extract \
     --weights runs/public_real_yolo_dogpose/train/weights/best.pt # ~10min, 全量提点
③ ST-GCN+BC 微调(backbone 冻结+4类新 head, init runs/p05_stgcn_bc_full/best.pt)
   → reports/p05-public-real-partialclass-result-<日期>.json
```
看护归属待用户指定（本窗继续等待 / 交 W18 GPU 看护窗口扩队列 / 手动触发）。

---

## 5. 三层口径声明

本报告所有数字属**公开真实层**（Animal Kingdom 犬科子集），禁止与合成层（97.3% ST-GCN+BC best_val_acc 口径）混排对比。真实 K9 层无新数字。

## 6. 领地合规自检

- 写入文件均在白名单内：`psd/data/ak_mapping.py`、`psd/data/tests/test_ak_mapping.py`、`configs/p05_public_real_partialclass.yaml`、`reports/p05-public-real-*`
- K9 仓全程只读 ✓；禁触清单（docs/paper、decisions、rescue-plan、HANDOVER、*ntu*）未触碰 ✓
- 统计脚本因白名单限制于临时目录执行，产物已归档 ✓

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 前置检查结论 + 映射重建 + 门禁统计 + 路线裁决请求 |
| v1.1 | 2026-08-24 | 复核会话补强：门禁解释分歧披露（§3.1）+ PE 犬科帧数分布实证（93 视频 max=13 帧、0 个达 T=30，强化 §1.4 结论）+ 双路实现交叉验证记录 |
| v2.0 | 2026-08-24 | 用户裁决 C 路线+宽松 4 类：选型定案（§4.1）+ 管线第一阶段完成证据（§4.2）+ GPU 接力队列（§4.3）；样本判定统一 R2 |
| v2.1 | 2026-08-24 | 复核会话补强（§4.2.1）：拓扑程序化验证/清单独立重算/R2 差异定量/tar 缓存抽验全 PASS；修复 manifest 覆盖缺陷与死参数 |
