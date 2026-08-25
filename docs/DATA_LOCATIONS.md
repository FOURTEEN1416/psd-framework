# 数据位置登记表（DATA_LOCATIONS）

> 盘点日期: 2026-08-23（歆歆，接管建仓会话）；数量复核: 2026-08-23（W2 窗口，证据见 `reports/data-inventory-2026-08-23.md`）
> 验证方式: PowerShell `Get-ChildItem` 目录存在性检查 ✓ → 数量级逐项复核 ✓（numpy/pandas 实测）
> 原则: 数据文件**不搬移不入库**（.gitignore 已排除 data/），本仓以绝对路径配置访问

## 1. 核心数据集

| 数据集 | 根路径 | 已验证结构 | 实测数量（2026-08-23 W2 复核） | 主用途 |
|--------|--------|-----------|---------|--------|
| **InterPet4D** | `D:\Desktop\k9-training-system\data\interpet4d` | smal_npy/ ✓ pet_npy/ smpl_npy/ mano_npy/ interpet_audio/ interpet_mert/ .cache/ | **226 个 .npz** ✓（与声明 226 吻合；骨架维度 pose_rotmat T×35×9 / kp_world T×24×3） | P0.1 AimCLR 预训练 / P0.2 SMQ 分割（smal_npy 3D 关键点） |
| **Animal Kingdom** | `D:\Desktop\k9-training-system\data\animal_kingdom` | action_recognition/ pose_estimation/ video_grounding/ README | **329 犬科视频**（train 231 / test 98），帧行 34,772（原声明 338/239 已溯源为规划期估计，K9 已修正） | P0.1 弱监督预训练 / P0.2 episode 分割信号 |
| **APTv2 全量** | `D:\Desktop\k9-training-system\data\APTv2\APTv2` | annotations/ + data/{easy,hard} | **83,304 文件**（原声明 242K 已溯源为规划期估计，K9 已修正） | 无标签池扩展 |
| **dog-pose（ultralytics 打包）** | `D:\Desktop\datasets\dog-pose` | images/{train,val} ✓ labels/{train,val} ✓ dog-pose.yaml（kpt_shape=[24,3] 单类 dog） | **8476 图 = 6773+1703** ✓（2026-08-25 W29 全量复核：配对/标注/包围盒零缺陷） | C5 静态池：预训练/增广（结论 b）；提点器微调源；规则引擎静态先验标定。⚠️ 无序列元数据、GT 有效关节 20/24（双眼/withers/throat 零标注）。详见 `reports/dogpose-report-2026-08-25.md` |

## 2. APTv2 伴生处理目录（K9 产品线产物，按需引用）

| 目录 | 说明 |
|------|------|
| `data\aptv2_annotations` | 标注处理产物（与主池 annotations/ 字节级同尺寸重复副本，truth 以主池为准，W26 核对） |
| `data\aptv2_canidae` | 犬科子集提取：24 mp4（43MB）+ MOT 框级 GT 852 行（class 单类）+ frames/hard 帧图 |
| `data\aptv2_yolo` / `data\aptv2_yolo_pose` | YOLO 检测/姿态训练格式（各 290 图；pose 版 kpt_shape=[17,3]，nc=3 dog/fox/wolf） |

### 2.1 C2/W26 挖掘登记（2026-08-25）

- **官方来源**: APTv2（ViTAE-Transformer/APTv2，v1.0，2023；据标注内嵌元数据）。许可：本地副本无独立 LICENSE 文件，学术引用待投稿前终审（R-C2-1）
- **总量对账**: 83,304 = data 池 83,245（jpg 41,569 + LabelMe 帧级 json 41,674 + 杂项 2）+ annotations 树 59 ✓
- **结构**: 41,179 标注图 ↔ 84,611 条 COCO 标注（train 58,029 / val 11,315 / test 15,267，标注级切分零交集），2,749 视频；30 物种共用唯一 17-kp 四足拓扑
- **判定**: **15 帧微序列池**（非静态池）：轨迹段内连续帧对 99.8%，但跨度被 ~15 帧窗口截断，canidae 无 ≥16 连续帧轨迹
- **canidae 规模**（公开真实层）: dog+fox+wolf = 11,410 标注 / 4,676 标注帧 / 326 序列组 / 646 轨迹
- **证据指针**: 报告 `reports/aptv2-report-2026-08-25.md`；可入库清单 `reports/aptv2-canidae-manifest-2026-08-25.json`；机读盘点 `reports/aptv2-inventory-summary-2026-08-25.json`；脚本 `scripts/mine_aptv2_inventory.py`（运行时落盘 `runs/data_campaign/aptv2/`，gitignore 可再生）
- **Format B 序列池（后续①已建）**: `runs/data_campaign/aptv2/sequences/canidae/` **503 条 T=15×V=17×C=3** .pkl（dog 203/fox 167/wolf 133；train 356/val 68/test 79），索引 `sequences/_manifest.json`；抽取器 `scripts/mine_aptv2_extract_sequences.py` 复跑可再生；拓扑 `aptv2_quadruped_17kp` 接入主管线前需 assets-map 映射

## 3. 待补充数据源

| 数据源 | 状态 | 用途 |
|--------|------|------|
| SyDog-Video（500 合成狗视频） | ⏳ 待评估下载 | 预训练增强 |
| samtwl 细粒度犬类行为 | ⏳ 待邮件请求 | 行为识别补充 |
| 真实 K9 训练视频 | ⚠️ 优先但非硬依赖 | 主动学习目标域 |

## 4. 合成层数据

| 数据集 | 根路径 | 实测数量 | 关键结构 | 主用途 |
|--------|--------|---------|---------|--------|
| **22 类合成骨架集（n=50/类）** | `data/synthetic/synthetic_22class_T30_n50.pkl` | **1100 样本**（22 类 × 50），9.8 MB | `.pkl`：`keypoints (T=30,24,3)` / `label (0-21)` / `boundary (T,)`；元数据 `_manifest.json` | P0.3 Phase B 类别映射输入 |
| **22 类合成骨架集（n=20/类）** | `data/synthetic/syn_22class_20per_class_seed42.pkl` | **440 样本**（22 类 × 20），3.8 MB | 同上；W11 冒烟基线（val_acc=18.2% @ epoch 5） | P0.5 W11 冒烟复现 |
| **22 类合成骨架集（n=100/类）** | `data/synthetic/syn_22class_100per_class_seed42.pkl` | **2200 样本**（22 类 × 100），19.0 MB | 同上；W12 主实验数据 | P0.5 完整训练 + E6 双贴合实验 |

> 权威 22 类清单：`docs/assets-map.md` §1（禁止另抄一份）
> 复现命令（n=100）：`.venv/Scripts/python.exe scripts/gen_synth_22class.py --samples-per-class 100 --output data/synthetic/syn_22class_100per_class_seed42.pkl`
> 元数据清单：`data/synthetic/_manifest.json`

## 5. 数据战役新增源（DATA-CAMPAIGN W25-W29）

| 数据集 | 根路径 | 实测数量 | 关键结构 | 主用途 | 登记窗口 |
|--------|--------|---------|---------|--------|---------|
| **MANN DogSet**（SIGGRAPH 2018 真实犬类动捕） | `external/dogset-mann-siggraph2018/raw`（原始 BVH，gitignore）+ `runs/data_campaign/mocap/sequences/*.pkl`（转换产物） | **51 BVH / 147,541 帧 / ≈41min @60fps** | BVH 21 关节厘米制；格式 B pkl `(T,V=21,3)` float32 + `manifest.jsonl`（qc_flag 打标） | 真实运动学先验（合成保真度拟合 C4 / 规则种子校准）；**非行为分类主粮**；许可=研究教育专用禁商用禁再分发（Edinburgh IP） | W27 (C3) |

> 调研 truth 与关节映射表：`dev-docs/research/MOCAP_DATASETS.md`；当次运行证据：`reports/c3-mocap-dogset-2026-08-25.md`
> 转换脚本：`scripts/bvh_dogset_to_sequence.py`（含 --self-test）

## 6. 回填记录

| 日期 | 数据集 | 复核项 | 结果 |
|------|--------|--------|------|
| 2026-08-23（W2） | InterPet4D smal_npy | 序列计数 + npz 抽查 + 骨架维度 | 226 ✓ 吻合；.npz 格式；pose_rotmat (T,35,3,3) / kp_world (T,24,3)，T=326/556/509 抽查 |
| 2026-08-23（W2） | Animal Kingdom 犬科 | 犬科视频 / 帧级标注计数 | 实测 329 视频（train 231/test 98）、帧行 34,772；≠ 声明 338/239 → 已溯源为规划期估计，K9 truth 已修正 |
| 2026-08-23（W2） | APTv2 全量 | 文件总数 + COCO 标注统计 | 实测 83,304 文件、annotations 84,611；≠ 声明 242K → 已溯源为规划期估计，K9 truth 已修正 |
<<<<<<< HEAD
| 2026-08-25（W26/C2） | APTv2 全量 + 伴生目录 | 83,304 精确分解 / 物种分布 / 序列连续性 / canidae 规模 / 拓扑与置信度结构 | 全部吻合无漂移：83,245+59=83,304 ✓；判定"15 帧微序列池"；canidae 11,410 标注；证据 `reports/aptv2-report-2026-08-25.md` |
| 2026-08-25（W27） | MANN DogSet | 51 BVH 拓扑一致性 + 帧数统计 + FK 自测 + pkl 抽验 | 51 条单拓扑 60fps ✓；147,541 帧 ✓；(T,21,3) 无 NaN ✓；45/51 带 suspect_glitch 标记（源数据毛刺，详见报告） |
=======
| 2026-08-25（W29） | dog-pose | 全量配对/标注解析/可见性/序列元数据探查 | 8476 图零缺陷；无序列分组元数据；GT 有效关节 20/24。证据 `runs/data_campaign/dogpose/inventory-evidence-2026-08-25.json`，结论 (b) 见 `reports/dogpose-report-2026-08-25.md` |

> 差异裁决与完整证据：`reports/data-inventory-2026-08-23.md`；K9 侧修正如上（commit `faaab28`，2026-08-23 双轨处置：回改 K9 truth + 本仓保留差异标注）

## 6. C1 公开视频主动抓取片段池（W25，2026-08-25 入库）

| 项 | 值 |
|----|----|
| 收敛契约 | dev-docs/handovers/DATA-CAMPAIGN-plan.md §0 格式 A |
| 片段目录 | runs/data_campaign/video/fragments/*.mp4（单段 ≤30s 实测 0 违规） |
| manifest | runs/data_campaign/video/manifest.jsonl，**642 行**（契约 7 字段逐行校验通过） |
| 来源平台 | Bilibili 299 / YouTube 343（yt-dlp 访问，B 站游客 cookie、YouTube 经系统代理 127.0.0.1:17890） |
| 关键词族 | 双语 12 族：IGP 训练 / Schutzhund / 马里努阿犬 训练 / 护卫犬 训练 / 警犬 训练 / 工作犬 训练 / 服从性 比赛 狗 / K9 police dog training / protection dog training / Belgian Malinois training / IGP dog training / dog obedience trial |
| 犬类粗筛 | ultralytics YOLO COCO 检测权重 yolo11n.pt（dog=class16），CPU 推理，均匀抽 4 帧，出现率 ≥0.5 过筛 |
| license_note（逐条） | research-use only; public platform content accessed via yt-dlp respecting ToS; no redistribution |
| 标签策略 | **W6 规则引擎自产种子草稿**（psd/data/rule_seeds.py 七类规则族机械复用）；外部标签零直用；当前 label_status=rule_seed_pending，待 Q3a/Q3b（GPU 接力队列）提点后由 scripts/harvest_rule_seeds.py 生成 |
| 管线代码 | scripts/harvest_video_pipeline.py + configs/harvest_video_w25.yaml |
| 报告 | reports/video-report-2026-08-25.md |
