# 数据位置登记表（DATA_LOCATIONS）

> 盘点日期: 2026-08-23（歆歆，接管建仓会话）；数量复核: 2026-08-23（W2 窗口，证据见 `reports/data-inventory-2026-08-23.md`）
> 验证方式: PowerShell `Get-ChildItem` 目录存在性检查 ✓ → 数量级逐项复核 ✓（numpy/pandas 实测）
> 原则: 数据文件**不搬移不入库**（.gitignore 已排除 data/），本仓以绝对路径配置访问

## 1. 核心数据集

| 数据集 | 根路径 | 已验证结构 | 实测数量（2026-08-23 W2 复核） | 主用途 |
|--------|--------|-----------|---------|--------|
| **InterPet4D** | `D:\Desktop\k9-training-system\data\interpet4d` | smal_npy/ ✓ pet_npy/ smpl_npy/ mano_npy/ interpet_audio/ interpet_mert/ .cache/ | **226 个 .npz** ✓（与声明 226 吻合；骨架维度 pose_rotmat T×35×9 / kp_world T×24×3） | P0.1 AimCLR 预训练 / P0.2 SMQ 分割（smal_npy 3D 关键点） |
| **Animal Kingdom** | `D:\Desktop\k9-training-system\data\animal_kingdom` | action_recognition/ pose_estimation/ video_grounding/ README | **329 犬科视频**（train 231 / test 98），帧行 34,772；⚠️ 与声明 338 视频 / 239 帧级标注不符，需 K9 口径确认 | P0.1 弱监督预训练 / P0.2 episode 分割信号 |
| **APTv2 全量** | `D:\Desktop\k9-training-system\data\APTv2\APTv2` | annotations/ + data/{easy,hard} | **83,304 文件**；⚠️ 与声明 242K 不符，需 K9 口径确认 | 无标签池扩展 |

## 2. APTv2 伴生处理目录（K9 产品线产物，按需引用）

| 目录 | 说明 |
|------|------|
| `data\aptv2_annotations` | 标注处理产物 |
| `data\aptv2_canidae` | 犬科子集提取 |
| `data\aptv2_yolo` / `data\aptv2_yolo_pose` | YOLO 检测/姿态训练格式 |

## 3. 待补充数据源

| 数据源 | 状态 | 用途 |
|--------|------|------|
| SyDog-Video（500 合成狗视频） | ⏳ 待评估下载 | 预训练增强 |
| samtwl 细粒度犬类行为 | ⏳ 待邮件请求 | 行为识别补充 |
| 真实 K9 训练视频 | ⚠️ 优先但非硬依赖 | 主动学习目标域 |

## 4. 回填记录

| 日期 | 数据集 | 复核项 | 结果 |
|------|--------|--------|------|
| 2026-08-23（W2） | InterPet4D smal_npy | 序列计数 + npz 抽查 + 骨架维度 | 226 ✓ 吻合；.npz 格式；pose_rotmat (T,35,3,3) / kp_world (T,24,3)，T=326/556/509 抽查 |
| 2026-08-23（W2） | Animal Kingdom 犬科 | 犬科视频 / 帧级标注计数 | 实测 329 视频（train 231/test 98）、帧行 34,772；≠ 声明 338/239，需 K9 口径确认 |
| 2026-08-23（W2） | APTv2 全量 | 文件总数 + COCO 标注统计 | 实测 83,304 文件、annotations 84,611；≠ 声明 242K，需 K9 口径确认 |

> 差异裁决与完整证据：`reports/data-inventory-2026-08-23.md`
