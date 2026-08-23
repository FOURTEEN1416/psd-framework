# 数据位置登记表（DATA_LOCATIONS）

> 盘点日期: 2026-08-23（歆歆，接管建仓会话）
> 验证方式: PowerShell `Get-ChildItem` 目录存在性检查 ✓
> 原则: 数据文件**不搬移不入库**（.gitignore 已排除 data/），本仓以绝对路径配置访问
> ⚠️ 「数量」列来自 K9 仓 truth 文档声明，**本仓未逐一复核**（未验证）——P0 对应子阶段使用前复核并回填

## 1. 核心数据集

| 数据集 | 根路径 | 已验证结构 | 声明数量 | 主用途 |
|--------|--------|-----------|---------|--------|
| **InterPet4D** | `D:\Desktop\k9-training-system\data\interpet4d` | smal_npy/ ✓ pet_npy/ smpl_npy/ mano_npy/ interpet_audio/ interpet_mert/ .cache/ | 226 序列 | P0.1 AimCLR 预训练 / P0.2 SMQ 分割（smal_npy 3D 关键点） |
| **Animal Kingdom** | `D:\Desktop\k9-training-system\data\animal_kingdom` | action_recognition/ pose_estimation/ video_grounding/ README | 338 犬科视频 / 239 帧级标注 | P0.1 弱监督预训练 / P0.2 episode 分割信号 |
| **APTv2 全量** | `D:\Desktop\k9-training-system\data\APTv2\APTv2` | APTv2/ 子目录 | 242K 文件 | 无标签池扩展 |

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
| （待 P0.1 启动时回填） | | | |
