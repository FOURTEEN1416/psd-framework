# DATA CAMPAIGN — 真实域数据五路攻坚总规划（W25-W29）

> Owner: dev-docs/handovers/DATA-CAMPAIGN-plan.md（战役唯一 truth）
> 立项: 用户裁决 2026-08-25 "降级不是我们的风格"，多方式同步解决真实域数据
> 目标: 为真实 K9 域 ≥85% 终验备足弹药；所有产物最终汇入既有管线（提点→规则种子→伪标签→warm-start 微调）

## 0. 收敛契约（五路必须遵守，防产物碎片化）

### 格式 A：视频片段（走 YOLO-pose 提点路径）
- 目录: `runs/data_campaign/<channel>/fragments/*.mp4` 单段 ≤30s
- manifest 每行: `{fragment_id, source_channel, origin_url_or_path, capture_context, species_note, license_note, collected_at}`

### 格式 B：关键点序列（直接进 ST-GCN 路径）
- 目录: `runs/data_campaign/<channel>/sequences/*.pkl`
- 每条: `{keypoints: (T,V,C), topology_name, V, fps_or_sampling, source, split}`

### 通用纪律
1. 每路开工先在 `docs/DATA_LOCATIONS.md` 登记来源与许可注记（df_action.xlsx 的语义教训已固化：外来标签必须过映射审查，优先用本仓规则引擎自产标签）
2. 全部 CPU/网络任务，**禁触 GPU 队列**（relay_executor Q1-Q3c 不可被挤占）
3. 标签策略默认 = 规则引擎物理先验自产（psd/data/rule_seeds.py 机械复用）；外部标签需映射审查后人工抽验
4. 每路完成出一份 `<channel>-report-<日期>.md`：样本量、质量分布、与 AK 域的分布对比、可训性门禁结论

## 1. 五路定义

| 路 | 窗口 | 产出 | 优先级 |
|----|------|------|--------|
| C1 公开视频主动抓取 | W25 | 格式 A 片段池 + 规则种子伪标签草稿 | 高 |
| C2 APTv2 83K 本地挖取 | W26 | canidae 关键点资产盘点+可用序列 | 最高（本地零协调） |
| C3 动捕犬类数据集 GitHub-First | W27 | 数据集目录 + Top1 转 (T,·,3) | 高 |
| C4 合成管线保真度提升 | W28 | syn_v2 生成器 + 分布距离报告 | 中 |
| C5 dog-pose 静态池评估 | W29 | 诚实可行性结论 + 可用部分入库 | 中 |

## 2. 各路任务书要点

### C1/W25 公开视频抓取
- 工具: yt-dlp；关键词族双语（IGP 训练/Schutzhund/马里努阿/K9 training/police dog/protection dog/obedience）
- 平台: Bilibili + YouTube；尊重 ToS，research-use 注记逐条登记
- 流水线: 搜索→下载→镜头切分(≤30s)→犬类出现率粗筛(抽帧+YOLO COCO 权重)→manifest 入库
- 目标量: ≥500 片段起步；标签=规则引擎种子草稿（W6 七类规则族直接复用）

### C2/W26 APTv2 挖取
- 盘点 83K 文件: 物种分布/图像vs序列/canidae 子集规模（HANDOVER §5 已知 aptv2_annotations/canidae 存在）
- 可用性判定: 静态姿态图为主则转"静态池"评估口径，与 C5 合流
- 产出: 挖掘报告 + 可入库清单 + DATA_LOCATIONS 更新

### C3/W27 动捕调研获取
- GitHub-First 检索: CMU mocap animal / DogMo / canine motion capture / quadruped mocap BVH
- 评估轴: 许可/拓扑/V 数/时长; Top1 做 BVH→(T,V,3) 转换 + 关节映射表文档
- 注意: 动捕犬非工作犬训练场景，作"真实运动学先验"用途而非行为分类主粮

### C4/W28 合成保真度
- 用 Q3b 真实提点产物（partialclass4_T30.pkl）统计真实分布: 关节角边缘/帧间速度谱
- 改造 make_synthetic_dataset → syn_v2: 参数向实测分布拟合; 保真度指标 = 逐关节 KS 距离 + 速度直方图差
- 产出: syn_v2 + 保真度报告; 三层口径不变（合成层自证）

### C5/W29 dog-pose 静态池
- 8476 图盘点: 是否有序列分组元数据（大概率无）
- 诚实结论三选一: a) 时序升采样构造过渡序列（标注"合成动态"风险） b) 仅作预训练/增广池 c) 不可用
- 与 C2 结论合流出"静态姿态资产统一处置方案"

## 3. 会师路径（各路成熟后）

```
C1 片段池 ──Q3a 权重提点──► 骨架序列 ─┐
C2/C3/C5 序列/静态池 ────────────────┼─► unified real-expansion pool
C4 syn_v2 ───────────────────────────┘        │
                                              ▼
                            规则种子 → 伪标签 → warm-start 微调
                                              ▼
                          真实 K9 域 ≥85% 终验（原始录像到位即插即跑）
```

## 4. 领地矩阵

| 路 | 可写 | 禁触 |
|----|------|------|
| C1 | scripts/harvest_*, runs/data_campaign/video/, configs/harvest_* | docs/paper, decisions, *ntu*, relay 队列文件 |
| C2 | scripts/mine_aptv2_*, runs/data_campaign/aptv2/ | 同上 + K9 仓只读 |
| C3 | dev-docs/research/MOCAP_DATASETS.md, scripts/bvh_*, external/(新克隆 gitignore) | 同上 |
| C4 | psd/data/synth_stgcn_v2.py(新增不改旧), tests, configs/syn_v2_* | 旧 synth_stgcn.py 行为 |
| C5 | scripts/assess_dogpose_*, runs/data_campaign/dogpose/ | 同上 |

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-25 | 用户裁决立项: 五路攻坚总规划+收敛契约+会师路径 |
