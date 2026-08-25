# C2/W26 — APTv2 本地 83K 池挖掘报告

> 窗口: W26（worktree `wt/W26`，B-full 协议）
> 任务书: `dev-docs/handovers/DATA-CAMPAIGN-plan.md` §2-C2（战役 truth，本报告为其 C2 路执行证据）
> 执行日期: 2026-08-25
> 挖掘脚本: `scripts/mine_aptv2_inventory.py`（CPU-only；K9 仓零写入）
> 机读产物: `runs/data_campaign/aptv2/aptv2_inventory_summary.json`、`runs/data_campaign/aptv2/canidae_sequence_manifest.json`
> 复跑命令: `& "D:\Desktop\psd-framework\.venv\Scripts\python.exe" scripts/mine_aptv2_inventory.py`（cwd=本仓根）

---

## 0. 执行摘要（三行版）

1. **判定修正**：APTv2 **不是静态姿态池**——标注帧在轨迹段内近乎全连续（连续帧对占比 99.8%）；但轨迹被 ~15 帧固定窗口截断（canidae 646 条轨迹无一 ≥16 连续帧），正确定性是 **"15 帧微序列池"**。
2. **canidae 规模**：主池 dog+fox+wolf = **11,410 条标注 / 4,676 个标注帧 / 326 个视频序列组**（646 轨迹）；另有 K9 产品线伴生子集 24 视频 MOT 框级 GT + 290 图 YOLO/YOLO-pose 训练格式。
3. **入库建议**：以 T=15 微序列直接进 Format B（关键点序列）通道；与 C5 合流面收窄为"T>15 长时序缺口"共享插值风险评估，不整体并入静态池处置。

---

## 1. 数据源登记

| 项 | 值 |
|----|----|
| 数据集 | APTv2（Animal Pose Track v2） |
| 官方出处 | 标注内嵌元数据：`url: https://github.com/ViTAE-Transformer/APTv2`，version 1.0，year 2023（GitHub-First 口径下以此内嵌元数据为准，未做外部检索） |
| 本地主池 | `D:\Desktop\k9-training-system\data\APTv2\APTv2\`（annotations/ + data/{easy,hard}），**只读** |
| 伴生目录 | `aptv2_annotations` / `aptv2_canidae` / `aptv2_yolo` / `aptv2_yolo_pose`（K9 产品线处理产物，只读引用） |
| 许可注记 | 本地副本未附独立 LICENSE 文件；学术研究用途引用官方仓库许可条款，投稿前需终审（登记为风险 R-C2-1） |

## 2. 总量对账（83,304 的精确分解）

| 组成 | 实测 | 说明 |
|------|------|------|
| data/{easy,hard} 文件池 | **83,245** | .jpg 41,569 + LabelMe 帧级 .json 41,674 + 杂项 2（见 §5 注） |
| annotations/ 目录树 | **59** | 主标注 7 + fewshot 10 + leaveoneout 两套 24 + tracking 18 |
| **合计** | **83,304** | 与 W2 复核值（DATA_LOCATIONS §5）逐位吻合，Δ=0 |

- 图像 = **41,179 张唯一标注图**（全部在盘、无缺漏：`annotated_missing_on_disk=0`）；另 390 张盘上 jpg 未入任何 split。
- 标注 = **84,611 条 COCO 关键点标注**，train 58,029 / val 11,315 / test 15,267；三个 split 共享同一 images 清单（41,179 图 / 2,749 视频），**按标注切分且身份交集为 0**（train∩val=test∩…=0，union=84,611 ✓）。
- easy/hard 变体是难度视图（val_easy 3,133 + val_hard 8,182 = 11,315 ✓ 同理 test 5,047+10,220=15,267 ✓），非新增标注。

## 3. 物种分布（30 类，联合 84,611 条）

| 科 | 物种(标注数) |
|----|----|
| **Canidae（犬科）** | **dog 4,770 / fox 3,667 / wolf 2,973 —— 合计 11,410（13.5%）** |
| Felidae | lion 5,169 / cheetah 3,725 / tiger 3,161 |
| Hominidae | gorilla 3,537 / chimpanzee 3,197 / orangutan 3,743 |
| Cercopithecidae | monkey 3,669 / spider monkey 3,171 / noisy night monkey 2,895 |
| Ursidae | black bear 3,081 / polar bear 3,370 / panda 2,963 |
| 有蹄类等 | deer 4,485 / horse 4,472 / zebra 4,151 / elephant 4,067 / giraffe 4,067 / buffalo 4,065 / antelope 4,057 / cow 3,946 / sheep 3,547 / hippo 2,277 / rhino 2,622 |
| 其他 | rabbit 4,813 / cat 4,320 / raccoon 3,462 / pig 3,751 |

分布相当均衡（2.3K–5.2K/种），无明显长尾塌缩。dog 在单物种榜排第 3。

## 4. 图像 vs 序列构成 —— "15 帧微序列"铁证

文件名模式 `<split>/<物种>/<clip目录>/NNNN.jpg`（如 `easy/10gorilla/v12C1/0000.jpg`），标注含 `video_id` + `track_id` 字段（覆盖率 100%）。

| 指标 | 全池（30 种） | canidae 三种 |
|------|--------------|--------------|
| 轨迹数 (video_id, track_id) | 5,974 | 646 |
| 轨迹跨度中位 / P90 | 15 / 15 帧 | 15 / 15 帧 |
| 轨迹最大连续段中位 | 15 帧 | 15 帧 |
| **连续帧对占比** | **99.88%** | **99.76%** |
| 连续段 ≥8 帧的轨迹 | 5,612 (93.9%) | 589 (91.2%) |
| 连续段 ≥16 帧的轨迹 | 47 (0.8%) | **0** |
| 连续段 ≥30 帧的轨迹 | 1 | **0** |

**解读**：标注以约 15 帧的固定窗口从源视频截取（span 中位=P90=max_run 中位=15，293/326 个 canidae 序列组恰为完整 15 帧段）。段内运动真实连续（99.8% 帧对相邻），因此：

- ❌ 不是"静态姿态为主"→ 任务书预设的"转静态池口径"条件**不成立**
- ✅ 是"短序列"→ 但 T 上限 ~15，对本管线常用 T=30 不够长

## 5. 关键点拓扑与置信度结构

**拓扑**：30 个 category 共用**唯一一套 17 关键点四足拓扑**（`unique_topology_count=1`）：
`left_eye, right_eye, nose, neck, root_of_tail, left_shoulder, left_elbow, left_front_paw, right_shoulder, right_elbow, right_front_paw, left_hip, left_knee, left_back_paw, right_hip, right_knee, right_back_paw`
skeleton 17 边（1-based，见 summary JSON `topology.skeleton_edges_1based`）。kp 数组长度恒 51（17×3，train 全量 58,029 条无一例外）。

**置信度结构**：
- GT 无 score/confidence 字段（`has_score_field=0`，预期内——这是人工 GT 不是模型输出）；唯一置信通道 = 可见性 v-flag
- v-flag 只取 {0, 2} 二值（train：0→344,667 占 34.9%；2→641,826 占 65.1%），**不存在 v=1"标注但遮挡"中间态**
- ⚠️ 数据质量坑：`num_keypoints` 字段有 ~150 条异常值（18–40 >17 上限，如 40 出现 1 次），而 kp 数组恒 51——该字段不可信，过滤时应按 v>0 现算计数
- bbox 覆盖率 100%

**杂项**：文件池含 41,674 个 LabelMe 格式帧级 JSON（COCO 标注的原生源，含 group_id=track 线索）；`easy/5rabbit/v25c1/frame8.txt(.bak)` 为扩展名笔误的 LabelMe JSON 及其备份（83,245 中的 +2 杂项）。

## 6. canidae 子集规模汇总（三层口径之公开真实层）

| 资产 | 规模 | 形态 |
|------|------|------|
| 主池 canidae（dog/fox/wolf） | 11,410 标注 / 4,676 唯一标注帧 / 326 视频序列组 / 646 轨迹 | COCO 17-kp 关键点 |
| └ dog 单种 | 4,770 标注 | 同上 |
| `aptv2_canidae` | 24 mp4（43 MB）/ MOT 框级 GT 852 行 / frames/hard 含 9 物种帧+JSON | 视频 + bbox 轨迹（class 列全为 '1' 单类） |
| `aptv2_yolo(_pose)` | 各 290 图/标签；pose 版 kpt_shape=[17,3]，nc=3（dog/fox/wolf） | YOLO 训练格式（K9 产品线产物） |
| `aptv2_annotations` | 与主池 annotations/ 字节级同尺寸（三主文件逐一核对 same_size=True） | **重复副本，truth 以主池为准** |

## 7. 可用性判定与 Format B 入库路径

任务书问句"静态姿态图为主？"→ **否**（§4 铁证）。定性改为 **"15 帧微序列池"**。入库选项对比：

| 选项 | 做法 | 得到 | 代价/风险 | 结论 |
|------|------|------|-----------|------|
| a) T=15 直接抽取 | 按 video_id+track_id 从 COCO JSON 抽 (T≤15,V=17,C=2/3)，滑窗出样本 | 326 组 canidae 微序列，零失真 | 下游 ST-GCN 管线当前按 T=30 设计，需 T 自适应或补齐策略 | ✅ **首选：登记即入库，改动最小** |
| b) 插值升采样 15→30 | 时序线性/样条插值翻倍 | 凑满 T=30 | "合成动态"标注义务（与 C5 方案 a 同款风险，需显式声明） | ⚠️ 仅当下游硬性要求 T=30 时启用 |
| c) 跨段拼接伪长序列 | 两段拼 30 帧 | 名义 T=30 | 段间跳变污染速度谱，污染 C4 保真度对照基线 | ❌ 否决 |

**推荐**：a 为主线（manifest 已就绪），b 作为后备并强制"合成动态"标记。拓扑映射前置风险：APTv2 17-kp 四足拓扑 ≠ InterPet4D smal 24-kp ≠ AK YOLO 提点拓扑，消费前需经 `docs/assets-map.md` 显式映射移植（本窗不做实现，超领地）。

## 8. 与 AK 域分布对比（任务书纪律 #4 要求）

| 维度 | APTv2 canidae | Animal Kingdom 犬科 |
|------|---------------|---------------------|
| 体量 | 4,676 标注帧 / 326 序列组 | 329 视频 / 34,772 帧行 |
| 监督形态 | 人工 17-kp GT（密集、干净） | 行为类别标签 + YOLO 自产提点 |
| 时序粒度 | ≤15 帧微序列 | 完整视频片段 |
| 场景 | 网络采集野生动物/宠物，多为单主体居中 | 野外/家庭监控风格 |
| 对管线价值 | 高质量**关键点监督**（预训练/warm-start 池） | 行为**语义标签**（分类监督） |

互补关系明确：AK 给语义、APTv2 给几何。两者均属公开真实层，禁止与合成层混报（三层口径纪律）。

## 9. 双向论证

**正方（值得入库）**：13.5% 犬科占比全池第 2 大科；GT 密集干净带 track 结构；序列连续性 99.8% 远超预期；本地已有、零下载协调成本；与规则种子→伪标签→warm-start 管线天然衔接（提点路径可跳过，直接真值骨架）。

**反方（冷水面）**：① 15 帧上限使其无法独立支撑长时序行为识别，T=30 主管线必须改造或插值，插值即引入"半合成"口径争议；② 场景偏动物园/网络图片，与真实 K9 工作犬训练场景域距大，直接迁移收益未知；③ num_keypoints 字段脏数据 + 无 LICENSE 文件两处合规卫生债；④ 390 张未标注图与 24 视频 MOT 子集价值密度低，不值得单独投入。

## 10. 与 C5 合流建议

APTv2 不并入 dog-pose 静态池处置方案（它有真序列）。合流点收窄为一项：**若最终需要 T>15 长序列，APTv2 微序列与 dog-pose 静态图面临同一道"时序构造 vs 半合成声明"的取舍**——建议 W29 出"静态姿态资产统一处置方案"时把 APTv2 作为"微序列参照系"纳入对比（其 15 帧真值动态可作为静态图插值方案的合理性校准基准），其余各自独立入库。

## 11. 局限与后续

- 未测主池源视频 fps（COCO JSON 无此元数据；如需可后续对 24 个伴生 mp4 用 cv2 探测，本窗未做以保纯 stdlib 依赖）
- 未做跨物种拓扑到本仓管线的映射实现（归 assets-map 移植流程）
- 许可条款待投稿前终审（R-C2-1）
- 后续动作建议：W29 合流时引用本报告 §10；若采纳选项 a，抽取器建议命名 `scripts/mine_aptv2_extract_sequences.py`（领地内顺延）

## 12. 产物清单（白名单提交物）

| 文件 | 性质 |
|------|------|
| `scripts/mine_aptv2_inventory.py` | 挖掘脚本（复跑即证） |
| `reports/aptv2-report-2026-08-25.md` | 本报告 |
| `reports/aptv2-inventory-summary-2026-08-25.json` | 全量机读盘点（提交快照） |
| `reports/aptv2-canidae-manifest-2026-08-25.json` | 可入库清单 326 序列组（提交快照，含 T 窗口统计与 format_b_ready 位） |
| `docs/DATA_LOCATIONS.md` | 登记更新 |
| `runs/data_campaign/aptv2/*.json` | 同上两 JSON 的运行时落点（`/runs/*` 已 gitignore，可再生，不入库） |

> 注：`runs/data_campaign/aptv2/` 为任务书领地内的运行时落盘位；因 `.gitignore #19 /runs/*` 不随 git 走，提交以 `reports/` 快照为准，truth = 脚本重跑输出。
