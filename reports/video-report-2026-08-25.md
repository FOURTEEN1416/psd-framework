# C1/W25 报告 — 公开视频主动抓取片段池（格式 A 入库）

> 窗口: W25（worktree `D:\Desktop\psd-framework-W25`，分支 `wt/W25`）
> 日期: 2026-08-25
> 收敛契约: `dev-docs/handovers/DATA-CAMPAIGN-plan.md` §0 格式 A
> 领地遵守: 仅写 `scripts/harvest_*`、`configs/harvest_*`、`runs/data_campaign/video/`、`docs/DATA_LOCATIONS.md`(§6 登记，通用纪律#1)、本报告；全程 CPU/网络，**未触 GPU 队列与 relay 文件**

---

## 1. 结论速览

| 项 | 值 |
|----|----|
| **入库片段** | **642**（任务书目标 ≥500 ✅） |
| 契约合规 | manifest 642 行，契约 7 字段逐行校验通过；fragment_id 全局唯一；单段 ≤30s **实测 0 违规**（max=30.0s） |
| 覆盖 | 双平台（B站 299 / YouTube 343）× 双语（zh 332 / en 310）× 12 关键词族全命中 |
| 标签 | 规则引擎自产策略已接线；当前 `label_status=rule_seed_pending`（Q3a/Q3b 未到位，fail-fast 有证） |

## 2. 样本量与采集漏斗

```
搜索候选 373 → 实下载成功 105/115 → 场景切分 1221 段 → YOLO 犬类粗筛过筛 642 (53%)
```

- 搜索：双语关键词族 × Bilibili(bilisearch) / YouTube(ytsearch)，flat 提取去重后 373 条候选
- 下载失败分类（10 例）：付费课/登录墙 4、格式不可用 3、时长越界(>15min) 2、其他 1 —— 均如实登记于 `_runtime/downloaded.jsonl`
- 切分：ffmpeg scene≥0.30 检测切点 → ≤30s 预算收段（纯函数 `budget_segments` 自检 6 用例通过）；重编码统一 360p/无音轨
- 粗筛：yolo11n.pt(COCO) dog=class16，CPU 推理，每段均匀抽 4 帧，出现率 ≥0.5 过筛；未过筛片段不留盘只留统计

## 3. 质量分布

| 维度 | 分布 |
|------|------|
| 时长(s) | min 2.4 / median 28.9 / mean 25.6 / max 30.0；分桶：≤10s 32、10–20s 78、20–30s 532 |
| dog_rate | mean 0.74；1.0×199、0.75×207、0.5×236（0.5 层含牵犬师为主镜头，下游可用该字段分层） |
| 分辨率 | 640x360 主体 558；竖屏 202x360 58、其余零星；最小 144x256 共 4（预探高度门 240 放行之竖屏） |
| 关键词(片段级) | Belgian Malinois training 81 / 马里努阿犬 训练 80 / protection dog training 76 / IGP 训练 犬 67 / 护卫犬 训练 67 / IGP dog training 56 / K9 police dog training 55 / 工作犬 训练 47 / 服从性 比赛 狗 40 / Schutzhund training 40 / 警犬 训练 31 / dog obedience trial 2 |
| 体量 | 88 个唯一源视频 URL，772 MB |

## 4. 与 AK 域的分布对比

| 轴 | Animal Kingdom 犬科（既有公开真实层） | 本池（C1 公开视频主动抓取） |
|----|----|----|
| 视角/场景 | 野生/纪录片视角为主，犬体常远距小目标 | 训练场手持/跟拍，人犬互动密集、犬体占幅大 |
| 行为覆盖 | W20 实测：12 类映射下仅 stay/track/watch 3 类可训，down/stand/scale **零覆盖** | 护卫/服从训练内容天然富含 sit/down/stay/heel/护卫科目动作——正是 AK 缺口的行为域 |
| 时序粒度 | 帧级行标注 34,772 行 / 329 视频 ≈106 行每视频；PE 子集稀疏（≈4.6 帧/视频）构不成 T=30 序列 | 中位 28.9s 长段；提点后按规则种子二次切段即可对齐秒级行为段粒度 |
| 结论 | 域差显著但**互补而非重叠**：本池补的是 AK 缺失的训练行为域，不是 AK 的重复采样 | |

> 口径声明：本池属**公开真实层**，与合成层、真实 K9 层三层分报，禁止混排。

## 5. 标签策略与状态（两段式）

任务书铁律：W6 规则引擎七类规则族自产种子草稿，禁止外部标签直用（df_action.xlsx 教训）。
本池 manifest 每行 `label_status=rule_seed_pending`，未携带任何行为标签。

依赖链（会师路径）：片段池 ──Q3a 犬类 pose 权重提点──► 24 点骨架 ──`scripts/harvest_rule_seeds.py` 机械复用 `psd/data/rule_seeds.py`──► 七类(lying/sitting/standing/walking/running/rise_transition/jump)种子草稿。

当次运行证据（2026-08-25）：

```
[rule-seeds] FAIL-FAST: 提点产物 0/642 到位于 runs\data_campaign\video\keypoints_q3b
[rule-seeds] 处置: 标签保持 label_status=rule_seed_pending; 权重到位后重跑本脚本即得七类种子草稿。 (exit=2)
```

Q3a/Q3b 属 GPU 接力队列（relay_executor），本窗按纪律不触碰；权重到位后重跑入口脚本即出草稿。

## 6. 合规与许可注记

- 访问方式：yt-dlp 尊重平台 ToS 的公开内容获取；B 站游客 cookie（buvid3）、YouTube 经系统代理 `127.0.0.1:17890`
- 每条 manifest `license_note`: research-use only; public platform content accessed via yt-dlp respecting ToS; no redistribution
- 付费课/登录墙内容全部失败退出，未绕过任何访问控制
- 片段池仅作学术研究训练用途，不对外再分发

## 7. 反方质疑与缓解

| 质疑 | 缓解 |
|------|------|
| B 站含大量 YouTube 搬运 → 同源重复膨胀 | manifest 以 origin_url 去重登记源；下游提点可按 URL 聚类去重；88 源中双语来源互补性已核对 |
| COCO-dog 粗筛可能混入其他四足动物或漏检 | dog_rate 字段保留分层能力；Q3a 犬类专用权重提点时自然二次过滤 |
| ≤30s 段内含多行为混合 | 属契约设计（格式 A 上限），规则种子切段阶段天然细分为秒级单行为段 |
| 0.5 过筛线偏宽松（牵犬师为主镜头混入） | dog_rate=1.0 层 199 条可直接作高置信子集；阈值是配置项可复跑 |
| YouTube 经代理的稳定性 | 断点续传设计（jsonl 状态机），单源失败不阻塞全局；本次 105/115 成功率佐证 |

## 8. 可训性门禁结论

| 门禁项 | 判定 |
|--------|------|
| 体量门（≥500 格式 A 片段） | **PASS**（642，余量 +142） |
| 合规门（契约字段/≤30s/许可注记逐条） | **PASS**（逐行校验 0 违规） |
| 标签门（自产种子草稿就绪） | **PENDING**（策略已接线，等 Q3a/Q3b 提点；非本窗领地） |
| 综合判定 | 片段池具备进入「提点→规则种子」会师路径的条件；可训性行为级结论待骨架序列产出后按 W20 同款口径复核 |

## 9. 复现命令

```
# 全管线(搜索→下载→切分→粗筛→manifest; 断点续传)
.venv/Scripts/python.exe scripts/harvest_video_pipeline.py --config configs/harvest_video_w25.yaml --stage all
# 分批防超时
... --stage download --limit 30   # 然后 --stage split / --stage filter / --stage manifest
# 种子草稿(需 Q3a/Q3b 提点产物先行)
.venv/Scripts/python.exe scripts/harvest_rule_seeds.py --manifest runs/data_campaign/video/manifest.jsonl --keypoints-root runs/data_campaign/video/keypoints_q3b
```

