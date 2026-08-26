# K9 真实域数据获取材料包（W42 交付）

> 版本: v1.0 | 日期: 2026-08-26 | 窗口: wt/W42（纯 CPU，零占卡）
> 目标: 把「等原始视频」从被动等待变为主动出击——邮件能发、拍摄能派、替代源能查、许可有据。
> 定位: 本包是**作战材料层**；战略研讨见 `RESEARCH_DATA_BLOCKADE_SOLUTION.md`（2026-08-19），战役收敛契约见 `dev-docs/handovers/DATA-CAMPAIGN-plan.md` §0。三者不重复 truth：本包只拥有"材料与来源清单"。
> 调研方法: GitHub-First（GitHub 搜索 → awesome 目录 → 逐仓验证活跃度/许可/内容物）+ 本地 C1 manifest 实证统计。零 WebSearch。

---

## §1 samtwl 细粒度犬类行为数据集——邮件申请包

### 1.1 数据集事实卡（逐仓验证 @2026-08-26）

| 项 | 值 |
|----|----|
| 仓库 | `samtwl/Deep-Learning-Fine-Grained-Action-Recognition-Canine-Behavior` |
| 内容物 | code/ + data/ + paper/（论文 PDF 在库）；README 声明 "The dataset will be available upon request." |
| 规模与类别 | **916 视频片段 × 4 类行为 × 2 大类**：负性 {angry, sad} / 正性 {happy, submissive}（源自仓库 `data/Permission-to-Use-Data.md`） |
| 官方申请协议 | 邮件注明 **①机构 ②用途** 即可申请 |
| 官方指定引用 | S., Tong & A., Theodore. (2019). Capturing Spatial and Temporal Context for Fine Grained Canine Behaviour Action Recognition. Retrieved from https://github.com/samtwl/Deep-Learning-Fine-Grained-Action-Recognition-Canine-Behavior |
| 许可状态 | ⚠️ 仓库无 LICENSE 文件；获取方式即"邮件授权制"——书面承诺必须随信给出 |
| 活跃度风险 | 最后推送 2019-06-19；主邮箱为 SMU MITB 学生邮箱（.2017），可能已失效 → **双地址同发**（samuel.tong.2017@mitb.smu.edu.sg 主 + samuel050590@gmail.com 备），30 天无回复则走 GitHub Issue 留言通道 |

### 1.2 邮件草稿文件（发送前替换【】占位符，一律填真实信息）

| 文件 | 语言 | 状态 |
|------|------|------|
| `dev-docs/research/K9-SAMTWL-email-zh.md` | 中文版（通用机构版措辞） | 备用 |
| `dev-docs/research/K9-SAMTWL-email-en.md` | English **v1.2 去AI痕迹+云南警官学院背书版** | **主发版本** |

> 纪律注记：警院身份表述以与云南警官学院的**真实关系**为准（学生/教师/科研/外部合作四选一，邮件文件内附对照表）；禁虚构职务；如有警院邮箱优先用它发送。

草稿已按作者协议内置四要素：数据用途（跨数据集泛化评估 + 细粒度表征分析）、学术框架（低资源动物行为识别研究、拟投 CV 期刊/会议）、引用承诺（逐字采用其指定格式）、许可承诺（组内学术使用/不再分发/不上传公开网/按要求删除并书面确认）。

---

## §2 基地拍摄清单——一页纸转发版

文件：`dev-docs/research/K9-BASE-FILMING-CHECKLIST.md`（独立成文，直接转发基地即可）

规格对齐依据（为什么这么定）：

| 清单条目 | 对齐的管线契约 |
|---------|---------------|
| 七类 = lying/sitting/standing/walking/running/rise_transition/jump | `psd/data/rule_seeds.py` 物理先验类别体系（单一 truth，YAML 可配但引擎内聚） |
| 单段 ≤30 秒、一段一主行为 | C1 格式 A 契约 `max_len_s: 30`（`configs/harvest_video_w25.yaml`）；超长会被切分器打散 |
| rise_transition/jump 标注稀缺重点补 | W35 实测五类高度依赖类规则命中全零、jump+stay 类不平衡（Q3c 公开真实层 44.90% 的结构性成因之一） |
| 单犬优先/多犬不重叠 | YOLO-pose 提点串犬污染风险（多犬重叠帧提点不可分） |
| 固定机位侧面全身 | 骨架序列时序连续性要求（镜头跟随=全局运动淹没行为运动） |

---

## §3 替代源清单（APTv2 之外）

### 3.1 公开数据集二次筛查表（全部经 GitHub 工具链逐仓验证 @2026-08-26）

| # | 数据集 | 类型 | 犬覆盖 | 规模 | 获取 | 许可/注记 | 对本管线用途 | 判定 |
|---|--------|------|--------|------|------|----------|-------------|------|
| D1 | **MammalNet**（Vision-CAIR，CVPR 2023） | 视频+行为标签 | ✅ 家犬属 173 哺乳动物类目之一 | 12 类高层行为；trimmed/full 视频 S3 直链 | wget 直下（README 内嵌链接） | 仓库无显式 LICENSE 文件；官方 BibTeX 提供 → 学术引用制 | 预训练池扩广度（视频级行为标签，非骨架） | **推荐评估下载 dog 子集** |
| D2 | **AP-10K**（Yu et al. 2021） | 静态姿态图 | ✅ 54 种含犬科 | 10,015 图 × 17kp × bbox | openreview 页面链接 | 学术发布，投稿前终审引用条款 | 静态池/预训练增广（沿 C5 判例 b 口径） | 推荐（静态池） |
| D3 | **Animal-Pose**（Cao et al. ICCV 2019） | 静态姿态图 | ✅ 含 dog | 4,000+ 图 × 20kp × bbox | 官方 Google Sites | 学术发布 | 同上；20kp 与 APTv2 17kp 拓扑近似 | 可选（与 D2 功能重叠） |
| D4 | **APT-36K**（pandorgan/APT-36K） | 视频姿态图 | ✅ 30 种 | 36,000 图 / 2,400 clip × 17kp | GitHub 指路 | 学术发布 | —— | **跳过**：APTv2 即其 v2 谱系，本地已有 83,304 文件全量 |
| D5 | **DogCentric Activity Dataset**（九州大学 AIT，2014） | 犬载第一视角活动视频 | ✅ 4 只犬 | 活动：turn/look/shake 等 | 项目页开放下载 | 学术发布（年代久，条款以项目页现行文本为准） | 视角鲁棒性研究素材（ego 域，非主粮） | 低优先候选 |
| D6 | **Inertial data for dog behaviour classification**（VetDataHub 收录，Kaggle benjamingray44） | IMU 时序 | ✅ 项圈+胸背 100Hz | 7 任务：gallop/lying chest/sit/sniff/stand/trot/walk | Kaggle | Kaggle 条款待下载页确认 | **跨模态先验**：7 任务与我们七类中 5 类同名对应，可做规则阈值旁证 | 推荐评估（非视频主管线） |
| D7 | Animals with skelet key points and action mark（Kaggle egorovalexeyd） | 关键点+动作标记 | 未验证物种构成 | 未验证 | Kaggle | 未验证 | —— | 待筛查（质量存疑，先小样验证再定） |
| D8 | Dog Behavior Action Classification 5015 图 9 类（lonlonago） | 静态图分类 | ✅ bite/lie/sit/sleep 等 9 类 | 5,015 图 720×720 | **Stripe 付费 $89** | ❌ 无许可条款、来源不明 | —— | **出局**（付费+无许可+仅静态图） |

> 已有资产不重复筛：Animal Kingdom / StanfordExtra(dog-pose) / DogSet 动捕 / InterPet4D / SyDog-Video(待评估) 均已登记于 `docs/DATA_LOCATIONS.md`。

### 3.2 警犬/搜救犬训练公开视频——频道目录与扩采协议

#### a) C1 已验证来源池实证（本地 manifest 统计 @2026-08-26）

- 642 片段来自 **88 个唯一源视频**：Bilibili 47 / YouTube 41；
- 关键词族覆盖失衡（片段数）：`Belgian Malinois training` 81 ≫ `警犬 训练` 31 > `dog obedience trial` **2**（近乎空白）；
- manifest 无 uploader 字段 → 本次抽样实拉元数据补齐（见 c）。

#### b) B 站频道目录（yt-dlp 实拉验证，按片段贡献排序）

| 上传者 | 贡献片段 | 内容定位 |
|--------|---------|---------|
| 宾宾与莜莜 | 32+12 | 训犬科普/随行讲解系列 |
| 运动AI | 22 | 马犬训练搬运 |
| 林小Jim | 18 | 护卫犬知识科普 |
| licensetolive | 16 | 大型犬警犬训练合集搬运 |
| 自然收集器 | 15 | 犬种服从性盘点 |
| 犬知者志哥的宠食小店 | 15 | IGP3 防卫视频解说 |

#### c) YouTube 侧状态与重跑协议

- 抽样当日代理 `127.0.0.1:17890` 不可达，41 个 YouTube 源 URL 全部 FETCH_FAILED（W25 曾成功，网络条件性故障）；
- 重跑命令模板（代理恢复后执行，CPU/网络任务不占卡）：
  `python -m yt_dlp --proxy http://127.0.0.1:17890 --skip-download --print '%(channel)s|%(title)s' <url>`
- 扩采关键词建议（对齐缺口）：中文族补「搜救犬 训练」「犬 越障 训练」；英文族补 `search and rescue dog training`、`IRO dog training`、`k9 agility training`、`police k9 deployment`。

#### d) 组织/社区线索（GitHub 证据 → 平台内容源）

| 线索 | 证据仓 | 说明 |
|------|--------|------|
| IRO（国际搜救犬组织）系赛事/训练视频 | `mwrnckx/K9-Mantrailing-Analyzer`（topics: iro/mantrailing/search-and-rescue/tracking） | 该工具面向 IGP/IRO/mantrailing 训导员 → 社区存在公开训练记录传统，YouTube/B站检索上述术语可发现频道 |
| IGP/Schutzhund 俱乐部圈层 | C1 关键词族已验证有效（IGP/Schutzhund 两族合计 239 片段） | 维持现有词族，向赛事全程录像方向加深 |

> 诚实声明：GitHub-first 检索未发现现成的"警犬训练频道目录"清单仓（多词 AND 检索零命中，短语检索命中的均为无关游戏模组/App）。上表目录为**本地实证池 + 实拉元数据**构建，扩采靠协议而非静态名单——这与 C1 判例一致。

### 3.3 兽医 / 动物行为学合作机构线索

| 机构/社区 | GitHub 存在 | 合作切入点 | 活跃度 |
|-----------|------------|-----------|--------|
| **UNL 犬类认知与人类互动实验室（CCHIL，University of Nebraska–Lincoln）** | `unl-cchil/canine_precise_dispenser`（犬认知实验器材开源） | 犬类认知实验合作/数据共享问询 | 2026-07 有更新 ✅ |
| **VetDataHub 兽医数据集社区** | `Vetdatahub/VetDataHub`（MIT 协议，25 个数据集分类栏含 canine/behavioral） | ① 提 Issue 征集犬行为视频数据集线索 ② 未来本仓规则种子产物反哺社区（需用户裁决） | 2026-07 更新 ✅ |
| **东京农工大 TUAT 生物信号信息学研究室** | `giovanni-gallerani/dog-behavior-dataset-creation-tools`（犬行为研究数据集建库工具，BIDS 规范） | 犬行为数据采集规范交流 | 2026-04 更新 ✅ |
| **动物行为学视频编码社区（BORIS 用户群）** | `olivierfriard/BORIS`（245★，动物行为观察标准软件） | 社区论坛/Issue 区触达动物行为学研究者（ethologists） | 持续维护 ✅ |
| 九州大学 AIT 机器人实验室 | DogCentric 数据集托管方（见 §3.1 D5） | ego 犬视角数据合作 | 项目页存续 |

---

## §4 许可与伦理注记汇总（逐项）

| 来源 | 许可/伦理要点 | 本仓动作约束 |
|------|--------------|-------------|
| samtwl 数据集 | 邮件授权制；须附指定引用；无 LICENSE 文件=默认保留所有权利 | 未获书面授权前不入库不入管线；获权后仍禁再分发 |
| MammalNet | 无显式 LICENSE；CVPR 2023 论文发布惯例=学术引用制 | 下载后仅研究用；引用 BibTeX；投稿前终审条款（沿 R-C2-1 判例口径登记 DATA_LOCATIONS） |
| AP-10K / Animal-Pose / APT-36K | 学术发布 | 引用原论文；拓扑映射前过 assets-map 审查（外来标签映射审查纪律） |
| DogCentric | 年代久远，条款以项目页现行文本为准 | 用前截图存档条款页面 |
| VetDataHub 收录 IMU 集 | 仓库 MIT（代码）≠数据集条款；Kaggle 页面条款为准 | 下载时逐条核对 |
| lonlonago 5015 图 | 付费+无许可 | 不接触 |
| C1 公开平台片段（既有 642 条） | research-use only; yt-dlp 尊重 ToS; no redistribution（manifest 逐条 license_note 判例） | 新扩采沿用同一注记模板 |
| **基地自采（新）** | 训导员知情同意留痕；正向强化无强迫；应激即停；素材不再分发；论文配图协商 | 登记表必填授权确认字段；沿格式 A 契约入库 |
| DogSet 动捕（已有） | Edinburgh IP 研究教育专用、禁商用禁再分发（DATA_LOCATIONS §5 已登记） | 不变 |
| 通用红线 | 三层指标口径禁止混报：任何新源入库必须在 DATA_LOCATIONS 登记"公开真实层"归属 | 沿 AGENTS 硬规则 2/3 |

---

## §5 下一步行动清单（按优先级）

1. **发 samtwl 邮件**（主发 gmail samuel050590@gmail.com、Cc SMU 学生邮箱；占位符填真实信息后即可发；2 周跟进 / 30 天无回复转 Issue 通道）；状态：待发送；
2. **转发拍摄清单一页纸**给基地联系人，启动首批 ≥20 段/类 × 7 类采集（稀缺类 rise_transition/jump 加倍）；
3. **C1 补采缺口**：新增关键词族「搜救犬 训练」「search and rescue dog training」「IRO dog training」「k9 agility training」（CPU/网络，沿 harvest_video_pipeline 复跑，需开新窗或排期）；
4. YouTube 41 URL 频道名补拉（代理恢复后，§3.2c 命令模板）；
5. MammalNet dog 子集评估下载（trimmed 包直链，先验规模再全量）；
6. VetDataHub 提 Issue 征询犬行为视频数据集线索（措辞可复用 samtwl 邮件框架）。

## §6 证据与复现

- GitHub 检索路径：user:samtwl → 18 仓定位目标仓；topic:animal-behavior（147 结果）→ BORIS/CCHIL/TUAT；topic:veterinary（420 结果）→ VetDataHub；"MammalNet"/"DogCentric"/topic:dog-training 逐一短语检索；awesome 目录两份（JackieZhai/awesome-behavior-datasets、ChaoYue0307/awesome-egocentric-atlas）作发现入口；
- 逐仓验证字段：stars/pushed_at/license/README 内容物，均于 2026-08-26 当次拉取；
- 本地统计复现：
  ```powershell
  $rows = Get-Content D:\Desktop\psd-framework\runs\data_campaign\video\manifest.jsonl | ForEach-Object { $_ | ConvertFrom-Json }
  ($rows.origin_url_or_path | Sort-Object -Unique).Count   # 88
  ```
- 频道名实拉样例（B 站 8/8 成功）：见 §3.2b；YouTube 0/6 成功（代理不可达）已如实披露。

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-26 | W42 首版：邮件双版+拍摄清单一页纸+替代源筛查 8 项+频道目录 6 频道+机构线索 5 条+许可伦理注记 |
| v1.1 | 2026-08-26 | 英文邮件改独立研究者诚实版（v1.1，禁伪造机构的纪律注记入 §1.2）；§5 行动 1 补收发件人安排与状态位 |
