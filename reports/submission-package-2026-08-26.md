# 投稿打包预备报告（W41 · Pattern Recognition 方向）

> Owner: `reports/submission-package-2026-08-26.md` · 执行窗: **W41**（wt/W41，B-full 协议）· 日期: 2026-08-26 · 任务类型: CPU（零占卡零 GPU 请求）
> 任务书: 投稿打包预备五件事（①格式核查 ②Cover Letter ③DA/Ethics 声明 ④fig1-4 终检 ⑤代码发布包清单）
> 输入: HANDOVER v2.4 快速启动节 / AGENTS.md / BOARD 尾部 / experiment-skeleton.md v1.0 / outline.md v1.0（R1-R11）/ figure-specs.md v0.1 / FIGURE_SOURCE.md v0.2 / introduction.md v0.3 / conclusion-limitations.md v0.4 / p01 报告 §7 / requirements.txt
> 外部信源: Elsevier 官方《Guide for Authors—Pattern Recognition》（sciencedirect.com/journal/pattern-recognition/publish/guide-for-authors，抓取日 2026-08-26）。注：本项非技术选型调研，系期刊投稿要件的一手信源核验，不触 AGENTS 硬规则 1（GitHub-First 技术调研禁 WebSearch）的适用域。

---

## 0. 结论速览

| 任务 | 判定 | 要点 |
|------|------|------|
| ① PR 格式核查 | 🟡 有硬缺口 | 仓库**零 .tex 文件**，LaTeX 工程未建；**投稿必须单栏双倍行距**（任务书中"双栏"预设不成立，见 F1）；Highlights 强制未备；四项强制声明中两项已起草（§3）、两项待用户 |
| ② Cover Letter | ✅ 草稿成 | 三支柱+双档 C1 证据+八条诚实边界，全文见 §2.1；配套 Highlights 五条全部实测 ≤77 字符（≤85 达标） |
| ③ DA + Ethics | ✅ 草稿成 | R7 缓解落地；数据许可逐源审计表 + 两处【待用户终审】标注（§3） |
| ④ fig1-4 终检 | ✅ 通过（附残留注记） | 四张 PDF 全矢量零栅格、字体全内嵌；fig3/fig4 PNG 600dpi 达标；色盲安全+双编码实证；描边灰度差偏近已由线型/标记/纹理补偿（§4） |
| ⑤ 代码发布包清单 | 🟡 待用户两项 | 物料盘点齐（psd/scripts/configs/README/requirements）；复现链命令节全部在档；**无 LICENSE 文件**——选型与开源时机待用户（§5） |

---

## 1. 任务①：Pattern Recognition 投稿格式核查

### 1.1 官方要件速查（2026-08-26 抓取）

| # | 要件 | 官方要求（原文口径） | 本仓现状 | 判定 |
|---|------|---------------------|---------|------|
| F1 | 版式 | **单栏、双倍行距（LaTeX）/1.5 倍距（Word）、两端对齐、编页码**；"double-column formatting is not allowed" | outline §7 内部页数预算按"PR 双栏终稿估算"书写 | ⚠️ 认知纠偏：终稿排版由出版社负责，**投稿稿单栏**；outline 该行是篇幅估算工具非版式指令，不需回改，但 LaTeX 工程必须按单栏搭 |
| F2 | 页数 | **20–35 页（含图表、参考文献、bio-sketches、附录）**；survey 至 40；**<20 页将被建议转投 Pattern Recognition Letters** | 内部预算 12.7 双栏密集页 ≈ 单栏双倍行距约 28–33 页（经验系数 2.2–2.6×） | 🟡 贴上限：附录计入页数 → 建议创新性排查矩阵与一条命令复现链移 Supplementary Material（不计页），成稿装配期复核 |
| F3 | 模板 | 指南当前链接指向 **els-cas-templates.zip**（cas-sc.cls 单栏 / cas-dc.cls 双栏）；经典 elsarticle 经 elsevier.com/latex 仍可用 | **仓库零 .tex**，工程未建 | 🔴 缺口：终稿装配前置项；推荐见 §6 决策 A |
| F4 | 标题页 | 标题精炼（理想 10–15 词）、作者/单位/通讯信息齐备 | 标题现 15 词（上沿）；作者信息空白 | 🟡 标题终裁本就挂 R2 待用户；作者名单待用户 |
| F5 | 摘要 | ≤250 词 | 现稿 ~200 词（introduction.md v0.3 定稿候选） | ✅ |
| F6 | 关键词 | 1–7 个 | 未定稿 | 🟡 终稿期定（建议 5–6：animal behavior recognition / skeleton-based action recognition / self-supervised learning / pseudo-labeling / decoupled framework / low-resource） |
| F7 | **Highlights** | **强制**：独立可编辑文件、文件名含 "highlights"、3–5 条、每条 ≤85 字符（含空格） | 未备 → **本窗已起草并程序化验证字符数**（§2.2） | ✅ 草稿就绪 |
| F8 | 图形摘要 | 鼓励非强制 | 无（fig1 可改造） | ⚪ 可选项，默认不做，待用户加购意愿 |
| F9 | 图片规格 | 矢量优先；halftone ≥300dpi；线图 ≥1000dpi（单栏宽 ≥3543px）；混合 ≥500dpi（≥1772px）；文字不得转图形；须照顾色觉障碍读者 | 见 §4 逐图核验 | ✅ 提交走矢量 PDF 即达标 |
| F10 | Research Data | **Option C 适用**；**投稿时强制 Data statement**（数据可得性声明，不可分享须说明原因） | 未备 → 本窗已起草（§3.1） | ✅ 草稿就绪 |
| F11 | Competing interests | 全体作者强制声明；经系统生成 Word 上传；无可声明选 "nothing to declare" | 未备（系统内操作） | 🟡 投稿操作项，待用户确认后一键完成 |
| F12 | CRediT 作者贡献 | 强制（14 角色taxonomy） | 未备 | 🟡 待用户定作者名单后按角色填写（模板句见 §3.3 附注） |
| F13 | **Generative AI 声明** | **投稿时强制声明**写作过程中 AI 工具的使用；AI 不可署名；作者对内容负全责 | 未备 → 本项目 AI 协作深度介入（多智能体窗口制），**必须如实声明** | 🔴【待用户终裁】声明措辞与范围（§3.3 给出建议稿） |
| F14 | 评审模式 | single anonymized（审稿人可见作者，反之不可）→ 稿件无需匿名化 | — | ✅ 无额外工作 |
| F15 | 预印本 | 允许；提供 SSRN 免费预印服务 | 尚无 arXiv/SSRN 预印本 | ⚪ 与开源策略联动，待用户 |
| F16 | 资助声明 | 强制（无资助用标准句） | 未备 | 🟡 标准句已入 §3.3 |

### 1.2 对照 experiment-skeleton 图表规范节（逐项过）

skeleton §图表规范六条 vs PR 官方要求交叉核验：

| skeleton 规范条 | 与 PR 官方的兼容性 | 核验结果 |
|----------------|------------------|---------|
| 白底 #FFFFFF + 浅灰网格，禁深底 | 兼容（Elsevier 无冲突要求） | ✅ 四图实测白底 |
| 单图系列色 ≤6、低饱和淡彩 | 兼容且优于最低要求 | ✅ fig1 用色 4 族、fig3 青/橙/灰三族、fig4 双系列 |
| 色盲安全：禁纯红绿、优先蓝橙、过灰度打印 | 与官方"readable ... impaired color vision"同向 | ✅ 青-橙对 + 双编码实证（§4.3） |
| 曲线/柱状矢量输出（PDF/EPS） | **正是 PR 首选形态**（editable source 要求矢量） | ✅ 四张 PDF 全矢量 |
| 照片类定性图 PNG ≥600 DPI | 官方线图线要求更高（≥1000dpi/3543px） | ✅ fig3 6470×4132px@600dpi、fig4 4000×2670px@600dpi，像素宽均超 3543px 线图门槛 |
| 表格：最优加粗+方向符号、数值右对齐小数位一致 | 兼容（另注意官方"避免竖线与单元格底纹"） | 🟡 tab2/tab3 成表排版期执行（素材已齐：tab3 六行零 PENDING） |
| 图内不放标题（caption 自足） | 兼容（官方 caption 规则同向） | ✅ fig1-4 均无内嵌标题，caption 草稿在 FIGURE_SOURCE.md |
| 误差棒：多种子必带 std/SE | 兼容 | ✅ fig4 mean±std over 3 seeds 已带 |

---

## 2. 任务②：Cover Letter 草稿 + Highlights

### 2.1 Cover Letter（英文全文草稿，占位符以【USER】标注）

> Dear Editors-in-Chief of *Pattern Recognition*,
>
> We are pleased to submit our manuscript "**A Physics-Semantics Decoupled Framework for Low-Resource Animal Behavior Recognition under Evolving Evaluation Criteria**" for consideration as a Regular Paper in *Pattern Recognition*.
>
> Animal behavior annotation is scarce and expensive, while operational evaluation criteria keep evolving—every taxonomy revision currently forces re-annotation and retraining. Our manuscript addresses this with three contributions:
>
> **(1) A physics-semantics decoupled architecture.** We formalize evolving evaluation criteria as taxonomy transitions absorbed by a lightweight semantic layer alone. The claim rests on dual-tier controlled evidence: at matched training budgets, taxonomy-transition retraining costs **≥3× less wall-clock time than full-pipeline retraining** (conservative bound backed by a measured 6.07× on the full tier and 7.32× on the small tier, same-seed paired minimum 4.00×), with accuracy statistically equivalent between arms (−0.91 pp full / +2.27 pp small, both below 2.3 pp).
>
> **(2) The first transfer of image-domain anchor–cluster–pseudo-labeling to temporal-skeleton recognition.** A repository-scale survey across ten query families found zero prior occupancy of this combination; arXiv/Scholar re-verification is scheduled before submission.
>
> **(3) Label-free validation on public real animal data under a strict three-caliber protocol.** Self-supervised pretraining reaches **20.89% kNN top-1 versus an 8.33% random baseline (2.51×)** on InterPet4D quadruped skeletons; unsupervised motion-word segmentation attains boundary IoU **0.458 ± 0.049**, above an equal-segment-count sliding-window arm (0.399 ± 0.035) and a random-cut null (0.323 ± 0.022) under a pre-registered criterion; and warm-started semantic-layer initialization makes a **20-clip budget usable (82.0% top-1 on the synthetic-offset tier)**.
>
> We believe this fits the journal's scope at the intersection of representation learning, temporal pattern segmentation, and low-resource recognition. In the spirit of transparent reporting, the manuscript explicitly discloses its boundaries: behavior-level evidence is tiered and caliber-labeled throughout; uncertainty-based active-learning sampling is reported strictly as an exploratory negative finding confirmed across three independent runs; real-domain validation covers a single species family (canids) with 20/24 effective supervision channels on extracted skeletons; one public-real result carries severe class imbalance disclosed per-class; all experiments ran on consumer-grade hardware; and our firstness claim is bounded by scheduled scholarly re-verification. None of these boundaries, we argue, undermines the central claim—they are precisely what the decoupling design absorbs.
>
> This manuscript is original, has not been published previously, and is not under consideration elsewhere. All authors have approved the submission and declare no competing interests.【USER 复核】The study uses exclusively publicly available datasets; no new animal experiments were conducted by the authors.
>
> Thank you for your consideration.
>
> Sincerely,
> 【USER：通讯作者姓名/单位/邮箱】
> on behalf of all authors

### 2.2 Highlights 草稿（强制件，字符数程序化实测）

| # | 内容 | 字符数（≤85） |
|---|------|--------------|
| H1 | Physics-semantics decoupled framework absorbs taxonomy evolution cheaply | 72 ✅ |
| H2 | Taxonomy transitions absorbed at >=3x lower wall-clock cost, matched accuracy | 77 ✅ |
| H3 | First anchor-cluster-pseudo-label transfer to temporal skeleton recognition | 75 ✅ |
| H4 | Label-free pretraining reaches 2.51x random kNN on real quadruped skeletons | 75 ✅ |
| H5 | Warm start makes a 20-clip budget usable: 82.0% top-1 under shift | 65 ✅ |

> 注：H2/H5 若终稿数字因预注册条款回改（full 档趋势矛盾/round2 复核），Highlights 同步改写。提交时存独立文件 `highlights.tex`（或 .docx），文件名含 "highlights"。

---

## 3. 任务③：Data Availability + Ethics 声明草稿（R7 缓解）

### 3.1 Data Availability Statement（英文全文草稿）

> **Data availability**
>
> All datasets used in this study are publicly available from their original providers: InterPet4D (Hugging Face, `ohicarip/interpet4d`, CC BY-NC 4.0)【待用户终审-A：派生骨架特征是否随包再分发，还是仅提供提取脚本】; Animal Kingdom (CVPR 2022 official release); APTv2 (official repository, ViTAE-Transformer/APTv2); NTU RGB+D 60 (obtained under the provider's research-use license terms)【待用户终审-B：许可证文本留档确认】; and a canine pose corpus derived from publicly released annotations (dog-pose/StanfordExtra lineage). The synthetic benchmark is fully generated by our released code (`configs/syn_v2_fidelity.yaml`, deterministic seeds). No new data were collected by the authors.
>
> All reported numbers are reproducible from archived configurations and one-command chains in the released repository【USER：仓库 URL 占位】; per-experiment evidence JSON files accompany the code. Data derived from third-party datasets are redistributed only within the terms of each source license; where a license restricts redistribution, we provide generation scripts instead of raw derivatives.【待用户终审-C：该承诺的最终边界】

### 3.2 Ethics Statement（英文全文草稿）

> **Ethics statement**
>
> This study exclusively re-analyzes publicly available, previously published datasets collected by third-party research groups; the authors conducted no new experiments on animals or human subjects. Original data collection complied with the protocols of the respective providers (InterPet4D, Animal Kingdom, APTv2, NTU RGB+D, and public canine pose corpora), cited in Section 4.1.【待用户终审-D：InterPet4D 含 ego-centric 人-宠互动内容，投稿前从数据集文档摘录其人体被试伦理/知情同意原句留档】 Sex- and gender-based analyses were not performed; this scope restriction is acknowledged within the limitations on cross-family generalization (Section 6). The application context (working-dog training) involves no intervention by this work; all reported systems are offline analysis pipelines.

### 3.3 配套强制声明的标准句（投稿系统直接粘贴）

- **Funding**: "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."【USER 如有资助替换】
- **Declaration of competing interest**: 经投稿系统问卷生成 Word 上传；建议选 "I have nothing to declare"【USER 复核】。
- **CRediT**: 待作者名单确定后按 14 角色填写；建议骨架 = Conceptualization / Methodology / Software / Validation / Investigation / Data curation / Writing – original draft / Writing – review & editing / Visualization / Supervision。
- **Generative AI declaration（建议稿，🔴 待用户终裁措辞与范围）**: "During the preparation of this work the authors used [tool names] in order to assist with code development, experiment orchestration, and language editing. After using this service/tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article." —— 本项目采用多智能体协作开发（实验编排/TDD/图表生成均有 AI 参与），按官方政策 AI 不可署名、作者负全责；**披露颗粒度（工具名列举 vs 概括表述）请用户拍板**。

---

## 4. 任务④：fig1-4 终检（对照 figure-specs.md + skeleton 图表规范）

### 4.1 矢量完整性（PDF 解压流逐张核验，2026-08-26 当次运行）

| 图 | 文件 | 栅格 XObject | 路径算符 'm' | 文本块 BT | Tj/TJ | 字体内嵌 | 后端 |
|----|------|-------------|-------------|-----------|-------|---------|------|
| fig1 | fig1_framework_overview.pdf (34.8KB) | **0** | 41 | 27 | 27 | ✅ FontFile×3 | Matplotlib 3.8.4 |
| fig2 | fig2_pseudo_label_loop.pdf (30.6KB) | **0** | 27 | 20 | 20 | ✅ ×3 | 3.8.4 |
| fig3 | fig3_segmentation_qualitative.pdf (27.7KB) | **0** | 264 | 61 | 69 | ✅ ×2 | 3.11.1 |
| fig4 | fig4_al_efficiency.pdf (34.6KB) | **0**（8 个 Do 均为 Form XObject 矢量组） | 64 | 20 | 30 | ✅ ×4 | 3.11.1 |

**判定：四张全矢量、零嵌入位图、字体全内嵌（Type-42/TrueType 子集），满足 PR "editable source + 矢量优先" 要求。**

### 4.2 分辨率（PNG 预览件 vs 提交件口径）

| 图 | PNG 预览 | pHYs DPI | 官方线图像素门槛（≥3543px 单栏） | 判定 |
|----|---------|----------|--------------------------------|------|
| fig1 | 1969×1141 | 200 | 不适用（架构示意图，提交走矢量 PDF） | ✅（PNG 仅预览用途） |
| fig2 | 1659×1277 | 200 | 同上 | ✅（同上） |
| fig3 | 6470×4132 | **600** | 6470px ✅ 超线图级门槛 | ✅ |
| fig4 | 4000×2670 | **600** | 4000px ✅ | ✅ |

> 注记：若未来需要 fig1/fig2 的栅格备份（如投稿系统仅收位图），须重跑生成脚本并以 ≥500dpi 导出（当前 200dpi 为 W17 预览设定，FIGURE_SOURCE.md 已注明 PDF 为正式产物）。

### 4.3 灰度打印 / 色盲安全（程序化取色 + 脚本双编码实证）

- **色相轴**：全系列采用青 (#DAFFFF/#0E7490) × 橙 (#FFE3DA/#C2410C) 对——蓝橙系为色盲安全组合，无红绿对。✅
- **淡彩填充灰度同亮度现象**：实测青填充 L=0.957 vs 橙填充 L=0.919，ΔL≈0.038——这是 skeleton 规范"低饱和淡彩"令的必然结果，**属设计预期而非缺陷**；判据落在第二编码通道：
  - fig4（脚本实证）：熵臂=实线+圆点标记，随机臂=虚线 (0,(5,2.5))+方块标记——**颜色+线型+标记三重编码**，纯灰度下完全可分辨。✅
  - fig3（脚本实证）：基线带 hatch="///" 纹理 + GT 带内嵌类文字标签 + GT/预测上下轨道空间分区——纹理+文字+位置多重编码。✅
  - fig1/fig2：物理层/语义层左右（上下）空间隔离 + 黑字模块名，色彩仅作氛围层。✅
- **描边色灰度间距**：#0E7490 L=0.348 vs #C2410C L=0.382，ΔL=0.035 偏近（<0.10 启发阈值）——在依赖描边区分系列的场景下偏弱；当前四图的系列区分均已由上述第二通道承担，故判 PASS；**建议**（非阻塞）：终稿若新增依赖描边辨别的数据图，优先加大明度差（如改用 #155E75 vs #EA580C，ΔL≈0.09→配合线型）。
- **fig4 负结果如实呈现**：图内注释显式标注随机反超区间（b≥100），caption 写明冷启动协议无效率优势——与 FIGURE_SOURCE.md v0.2 登记、E5 降级裁决一致。✅

### 4.4 figure-specs.md 符合性偏差登记（沿用 FIGURE_SOURCE v0.2 五条）

W22 已登记的五条偏差（fig4 不画 85% 目标线属三层口径纪律、fig3 用 recheck JSON、防樱桃采摘 4/4 episode 全展示等）本次逐条复核仍成立，无新增偏差；figure-specs.md 中 fig3/fig4 "规格待定项"实际已由 FIGURE_SOURCE.md v0.2 + 脚本固化——**规格 truth 仍在 figure-specs（owner），执行细节在 FIGURE_SOURCE（溯源册）**，符合 truth 单一性分工，不改文件。

---

## 5. 任务⑤：代码发布包清单

### 5.1 发布物料盘点（当次 worktree 实测）

| 类别 | 内容 | 状态 |
|------|------|------|
| 核心 Python 包 | `psd/`（data/models/training + tests） | ✅ 在库 |
| 入口脚本 | `scripts/*.py` 47 个（p01-p05/ntu/harvest/ablation/seg_ablation 全链） | ✅ 在库 |
| 实验配置 | `configs/*.yaml` 25 个（逐实验 config_echo 可溯源） | ✅ 在库 |
| 环境锁定 | `requirements.txt`（torch 2.11.0+cu128 口径注释在案） | ✅ |
| 项目 README | 根 `README.md` | ✅ 存在（发布前须补快速上手节，见 5.3） |
| 数据指针 | `docs/DATA_LOCATIONS.md` | ✅ |
| 证据链 | `reports/*.json`（knn-result/c1-cost/warmstart/Q3c/eC-seeds 等） | ✅ |
| **LICENSE** | **不存在** | 🔴【待用户终审-E】 |
| CITATION.cff / .zenodo.json | 不存在 | ⚪ 可选项 |

### 5.2 一条命令复现链核对（硬规则 4：只认在档命令节）

| 实验 | 命令归档处 | 核验 |
|------|-----------|------|
| P0.1 预训练+kNN | `reports/p01-aimclr-2026-08-23.md` §7（venv→export→train→eval 四步全命令） | ✅ 当次复读在档 |
| P0.2 SMQ/E-C | `reports/p02-*` + `scripts/train_smq_segmentation.py`/`eval_smq_segmentation.py` + `configs/p02_smq_eC.yaml` | ✅ 物料在库 |
| P0.3 PhaseA/B | `run_p03_phasea.py`/`run_p03_phaseb.py` + `configs/p03_jia_*.yaml` | ✅ |
| P0.4 TCL | `run_p04_tcl.py` + `configs/p04_tcl.yaml` | ✅ |
| P0.5 AL/warm-start | `run_p05_al_efficiency.py`/`run_p05_al_warmstart.py` + `configs/p05_al_{short,full}.yaml`/`p05_al_warmstart_*.yaml` | ✅ |
| C1 解耦成本 | `run_c1_decouple.py` | ✅ |
| 公开真实微调 Q3c | `run_p05_public_real_pipeline.py`/`run_p05_public_real_finetune.py` | ✅ |
| tab3 三臂消融 | `seg_strategy_ablation.py` + `configs/seg_ablation_p02.yaml`；`run_ablation_pretrain.py` + `configs/ablation_pretrain.yaml` | ✅ |
| 合成保真 syn_v2 | `configs/syn_v2_fidelity.yaml`（内嵌一键刷新命令） | ✅ |
| 回归测试 | `pytest psd -q`（master 最近全量绿：402，BOARD 08-26 00:28 W38 收编门禁） | ✅ |

### 5.3 打包结构建议（录用/投稿随附二选一，见决策 B）

```
psd-framework-release/
├── LICENSE                      ← 【待用户终审-E】
├── README.md                    ← 补：环境/数据获取/三条主命令/测试
├── requirements.txt
├── psd/  scripts/  configs/
├── docs/DATA_LOCATIONS.md       ← 数据获取指针
├── reports/                     ← 证据 JSON（论文数字溯源）
└── external/README.md           ← 不随包分发上游仓；给 clone 指针+AimCLR 权重初始化说明
排除项: data/（三方数据集本体）、runs/（可再生）、.venv/、dev-docs/（内部治理）、*.pt 大文件（提供下载链接或导出脚本）
```

---

## 6. 需要默默拍板的决策点（对比表 + 推荐）

### 决策 A：LaTeX 模板（cas-sc vs elsarticle）——✅ 已裁决并落地（2026-08-26 用户拍板 cas-sc）

> **执行结果（W41 同日交付）**：`docs/paper/latex/` 脚手架建成并冒烟编译全绿（pdflatex+bibtex 三段 exit=0，
> main.pdf 3 页/highlights.pdf 1 页；类选项 review=doublespacing 经 cls L67/L138 实证；题录首条取官方仓
> AimCLR BibTeX GitHub 取证；thumbnails/ 官方资产随包）。装配纪律与排障记录见 `docs/paper/latex/README.md`。

| | els-cas `cas-sc.cls`（官方现行推荐） | 经典 `elsarticle`（老牌通用） |
|--|--|--|
| 得到什么 | 与当前 Guide for Authors 直链一致；单栏评审版开箱即用；CAS 系是 Elsevier 主推新模板 | 社区资料最多、教程最全；历史兼容性最好 |
| 代价 | 模板较新，中文社区示例少；宏包行为偶有差异 | 官方页面已不再直推；部分新功能（CRediT 块等）要手搓 |
| 风险 | 低（Elsevier 自家维护） | 低（长期可用） |
| 反方质疑 | "cas 系会不会不稳定？"——Elsevier 自家 CI 维护，投稿系统原生支持 | "老模板会不会被嫌弃？"——不会，官方仍收录；但评审版格式需自行配置 review 选项 |

**歆歆推荐：cas-sc（els-cas 单栏）**——理由一句话：跟着官方指南当前直链走，把"格式不符被打回"的风险压到最低。最终决定权在你。

### 决策 B：开源许可（发布包 + 代码公开时机）

| | MIT | Apache-2.0 | CC BY-NC 4.0（仅文档/数据类） |
|--|--|--|--|
| 定位 | 最宽松代码许可 | 宽松+显式专利授权 | 不适合软件代码（非软件许可） |
| 好处 | 采用摩擦最小，利于论文影响力传播 | 企业采用无忧；专利条款保护贡献者 | 保护免于商用 |
| 代价/风险 | 他人可闭源商用（含 K9 产品线竞品场景） | 文本较长，其余几乎无 | 代码用它=许可错配，审稿人可能挑刺 |
| 反方质疑 | 学术发表通常不在乎商用封闭 | 无显著反方 | 若想禁止商用应代码 Apache/MIT+数据 NC 分层 |

**歆歆推荐：代码 Apache-2.0 + 数据/派生特征遵循各源许可分层声明**——理由一句话：专利条款对本仓大量移植改编代码（AimCLR/SMQ 适配层）更稳妥，且与 DA 声明的"按源许可分层"天然咬合。开源时机（投稿即公开匿名仓 vs 录用后）与许可终裁一起归你拍板【终审-E】。

### 其余待用户项汇总（红色通道）

| 编号 | 事项 | 出处 |
|------|------|------|
| 终审-A | InterPet4D CC BY-NC 4.0 下派生骨架特征的再分发边界 | §3.1 |
| 终审-B | NTU60 许可文本留档确认 | §3.1 |
| 终审-C | "license 受限时给脚本不给衍生数据"承诺的最终边界 | §3.1 |
| 终审-D | InterPet4D 人体被试伦理原句摘录留档 | §3.2 |
| 终审-E | LICENSE 选型 + 开源时机 + 仓库 URL | §5/§6-B |
| 终审-F | Generative AI 声明措辞与工具名披露颗粒度 | §3.3/F13 |
| （既有） | 标题去留（R2）/作者名单/C6 措辞收窄/Scholar 终审 | outline R2、C6 注记、L3 |

---

## 7. 未验证项与移交

- [ ] **NTU 三流融合数落地后的 NTU 列升级：不归本窗**（协调者派单，BOARD 01:20 协调者既定分工）；本报告所有引用数字均为当前在档版本，若 full 档预注册条款触发回改，本报告 §2.2 Highlights 与 Cover Letter 数字须同步刷新
- [ ] LaTeX 工程搭建（决策 A 定案后）：`docs/paper/latex/` 脚手架 + 五篇 md → tex 装配 + Highlights 独立文件 + 参考文献池（related-work 题录已 11 条全解决/3 条作者待补）
- [ ] Scholar/arXiv 首次性终审（L3 既定投稿前置）
- [ ] 页数实测：md→tex 装配后跑一次 20–35 页窗口校验（F2 风险监控点）
- [ ] round2/W40 结果落地后复核 E4 第二行表述（conclusion-limitations 待办既有项）
- [ ] 记忆 MCP 故障窗口（BOARD 01:20 登记）恢复后由协调者补写本窗关键结论

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-26 | W41 投稿打包预备五件套交付：格式核查矩阵（F1-F16，官方指南当日抓取）/Cover Letter+Highlights（字符数实测）/DA+Ethics 草稿（四处终审标注）/fig1-4 程序化终检（矢量零栅格+双编码实证）/代码发布包清单（LICENSE 缺口上报）；决策 A/B 呈报待用户 |
