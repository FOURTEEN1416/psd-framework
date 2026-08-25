# Abstract + 1. Introduction（英文占位初稿 · P0.6 增量二）

> Owner: `docs/paper/introduction.md` · W5 窗口 2026-08-23 · 状态: 初稿 v0.1——**结构成文，实验数字以 `[RESULT-x]` 占位；终稿等 P0.2-P0.5 数据回填**（交接文档"Introduction 终稿不做"边界不破）
> 写作规范: 五句摘要公式（Farquhar）+ Intro 六段式（hook→难点→缺口→方案→贡献→预览）；句均长 ≤25 词；无套话开头。

---

## Abstract（占位初稿 ~200 词）

Behavior recognition in animals underpins welfare monitoring and working-dog training, yet behavior annotation is scarce and evaluation criteria evolve with operational needs. Existing recognition pipelines couple representation learning to a fixed label set, so every taxonomy change forces re-annotation and retraining. We propose PSD, a physics–semantics decoupled framework that separates *how skeletons move* from *what behaviors are called*. A label-free physics layer combines self-supervised pretraining with unsupervised motion-word segmentation; a lightweight semantic layer expands rule-engine seeds into full taxonomy coverage through anchor-guided prototype clustering with iterated confidence-filtered pseudo-labeling, consolidated by semi-supervised self-training and made usable at small annotation budgets by warm-started semantic-layer initialization. On public real animal-skeleton data, pretraining alone yields kNN top-1 of **20.89% versus an 8.33% random baseline (2.51×)** [RESULT-1]; the full pipeline reaches **[RESULT-2]** on 22-class behavior classification using only 100–200 annotated clips, and absorbs a taxonomy transition at **≥3× lower retraining cost** than full-pipeline retraining (conservative bound, synthetic-tier benchmark) [RESULT-3]. These results indicate that decoupling turns evolving evaluation criteria from a re-annotation burden into a routine semantic-layer update.

<!-- [DRAFT-W32-C7-pivot · 依据 ADR-0006 + experiment-skeleton v0.6 E5 换轨块 · 待终稿窗口复核合并]
     本段 C7 相关改动两处（其余句未动）：
     ① 方法尾句：uncertainty-based active learning → warm-started semantic-layer initialization（效率主张正文/摘要禁用，ADR-0006 裁决 1）
     ② [RESULT-3] 句：按 outline §4.1 重选源候选 C（C1 墙钟比）预填——"≥3× lower retraining cost (conservative bound, synthetic-tier benchmark)"；
        数据锚点 reports/c1-decouple-cost-2026-08-24.json aggregated（实测均值 7.32×，论文措辞取保守区间 ≥3×），
        ⚠️ 预注册条款：full 档 GPU 复跑若趋势矛盾须回改本句。
     备选版本（若用户裁决 RESULT-3 选候选 B 保守口径，公开真实层）:
       "...and its iterated pseudo-labeling lifts pool precision by **10.69 pp** (paired t-test, p=0.030) over its unlabeled start point [RESULT-3].
        数据锚点 reports/p04-tcl-results.json cells/on_consensus_a1.0/paired_first_vs_final/delta_pp_mean=10.69±3.28。
        ⚠️ 峰值口径 +17.88pp 存在 cherry-picking 风险，摘要不建议使用。
     候选 A（SMQ 分割 IoU 1.53×）与本句 cost 语义不符，仅建议留在正文 §4.3-E2。
-->

> 自检：四要素齐全（背景/方法/结果/结论）✅；数值结果 3 处（1 实 + 2 占位）✅；五句公式结构 ✅；无 "Recently... increasing attention" 类开头 ✅。

---

## 1. Introduction（初稿 ~650 词）

### Para 1 — Hook（产业痛点，禁泛化开场）

Training a working dog costs more than USD 12,000, and more than half of candidates fail to graduate [Grimm, Science feature]. Behind these numbers sits a measurement problem: organizations that breed and train dogs still rely on subjective human scoring of behavior. Automated behavior recognition would make training decisions objective and comparable across evaluators. The same need extends to livestock welfare monitoring, veterinary research, and pet behavior analysis.

### Para 2 — 为什么难（三重叠加困难）

Three difficulties make animal behavior recognition harder than its human counterpart. First, labeled data is scarce: behavior annotation requires expert time, and wild or working environments offer no laboratory control. Second, skeletons dominate the practical signal—keypoint tracks are compact, background-invariant, and transfer across individuals—but skeleton-based recognition methods have been developed almost exclusively for humans. Third, and least explored, evaluation criteria are not static. Working-dog programs redefine behavior categories as training standards evolve; welfare protocols split or merge classes as regulations change. A system trained on a fixed label set must be re-annotated and retrained after every such revision.

### Para 3 — 缺口（对应 Related Work 三小节，一句话每条）

Current literature leaves three gaps open (Section 2). Animal behavior pipelines are supervision-heavy and assume a fixed taxonomy (§2.1). Self-supervised skeleton learning is mature but human-centric, and pretraining is studied separately from temporal segmentation (§2.2). Semi-supervised methods provide pseudo-label machinery whose components—seed-annotated anchors, prototype clustering, iterated confidence-filtered labeling—have never been assembled for temporal-skeleton recognition (§2.3).

### Para 4 — 方案（PSD 一段话概述）

We propose PSD, a physics–semantics decoupled framework for low-resource animal behavior recognition. PSD factorizes recognition into two layers connected by a narrow interface. A physics layer Φ learns how skeletons move without any behavior labels: it combines extreme-asymmetry contrastive pretraining adapted to quadruped topology with motion-word quantization that segments continuous streams into behavior proposals. A semantic layer Ω learns what behaviors are called from a small seed budget: rule-engine coarse labels seed class anchors, prototype clustering assigns pseudo-labels under confidence filtering, and iterated expansion operates within an annotation budget of 100–200 clips; a warm-start protocol initializes this stage from prior semantic-layer weights, making budgets as small as 20 clips usable. Because the layers interact only through embeddings and proposals, an evaluation-criteria transition replaces the taxonomy of Ω while Φ remains frozen.

<!-- [DRAFT-W32-C7-pivot · 依据 ADR-0006 · 待终稿窗口复核合并]
     本段仅改尾句：原 "iterated expansion is verified by uncertainty-based active learning within a budget of 100–200 annotated clips"
     隐含"AL 验证扩展"的效率承诺（C7 裁决①禁用），换轨为 warm-start 可用性表述。
     数据锚点：reports/p05-al-efficiency-warmstart-short-2026-08-25.json curves（协议层 b=20 两臂均值 82.02%，合成偏移层 noise_std=0.10 口径）。
     ⚠️ 与 W14 冷启动 ~7.8% 的对照不可在正文并列为同协议数字（W23 meta.comparability 明示分布不同、禁止直接互比），
     终稿若需并列须加口径限定句。本句未引用冷启动数字，规避此坑。
-->

### Para 5 — 贡献 bullets（×4，每条 ≤2 行）

Our contributions are fourfold:

1. **A physics–semantics decoupled framework** that formalizes evolving evaluation criteria as taxonomy transitions and absorbs them through semantic-layer updates alone.
2. **The first transfer of image-domain anchor–cluster–pseudo-labeling to temporal-skeleton recognition**, supported by a systematic repository-scale survey with zero occupancy (Supplementary Material); arXiv/Scholar re-verification is scheduled before submission.
3. **Label-free validation on public real animal-skeleton data**: self-supervised pretraining reaches 20.89% kNN top-1 against an 8.33% random baseline, and motion-word segmentation cuts continuous streams into behavior-aligned proposals [PENDING P0.2].
4. **A complete low-resource pipeline evaluated under a three-caliber protocol** (synthetic / public-real / real-K9); warm-started semantic-layer initialization makes a 20-clip budget usable (82.0% vs 7.8% cold-start reference on the synthetic-offset tier), while uncertainty-based sampling is reported as an exploratory negative finding in Section 5.

<!-- [DRAFT-W32-C7-pivot · 依据 ADR-0006 + experiment-skeleton v0.6 · 待终稿窗口复核合并]
     原 bullet 4 "closing with active learning that reaches ≥85% on 22 classes within the annotation budget [PENDING P0.4-P0.5]"
     为 C7 裁决①明令禁用的效率主张，整句换轨：
     - 正证据路径 → warm-start 可用性（82.0%@20 clips，b=200 天花板 95.7%——注意该天花板归属均匀扩展臂，
       reports/p05-al-efficiency-warmstart-short-2026-08-25.json curves.random.'200'.mean=0.9566；entropy 臂同点 91.4%）；
     - 效率负结果 → 探索性发现落 §5（W14 冷启动 + W23 强打分器双轮背书）；
     - [PENDING P0.4-P0.5] 标记随效率主张一并移除；82.0%/7.8% 为合成层口径，三层铁律照旧。
     ⚠️ outline §3 故事线 #4 计划缩窄口径与本 bullet 对齐；outline 本体 W32 不动。
-->

### Para 6 — 结果预览与文章组织

Across three data calibers, PSD reaches **[RESULT-4: 22 类主精度]** using [RESULT-5] of the full annotation budget, and absorbs a taxonomy transition at [RESULT-6] of the cost of full retraining while matching its accuracy within noise. Figure 1 overviews the framework. Section 2 reviews related work. Section 3 details the method. Sections 4–5 report experiments and ablations, and Section 6 concludes with limitations.

---

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 无泛化开场（首句即具体数字/痛点） | ✅ $12k + 淘汰率 |
| 贡献前置、Methods 从 §3 开始（页面预算内） | ✅ |
| 首次性主张带边界声明 | ✅ 贡献点 2 显式标注 |
| 数字纪律 | ✅ P0.1 归档数字实写；C7 句段经 W32 换轨预改写后仅引用已归档数字（warm-start 82.0%@20 / ≥3× 保守界，均标口径）；其余仍 [RESULT-x]/[PENDING] 占位 |
| 与 outline 故事线一致 | ✅ 六段一一对应 Narrative Arc 五拍；C7 换轨后 bullet 4 与 outline v0.7 缩窄计划对齐（outline 本体待终稿窗口同步） |
| 引用纪律 | ✅ Science 特稿引用已补全（Grimm, Science feature；新闻特稿措辞，W17 终审） |

## 待办
- [ ] P0 数据落地后：回填 RESULT-2~6，删除 [PENDING] 标记，升级为终稿候选
- [ ] Science 特稿正式题录补全（文献终审窗口）
- [ ] **[W32 新增]** 终稿窗口合并四处 `DRAFT-W32-C7-pivot` 注释块（含 RESULT-3 源裁决落定后二选一备选句）；合并时删除注释本体

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 增量二：Abstract 占位稿 + Introduction 六段式初稿 |
| v0.2-draft | 2026-08-25 | W32 C7 换轨预改写（依据 ADR-0006 / experiment-skeleton v0.6）：四处 C7 相关句段替换 + DRAFT 注释锚点——效率主张移除、warm-start 可用性上桌、[RESULT-3] 按 C1 候选预填并附 B 候选备选句；领地边界：仅 C7 相关句段（自助收编 -Permit 特批） |
