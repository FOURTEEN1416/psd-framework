# Abstract + 1. Introduction（英文占位初稿 · P0.6 增量二）

> Owner: `docs/paper/introduction.md` · W5 窗口 2026-08-23 · 状态: 初稿 v0.1——**结构成文，实验数字以 `[RESULT-x]` 占位；终稿等 P0.2-P0.5 数据回填**（交接文档"Introduction 终稿不做"边界不破）
> 写作规范: 五句摘要公式（Farquhar）+ Intro 六段式（hook→难点→缺口→方案→贡献→预览）；句均长 ≤25 词；无套话开头。

---

## Abstract（占位初稿 ~200 词）

Behavior recognition in animals underpins welfare monitoring and working-dog training, yet behavior annotation is scarce and evaluation criteria evolve with operational needs. Existing recognition pipelines couple representation learning to a fixed label set, so every taxonomy change forces re-annotation and retraining. We propose PSD, a physics–semantics decoupled framework that separates *how skeletons move* from *what behaviors are called*. A label-free physics layer combines self-supervised pretraining with unsupervised motion-word segmentation; a lightweight semantic layer expands rule-engine seeds into full taxonomy coverage through anchor-guided prototype clustering with iterated confidence-filtered pseudo-labeling, consolidated by semi-supervised self-training and uncertainty-based active learning. On public real animal-skeleton data, pretraining alone yields kNN top-1 of **20.89% versus an 8.33% random baseline (2.51×)** [RESULT-1]; the full pipeline reaches **[RESULT-2]** on 22-class behavior classification using only 100–200 annotated clips, and reduces taxonomy-transition cost by **[RESULT-3]** relative to full retraining. These results indicate that decoupling turns evolving evaluation criteria from a re-annotation burden into a routine semantic-layer update.

> 自检：四要素齐全（背景/方法/结果/结论）✅；数值结果 3 处（1 实 + 2 占位）✅；五句公式结构 ✅；无 "Recently... increasing attention" 类开头 ✅。

---

## 1. Introduction（初稿 ~650 词）

### Para 1 — Hook（产业痛点，禁泛化开场）

Training a working dog costs more than USD 12,000, and more than half of candidates fail to graduate [CITATION-NEEDED: Science feature]. Behind these numbers sits a measurement problem: organizations that breed and train dogs still rely on subjective human scoring of behavior. Automated behavior recognition would make training decisions objective and comparable across evaluators. The same need extends to livestock welfare monitoring, veterinary research, and pet behavior analysis.

### Para 2 — 为什么难（三重叠加困难）

Three difficulties make animal behavior recognition harder than its human counterpart. First, labeled data is scarce: behavior annotation requires expert time, and wild or working environments offer no laboratory control. Second, skeletons dominate the practical signal—keypoint tracks are compact, background-invariant, and transfer across individuals—but skeleton-based recognition methods have been developed almost exclusively for humans. Third, and least explored, evaluation criteria are not static. Working-dog programs redefine behavior categories as training standards evolve; welfare protocols split or merge classes as regulations change. A system trained on a fixed label set must be re-annotated and retrained after every such revision.

### Para 3 — 缺口（对应 Related Work 三小节，一句话每条）

Current literature leaves three gaps open (Section 2). Animal behavior pipelines are supervision-heavy and assume a fixed taxonomy (§2.1). Self-supervised skeleton learning is mature but human-centric, and pretraining is studied separately from temporal segmentation (§2.2). Semi-supervised methods provide pseudo-label machinery whose components—seed-annotated anchors, prototype clustering, iterated confidence-filtered labeling—have never been assembled for temporal-skeleton recognition (§2.3).

### Para 4 — 方案（PSD 一段话概述）

We propose PSD, a physics–semantics decoupled framework for low-resource animal behavior recognition. PSD factorizes recognition into two layers connected by a narrow interface. A physics layer Φ learns how skeletons move without any behavior labels: it combines extreme-asymmetry contrastive pretraining adapted to quadruped topology with motion-word quantization that segments continuous streams into behavior proposals. A semantic layer Ω learns what behaviors are called from a small seed budget: rule-engine coarse labels seed class anchors, prototype clustering assigns pseudo-labels under confidence filtering, and iterated expansion is verified by uncertainty-based active learning within a budget of 100–200 annotated clips. Because the layers interact only through embeddings and proposals, an evaluation-criteria transition replaces the taxonomy of Ω while Φ remains frozen.

### Para 5 — 贡献 bullets（×4，每条 ≤2 行）

Our contributions are fourfold:

1. **A physics–semantics decoupled framework** that formalizes evolving evaluation criteria as taxonomy transitions and absorbs them through semantic-layer updates alone.
2. **The first transfer of image-domain anchor–cluster–pseudo-labeling to temporal-skeleton recognition**, supported by a systematic repository-scale survey with zero occupancy (Supplementary Material); arXiv/Scholar re-verification is scheduled before submission.
3. **Label-free validation on public real animal-skeleton data**: self-supervised pretraining reaches 20.89% kNN top-1 against an 8.33% random baseline, and motion-word segmentation cuts continuous streams into behavior-aligned proposals [PENDING P0.2].
4. **A complete low-resource pipeline evaluated under a three-caliber protocol** (synthetic / public-real / real-K9), closing with active learning that reaches ≥85% on 22 classes within the annotation budget [PENDING P0.4-P0.5].

### Para 6 — 结果预览与文章组织

Across three data calibers, PSD reaches **[RESULT-4: 22 类主精度]** using [RESULT-5] of the full annotation budget, and absorbs a taxonomy transition at [RESULT-6] of the cost of full retraining while matching its accuracy within noise. Figure 1 overviews the framework. Section 2 reviews related work. Section 3 details the method. Sections 4–5 report experiments and ablations, and Section 6 concludes with limitations.

---

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 无泛化开场（首句即具体数字/痛点） | ✅ $12k + 淘汰率 |
| 贡献前置、Methods 从 §3 开始（页面预算内） | ✅ |
| 首次性主张带边界声明 | ✅ 贡献点 2 显式标注 |
| 数字纪律 | ✅ 仅 P0.1 已归档数字实写，其余 [PENDING]/[RESULT-x] 占位 |
| 与 outline 故事线一致 | ✅ 六段一一对应 Narrative Arc 五拍 |
| 引用纪律 | ✅ Science 特稿标 CITATION-NEEDED（新闻特稿措辞） |

## 待办
- [ ] P0 数据落地后：回填 RESULT-2~6，删除 [PENDING] 标记，升级为终稿候选
- [ ] Science 特稿正式题录补全（文献终审窗口）

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 增量二：Abstract 占位稿 + Introduction 六段式初稿 |
