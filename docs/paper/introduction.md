# Abstract + 1. Introduction（英文占位初稿 · P0.6 增量二）

> Owner: `docs/paper/introduction.md` · W5 窗口 2026-08-23 · 状态: **终稿候选 v0.3（2026-08-25 W36 合并定稿）**——W32 四处 C7 换轨 DRAFT 注释块已复核合并并删除注释本体；[RESULT-1/2/3] 已按归档数字终填（来源见文末对账节）；[PENDING P0.2] 已填；Para 6 三占位符按三层口径重写为诚实版
> 写作规范: 五句摘要公式（Farquhar）+ Intro 六段式（hook→难点→缺口→方案→贡献→预览）；句均长 ≤25 词；无套话开头。

---

## Abstract（占位初稿 ~200 词）

Behavior recognition in animals underpins welfare monitoring and working-dog training, yet behavior annotation is scarce and evaluation criteria evolve with operational needs. Existing recognition pipelines couple representation learning to a fixed label set, so every taxonomy change forces re-annotation and retraining. We propose PSD, a physics–semantics decoupled framework that separates *how skeletons move* from *what behaviors are called*. A label-free physics layer combines self-supervised pretraining with unsupervised motion-word segmentation; a lightweight semantic layer expands rule-engine seeds into full taxonomy coverage through anchor-guided prototype clustering with iterated confidence-filtered pseudo-labeling, consolidated by semi-supervised self-training and made usable at small annotation budgets by warm-started semantic-layer initialization. On public real animal-skeleton data, pretraining alone yields kNN top-1 of **20.89% versus an 8.33% random baseline (2.51×)**; the warm-started semantic layer reaches **82.0% top-1 on 22 classes from only 20 labeled clips** (synthetic-offset benchmark); and a taxonomy transition is absorbed at **≥3× lower retraining cost** than full-pipeline retraining (conservative bound, synthetic-tier benchmark). These results indicate that decoupling turns evolving evaluation criteria from a re-annotation burden into a routine semantic-layer update.

> [W36 合并记录] 原 DRAFT-W32-C7-pivot 注释块①②已复核合并：[RESULT-2] 终填 82.0%@20 clips（`reports/p05-al-efficiency-warmstart-short-2026-08-25.json` curves.*.20.mean=0.8202，22 类、合成偏移层 noise_std=0.10、3 seeds mean±4.3pp）；[RESULT-3] 按用户裁决候选 C 定稿（实测均值 7.32×，措辞保守 ≥3×，锚点 `reports/c1-decouple-cost-2026-08-24.json` aggregated + full 档 `reports/c1-decouple-cost-full-2026-08-25.json` 6.07× 背书）；候选 B（+10.69pp 配对口径）/A（SMQ 1.53×）否决留痕见 BOARD 与记忆库。预注册条款照旧：full 档趋势若矛盾须回改。

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

> [W36 合并记录] 原 DRAFT-W32-C7-pivot 注释块（Para 4 尾句换轨）已复核：warm-start 表述保留，未引用冷启动数字，规避 W23 meta.comparability 跨分布不可比坑。

### Para 5 — 贡献 bullets（×4，每条 ≤2 行）

Our contributions are fourfold:

1. **A physics–semantics decoupled framework** that formalizes evolving evaluation criteria as taxonomy transitions and absorbs them through semantic-layer updates alone.
2. **The first transfer of image-domain anchor–cluster–pseudo-labeling to temporal-skeleton recognition**, supported by a systematic repository-scale survey with zero occupancy (Supplementary Material); arXiv/Scholar re-verification is scheduled before submission.
3. **Label-free validation on public real animal-skeleton data**: self-supervised pretraining reaches 20.89% kNN top-1 against an 8.33% random baseline, and motion-word segmentation cuts continuous streams into behavior-aligned proposals (boundary IoU **0.458 ± 0.049** versus an ≈0.30 equal-segment-count random-cut null under a seeds pseudo-ground-truth protocol).
4. **A complete low-resource pipeline evaluated under a three-caliber protocol** (synthetic / public-real / real-K9); warm-started semantic-layer initialization makes a 20-clip budget usable (**82.0% top-1 on the synthetic-offset tier**; cold-start protocols remain near chance at this budget under their respective distributions), while uncertainty-based sampling is reported as an exploratory negative finding in Section 5.

> [W36 合并记录] 原 DRAFT-W32-C7-pivot 注释块（bullet 4 换轨）已复核合并：82.0% 保留、7.8% 冷启动直接数字对改为定性限定句（遵守 W23 comparability 护栏——两轮分布不同禁止同协议并列）；[PENDING P0.4-P0.5] 效率主张随裁决①移除；95.7% 天花板（归属均匀扩展臂）不入正文贡献句，防臂属错配。

### Para 6 — 结果预览与文章组织

Across three data calibers, PSD reaches 96.6% top-1 on the synthetic 22-class benchmark and 82.0% from 20 labeled clips under distribution shift (synthetic-offset tier), attains 44.9% on the four-class public-real subset (1.80× random; severe class imbalance disclosed in Section 6), and absorbs a taxonomy transition at ≥3× lower wall-clock retraining cost while matching full-retraining accuracy within statistical noise. Figure 1 overviews the framework. Section 2 reviews related work. Section 3 details the method. Sections 4–5 report experiments and ablations, and Section 6 concludes with limitations.

> [W36 终填记录] 原 [RESULT-4/5/6] 三占位符按三层口径诚实重写（禁止单数字跨层混排，AGENTS 硬规则 3）：
> - 合成层 96.6% = `reports/p05-stgcnbc-synthetic-100perclass-Y.json` summary/best_val_acc=0.9659（等预算 50ep 协议，引用须注明）；
> - 合成偏移层 82.0%@20 = warmstart JSON curves.*.20.mean；
> - 公开真实层 44.9%（4 类部分口径）= `reports/p05-public-real-partialclass-result-2026-08-25.json` best_val_acc=0.4490，1.80×随机（4 类随机基线 25%），per-class watch 100%/track 23.5%/jump+stay 0 类不平衡如实披露并前送 §6 Limitations L7；
> - 成本句 = 候选 C 裁决定稿（≥3× 保守界，full 6.07× 背书；精度统计等效 −0.91pp/+2.27pp）。

---

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 无泛化开场（首句即具体数字/痛点） | ✅ $12k + 淘汰率 |
| 贡献前置、Methods 从 §3 开始（页面预算内） | ✅ |
| 首次性主张带边界声明 | ✅ 贡献点 2 显式标注 |
| 数字纪律 | ✅ 全部数字已归档溯源：20.89%/2.51×（p01-knn-result.json）、0.458±0.049（p02-smq-iou-eC-seeds-recheck.json）、82.0%@20（p05-al-efficiency-warmstart-short-2026-08-25.json）、96.6%（p05-stgcnbc-synthetic-100perclass-Y.json 等预算协议）、44.9%/1.80×（p05-public-real-partialclass-result-2026-08-25.json）、≥3× 保守界（c1-decouple-cost 两档 JSON）；效率主张零残留 |
| 口径纪律 | ✅ 三层口径逐句标注；warm-start 与冷启动不并列同协议数字（comparability 护栏）；44.9% 明示 4 类部分口径 + 类不平衡前送 L7 |
| C7 换轨合规 | ✅ 四处 DRAFT 注释块已合并删除；无"≥85% within budget"类禁用表述；AL 仅以探索性发现出现 |
| 与 outline 故事线一致 | ✅ 六段一一对应 Narrative Arc 五拍；outline 矩阵已由 W36 同步刷新（v0.8） |
| 引用纪律 | ✅ Science 特稿引用已补全（Grimm, Science feature；新闻特稿措辞，W17 终审） |

## 待办
- [x] ~~P0 数据落地后：回填 RESULT-2~6~~ ✅ W36 终填完成（2026-08-25）
- [ ] Science 特稿正式题录复核（文献终审窗口保留项）
- [x] ~~终稿窗口合并四处 `DRAFT-W32-C7-pivot` 注释块~~ ✅ 已合并并删除注释本体，合并记录改为行内引注块（2026-08-25 W36）
- [ ] 投稿前 Scholar 终审时复核 96.6% 引用协议标注（等预算 50ep vs 早停两种口径并存，见 number-index E1 note）

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 增量二：Abstract 占位稿 + Introduction 六段式初稿 |
| v0.2-draft | 2026-08-25 | W32 C7 换轨预改写（依据 ADR-0006 / experiment-skeleton v0.6）：四处 C7 相关句段替换 + DRAFT 注释锚点——效率主张移除、warm-start 可用性上桌、[RESULT-3] 按 C1 候选预填并附 B 候选备选句；领地边界：仅 C7 相关句段（自助收编 -Permit 特批） |
| v0.2.1-draft | 2026-08-25 | 用户裁决 [RESULT-3] 选候选 C（解耦墙钟 ≥3× 保守界）落定：Abstract 注释块改为裁决已定态（B/A 备选否决留痕），终稿窗口零决策合并 |
| v0.3 | 2026-08-25 | **W36 终稿合并**：四处 DRAFT 注释块复核合并删除；[RESULT-1] token 移除（数字已在文）、[RESULT-2]=82.0%@20 clips（合成偏移层口径随句）、[RESULT-3] 定稿 ≥3× 保守界；bullet 3 [PENDING P0.2]→SMQ IoU 0.458±0.049 vs ≈0.30 null（W34 勘误口径：非滑窗基线）；bullet 4 冷启动直接对改定性限定句（comparability 护栏）；Para 6 RESULT-4/5/6 重写为三层口径诚实版（96.6% 合成 / 82.0% 合成偏移 / 44.9% 公开真实 4 类 + 类不平衡前送）；自审记录与待办同步 |
