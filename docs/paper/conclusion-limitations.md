# 6. Conclusion and Limitations（英文骨架 · P0.6 增量二）

> Owner: `docs/paper/conclusion-limitations.md` · W5 窗口 2026-08-23 · 状态: **终稿候选 v0.4（2026-08-25 W36）**——[RESULT-7/8] 已按归档数字终填；Limitations 扩至八条：v0.2 六条 + W36 增 L7（公开真实层 44.90% 类不平衡）与 L8（真实域骨架有效监督 20/24），L5 升级为 AL 双轮负结果（W14 冷启动 + W23 强打分器）
> 写作规范: galaxy 诚实原则——Limitations 独立成段、先发制人；结论不超出已测范围。

---

## 6.1 Conclusion（终稿段）

We presented PSD, a physics–semantics decoupled framework for low-resource animal behavior recognition. PSD separates how skeletons move from what behaviors are called: a label-free physics layer provides self-supervised dynamics representations and unsupervised behavior proposals, while a lightweight semantic layer grows rule-engine seeds into full taxonomy coverage through anchor-guided clustering, iterated pseudo-labeling, and warm-started small-budget initialization. Experiments across three data calibers showed that pretraining alone yields discriminable dynamics representations (**2.51×** random baseline), that the semantic layer attains **82.0% top-1 on 22 classes from a 20-clip budget via warm-started initialization** (synthetic-offset tier; ±4.3 across three seeds), and that taxonomy transitions are absorbed at **≥3× lower wall-clock retraining cost** than full-pipeline retraining at matched accuracy (conservative bound backed by a measured 6.07× on the full tier; accuracy statistically equivalent at −0.91 pp full / +2.27 pp small, both <2.3 pp). Beyond animal behavior, the decoupling pattern—frozen physics, revisable semantics—applies to any recognition task whose evaluation criteria evolve with operational practice.

> [W36 终填记录] [RESULT-7] = `reports/p05-al-efficiency-warmstart-short-2026-08-25.json` curves.*.20.mean=0.8202（22 类、合成偏移层 noise_std=0.10、验证集 n=330、3 seeds）；[RESULT-8] = `reports/c1-decouple-cost-full-2026-08-25.json`（full 6.07×、同 seed 配对最小 4.00×）+ `reports/c1-decouple-cost-2026-08-24.json`（small 7.32×）→ 措辞保守 ≥3×，精度措辞按 W34 校准为统计等效。语义层组件句同步 C7 换轨：active learning → warm-started small-budget initialization。

## 6.2 Limitations（独立成段，先发制人）

Eight limitations bound our claims.

1. **Proxy-caliber evidence for representation quality.** Our pretraining evidence uses subject identity as a kNN probe on InterPet4D, which measures representation discriminability rather than behavior accuracy; behavior-level evidence now rests on the downstream experiments of Sections 4–5 (segmentation IoU 0.458±0.049 under a seeds pseudo-ground-truth protocol, pseudo-label pool precision 0.691±0.013, and the four-class public-real fine-tuning result below), each with its caliber disclosed inline.
2. **Single species family.** All real-data validation covers canids; cross-family generalization (felids, equids, primates) is untested and left to future work.
3. **Survey boundary on firstness.** Our novelty claim rests on repository-scale search with zero occupancy; arXiv/Google Scholar re-verification was scheduled before submission, and we cannot exclude unpublished or non-indexed concurrent work.
4. **Scale of compute.** All experiments ran on a single consumer GPU (RTX 5060 Laptop 8GB); scaling behavior to multi-GPU pretraining and web-scale unlabeled pools is unverified.
5. **No advantage of uncertainty sampling—confirmed across three independent runs.** In our short-budget active-learning studies (synthetic tier), entropy sampling did not outperform random selection under either weak or strong scorers: with cold-start scorers at the short schedule, random led by 7.9 pp at budget 100 and 7.1 pp at budget 200 (all three seeds agreeing); an extended-schedule cold-start rerun reproduces the direction (random ahead by 12.3 pp at budget 100 and 7.8 pp at budget 200); with warm-started in-domain fine-tuned scorers, random again led by 4.2–5.0 pp across all three budget points (three seeds each). The synthetic-trained checkpoint additionally scored the public-real candidate pool with fully saturated softmax (mean top1−top2 logit margin 100.9 vs ≈10.8 within the synthetic validation domain), degrading uncertainty signals to numerical zero (`reports/w14-p05-al-efficiency-2026-08-24.md`, `reports/w23-p05-al-warmstart-2026-08-25.md`, `reports/p05-al-efficiency-full-2026-08-25.json`). We therefore report uncertainty sampling strictly as an exploratory negative finding and make no sampling-efficiency claims.
6. **Structural constraints of the public-real tier.** Under the pre-registered partial-class protocol, only 4 of 12 target classes pass the canine sample-size gate under the relaxed rule (3 under the strict rule; down/stand/scale have zero coverage), and the official pose-estimation subset averages ≈4.6 frames per video with no canine-overlapping video reaching the T=30 temporal length required by skeleton pipelines; a self-extraction pipeline (YOLO11-pose fine-tuned on dog-pose → skeleton extraction → ST-GCN fine-tuning on the 4-class subset) was adopted as mitigation (`reports/p05-public-real-partialclass-2026-08-24.md`), and its first-round outcome is quantified in L7.
7. **Severe class imbalance dominates the public-real result.** The self-extraction round-one model reaches 44.9% overall validation accuracy on the four-class subset (1.80× the 25% random baseline) under heavily skewed support (train: watch 72 / track 46 / stay 27 / jump 27); per-class accuracies collapse to watch 100% / track 23.5% / jump 0% / stay 0% — majority-class recall accounts for essentially all of the aggregate score (`reports/p05-public-real-partialclass-result-2026-08-25.json`). We therefore present this number with its per-class breakdown attached, restrict all public-real claims to the four-class partial caliber, and treat aggregate-only reporting as prohibited for this tier.
8. **Real-domain skeletons carry 20/24 valid supervision channels.** The dog-pose annotations never label four of the 24 graph joints (both eyes, withers, throat — the last four slots are absent from the StanfordExtra source), so every real-domain keypoint product in this paper is topology-isomorphic rather than fully supervised: the four dead channels are hard-masked at assembly and NaN-guarded downstream. Any real-domain skeleton-based number therefore rests on 20-channel effective supervision, and rules that depend on the withers landmark are structurally degraded in the pixel domain.

> 每条 limitation 均配"为何不动摇核心主张"的回应策略（rebuttal 预案）：
> 1 → 口径披露在 §3.2.1/§4.3 双处声明，行为级证据由 E2-E6 承担；
> 2 → 物理层按物种族训练（trained once per species family），架构本身跨族可复用；
> 3 → 首次性主张措辞已带边界（"to the best of our knowledge"），非绝对断言；
> 4 → 消费级算力可行性恰是低资源叙事的佐证，非缺陷。
> 5 → 负结果按诚实原则先发制人如实呈现（C7 已按用户裁决①降级为探索性发现，2026-08-25；W23 第二轮强打分器复证后双轮负结果闭环，效率主张正文与摘要永久禁用）；不确定性采样的经典前提（较强打分器 + 域内校准）缺失反而强化渐进式标注叙事（先标注→校准打分器→再选样）；warm-start 正证据已按裁决 A 落地为 C7 新主张；
> 6 → 结构性约束显式披露而非隐藏；自提取管线沉淀为可复用资产；公开真实层相关主张全部限定在 4 类子集口径内，禁止升格；
> 7 → 44.9% 永远与逐类分解同框呈现（防聚合分数掩盖多数类偏置）；类不平衡处理是 §5 敏感性扫描的显式开关（frequency-aware margin vs 重采样），扩展池增强的第二轮验证（round2）已立项独立窗口，不与本口径混报；
> 8 → 死关节硬掩码是组装出口的确定性机制（非统计修补），20/24 口径在 §4.1 与本节双处披露；依赖 withers 的规则降级已在数据飞轮报告中量化归档。

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 结论不超出已测范围 | ✅ 全部数字已按归档报告终填（82.0%@20 / ≥3× 保守界 / 统计等效精度带），无占位残留 |
| Limitations 独立成段 + 每条有回应预案 | ✅ 8 条（v0.2 六条 + W36 增 L7/L8） |
| 与风险登记册一致 | ✅ L1↔R1、L3↔W1 边界声明、L4↔HANDOVER §4、L5↔R11/W14+W23+full 复跑三证据、L6↔W20 报告结构约束披露、L7↔Q3c JSON 类不平衡如实、L8↔死关节 ADR/20-24 口径块 |
| 三层口径 | ✅ 82.0% 标合成偏移层；44.9% 标公开真实层 4 类部分口径；成本数字标合成层基准——零混报 |

## 待办
- [x] C7 措辞用户裁决落地并同步本文件（2026-08-25，选项①：降级探索性发现）
- [x] ~~RESULT-7/8 回填后收口终稿~~ ✅ W36 终填（2026-08-25）：RESULT-7=warm-start 82.0%@20 clips（合成偏移层）、RESULT-8=≥3× 保守墙钟成本界（full 6.07× 背书 + 精度统计等效）
- [ ] round2 扩展池增强结果落地后，评估 L7 是否增补"飞轮效力"实证句（独立窗口执行，本窗不动其数据）

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 增量二：结论模板 + 四条 Limitations 及 rebuttal 预案 |
| v0.2 | 2026-08-24 | W21 诚实刷新：新增 L5（冷启动弱打分器场景不确定性采样无优势 + softmax 跨域饱和诊断 100.9 vs ≈10.8）与 L6（AK 公开真实层结构约束：宽松门禁 4/12 类 + PE ≈4.6 帧/视频，自提取管线为缓解方案待 Q3 接力）；rebuttal 预案同步扩至 6 条 |
| v0.3 | 2026-08-25 | 用户裁决 C7 选项①落地：结论模板删除"100–200 片段预算内达标"句式（RESULT-7 改指语义层精度）；L5 rebuttal 更新为裁决①表述并注明升级通道 |
| v0.4 | 2026-08-25 | **W36 终稿轮**：[RESULT-7] 终填 warm-start 82.0%@20 clips（±4.3，合成偏移层口径随句）；[RESULT-8] 终填 ≥3× 保守墙钟成本界（实测 full 6.07×/small 7.32× 背书 + 精度统计等效 −0.91/+2.27pp）；语义层组件句随 C7 换轨改 warm-started initialization；L1 去除 [PENDING] 改指向 §4-§5 已落地证据；**L5 升级三重负结果证据链**（短档冷启动 + extended 复跑 + W23 强打分器，random 全线反超）；**新增 L7**（公开真实层 44.9%=1.80×随机但 watch 100%/track 23.5%/jump+stay 0 类不平衡主导聚合分数，逐类分解强制同框）；**新增 L8**（真实域骨架拓扑同构有效监督 20/24 死关节硬掩码口径）；rebuttal 预案扩至 8 条 |
