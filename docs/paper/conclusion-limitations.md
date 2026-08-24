# 6. Conclusion and Limitations（英文骨架 · P0.6 增量二）

> Owner: `docs/paper/conclusion-limitations.md` · W5 窗口 2026-08-23 · 状态: 骨架 v0.2——结论句式与 Limitations 清单已成文，数字 `[RESULT-x]` 占位；v0.2 增 W21 两条诚实发现（L5 AL 负结果 / L6 公开真实层结构约束）
> 写作规范: galaxy 诚实原则——Limitations 独立成段、先发制人；结论不超出已测范围。

---

## 6.1 Conclusion（模板段，数字占位）

We presented PSD, a physics–semantics decoupled framework for low-resource animal behavior recognition. PSD separates how skeletons move from what behaviors are called: a label-free physics layer provides self-supervised dynamics representations and unsupervised behavior proposals, while a lightweight semantic layer grows rule-engine seeds into full taxonomy coverage through anchor-guided clustering, iterated pseudo-labeling, and active learning. Experiments across three data calibers showed that pretraining alone yields discriminable dynamics representations (**2.51×** random baseline), that the full pipeline reaches **[RESULT-7]** on 22-class recognition within a 100–200 clip budget, and that taxonomy transitions are absorbed at **[RESULT-8]** of the full-retraining cost at matched accuracy. Beyond animal behavior, the decoupling pattern—frozen physics, revisable semantics—applies to any recognition task whose evaluation criteria evolve with operational practice.

## 6.2 Limitations（独立成段，先发制人）

Six limitations bound our claims.

1. **Proxy-caliber evidence for representation quality.** Our pretraining evidence uses subject identity as a kNN probe on InterPet4D, which measures representation discriminability rather than behavior accuracy; behavior-level claims rest on downstream experiments (Sections 4–5) [PENDING].
2. **Single species family.** All real-data validation covers canids; cross-family generalization (felids, equids, primates) is untested and left to future work.
3. **Survey boundary on firstness.** Our novelty claim rests on repository-scale search with zero occupancy; arXiv/Google Scholar re-verification was scheduled before submission, and we cannot exclude unpublished or non-indexed concurrent work.
4. **Scale of compute.** All experiments ran on a single consumer GPU (RTX 5060 Laptop 8GB); scaling behavior to multi-GPU pretraining and web-scale unlabeled pools is unverified.
5. **No advantage of uncertainty sampling under cold-start weak scorers.** In our short-budget active-learning study (synthetic tier), entropy sampling did not outperform random selection—random led by 7.9pp at budget 100 and 7.1pp at budget 200, with all three seeds agreeing—and the synthetic-trained checkpoint scored the public-real candidate pool with fully saturated softmax (mean top1−top2 logit margin 100.9 vs ≈10.8 within the synthetic validation domain), degrading uncertainty signals to numerical zero; full-budget confirmation remains queued on the GPU (`reports/w14-p05-al-efficiency-2026-08-24.md`).
6. **Structural constraints of the public-real tier.** Under the pre-registered partial-class protocol, only 4 of 12 target classes pass the canine sample-size gate under the relaxed rule (3 under the strict rule; down/stand/scale have zero coverage), and the official pose-estimation subset averages ≈4.6 frames per video with no canine-overlapping video reaching the T=30 temporal length required by skeleton pipelines; a self-extraction pipeline (YOLO11-pose fine-tuned on dog-pose → skeleton extraction → ST-GCN fine-tuning on the 4-class subset) was adopted as mitigation, with results deferred to the Q3 relay queue (`reports/p05-public-real-partialclass-2026-08-24.md`).

> 每条 limitation 均配"为何不动摇核心主张"的回应策略（rebuttal 预案）：
> 1 → 口径披露在 §3.2.1/§4.3 双处声明，行为级证据由 E2-E6 承担；
> 2 → 物理层按物种族训练（trained once per species family），架构本身跨族可复用；
> 3 → 首次性主张措辞已带边界（"to the best of our knowledge"），非绝对断言；
> 4 → 消费级算力可行性恰是低资源叙事的佐证，非缺陷。
> 5 → 负结果按诚实原则先发制人如实呈现；不确定性采样的经典前提（较强打分器 + 域内校准）缺失反而强化 100–200 片段预算叙事（先标注→校准打分器→再选样）；C7 最终措辞待用户裁决，warm-start 协议变更属预注册修正另开窗口；
> 6 → 结构性约束显式披露而非隐藏；自提取管线沉淀为可复用资产；公开真实层相关主张全部限定在 4 类子集口径内，禁止升格。

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 结论不超出已测范围 | ✅ 未落地数字全部占位 |
| Limitations 独立成段 + 每条有回应预案 | ✅ 6 条（v0.2 增 L5/L6） |
| 与风险登记册一致 | ✅ L1↔R1、L3↔W1 边界声明、L4↔HANDOVER §4、L5↔R11/W14 报告、L6↔W20 报告结构约束披露 |

## 待办
- [ ] RESULT-7/8 回填后收口终稿（RESULT-7 语义随 C7 用户裁决联动）
- [ ] C7 措辞用户裁决落地后同步本文件结论段与 rebuttal 预案

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 增量二：结论模板 + 四条 Limitations 及 rebuttal 预案 |
| v0.2 | 2026-08-24 | W21 诚实刷新：新增 L5（冷启动弱打分器场景不确定性采样无优势 + softmax 跨域饱和诊断 100.9 vs ≈10.8）与 L6（AK 公开真实层结构约束：宽松门禁 4/12 类 + PE ≈4.6 帧/视频，自提取管线为缓解方案待 Q3 接力）；rebuttal 预案同步扩至 6 条 |
