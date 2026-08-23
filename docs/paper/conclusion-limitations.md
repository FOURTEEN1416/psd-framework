# 6. Conclusion and Limitations（英文骨架 · P0.6 增量二）

> Owner: `docs/paper/conclusion-limitations.md` · W5 窗口 2026-08-23 · 状态: 骨架 v0.1——结论句式与 Limitations 清单已成文，数字 `[RESULT-x]` 占位
> 写作规范: galaxy 诚实原则——Limitations 独立成段、先发制人；结论不超出已测范围。

---

## 6.1 Conclusion（模板段，数字占位）

We presented PSD, a physics–semantics decoupled framework for low-resource animal behavior recognition. PSD separates how skeletons move from what behaviors are called: a label-free physics layer provides self-supervised dynamics representations and unsupervised behavior proposals, while a lightweight semantic layer grows rule-engine seeds into full taxonomy coverage through anchor-guided clustering, iterated pseudo-labeling, and active learning. Experiments across three data calibers showed that pretraining alone yields discriminable dynamics representations (**2.51×** random baseline), that the full pipeline reaches **[RESULT-7]** on 22-class recognition within a 100–200 clip budget, and that taxonomy transitions are absorbed at **[RESULT-8]** of the full-retraining cost at matched accuracy. Beyond animal behavior, the decoupling pattern—frozen physics, revisable semantics—applies to any recognition task whose evaluation criteria evolve with operational practice.

## 6.2 Limitations（独立成段，先发制人）

Four limitations bound our claims.

1. **Proxy-caliber evidence for representation quality.** Our pretraining evidence uses subject identity as a kNN probe on InterPet4D, which measures representation discriminability rather than behavior accuracy; behavior-level claims rest on downstream experiments (Sections 4–5) [PENDING].
2. **Single species family.** All real-data validation covers canids; cross-family generalization (felids, equids, primates) is untested and left to future work.
3. **Survey boundary on firstness.** Our novelty claim rests on repository-scale search with zero occupancy; arXiv/Google Scholar re-verification was scheduled before submission, and we cannot exclude unpublished or non-indexed concurrent work.
4. **Scale of compute.** All experiments ran on a single consumer GPU (RTX 5060 Laptop 8GB); scaling behavior to multi-GPU pretraining and web-scale unlabeled pools is unverified.

> 每条 limitation 均配"为何不动摇核心主张"的回应策略（rebuttal 预案）：
> 1 → 口径披露在 §3.2.1/§4.3 双处声明，行为级证据由 E2-E6 承担；
> 2 → 物理层按物种族训练（trained once per species family），架构本身跨族可复用；
> 3 → 首次性主张措辞已带边界（"to the best of our knowledge"），非绝对断言；
> 4 → 消费级算力可行性恰是低资源叙事的佐证，非缺陷。

## 自审记录

| 检查项 | 结果 |
|--------|------|
| 结论不超出已测范围 | ✅ 未落地数字全部占位 |
| Limitations 独立成段 + 每条有回应预案 | ✅ 4 条 |
| 与风险登记册一致 | ✅ L1↔R1、L3↔W1 边界声明、L4↔HANDOVER §4 |

## 待办
- [ ] RESULT-7/8 回填后收口终稿

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 增量二：结论模板 + 四条 Limitations 及 rebuttal 预案 |
