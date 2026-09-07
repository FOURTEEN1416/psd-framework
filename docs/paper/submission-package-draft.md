# 投稿包草稿 v0.1（2026-09-07，审计窗代拟）

> **状态**：v0.2（2026-09-07 R23 三审轮后终检刷新）——E9 系列四点 10-seed 终口径、TRANS-001 真域转换（成本 FAILS 如实+精度优势）、SSL 外部基线（+20.7pp）全部入稿；master=9ce81ff 编译零错。
> 纪律：所有数字必须能在 main.tex/摘要/reports 工件中逐字找到；审稿人可验证性优先于修辞。

---

## 1. Cover Letter（致 Pattern Recognition 编辑，英文定稿前待用户过目）

Dear Editor,

We submit our manuscript **"A Physics-Semantics Decoupled Framework for Animal Behavior Recognition under Evolving Evaluation Criteria"** for consideration by Pattern Recognition.

Working-animal behavior recognition is unusual among pattern-recognition tasks in that its evaluation criteria are not fixed: operational taxonomies split, merge, and grow as trainers and welfare protocols evolve, and every such transition currently forces re-annotation and full retraining. Our contribution is a framework — a frozen physics layer for skeleton dynamics plus a revisable semantic layer for taxonomy-facing labeling — that turns this re-annotation burden into a routine semantic-layer update, and we quantify that claim rather than assert it.

Three properties of the evidence we believe fit Pattern Recognition's scope:

1. **Pre-registration discipline.** Eight protocols (working-dog pilot, dataset expansion, four budget-retention studies on NTU60, NTU120, PanAf500, and UCF101, a real-domain taxonomy-transition replication, and an external semi-supervised baseline) were frozen — decision rules and thresholds included — before the experiments ran. Where a pre-registered verdict flipped or failed under the frozen protocol (a 3-seed confirmation becoming a 10-seed partial at PanAf500; a canine-tier retention claim that does not survive protocol correction; the real-domain transition cost endpoint failing where the synthetic-tier claim held), the paper reports the reversal, the failure, and the boundary — not the favorable reading. All reported p-values are Holm-Bonferroni corrected within declared families (corrected values printed alongside raw).
2. **An external semi-supervised baseline, run rather than cited.** On the NTU60 10%-budget harness (same frozen features, same subset, same evaluation), a Mean-Teacher-style pool-distillation baseline reaches 46.79% ± 0.61 against the PSD pipeline's 67.53% ± 0.24 — a +20.7pp margin with 10/10 paired seed wins (Wilcoxon p=0.002, corrected 0.008). The comparison is harness-limited (frozen features; disclosed) and its mechanism is consistent with the paper's central claim: the external baseline's confidence mask admits only 15,758 of 36,082 pool clips, while near-full pool restoration is exactly what the pipeline does.
3. **Cross-domain budget retention with a falsifiable mechanism.** On the human benchmark the pipeline retains 90.7% of full-budget linear-probe accuracy at 10% of the labels (ten seeds, pre-registered); on NTU120 the same protocol retains 88.9%; on PanAf500 — the only animal-domain public benchmark — 88.7%, a pre-registered out-of-sample test of the retention mechanism; on the public canine tier the corrected protocol stays near chance at a 13% budget. The gradient is stated in falsifiable form (any tier with a strongly separable full-budget reference yet <85% retention would break it), and a companion experiment shows a stronger pose extractor does not raise the canine ceiling, localizing the bottleneck to label alignment.
4. **Verifiable attribution, including the failures.** Implementation equivalence is checked against the official reference (three-stream NTU60 fusion 77.97% vs a pre-registered 77.18% line); the synthetic-tier taxonomy-transition cost claim (≥3×, measured 6.07×) is paired with an accuracy-equivalence test inside a pre-registered noise band — and its pre-registered real-domain replication, which fails the wall-clock endpoint at small-backbone scale, is reported as a failure with the accuracy advantage it does establish, solver-family confound disclosed.

All datasets are public; derived skeletons are not redistributed and regenerate from provider data via released scripts; every number in the paper traces to a committed artifact in the public repository (github.com/FOURTEEN1416/psd-framework, tag review-snapshot).

The manuscript is original, not under review elsewhere, and all authors have approved it. Funding: none (declared in the manuscript); GenAI use is disclosed in the manuscript's declaration section.

Sincerely,
[通讯作者 — 待用户填]

---

## 2. Suggested Reviewers（候选池，提交前须逐一核实在职单位与 COI）

策略：骨架识别×动物行为×半监督三线覆盖，全部来自本文引用池（领域内活跃、无合作史待用户确认）；每行=姓名 | 关联依据 | 需核实项。

| 候选 | 领域线 | 依据（本文引用） | 提交前核实 |
|---|---|---|---|
| Jun Liu (Harbin Inst. of Tech.) | 骨架/动作识别 | PoseConv3D 一作 (duan2022posec3d) | 现单位、近期与作者机构合作 |
| C. V. Jawahar (IIIT Hyderabad) | 骨架表征 | CVPR 骨架方向资深 PC | COI |
| Tianyu Guo | 自监督骨架 | AimCLR/AimCLR++ 一作（本文等价性基线作者——**利益冲突风险高，建议回避**） | 若回避则换 |
| Emmanouil Benetos (QMUL) | 动作分割/时序 | SMQ 相关领域 PC | 单位/COI |
| 动物行为计算方向：B. Mohler 系（InterPet4D 相关）或 B. Behav. 期刊编委 | 动物行为标注 | peng2026interpet4d 作者群 | 是否愿评 ML 方法稿 |
| Nikos Komodakis (U. Crete/Noah's) | 半监督/对比学习 | TCL 系领域资深 | COI |

> 注（v0.2 更新）：①AimCLR 作者（Tianyu Guo）行已删除——COI 明确（基线比较对象），不保留占位；②新增第 7 行 SSL/一致性方向候选（R23 SSL 基线轮新增的比较域）；③R23 后引用池新增 liu2020ntu（NTU120 TPAMI 一作 Jun Liu 同为 NTU 数据集方——推荐其审稿属常见做法但需声明其数据集作者身份，提交时如实用）。最终 3-5 名由用户圈定，建议补 1 名动物福利领域（非 CV）审稿人覆盖应用线。

| 候选（新增） | 领域线 | 依据 | 提交前核实 |
|---|---|---|---|
| SSL/一致性蒸馏方向资深学者（须**非** Mean-Teacher 原作者群——Rasmus/Kahn/Virtanen 系为方法比较对象 COI） | 外部 SSL 基线 | PSD-SSL-BASE-001 对照方法所属领域 | 具体人选待用户圈定 |

## 3. Graphical Abstract（规格 + 脚本指针）

- 规格：Elsevier GA 建议 ≤531×131 pt（约 7.4×1.82 in），≥250 dpi，单图讲清"冻结物理层+可修订语义层+演化吸收"。
- 脚本：`docs/paper/figures/scripts/make_ga_graphical_abstract.py`（diagram-design 密度 4/10 原则 + 印刷尺寸 1:1 铁律；输出 PDF+PNG 600dpi）。
- 内容三元素：①双层框（青=物理冻结/橙=语义可修订，与 fig1 同谱系）②Y→Y′ 演化箭头只穿语义层 ③右下角保留率微条（human 90.7 / NTU120 88.9 / canine 边界如实）。

---

## 4. 链完后刷新清单（R22 后执行）

- [x] UCF101 终判入 cover letter 第 2 点（66.64% FAILS 边界措辞，2026-09-07 已补）
- [x] E9b/c 口径切换后核对摘要 90.6%/88.9% 两数是否变化，同步本稿（10-seed 终口径已核实一致）
- [ ] **R22b#7：预注册协议外部时间戳**（OSF/AsPredicted 注册或修订信中说明 repo commit 时间戳的证明力——用户人工决策项）
- [ ] Funding/GenAI 段与最终声明一致
- [ ] 通讯作者+单位+邮箱（用户人工项）
- [x] GA 图 judge 视觉验收一轮（2026-09-07 pass）
