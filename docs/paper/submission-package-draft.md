# 投稿包草稿 v0.1（2026-09-07，审计窗代拟）

> **状态**：UCF101 10-seed 在飞、E9b/c 口径切换与 R22 审计未完成——本稿数字以**当前 36 页摘要**为准，链完后按最终版刷新一次（占位符已标注）。
> 纪律：所有数字必须能在 main.tex/摘要/reports 工件中逐字找到；审稿人可验证性优先于修辞。

---

## 1. Cover Letter（致 Pattern Recognition 编辑，英文定稿前待用户过目）

Dear Editor,

We submit our manuscript **"A Physics-Semantics Decoupled Framework for Animal Behavior Recognition under Evolving Evaluation Criteria"** for consideration by Pattern Recognition.

Working-animal behavior recognition is unusual among pattern-recognition tasks in that its evaluation criteria are not fixed: operational taxonomies split, merge, and grow as trainers and welfare protocols evolve, and every such transition currently forces re-annotation and full retraining. Our contribution is a framework — a frozen physics layer for skeleton dynamics plus a revisable semantic layer for taxonomy-facing labeling — that turns this re-annotation burden into a routine semantic-layer update, and we quantify that claim rather than assert it.

Three properties of the evidence we believe fit Pattern Recognition's scope:

1. **Pre-registration discipline.** Six protocols (working-dog pilot, dataset expansion, and four budget-retention studies on NTU60, NTU120, PanAf500, and UCF101) were frozen — decision rules and thresholds included — before the experiments ran. Where a pre-registered verdict flipped under the frozen protocol (a 3-seed confirmation becoming a 10-seed partial at PanAf500; a canine-tier retention claim that does not survive protocol correction), the paper reports the reversal and the boundary, not the favorable reading. All reported p-values are Holm-Bonferroni corrected within their experiment families.
2. **Cross-domain budget retention with an honest boundary.** On the human benchmark the pipeline retains 90.6% of full-budget linear-probe accuracy at 10% of the labels; on NTU120 the same protocol retains 88.9%; on the public canine tier the corrected protocol stays near chance at a 13% budget — a tier- and label-granularity-dependent boundary we analyze rather than hide (a companion experiment shows a stronger pose extractor does not raise the ceiling, localizing the bottleneck to label alignment).
3. **Verifiable attribution.** Implementation equivalence is checked against the official reference (three-stream NTU60 fusion 77.97% vs a pre-registered 77.18% line); the taxonomy-transition cost claim (≥3× lower wall-clock, measured 6.07×) is paired with an accuracy-equivalence test inside a pre-registered noise band; and an end-to-end fine-tuning control shows the frozen-physics design is superior where both metrics agree, not merely cheaper.

All datasets are public; derived skeletons are not redistributed and regenerate from provider data via released scripts; every number in the paper traces to a committed artifact in the public repository (github.com/FOURTEEN1416/psd-framework, tag review-snapshot).

The manuscript is original, not under review elsewhere, and all authors have approved it. [FUNDING/GENAI: 按最终声明段同步]

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

> 注：AimCLR 作者与本文有基线比较关系，按多数期刊 COI 规则应**排除**而非推荐；上表保留仅为记录理由。最终 3-5 名由用户圈定，建议补 1 名动物福利领域（非 CV）审稿人以覆盖应用线。

## 3. Graphical Abstract（规格 + 脚本指针）

- 规格：Elsevier GA 建议 ≤531×131 pt（约 7.4×1.82 in），≥250 dpi，单图讲清"冻结物理层+可修订语义层+演化吸收"。
- 脚本：`docs/paper/figures/scripts/make_ga_graphical_abstract.py`（diagram-design 密度 4/10 原则 + 印刷尺寸 1:1 铁律；输出 PDF+PNG 600dpi）。
- 内容三元素：①双层框（青=物理冻结/橙=语义可修订，与 fig1 同谱系）②Y→Y′ 演化箭头只穿语义层 ③右下角保留率微条（human 90.6 / NTU120 88.9 / canine 边界如实）。

---

## 4. 链完后刷新清单（R22 后执行）

- [x] UCF101 终判入 cover letter 第 2 点（66.64% FAILS 边界措辞，2026-09-07 已补）
- [x] E9b/c 口径切换后核对摘要 90.6%/88.9% 两数是否变化，同步本稿（10-seed 终口径已核实一致）
- [ ] **R22b#7：预注册协议外部时间戳**（OSF/AsPredicted 注册或修订信中说明 repo commit 时间戳的证明力——用户人工决策项）
- [ ] Funding/GenAI 段与最终声明一致
- [ ] 通讯作者+单位+邮箱（用户人工项）
- [x] GA 图 judge 视觉验收一轮（2026-09-07 pass）
