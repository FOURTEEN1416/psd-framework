# R32 · Blind Review Report — Pattern Recognition 投稿级独立外审

**日期**: 2026-09-11
**评审对象**: `docs/paper/latex/main.pdf`（41 页评审版，双倍行距；含 main.tex、sections/*.tex、refs.bib、highlights.tex、5 图 4 表）
**评审模式**: 盲审（仅依据论文本体；未读任何内部文档/实验报告/git 历史；仅写入本报告文件）
**技能加载声明**: `scholar-critique-manuscript`（按其七维准则逐条执行，跳过 AskUserQuestion 环节，按自有稿件处理）+ `research-review`（其 `reviewer_client.py` 不可用，环境变量为空且技能目录无脚本，按技能规则转为主评审模型自行深度评审）。
**页码约定**: 引用页码为 PDF 印刷页脚 "Page N of 41"（= 物理页 − 1，物理第 1 页为 Highlights 预页）。

---

## 1. 中立摘要（Summary, ~150 words）

> The paper proposes PSD, a two-layer framework for animal behavior recognition from skeleton sequences. A frozen "physics layer" learns dynamics representations via AimCLR-style contrastive pretraining and cuts streams into behavior proposals via unsupervised motion-word quantization; a revisable "semantic layer" expands rule-engine seed labels through anchor-guided prototype clustering with iterated confidence-filtered pseudo-labeling, self-training, and warm-start initialization. Because the layers interact only through embeddings and proposals, taxonomy changes are claimed to require retraining the semantic layer alone. Evidence spans a synthetic tier (96.6% top-1; taxonomy transition absorbed at 6.07× lower wall-clock), a public-real canine tier (representation probe 2.51× random; end-to-end near chance at a 13% budget), synthetic-offset warm-start (82.0% top-1 from 20 clips, marginal budget), and pre-registered budget-shrinkage studies on NTU60/NTU120/UCF101/PanAf500 (retention 90.7%/88.9%/66.6%/88.7% of full-budget accuracy at 10% labels). The authors claim the first anchor–cluster–pseudo-label loop for temporal skeletons, supported by a repository-scale novelty survey.

**中文对照**：论文提出 PSD 双层框架——冻结的"物理层"（自监督预训练 + 无监督运动词分割）与可重训的"语义层"（规则引擎种子 → 锚点引导原型聚类 → 置信过滤迭代伪标签 + 自训练 + 热启动），两层仅经嵌入与提议段交互，故分类学变更只需重训语义层。证据覆盖合成层、公开真实犬类层（端到端近随机）、合成偏移热启动，以及 NTU60/120、UCF101、PanAf500 四个公开基准的预注册低预算保持率研究；声称时间序骨架上首个"锚点–聚类–伪标签"闭环，附仓库级新颖性调查。

---

## 2. Major concerns

### MC1 · 标题与摘要的承诺层级高于动物域端到端证据所能支撑的水平

**定位**: 标题（p1）；摘要末句 "decoupling turns evolving evaluation criteria from a re-annotation burden into a routine semantic-layer update"（p2）；E7/E7b（§4.3, p.16–18）；L13（§6.2, p.36）。

**证据**:
- E7: "the pipeline reaches **9.8% ± 7.5** top-1---at the tier's 11.1% chance level---against 33.93% under full supervision"（p.16）；
- E7b: "the warm-start arm reaches 13.1% ± 5.3 ... against a 12.5% chance rate---near chance, as on v1"（p.17）；
- E7 的显著同种子效应全部**不利于**所提初始化："the only significant same-seed effects are on macro-F1, both favoring the generic-SSL arm (warm trailing by 5.1 pp at 2 clips/class, corrected p=0.023, and 6.6 pp at 4, corrected p=0.049)"；
- L13 承认："no reported number of any published system is directly comparable under our calibers ... no superiority claim over any published method is made anywhere"。

**为什么动摇结论**: 论文的动机域是工作犬行为识别（引言第一句即训练成本与淘汰率），但全部端到端正结果落在合成数据与人类基准上；在仅有的两个真实动物端到端层（AK v1/v2）上，管线处于随机水平，且唯一显著的对照效应方向相反。PanAf500 的 88.7% 保持率是唯一的真实动物域亮点，但论文自己写明 "PanAf's retention is carried predominantly by the frozen-feature linear head, and no pipeline advantage is claimed on this tier"（p.25）。摘要末句的因果表述（"turns ... into a routine semantic-layer update"）因此只在合成层与人类基准上成立，在标题所指的应用域上没有端到端支撑。论文的披露是诚实的，但诚实披露不能替代框架层面的一致性：标题、摘要的承诺量级与正文可交付量级之间存在系统性落差，这是 framing 级的拒稿面。

### MC2 · 被声称"首创"的语义层机制，在唯一通过预注册线的基准上实测贡献 ≈0，且被论文自己的分解实验归因于普通自训练

**定位**: E9（§4.4, p.22–23）；L11（§6.2, p.35）；§3.3（方法主张）；Contribution 2（§1, p.3–4）。

**证据**:
- E9: 管线 67.53% vs 线性头 66.05%，"the pseudo-label iteration contributes **+1.48 pp** over the linear-only arm"（p.23）；
- E9 披露: "the confidence-filtered pool adds ≈35.4k pseudo-labeled clips per seed---the near-restored training-set size *is* the mechanism---the gate accepts **≈98%** of the unlabeled pool at this scale (and the prototype-path consensus gate is inert on the numeric NTU taxonomy)"；
- L11 的 2×2 分解（PSD-POOLDECOMP-PREREG-001）: "the filtering main effect is **−0.02 pp** and the pool-size main effect is **0.0 pp** (all four arms within 67.53–67.58%)"——即门控与池规模两个论文核心设计在 NTU 工作点均无可测贡献，+1.48pp 归属于"pseudo-label self-training over a near-full pool"；
- 犬类层上机制同样失效: "pool pseudo-label precision ... averages **0.11** across seeds"（E7, p.16）；对齐补救预注册为 NULL（§5, p.29–30: "the pre-registered rescue bar (mean ≥20%) is not met ... returns NULL"）。

**为什么动摇结论**: Contribution 2 与 C5/C6 的核心是"锚点引导 + 原型聚类 + 置信过滤迭代"这一组合机制的首创性与价值。但论文内部的证据链显示：在其唯一 clears 预注册线的场景（NTU60），该机制退化为"接受 98% 池的置信自训练"，其区别性组件（门控、原型通路）实测效应为零或负；在真实动物层上则彻底失败（精度 0.11）。首创性（组合无人做过）与有用性（组合带来可测收益）是两个命题，论文证明了前者、未能在任何真实层上证明后者。审稿人会问：该机制的适用工作点到底存在与否？

### MC3 · "≥3× 更低迁移成本"主张的实际意义边界：合成层专属、不体现"省标注"、且架构上非 PSD 专属

**定位**: E6（§4.3, p.19–20）；E6-real/E6-real-2/MEXP（§4.4, p.20, 25–26）；L12（§6.2, p.35–36）；Highlights 第 1–2 条（p.1）。

**证据**:
- E6 自认: "Annotation units are identical across arms by construction, so the saving is computational rather than labeling-driven"，且 "the wall-clock ratio reflects the frozen-versus-retrained architecture distinction and is **available to any design that freezes a backbone and retrains its head**"（p.20）；
- 真实域复制一（TRANS-001）: "The pre-registered cost endpoint **fails**: the decoupled arm's head retraining (466 s median) is not cheaper than end-to-end retraining of the small coupled MLP (35 s)"（p.20）；
- 真实域复制二（TRANS-002）: 15.8× 通过成本线，但 "accuracy 79.8% (coupled) vs 75.3% (decoupled, gap -4.5 pp)"，超出 ±2.3pp 等价带，判定 PARTIAL（p.25–26; L12）。

**为什么动摇结论**: 论文动机是"评价标准演化导致重标注负担"，但 E6 的两臂标注单位相同，测得的比例只是"冻骨架训头部比全管线重训快"这一普适事实；省标注的收益（rule-engine seeds + 无监督提议段 vs 全量重标注）被明确承认为"not isolated by this experiment"。两个真实域复制分别 FAILS 与 PARTIAL（后者精度等价性破带）。因此摘要中 "≥3× lower retraining cost" 的头条数字只在合成层成立，而摘要末句"re-annotation burden → routine update"的实践承诺没有任何真实域实验支撑。主张的作用域标记是做了（值得肯定），但主张本身的实践价值与标题动机的耦合是弱的。

### MC4 · 全文没有任何与已发表方法的同口径对比，外部 SSL 基线比较存在设计性混淆

**定位**: L13（§6.2, p.36）；E9 的 Mean-Teacher 比较与 L11 末段（p.23, 35）；Scope statements（§4.4, p.26）。

**证据**:
- L13: "All baselines in this paper are internal controls ... no reported number of any published system is directly comparable under our calibers"；
- E9 的外部基线: "a Mean-Teacher-style pool-distillation arm (soft pseudo-labels, τ=0.95 confidence mask ...) reaches 46.79% ± 0.61, so the PSD pipeline exceeds the classical external SSL baseline by +20.7 pp"；
- 但 L11 自己改写该差距的含义: "the +20.7 pp gap measures the value of near-full pool restoration over selective pseudo-labeling, **not the superiority of pseudo-label restoration over SSL in general**"。

**为什么动摇结论**: 论文相关工作里引用了 MCT（TIP）、GRA（TNNLS）、MAC-Learning（TPAMI）等半监督骨架方法，却在实验中一个都不比；唯一的外部基线（Mean-Teacher）在自建 harness 内以冻结特征 + 高置信阈值（仅接纳 15,758/36,082 池样本）运行——正是论文承认对其最不利的设定。对 Pattern Recognition 而言，NTU60 10% 标签是公开、标准、可复算的设定，至少一个已发表半监督基线（如 MCT 的公开口径）在共享协议下的对比是可行且必要的。在补上之前，所有"相对优势"表述只能停留在内部对照层面。

### MC5 · 关键归因链的统计功效不足：多个"判定级"结论建立在不具备检验功效的零结果或单次运行之上

**定位**: E7c（§4.3, p.18）；E4（§4.3, p.16–17）；E9 披露第 3 条（p.23）；§4.2 统计协议（p.13–14）。

**证据**:
- E7c: "30.36% versus 33.93% (**−3.57 pp**, below the pre-registered +3 pp line; verdict LABEL_BOTTLENECK; **single-run references on both sides, no seed dispersion reported**)"，并承认 "the −3.57 pp shift lies within this tier's three-seed spread of ±5.9 pp and is read as decision-grade per the frozen rule"；
- E4 端点: "Holm-corrected p=0.090 ... direction-consistent but not significant after family correction"，但该 pool precision 0.691 仍作为 Table 2 行与 C5 证据链一环；
- E9: "the 10% subset is a single stratified draw (seed 42), so the ±0.24 spread covers self-training seeds but not subset resampling"；事后四重采样给出线性臂 σ≈0.48pp——而管线对 90% 预注册线的超出量仅 0.7pp（≈1.5σ_subset）。

**为什么动摇结论**: 预注册固定了判定规则，但不能替代统计功效。犬类层的核心归因（"瓶颈是标签对齐而非骨架质量"）依赖 E7c 的单次运行零结果，其观测位移在本层种子噪声带以内；E9 对 90% 线的通过幅度与子集重采样噪声同量级。论文对各点均有一致披露，但从外部审稿视角，"预注册判定规则 + 低功效观测"产生的 verdict（LABEL_BOTTLENECK、GENERALIZES）应降格为方向性证据，相应的机制叙事（L9 的对齐诊断、90.7% 头条）需要更谨慎的定级。

### MC6 · 方法规格存在可复现性缺口：规则引擎从未被定义，语义层 Ω 的参数化跨层不一致且无统一说明

**定位**: §3.3.1（p.9）；§3.3.3（p.11）；E6/E9 各臂描述（p.19–20, 22–26）；Algorithm 1（p.11）。

**证据**:
- 规则引擎——语义层的起点——全文仅一句: "Rule-engine coarse labels over physical priors provide seed anchors A = {(e_i, y_i)}"（§3.3.1, p.9）。任何一条具体规则（物理先验是什么、阈值如何定、每层种子规则是否相同）均未给出；图 3 标题里出现的 "rule-engine seeds, confidence ≥0.8 and duration ≥0.5s" 是伪 GT 过滤器而非规则本体；
- Ω 的形态在各层漂移：合成层为 ST-GCN 头（E6 "rebuilds only the semantic head"）、NTU 为 "StandardScaler plus logistic regression"（E9）或 CPU 逻辑回归（TRANS-001）、AK 为重训头——方法章没有定义 Ω 的函数族与选择依据；
- 置信度 "κ is the calibrated top1−top2 margin"（§3.3.2）——校准方法（温度缩放？分位映射？）未说明。

**为什么动摇结论**: 框架的两个接口（种子如何产生、Ω 是什么）是复现的必经之路，目前论文内信息不足以独立复现语义层；可复现性完全外包给"released repository"。对期刊审稿而言，方法章自足性不足是标准的首要修改要求。

### MC7 · 分割组件（SMQ）的必要性未被自家消融支持，但仍在贡献链与流水线中承担载荷角色

**定位**: E2（§4.3, p.15）；Table 3 第 3 行（p.27）；§3.2.2（p.9）；fig3（p.15）。

**证据**:
- E2/Table 3: "SMQ 0.458±0.049 > uniform equal-count window 0.399±0.035 > random-cut null 0.323±0.022; all three pre-registered gates pass. **A fixed-grid probe (0.453±0.027) is statistically on par with SMQ**; uniform windows lead on boundary F1 (0.396 vs. 0.343)"；
- 摘要亦承认: "a fixed-grid probe is statistically on par"（p2, Contribution 3）。

**为什么动摇结论**: 物理层的第二个组件（提议段生成）与一个固定网格打分在主指标上统计不可分、在次指标（boundary F1）上更差。若提议段质量不敏感于分割方法，则"motion-word quantization"作为贡献组件（C4、fig3、方法 §3.2.2 整节）的地位应降格为"任一切分即可"，或论文需给出 SMQ 影响下游（而非 IoU 代理）的证据。目前 C4 的证据状态与其在方法叙事中的载荷地位不匹配。

---

## 3. Minor issues

1. **正文泄漏内部路径与工件名**（违反论文自身 §4 开头的装配纪律）: 印刷 p.29（§5）两处 `Evidence: reports/p15-label-alignment-2026-09-05.json`、`reports/p16-dap-aptv2-2026-09-05.json, reports/p17-budget-curve-2026-09-05.json`；p.17（E7b）`(released artifact r12-holm-eightclass, eightclass_control block; that file's separate holm block pertains to the superseded leaky protocol and is retained only for the audit trail)`——内部审计记账不属于正文；Table 4 标题（p.37）`the full check table is in the released survey log (dev-docs/research/)`。应统一改为"released repository"级别的表述。
2. **比值数值失真**: p.25–26 E6-real-2: "reads 32.5x (coupled 1520 s vs decoupled 47 s)"——1520/47 = 32.3×；如系中位数未取整所致，请给出与分量一致的比值或注明。
3. **摘要超长**: 约 312 词（PR 惯例上限 ~250 词），且单段信息密度过高（一个句子里嵌套三层括号披露）。建议拆分并压缩至 250 词内，把分层披露下沉到正文。
4. **数字重复疲劳**: 头条数字（90.7%/88.7%、82.0%、9.8%、44.9%、≥3×、2.51×）在摘要、Highlights、贡献列表、§1 末段、对应实验段、结论各重复一遍（每数 ≥5 次）。§1 第 5 段几乎逐字复述摘要。建议贡献列表给数字、末段给定性（或反之）。
5. **未定义内部行话**: "spc2/spc4/spc12"（p.17 E7b "ordered as at spc2"; §5 p.30 "from spc2 (13% labels) to spc12 (77% labels)"）——"spc"（samples per class）从未展开定义；"R16"（p.16 E7 "a corrected protocol (R16)"）对外部读者无意义。均需在首次出现处定义或删除。
6. **引用缺口**: ① Mean-Teacher 基线（E9/L11）未引 Tarvainen & Valpola (2017)；② L9 的 "AdaBN-style statistics" 未引 Li et al. (2017)；③ §3.2.1 的 InfoNCE 未引 van den Oord et al. (2018)；④ "prototype clustering" 谱系未引 prototypical networks（Snell et al. 2017）；⑤ Table 4 中 "hierarchical ST-VQ" 无对应参考文献条目；⑥ Hungarian matching 未引 Kuhn (1955)（惯例可免，但 ①② 应补）。
7. **排版/文字**: p.29 双句点 "(L9).. Evidence:"；Algorithm 1 第 7 行 until 子句的括号结构别扭（"or (where a rule-derived consensus reference exists; Section 3.3.2) pool precision ... drops"），建议改写；L12 内 "15.8x/6.2x/32.5x" 用 ASCII x 而他处用 $\times$，全文统一。
8. **长段落**: E7b（p.17–18）、E9（p.22–23）、E9b/c（p.23–24）、E9d（p.25）均为 400+ 词单段，含多层破折号插入语；建议按"设置—结果—披露"拆段或改用小标题。§4.4 与 §4.3 的实验编号系统（E1–E9d + MEXP + 五臂研究）需要一张实验索引小表帮助读者导航。
9. **Table 2 信息不全**: "Synthetic-offset | 20-clip warm-start | 82.0%" 未带正文已有的 ±4.3；"96.6% (single run, seed 42)" 的单次运行披露值得肯定，建议全部行统一给出离散度或"single run"标记。
10. **Highlights 第 5 条** "Warm start: 82.0% top-1 from 20 clips (synthetic-offset; marginal budget)"——即使带括号限定，"from 20 clips" 在 highlight 层面仍具误导性（初始化权重来自 2,200 全标注源域片段；cold-start 同预算仅 8.0%）。建议改为 "82.0% top-1 adapting to offset with 20 clips"之类明确"适配预算"措辞。
11. **匿名合规核查**: Data availability 给出 `github.com/FOURTEEN1416/psd-framework`（含用户名）。PR 若为单盲评审则无碍；若投稿时切双盲需置换为匿名仓库。页脚 "USER: surname et al." 与作者区占位符投稿前必须完成（作者已自知，列此备查）。
12. **E7c 未给对照 mAP**: 只给强提取器 "pose mAP50 0.834 versus the production extractor's smaller model"，未给生产提取器的 mAP——归因强度（"substantially stronger"）应可复核。
13. **结论措辞**: §6.1 末句 "we verify the pattern on skeleton data in two domains and state it as a design claim elsewhere"——"elsewhere" 指代不明，删或改为具体出处。
14. **附录 B 可读性**: 复现链一段塞入 ~25 个脚本名的 run-on 句（p.38）；建议编号列表化，并与实验编号（E1–E9d）对齐成映射表。

---

## 4. 总推荐与置信分

### Overall recommendation: **Major Revision**

**Confidence: 8/10**（已全文精读含附录与参考文献，完成全部数字的内部一致性与算术复核；未做外部文献检索核证新颖性主张——这是置信度未到 9–10 的主因之一）。

### 值得保留的优点（修改中不要丢掉）

1. **协议文化罕见**: 预注册协议（PSD-*-PREREG-001 系列）、Holm 校正族、保守界规则（"≥3× claimed, 6.07× measured"）、错误修正的完全披露（withdrawn oracle-leakage 数字、superseded readouts 双轨保留），这套纪律高于本刊平均水平，是论文最硬的资产。
2. **MEXP 结果有独立科学价值**: "linear head on the coupled arm's own frozen penultimate features beats the coupled arm's end-to-end output (10/10 seeds, p=0.002)" 是一个干净、可证伪、对社区有用的观察。
3. **可证伪化的机制表述**: 保持率梯度写成 "any tier whose full-budget reference is strongly separable (≥70%) yet retains <85% would break it"，并预留 K9 试点作为 falsification test——这是正确的科学姿势。
4. **负面结果的质量**: E5（不确定性采样失效 + 饱和 softmax 诊断）、E7/E7b/E7c 三角归因、L9 的 AdaBN 阴性——负结果附机制诊断，比多数正结果论文更有信息量。
5. NTU60 实现等价性验证（77.97% vs 预注册线 77.18%）是重实现论文应有的做法。

### 给编辑的意见（3–5 句）

This manuscript tackles a well-motivated problem (taxonomy evolution in animal behavior recognition) with an unusually disciplined experimental protocol, including pre-registration, corrected statistical families, and unusually honest negative reporting. However, the headline claims outrun the animal-domain evidence: end-to-end recognition on the canine tier is at chance, the novel anchor–cluster–pseudo-label machinery is measured to contribute approximately nothing beyond plain self-training on the only benchmark that passes its pre-registered line, and the ≥3× transition-cost claim is synthetic-only with real-domain replications returning FAILS and PARTIAL. There is no same-protocol comparison against any published method, and the method section omits the rule-engine and semantic-layer specifications needed for independent reproduction. None of these flaws is fatal—the underlying measurements are careful and internally consistent—but they require either a substantive reframing of claims (title, abstract, contribution list) plus at least one published external baseline, or new target-domain evidence. I recommend **Major Revision** with re-review.

---

## 5. 最想问作者的两个问题

**Q1（机制价值）**: 你们自己的 2×2 分解（L11）显示在 NTU60 工作点上门控主效应为 −0.02pp、池规模主效应为 0.0pp，管线对线性头的净贡献是 +1.48pp，而犬类层上伪标签池精度只有 0.11、对齐/骨干/预算三个杠杆全部 NULL。**请指出一个具体工作点（数据层 × 预算 × 类空间），在该点上"锚点引导 + 原型聚类 + 置信过滤迭代"这套被声称首创的机制，能在同一 corrected protocol 下显著优于"对近全池做置信自训练"这一最平凡基线——如果目前不存在这样的工作点，这套机制的适用边界应该如何在摘要与贡献列表中重写？**

**Q2（迁移成本主张的标注经济学）**: E6 承认两臂标注单位相同、比值"available to any design that freezes a backbone and retrains its head"，而两个真实域复制分别 FAILS（无成本可省）与 PARTIAL（精度等价破带 4.5pp）。**论文的真正卖点是省重标注——你们能否设计并运行一个实验，使taxonomy 变更时语义层确实只需 rule-engine seeds 而基线臂需要全量重标注（即让标注单位不对称），哪怕在一个公开数据集的人造 taxonomy 迁移上？如果现在就能跑，为什么没跑；如果不能，"routine semantic-layer update" 这一摘要级承诺依据什么成立？**

---

## 附：程序化核查记录（评审过程留痕）

- 页数 41（pdfinfo）；未解析引用 "??"/空方括号：0 处；`\TODO` 残留：0 处。
- 算术复核通过项：2.51×、6.07×、15.8×、88.9%、88.7%、90.7%、86%、1.80×、1.99×、+4.17pp、+11.54pp、85.8%、66.6%、28×、4.55pp、1/22=4.5%；失真项：32.5×（见 Minor #2）。
- 图表视觉检查（100dpi 渲染）：fig1（p.4）、fig2（p.10）、fig3（p.15）、fig4-budget（p.20）、fig5-AL（p.31）、Table 1–4 渲染均清晰、无溢出、无图例碰撞；交叉引用（Section 4.1/6）解析正确。
- 交叉引用泄漏扫描：正文 3 处内部路径 + 1 处内部工件名 + 1 处 dev-docs 引用（见 Minor #1）。

*评审独立完成于 2026-09-11；本报告为唯一写入产物，未改动论文任何文件。*
