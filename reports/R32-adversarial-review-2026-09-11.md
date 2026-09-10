# R32 · 最敌意审稿人三重人格 · 对抗审稿报告

- **日期**: 2026-09-11
- **对象**: `docs/paper/latex/{main.pdf (41页), main.tex, sections/*.tex, refs.bib, highlights.tex}` + `docs/paper/figures/*.png`
- **方法**: ars-adversarial-reviewer 三敌意人格方法论（每人格强制 ≥1 条发现；≥2 人格同捕升一级；BLOCK/CONCERNS/MINOR）+ scholar-critique-figures 图表检查单。迁移到论文审稿：A=拒稿狂 AE（framing）、B=统计检察官、C=复现刁难者。
- **证据纪律**: 每条给印刷页码（页脚 "Page N of 41"，物理页=印刷页+1）+ tex 文件:行；已用 pdftotext 全文+逐页渲染 PNG 目检（Table 1–4 文本层错位经视觉复核为提取伪影，**排版无恙**，不做指控）；悬空引用与零引用条目经 grep 全文核verified。
- **边界**: 只读论文；唯一写入本报告。

---

## 人格 A · 拒稿狂 AE（framing 层攻击）

**A1 [CONCERNS] 标题域与证据域系统性错位——标题域恰是全文唯一全负结果域。**
定位: 标题+摘要 p.1；贡献4 p.2–3；tab:main p.26；E7 p.16；E9d p.23–24；L7 p.33；L13 p.35。
攻击链: 标题承诺 "Animal Behavior Recognition"。全文的正向数字依次落在：合成生成器数据（96.6%，单 run）、人类骨骼基准（NTU60/120 的 90.7%/88.9% retention）、subject-ID 代理探针（20.89%，L1 自认非行为精度）。而标题域内：canine 端到端 9.8%±7.5（chance 11.1%）与 v2 13.1%（chance 12.5%）双双近随机；44.9% 四类聚合被 L7 自认"majority recall accounts for essentially all"；PanAf500 上 "no pipeline advantage is claimed"（p.24），retention 由线性头独立承载。目标应用 working-dog（real-K9）整行空白。论文对每一条都做了披露，但 framing 层结论不变：**唯一没有正结果的就是标题卖点域**。"tier-dependent boundary" 是诚实的，但它同时是一份自供状——把标题域证伪、把正结果寄存在人类域与合成域。

**A2 [CONCERNS→升级] 卖点机制在全部头号数字处被论文自家证据测得≈惰性（与 B4 合并升级 BLOCK，见 M1）。**
定位: §3.3.2 p.10–11（共识门只管 standing 类；AK/NTU 分类空间"passes all proposals by construction"）；E9 p.21（gate 接受 ≈98% 池）；L11 p.34（POOLDECOMP：filtering 主效应 −0.02pp、pool-size 主效应 0.0pp）；E4 p.14–15（校正后 p=0.090 n.s.）；E7 p.16（canine 池精度 0.11）。
攻击链: 标题/摘要/贡献2 卖的是 anchor-guided + confidence-filtered + prototype-consensus 机制；但全部亮眼数字所在的操作点上：共识门构造性失效、置信门近似不过滤（98% 接受）、2×2 分解测得门与池规模效应均为零、唯一正向端点（+17.9pp）过不了自家校正。机制真正"工作"的证据只剩合成层（E3 purity 0.5339，且无任何显著性检验）。**方法论文的方法在其证据的 nowhere 是 load-bearing 的**——这是 framing 级可拒稿点。

**A3 [CONCERNS→升级] 与任何已发表方法零对比；外部基线是自实现+阈值未调的反向选择性设置（与 B3 合并升级 BLOCK，见 M2）。**
定位: L13 p.35（自认全部基线为内部对照）；E9 p.21–22；L11 p.34（自认 Mean-Teacher τ=0.95 的选择性"is the opposite of the pipeline's near-full pool acceptance"，+20.7pp 测的是池恢复价值非 SSL 劣势）；§2.1 p.4（ASBAR 75.3%、BCST-GCN 95.36% 有公开数字却未在任何同口径下对比）。
攻击链: Pattern Recognition 级方法论文，基线全为 random/majority/scratch/自实现 AimCLR/自实现 Mean-Teacher。唯一的"外部"对比在自家Harness里给基线设了一个与自家门相反且未调的阈值，然后论文自己承认该 gap 不可解释为优势。AE 结论：贡献声明（firstness+机制优势）没有任何已发表参照系。

**A4 [MINOR] Firstness 声明的证据强度薄弱：最近邻是未评审 arXiv 预印本，矩阵多行无法在印刷品中核验。**
定位: 贡献2 p.2；§2.3 p.6–7；Table 4 p.37（"HAA4D; ISBFSAR; FICAMA; SMAM; UMEG-Net; SkelHCC" 等 7 作捆绑一行且无引键）；refs.bib:293–298（pointsup2026 = arXiv 2603.06201）；L3 p.32。
攻击链: "first of its kind" 依赖 GitHub 代码搜索 + awesome-list 扫描（ten query groups）；differentiation 的对象是未评审预印本（可随时更新为含锚点+迭代）；附录矩阵的 Difference 列对无引文行不可核验。survey log 在 `dev-docs/research/`（仓内路径）。L3 已披露边界，但披露不等于豁免。

**A5 [MINOR] 动机统计引用新闻特稿；动机域与交付域脱钩。**
定位: p.1 §1 与 p.3 §2 开篇两次引用 [1]（grimm2026science，note "News feature"）；tab:main Real-K9 行 "not exercised"（p.26）。
攻击链: "数万美元/只、过半淘汰" 的立项数字来源是 Science 新闻特稿而非同行评审数据；而以该动机命名的 target tier 全文未动。动机的"痛点演示"与交付物处于两个域。

**A6 [MINOR] 边际预算限定词在 abstract→intro→conclusion 逐级衰减，结论裸奔。**
定位: 摘要 p.1（有限定 "(the marginal budget on top of a fully labeled source domain)"）；§1 p.2–3（"82.0% from 20 labeled clips" 无限定 ×2 处）；§6.1 p.30（"82.0% top-1 on 22 classes from a 20-clip budget" 无限定）。
攻击链: 82.0% 的真实语义 = 2,200 片全标注源域饱和训练 + 20 片边际适配（§4.3 p.18 有完整披露）。限定词在摘要存在、在引言消失、在结论完全消失。读者的最后记忆是"20 clips → 82%"。

---

## 人格 B · 统计检察官

**B1 [CONCERNS→升级] 44.90% 是 3 种子最优值上表；乘数用的最弱基线（与 C5/A7 合并升级 BLOCK，见 M3）。**
定位: E4 p.15（"44.90% overall (seed 42; three-seed mean 41.5%, ±5.9)"）；tab:main p.26（印 44.90%）；§1 p.3（"44.9% ... (1.80× the 25% random baseline)"）；L7 p.33（watch 72/track 46/stay 27/jump 27；per-class 100/23.5/0/0）。
攻击链: (1) 摘要链与主结果表报最优单种子，均值 41.5±5.9 藏在 E4 正文与 L7——这不是"报均值顺带提极值"，是反向。(2) 1.80× 的分母选了 25% 均匀随机；按 L7 印刷的 support（val 全表 watch≈41%），多数类基线≈41%，真实乘数≈1.09×。(3) per-class 100/0/0 表明聚合分被单类扛走——L7 承认。三件事叠加 = 选择性报告的教科书样本，且发生在论文唯一"看着像正结果"的真实动物数字上。

**B2 [CONCERNS] MEXP 十种子 Wilcoxon p=0.002 逃逸自家 Holm 族——违反论文自己的统计协议。**
定位: MEXP p.24（"paired Wilcoxon signed-rank p=0.002; sign test 10/10, p=0.002"，无任何 family/correction 归属）；§4.2 p.13–14（全文唯一定义的族 = E9-series eight-test family，2026-09-07 固化；且自定协议 "within each experiment family the reported p values are Holm–Bonferroni corrected"）。
攻击链: n=10 的 Wilcoxon 最小可达 p 恰为 0.00195——报告的就是这个理论下限值，而 M 对 coupled 的检验不在任何声明的校正族里（MEXP addendum 晚于 2026-09-07 族固化）。p=0.002 大概率扛得住任何合理校正，**但协议合规性崩了**：自家立的规矩"每个 family 必须 Holm"，新检验不带族上线。检察官问：下一个不进族的检验会在哪？

**B3 [CONCERNS→升级] 校正族治理全线事后：未定义的 "three-test family"、结果已知后固化的 eight-test 族、事后豁免的 PanAf、阈值伪影包装成 +20.7pp 头条（与 A3 合并升级 BLOCK，见 M2）。**
定位: E4 p.14（"over the three-test family"——全文仅此一处，三检验是什么从未列出，grep verified）；§4.2 p.13（"the consolidated eight-test family was fixed before corrected inference entered this manuscript"，同日 2026-09-07 十种子约定采纳，3-seed readouts 已存在且"both readouts are released"）；p.13（PanAf 以"ties ⇒ uninformative"排除出族，ninth-member 敏感性只有一句断言）；L11 p.34（+20.7pp = 阈值不对称的自认）。
攻击链: 预注册管住了单点决策规则，管不住族结构：族是结果部分可见后合并的；E4 的族连成员清单都没给；E3 purity 头条（0.5339, 1.61×）从头到尾没有任何检验。敌意读法：**"pre-registered" 品牌被用作了统计信誉抵押，而族的边界在需要时移动**。

**B4 [CONCERNS→与A2合并] "Statistical equivalence" 由 band-in 宣告，无等效性检验；旗舰 margin 的归因被自家分解清零（与 A2 合并升级 BLOCK，见 M1）。**
定位: E6 p.18–19（−0.91/+2.27 双双 <2.3pp ⇒ "statistical equivalence"；+2.27 距带边 0.03pp；n=3；CPU 小档）；L9 p.33（mean −2.04 "inside the ±5.9 pp band" 同款推理）；L11 p.34（POOLDECOMP 四臂 67.53–67.58%）。
攻击链: band-in 是点估计落域，不是 TOST/CI 包含；n=3 的等效性声明几乎无功效，+2.27 这个数离违约只有 0.03pp——保守界 ≥3× 的另一半（accuracy 端）建立在这样的"等效"上。同时 POOLDECOMP 把旗舰 +1.48pp 的机制归因清零后，论文改口"pseudo-label self-training over a near-full pool"——那是一个无人声明的朴素机制。

**B5 [CONCERNS] 单次抽样预算子集 + 噪声量级单 run 裁决被 frozen rule 授予 "decision-grade"。**
定位: E9 p.22（10% 子集=seed 42 单次分层抽样；重抽诊断只做了 linear 臂 4 抽 ±0.48；"the decoupled arm's subset sensitivity is not measured"）；E9b/c p.22（同款声明）；E7b p.17（重编码移动参照 2.1pp = 两个验证片段——自认）；E7c p.18（−3.57pp、单 run、低于该 tier 自身 ±5.9pp 三种子散布，却依 frozen rule 判 LABEL_BOTTLENECK）。
攻击链: 全部 E9 系列 retention 头条条件在每 tier 一次未重抽的预算子集上；管线臂的子集敏感度在四个 tier 上全都没测。E7c 则是"预注册规则把小于噪声的差值升格为归因裁决"——预注册防 p-hacking，不防 rule 本身无效。检察官结论：±band 系统性窄于真实不确定性。

**B6 [MINOR] C4 分割头条 n=4 episodes，且对 null 无印刷在案的检验；trivial 探针打平被降格处理。**
定位: E2 p.14；fig3 p.15（bar 图四根柱，"4/4 > baseline"）；tab:3 p.27（"all three pre-registered gates pass"——门槛内容未列）；§1 贡献3 p.2 与 §5 p.27（"fixed-grid probe is statistically on par"）。
攻击链: 0.458±0.049 的 ± 来自 4 个 episode；与 random-cut null 的差异无检验记录（三道"预注册门槛"内容不透明）；而与 trivial fixed-grid 打平这一事实同时出现在贡献句的括号里——贡献3 的第二半句在为第一半句拆台。

**B7 [MINOR·疑点] 若干派生数字无法从印刷值复算（疑点，confidence=medium）。**
定位/算账: E7b p.17 "above-chance margin 13.5→25.0pp"：33.93−11.1=22.8、33.93−8.33=25.6，无一路径得 25.0；"above-majority −4.8→+3.1" 的多数类值未印。E6-real-2 p.24 "32.5×"：1520/47=32.34。MEXP "+2.9"：82.57−79.71=2.86（四舍五入可容）。tab:main p.26 "67.5%" vs 正文 67.53（可容）。
攻击链: 单条皆可用"底层未舍入值"辩护，但合起来构成一个模式：**派生量不附输入值，审稿人无法在印刷品上复核**。另：96.6% 合成头条为单 run（tab:main 已注明）。

---

## 人格 C · 复现刁难者（只凭正文+附录 Reproduction chain）

**C1 [CONCERNS] 悬空披露：成本端点的测量条件在全文没有任何前置交代——"disclosed above" 指向不存在的内容。**
定位: E6-real-2 p.24 = 04-experiments.tex:80（"All three coupled-arm seeds include host-restart segments (the RAM-pressure crashes disclosed above)"）；grep 全文（sections/+main.tex）："RAM/crash/restart" 仅此一处。
攻击链: 论文旗舰成本复制（15.8×/−4.5pp，tab:main 同款）的全部三个 coupled 种子都含宿主崩溃-重启段，wall-clock 是"每种子贡献段求和"——而这些事实的"上文披露"根本不存在。这是主端点测量有效性的关键上下文（消费级笔记本、RAM 压力、崩溃重启计账），第一次出现竟是一个自我引用。6.2× final-segment 界有设防，但"披露链断裂"本身坐实：读者无法从论文评估 9897s 这个 median 的构成。

**C2 [CONCERNS] 引用真空：核心基准与方法零条目 + 参考文献渲染破损。**
定位（grep verified，refs.bib 全 33 条无对应键）: PanAf500（E9d 全节、tab:main、fig4、结论——**零引用**）；UCF101（零引用，无 Soomro）；Mean-Teacher（外部基线本体，无 Tarvainen & Valpola）；HRNet 2D 标注（E9b/c）；AdaBN（L9）；YOLO11/YOLO11x-pose（L6/E7c/E9d）；PYSKL（p.22）。渲染破损：ref [20] p.39 "Pattern **Recognitionrepo** github.com/..."（期刊名与 note 粘连）；ref [23] p.39 DOI 直出 `10.1007/978-3-031-73347-5\_14`（反斜杠入印刷品）。
攻击链: 用了谁的基准/方法就得引谁——这是期刊硬规范；外部基线（Mean-Teacher）恰是论文对比故事的主角，无引文=无法溯源版本与协议。R23 曾修过 liu2020ntu 引用违规，同族缺陷在新增内容里回潮。

**C3 [CONCERNS] 自包含性失败：语义层核心超参全部不在论文里；复现链自带"作废脚本"混排。**
定位: §3.3.2 p.9–10（κ "calibrated" 无校准法；τ "quantile targeting on estimated coverage" 无分位数/覆盖目标值；"frequency-aware margins" 无公式；K 如何选无说明）；§4.2 p.13（种子清单外置 "released experiment ledger"；"first working setting, not the outcome of a systematic search"——LR 0.05/queue 1024/T=64 之外无任何训练超参如 batch/optimizer/schedule）；附录 B p.38（链内明确混入 "superseded protocol, see the errata notes in the p07/p08/p10/p12/p14 reports" 的脚本——哪个表/图属于哪版脚本的映射在论文之外）；Data availability p.36（"review snapshots are tagged review-snapshot"——单一 tag 名无法寻址多个快照）。
攻击链: 论文把复现性外包给仓库考古学：读者必须知道去哪个 commit、读哪些 errata、对哪个 JSON 才能知道 E7 的 9.8% 出自 r16 而非 p07/p08/p10。附录本为消除"补充材料不存在"的指控而设，现在它自己承认链内有五处脚本作废需外部勘误。单 tag 指针在多次重锚的工作流里无法充当不可变地址。

**C4 [MINOR·疑点] 82.0% 头条的协议血统未声明；v2 warm 臂的暴露混淆使其不可作为方法证据。**
定位: warm-start 段 p.18（未声明该结果出自 corrected protocol 还是旧协议——对比之下 E7/E7b/E9 每处都声明 "corrected protocol of E7"）；§5 p.28（自认 public-real 的早期 warm-start 优势"was an artifact of a true-label head and is superseded"）；E7b p.17（v2 warm 臂"partly co-occurs with the warm arm's source-video exposure"）。
攻击链: 论文自己证明过 warm-start 叙事能在泄露协议下产幻；而合成档 82.0% 恰恰没有贴协议血统标签。疑点（confidence=medium），非指控：一段话即可澄清，但审稿人有权要求先澄清再采信头条。

**C5 [MINOR→与B1/A7合并] Cherry-picking 面向清单（按本人格章程专项排查）。**
定位: 挑种子 = tab:main p.26 印 44.90（best-of-3，见 M3）；挑 episode = fig3 p.15 展示 lowest/highest-IoU 两极（caption 已披露）；挑预算点 = fig4 p.20 对数轴位移 6 个点（caption 已逐点列出位移量）。
攻击链: 后两项有披露，属可接受但需复审的操作；第一项无豁免。三处叠加构成模式：**展示层总在取有利切面，防御全靠 caption/局限节的远端披露**。

**C6 [MINOR] 正文携带仓内路径/内部编号；占位符处于 desk-reject 状态。**
定位: §5 p.28–29（`reports/p15-label-alignment-2026-09-05.json`、`p16-dap-aptv2-...json`、`p17-budget-curve-...json` 直书正文；`(L9)..` 双句点）；附录A p.37（`dev-docs/research/`）；摘要 p.1（"; On the synthetic tier" 分号后大写）；作者/单位/邮箱/ORCID 全为 "USER:" 占位（p.1–2、p.35）；GenAI 披露颗粒度未定稿（main.tex:132 注释自认待终裁）。
攻击链: 内部路径违反自家装配纪律（04:3 "内部路径/文件名禁入正文(ROUND2 发现#7)"）；占位符作为"投稿前状态"可原谅，但若以此状态出稿即是 desk-reject。

---

## 合并与升级说明

去重合并（同一问题多人格命中）：

| 合并项 | 构成 | 基级 → 终级 | 一句话 |
|---|---|---|---|
| **M1** | A2 + B4（POOLDECOMP 零效应 = 归因崩塌的统计面） | CONCERNS ×2 → **BLOCK** | 卖点机制在其全部头号数字处被自家证据测得惰性；"statistical equivalence" 无等效检验支撑 |
| **M2** | A3 + B3（+20.7pp 阈值伪影 = 无对比的统计面） | CONCERNS ×2 → **BLOCK** | 与任何已发表方法零对比；唯一外部基线为未调阈值的反向选择性自实现 |
| **M3** | B1 + C5 + A7（intro/表头框架面 + 复现完整性面） | CONCERNS ×3 → **BLOCK** | 44.90=best-of-3 上主结果表，均值与类崩塌远端披露；乘数用最弱基线 |

未合并的独立 CONCERNS（6）: A1（标题域错位）、B2（MEXP 逃逸 Holm 族）、B5（单次子集+单 run 裁决）、C1（RAM 悬空披露）、C2（引用真空）、C3（自包含性失败）。
MINOR（7）: A4、A5、A6、B6、B7、C4、C6。
注：Table 1–4 的文本层错位与标题区 117pt overfull 经逐页渲染目检排除（不可见或为提取伪影），不做指控——本报告不收化妆品级假线索。

---

## 最终裁决

- **裁决: REJECT**（三人格合并后含 3 条 BLOCK；按 ars 裁决制 BLOCK≥1 即拒——本人格按"最敌意"设定行使该规则）。
- **拒稿概率: 55%**。一句话理由：三条 BLOCK 全部是"可修复型"（L7 的均值、POOLDECOMP 的四臂、已发布的脚本链都存在，修复以重排+补引+族治理为主，不需新实验），真实审稿面板更可能给 MAJOR REVISION，但以当前印刷稿状态送审，半数面板会直接拒。
- 修复优先级（若进入 revision）: M3 一行表改（Tab2 印均值+乘数换多数类基线）→ C2 补引 + ref[20]/[23] 修复 → C1 补一段崩溃-重启测量条件披露 → M1 在 abstract/§1 把机制卖点降级为"合成档验证的机制 + 真实档边界" → M2 补一个已发表方法同口径跑分或把 L13 升进正文 Discussion → B2/B3 把 MEXP 与 E4 纳入声明族并打印族成员清单。

### 模拟审稿意见信（英文，145 words）

> The manuscript reports an elaborate self-audit, yet its central selling point is contradicted by its own evidence. The anchor-guided, confidence-filtered pseudo-label machinery — the claimed contribution — is measured inert at every operating point where the headline numbers live: the NTU60 gate accepts ~98% of the pool, your own 2×2 decomposition attributes −0.02 pp to filtering, the consensus gate is disabled by construction on all real tiers, and canine pool precision is 0.11. No comparison against any published method is made; the in-house Mean-Teacher baseline runs with an untuned threshold that your own text identifies as the source of the +20.7 pp gap. The strongest real-animal number in the Introduction (44.9%) is the best of three seeds; the mean (41.5±5.9) and per-class collapse appear only in Limitations. I see no path to acceptance within this framing. Reject.

---
*审计闭环: 本报告为唯一写入物；论文零改动。技能加载: ars-adversarial-reviewer v2.9.0 + scholar-critique-figures（均按 SKILL.md 全流程执行，人格纪律/升级机制/图表检查单逐项对照）。*
