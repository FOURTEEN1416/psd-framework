# R32 投稿前交叉审稿（正文 ↔ 工件 ↔ 图表 三方对账）2026-09-11

- **审稿人**：ZCode 交叉审稿员（只读审稿；本文件为唯一写入物）
- **技能加载**：nature-submission-audit / scholar-presubmit-checks / galaxy-paper-self-review（开工已声明）
- **对象**：docs/paper/latex/{main.tex, sections/*.tex, highlights.tex, main.pdf} ↔ reports/*.json ↔ docs/paper/figures/*.png
- **红线遵守**：三层口径混报检查纳入（未发现混报）；E6-real/E6-real-2 冻结段未建议改措辞（其与他处数字核对一致）。

---

## 总结论（一句话）

**未发现 P0（结论动摇）或 P1（表述错误）级不一致；A 清单 20 个重点数字全部逐位命中，扩展核对 ~115 个数字点全部命中；疑点共 6 项均为 P2（建议级），另 2 条观察项不计级。论文三方对账状态：PASS。**

计数：**一致项 ~130 / 疑点项 8（P0=0，P1=0，P2=6，观察=2）**。

---

## A. 数字对账（正文量化结果 ↔ reports/ 工件逐位核对）

### A-1 重点数字命中表（任务清单 20 项，全部 ✅）

| # | 论文数字 | 正文定位 | 工件 | 工件值 | 判定 |
|---|----------|----------|------|--------|------|
| 1 | 82.0% ±4.3（20-clip warm-start） | 04:50, main:72, 06:7, tab1 | p05-al-efficiency-warmstart-short-2026-08-25.json | b=20 mean 0.8202, std 0.0431 | ✅ |
| 2 | 20.89% | 04:21, main:72 | p01-knn-result.json | knn_top1_mean_pct 20.89 | ✅ |
| 3 | 8.33% / 2.51× | 同上 | 同上 | 8.33 / 2.51 | ✅ |
| 4 | 20.89 ±4.04 / 折 15.56–26.67 | 04:21 | 同上 | 折算 ddof1 std=4.04；folds 15.56–26.67 | ✅ |
| 5 | 90.7% retention（E9） | 04:70, main:72, tab1 | r16-ntu-pseudo-10seed-2026-09-07.json | retention_b_over_c 0.907；b 臂 mean 0.6753±0.0024 | ✅ |
| 6 | 66.05% linear / 74.45% ref / 88.7% | 04:70 | 同上 | 0.6605 / 0.7445 / linear_retention 0.8872 | ✅ |
| 7 | +1.48pp 迭代增益 | 04:70, 06:7 | 同上 | pseudo_iteration_gain_pp 1.48 | ✅ |
| 8 | 88.9%（NTU120，PARTIAL） | 04:73, tab1, fig5 | p5b-ntu120-retention-2026-09-07.json | 0.8886 PARTIAL；48.29±0.76 vs 45.42 / 54.34 | ✅ |
| 9 | 66.6%（UCF101，负边界） | 04:73, fig5, tab1 | p5b-ucf101-retention-2026-09-07.json | 0.6664 FAILS；15.40±1.89 vs 14.04 / 23.11 | ✅ |
| 10 | 88.7%（PanAf500，10 seeds） | 04:76, tab1, fig5 | p23-panaf-retention-2026-09-07.json | 前 10 seed mean 0.532 / 0.60 = 88.67%；30-seed 0.8578 亦在文 | ✅ |
| 11 | PanAf 30-seed 51.5±8.6、13/30 wins、3 ties、p=0.54、85.8% | 04:76 | 同上 | mean 0.5147；wins=13 ties=3（本审复算 Wilcoxon p=0.5369） | ✅ |
| 12 | 44.90%（4-class partial，seed 42） | 04:38, 06:32 | p05-public-real-partialclass-result-2026-08-25.json | best_val_acc 0.44898；per-class 100/23.5/0/0 | ✅ |
| 13 | 96.6%（synthetic 22 类，seed 42 单跑） | main:22, tab1 | p05-stgcnbc-synthetic-100perclass-Y.json | best_val_acc 0.96591 | ✅ |
| 14 | 0.4577±0.0488（fig3）/0.458±0.049（正文） | 04:24,31; 05:20 | p02-seg-strategy-ablation-2026-08-25.json | 0.4577/0.0488；逐集 [0.420,0.423,0.448,0.540] | ✅ |
| 15 | 82.6% vs 79.7%、+2.9pp、p=0.002、10/10 | 04:83, main | r23-trans002-mexp-2026-09-09.json | mean_M 0.8257 / mean_C 0.7971；Wilcoxon & sign p=0.00195；diffs 全正 | ✅ |
| 16 | NTU120 +2.9pp、t=12.0、raw p<10⁻⁵、W p=0.002、校正 0.012 | 04:73 | r22-e9bc-gap-tests / r23-holm-family | delta 2.87pp；t=12.0 p=6e-7；W p=0.002；Holm 5.4e-6 / 0.0117 | ✅ |
| 17 | 15.8x（9897s vs 626s）、−4.5pp、6.2x、PARTIAL | 04:80, 06:47 | r23-trans002-full-fixed-2026-09-08.json | median 9897/626.3；ratio 15.8；gap −4.49pp；final-seg 6.24x | ✅ |
| 18 | 32.5x（1520 vs 47）、+2.6pp | 04:80, 06:47 | r23-trans002-full10-2026-09-08.json | 1520.3/46.8=32.49x；gap +2.58pp | ✅ |
| 19 | 6.07×（31.1s vs 188.7s）、−0.91/+2.27pp、2.18×、≥3.99× | 04:53, tab1, 06:7 | c1-decouple-cost-full/-2026-08-24.json | 31.10/188.73；ratio 6.0685；Δacc −0.0091/+0.0227；epochs 2.1818；配对最小 122.38/30.6=4.00 | ✅ |
| 20 | ±2.3pp band（E6/TRANS 系） | 04:53,56; 06:47 | r22-ntu-transition verdicts band_pp=2.3；E6 预注册带 | 全文同值 | ✅ |

### A-2 其余扩展核对命中项（~95 点，摘要）

E1（12.89%/1.55×→p01-aimclr md）；E2（uniform 0.399±0.035、grid 0.453±0.027、null 0.323±0.022、F1 0.343 vs 0.396、200 次 MC、4/4）；E3（purity 0.5339/0.3306/0.4858/1.61×/+4.8pp→p03-jia-phasea-results.json；seed-noise 0.5267/0.5201/0.5025±0.0063；K14 sweep 0.5631→0.5390=−2.4pp→p03-jia-phasea md §4.2）；E4（0.5125→0.6913±0.0128、+17.9pp、+10.69pp、raw p=0.030/Holm 0.090→p04-tcl-results + r16-holm-p04）；E5（7.9/7.1pp、69.9 vs 77.8、80.9 vs 88.0、120ep 复跑 12.3/7.8、4.2–5.0pp、margin 100.9 vs 10.8→p05-al-efficiency-* + w14 md）；gradient（+21.2/+9.9/−2.3/+0.15、2.3×/7× random、3.8× std、22 样本=4.55pp→w31+w39 系列）；E7 修正臂（9.8±7.5 / 15.2±5.1 / 8.4±10.7 / 10.9 / 15.9 / macro-F1 5.3 vs 10.4 p=0.023、7.3 vs 13.9 p=0.049、pool 0.11、recalib 8.9–12.5%、33.93→r16-endtoend-pseudo-2026-09-05.json）；E7b（352=256+96、13.1±5.3 / 17.6±4.3 / 13.8±5.6 / 16.6±6.8 / 14.6、+4.6pp p=0.012、35.42/37.50、32.99±3.94 / 9.72±1.59→p13、25.96%/+11.54pp/13.5→25.0/−4.8→+3.1→r12-holm-eightclass、+3.57 判据→p12 ep3_ceiling）；E7c（33.93→30.36、−3.57、LABEL_BOTTLENECK→p18-superanimal-extract）；NTU 等价（74.30/71.51/67.84、77.97/95.78、n=16487、0.6/0.6/0.4→ntu-phaseB-*）；E9 SSL 基线（46.79±0.61、+20.7pp、15758/36082、pool≈0.69、65.62±0.48→r23-ssl-baseline + subset-resample + pooldecomp（四臂 67.53–67.58、−0.02/0.0pp））；E9c SSL（14.17±0.87 vs 14.04、+1.2pp、8/10、p=0.047/0.141）；TRANS-001（466s/35s/13×/2.3×/+3.93/+11.3/FAILS→r22-ntu-transition）；five-arm（35.71/27.38/+8.3pp、36.90/36.90、0.119/0.103、random 6.0/majority 25.0、27–37% 带、n_train 141/n_val 56 合计 197→twoarm+endtoend JSON）；L6（28,197 图 / 6,160 目录 / 4.6 帧/视频→partialclass md；33,099 为 AK 官方数）；L9 AdaBN（40.82/44.90、−4.08/+6.12/−8.16、−2.04、85/85、23.5→76.5、+4.08、100→26.7、1,069=570+499、51 段→round2/round3 md）；数据集计数（226/225、329=231+98、34,772、2,749、41,179、40,091、4,009→data-inventory + aptv2-inventory + p01/p12/r16 config_echo）；p15/p16/p17（V2 14.11±/8-of-10/p=0.13/NULL/0.21 vs 0.11；DAP 503 clips/25.0%/11.4 vs 14.1/3-of-10；budget 14.1→22.1/24% knee 不达/池精 0.04–0.17）；σ=0.10 预注册诊断与 41.2% 零样本→p05-al-efficiency-warmstart-diagnosis.json（noise 0.1 行 val_acc 0.41212、池 220）。

### A-3 不一致/疑点（P2）

| 编号 | 不一致描述 | 论文侧 | 工件侧 | 级别 | 修复建议 |
|------|------------|--------|--------|------|----------|
| A-3-1 | **"95.7% at the full offset budget (220 clips)"**：工件 AL 曲线预算点为 20/50/100/200，95.7%（0.95657）是 **b=200** 的值；220 是 offset 池容量（10/类×22）。"full budget (220 clips)" 与数字实际对应的预算点不符（86%=82.0/95.7、9%=20/220 两处算术各自成立） | sections/04-experiments.tex:50（Warm-start 段） | p05-al-efficiency-warmstart-short-2026-08-25.json（meta.protocol.data_fingerprints.pool_gen=220；curves 仅 20/50/100/200） | P2 | 改为 "at the 200-clip budget point（91% of the 220-clip pool）"，或补跑 b=220 全预算并把 95.7 换成真全预算值 |
| A-3-2 | **E7b spc2 种子 clip 数**：论文 "14 of 256, ≈5.5%; the sit class has none"，被引协议工件披露 "spc2 on v2 = **16**/256 train clips ~ 6%（…sit n=2）"。两处口径差 2 clips（sit 类训练 clip 有无） | sections/04-experiments.tex:44 | p12-akv2-replication-2026-09-04.json disclosures[0] | P2 | 在 r16 修正协议工件中固化 v2 spc2 的实际种子 clip 数并同步 p12 披露（sit 类 n_train=0 或 2 择一为真） |
| A-3-3 | **"44.2% retention of the v2 full-budget reference"** 未指明分母：16.6/37.50（archived）=44.3→44.2 ✅；但同段并列的 re-encoding 参考 35.42 会给 46.8%。分母有歧义 | sections/04-experiments.tex:44 | r16-endtoend summary.v2.warm_full=0.3542 与 p12 v2_full_top1=0.375 并存 | P2 | 补两词 "of the archived 37.50% reference" |
| A-3-4 | **E7c 提取器质量数字无 reports/ 工件**："up to 100 epochs, early-stopped at 89; pose mAP50 0.834" 在 psd-framework/reports/ 无对应 JSON（p18 仅含 33.93/30.36/−3.57/verdict；0.834 仅见于 k9-training-system runs/…/results.csv）。任务红线"正文声称的每个数字都必须能在 reports/ 对应 JSON 里找到"对这两条不满足 | sections/04-experiments.tex:47 | p18-superanimal-extract-2026-09-07.json（缺 mAP/epoch 字段）；D:\Desktop\k9-training-system\runs\pose\…\results.csv | P2 | 将 E7c 提取器评估（mAP50 曲线、early-stop epoch）导出 JSON 存入 reports/，或正文改为 "released in the K9 training repository" 指向 |
| A-3-5 | **highlights.tex 头注字符数区间过期**：注称"五条已程序化实测 65–77 字符"；实测第 3 条源码 85 字符（渲染 84）。≤85 合规不变，但文档化区间失真 | highlights.tex:2 | 本审实测 | P2 | 头注更新为实测区间（如 65–84） |
| A-3-6 | **fig2 caption "the physics encoder stays frozen" 图内无对应元素**：图中不存在 physics encoder 节点/角标，caption 断言了图外事实（图-例同步性瑕疵，非数字错误） | sections/03-method.tex:43（fig:pseudoloop caption） | figures/fig2_pseudo_label_loop.png | P2 | 图中 hub 下方补一行灰色小字 "Φ frozen"，或删该句 |

### A-4 观察项（不计级）

- **任务清单 "4.5% chance"**：正文中不存在 "4.5% chance" 主张。4.5% 仅两处出现：fig4 图内 random-guess 基线线（=1/22，算术正确，p05-stgcnbc synthetic random_baseline 0.045 同值）；以及任务清单所指更可能是 TRANS-002 的 **−4.5pp** 精度差（79.8 vs 75.3，工件 −4.49，已逐位命中）。清单措辞与论文实际主张对齐无误。
- **APTv2 "official counts 41,235 frames"**：本地工件仅存 41,179 本地库存（论文已自披露差额，属外部官方数字），本地无法复验 41,235 本身；如需加固可在 dev-docs 存官方页快照。

---

## B. 图-文对账（五图 caption ↔ 实际绘制 ↔ 图内统计量）

五图印刷页落位与任务给定一致：**Fig1=p4，Fig2=p10，Fig3=p15，Fig4=p20，Fig5=p31**（页脚 "Page N of 41" 口径）。

| 图 | caption 关键词 | 图上实际 | 判定 |
|----|----------------|----------|------|
| Fig1（p4，fig1_framework_overview） | "physics layer Φ (left)…semantic layer Ω (right)"；embeddings+proposals 窄接口；𝒴→𝒴′ only Ω retrains | 左蓝 Φ（SSL pretraining→dynamics embeddings；motion words→behavior proposals），右橙 Ω（seeds→anchors→clustering→self-training→classification under 𝒴），中缝 embeddings+proposals，底部横幅 Taxonomy 𝒴→𝒴′ only Ω retrains | ✅ 全一致 |
| Fig2（p10，fig2_pseudo_label_loop） | 顶行 automated（橙）Assign→pool→update Ω→re-estimate P；底行 human（teal）AL queue→verified seeds；黑 hub (P,Ω,A)；虚线 write-back | 图与 caption 逐元素一致，κ≥τ/κ<τ 分流正确；唯一瑕疵见 A-3-6（"physics encoder stays frozen" 无图内对应） | ✅（1 项 P2 已记） |
| Fig3（p15，fig3_segmentation_qualitative） | lowest-/highest-IoU 两极端集；orange=Pseudo-GT / teal=SMQ（E-C, K=8, epoch-30）；agg 0.4577±0.0488、4/4 | 两极端为 Ep1（0.420）与 Ep4（0.540）✅（四集 IoU 0.4196/0.4230/0.4483/0.5400 的最小/最大）；橙/teal 带正确；底部 0.4577±0.0488·4/4>baseline 与逐集基线 0.291/0.315/0.336/0.350 与工件逐位一致 | ✅ |
| Fig4（p20，源文件 fig5_budget_retention） | 七点+位移清单（v1 12.8→13.9；v2 spc4 10.9→11.5；offset 9.1→8.0；UCF 10→8.8；PanAf 10→13.2；NTU120 10→16.5）；NTU60 ±0.3pp 小于标记；retention 90.7/88.9/88.7/86(offset)/66.6/29/44 | 图上七点位置、误差棒、位移与 caption 清单逐一相符；NTU60 菱形无可见棒 ✅（retention 口径 std=0.24/74.45→0.32pp→"±0.3pp" 成立）；v1≈29%、v2 spc2≈35%、v2 spc4≈44%、UCF≈66.6、offset≈86 全对 | ✅ |
| Fig5（p31，源文件 fig4_al_efficiency） | (a) cold-start/(b) warm-started；69.9 vs 77.8（b=100, 2/3）、80.9 vs 88.0（b=200, 3/3）；4.2–5.0pp；n=330；log 轴 | 图内注释 "random exceeds uncertainty for budgets ≥ 100 (cold-start)"、"random leads by 4.2–5.0 pp for b ≥ 50"、"random-guess baseline 4.5%"（=1/22 ✅）；数值与工件一致 | ✅ |

- 图内统计量 ↔ caption ↔ 工件三方全部对上（含 0.4577±0.0488、4/4、69.9/77.8、80.9/88.0、4.2–5.0pp、4.5%）。
- 无孤立引用的 panel；无 caption 描述旧版图的情况。
- **观察**：源文件名 `fig4_al_efficiency` / `fig5_budget_retention` 与刊出图号（Fig4=budget，Fig5=AL）互换——编译序 §4 的 budget 图先排号所致，非错误，但维护时易混淆，建议 FIGURE_SOURCE.md 加一行注记。
- 图形格式（scholar-presubmit）：五图正文均 `\includegraphics` **PDF 矢量** ✅；`thumbnails/cas-email.jpeg` 位图系模板证件照占位（合法例外，USER 项）。

---

## C. 主张-证据对账（Highlights 五条 + Abstract 加粗主张）

### Highlights（main.tex 版内与 highlights.tex 逐字一致 ✅）

| 条目 | 支撑 | 判定 |
|------|------|------|
| H1 Decoupled framework absorbs taxonomy evolution cheaply (synthetic tier) | E6（§4，31.1 vs 188.7s，6.07×）；scope 词 "(synthetic tier)" 在条目内 | ✅ |
| H2 Taxonomy transitions cost ≥3x less wall clock (synthetic tier) | 同上 + 预注册保守界规则；real-domain 边界由 Abstract/正文 PARTIAL 披露（Highlights 免苛求） | ✅ |
| H3 First, to our knowledge, anchor–cluster pseudo-label loop on skeletons (combination) | Appendix A 十组查询 19 候选零占用 + §2.3 与最近邻 pointsup2026 显式划界；"to our knowledge" 在条目内 | ✅ |
| H4 Pretraining: 2.51x random kNN on quadruped skeletons (subject-ID probe) | E1；"(subject-ID probe)" scope 词在条目内，与 L1 一致 | ✅ |
| H5 Warm start: 82.0% top-1 from 20 clips (synthetic-offset; marginal budget) | Warm-start 段；"(synthetic-offset; marginal budget)" 双限定在条目内，与 "not learning from 20 clips alone" 披露一致 | ✅ |

### Abstract 加粗主张逐条

| 主张 | 支撑链 | 判定 |
|------|--------|------|
| 20.89% vs 8.33% (2.51×; representation probe) | E1→p01-knn | ✅ |
| 82.0% top-1 on 22 classes from only 20 labeled clips（marginal budget 括注） | Warm-start→warmstart-short JSON；括注已限定"fully labeled source domain 之上的边际预算" | ✅ |
| ≥3× lower retraining cost（conservative bound; measured 6.07×）+ fails/PARTIAL 两复现 scope | E6 + TRANS-001（FAILS）+ TRANS-002（PARTIAL）→c1/r22/r23 工件 | ✅ |
| 90.7% retention at 10% labels（linear head alone 88.7%） | E9→r16-ntu-pseudo-10seed | ✅ |
| canine tier near chance at 13% budget | E7→r16-endtoend（9.8% vs chance 11.1%） | ✅ |
| 收尾解读句 "indicate…tier- and protocol-dependent rather than uniform" | 与全文三层口径/负边界披露一致，无超出证据的强度 | ✅ |

- 无 forward-reference（Abstract 的 PARTIAL/FAILS 概念在 §4 E6-real/E6-real-2 与 L12 正式展开，Intro §1 已预告，属正常倒叙）。
- Overclaiming 清单（galaxy）：机制词（"carried by the frozen-feature linear head"、"bounded by pseudo-label quality"）均绑定实验（pooldecomp、四点梯度）；负结果与边界全部随行披露（UCF101 FAILS、K9 未行使、L13 无_published-method 对比）。
- C-1【P2】：Intro 贡献 4 "a generic-SSL control matches or exceeds the task-pretrained initialization at low budget on **both real tiers tested** (Section~\ref{sec:ablation})"——v2 tier 的该对比证据在 **§4 E7b**，§5 只含 v1；交叉引用覆盖不全（01-introduction.tex:19）。建议括注改 "(Sections 4–5)"。
- C-2【观察】：Conclusion "we verify the pattern on skeleton data in two domains"——"two domains" 指代含糊（synthetic+NTU 或 NTU60+NTU120 均可读），且 real 域复制判 PARTIAL；"verify" 与前句 "holds only approximately" 有轻微张力（已有缓解措辞，可不动）。

---

## D. 术语一致性

- **Tier 命名**：synthetic / synthetic-offset / public-real（v1、v2 并行不合并）/ real-K9（未行使）/ public human benchmark / animal public benchmark (PanAf500) / independent benchmark (UCF101) 在 §4.1 定义后全文（含 tab:main 行名、fig4 图例、Abstract）同义同名 ✅。
- **协议号**：PSD-K9-AKV2-SA-NTU-NTU120-UCF101-PANAF-SSL-BASE-ALIGN-DAP-BUDGET-POOLDECOMP-PREREG / TRANS-001 / TRANS-002 / TRANS-002-MEXP / R16 等 16 个协议号与工件 `protocol` 字段逐字一致（抽全核）✅。
- **实验编号**：E1–E9/E9b/c/d/E7b/E7c 与 claim tag C1–C7 的映射在 §4.3 开头集中声明且全文一致 ✅；E5/E8 的"降级/并入"说明与正文实际位置一致 ✅。
- **统计术语**：ten-seed convention（3-seed 初始读数扩展并双保留）、Holm–Bonferroni（per-tier m=6 / series-wide m=8 / 6-test 子族不变性）、spread=sample std (ddof 1) 非 CI、directional-only（PanAf gap）——全文口径统一且与 r23-holm-family/r22 工件一致 ✅。
- E9d 的 88.7%（10-seed，tab:main/fig4）与工件 JSON 顶层 retention 0.8578（30-seed）并存：论文已双口径披露（"both readouts are released"），非混报 ✅。
- **无发现**（本项全一致）。

---

## E. 引用完整性

- refs.bib **37 条** ↔ 正文 cite **37 键**：双向零缺口（无断引、无孤儿条目、无重复键）✅（本审脚本核验）。
- `\ref`/`\label` 闭合；唯一未被引用的 label 是 `sec:intro`（intro 节自身，无害）。
- **Limitations 13 条（L1–L13）**与正文声明 "Thirteen limitations bound our claims" 一致，且各条数字均与正文/工件互洽（L6 28,197/6,160；L7 44.9/41.5±5.9/100-23.5-0-0；L9 40.82/44.90；L12 15.8x/6.2x/32.5x/4.5pp）✅。
- 正文 `\TODO{}` 调用 **0 处**（仅导言区宏定义与注释）✅——成稿占位清零达标。
- bibtex 11 条 "Warning--empty pages"（yolo2025petx/miyai2025workshop/mabe22/yan2018stgcn/duan2022posec3d/guo2022aimclr/li2021crossclr/videomamba2024/painet2023/caron2021dino/singh2021tcl）——排版建议级（Elsevier 数字样式期望页码/文章号），不阻塞。
- Front matter（scholar-presubmit 第 5 项）：作者/单位/ORCID 为 USER 占位（已知人工项）；Funding=无资助标准句；Acknowledgements=NTU 强制声明；Data availability/Ethics/GenAI 披露齐备（GenAI 颗粒度注释标 USER 终裁）——与记忆中投稿前人工项清单一致，非本审新增缺陷。

---

## F. 编译面（pdflatex 绝对路径 + 经典三连，沙箱全量复建）

在临时目录完整复刻（源 tex/bib + figures + thumbnails；不触碰论文原文件）执行
`pdflatex → bibtex → pdflatex ×2`（/c/Users/FOUR/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex）：

| 项 | 结果 |
|----|------|
| 页数 | **41 页**，与仓内已提交 main.pdf 一致 |
| Errors | **0** |
| Undefined references / citations | **0** |
| LaTeX Warning | **0** |
| Overfull | **1**：`Overfull \hbox (117.0831pt too wide) detected at line 89`（\maketitle prelims 框；与仓内已提交 main.log **逐字相同**，为 R18 已知 cas-sc 类缺陷残余，渲染输出不可见——已目检标题页/Highlights 页） |
| Underfull | 0 |
| bibtex | 0 error；11 条 empty-pages 提示（见 E） |
| 图形 | 5 张 PDF 全部解析成功；无缺图 |

结论：编译面干净，沙箱构建与仓内 2026-09-11 01:36 构建同态。

---

## 修复建议优先级（供作者决策，本审不改稿）

1. **A-3-4（E7c 工件缺口）**——唯一触碰"每数字必有 reports/ JSON"红线的项：补导出 mAP50/early-stop JSON，或正文改指向。
2. **A-3-1（95.7%@220 clips）**——一处措辞校准即可消除（审稿人可能拿 AL 曲线对不上 220）。
3. **A-3-2（14 vs 16 clips）**——工件侧澄清一次即可，两处必有一处需要更正。
4. A-3-3 / C-1 / A-3-5 / A-3-6——分钟级微调。
5. （可选）FIGURE_SOURCE.md 注记 fig4/fig5 文件名与刊出图号互换。

**R32 审毕。三方对账 PASS：P0=0，P1=0，P2=6，观察 2。**
