# 2. Related Work（英文初稿 · P0.6）

> Owner: `docs/paper/related-work.md` · W5 窗口 2026-08-23 · 状态: 初稿 v0.4（两轮评审 + W17 文献终审标记同步，见文末审查记录与修订历史）
> 引用纪律: 全部条目溯源 `dev-docs/research/RESEARCH_LITERATURE.md`（17 篇池）与 `NOVELTY_CHECK_YAOQING_JIA.md`；题录不全者标 `[CITATION-NEEDED]`。
> 写作规范: 主题式综合（禁逐篇罗列）；句均长 ≤25 词；被动语态慎用；每小节以差距句收尾。

---

Animal behavior recognition supports applications ranging from livestock welfare monitoring to working-dog training. Training a single guide dog costs more than USD 12,000, and more than half of candidate dogs fail to graduate, which motivates objective, automated behavior assessment [Grimm, "Can science build a better working dog?", Science feature]. Deployed systems already combine wearable sensors with classical machine learning for canine posture estimation [PLOS ONE 2023, PMC10284380], yet the field lacks a recognition framework that works when labels are scarce and evaluation criteria keep evolving. We review three bodies of work that bear on this gap: animal behavior understanding, skeleton-based self-supervised learning, and pseudo-label-driven semi-supervised learning.

## 2.1 Animal Behavior Understanding

Existing animal behavior pipelines divide into appearance-based, pose-based, and sensor-based lines.

Appearance-based methods classify behavior directly from RGB frames or video. YOLO-PetX adapts a detection backbone to anomalous dog behavior ["YOLO-PetX: Enhanced YOLO-Based Recognition of Abnormal Dog Behaviors in Intelligent Pet Care Applications", IEEE CEECT 2025; CITATION-NEEDED: author list + DOI]. TP-CanineNet applies temporal contrastive learning with pseudo-labels to canine separation-anxiety video [MDPI Animals 2025]. These methods inherit the strengths of image recognition but depend on large annotated corpora, and their accuracy degrades under background change, occlusion by handlers, and breed variation in appearance.

Pose-based methods first extract keypoints, then classify dynamics. DeepLabCut [DOI 10.1038/s41593-018-0209-y] and SLEAP [DOI 10.1038/s41592-022-01426-1] made markerless animal pose estimation practical. Building on such inputs, ASBAR unifies skeleton-based animal action recognition with PoseConv3D and reports 75.3% Top-1 on primate benchmarks [github.com/MitchFuchs/asbar, eLife 2024]. BCST-GCN adapts ST-GCN-style graph convolutions with bidirectional cross-attention to pig behavior and reaches 94.43% overall accuracy on a private four-behavior dataset [DOI 10.3389/fvets.2026.1782396]. Hierarchical representation learning of dog behavior from single-view 3D pose has also been explored [Miyai et al., NeurIPS 2025 Workshop on AI for Animal Communication]. Skeleton representations are compact, background-invariant, and robust across individuals—properties that suit low-resource settings—but every method above is trained fully supervised on a fixed label set.

Sensor-based methods use accelerometers or IMUs. They capture coarse daily activities reliably [PMC10284380] but cannot resolve fine-grained behaviors that differ mainly in body configuration rather than global motion.

**Gap.** Current animal pipelines assume a fixed taxonomy and supervision-heavy training. None of them exploits unlabeled video streams at scale, and none is designed so that changing evaluation criteria can be absorbed without re-annotating the corpus.

## 2.2 Skeleton-Based Self-Supervised and Unsupervised Learning

Self-supervised representation learning for skeletons is mature in the human domain. ST-GCN established graph convolutions over joint topologies [arXiv 1801.07455], and PoseConv3D recasts keypoints as heatmap volumes for 3D convolution [Duan et al., CVPR 2022, arXiv 2104.13586]. Among purely unsupervised objectives, AimCLR maximizes asymmetry across extreme augmented views with an InfoNCE queue and reports 79.18% on NTU60 cross-subject under the linear-evaluation protocol (frozen encoder + FC-softmax classifier; three-stream joint/motion/bone fusion; the paper body itself reports 78.9%, with 79.18% being the authors' re-measured score of the released checkpoints) [AAAI 2022, DOI 10.1609/aaai.v36i1.19957; repo github.com/Levigty/AimCLR; protocol verified against the original paper, see reports/ntu-phasea-2026-08-24.md]; its journal extension AimCLR++ further improves NTU cross-subject accuracy to 77.2% [Pattern Recognition 2024; repo github.com/Levigty/AimCLR-v2]. Cross-view contrastive objectives form a parallel self-supervised family for skeletons [e.g., CrosSCLR, "3D Human Action Representation Learning via Cross-View Consistency Pursuit", CVPR 2021; repo LinguoLi/CrosSCLR], and we treat the encoder choice as replaceable within our framework; AimCLR serves as the reference physics-layer instantiation rather than a load-bearing design commitment. For temporal structure, unsupervised segmentation methods quantize motion into discrete vocabularies; Skeleton Motion Words (SMQ) derives behavior-proposal boundaries from motion-word sequences [Gökay et al., ICCV 2025, arXiv 2508.04513].

Two observations motivate our treatment of this literature. First, virtually all benchmarks, augmentations, and graph templates assume the human body plan; transferring these objectives to quadruped kinematics requires adapting joint topology, augmentation priors, and view construction. Second, pretraining and segmentation are studied separately, although a recognition system needs both: representations without proposals cannot localize behaviors, and proposals without semantics cannot name them. State-space backbones such as VideoMamba [ECCV 2024] and Mamba-MSQNet for animal action recognition [Fazzari et al., Ecological Informatics 2024, art. 102955] offer efficiency alternatives but do not address label scarcity either.

**Gap.** Self-supervised skeleton machinery exists almost exclusively for humans, is validated on human benchmarks, and is decoupled from unsupervised temporal decomposition. Animal-domain adaptation remains largely unexplored; our repository-scale survey found no published self-supervised animal-skeleton recognition pipeline (systematic internal survey; to be migrated to supplementary material before submission).

## 2.3 Semi-Supervised, Few-Shot, and Pseudo-Label Learning

Semi-supervised skeleton action recognition in the human domain relies mostly on consistency regularization and teacher-student schemes, including momentum-contrastive teachers ["Momentum Contrastive Teacher for Semi-Supervised Skeleton Action Recognition", IEEE TIP; CITATION-NEEDED: author list] and graph-representation alignment ["GRA: Graph Representation Alignment for Semi-Supervised Action Recognition", IEEE TNNLS; CITATION-NEEDED: author list]. These methods regularize predictions under perturbation but receive no external guidance about which classes matter. MAC-Learning introduces anchor-based contrastive learning, where "anchors" are contrastive sample pairs selected to structure the embedding space [TPAMI 2022; repo github.com/1xbq1/MAC-Learning]; it does not maintain seed-annotated anchors that supervise cluster assignment, and it includes no iterative pseudo-label loop. Few-shot approaches such as PAINet learn episode-based metric classifiers [ICCV 2023; repo github.com/starrycos/PAINet] and likewise lack a mechanism for expanding labels beyond the support set.

Vision foundation models (VFMs) supply another route to low-shot recognition [e.g., DINO, Caron et al., ICCV 2021]. Skeleton-to-image encoding maps skeletons into images consumed by VFM-pretrained encoders [arXiv 2603.05963], demonstrating that foundation-model features transfer to skeleton data. This line uses VFMs for representation only; it does not close the loop from a handful of annotated seeds to full taxonomy coverage. In the animal domain specifically, TP-CanineNet is the closest use of pseudo-labels, operating on RGB video rather than skeletal dynamics [MDPI Animals 2025].

**Positioning against the three nearest neighbors.**

| Work | Domain | Seed-annotated anchors | Cluster + pseudo-label iteration | VFM-equivalent backbone | Output |
|------|--------|------------------------|----------------------------------|--------------------------|--------|
| MAC-Learning [TPAMI 2022] | Human skeleton | ✗ (contrastive pairs) | ✗ | ✗ | Representation |
| Skeleton-to-Image [arXiv 2603.05963] | Skeleton representation | ✗ | ✗ | ✓ (VFM encoder) | Representation |
| TP-CanineNet [MDPI Animals 2025] | Canine RGB video | ✗ | Partial (video pseudo-labels) | ✗ | Classification |
| **Ours** | **Quadruped skeleton stream** | **✓ rule-engine seeds** | **✓ confidence-filtered, iterated** | **✓ physics-layer self-supervised encoder** | **Recognition + segmentation** |

To the best of our knowledge, transferring the combination of seed-annotated anchor learning, prototype clustering, and iterated confidence-filtered pseudo-labeling to temporal-skeleton recognition has not been reported. Our claim of firstness rests on a systematic GitHub-scale survey covering ten query groups and fourteen candidate works with zero occupancy [internal survey; arXiv/Google Scholar re-verification is scheduled before submission]. We state this claim with that explicit boundary rather than as an absolute assertion.

**Gap.** Three absences compound across the three lines reviewed above. Animal behavior understanding offers no label-efficient path and no mechanism for taxonomy change (§2.1). Skeleton-based self-supervision supplies representation machinery that has never been adapted to quadruped data, and treats segmentation as a separate problem (§2.2). Semi-supervised learning supplies pseudo-label machinery whose components—seed-annotated anchors, prototype clustering, iterated confidence-filtered labeling—have never been assembled for temporal-skeleton recognition (§2.3). No existing method couples a label-free physics layer (self-supervised dynamics modeling plus unsupervised segmentation) with an anchor-guided semantic layer that grows from tens of seeds to a complete, revisable taxonomy. This coupling is precisely what evolving evaluation criteria demand, and it defines the contribution of this paper.

---

## 自审记录（auto-review-loop 清单人工执行 + check-citations 核查）

| 检查项 | 结果 |
|--------|------|
| 主题式组织、无逐篇罗列 | ✅ 三小节均为"路线综述→代表工作→差距句"结构 |
| 每小节差距句铺垫本文动机 | ✅ 2.1 固定标签集+重标注成本 → 2.2 人类域集中+预训练分割割裂 → 2.3 无锚点引导闭环 → 总 Gap 对应 C1/C2 |
| 词数 | 正文 1074 词（≥800 达标） |
| 句长 ≤25 词为主、被动语态克制 | ✅ 抽查通过 |
| 首次性主张带边界声明 | ✅ 显式写明内部调研 + Scholar 终审保留（对齐 W1 §5 边界声明） |
| 引用全部有真实出处或显式 [CITATION-NEEDED] | ✅ W17 终审后剩 3 处缩窄标记（YOLO-PetX / TIP Teacher / GRA 的作者列表+DOI 级待补），其余全部补全为官方渠道题录；无凭记忆生成条目 |
| Science 特稿引用措辞 | ✅ 用作产业背景 motivation，未当作研究证据 |
| AI 腔检查 | ✅ 无 "delve/moreover/furthermore 堆砌"、无空泛开头 |

### 本轮审查实际发现并已修复（2026-08-23）
1. ❌→✅ "79.18% **linear-probe** accuracy"：协议名系池外未验证细节（材料仅记 NTU60 参照成绩）——已删除协议限定词并加"待对照原文复核"标注。
2. （outline.md 同轮发现）❌→✅ §4 Method 节 "AimCLS" 笔误 → 已改 "AimCLR"。

### 待办（移交后续窗口）
- [x] 补全待补题录（W17 终审 2026-08-24：11 条完全解决，3 条缩窄为作者列表级；溯源明细见 review-log.md 终审节）
- [x] **补三个数据集原始论文题录**（InterPet4D / Animal Kingdom / APTv2）——✅ W17 已补齐官方 BibTeX，R5 解除
- [ ] 投稿前 Scholar 终审补 3 处作者列表：YOLO-PetX（CEECT 2025）、Momentum Contrastive Teacher（TIP）、GRA（TNNLS）的作者与正式 DOI
- [ ] TCL 数字复核：method.md 中 82.7%（10% 标注）/88.6%（全监督）沿用池内记载，投稿前对照 CVPR 2021 原文核验
- [ ] **内部引用迁移**：正文 2 处 "(internal survey...)" 措辞在投稿时替换为 Supplementary Material 正式引用（内部报告整理为附录）；"[internal survey; arXiv/Google Scholar re-verification...]" 同规则处理
- [ ] SMQ 与 AimCLR 的具体数字（79.18%/77.2%/75.3%）在投稿前对照原文复核一遍（✅ AimCLR 79.18% 已于 2026-08-24 W9 核实并改写正文口径：linear eval/xsub/3s，论文正文 78.9%、79.18% 为 released-model 复测——见 `reports/ntu-phasea-2026-08-24.md`；SMQ 75.3% 与 AimCLR++ 77.2% 仍待核）
- [ ] 若 P0.2-P0.5 结论变化，回改 2.2/2.3 的差距句强度；C6 的 "≤20%" 措辞按 E4 实测收紧

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1 | 2026-08-23 | W5 窗口初稿：三小节 + 三近邻表 + 自审一轮 |
| v0.2 | 2026-08-23 | 对抗评审加固：§2.2 补跨视角 SSL 谱系承认句（防"文献覆盖不全"指控）+ 总 Gap 改三缺口合成结构 + 待办增补数据集论文引用项 |
| v0.3 | 2026-08-23 | 第二轮对抗评审：元数据一致性修复（状态行/词数/C-N 计数）+ 内部引用迁移策略入待办 |
| v0.4 | 2026-08-24 | W17 文献终审：9 处 [CITATION-NEEDED] 中 6 处补全（Science 特稿/层次化犬行为 workshop 论文/PoseConv3D/CrosSCLR/SMQ/Mamba-MSQNet），3 处缩窄为作者列表级（YOLO-PetX/TIP Teacher/GRA）；VFM 句补 DINO 代表引用；待办清单同步 |
