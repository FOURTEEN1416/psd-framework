# submission-package-final-2026-09-12 — 投稿包终稿（Pattern Recognition · Regular Paper）

> 取代 `submission-package-2026-08-26.md`（v1.0 历史版，留档不删）与 checklist-2026-08-26。
> 对齐基线：master `ccc2636`（GA v15 双行定稿版），42 页，pdflatex+bibtex 全链 0 错误 0 浮动
> 警告，.bbl 重跑零漂移。
> 本文件是投稿操作的单一真源；提交后如有 rebuttal 阶段改动，另起新日期文件。

## §1 Editorial Manager 上传四件套（规格已程序化核验）

| # | 件 | 路径/来源 | 规格核验 |
|---|----|----------|---------|
| 1 | LaTeX 源（含 PDF） | `docs/paper/latex/`（main.tex + sections/*.tex + refs.bib + **main.bbl** + main.pdf）+ `docs/paper/figures/fig1..fig5/*.pdf` | cas-sc `[a4paper,fleqn,review]`（PR 单栏双倍行距口径）；编译链 pdflatex+bibtex 0 错误；\TODO 清零 |
| 2 | Highlights | `docs/paper/latex/highlights.tex`（独立文件、文件名含 highlights） | 5 条，渲染态 62–84 字符（≤85 ✅），与 main.tex 版内环境逐字同步 |
| 3 | Graphical Abstract | `docs/paper/figures/fig_ga_graphical_abstract.pdf`（**v15 双行构图定稿版**） | MediaBox 实测 **531×131.04pt**（≤531×131 上限 ✅）；纯矢量（零嵌入位图）；色盲模拟 PASS |
| 4 | Cover Letter | 本文件 §2 全文（粘贴为 EM 文本或 PDF） | 数字对齐 ccc2636 终稿（2026-09-12 刷新） |

随附非强制件：suggested reviewers + COI 回避说明（§4）、declaration of competing interest
（EM 问卷生成 Word）、CRediT（已在源内 \printcredits）。

## §2 Cover Letter 终稿（数字对齐 ccc2636；替换 08-26 版）

> Dear Editors-in-Chief of *Pattern Recognition*,
>
> We are pleased to submit our manuscript "**A Physics–Semantics Decoupled Framework for
> Animal Behavior Recognition under Evolving Evaluation Criteria**" for consideration as a
> Regular Paper in *Pattern Recognition*.
>
> Animal behavior annotation is scarce and expensive, while operational evaluation criteria
> keep evolving—every taxonomy revision currently forces re-annotation and retraining. Our
> manuscript addresses this with three contributions:
>
> **(1) A physics–semantics decoupled architecture whose transition-cost claim is measured,
> bounded, and replicated.** We formalize evolving evaluation criteria as taxonomy
> transitions absorbed by the lightweight semantic layer alone. On the synthetic tier the
> decoupled rebuild costs **≥3× less wall-clock retraining than full-pipeline retraining at
> matched accuracy** (conservative bound backed by a measured 6.07×; accuracy differences
> inside the pre-registered ±2.3 pp noise band), and we pursue the claim into its failure
> mode: two pre-registered real-domain replications scope the result honestly — at
> backbone scale the cost saving is real (15.8× full / 32.5× at 10% budget) while matched
> accuracy holds only approximately (**PARTIAL**, disclosed as such), and the claim is
> never stated as domain-universal.
>
> **(2) The first transfer of image-domain anchor–cluster–pseudo-labeling to
> temporal-skeleton recognition, to our knowledge.** A repository-scale survey across ten
> query groups (nineteen candidate works, Appendix A) found zero prior occupancy of the
> three-part combination; an arXiv/CrossRef re-verification at submission time surfaced the
> closest neighbor (point-supervised skeleton segmentation), which we differentiate
> explicitly in Section 2.
>
> **(3) Low-resource behavior characterized as a tier-dependent law, not a universal
> promise.** A pre-registered cross-domain series spanning four public benchmarks and two
> biological kingdoms shows the end-to-end pipeline retaining **90.7% of full-budget
> accuracy at 10% of the labels on NTU60** (88.9% NTU120, 88.7% PanAf500 — an animal-domain
> benchmark), collapsing to 66.6% on UCF101, and staying near chance on the canine tier at
> a 13% budget — retention tracks full-budget linear separability, not budget arithmetic.
> Warm-started initialization additionally makes a **20-clip budget usable under
> distribution shift (82.0% top-1, 22 classes)**, and self-supervised pretraining reaches
> **20.89% k-NN top-1 vs. an 8.33% random baseline (2.51×)** on real quadruped skeletons.
>
> We believe this fits the journal's scope at the intersection of representation learning,
> temporal pattern segmentation, and low-resource recognition. In the spirit of transparent
> reporting, the manuscript discloses its boundaries explicitly: behavior-level evidence is
> tiered and caliber-labeled throughout; uncertainty-based active-learning sampling is
> reported strictly as an exploratory negative finding confirmed across three independent
> runs; real-domain validation covers a single species family (canids) with 20/24 effective
> supervision channels; one public-real result carries severe class imbalance disclosed
> per-class; all experiments ran on consumer-grade hardware; no head-to-head comparison
> against any published method is claimed; and the target working-dog tier is pre-registered
> for a follow-up study rather than exercised selectively here. None of these boundaries
> undermines the central claim—they are precisely what the decoupling design absorbs.
>
> This manuscript is original, has not been published previously, and is not under
> consideration elsewhere. All authors have approved the submission and declare no
> competing interests. The study uses exclusively publicly available datasets; no new
> animal experiments were conducted by the authors.
>
> Thank you for your consideration.
>
> Sincerely,
> Zhimai Hou (Yunnan Police College; 3389812293@qq.com) and
> Ying Geng (Shanghai Jiao Tong University; gingying@sjtu.edu.cn),
> corresponding authors, on behalf of all authors.

## §3 终稿核对矩阵（2026-09-12 全部实测）

| 项 | 结果 |
|----|------|
| 编译 | pdflatex→bibtex→pdflatex×2：0 错误、0 浮动警告、42 页；**.bbl 零漂移** |
| 图 1–5 落位 | 印刷页 p4/10/15/20/31（pdftotext 断言）✓ |
| 图件 | 六图矢量+600dpi PNG 双导出；pdfimages 零位图；统一门禁（CVD/碰撞/穿线/MediaBox）PASS |
| GA | v15 双行版，531×131.04pt 实测 ✅ |
| Highlights | 62–84 字符 ×5 ✅，两处同步 ✅ |
| 摘要 | ~252 词（PR 无硬上限，建议 ≤250 已对齐） |
| 声明区 | DA/Ethics/Funding/COI/GenAI（单工具 GLM-5.3-Flash，重复碎片已修）/CRediT 全在位 |
| 引用 | 37 条全量复验闭环（含 plosone2023imu 卷期页、yolo2025petx DOI）；AI 腔扫描零命中 |
| 残留占位 | `\TODO{}` 清零；USER 占位词零残留（作者区真实姓名除外） |

## §4 Suggested Reviewers（EM 用，提交前用户复核名单）

- 沿用投稿包草稿 v0.1 名单（BOARD 09-07 登记：含 **AimCLR 作者 COI 回避建议**——
  郭亚栋组等骨架 SSL 近邻作者不提名或声明 COI）。
- 用户提交前在 EM 界面复核人数与单位分布（≥3 名、跨机构、近三年无合著）。

## §5 提交前剩余动作（按序）

1. **仓库同步**（等用户确认 push）：`git tag -f review-snapshot`（重锚至最终提交）→
   `git push origin master --tags`。正文 Data availability 公开承诺 dated commit chain +
   tagged snapshots，此步执行后承诺成立。
2. **EM 上传**（用户人工）：§1 四件套 + §4 名单。
3. 提交后：BOARD 登记投稿流水号；进入审稿静默期（rebuttal 弹药已备：fig3b 匹配细节图、
   双栏 7in 资产、GA 备选构图）。

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| final-1.0 | 2026-09-12 | 终稿版：CL 全面刷新（标题去 Low-Resource 对齐 R20；补 90.7%/四基准两王国/PARTIAL 复制；firstness 措辞对齐 highlights；署名对齐 R41 双通讯）；核对矩阵全实测；取代 2026-08-26 v1.0 |
