# Editorial Notes — paper_manuscript_v1.md

> 本文件为从 `paper_manuscript_v1.md` 正文移出的中文编辑注记与 TODO，不属于论文正文。
> 移出日期：2026-11-20（排版整理）。论文正文（`paper_manuscript_v1.md`）仅保留英文内容。

## TODO（投稿前）

1. ✅ 依本稿在 LaTeX（elsarticle）中排版完成（2026-11-21）：`latex/main.tex`，17 页、0 error/0 warning/0 overfull，41 条文献全引用；机检脚本 `scripts/p1_sprint/check_latex_next_paper.py`；
2. ✅ 参考文献核实完成（2026-11-21）：[1] AimCLR=Guo et al. AAAI 2022、[2] CrosSCLR=Li et al. CVPR 2021（原条目作者/标题有误，已更正）；[3] 改为 hBehaveMAE（Stoffl et al., ECCV 2024）；[5] 改为 ProSL（Qin et al., Machine Learning 2025，原文描述已改正）；[15] PanAf20K（Brookes et al., IJCV 2024，PanAf500 为其子集）；[18] Cortes et al., ALT 2016；[39] Breiman 2001 补全卷期页；[41] 前作 under review；
3. 🟡 附录 A–E 从 evidence 目录填充；
4. ✅ 图 1–4 已生成（`docs/next_paper/figures/`），LaTeX 版已用 PDF 矢量版；
5. 🟡 Highlights 已成稿（`latex/highlights.tex`，5 条 65–74 字符，≤85 合规）；cover letter 待作者信息；
6. 🟡 CVIU 三动作（作者信息/COI/EM 提交）仍等用户

## 数据与图源（原正文横幅内容移入，2026-11-21）

- 正文所有数字均为实测：来源 `runs/p1_sprint/` 汇总；证据镜像 `reports/p1_sprint_evidence/adaptation/`。
- 图 1–4 位于 `docs/next_paper/figures/`，在正文首次引用处内嵌。

## 参考文献备注（原 References 节末中文注）

> **注（2026-11-21 更新）**：全部 41 条均已核实或为领域标准引文；原 [15]/[18]/[39]
> 占位与 [41] 前作占位已补全（见 TODO 第 2 条）。[1][2] 原条目作者与标题有误，
> 已按 Crossref 元数据更正。成稿时用 BibTeX 管理并保留 DOI。
