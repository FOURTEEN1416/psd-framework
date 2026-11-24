# Neurocomputing 投稿声明文件（占位待填）

> 目标期刊：Neurocomputing（Elsevier，elsarticle 直投）。
> 以下四份声明按 Neurocomputing/Elsevier 通用要求起草，`[方括号]` 处需用户
> 填入真实信息后即可粘贴进投稿系统。主稿 `latex/main.tex` 为 elsarticle 模板，
> 无需转排；highlights 已在 `latex/highlights.tex`。

---

## 1. CRediT authorship contribution statement（作者贡献声明）

**[Author 1 Full Name]**: Conceptualization, Methodology, Software, Investigation,
Formal analysis, Data Curation, Writing – Original Draft, Visualization.
**[Author 2 Full Name]**: [e.g., Conceptualization, Supervision, Writing – Review & Editing,
Resources]. **[Author 3 Full Name]**: [e.g., Investigation, Validation, Writing – Review & Editing].

> 填写说明：按实际贡献从 CRediT 十四类中选取（Conceptualization / Methodology /
> Software / Validation / Formal analysis / Investigation / Resources / Data Curation /
> Writing – Original Draft / Writing – Review & Editing / Visualization /
> Supervision / Project administration / Funding acquisition）。

## 2. Declaration of competing interest（利益冲突声明）

The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

## 3. Data availability statement（数据可用性声明）

The skeleton datasets used in this study are publicly available: NTU RGB+D 60
and NTU RGB+D 120 from the original authors' release, PanAf20K (the PanAf500
subset used here) from the eLife-associated release, and UCF101 from the
original publication. The measurement harness, configuration files, and all
per-seed results supporting the findings of this study are included with the
paper and in the released repository.

## 4. Acknowledgments（致谢，可选）

[Optional: funding sources, computing facilities, or individuals to thank.]

## 5. 提交系统待办（用户操作）

- [ ] 在 Neurocomputing 投稿系统注册/登录（Elsevier Editorial Manager）
- [ ] 填入作者信息（姓名、单位、邮箱、ORCID）
- [ ] 粘贴上方 CRediT / COI / Data availability 三段
- [ ] 上传 `latex/main.pdf`（或源文件 main.tex + 图源）
- [ ] 上传 `latex/highlights.tex` 对应的 5 条 highlights
- [ ] 粘贴 cover letter（`cover_letter.md`，已定稿 Neurocomputing 版）
- [ ] 提交前最后跑一次：`python scripts/p1_sprint/audit_next_paper.py` 与
      `python scripts/p1_sprint/check_latex_next_paper.py`（应全绿）
