# docs/paper/latex/ — PR 投稿 LaTeX 工程（cas-sc）

> 决策 A（用户裁决 2026-08-26）：**els-cas `cas-sc`**——官方 Guide for Authors 直链模板族，单栏。
> 脚手架窗：W41 · 状态：骨架可编译、正文待装配。

## 文件地图

| 文件 | 作用 | truth 源 |
|------|------|---------|
| `main.tex` | 主装配文件（前置区已填：摘要/Highlights/关键词/声明框架） | introduction.md v0.3 + submission-package 报告 §2-§3 |
| `highlights.tex` | **投稿系统独立文件**（文件名含 "highlights"，强制要件） | 同 main.tex 版内环境，两处同步改 |
| `sections/01-06*.tex` | 六节装配桩（头部注释写明纪律） | 各 owner md（W36 v-final） |
| `refs.bib` | 题录库（已含官方仓 AimCLR BibTeX 首条取证） | outline.md §6 引用计划（W17 终审池） |
| `thumbnails/` | 官方 CAS 模板自带作者信息小图标（编译必需资产，随包分发） | els-cas-templates 官方 zip 原样拷贝（2026-08-26） |

## 编译实证与排障（2026-08-26 W41 当次运行）

- **冒烟全绿**: pdflatex → bibtex → pdflatex ×2 三段 exit=0；`main.pdf` 3 页 / `highlights.pdf` 1 页；日志零错误。
- `latexmk` 在本机不可用（MiKTeX 报缺 Perl 脚本引擎）——用经典三连即可，见顶部注释命令。
- 控制台出现 `security risk: running with elevated privileges` 为管理员终端运行 MiKTeX 的提示，无害可忽略。
- **排版纪律两条**（违者编译炸或成稿污染）：① pdflatex 非 Unicode 引擎，非注释文本禁 CJK；
  ② 前置区字段（shortauthors/author/ead/affiliation）被类内 LaTeX3 全展开，禁放自定义宏——占位纯文本，`\TODO{}` 仅用于正文区。
- `\TODO{}` 红色占位清零方式：`Select-String -Path *.tex,sections\*.tex -Pattern '\\TODO{'`。

## 类选项依据（实证，非记忆）

- `review` → cas-sc.cls L67 加载 setspace、L138 施加 `\doublespacing` ⇒ **精确满足 PR "single-column, double-spaced"**；
- 终稿清样期：去掉 `review` 改 `final`；
- 标题页超一页时加 `longmktitle`；匿名化不需要（PR 为 single anonymized review）。

## 编译

```powershell
cd docs\paper\latex
latexmk -pdf -interaction=nonstopmode main.tex      # 主稿
latexmk -pdf -interaction=nonstopmode highlights.tex
```

MiKTeX 已自带 els-cas-templates（cas-sc.cls / cas-common.sty / elsarticle-num.bst 实测在 TEXMF）。
投稿打包时将 cas-sc.cls、cas-common.sty、elsarticle-num.bst 一并复制进上传 zip（Elsevier 要求可编辑源文件自足）。

## 装配顺序建议与检查点

1. sections/01→06 逐块搬运（数字禁改写，对账跑 `scripts/gen_number_index.py`）；
2. tab1/tab2/tab3 成表（规范：最优加粗+方向符号+右对齐+无竖线无底纹）；
3. fig1-4 引入（`\graphicspath` 已指 ../figures/；caption 取 figure-specs.md 与 FIGURE_SOURCE.md v0.2）;
4. **页数校验**：编译后查总页数 ∈ [20,35]（含参考文献与附录）；不足 20 页会被期刊建议转投 PR Letters；
   超页先移附录（复现链/新颖性矩阵）至 Supplementary Material；
5. 【USER】项清零后方可提交（清单见 reports/submission-package-checklist-2026-08-26.md）。
