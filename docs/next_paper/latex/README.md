# docs/next_paper/latex/ — 下一篇论文 LaTeX 工程（elsarticle）

> 真源：`../paper_manuscript_v1.md`（v1.0）。本目录只做排版，不改内容与数字；
> 任何文字修订先回写真源 md，再同步到 `main.tex`。

## 文件地图

| 文件 | 作用 |
|------|------|
| `main.tex` | 全文（elsarticle preprint，单栏 12pt；三张 booktabs 表 + 四张图） |
| `highlights.tex` | Elsevier 投稿独立要件（3–5 条，每条 ≤85 字符） |
| `preview/` | 视觉验收用页面 PNG（110 dpi，gitignored） |

## 编译

```bash
cd docs/next_paper/latex
pdflatex --disable-installer -interaction=nonstopmode main.tex
pdflatex --disable-installer -interaction=nonstopmode main.tex
```

- 本机 MiKTeX：**必须加 `--disable-installer`**。`newtx` 字体包会触发在线安装并在无交互终端挂起；正文用 `mathptmx`（Times 系）替代。
- 不跑 BibTeX：41 条文献以 `thebibliography` 内联，编号与 md 版一一对应；改动条目顺序会破坏正文 `\cite` 编号，改前先跑一致性检查。
- 机器检查：`python scripts/p1_sprint/check_latex_next_paper.py`
  （数字逐项对齐 md、41 条引用全覆盖、摘要词数、PDF 无 `??`）。

## 当前状态（2026-11-21）

- 17 页；0 error / 0 warning / 0 overfull；字体嵌入为 Type1 Times。
- 图 1–2 全宽（165 mm 原尺寸等比缩至版心），图 3–4 为 0.72 版心宽。
- 表 2 的 “Anchors / budget” 合并列以控制宽度；Δ 与 Verdict 之间留 10 pt。

## 投稿前 TODO

1. 作者块：`\author` / `\affiliation` 现为 `Anonymous Authors` 占位，替换为正式作者与单位；
2. 期刊名：`\journal{Pattern Recognition Letters}` 按最终投稿方向改（TMM 需换 IEEEtran）；
3. 双盲/审稿版：将类选项改为 `\documentclass[review,12pt,number]{elsarticle}`（双倍行距）；
4. 附录 A–E 从 `reports/p1_sprint_evidence/` 填充后并入（当前稿不含附录）；
5. Highlights 与 cover letter 定稿（cover letter 需作者信息）。
