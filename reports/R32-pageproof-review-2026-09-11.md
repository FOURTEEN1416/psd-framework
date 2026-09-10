# R32 投稿前逐行逐页校对报告（page-proof review）

- **日期**：2026-09-11
- **对象**：`docs/paper/latex/main.pdf`（41 物理页，cas-sc 单栏双倍行距评审版；印刷页码 = 物理页 − 1，首页 Highlights 不编号）
- **方法**：pdftoppm 150dpi 全 41 页逐页实看（零抽样）+ pdftotext -layout 文字层正则扫描 + tex 源码逐文件审查（scholar-latex-cleanup 全清单）+ 300dpi zoom 双核复核可疑项（PNG 双核纪律）
- **技能**：scholar-latex-cleanup + comp-visual-review（--review 纪律：全部结论以实际渲染实看/zoom 为证据，pdftotext 异常一律 PNG 双核裁定）
- **边界**：只读校对，未修改论文任何文件；临时渲染产物（PNG/脚本）已清理
- **源码基线**：main.pdf 2026-09-11 01:36 构建；main.tex 01:31；sections 01:35

---

## A 部：逐页巡查表（41/41 页实看）

物理页 = PDF 页序；印刷页码为页脚 "Page N of 41"。

| 物理页 | 印刷页 | 判定 | 发现与定位 |
|---|---|---|---|
| 1 | —（Highlights） | pass | Highlights 单页，下半页留白为 Elsevier 惯例；"≥3x/2.51x" 字母 x 与摘要乘号不一（→B-07） |
| 2 | 1 | **P1** | 作者行 "USER, USER, , USER" 双逗号空槽（postcode 空占位）+ ORCID(S): 空白 + email 占位（→B-05，用户人工项）；摘要 "; On the synthetic tier" 句中大写（→B-06）；其余版面正常 |
| 3 | 2 | pass | Φ/Ω 数学渲染正常；pdftotext 报"缺字符"经 PNG 证为提取伪影 |
| 4 | 3 | pass | |
| 5 | 4 | pass | Figure 1 清晰无压叠，caption 紧贴 |
| 6 | 5 | pass | 引号源码为 ``scarce''（150dpi 视似直引号，源码正确） |
| 7 | 6 | pass | 同上，``anchors'' 源码正确 |
| 8 | 7 | pass | Table 1 未超版心；Skeleton-to-Image 单元格两行换行可读 |
| 9 | 8 | pass | 页底约 1/4 留白（3.2.2 节整体挪次页防标题孤悬，可接受）；Section/§ 混用此页可见（→B-10） |
| 10 | 9 | pass | |
| 11 | 10 | pass | Figure 2 图例/虚线回写箭头完整 |
| 12 | 11 | pass | Algorithm 1 框线/下标正常 |
| 13 | 12 | pass | 协议编号 PSD-K9-PREREG-001 等为公开预注册编号，非内部代号 |
| 14 | 13 | pass | \texttt{review-snapshot} 未撑行 |
| 15 | 14 | P2 | ± 空格漂移：E3 "(0.5025±0.0063" 紧贴 vs E1/E2/E4 "± " 带空格（→B-09） |
| 16 | 15 | pass | Figure 3 清晰；"agg 0.4577…" 注释文字经 300dpi zoom 证实与轴线无压叠 |
| 17 | 16 | pass | 脚注 1 小字印刷可辨 |
| 18 | 17 | pass | 长 \texttt{r12-holm-eightclass…} 未撑行 |
| 19 | 18 | P2 | 正文单词级粗体强调 "do **not**"（04-experiments.tex:47，风格突兀，→B-12） |
| 20 | 19 | pass | |
| 21 | 20 | pass | Figure 4 图例 7 项/log 轴/误差棒完整 |
| 22 | 21 | P2 | E9 段约 6 行连续粗体长句，强调块过重（04-experiments.tex:70，→B-12） |
| 23 | 22 | pass | |
| 24 | 23 | pass | |
| 25 | 24 | pass | 同段 "15.8×" 此处为乘号（与页 36 字母 x 对照，→B-08） |
| 26 | 25 | P2 | ± 紧贴式再现（Segmentation strategy 段，→B-09） |
| 27 | 26 | pass | Table 2 未超版心；300dpi zoom 证实 "(HRNet 2D)" 为大写 N（150dpi 误读撤销）；"wall–clock" 窄列断行轻微（→B-14） |
| 28 | 27 | pass | Table 3 密集但可读，caliber 斜体正常 |
| 29 | 28 | P2 | verdict 词风格：NULL 粗体 vs LABEL_BOTTLENECK 正体（→B-11） |
| 30 | 29 | **P1** | **"(L9)..)" 双句点残留**（05-ablation-analysis.tex:36，→B-04）；Evidence: reports/*.json 为仓库相对路径且论文声明公开，复验不构成内部信息泄漏 |
| 31 | 30 | pass | |
| 32 | 31 | pass | Figure 5 双面板完整 |
| 33 | 32 | pass | |
| 34 | 33 | pass | |
| 35 | 34 | pass | |
| 36 | 35 | pass | L12 段 "15.8x/6.2x/3x/32.5x" 字母 x（→B-08）；CRediT/声明区开始 |
| 37 | 36 | pass | github.com/FOURTEEN1416/psd-framework 为论文公开仓库 URL（Data availability 有意披露），非泄漏 |
| 38 | 37 | pass | Table 4 浮动至页顶、位于附录 A 标题前（pos=t 规范行为）；GenAI 声明正常 |
| 39 | 38 | pass | 附录 B 等宽脚本链可读；"train_aimclr.py" 无断名（150dpi 视误，pdftotext 证完整）；References [1]–[6] 开始 |
| 40 | 39 | **P1** | **[20] "Pattern Recognitionrepo" 粘连**（refs.bib:180，→B-01）；**[23] DOI "…73347-5\_14" 反斜杠残留**（refs.bib:197，→B-02）；[36][37] "Ntu rgb+d" bst 小写化（→B-13） |
| 41 | 40 | **P1** | **[31] "H. J'egou" 重音渲染错误**（refs.bib:259，→B-03；同页 Gökay/Kjellström 的 ö 正常，证为该条转义写错）；末页留白正常 |

**页级统计**：完全 pass 34/41；带 P1 页 4；仅带 P2 注记页 3。

---

## B 部：逐行 findings 表

| 编号 | 类别 | 定位 | 原文摘录 | 级别 | 修改建议 |
|---|---|---|---|---|---|
| B-01 | 引用错位/粘连 | refs.bib:180（印于物理页 40，条目 [20]） | `journal = {Pattern Recognition}` + `note = {{repo github.com/Levigty/AimCLR-v2}}` → 印刷为 "Pattern **Recognitionrepo** github.com/…" | **P1** | elsarticle-num 对 article 的 note 不加分隔符。改为 `note = {{. Repo github.com/Levigty/AimCLR-v2}}`（前置句点），或并进 journal 字段后手动标点。修复后需重跑 bibtex 三连 |
| B-02 | 乱码残留 | refs.bib:197（物理页 40，条目 [23]） | `doi = {10.1007/978-3-031-73347-5\_14}` → 印刷为 "doi:10.1007/978-3-031-73347-5**\\_**14" | **P1** | doi 字段去 `\_` 改裸 `_`（该 bst 的 doi 输出宏已按安全模式排印，比对 [12][24] 等正常条目）。改后 DOI 可点击/复制 |
| B-03 | 乱码残留（重音） | refs.bib:259（物理页 41，条目 [31]） | `J'egou, Herv'e` → 印刷为 "H. **J'egou**"（直撇号+e） | **P1** | 改 `J\'egou` 与 `Herv\'e`（正确作者名 Hervé Jégou）。同页 Gökay/Kjellström 转义正确可作对照 |
| B-04 | 字词重复（标点） | sections/05-ablation-analysis.tex:36（物理页 30） | "…diagnosis of Section 6 (L9)**..** Evidence: reports/p15-…" | **P1** | 删多余句点：`(L9). Evidence:` |
| B-05 | 排版混乱（占位符） | main.tex:60–68（物理页 2） | 作者行渲染 "USER, USER, , USER"（postcode={} 空槽）；ORCID(S): 空白；email 占位 | **P1**（用户人工项） | 已知作者名单为用户终审项；提醒： affiliation 各字段须全非空或删空字段，否则双逗号空槽印在首页。投稿前必清 |
| B-06 | 文字病（句中大写） | main.tex:72 摘要（物理页 2） | "…source domain); **On** the synthetic tier, a taxonomy transition…" | P2 | 分号并列子句改小写 "on the synthetic tier" |
| B-07 | 数字格式漂移 | main.tex:78/80 + highlights.tex:12/14（物理页 1） | Highlights "cost ≥**3x** less wall clock"、"**2.51x**" 用字母 x；摘要/正文同数字用 "$2.51\times$"、"$\geq 3\times$" | P2 | highlights 两处改 `$\geq$3$\times$` / `2.51$\times$`，与摘要统一（改后 highlights.tex 需同步重编译） |
| B-08 | 数字格式漂移（整段） | sections/04-experiments.tex:80 + sections/06-conclusion-limitations.tex:47（物理页 36，L12 段） | "ratio is **15.8x** … bound of **6.2x** still clears the **3x** line … reaches **32.5x** with a **+2.6**\,pp … gap **-4.5**\,pp" | P2 | 该两段为全篇唯二字母 x 区（E6 段同实验用 $6.07\times$/$\geq 3\times$）。统一改 `$15.8\times$` 等；裸文本 `+2.6`/`-4.5` 改数学模式 `$+2.6$`/`$-4.5$`（文本负号是连字符，短于数学负号） |
| B-09 | 排版风格漂移（±） | 04-experiments.tex:35（E3 段）、:86（Segmentation strategy）、05-ablation-analysis.tex:14/20/24、Table 3 多处 vs 04:21/24（E1/E2）、Table 2 | "(0.5025**±**0.0063" 紧贴 vs "20.89% **± **4.04%" 带空格；同一数字 0.323±0.022 两处两种式样 | P2 | 全文统一为 `$\pm$` 带空格式（推荐，主流 Elsevier 排版）；或全部紧贴。以宏统一最稳：`\newcommand{\pms}{\,$\pm$\,}` |
| B-10 | 引用语法不统一 | sections/02-related-work.tex:62、03-method.tex:14/30/69 用 `§\ref{ssec:…}`；04-experiments.tex:41 等用 `Section~\ref{ssec:…}` 指同级小节 | "no mechanism for taxonomy change (**§2.1**)" vs "(the oracle-label leakage …, **Section 3.3.2**)" | P2 | 定一条规则（推荐：节级 "Section~\ref"、小节 "§\ref"）后全文 grep 归一；当前 §/Section 对小节的指代随机混用 |
| B-11 | 术语强调风格 | 04-experiments.tex:47（LABEL\_BOTTLENECK 正体）vs 05:36（**NULL** 粗体）vs 04:56（**fails**/**PARTIAL** 粗体） | "verdict LABEL_BOTTLENECK" / "returns **NULL**" / "endpoint **fails**" | P2 | verdict/决策词统一一种排版（推荐全部 \texttt 或全部粗体） |
| B-12 | 强调块过重 | 04-experiments.tex:47（"do **not**"）、:70（约 6 行连续粗体句） | "The stronger skeletons do **not** raise the ceiling"；"**67.53% ± 0.24 (ten seeds, …) once the PSD semantic pipeline … is applied under the corrected protocol of E7—90.7% …**" | P2 | 单词强调改 \emph；6 行粗体句收窄至关键数字短语（粗体整段失去强调功能且双倍行距下视觉沉重） |
| B-13 | bst 大小写 | refs.bib（[36][37]，物理页 40–41） | "Ntu rgb+d: A large scale dataset…" / "Ntu rgb+d 120: …" | P2 | title 加保护：`{NTU RGB+D}: A Large Scale Dataset…`（bst 句式小写化吞掉缩略词；与既有"bst note 大小写"教训同族） |
| B-14 | 窄列断行 | 04-experiments.tex:106 Table 2 Metric 列（物理页 27） | "wall–clock" 断为 "wall–"/"clock" 两行 | P2 | Metric 列宽从 1.2cm 略加宽，或该格写 `wall\-clock` 控断点；轻微，可留 |

**泄密层复验结论（B 附）**：全文与脚注无 `D:\`、`C:\`、绝对路径、窗口号、任务书编号；出现的 `reports/p15-…json`、`dev-docs/research/`、`ADR 0008`、`spc2/spc4`、协议编号（PSD-*-PREREG-001）、`github.com/FOURTEEN1416/psd-framework` 均为论文明示公开的仓库工件指针/预注册编号，R32 盲审轮所报"内部路径"类 minor 已清零（当时所指即相对工件指针，现均有公开声明兜底）。R32 盲审"正文泄漏内部路径"复核口径：**未发现绝对路径级泄漏**。

**pdftotext 伪影备忘（非成稿缺陷）**：页 2 "taxonomy of Ω while Φ"（数学符号丢失）、附录 B 箭头 → 丢失、表格对齐双空格——均经 PNG/zoom 证为提取层伪影；"the classification and "+17.9 pp 等处无真实双空格。

---

## C 部：结构连续性结论

| 检查项 | 结果 |
|---|---|
| 节号 | 1 Introduction → 2 Related work → 3 Method → 4 Experiments → 5 Ablation and analysis → 6 Conclusion and limitations，连续无跳；附录 A（Novelty survey matrix）/B（Reproduction chain）齐 ✅ |
| 图 | Figure 1–5 编号连续单调：fig1(§1 尾)、fig2(§3.3.2)、fig3(§4.3 E2)、图 4（源文件 fig5_budget_retention.pdf，§4.4）、图 5（源文件 fig4_al_efficiency.pdf，§5）；正文首引均先于实体、\ref 渲染编号正确 ✅（文件名与图号交叉是命名遗留，不影响成稿） |
| 表 | Table 1(§2.3) → 2(§4.3) → 3(§5) → 4(附录 A) 连续；正文引用齐全 ✅；Table 4 浮动到附录 A 标题前属 pos=t 规范行为，非错序 |
| 算法 | Algorithm 1 唯一，§3.3.2 引用正常 ✅ |
| 页码 | "Page 1 of 41"–"Page 40 of 41" 连续无跳、无重复（40 个页脚 + 首页 Highlights 不编号 = 41 物理页）✅ |
| 引用 | `??` 断引用全文零命中 ✅；References [1]–[37] 编号连续；bibtex 正常收敛 |
| 声明区顺序 | CRediT(\printcredits，无重复标题) → Competing interest → Funding → Acknowledgements → Data availability → Ethics → GenAI 声明 → 附录 → References，符合 cas-sc/Elsevier 评审版惯例 ✅ |
| 版心/浮动 | 全 41 页无 overfull 撞缘、无图表压叠正文、无浮动体滞留文末（R31 修复后正常）、双栏表全部未超版心 ✅ |
| 一致性总体 | 弯引号统一（``…''）、i.e.\ 格式统一（2 处同式样）、Figure~/Table~\ref 语法统一（仅 §/Section 分层混用见 B-10）、正文零 CJK/中文标点（仅注释行）、\TODO 调用清零（仅 main.tex:26 宏定义）、千分位 {,} 统一 ✅ |

---

## 总结

**页级 pass 率**：34/41 页完全 pass（83%）；其余 4 页带 P1、3 页仅带 P2 注记。
**计数**：**P0 = 0**；**P1 = 5**（[20] 粘连、[23] DOI \_、[31] J'egou、(L9).. 双句点、作者占位符空槽【用户人工项】）；**P2 = 12**。
**裁决**：**版面骨架已达可投质量（无 P0、无溢出、无断引用、无浮动体事故），但尚未可投**——修完 4 处机械级 P1（三个 bib 条目 + 一处双句点，均为分钟级修复、修后需重跑 pdflatex→bibtex→pdflatex×2）并确认作者区占位符（B-05，用户终审项）后即可达可投状态；P2 各项（乘号/±/§风格归一、强调块收窄）建议顺手归一但均不构成拒稿面。
