# R31 投稿前交叉重排+全文扫描记录（2026-09-11）

## 触发
用户令："进行交叉重排和扫描为最后的投稿做最后检查"。R30 收官后对 master(23ba8ea) 全文体检。

## 发现与修复（3 实锤 + 2 记录）
1. **【致命·已修】五张图浮动体全部滞留文末 37-41 页**（正文引用点在 p4/9/13/15/27）。
   根因三层剥离（全部实测复现，最小文档定位）：
   - `cas-common.sty` 用 LaTeX3 重定义 figure/table 环境，定位符走 key-value 语法（`[pos=t]`）；
   - 裸 `[t]` 语法落入 unknown-key 分支把 `\fps@figure` 写成空串（运行时实测 undefined），
     `\@float{figure}` 无参调用拿到空定位符 → 内核 "No positions in optional float specifier"
     ×5 → 浮动体链式滞留到 `\clearpage` 文末冲刷；
   - 修复双保险：①preamble `\def\fps@figure{tbp}\def\fps@table{tbp}`（`\renewcommand` 对
     undefined 会报错且被 nonstopmode 吞掉——第一版修复无效的原因）②全部 5 figure + 4 table
     环境改类约定的 `[pos=t]`→再放宽 `[pos=tbp]`（pos=t 只许页顶，fig5 被占顶后滚到 p41）。
   修复后：警告 5→0，图 1-5 落位 p5/p11/p16/p21/p32，全部紧邻引用页。
2. **【已修】tab3 右缘 10.07pt overfull**：表级 `\hyphenpenalty=10000` 使 `\-` 断字失效
   （第一版断字 hack 无效的原因），"(exploratory)" 在 p{1.4cm} 列整体放不下；列宽 1.4→1.8cm，
   X 弹性列吸收，overfull 消除，文字逐字未动。
3. **【记录·不可见】标题区 117.0831pt overfull**（\maketitle，R18 注释同值）：渲染 p1/p2
   目检完全干净，属 cas 前置区内部盒子日志残留，非可见缺陷，不追。
4. **【记录】underfull 29 条**：review 双倍行距+float flush 常态，目检无可见松行，不阻塞。
5. **【甄别·非缺陷】pdftotext "叠词"**：p2 "recognition recognition" 为关键词栏与摘要栏
   双栏提取交错假阳性；p38 "no no" 为新颖性矩阵相邻单元格提取假阳性。

## 扫描断言（全绿）
- 断引用 "??"：0；未定义引用/citation 警告：0（main.log）；multiply-defined：0；
- `\TODO{` 实际用例：0（2 处命中均为注释）；red 占位：0；
- 24 个头条数字在文抽查：全部命中（MEXP 以舍入口径 82.6/79.7/+2.9pp/p=0.002/32.5x 在文；
  −4.49pp 在文写作 −4.5 pp×2 处均语境正确；28.9 仅在 GA 图不属 main.pdf，预期内）；
- highlights.tex 与版内 highlights 五条逐字同步 ✓；页脚 "Page N of 41" 计数含图页 ✓。

## 终验
- pdflatex ×2：41 页 0 错误、0 浮动警告、overfull 1（不可见类残留）；
- judge 对重排后五个图页（p5/p11/p16/p21/p32）：**5/5 pass**（落位/图注配对/无碰撞/无残留）。

## 人工项提醒（投稿前用户终裁，与 R22 清单一致）
作者名单/单位/ORCID（现为 USER 占位）、GenAI 披露颗粒度、通讯作者信息。
