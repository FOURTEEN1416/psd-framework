# R34 逐页校对窗 P2 意见修复记录（2026-09-11）

## 范围：R32-pageproof-review 剩余可修项（B-09/B-10/B-11/B-12；B-05 用户人工项不动，B-14 审稿人自判可留）

- **B-09 ± 空格归一**：20 处 `$\pm$数字` 紧贴式 → `$\pm$ 数字` 带空格式（04×10 / 05×4 / 06×6；源码级真实漂移，非 pdftotext 伪影，Python 盘点实证）；
- **B-10 小节引用归一**：`§\ref{ssec:}` ×12（02×3 + 03×9）→ `Section~\ref{ssec:}`，全文小节引用统一 Elsevier 惯用式（审稿人推荐方向为 §，取改动最小且同样达成统一）；
- **B-11 verdict 词加粗统一**：LABEL\_BOTTLENECK、(PARTIAL at the 10% tier)、returns PARTIAL at both budgets → \textbf（与 NULL/fails/PARTIAL 既有粗体一致）；
- **B-12 强调块降级**：E7c "do \textbf{not}" → \emph{not}；E9 六行连续粗体收窄为仅 67.53%±0.24 与 90.7% 两处关键数字。

## 验证
pdflatex ×2：41 页 0 错误 0 浮动警告，overfull 1（cas 前置区不可见残留）；"?? " 零；图 1-5 落位 p5/11/16/21/32 不回退。

## 维持不变（有据）
- Highlights 内 3x/2.51x 字母 x：R24 LaTeX3 咬 \t 实锤，逐页 B-07 的 $\times$ 建议拒绝；
- 作者区 USER 占位（B-05）：投稿前用户人工项。
