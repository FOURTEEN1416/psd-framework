# 真源交割治理记录（2026-09-10）

## 裁决
用户指令"进行治理"（选中治理债条目）= 批准推荐方案：**改声明不追内容**——
论文正文装配 truth 由 `docs/paper/*.md` 正式移交 `docs/paper/latex/`（tex），
md 冻结为历史设计稿。理由：R22 之后 fixfull/MEXP/R24/R25 五轮全部直改 tex，
md 从未回同步，"truth=md" 条款事实失效；追平五份 md 成本高且结构已陈旧，
维持"Truth 单一性"（AGENTS.md L18）的唯一诚实解是把声明改成与执行一致。

## 变更（14 文件，全部注释/注记级，零正文内容改动）
1. **六节 tex 头**（sections/01..06）：`装配源 truth: docs/paper/xxx.md` →
   `truth: 本文件（2026-09-10 交割）`。
2. **main.tex**：装配纪律段、正文装配区标题、六条 `\input` 尾注（`<= xxx.md` →
   `hist-draft: xxx.md`）、声明区标题（truth=本 tex，submission-package md 降为
   历史建议稿）。声明区 L107/118/122 的取证来源注（许可承诺/InterPet4D 卡逐字引用/
   GenAI 🔴 待终裁）**保留**——那是证据出处不是装配真源。
3. **七份 md 头部治理注记**：introduction / related-work / method /
   conclusion-limitations / figure-specs / figures/FIGURE_SOURCE 六份冻结降权；
   **experiment-skeleton.md 例外条款**：实验协议/结论台账记录（R4 终判行等）不受
   影响，仅撤销"正文装配 truth"身份。

## 边界（未动）
- 全部预注册协议 md（k9-pilot / budget-curve / panaf-a2 / ntu-transition-002 等）：
  它们是协议真源非正文装配源，**继续有效且不可改写**（预注册纪律）。
- reports/ 工件、dev-docs/（代码侧 truth root 不变）。
- 正文文字与数字：零改动（编译 41 页与交割前一致，0 错误）。

## 后续纪律
- 正文修订直接改 tex（原流程事实上已如此）；md 不再要求同步。
- 若未来需要"设计稿→正文"新流程，另立 ADR，不复活旧条款。
- 补丁脚本（一次性）：workspaces/r24-refine/patch_truth_governance.py（数模侧，未入 psd 仓）。
