# paper-review-c-2026-09-11 — 旗舰级对抗审稿（任务包 C · 只读窗）

> 审稿对象：main.tex @ commit 1f8e06e（figA 增量收编后），cas-sc review 版 42 页。
> 方法：全文逐节精读（main/highlights/01–06）+ 数字↔工件↔图表交叉对账 + 程序化扫描
> （Highlights 字长/摘要词数/AI 腔九模式/括号平衡）。三轮人格：恶意 REJECT / 交叉核对 / 方法论 CONCERNS。
> 冻结清单冲突项：只记录，不建议改动（见任务书 §四）。

## 一、总评

**判定：P0×1、P1×2、P2×4，无结构性新攻击面。** 论文的诚实边界架构（L1–L13 + 预注册链 +
勘误链）依然是其最强防御——本轮全部六张图换新后，图文一致性（颜色词/方位词/数值）
交叉核对零失配。两个文字级硬伤（P0/P1-1）属于历史编辑事故残留，修复后即回到可投状态。
结构性弱点（K9 pilot 未行使、无已发表方法正面对比）已被 L9/L13 预先承认，属 GPU 实验
级事项，不在文本修复范围（任务书 §五裁决项）。

## 二、P0（致命 · 投稿前必须修）

### P0-1 GenAI 声明存在编辑残留重复碎片（main.tex L179）
原文："...the authors used GLM-5.3-Flash (Zhipu AI) to assist with code development,
experiment orchestration, and language editing. After using this tool, ... published
article. **to assist with code development, experiment orchestration, and language
editing. After using these tools, the authors reviewed and edited the content as needed
and take full responsibility for the content of the published article.**"
——R40"单工具化"改写（this tool / these tools）时旧句尾未删净，声明区出现 garbled
重复。这是 Editor/审稿人第一眼可见的整洁度硬伤，且位于合规声明区，杀伤力放大。
**修复**：删除第二段重复尾巴（不改任何实质措辞——工具名/颗粒度保持用户终裁原样）。

## 三、P1（重大）

### P1-1 E6-real-2 段落缺右括号（04-experiments L86）
"a final-segment-only bound **(6.2$\times$** still clears the $3\times$ line."
——括号未闭合。注意 06-conclusion L12（L12 段）同内容写法正确（"...a final-segment-only
bound of 6.2$\times$ still clears..."），仅 04 此处漏。修复：补 ")"。

### P1-2 E4 冠词错误："an +17.9 pp"（04-experiments L38）
"+" 读作 "plus"（辅音），应为 "a +17.9 pp"。同节 L95 "by $+8.3$\,pp" 用法正确，
此处为孤立不一致。

## 四、P2（次要 · 建议修复或留痕）

| # | 位置 | 发现 | 处置建议 |
|---|------|------|---------|
| P2-1 | 04 L63 fig5 caption | "the NTU60 **bar**, $\pm$ 0.3 pp, is smaller than its marker" —— 图中无条形，该元素是误差棒（直标版散点图） | 改 "error bar"（事实精确化，非语义变更） |
| P2-2 | highlights.tex 头注 | 注释声称"五条已程序化实测 65–85 字符" | 本轮复测 62–84 全部合规，注释可保留（无需改） |
| P2-3 | 04 L18 E2 | "boundary IoU 0.458 ± 0.049" 与 fig3 caption "0.4577 ± 0.0488" 为不同有效位口径 | 两者各自语境一致（正文两位小数/图注四位原始值），留痕不改 |
| P2-4 | 04 L41 E7 | "9.8% ± 7.5" 等处 ± 值缺 % 后缀 | 与 tab:main 及全文风格一致（%挂在主值上），留痕不改 |

## 五、交叉对账结果（数字↔工件↔图表）

- 五图落位 p4/10/15/20/31 ✓；caption 颜色词（orange/teal）与 v17 在位图一致 ✓；
- fig5 caption 位移清单（13.9/11.5/8.0/8.8/13.2/16.5）与 v6 脚本 dodge 逐位一致 ✓；
- 冻结口径抽核全对：90.7/88.9/88.7/66.6/41.5(±5.9, best 44.90)/82.0(±4.3)/6.07×/
  15.8×(9897s vs 626s)/32.5×/+2.9pp(82.6 vs 79.7, Wilcoxon p=0.002, Holm 0.004)/
  0.4577±0.0488/77.97 vs 77.18/74.45 参考值 ✓；
- E9 八检验族/E9d 三十种子扩展（85.8% PARTIAL, p=0.54）/POOLDECOMP 四臂（67.53–67.58）
  /E7b 同空间对照（25.96%）与冻结记录逐位一致 ✓；
- 摘要 252 词（R40 终裁 250±）✓；Highlights 五条 62–84 字符（≤85）✓ 且与 main.tex
  版内环境逐字同步 ✓；AI 腔九模式零命中（R24/R42 反防御存量保持）✓；
- 章节交叉引用/图号/公式引用完好（编译 0 错误 + ?? = 0）✓。

## 六、恶意 REJECT 人格三轮攻击尝试（均被既有防线化解，留痕）

1. *"标题卖 evolving criteria 但 canine 层近随机"* → 已被标题删 Low-Resource（R20）、
   摘要 tier-dependent boundary 句、L9 边界分析三重防御；无新攻击面。
2. *"90.7% 是 human-domain 不是 animal"* → E9 系已显式标注 implementation-equivalence
   + E9d PanAf500 动物域正结果；L2/L9 边界完整。
3. *"零已发表方法对比"* → L13 显式承认 + SSL-BASE 外部基线下界 + equivalence 验证；
   属预注册 K9/GPU 级遗留（§五），非文本可修。

## 七、修复路由

P0-1 / P1-1 / P1-2 / P2-1 均为文本层修复 → 移交任务包 B 当轮执行（见
reports/paper-sprint-b-2026-09-11.md 逐处 diff 举证）。P2-3/P2-4 留痕不改。
无冻结清单冲突：四处修复均为事故残留清理/事实精确化，不触碰任何冻结措辞、
数值、口径与披露结构（GenAI 声明的工具名与颗粒度保持用户终裁原文）。
