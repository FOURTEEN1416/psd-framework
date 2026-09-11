# paper-sprint-b-2026-09-11 — 正文精修（任务包 B · 单窗串行执行）

> 输入：reports/paper-review-c-2026-09-11.md（P0×1 / P1×2 / P2×4）。
> 边界：仅散文层事故残留清理与事实精确化；冻结清单（§四）零触碰；
> 数字/口径/披露结构零改动；不新增引用；Highlights/摘要经程序化复测已合规不动。
> 每处改动附前后 diff 举证。

## 一、修复清单（4 处，对应 C 报告 P0-1/P1-1/P1-2/P2-1）

### B-1（P0）GenAI 声明去重 —— main.tex 声明区
```diff
-During the preparation of this work the authors used GLM-5.3-Flash (Zhipu AI) to
-assist with code development, experiment orchestration, and language editing. After
-using this tool, the authors reviewed and edited the content as needed and take full
-responsibility for the content of the published article. to assist with code
-development, experiment orchestration, and language editing. After using these tools,
-the authors reviewed and edited the content as needed and take full responsibility
-for the content of the published article.
+During the preparation of this work the authors used GLM-5.3-Flash (Zhipu AI) to
+assist with code development, experiment orchestration, and language editing. After
+using this tool, the authors reviewed and edited the content as needed and take full
+responsibility for the content of the published article.
```
理由：R40 单工具化改写残留（第二句为旧"these tools"版本尾巴）。工具名/颗粒度保持
用户终裁原文，仅删除重复碎片。

### B-2（P1）E6-real-2 缺右括号 —— 04-experiments.tex TRANS-002 段
```diff
-a final-segment-only bound (6.2$\times$ still clears the $3\times$ line.
+a final-segment-only bound (6.2$\times$) still clears the $3\times$ line.
```

### B-3（P1）冠词 "an +17.9" → "a +17.9" —— 04-experiments.tex E4 段
```diff
-(an $+17.9$\,pp endpoint gain; ...
+(a $+17.9$\,pp endpoint gain; ...
```
理由："+"读作 plus（辅音）；同节 L95 "by +8.3 pp" 已是正确用法。

### B-4（P2）fig5 caption "bar" → "error bar" —— 04-experiments.tex
```diff
-the NTU60 bar, $\pm$ 0.3\,pp, is smaller than its marker
+the NTU60 error bar, $\pm$ 0.3\,pp, is smaller than its marker
```
理由：图内无条形元素，指称对象是误差棒；事实精确化，非语义/颜色/方位变更。

## 二、复测合规、判定不动项

- Highlights 五条渲染态 62–84 字符（≤85）✓；与 main.tex 版内环境逐字同步 ✓；
- 摘要 ~252 词（R40 终裁 250±）✓；
- AI 腔九模式（delve/leverage/showcase/it is worth noting/not only/...）零命中 ✓；
- C 报告 P2-3（两位/四位小数口径）与 P2-4（±后缀风格）判留痕不改（全文风格自洽）；
- 冻结清单全文比对：0 处触碰（90.7/88.9/88.7/66.6/41.5/82.0/6.07×/15.8×/32.5×/PARTIAL/
  FAILS 声明与全部披露结构原样）。

## 三、验收

pdflatex ×2 零错误零浮动警告、42 页；断言：图 1–5 落位 p4/10/15/20/31 全 OK、
`??`=0、USER=0；编译后 GenAI 声明段 pdftotext 抽查无重复碎片。
