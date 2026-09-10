# R36 Path-A 执行记录（承诺降级）+ Path-B 预研交付（2026-09-11）

> 用户裁决（2026-09-11）：按推荐执行——A 先行投稿不空转，B 最小版并行预研，K9 数据到货进 camera-ready/第二篇。

## A 线：叙事收敛（8 处编辑，main/01/04）
1. **摘要末句降级**："turns ... into a routine semantic-layer update" → "reduces taxonomy evolution to a semantic-layer update whose benefits are architecture- and scale-dependent rather than universal"——"routine" 承诺撤回，收益限定为架构/规模依赖；
2. **贡献 2 机制卖点前置校准**：补 "its marginal value over plain near-full-pool self-training is itself measured (Section 6, L11)"——POOLDECOMP 零效应从"远端辩护"升为"贡献句内自披露"（恶意 M1 叙事解）；
3. **44.9 → 41.5±5.9 主值切换（三处联动）**：摘要无此数✓；§1 末段、E4 正文、tab:main 行全部切为三种子均值主报（best 44.90% 括注；乘数 1.80×→1.66× 主报、1.80× best 括注）——恶意 M3"best-of-3 上主表"从结构上消灭；
4. **E4 三检验族成员印刷**（恶意 B3）：主检验/无共识消融/α=0 对照 + r16-holm-p04 工件指向；
5. **MEXP 双检验族声明**（恶意 B2）："two-test addendum family, Holm inside, adjusted p=0.004, declared here rather than retrofitted"——族治理从"事后"改"声明在先"；
6. **无已发表对比 scope 句升入 §4.2**（恶意 M2 叙事解）："no superiority over any published system is claimed anywhere (L13)"。

**标题维持不改**：R20 已删 Low-Resource；现标题描述框架归属域而非结果承诺，摘要/贡献降级后标题-证据落差已闭合，再改标题伤身份无增益。

## B 线：预研交付
`dev-docs/research/pathB-min-asym-transfer-prereg-draft-2026-09-11.md`——B-MIN-TRANS 草案：NTU60 平台 60→49 迁移三臂设计（D=规则种子 0 人工标签 / C=全量重标注 / M=匹配求解器），规则来源两方案（几何先验规则优先，规则代理有循环性风险），预注册判据（wall-clock ≥3× 且标注量比 ≥10×，±2.3pp 带），机时估算 ≈1 次 GPU 排期（~2 天）+ 1 人日规则编写，无新数据依赖。K9 试点的方法论彩排。

## 验证
pdflatex ×2：41 页 0 错误 0 浮动警告；断言：routine 措辞清零、bounded 措辞在印、mean-primary 三处生效、族成员/族声明/scope 句在印、1.66× 在印、"?? " 零、图页落位不回退。

## R33 决策清单状态更新
- 已执行（用户裁决 2026-09-11）：#1 摘要承诺降级 ✓ / #2 机制重述 ✓ / #4 主值切换 ✓ / #5 族治理（B2 声明 + B3 成员印刷）✓ / #3 叙事部分（scope 句升正文）✓——MCT 跑分归 B 线待议；
- 仍挂起：摘要压缩 250 词、长段落拆分+实验索引表、E7c 工件归 K9 仓、spc2 14/16 口径、双盲匿名化、B-05 作者占位。
