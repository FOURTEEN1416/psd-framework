# R33 六窗审稿意见回收与修复记录（2026-09-11）

## 意见来源（六窗全落盘）
- R32-blind-review（独立盲审）：Major Revision/置信8 —— 7 MC + 14 minor；
- R32-adversarial-review（恶意审）：REJECT/概率55% —— 3 BLOCK + 6 CONCERNS + 7 minor；
- R32-cross-review（交叉审）：三方对账 **PASS**（P0=0/P1=0，P2=6，观察2；20 重点数字+~115 扩展点全命中）；
- R32-pageproof-review（逐页校对）：41/41 页实看，P0=0、P1=5、P2=12，页级 pass 34/41；
- R32-visual-review + VERDICT.json（视觉验收）：**PASS**（fatal0/major0/minor4）——图件零改动。

## 本轮已修（两轮共 44 处编辑，全部程序化断言复核）
### 第一轮（31 编辑 + YOLO 引用 + 8 条 bib）
- A 泄密清理：§5 两处 `Evidence: reports/p15/p16/p17-*.json` → released ledger；E7b 内部审计括注中性化；Table 4 caption dev-docs 路径移除；
- B 引用补全：新增 8 条权威 bib（Tarvainen/Mean-Teacher、Li/AdaBN、van den Oord/CPC、Snell/ProtoNet、Kuhn/Hungarian、Soomro/UCF101、Sun/HRNet、Ultralytics/YOLO11）并正文落点；修复渲染破损 [20] 粘连（删 repo note）与 [23] doi `\_` 直出（删 doi 字段）；
- C 数值呈现：32.5× 加注 unrounded seconds；15.8x/6.2x/32.5x → $\times$（六处）；tab:main 44.90 行加 best-of-3+mean 披露；82.0 行补 ±4.3；uniform-random 措辞；
- D 悬空披露：RAM-restart 句自包含化（冻结判据句零触碰）；
- E 行话：spc/R16 首现释义；
- F 措辞：(L9).. 双句点、elsewhere→beyond them、摘要分号大写、82.0% 边际预算限定词三处补齐、Highlights#5 改 "adapting to an offset domain with 20 clips"（main+standalone 双同步）。

### 第二轮（11 编辑，来自交叉审+逐页校对）
- refs.bib：J'egou→J\'egou（[31] 重音修复）；NTU60/NTU120 两条 title 双括号保护（bst 小写化 "Ntu rgb+d" 修复）；
- 正文：E7c 加 "extractor-training logs released with the K9 training repository" 指向（A-3-4 工件缺口）；"3x line" ×2 → $3\times$；E7 warm-start "full offset budget (220 clips)" → "200-clip budget point (91% of the 220-clip offset pool)"（A-3-1 预算点标注失实）；44.2% 分母消歧 "archived 37.50%"（A-3-3）；fig2 caption "physics encoder stays frozen" → "remains frozen throughout (not shown)"（A-3-6 图外断言）；Intro 贡献4 交叉引用扩为 Sections 4–5（C-1）；highlights.tex 头注字符数区间 65–77→65–85（A-3-5）。

### 拒绝执行项（有据）
- 逐页 B-07 建议 Highlights 内 3x/2.51x 改 $\times$：**拒绝**——R24 实锤 cas-sc LaTeX3 将 \t 展开为 tab 咬掉 \times（commit ac1d916 fatal 修复），字母 x 是当时的修复方案，维持。

## 验证链（修复后终态）
- pdflatex→bibtex→pdflatex×2 全链：**41 页 0 错误、0 浮动警告、overfull 1（cas 前置区不可见残留）**；
- 断言组全绿：泄漏词五项零残留、"Recognitionrepo"/`\_14`/"Ntu rgb" 清零、新引作者题名全部入印（H. Jégou/X. Sun 倒装确认）、"?? " 零、图 1-5 落位 p5/11/16/21/32 不回退；
- 交叉审独立复核：20 重点数字逐位命中 + ~115 扩展点命中 + 协议号/术语/三层口径全一致 + Limitations 13 条互洽。

## 未修·框架级决策项（动论点/动协议/需新实验，交用户裁决）
1. MC1/恶意A1：标题与摘要承诺量级 vs 动物域端到端全负结果的 framing 降级；
2. MC2/恶意M1：机制卖点重述（POOLDECOMP 零效应 → "合成档验证的机制 + 真实档边界"）；
3. MC4/恶意M2：已发表方法同口径对比（补 MCT 公开口径跑分，或将 L13 升入正文 Discussion）；
4. 恶意M3 深修：tab:main 主值 caliber 切换（44.9 best → 41.5 mean 作头条）——已加注，切换属头条变更；
5. 恶意B2/B3：MEXP/E4 纳入声明统计族并印刷族成员清单；
6. 盲审M#3：摘要压缩至 250 词；M#8/M#14：长段落拆分 + 实验索引表；
7. 盲审M#12：E7c mAP50/early-stop 数字导出 reports/ JSON（本轮已加仓外指向，归档属 K9 仓）；
8. 交叉审A-3-2：E7b spc2 14 vs 16 clips 口径（需工件侧裁决 sit 类 n_train）；
9. 恶意C4：82.0% 协议血统标注；盲审M#11：双盲切换时匿名仓库置换。
10. 页面校对 B-05：作者区 USER 占位（既有用户人工项）。
