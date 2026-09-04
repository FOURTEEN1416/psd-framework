# Auto-Review-Loop Round 5（2026-09-04 · 恶意审稿极端批判 + 引用实际查询）

> 对象: 23 页成稿（fig2 v3 flywheel + fig3 定性入稿后）
> 引用验证渠道: ghproxy 上游仓库 / openaccess.thecvf / IEEE Xplore / cvir 项目页（arXiv 直连与 HF 本轮网络受限，受限项标注待补）

## A. 引用实际查询结果（用户点名的关键项）

| 条目 | 验证渠道 | 结果 |
|------|---------|------|
| guo2022aimclr (AimCLR, AAAI 2022) | 官方仓 Levigty/AimCLR README | ✅ 标题逐字一致 + "3s-AimCLR **79.18**" 官方口径证实 + AAAI 2022 |
| gokay2025smq (SMQ, ICCV 2025) | awesome 目录 + openaccess.thecvf PDF | ✅ 标题一致 + ICCV 2025 CVF 开放获取页 200 |
| singh2021tcl (TCL, CVPR 2021) | 官方项目页 cvir.github.io/TCL + 官方仓 CVIR/TCL | ✅ **作者七人逐一核实**：Singh(IIT-M)/Chakraborty(IIT-K)/Varshney(IIT-K)/Panda(MIT-IBM)/Feris(MIT-IBM)/Saenko(MIT-IBM+BU)/Das(IIT-K)——**⚠️ bib 现有作者序与官方页出入：官方第二~三为 Omprakash Chakraborty / Ashutosh Varshney，bib 写 Prithvijit Chakraborty / Ameesh Varshney——名字错，须改**；82.7%/88.6% 数字本轮渠道未直接见原文表格（README 只有 x% 协议），保留投稿前对原文 PDF 复核项 |
| mac2022learning (MAC-Learning, TPAMI 2022) | awesome 目录 | ✅ repo 1xbq1/MAC-Learning + IEEE 9954217 一致 |
| mct2024tip / gra2024tnnls | IEEE Xplore 直连 | ✅ 10820022 / 10398229 均 202（存在） |
| ng2022animalkingdom | CVF openaccess（URL 变体 404）| ⚠️ 本轮渠道受限未直证——本地数据集自带文档无 citation 节；**保留 Scholar 终审** |
| yang2023aptv2 | arXiv 直连断 | ⚠️ 待补 |
| yolo2025petx / 犬行为 workshop 等 | CEECT/workshop 无独立开放页 | ⚠️ 待 Scholar 终审（已在清单） |

**行动项**：singh2021tcl 作者名修正（Omprakash Chakraborty / Ashutosh Varshney）——已执行。

## B. 恶意审稿极端批判（Review 4 前的"击沉演练"）

### 🔨 恶意审稿人 #1："我要证明你们的方法没用"
1. **"你们最硬的行为识别结果是 44.9%（4类，watch 100% 主导）——这不就是多数类分类器吗？"**
   → 防线已在（L7 per-class 同框 + "majority-class recall accounts for essentially all" 自认）。**残余风险**：Abstract/Intro 未提 44.9% 的弱点只报数字——检查通过：Para 6 写明 "(1.80× random; severe class imbalance disclosed in Section 6)" ✅
2. **"解耦成本 6.07× 是合成层玩具实验，真实场景谁在乎 188 秒 vs 31 秒？"**
   → 防线：E5 明示合成层口径 + 保守 ≥3× 措辞 + 成对披露。**新发现缺口**：未在正文讨论"成本的量级重要性"（实际业务中重训是小时/天级，31s→188.7s 的比例可外推但绝对值不可）——**补一句限制性说明进 E5 段**。
3. **"NTU 等价性只证明 AimCLR 复现对，SMQ 和 TCL 呢？"**
   → 真实缺口！R4 原文只覆盖 AimCLR 管线。SMQ 有 W38 消融（0.458 vs null）但无官方数字对照；TCL 无官方代码无从对照（p04 §6 已披露）。**处置**：E5 段 scope 声明扩一句，明示"等价性验证覆盖 AimCLR 预训练与融合协议；SMQ 以预注册消融对照替代官方数字对照（官方无公开基线协议），TCL 因无官方代码以等效迭代闭环交付（§3.3.3）"。

### 🔨 恶意审稿人 #2："我要证明你们的实验不可信"
4. **"单卡 8GB + CPU/GPU 混跑（L10 自曝）——同一篇论文里 CPU 档和 GPU 档数字不可比，你们自己都说不能 poolable，那 tab3 怎么还放在一张表里？"**
   → tab3 表内各格已带 tier/device 标注，梯度叙事段已声明 within-tier paired deltas only。**残余**：tab3 预训练行内 spc5/10/20（CPU）与 full（GPU）确实同格——**加一行表注明示**。
5. **"warm-start 82.0% 的合成偏移层 noise_std=0.10 是你们自己造的分布——分布是不是故意调到对 warm-start 有利？"**
   → W23 诊断选档 0.10 有预注册（ADR），非事后调参。**补**：method §3.3.3 或 E5 段加"noise_std=0.10 为预注册诊断选档（ADR-0005），非事后选择"一句。

### 🕵️ 主张审计员："措辞与证据的 0.5pp 差距也要抓"
6. Abstract "82.0% top-1 on 22 classes from only 20 labeled clips (synthetic-offset benchmark)" vs method.md "±4.3 across three seeds"——摘要未带 std。按 galaxy 规范摘要可不含，但 Intro Para 6 也未带——**一致性通过（摘要/Intro 均不带 std 是统一纪律），结论段带了 ±4.3**。零变更。
7. "first anchor-cluster-pseudo-label transfer"（highlights）vs 正文 "to the best of our knowledge"——highlights 作为独立文件可用更强措辞（投稿系统分开展示），但为稳妥**改为 "First (to our knowledge)"**。

### 📐 方法学破坏者
8. "12 类两臂实验 n_val=56，±1.46pp 的种子噪声下 +8.3pp 是 5.7σ？三种子的 std 根本不是真正的 CI"——**真问题**。E5 对比段补 "three seeds; we report seed-level std as dispersion, not as a confidence interval"。

## C. 处置汇总

| # | 级别 | 处置 |
|---|------|------|
| TCL 作者名 | **错误修正** | refs.bib 已改（Chakraborty, Omprakash / Varshney, Ashutosh） |
| #2 成本量级 | MINOR | E5 段补一句 |
| #3 等价性覆盖面 | MAJOR | §4.4 scope 声明扩句（SMQ/TCL 覆盖边界明示） |
| #4 tab3 设备混杂 | MINOR | tab3 表注补 |
| #5 noise_std 预注册 | MINOR | E5 段补 |
| #7 highlights | MINOR | 改 "First (to our knowledge)" |
| #8 std≠CI | MINOR | 五臂段补 |

全部修复入本轮 commit；AK/APTv2/CEECT/workshop 四条渠道受限项并入 Scholar 终审清单。
