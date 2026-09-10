# R24 精修轮审查记录（2026-09-10）

> 方法论：数模竞赛科研工具箱 `auto-review-loop`（串行三模式 + 独立视觉 judge 双层，对齐 R23"串行单智能体"用户令与"终端质检补同源偏差"教训）。
> 三模式文字审在主会话执行；图表视觉审由两路独立 judge 子代理执行（恶意尺度标准，两轮循环）。
> 状态文件：`REVIEW_STATE.json`。本文件为累积日志（round 语义 = 审查→修复→复核循环）。

## 输入基线

- master=01f35c7（MEXP 收官），tag review-snapshot，PDF 42 页。
- 本轮范围：交叉审稿+对抗审稿+恶意批判+图表优化+减防御性写作（用户指令 2026-09-10）。

## 恶意批判审（reject 人格）findings

| # | sev | 发现 | 处置 |
|---|-----|------|------|
| M1 | fatal | tab:main 主结果表内 `\ref` 被源码换行劈断，PDF 实渲染 "Section ef-"（全稿最醒目表格） | 已修（04 tex） |
| M2 | fatal | Highlights 区 "≥3imes/2.51imes"——cas-sc 类 LaTeX3 展开吞 `\times` 前缀（`\t` 为 LaTeX3 tab 缩写）；同页摘要正常形成对照 | 已修：main.tex+highlights.tex 双处改纯 x 记法（与正文 15.8x 风格一致） |
| M3 | major | 摘要 ≥3× 主张后 90+ 词双层嵌套括号披露堆叠（claim-then-caveat 反噬形态） | 已修：重组为两短句，全部事实保留（fails/PARTIAL 口径不降） |
| M4 | major | MEXP 段引用 "three-seed observation" 但正文从未载三 seed M 臂数字（悬挂引用，审稿人找不到出处） | 已修：MEXP 段内补 "At the frozen three-seed budget the matched-solver arm already outscored the coupled arm (mean 82.6% vs 79.8%)" |
| M5 | major | 工件 r23-mexp json 的 s49 wall=3875.1s 系续训漏段（真值≈7933s），而新段称 "wall-times in the released artifact" | 已修：段内改为 host-restart 分段记账指向 + commit 记录修正值；冻结规则下 wall 不进 verdict 故工件不改 |
| M6 | major | Data availability 称 review-snapshot tag "immutable"，与治理惯例（每次收官 `git tag -f` 重锚）矛盾——审稿人克隆仓库即可证伪 | 已修：改为 "re-anchored against the dated commit chain" 表述 |
| M7 | minor | L7 引用无上下文时易被读作行号（版面 judge 实际误读为"源码行号残留"） | 部分缓解：per-class 义务句改指 "the L7 class-imbalance caveat" 等正文化表述；Limitations L1-L13 编号制保持 |
| M8 | minor | PanAf 30-seed 85.8% 与 10-seed 决策数 88.7% 并呈 | 保留（E9d 段已明确 10-seed 为决策、30-seed 为 gap 诊断）；但见 X-fig5 |

**拒稿概率主观评估（修复前 42 页版）**：约 60-65%（PR 初审）——主要风险面=版面级硬伤（M1/M2 直接暴露"未做终审"观感）与摘要观感（M3）。

## 交叉审稿（数字脊柱/图表文三方）findings

| # | 结果 |
|---|------|
| X1 | E9 全链复算通过：67.53/74.45=90.7 ✓ 66.05/74.45=88.7 ✓ +1.48=67.53−66.05 ✓；E9b 48.3/54.3=88.9 ✓；E9c 15.4/23.1=66.6 ✓；E9d 53.2/60.0=88.7 ✓ 51.5/60.0=85.8 ✓；MEXP 82.6/79.7/+2.9 与工件 mean 82.57/79.71/2.86 四舍五入口径一致 ✓；p=0.002=0.00195 报告惯例 ✓；15.8x=9897/626 ✓ 32.5x=1520/47 ✓ |
| X2 | tab3 与 E1-E4 正文数字逐项一致 ✓；fig5 caption 四基准 90.7/88.9/88.7/66.6 与正文一致 ✓；tab2 各行与正文一致 ✓ |
| X3 | **fig5 再生成陷阱（修复）**：p23-panaf 工件被 Amendment 2 扩为 30 seeds 后，重绘脚本 b_arms 全量读到 30 → 图上 PanAf 点 85.8 与 caption/决策数 88.7 不一致。旧 PDF 渲染早于文件扩展故侥幸一致。已修：fig5 取 b_arms[:10]（决策口径）并注记。教训入记忆：**数据文件可被扩展重写，重绘图必须锚定决策口径而非全量字段** |
| X4 | main.tex 注释"十二条 L1-L12"过时（正文 Thirteen/L13）→ 注释已更新；反防御适配块所称 "10 条 Limitations" 同样过时（套件侧文件，不动，记债） |
| X5 | 净评估口径复查：tab2/E7/E7b/E9 全为 eval 模式净值；文件名 acc 虚高（BN 双口径）未混入正文 ✓；13×=466/35 与正文 "13x" ✓ |
| X6 | 版面基线：2 overfull 均为既有（MEXP 轮日志同样存在），非本轮引入；p34 novelty 表 117pt overfull 待 judge 复核轮实证 |

## 对抗审稿（三 persona 迁移）findings

- **Saboteur（找破法）**：M5（工件对不上真实时间线）与 X3（图读被扩展的工件）为本 persona 最大收获——两处都是"再生成时数据源漂移"型缺陷，一次修复+一次制度性教训。检查了 GA 531×131pt 规格保持 ✓。
- **New Reader（读不懂）**：摘要 120 词单元句（M3 处理）；E7c 冒号长句保留（信息密度换可读性不划算）；tab3 "favoring generic-SSL" 方向词保留（表内语境明确）。
- **Integrity Auditor（诚实落差）**：headline 90.7% 的"linear-only 88.7% +1.48pp"锚句在摘要/intro/结论三处齐备 ✓ 非冗余不可删（R22 头条必配检验裁决）；L13 "no superiority claim anywhere" 与全文抽查相符（2.51×/1.61× 均内参照）✓；per-class 义务句收敛到 L7 唯一义务源 ✓。
- 三 persona 交叉命中提升：无（各发现单一来源，已按最高 sev 记账）。

## 图表专项（用户最高优先）

第一轮（独立 judge，恶意尺度）：**6/6 fail**。
- fig2：κ<τ 分叉画在 Re-estimate→AL queue 弧（拓扑错，spec 分叉点在 Assign）→ **v6 双环重设计**（自动环左逆时针 + 人工分支右链 + 写回辐条）。
- fig4：线性 x 轴承载 20→200 指数预算域比例失真 + 橙压青 + 基线文字骑线 → **v2 log 轴 + ±4.5% dodge + 白底注**。
- fig1：反向弧箭头与"单向通信"矛盾（major）+ 左容器失衡 + 贴边 + 虚线落点含糊 → v6 四修。
- fig3：灰基线标签叠青柱 + 图例归属含混 + 缺 frame index → v3 三修（ytick 配色绑定轨道）。
- fig5：顶部点簇净空<3pt（历史问题复发）+ 三系列同深板岩 → v2 白描边+位移+紫色分离+**X3 口径回正**。
- GA：canine 条无标数与 near-chance 注量纲混读 + NTU 条撞 Φ 青色 + 悬空箭头 → v2 标数注绝对值+灰系对齐 fig5+弧化。
caption 同步：fig2 双环描述重写、fig3 去 "not a best-of cherry-pick"、fig4 加 log/dodge 说明、fig5 位移清单更新（8.0/13.2 新增，PanAf 12.5→13.2）。

复核轮：两路 judge 进行中（v2 产物），循环直至全 pass。

## 减防御性写作三件套（技能输出契约）

**改写文本**：37 处已落地（脚本化 assert 唯一命中，见 git diff）。
**六类分类清单**：
- 类 6 冗余澄清（删）：`we cannot exclude unpublished concurrent work`（L3 已有正式边界）/`We state this claim with that explicit boundary...`/`rather than hiding it`/`disclosed as such`×5/`we disclose`×4/`An honest boundary/attribution boundary... applies` 引导语×2/`reported as-is`×3/`careful not to over-read`/`No rescue claim is made anywhere`/`replaces a would-be hand-wave`/`we make no sampling-efficiency claims`（与段题重复）/`not a best-of cherry-pick`（选择规则已明示）/`to forestall cherry-picking`/`rather than fabricate`/fig2 caption `palette axis independent...`。
- 类 5→正面化（改写不删）：摘要双层括号重组；intro L22 括注压缩为 "fails at small-backbone scale, PARTIAL at backbone scale"；E4 per-class 句简化（义务源移 L7）；E7b `Disclosed leakage boundary:`→`Leakage boundary:`；registration 边界句正面化。
- 类 2/3 必要限定（**全部保留**）：三层口径、13 条 Limitations、single-draw spread 口径、PARTIAL/fails 判据词、净评估口径、E6-real/E6-real-2 **冻结段措辞零改动**（预注册协议红线："冻结判据段落的措辞不改动"——E6-real-2 段的 NaN/重启/记账程序性披露密度虽高但协议不可动，残余防御感留在此两段是已支付的协议成本）。
**保留限定理由表**：逐句命中五项必要条件（论断有效性 12/证据解释 6/应用范围 9/研究设计 5/读者正确使用 5），完整表见 git diff 各段注释与本节分类；拿不准一律保留（技能铁律）。

## 治理债（上报不擅动）

1. docs/paper/*.md 真源自 R22 文本补丁轮起未随 tex 同步（fixfull/MEXP 写回均直改 tex）——truth 单一性条款事实失效，建议用户裁决：追平 md 或正式改声明 truth=tex。
2. 套件 anti-defensive-writing 适配块所称 "10 条 Limitations" 过时（现 13）。
3. author/GenAI 人工项不变（USER 占位为投稿前替换项，评审版保留）。

## 图表迭代全记录（六轮 judge 循环，2026-09-10）

| 轮 | judge 结论 | 修复动作 |
|----|-----------|---------|
| R1 | 6/6 fail（fig2 拓扑错/fig4 轴失真重绘级；fig1/3/5/GA 精修级） | fig2 v6 双环、fig4 v2 log+dodge、fig1 v6（**写了未跑**）、fig3 v3、fig5 v2（含 PanAf 口径回正）、GA v2 |
| R2 | fig1 假修复暴露（脚本未执行）；fig2 双箭头并排+长弧穿 hub；fig3 图例挪位反叠 Ep4 标签；fig4 dodge 致刻度失配+图例撞注；GA 弧穿字；fig5 pass | fig1 补执行+配平 v7、fig2 绕行折线、fig3 ylim 抬高、fig4 去 dodge 空心标记、GA 弧下压 |
| R3 | fig1/fig5/GA pass；fig2 折线穿盒+箭头重影；fig3 图例锚点未到位；fig4 注记撞图例 | fig2 彻底改双行管线 v7（弃圆环）、fig3 图例左上、fig4 图例中左+注记缩短 |
| R4(自验) | 作者目检抓到 arc3 rad 符号错（回环弧沉盒后被遮）+re-anchor 被盒吃 | 折线三段显式坐标、删 re-anchor |
| R5 | fig2 副标签越框×2+辐条悬空；fig4 注记仍越右界；fig3 图例与柱标签贴近（页 judge）；fig2 caption 方位词未随布局更新 | 副标签缩短、辐条改算 hub 盒缘、注记右锚定、caption 改 top/bottom row、fig2 图例上移、fig3 图例出轴下横排 |
| R6(终验) | 运行中 | — |

**制度性教训（入记忆）**：①脚本写了≠跑了——图修复必须执行+验证产物 mtime，否则 judge 对着旧图空转一轮；②数据工件会被后续扩展重写（p23 30-seed 覆盖 10-seed），重绘图必须锚定决策口径字段而非全量读取；③arc3 rad 符号靠推理不可靠，几何敏感处用显式坐标折线；④图例位置是高频碰撞源，优先出轴放置；⑤双 judge（图细验+页版面验）独立收敛后结论可信度高，单 judge 的通过需另一路佐证。

## 终验与冻结（R7-R8）

- R7（v9 微审）：fig2（Assign 盒加宽）/fig4（(a) 注记移入下方空带+实白底）闭环 pass，pg-38 pass；pg-40 映射错位系 staging 页码漂移，非缺陷。
- R8：v8 页面轮五页（37-41）全 pass 可冻结；pg-41（al_efficiency 真实所在页）由作者目检 v9 渲染确认（注记无穿字、caption 数值逐项吻合、图题同页）。
- **六图终态：fig1/fig2/fig3/fig4/fig5/GA 全 pass；版面 pg-01/07/24/25/34/37-41 全 pass。冻结成立。**
- 页脚 Page N of 41 与物理页恒 -1 偏移 = cas-sc 首页不计数模板行为，全稿一致，非缺陷。

## 状态

终验通过，簿记提交+tag 重锚+记忆回流后闭环。
