# Auto-Review-Loop Round 1（2026-09-03 · 21 页成稿五视角评审）

> 对象: docs/paper/latex/（六节成文 + tab1/tab2/tab3 + fig1/fig4 + Algorithm 1 + 五节声明区, 21 页 PDF 编译零错）
> 方法: ars-academic-paper-reviewer 五视角 + ars-adversarial-reviewer 强制发现制（每人格至少一条, 禁止 LGTM）
> 预扫描: \citep↔bib 对账零缺失 / 排版面零内部路径 / TODO 残留=用户终审项（红色醒目合规）

## Round 1 发现清单

| # | 视角 | 发现 | 级别 | 处置 |
|---|------|------|------|------|
| 1 | 🔨方法论破坏者 | L5 的 "extended-schedule cold-start rerun" 实为 **full-budget 档**（epochs 120 vs 短档 50; 12.3/7.8pp 复算自 p05-al-efficiency-full JSON 全部吻合）——审稿人按字面理解为"延长调度"会追问训练预算差异，措辞有歧义 | MINOR | 05/06 两处 "extended-schedule" → "full-budget schedule (120 epochs vs. 50)" 措辞精化 |
| 2 | 🆕新人审稿人 | S4 五臂段与 S5 消融表的 AL 行、L5 三处出现同一组数字（7.9/7.1/12.3/7.8/4.2-5.0）——读者第三次遇到时会怀疑 padding | MINOR | S5 AL 行已是最简形态; S4 五臂段不含 AL 数字✅; L5 是 Limitations 本职✅——**维持现状**, 记录为"三处出现各司其职"判定 |
| 3 | 🆕新人审稿人 | tab2 无表格编号交叉引用（正文未 \ref{tab:main}）——cas-sc 模板自动编号后正文不引用会被批"表格孤儿" | MAJOR | S4 正文加 "Table~\ref{tab:main} consolidates..." 引用句 |
| 4 | 🕵️主张审计员 | 结论段 "82.0% top-1 on 22 classes from a 20-clip budget" 未随 tier 标注（synthetic-offset）——与摘要口径不一致（摘要有随句） | MINOR | 06 结论段补 "(synthetic-offset tier)"——已有, 复核确认在; ✅零变更 |
| 5 | 🕵️主张审计员 | refs.bib 13 条 "[author list pending Scholar review]" 占位作者——投稿前必须换真实题录, 红线（禁凭记忆生成作者名单）维持, 但需导出待补清单防遗漏 | MAJOR | citation_checker 阶段导出 13 条 Scholar 清单（下一工序） |
| 6 | 😈魔鬼代言人 | 端到端五臂窄带结果（warm_full≈scratch_full, 全参微调无优势）写进 S4 是给审稿人递刀? | MAJOR | 反方论证维持: 该段以"数据瓶颈证据"口径呈现且与梯度消融互证——**先发制人的诚实展示优于被发现**; 措辞再硬化一处: "independently reproduces" → "corroborates"（窄带现象与梯度消融是相关佐证非独立复现, 不同数据集不同协议） |
| 7 | 🎯主编 | 摘要四要素齐但缺 "NTU 等价性" 一句——审稿人视角这是 R4 防线证据非贡献, 但 PR 审稿重实现类稿件会先找实现可信度信号 | MAJOR | 维持"R4 不进 Abstract"纪律（成文纪律③）; 实现等价性在 S4.4 有完整段, 结论段已有一句——**摘要不加**, 记录判定理由 |
| 8 | 🎯主编 | Data Availability 的 NTU RGB+D 许可证【终审-B】注释与 S4.4 "publicly available" 表述有张力——若许可留档未确认, "publicly available" 措辞过硬 | MINOR | S4.4 无 "publicly available" 字样（已核）; DA 句用 "obtained under the provider's research-use license terms" 已是准确表述——✅零变更, 终审-B 照旧 |

**裁决**: 无 CRITICAL; 3 MAJOR（#3 表格孤儿 / #5 Scholar 清单 / #6 措辞硬化）+ 4 MINOR。

## Round 1 修复提交

- #1: extended-schedule → full-budget schedule（05/06 两处）
- #3: tab2 正文引用句（04）
- #6: independently reproduces → corroborates（04）
- #5: 转 citation_checker 工序
- #2/#4/#7/#8: 复核后判定零变更, 理由留痕如上

## 复审计划

Round 2 聚焦: 修复项复核 + 摘要/Intro/S2 逐句句长与 AI 腔扫描（前两轮未覆盖的文风层）。

---

# Round 3（2026-09-03 · anti-defensive-writing 防御性写作专项）

> 工具: 本轮起用新融入技能 anti-defensive-writing（Kiterlin 上游, MIT; 见工具箱 skills/anti-defensive-writing/）
> 方法: 十项检测清单模式化扫描 + 高影响力位置 hedge 词密度 + 保护句式密度 + 人工六类分类判读
> 保护区声明: 10 条 Limitations、三层口径标注、[CITATION-NEEDED]、首次性边界声明 = 真实方法学限定（第 3 类），一律保留

## 扫描结果

| 扫描项 | 结果 |
|--------|------|
| 十项检测清单（does not claim / worth noting / to be clear / not-X-but-Y 滥用 / 双重转折 / 防批评解释等） | **0 命中** |
| 高影响力位置 hedge 词（abstract/贡献 bullets/结论第一段） | **0 命中**（"overall accuracy" 为指标术语，非对冲） |
| "is disclosed as such" 保护句式 | 4 处（04×2 表格与对比段、05×1 tab3、06×1 L5），全部位于正确章节 |

## 六类分类判读

| 类别 | 数量 | 处置 |
|------|------|------|
| 1 不必要免责 | 0 | — |
| 2 必要范围条件 | 多处 | 保留（tier 标注随行） |
| 3 真实方法学限定 | 多处 | 保留（Limitations/口径/边界） |
| 4 有用概念对比 | 少量 | 保留（"frozen physics, revisable semantics" 等对比本身是论点） |
| 5 基于证据的限定 | 多处 | 保留（"within 0.93pp" 等） |
| 6 冗余澄清 | 0 必修 | "is disclosed as such" ×4 属统一纪律句式——**保留统一不变体**：审稿人识别其为系统性纪律而非疏忽（防 R2#4 元数据不一致教训重演） |

## 裁决

**零必修项。** 成稿防御性写作已达到该技能的 Final Pass 标准（deliver text free of unnecessary disclaimers）——归因于项目自大纲阶段起的正向框架纪律（outline 措辞纪律 v0.3、claim ledger 门禁）。本技能在本稿的价值 = 提供了系统性验证手段（而非修复）；后续修订轮（审稿回复信）为主要应用场景。
