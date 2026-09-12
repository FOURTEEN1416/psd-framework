# figB 终报告 · 六图审美升级（阶段 I 全量实施）

> 窗口 wt/figB（base = master `806f78d`）；方向由用户拍板：**C 分阶色标 + 制图精度**。
> 本报告按任务书 §6 自检格式：先证明，再宣称。全部证据可复现（命令与日志见下）。

## 0. 结果一览

| 图 | 版本 | MediaBox 前（tight） | MediaBox 后（固定画布） | 装入系数 | 门禁 |
|---|---|---|---|---|---|
| fig1 框架总览 | v13 → **v15** | 245.2 × 186.4 pt | **246.24 × 187.20 pt** | 1.004 → **1.000** | G1/G2/G5/G6/G4 全 PASS |
| fig2 伪标签环 | v17 → **v18** | 245.2 × 169.2 pt | **246.24 × 169.92 pt** | 1.004 → **1.000** | G1/G2/G5/G6/G4 全 PASS |
| fig3 分割定性 | v8 → **v9** | 233.65 × 207.21 pt | **246.24 × 218.16 pt** | 1.054 → **1.000** | G1/G5/G6/G4 全 PASS |
| fig4 AL 效率 | v6 → **v7** | 227.28 × 260.92 pt（实排 3.93 in，超上限 7.3%） | **246.24 × 263.52 pt = 3.66 in 整** | 1.083 → **1.000** | G1/G3(双面板)/G5/G6/G4(max_h 263.6) 全 PASS |
| fig5 预算保留 | v6 → **v8** | 220.2 × 177.5 pt | **246.24 × 198.00 pt** | 1.118 → **1.000** | G1–G6 + 引线洁净度本地断言全 PASS |
| GA 图形摘要 | v15（不动） | 531×131pt 硬上限内 | 未动（用户裁决定稿，见待裁决 D5） | — | — |

**全部 5 张单栏图 MediaBox 宽严格 = 246.24 pt = 版心宽**：LaTeX 按 `\linewidth` 装入零缩放，图内字号所见即所得（提案期诊断的头号根因"同一声明字号跨图差 11.8%"就此消除）。

## 1. 逐图改动明细

### fig1 v15（方向 C 重绘）
- 图内文字**逐字承 v13**（面板标题×2 / 卡片×9 / 桥标签 / 底部条），零新增删除。
- 分阶色标：面板浅底 + 0.6pt 发丝边 → 卡片深一阶底 + 0.55pt 边 → 焦点 DEEP 满块（焦点块 = Semantic hub + Label pool ≤2 的纪律保持）。
- 字重阶梯：面板 bold 6.8 → 卡片 regular → 焦点白字 bold；桥标签 5.2 → 5.5pt（D3 字号下限修复）。
- 主流线统一 1.6pt/head11（接口主箭 1.8 → 1.6）；虚线仅留 Taxonomy 回路一种语义。
- 两条顶部长横箭终点 52.5 **精确触桥缘**（v13 悬停在空白里的缺陷修复）。

### fig2 v18（方向 C 重绘）
- 图内文字**逐字承 v17**（六站名/站注/chip/图例/流注），κ≥τ 标签修复保持。
- 站卡白底 → 同相浅底（mix 0.58）+ 更深同相描边（mix 0.30）；焦点 Label pool = SEM_DEEP 满块、hub = INK 满块（深色焦点 ≤2 保持）。
- 字号下限合规：chip 5.2 → 5.5、write-back 注记 5.4 → 5.5（D3）。
- 颜色语义与 caption 逐词对应：橙（SEM_* 系）= 自动环、teal（HUMAN_* 系）= 人工支，top/bottom row 方位不变。

### fig3 v9（尺度修复 + 显式版面）
- 数据/构图/文字/字号/色板/段内标签门控**全部承 v8**；aggregate 0.4577 ± 0.0488 与 caption 逐位一致。
- gridspec 显式边距（left=0.155 / right=0.972 / top=0.93 / bottom=0.06），轴外 a/b/c 面板字母与右缘 T 刻度全落画布内（G5 实证修复两轮后全绿）。
- 橙=种子带（SEM_FILL）、teal=SMQ 带（PHYS_FILL），方位与 caption 一致（upper/lower band 不变）。

### fig4 v7（尺度修复 + 高度上限合规）
- 数据/构图/文字**全部承 v6**：双面板 sharex、图级单行共享图例、0–100 诚实轴、机会线 4.5%。
- caption 数值逐点核对一致：69.9 vs 77.8 @b=100、80.9 vs 88.0 @b=200；**负结果事实保留并断言**（cold-start random ≥ entropy 原样诚实呈现）。
- v6 的 tight 墨迹框被 LaTeX 放大 1.083 倍、实排 3.93 in 顶破"fig4 高 ≤ 3.66 in"上限 7.3%；本版固定画布 3.42×3.66 in，MediaBox = 246.24×263.52 pt = 3.66 in 整，**高度精确合规**（对应提案待裁决 D4 的"修 tight 回落"路径，判为构图职能内，未加内容）。

### fig5 v8（方向 C 重绘，样板 v7c 的生产化）
- 数据零改动：JSON 直读、100·mean/ref 口径、dodge 位移（13.9/5.5/11.5/10.0/16.5/8.8/13.2/8.0）、PanAf 前 10 种子口径逐位承 v6；8 点位数值与 v6 逐位一致（脚本尾部打印留痕）。
- tonal ring 同色相浅阶环造点厚度；仅 public-real v1/v2 两远点系列走正交引线轨，其余就近直标；引线洁净度本地断言（穿字/相交/穿点零命中）。
- 预注册 PARTIAL 带 85–90%：极浅参考带 + 左端两行注记（实测文字宽度 0.479/0.784 in 锚定，与 synthetic-offset marker 环留 3.7 px 净空）；数值 85/90 取自 04-experiments.tex 既有预注册口径，**非新统计量**。
- `own full budget` 注记移至 100% 基线右端下方（原 va=bottom 位置被基线穿字，实看发现修复）。

## 2. 门禁证据（复跑原始 stdout）

留档：`reports/figB-proposal/gates-final.log`（2026-09-12 复跑，五脚本 EXIT 全 0、**零 FAIL**）。

关键行摘录：

```
[G2 label-collision] fig1-v15: 15 labels -> PASS
[G5 typography] fig1-v15: 15 texts, min fontsize 5.50pt -> floor 5.5pt PASS
[G5 typography] fig1-v15: pairwise-overlap + canvas-fit -> PASS
[G4 file] fig1_framework_overview.pdf: MediaBox 246.24 x 187.20 pt / embedded images = 0 -> PASS
[G2 label-collision] fig2-v18: 26 labels -> PASS
[G5 typography] fig2-v18: 26 texts, min fontsize 5.50pt -> PASS
[G4 file] fig2_pseudo_label_loop.pdf: MediaBox 246.24 x 169.92 pt -> PASS
[G5 typography] fig3-v9: 37 texts, min fontsize 5.50pt -> PASS
[G4 file] fig3_segmentation_qualitative.pdf: MediaBox 246.24 x 218.16 pt -> PASS
[G3 whisker-crossing] fig4-v7-a: 2 labels x 8 errbars -> PASS
[G3 whisker-crossing] fig4-v7-b: 1 labels x 8 errbars -> PASS
[G6 marks-in-axes] fig4-v7: 48 marks -> PASS
[G4 file] fig4_al_efficiency.pdf: MediaBox 246.24 x 263.52 pt -> PASS
[G6 marks-in-axes] fig5-v8: ... -> PASS（tonal ring 全部落轴域）
[G4 file] fig5_budget_retention.pdf: MediaBox 246.24 x 198.00 pt -> PASS
```

G1 均带冗余编码声明（marker 形状 + 直标等），色盲模拟豁免 WARN 数与提案期持平、未新增。五脚本 600 dpi PNG 全部再导出；150 dpi 预览逐张实看通过（`reports/figB-proposal/preview_fig*.png`）。

## 3. 编译断言（pdflatex ×4 + bibtex，稳态 main.log）

| 断言 | 要求 | 实测 | 判定 |
|---|---|---|---|
| TeX 错误行（`^!`） | 0 | **0** | PASS |
| 未解析引用 `??` | 0 | **0** | PASS |
| `USER` 标记 | 0 | **0** | PASS |
| LaTeX Warning（含浮动） | 0 | **0** | PASS |
| 图 1–5 落位 | p4/10/15/20/31 不回退 | Figure 1→p4、2→p10、3→p15、4→p20、5→p31（印刷页码；物理页 5/11/16/21/32 = 印刷+1） | PASS |
| 颜色词一致 | orange/teal 语义不变 | fig2 橙=自动/teal=人工（SEM_*/HUMAN_* 系）、fig3 橙=种子带/teal=SMQ 带（SEM_FILL/PHYS_FILL），与 caption 逐词对应 | PASS |
| 总页数 | — | 42 页（544.25×742.68 pt，elsarticle 版式） | — |

说明两点：
1. **仓库文件名与印刷图号是交叉的**（既有状态，非本次变更）：`fig5_budget_retention.pdf` 在正文是印刷 **Figure 4**（p20），`fig4_al_efficiency.pdf` 是印刷 **Figure 5**（p31）。落位断言按印刷图号口径全部命中。
2. 唯一 Overfull \hbox（117.08 pt）来自 MiKTeX stix 字体定义文件 `ls2stixtt.fd` 装载期（"detected at line 135"），与图件和正文无关，属环境既有项；浮动警告为零，符合"0 错误 0 浮动警告"要求。

## 4. 提交清单（git log 比对）

wt/figB 相对 master `806f78d` 共 **12 笔提交、37 个文件**，每图独立成笔（不攒批）：

```
f1beb8e feat(fig4): 正典 v7 尺度修复 + 高度上限合规（3.93 -> 3.66in）
5f364b3 feat(fig3): 正典 v9 尺度修复——固定画布 + 显式版面
35e3e09 feat(fig2): 正典 v18 方向 C 重绘——分阶色标 + 尺度修复
8dc994d feat(fig1): 正典 v15 方向 C 重绘——分阶色标 + 尺度修复
d99e1c5 feat(fig5): 正典 v8 方向 C 重绘——分阶色标 + 尺度修复
67e95fa docs(figB): 更正事故口径——git 写 ref 不落地（auto-gc 初判证伪）
daeb4ec docs(figB): 阶段 S0 风格方向提案（诊断 + 三方向 + fig5 样板 + 推荐）
861e43a chore(fig5): 三方向样板门禁留档与四联对比图
caff7e0 feat(fig5): 方向 C 提案样板 v7c（分阶色标 + 制图精度）
a8a0ad6 feat(fig5): 方向 B 提案样板 v7b（高密度双面板 small multiples）
a69689e feat(fig5): 方向 A 提案样板 v7a（编辑极简·单色骨架）
9235490 feat(fig-gates): 共享门禁增补 G5 排印与 G6 标记落位
```

文件域合规：改动只落在 `docs/paper/figures/`（脚本 + 产物）与 `reports/figB-*`；**main.tex、sections、refs、psd_style.py、reports/ 其他文件零触碰**（`git diff 806f78d HEAD -- <域>` 验证为空）。旧版脚本（v13/v17/v8/v6）与提案期产物（v7a/b/c、四联图）全部保留未删。

## 5. 未完成项 / 不动项

1. **GA 未动**（用户裁决定稿 v15；其反馈虚线箭头指向 frozen Φ 的语义反转仍在，见待裁决 D5）。
2. **双栏资产未同步**：`fig1_*_2col` / `fig2_*_2col` 仍为旧风格；若投稿系统需要双栏替换件需另行指示（单栏正典已是方向 C）。
3. **D2 方向 B（双面板拆分）未实施**——用户拍板 C，B 的图型变更（含 caption 连带）自然搁置。

## 6. 待裁决项（登记看板）

- **D5（保留）**：GA 唯一反馈虚线箭头指向被标注 frozen 的 Φ 且与图内 "only Ω retains" 矛盾（语义反转，已放大确证）。GA 属不动资产，是否单列小修待定。
- **D6（保留）**：新门禁 G5（排印：字号下限/互压/出画布）+ G6（标记落位）是否合并回 master——本轮已实测拦下 3 类 G1–G4 盲区缺陷（双轴名重印叠字、面板字母出画布、色键点压字）。
- **D3（已随实施落实）**：fig2/fig5 字号下限统一 5.5pt（fig2 chip 5.2→5.5）；属构图职能内，特此报备。
- **D4（已按"修 tight 回落"路径解决）**：fig4 高度 3.93 → 3.66 in 整，未加内容、未放弃上限。
- **新增 D7**：双栏资产（fig1/fig2 _2col 版）是否需要按方向 C 同步重绘。

## 7. 自检清单（先证明，再宣称）

- [x] 动了哪些文件：§4 清单（12 commits / 37 files），逐笔 `git add <精确文件>`，无全量 staged。
- [x] 门禁证据：`gates-final.log` 原始 stdout 复跑留档（§2 摘录，零 FAIL、EXIT 全 0）。
- [x] 编译证据：§3 断言表（0 错误 / 0 ?? / 0 USER / 0 警告 / 落位不回退 / 颜色词一致）。
- [x] 数据零改动：五脚本数据逻辑逐位承袭对应正典前版（各 commit message 逐一声明；fig5 8 点位打印留痕）。
- [x] 冻结清单未弱化：90.7/88.9/88.7/66.6/41.5/82.0、6.07×/15.8×/32.5×、PARTIAL/FAILS、random≥entropy 负结果——全部原样保留（fig4 断言在脚本内）。
- [x] 正典产物与提交逐字节一致（复跑取证后还原备份，`git diff` 为空）。
- [x] 工作区收尾干净：仅剩本报告与 gates-final.log 待提交（本笔提交后归零）。
