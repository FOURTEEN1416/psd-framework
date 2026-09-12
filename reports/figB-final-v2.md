# figB 终报告 v2 · 阶段 I（方向 C → C+ 强化）全量实施

> 窗口：`wt/figB`（worktree `D:\Desktop\psd-framework-figB`）
> 基线：`806f78d`（master） · 当前 head：`a38b0b4` · 提交 20 笔 / 相对 master 47 文件
> 本轮：用户裁定「六张全部重画」，风格方向 **C+（方向 C 的强化版）**
> 范围：fig1–fig5（印刷 Figure 1–5）+ GA（graphical abstract）；**双栏资产不在本轮范围**

---

## 一、本轮做了什么（相对上一轮交付的增量）

上一轮（方向 C）已把五图做完尺度修复 + 分阶色标。本轮 C+ 在其上再收紧三件事，
并把此前因属「待裁决资产」而**故意未动**的 GA 一并重画，同时修掉 GA 的语义反转缺陷。

| 图 | 版本 | 变化 | MediaBox（宽 × 高 pt） | 字号下限 |
|---|---|---|---|---|
| fig1 框架总览 | v15→**v16** | 面板加宽 44→48 单位；标题 6.8→7.0（并修 v15 实测越框 0.9 单位的净空违规）；焦点块 2→1；流线 1.6→1.2 | 246.24 × 216.00 | 5.5→**5.80** |
| fig2 伪标签环 | v18→**v19** | 站卡去边（仅靠色阶）；焦点 2→1（Label pool 降为浅底，hub 保留深块——caption 明文 dark hub）；`κ≥τ` 段标签从段下移到段上（修与两侧站注「连读成串」的真缺陷） | 246.24 × 194.40 | 5.5→**5.50** |
| fig3 分割定性 | v9→**v10** | 修 v9 缺陷：`set_yticklabels` 被后续 `tick_params` 覆盖导致字号失效；左边界距 0.155→0.185（修刻度标签出画布）；ylim 0.86→0.95 加顶部余量使 agg 注记与图例各自净空 | 246.24 × 237.60 | 5.5→**5.80** |
| fig4 AL 效率 | v7→**v8** | 守 3.66in 硬上限**不加高**，留白改走面板间距（hspace 0.32→0.40）；误差棒 0.9→0.6、曲线 1.4→1.0 | 246.24 × 263.52 | 5.5→**6.20** |
| fig5 预算保留 | v8→**v9** | 加高 2.75→2.95in（ylim 不变 ⇒ 净空单调改善）；tonal ring 8.6→7.6、marker 6.0→5.2 | 246.24 × 212.40 | 5.5→**5.60** |
| GA 图形摘要 | v15→**v16** | **修 D5 语义反转**（见下）；接口边标签 4.9→5.6、100% 注记 4.6→5.6；曾试加高至 1.95in 但 MediaBox 变 531×140.4pt 超 131pt 上限 9.4pt，**已收回** | 531.00 × 131.04 | 4.6→**5.60** |

五张单栏图 MediaBox 宽**全部严格 246.24pt = 版心宽**，LaTeX 装入系数 1.000
（修复前 1.004–1.118）。GA 531×131.04pt 在投稿系统 131pt 上限内。

---

## 二、D5 修复（本轮的语义级改动）

**问题**（提案期已放大确证）：v15 的演化回线 dashed 从 Ω 底绕到 Φ 底，
`FancyArrowPatch((21.0, 11.4) → (21.0, 12.4))` 在 **Φ 底端朝上指向 Φ**。
视觉语义 = "Φ 被重训"，与图内同时存在的两处文字直接冲突：
- `Physics layer Φ (frozen)`
- `only Ω retrains`

**v16 处置**：
1. 删掉指向 Φ 的箭头；
2. dashed 改为**单横段**从 Ω 底左折、左端上行至 Φ 底但**无箭头**，
   改以 `×` 形终止符明示"Φ 不接收重训"；
3. **Ω 端保留朝上实心箭头**，明示重训闭环终点在 Ω；
4. 新增**本地语义断言**（防回归）：遍历 `ax.patches` 中全部 `FancyArrowPatch`，
   断言不存在终点落在 Φ 块内（x≤39, y≥12.9）的箭头。

实测输出：
```
[local D5] ga-v16: 2 FancyArrowPatch, 指向 Phi 的箭头 0 个 -> PASS
[local D5] 闭环终点 = Omega（x>39），Phi 底为 x 形终止符 —— 与 'only Omega retrains' 一致
```

---

## 三、门禁证据

六脚本复跑原始 stdout：`reports/figB-proposal/gates-final-v2.log`（167 行，0 FAIL，EXIT 全 0）。

```
[G2 label-collision] fig1-v16: 15 labels -> PASS
[G5 typography] fig1-v16: 15 texts, min fontsize 5.80pt -> floor 5.5pt PASS
[G6 marks-in-axes] fig1-v16: 0 marks -> PASS
[G4 file] fig1_framework_overview.pdf: MediaBox 246.24 x 216.00 pt -> PASS
[local clearance] Physics layer title   span 6.26..43.74 | panel 1..49 | clear L=5.26 R=5.26 -> PASS
[local clearance] Semantic layer title  span 63.12..106.88 | panel 61..109 | clear L=2.12 R=2.12 -> PASS

[G2 label-collision] fig2-v19: 26 labels -> PASS
[G5 typography] fig2-v19: 26 texts, min fontsize 5.50pt -> floor 5.5pt PASS
[G4 file] fig2_pseudo_label_loop.pdf: MediaBox 246.24 x 194.40 pt -> PASS
[local clearance] card assign    name+sub inside box, min clearance = 2.77 px
[local clearance] card verified  name+sub inside box, min clearance = 7.32 px
[local clearance] fig2-v19: 6 station cards -> PASS

[G5 typography] fig3-v10: 37 texts, min fontsize 5.80pt -> floor 5.5pt PASS
[G4 file] fig3_segmentation_qualitative.pdf: MediaBox 246.24 x 237.60 pt -> PASS
[local clearance] fig3-v10: 0 in-segment labels, all inside own segment -> PASS

[G3 whisker-crossing] fig4-v8-a: 2 labels x 8 errbars -> PASS
[G3 whisker-crossing] fig4-v8-b: 1 labels x 8 errbars -> PASS
[G5 typography] fig4-v8: 26 texts, min fontsize 6.20pt -> floor 5.5pt PASS
[G6 marks-in-axes] fig4-v8: 48 marks -> PASS
[G4 file] fig4_al_efficiency.pdf: MediaBox 246.24 x 263.52 pt -> PASS

[G3 whisker-crossing] fig5-v9: 9 labels x 8 errbars -> PASS
[LOCAL leader-clean] text=0 leaderXleader=0 nearMarker=0 crossErrbar=0 -> PASS
[G5 typography] fig5-v9: 20 texts, min fontsize 5.60pt -> floor 5.5pt PASS
[G4 file] fig5_budget_retention.pdf: MediaBox 246.24 x 212.40 pt -> PASS

[G5 typography] ga-v16: 18 texts, min fontsize 5.60pt -> floor 5.5pt PASS
[G4 file] fig_ga_graphical_abstract.pdf: MediaBox 531.00 x 131.04 pt -> PASS
[local D5] ga-v16: 2 FancyArrowPatch, 指向 Phi 的箭头 0 个 -> PASS
```

**G1 冗余豁免 WARN**（有 redundancy 声明即降级为 WARN，不计失败）：
fig1/fig2/fig3/fig4 各 1 对，fig5 14 对——与提案期同口径，**未增多**。

**fig3 的一条自检是「空过」，如实登记**：`0 in-segment labels`。
经核查 v8/v9/v10 三者该集合**均为空集**（段长 17–225 帧 vs 门控 need 209–389 帧），
非本轮引入。属既有状态，登记为可选后续项（若要显示段内标签，需放宽门控或缩短标签）。

---

## 四、编译断言

`pdflatex ×4 + bibtex`（MiKTeX）：

| 断言 | 结果 |
|---|---|
| TeX 错误（`^!`） | **0** |
| 未解析引用 `??` | **0** |
| `USER` 标记 | **0** |
| `LaTeX Warning` | **0** |
| Overfull / Underfull | 1 / 29（唯一 Overfull 117.08pt @ line 135，来自 MiKTeX `ls2stixtt.fd` 字体装载，与图件/正文无关，属环境既有项） |
| 页数 | 42 |
| 图 1–5 落位（印刷页） | **p4 / p10 / p15 / p20 / p31，零回退** |
| 颜色词一致 | `03-method.tex` 1×teal；`04-experiments.tex` 1×orange + 1×teal —— 与图内语义色映射逐词对应（fig2：橙=automated / teal=human；fig3：橙=seed pseudo-GT 带 / 青=SMQ 带），全部承袭前版 |

> 落位核对口径：**印刷页码 = PDF 物理页 − 1**（前言部分偏移）。实测物理页
> 5/11/16/21/32 → 印刷页 4/10/15/20/31。
> 另注（既有状态，非本轮变更）：仓库文件名与印刷图号**交叉**——
> `fig5_budget_retention.pdf` 是印刷 Figure 4（p20），
> `fig4_al_efficiency.pdf` 是印刷 Figure 5（p31）。落位断言按**印刷图号**口径。

---

## 五、数据零改动核验

- 五脚本数据逻辑逐位承袭前版；fig5 v9 的 8 个点位与 v6 逐位一致（脚本尾部打印留痕）。
- **冻结清单原样保留**：`90.7 / 88.9 / 88.7 / 66.6 / 41.5 / 82.0`、
  `6.07× / 15.8× / 32.5×`、PARTIAL/FAILS 判定、fig4 的 `random ≥ entropy` 负结果
  （脚本内 `assert` 保留，实测 True）、GA 的 `90.7 / 88.9 / 28.9` 与
  `canine 9.8 vs 11.1% chance @13% budget`。
- 未改：`psd_style.py`、任何 `.tex`、任何 caption、任何图号顺序。

## 六、动了哪些文件

`git diff --name-only 806f78d HEAD` = 47 文件，其中本轮的 6 笔提交为：

```
a38b0b4 feat(ga): v16 方向 C+ 强化重画 + 修 D5 箭头语义反转
784fb9c docs(figB): 风格方向 C+ 全局规格落档
22dbae1 feat(fig5): v9 方向 C+ 强化重画
f9defba feat(fig4): v8 方向 C+ 强化重画
ee3149d feat(fig3): v10 方向 C+ 强化重画
f388796 feat(fig2): v19 方向 C+ 强化重画
353571e feat(fig1): v16 方向 C+ 强化重画
```

每图为独立提交（精确 `git add` 脚本 + PDF + PNG + 预览），无攒批。

---

## 七、自检清单

- [x] 六图逐张重画完成，每图独立提交
- [x] 每图 G1–G6 全 PASS（fig3 无误差棒故无 G3；GA 无坐标轴故只调 G1/G2/G5/G6/G4）
- [x] 六图 150dpi 预览逐张本人实看（fig1–fig5 + GA）
- [x] caption 逐词对照（fig1 `retrains Ω alone`；fig2 top/bottom row、dark hub、`(P,Ω,A)`、κ 两分支；fig3 上带橙/下带 teal、4/4 above baseline；fig4 两面板标题、`random leads by 4.2–5.0 pp`；fig5 七 tier 文字）
- [x] 数据零改动 + 冻结清单保留
- [x] 尺度修复：五图 MediaBox 宽统一 246.24pt、装入系数 1.000；GA 531×131.04pt
- [x] 编译 0 错误 / 0 `??` / 0 USER / 0 警告；落位零回退
- [x] D5 修复并加防回归断言
- [x] 门禁日志留档 `gates-final-v2.log`
- [ ] 双栏资产 `fig1/fig2_*_2col` —— **未做**（不在本轮「六张」范围，见待裁决 D7）
- [ ] fig3 段内标签 —— **空过**（既有状态，见 §三）

## 八、待裁决遗留

| 项 | 内容 |
|---|---|
| **D6** | 新共享门禁 G5（排印：字号下限/互压/出画布）+ G6（标记落位）是否合并回 master（本轮实测拦下 fig3 字号失效、fig2 卡内净空 1.7px 临界等 G1–G4 盲区缺陷） |
| **D7** | 双栏资产 `fig1_framework_overview_2col` / `fig2_pseudo_label_loop_2col` 仍为旧风格，是否按 C+ 同步重绘 |
| **D8**（新增） | fig3 段内标签门控长期空过：是否放宽门控（need 209–389 帧 vs 实际段长 17–225 帧）使段内标签真正可见 |

原 D1–D5 均已消解：D1/D2（色板用法与图型）方向 C 不涉及；D3（字号下限 5.5pt）已随实施落实（六图字号下限 5.50–6.20pt）；D4（fig4 高度）已由 3.93→3.66in 解决；D5（GA 箭头）本轮已修。
