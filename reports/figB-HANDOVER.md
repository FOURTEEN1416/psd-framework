# figB 窗口交接文档 · 六图审美升级（全周期）

> **窗口**：`wt/figB` · worktree `D:\Desktop\psd-framework-figB`
> **基线**：`806f78d`（= master = origin/master，全程未动）
> **最终 head**：`ca086e8` · 提交 **23 笔** · 相对 master **57 文件**
> **状态**：全部任务完成，工作区干净，等待协调窗收编
> **写作时间**：2026-09-12 15:30 前后

---

## 一、这个窗口做了什么（一句话版）

用户对论文六张图不满（"丑"）→ 本窗口把"丑"翻译成三类**结构性**工程问题
（tight bbox 尺度失控 / 层级靠颜色抢通道 / 空区与漂字共生）→ 提三方向样板 →
用户拍板 **方向 C**（分阶色标 + 制图精度）→ 五图实施 → 用户追加"六张全部重画
+ C 强化"→ **C+ 轮**六图重画 + 修 GA 语义反转 → 用户裁决 D6/D7/D8 →
D7 双栏资产同步完成、D8 有证据地不放宽、D6 随收编自动生效。

## 二、裁决记录（本窗口收到过的用户拍板）

| 裁决 | 内容 | 执行结果 |
|---|---|---|
| S0 拍板 | 方向 C（v7c：tonal ring + 引线 + 预注册带） | ✅ 五图 v15/v18/v9/v7/v8 落地 |
| 追加拍板 | 「六张全部重画」+ 方向 C 强化版 | ✅ C+ 轮：fig1 v16 / fig2 v19 / fig3 v10 / fig4 v8 / fig5 v9 / GA v16 |
| **D6** | 新门禁 G5/G6 是否并回 master → **按推荐：采纳** | G5/G6 已在 `make_common_gates.py`（分支内），**随收编并入 master，无需单独操作** |
| **D7** | 双栏资产是否同步 C+ → **按推荐：同步重绘** | ✅ `3f761c7`（fig1 v16_2col）+ `ca086e8`（fig2 v19_2col），MediaBox 504×212.4 / 504×208.8pt |
| **D8** | fig3 段内标签门控是否放宽 → **按推荐：先核实** | 核实结果 = **不放宽**。证据：caption 原文 *"band colors encode segmentation quality, not class semantics"*——边界匹配是 label-agnostic 的，段内语义标签显示出来反而误导。门控空集是**语义正确**的既有状态，非缺陷。D8 关闭。 |

原 D1–D5 全部消解（详见 `reports/figB-final-v2.md` §八）。

## 三、最终产物清单（相对 master 的 57 文件）

### 正典六图（docs/paper/figures/，每图脚本+PDF+600dpi PNG+150dpi 预览）

| 产物 | 生成脚本 | MediaBox（pt） | 字号下限 |
|---|---|---|---|
| fig1_framework_overview | make_fig1_overview_v16.py | 246.24×216.00 | 5.8 |
| fig2_pseudo_label_loop | make_fig2_pseudo_label_loop_v19.py | 246.24×194.40 | 5.5 |
| fig3_segmentation_qualitative | make_fig3_segmentation_qualitative_v10.py | 246.24×237.60 | 5.8 |
| fig4_al_efficiency | make_fig4_al_efficiency_v7→v8.py | 246.24×263.52（=3.66in 上限内） | 6.2 |
| fig5_budget_retention | make_fig5_budget_retention_v9.py | 246.24×212.40 | 5.6 |
| fig_ga_graphical_abstract | make_ga_graphical_abstract_v16.py | 531.00×131.04（131 上限内） | 5.6 |

### 双栏资产（D7）

| 产物 | 生成脚本 | MediaBox（pt） |
|---|---|---|
| fig1_framework_overview_2col | make_fig1_overview_v16_2col.py | 504.00×212.40 |
| fig2_pseudo_label_loop_2col | make_fig2_pseudo_label_loop_v19_2col.py | 504.00×208.80 |

### 共享模块（D6 的本体）

- `docs/paper/figures/scripts/make_common_gates.py`：新增 **G5 gate_typography**
  （字号下限/两两互压/出画布）与 **G6 gate_marks_in_axes**（PathCollection 落位，
  跳过 errorbar 的 LineCollection 占位假阳性）。实测拦下过：双面板轴名重印叠字
  168.8px²、面板字母出画布、fig3 字号失效、fig2 卡内净空临界——全是 G1–G4
  全绿状态下漏掉的。

### 报告与证据（reports/）

| 文件 | 内容 |
|---|---|
| figB-S0-style-proposal.md | 阶段 S 提案（诊断 + 三方向 + 推荐） |
| figB-final.md | 方向 C 五图终报（第一轮） |
| figB-final-v2.md | **C+ 六图终报（主报告）** |
| figB-cplus-spec.md | C+ 全局规格定义 |
| figB-proposal/figB-fig5-fourway.png | v6/v7a/v7b/v7c 四联对比 |
| figB-proposal/gates-final.log / gates-final-v2.log | 两轮终版门禁复跑原始 stdout |
| figB-proposal/preview_*.png | 全部 150dpi 预览 |
| figB-proposal/make_fig5_budget_retention_v7{a,b,c}.py | 三方向提案样板（留档） |

## 四、门禁与编译证据（全部可复现）

**门禁**：`gates-final-v2.log` = 六脚本复跑，0 FAIL、EXIT 全 0。
G1 冗余豁免 WARN：fig1–4 各 1 对、fig5 14 对（与提案期同口径未增）。
本地自检：fig1 面板标题净空、fig2 卡内净空 1.51–3.10px、fig5 引线洁净度
（text=0 / leaderXleader=0 / nearMarker=0 / crossErrbar=0）、GA D5 语义断言
（2 个 FancyArrowPatch、指向 Φ 的箭头 0 个）。

**编译**：pdflatex×4 + bibtex → **0 错误 / 0 `??` / 0 USER / 0 LaTeX Warning**，
42 页；图 1–5 印刷落位 **p4/10/15/20/31 零回退**；颜色词一致。
唯一 Overfull 117.08pt 来自 MiKTeX `ls2stixtt.fd` 字体装载，与图件无关。

**复现命令**（协调窗可直接粘贴）：

```bash
cd D:\Desktop\psd-framework-figB
D:\Desktop\psd-framework\.venv\Scripts\python.exe docs\paper\figures\scripts\make_fig1_overview_v16.py
# ...其余五个脚本同理；门禁输出在 stdout
cd docs\paper\latex ; pdflatex main ; bibtex main ; pdflatex main ; pdflatex main
```

注意：复跑脚本会因 PDF CreationDate 改变产物字节——**门禁验证用临时跑，
正式产物以 git 里的版本为准**（本轮做法：备份 → 复跑取证 → 还原）。

## 五、硬边界遵守情况（自检）

- **数据零改动**：五脚本数据逻辑逐位承袭；fig5 八点位与 v6 逐位一致。
- **冻结清单原样**：90.7/88.9/88.7/66.6/41.5/82.0、6.07×/15.8×/32.5×、
  PARTIAL/FAILS、fig4 `random ≥ entropy` 负结果（脚本内 assert 保留）、
  GA 90.7/88.9/28.9 与 9.8 vs 11.1% @13%。
- **未碰**：psd_style.py、main.tex、sections/*.tex、refs、图号顺序、caption。
- **D5 修复**是唯一语义级改动（GA 箭头），有防回归断言且经用户裁决路径登记。

## 六、给协调窗的收编指引

1. **合并**：`wt/figB`（ca086e8）→ master。23 笔提交、57 文件，历史线性
   （全部基于 806f78d，无分叉）。合并后 G5/G6（D6）自动进入 master——
   **其他窗口的旧 make_* 脚本若未过 G5/G6，下次复跑会被新门禁拦下**：
   这是预期行为（它们存在真实的历史缺陷），逐图补修即可，参考本轮六脚本的
   写法（门禁调用在 savefig 之后、gate_pdf 之前）。
2. **卸窗**：按协议用 `scripts/window_board.ps1` 对应脚本走，禁止直接
   `git worktree remove`（本窗口未执行任何 remove）。
3. **看板**：本窗口已登记五轮（提案/待裁决/C 终报/C+ 终报/D6-D8 处置）。
4. **投稿系统**：正典六图已在系统里排队的旧版本不受影响；新图作为 Approve 前
   替换或 revision 素材由用户决定。GA 的 531×131pt 是投稿系统栅格上限，
   v16 已严格合规。

## 七、环境事故与坑（下一个窗口必读）

1. **git 写 ref 不落地**（本轮两次实证）：`git commit`/`update-ref`/`branch`
   返回 0 但 ref 文件不出现，Bash 与 PowerShell 双通道同样失败；置 `gc.auto=0`
   无效。**可靠做法**：`read-tree → git add → write-tree → commit-tree -p <父>`，
   最后 PowerShell `[System.IO.File]::WriteAllText` 手写 loose ref
   （完整 40 位 hex + 换行；缩写 SHA 会报 branch broken；嵌套 ref 需先 mkdir）。
   详见用户级 skill `worktree-git-safe-commit`。
2. **PowerShell 输出通道偶发不可见**：命令 exit 0 但 stdout 空。**所有验证
   结果必须 `Out-File` 落盘再 Read 读回**，否则等于没验证（本轮一次提交
   翻车即因消息文件路径笔误 + 空值 ref，靠落盘读回才发现）。
3. **GA 不可加高画布换留白**：1.95in → MediaBox 531×140.4pt，超 131pt 上限
   9.4pt。只能走内部净空。
4. **matplotlib**：`FancyArrowPatch` 无 `get_posB()`，端点读私有 `_posA_posB`；
   `set_yticklabels(fontsize=)` 会被后续 `tick_params(labelsize=)` 覆盖——
   先 tick_params 再 set_ticklabels。
5. **Bash 前置**：每次先
   `export PATH="/c/Users/FOUR/.workbuddy/binaries/PortableGit/versions/1.2.0/usr/bin:.../mingw64/bin:$PATH"`
   （coreutils 缺失）。stderr 里的 `dirname: command not found` 是 shim 噪声，可忽略。

## 八、未完成 / 遗留

- **无未完成任务**。D1–D8 全部闭环。
- 可选后续（非本窗职责）：若未来某轮要显示 fig3 段内标签，需先推翻
  "label-agnostic" 的 caption 措辞——那是新的裁决项，不是工程项。
- 中途产生的孤立提交对象（unborn 期间的 root commit 等）已成为不可达对象，
  `gc.auto=0` 下不回收、不影响任何分支，无需清理。

## 九、本窗口提交链全貌（806f78d..HEAD）

```
ca086e8 feat(fig2-2col): 同步方向 C+（v19_2col）…          ← D7
3f761c7 feat(fig1-2col): 同步方向 C+（v16_2col）…          ← D7
3e655f7 docs(figB): 阶段 I（C+ 强化）终报告 …
a38b0b4 feat(ga): v16 C+ + 修 D5 箭头语义反转 …
784fb9c docs(figB): C+ 全局规格落档
22dbae1 feat(fig5): v9 C+ …
f9defba feat(fig4): v8 C+ …（守 3.66in）
ee3149d feat(fig3): v10 C+ …
f388796 feat(fig2): v19 C+ …
353571e feat(fig1): v16 C+ …
dfd5c01 docs(figB): 阶段 I 终报告（方向 C 五图）
f1beb8e feat(fig4): v7 尺度修复 + 高度合规
5f364b3 feat(fig3): v9 尺度修复
35e3e09 feat(fig2): v18 方向 C
8dc994d feat(fig1): v15 方向 C
d99e1c5 feat(fig5): v8 方向 C
67e95fa docs(figB): 事故口径更正（ref 不落地实证）
daeb4ec docs(figB): 阶段 S0 提案
861e43a chore(fig5): 三样板门禁留档 + 四联对比图
caff7e0 feat(fig5): 方向 C 样板 v7c
a8a0ad6 feat(fig5): 方向 B 样板 v7b
a69689e feat(fig5): 方向 A 样板 v7a
9235490 feat(fig-gates): 共享门禁增补 G5/G6                ← D6 本体
```
