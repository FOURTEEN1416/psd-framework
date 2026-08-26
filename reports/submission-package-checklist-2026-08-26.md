# PR 投稿打包清单（W41 · 2026-08-26）

> Owner: `reports/submission-package-checklist-2026-08-26.md` · 配套报告: `reports/submission-package-2026-08-26.md`（证据与草稿全文均在彼处）
> 用法: 投稿装配窗口按此逐格打勾；【USER】项未清零前不得点击提交。

## A. 稿件本体

| ✓ | 项 | 状态 | 责任 |
|---|----|------|------|
| ☑→🔄 | LaTeX 工程（**用户决策 A 已裁 cas-sc**，2026-08-26）：脚手架已建且编译全绿（main.pdf 3页/highlights.pdf 1页），六节正文待装配 | 🔄 脚手架完成 | 终稿装配窗（入口 `docs/paper/latex/README.md`） |
| ☐ | 五篇 md → tex：introduction / related-work / method / experiment-skeleton / conclusion-limitations | ⏳ 待装配 | 终稿装配窗 |
| ☐ | 单栏双倍行距、两端对齐、编页码（官方硬约束，禁双栏） | ⏳ 模板选项即含 | 终稿装配窗 |
| ☐ | 页数窗口校验 **20–35 页含图表参考文献附录**；<20 页会被建议转投 PR Letters | ⏳ 装配后实测 | 终稿装配窗 |
| ☐ | 标题终裁（现 15 词上沿 + R2 副句去留） | 【USER】 | 用户 |
| ☐ | 作者名单/单位/通讯作者信息 | 【USER】 | 用户 |
| ☐ | 关键词 5–6 个定稿 | ⏳ 建议稿在报告 F6 | 终稿窗 |
| ☐ | NTU 三流融合数落地后回写 §4.4/R4 及全文引用数字联动刷新 | ⏳ 不归 W41（协调者派单） | 协调者 |

## B. 强制随附文件（缺一系统不放行）

| ✓ | 项 | 状态 | 备注 |
|---|----|------|------|
| ☐ | **Highlights** 独立可编辑文件（3–5 条 ≤85 字符） | ✅ 草稿成（五条实测 65–77 字符，报告 §2.2）→ 存为 `highlights.tex` | 文件名须含 "highlights" |
| ☐ | **Data statement** | ✅ 草稿成（报告 §3.1）｜终审-A/B/C 三处许可标注待清 | 【USER】 |
| ☐ | **Declaration of competing interests**（系统问卷生成 Word 上传） | ⏳ 建议 "nothing to declare" | 【USER】复核 |
| ☐ | **CRediT author statement** | ⏳ 待作者名单 | 【USER】+终稿窗 |
| ☐ | **Generative AI use declaration** | 🔴 必填项，建议稿在报告 §3.3；披露颗粒度待拍板 | 【USER】终审-F |
| ☐ | Funding statement（无资助标准句已备） | ⏳ 一键粘贴 | 终稿窗 |
| ☐ | Ethics statement | ✅ 草稿成（报告 §3.2）｜终审-D InterPet4D 人被试原句留档待清 | 【USER】 |
| ☐ | Cover Letter | ✅ 草稿成（报告 §2.1）｜签名与利益声明句待用户 | 【USER】 |
| ☐ | Graphical abstract（可选） | ⚪ 默认不做 | 【USER】意愿 |

## C. 图表包

| ✓ | 项 | 状态 | 备注 |
|---|----|------|------|
| ☑ | fig1-4 矢量 PDF 终检通过（零栅格/字体内嵌） | ✅ 报告 §4.1 当次运行证据 | 直接上传 `docs/paper/figures/*.pdf` |
| ☑ | fig3/fig4 PNG 600dpi 达线图门槛（≥3543px） | ✅ 6470px / 4000px | 备用位图 |
| ☑ | 色盲安全 + 灰度打印双编码 | ✅ 青-橙对 + 线型/标记/纹理实证 | 新增数据图注意描边 ΔL 注记（§4.3） |
| ☐ | tab1/tab2/tab3 成表排版（最优加粗/方向符号/右对齐/无竖线无底纹） | 🟡 素材全齐（tab3 六行零 PENDING） | 终稿装配窗 |
| ☐ | caption 定稿（fig1/2 草稿在 figure-specs.md，fig3/4 在 FIGURE_SOURCE.md v0.2） | ⏳ 随装配收口 | 终稿装配窗 |
| ☐ | fig1/fig2 若需位图备份 → 重跑脚本 ≥500dpi 导出 | ⚪ 条件触发 | 同上 |

## D. 代码/数据发布包

| ✓ | 项 | 状态 | 备注 |
|---|----|------|------|
| ☐ | LICENSE 文件 | 🔴 不存在 | 【USER】终审-E（推荐 Apache-2.0 + 数据分层，报告 §6-B） |
| ☐ | 开源时机（投稿匿名仓 vs 录用后）+ 仓库 URL 回填 DA/Cover Letter 占位符 | 【USER】 | 与上一条同批裁决 |
| ☐ | README 快速上手节（环境→数据获取→三条主命令→测试） | ⏳ 结构建议在报告 §5.3 | 发布装配窗 |
| ☐ | 打包白名单: psd/ scripts/(py) configs/ requirements.txt README docs/DATA_LOCATIONS reports/*.json external→指针 | ⏳ 排除 data//runs//.venv/dev-docs/*.pt | 发布装配窗 |
| ☐ | 复现链命令节核对 | ✅ 十条链全部在档（报告 §5.2 表） | 已核，发布时复核最新 master |
| ☐ | external/AimCLR·SMQ 上游许可合规注记（不随包分发，clone 指针 + 权重初始化说明） | ⏳ | 发布装配窗 |

## E. 提交操作序列（Editorial Manager, single anonymized）

| ✓ | 步骤 |
|---|------|
| ☐ | 清零上文全部【USER】红色项 |
| ☐ | Scholar/arXiv 首次性终审执行并留档（L3 既定前置） |
| ☐ | 上传: 主稿 .tex 源文件包 + highlights + figures(pdf) + tables + cover letter + declarations |
| ☐ | 系统内生成 competing interest Word 并上传 |
| ☐ | 建议审稿人名单（可选，【USER】决定） |
| ☐ | 提交前最后跑一次 `pytest psd -q` 全绿留证（硬规则 4） |

## 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-26 | W41 初版：A-E 五区清单，6 个【USER】终审项编号与主报告 §6 汇总表一一对应 |
