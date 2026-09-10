# R30 论文图「海洋清风」编辑级重绘记录（2026-09-11）

## 触发与裁决
用户否定 R26–R29 的 NS 九色板+软阴影卡片观感（"不是顶刊风格"），两项硬指令：
1. 强制加载 `diagram-design` 技能（github.com/FOURTEEN1416/diagram-design，已克隆至 `C:\Users\FOUR\.zcode\skills\`）+ `figures4papers`（ChenLiu-1996）落地本地；
2. 全论文图改用「海洋清风」八色板（用户参考图标定色值）。

## 技能依据（本轮声明）
- diagram-design v2.6（编辑级设计系统）：阴影出局/边线入场、焦点强调 ≤2、密度 4/10、墨色文字；
- figures4papers + 内置 scientific-figure-making（房子风格）：top/right spine 关、无框图例、Arial、600dpi、语义色板纪律；
- 同轮部署图片清单全部 13 技能：12 个定位成功（diagram-design / figures4papers / academic-figure-skill(TingxiYu) / scipilot-figure-skill / pubfig / scientific-figure-making(内置于figures4papers) / paper-framework-figure-studio-pro / visio-image-rebuilder / paper-plot-skills / nature-figure-skill / AgentFigureGallery / K-Dense=scientific-agent-skills）；`scientific-figure-reference-generator` 全 GitHub 检索 total=0 不存在独立仓。

## 变更清单（数据口径零触碰）
- `figures/scripts/psd_style.py`：色板真源换海洋清风八色（#BFDDD2/#53999D/#4098AC/#7CC0CE/#DCC992/#ECB66B/#EC9E59/#EC8E5A）；语义映射 冷青=物理 / 深橙=语义+HERO / 青绿=human-in-loop / 灰=基线（蓝=物理红=语义的旧映射随板废止）；card() 阴影参数保留签名但不再生效（历史脚本零改动去阴影）；flowline 默认 2.2→1.6pt；NS_* 别名兜底映射。
- 新脚本：fig1 v11（接口带=浅绿桥）、fig2 v13（白卡+家族色描边、图例与实际编码对齐）、GA v11（NTU120 浅青条白字→墨字保对比）。
- 零代码换装重跑：fig3 v6 / fig4 v4 / fig5 v4（token 化架构自动生效）。
- caption 颜色词同步 2 处：03-method.tex fig2 "Red/orchid-labeled"→"Orange/teal-labeled"；04-experiments.tex fig3 "(red;/blue;"→"(orange;/teal;"。

## 验证链
1. 六图重生成成功（.venv python，600dpi png + 矢量 pdf）；
2. pdftotext 断言：旧颜色词（Red-labeled/orchid/(red;/(blue;/purple 等）零残留；新词落位 38/39 页；
3. pdflatex ×2：41 页 0 错误（与 R29 基线同页数）；
4. judge 视觉终审 7/7 pass（五张图页 + fig1 600dpi + GA）。minor 未阻塞项：fig1 顶部一处线交叉与肘部间隙（v9 冻结拓扑 legacy）；GA canine 灰斜纹为既定边界案例编码；judge 派发时 Figure4/5 页标注对调（页内 caption 与内容一致，论文无错位）。

## 遗留（不阻塞）
- fig1 深色满块 4 处 > 焦点 ≤2 指南（入口/出口锚点+演化带为本图设计语言，judge 认可）；
- 工具箱 `shared-scripts/figure_style_guide.md` 是否随海洋清风更新属跨项目文件（Ask first），未动；
- fig5 v2(深橙 s) 与 PanAf(橙 X) 色相接近，靠 marker+x 位冗余区分（judge 判定可辨识）。
