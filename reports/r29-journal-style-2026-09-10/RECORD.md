# R29 顶刊方法图语言重做轮 — 留痕（2026-09-10）

## 背景与授权
R27/R28 后用户第三次判丑，并给出方向性裁决："**学学顶刊的科研绘图**，我们这个
像小学生"。诊断升级：R27 只换了卡片皮肤，没换**设计语言**——顶刊方法图的质感
来自柔色满块+文字层级+粗流线+留白节奏，而非"描边盒+细箭头"。

## 设计语言（R29，写入 psd_style.py）
- `PHYS_DEEP #1F6FA8 / SEM_DEEP #B02128 / HUMAN_DEEP #7E4678`：NS 板色相的
  印刷明度深化版（白字对比达标；色相不变=不违背"直接用九色板"裁决）；
- `PHYS_TINT / SEM_TINT / HUMAN_TINT / BODY_TINT`：柔色填充梯度；
- `flowline()`：粗流线箭头（lw 2.0-2.4，箭头头 11-14）；
- `module()`：柔色圆角块+DEEP 粗体标题+灰副文两级文字。
- 容器=柔色满块（PHYS_TINT/SEM_TINT，无边大圆角+阴影），标题 DEEP 内嵌容器顶部；
- 模块=白色扁平卡+DEEP 粗体标题+灰副文；**入口/出口锚点模块=DEEP 满色白字**
  （Unlabeled streams / Rule-engine seeds / Classification / Label pool 焦点块）；
- fig1 版式承 v9 坐标，fig2 承 v11，GA 承 v9（数据条不动）。

## 版本与修复
fig1 v10（自验抓到输出流线穿容器标题→降线 y65/71→63/68.3）；
fig2 v12（automated=白卡深红字、焦点 Label pool=深红满块白字、human=白卡兰紫字、
hub 黑保留、flowline lw2.0）；GA v10（双柔色块+DEEP 标题+粗接口流线+演化弧
zorder=6 承 R28 修复）。

## caption 同步（教训条款）
fig2 caption "Orchid boxes are human-in-the-loop steps, red boxes automated steps."
→ "Red-labeled steps are automated; orchid-labeled steps are human-in-the-loop."
（图已是"红字白卡"非"红盒"）。pdftotext 断言新句 ×1 旧句 ×0。

## 验证链
三图重渲 mtime 一致；作者自验（抓到并修复流线穿标题）；judge 严格顶刊标准
三张全 pass（"does this now read as a professional journal method figure"）；
编译 41 页 0 错误。数据图三张（fig3 v6/fig4 v4/fig5 v4）R28 已 pass 未动——
顶刊数据图形态本就如此，判丑集中于流程图。

## 红线
坐标拓扑承前版、数据口径零触碰、预注册措辞未动、GA 尺寸 531×131pt 合规。

## 教训
1. "换色板救不了丑"（R27 结论）之后还有一层：**"换卡片皮肤也救不了"——
   要换的是设计语言**（色块 vs 描边、文字层级、线宽等级、留白）。
2. DEEP/LIGHT 双色阶是"用户指定明亮色板"与"印刷对比度"的兼容解：
   色相保持用户裁决，明度按 WCAG 对比适配。
3. 自验必须在缩略图+整图两级都做：本轮流线穿标题在整图自验才看到。
