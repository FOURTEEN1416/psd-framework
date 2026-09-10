# R26 NS 九色板美化轮 — 留痕（2026-09-10）

## 背景与授权
R25 后用户看图仍判"配色很丑"，亲令：① 深度搜索数模工具箱绘图 skills；② **指定参考图
`D:\Desktop\参考图\微信图片_2026-09-10_204128_415.png` 的配色**（NS 单细胞九色板，图底部
自带色值标注）。方向=美化（工程域，用户已给明确方向+指定色板，无需再问）。

## 深度扫描结果（工具箱 247 技能）
绘图相关命中：figure-spec / paper-figure(+drawio/html) / nature-figure(-planner) /
scientific-visualization / scholar-critique-figures / comp-visual-review / diagram-design /
**shared-scripts/figure_style_guide.md（1045 行，本轮主要依据）** + 5 套 recipes 库
（basic/advanced/empirical/academic/competition）+ plot_utils.py 7 组配色方案
（soft/tableau/npg/nejm/science/colorblind...）。
style guide 关键纪律已吸收：数据色走 PALETTE token 禁硬编码；多系列=前 n 色+不同 marker；
浅填充(PALETTE_LIGHT)+深边(PALETTE)；同论文配色统一；禁红黄绿红绿灯对；禁默认蓝单色。

## 色板（参考图九色，写入 psd_style.py 为单一真源）
#E53A46 红 / #EE726D 珊瑚 / #FEDB65 黄 / #86BB4A 绿 / #75D3E3 天青 / #59ABDD 中蓝 /
#B67BB2 兰紫 / #F5B5B0 浅粉 / #FBCDB5 杏 + 蓝灰中性系 + 墨色 #22313F。
语义映射（结构承 R25，只换色值）：蓝=物理层、红=语义层/HERO、marker 七色一系一色、
灰=基线/参考。**fig1/fig2/fig4 零代码改动自动换装（token 架构收益实证）**；
fig3 v5（基线带独立蓝灰防过浅）、fig5 v4（七系列 NS 全色板+黄系深描边 S_SYNTH_EDGE）、
GA v4→v5（judge 修复轮）。

## caption 颜色词同步（三处，教训条款执行）
fig3 caption orange→red / cyan→blue；fig2 caption "orange boxes automated"→"red boxes"。
pdftotext 断言：新词各×1、旧词×0。

## judge 循环（1 轮 fail→修→复核）
首验 5/6 pass；**GA fail 两处（judge 像素证据）**：① 绿条白字 90.7% 对比 ~2.3:1 不达
3:1 大字下限→改墨色 #22313F（~5.8:1）；② 浅蓝填充上蓝标题弱于红标题孪生→新增
PHYS_TEXT #2E77AE 加深。复核 pass。**6/6 全绿。**

## 验证链
六图重渲 mtime 21:32-21:46；fig4/fig5 数据口径逐位与 R24 一致（负结果事实保留=True）；
作者自验六图；双 judge+复核 judge；编译 41 页 0 错误；pdftotext caption 断言。

## 红线（未动）
布局/坐标/数据/预注册判据措辞全冻结；E6-real/E6-real-2 零触碰；GA 尺寸 531×131pt 合规。
