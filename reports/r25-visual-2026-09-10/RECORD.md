# R25 视觉审美升级轮 — 留痕（2026-09-10）

## 背景与授权
R24 精修轮冻结（ac1d916）后，用户查看六图并亲自定方向："需要修审美配色结构"，
指定从数模竞赛工具箱加载技能。本域=工程域（图表风格），项目主权；行为域走
轻量商讨协议（复述对齐+完成后 diff 举证）。

## 加载技能（工具箱 D:\Desktop\数模竞赛\科研工具箱\skills）
- **nature-figure**：PALETTE_NATURE 语义（蓝=hero 方法/灰=基线参考）、禁 matplotlib
  默认色、legend 无框、去 chart junk、Arial 系字族；
- **scientific-visualization**：色盲安全（teal/orange 替代红绿对）、灰度可辨、
  最终尺寸字号检查、spines 精简；
- **comp-visual-review + judge 子代理**：沿用 R24 复验门（双 judge 并行）。
- 弃用 figure-spec（JSON→SVG 渲染器）：现有 matplotlib 矢量脚本已版本化，
  迁移风险大于收益。

## 变更（布局/坐标/数据口径零改动，只动外观）
1. **新增 `docs/paper/figures/scripts/psd_style.py` 单一真源色板**：
   PHYS teal #0E7490=物理层 / SEM orange #C2410C=语义层 / HERO blue #0F4D92=
   proposed 臂 / GRAY_1..4 单调明度灰阶=基准与基线 / VIOLET #9A4D8E=synthetic 特例；
   `apply_style()` 统一 Arial（回退 DejaVu）+ spines 0.8 + legend frameon=False。
2. **六图脚本版本递进**（v7/v8/v4/v3/v3/v3，补丁=patch_r25.py 全 assert 计数）：
   - fig1/fig2/GA：取色改 token（值多数不变），INK 统一 #111111，字体 Arial；
   - fig3：cyan/orange 语义保留（caption 颜色词不动），BAND_GRAY 归 token；
   - **fig4 语义重组**：entropy(我方策略)=HERO 蓝实线实心圆，random(基线)=GRAY_2
     灰虚线空心方块（原 teal/orange 与层语义撞色）；
   - **fig5 七系列灰阶单调化**：UCF #A6A6A6 < NTU60 #767676 < NTU120 #4D4D4D <
     PanAf #272727（原 #9CA3AF/#6B7280/#4B5563/#1F2937 明度乱序）；紫 #6D28D9→#9A4D8E
     归队；teal/orange 保留（public-real v1/v2 层语义，caption 无颜色词，安全）。
3. **caption-图失配修复两处**（看图时新发现的 R24 漏网）：
   - 03-method.tex fig2 首句 "two-ring flywheel sharing the Assign node" →
     "two-row pipeline around a shared state hub"（图 R24 已重设计为双行管线）；
   - 05-ablation-analysis.tex fig4 caption 删 "arms dodge ±4.5% at shared budgets"
     （图 R24r3 已弃 dodge，DODGE=1.0）。
   - 06-conclusion "future flywheel claim" 比喻句=正文独立隐喻，有意保留。

## 验证链（先证明再宣称）
- 渲染：六脚本全执行，12 工件 mtime 统一 20:31；fig4/fig5 打印数值逐位与 R24
  一致（69.9/77.8/80.9/88.0；28.9/35.0/44.2/90.7/88.9/66.6/88.7/85.7）。
- 作者自验六图缩略（R24 教训：作者先自验再派 judge）。
- 双 judge 并行复验：**6/6 pass**（流程图 judge 附逐处像素证据；数据图 judge 零 issue）。
- 编译：pdflatex 两连，41 页 0 错误；pdftotext 验证 "two-row pipeline"×1、
  "two-ring flywheel"×0、"dodge"×0。
- 提交 8841c19（25 文件精确 staging，无 add .），tag review-snapshot 重锚 HEAD。
- 附带簿记：MEXP 战役三件运行时日志（bootstrap/console/DONE.flag）入 reports/。

## 教训
- **caption 颜色词与图形态是失配高发区**：R24 八轮 judge 只审"图内自洽"，
  没人对 caption 首句比喻与图拓扑——作者通读图+caption 配对才抓出 flywheel/dodge
  两处。图重设计后必须 grep caption 方位词/颜色词/机制词全文同步。
- 色板统一成本最低的落地方式=共享模块+字面量替换（psd_style.py），而非重写脚本；
  下次全谱系调色只动一个文件。
- bash grep 对 pdftotext 的 ISO-8859+CRLF 输出静默失灵（rc=1 无计数行），
  文本断言验证一律用 python 读字节解码。

## 未动清单（红线）
E6-real/E6-real-2 冻结判据段措辞；fig 版式几何与数据口径；_patch2.py（上轮遗留
untracked，非本轮产物，未删未提交）；docs/paper/*.md 真源漂移治理债（仍待用户裁决）。
