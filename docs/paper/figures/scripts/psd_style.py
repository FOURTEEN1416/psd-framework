# -*- coding: utf-8 -*-
r"""psd_style.py — PSD 论文图统一样式真源（R25 审美/配色/结构升级, 2026-09-10）。

依据数模工具箱两个技能的规范合成：
- nature-figure: PALETTE_NATURE 语义（深蓝=hero 方法 / 灰阶=参考基线 / 克制单族），
  禁 matplotlib 默认色、legend 无框、去 chart junk；
- scientific-visualization: 色盲安全（teal/orange 替代红绿对）、灰度可辨、
  Arial 系字体、轴 spines 精简。

跨图色彩语义铁律（同色=同意图，全六图一致）：
  PHYS   teal #0E7490      = 物理层 / Φ 动态流 / 无监督提案
  SEM    orange #C2410C    = 语义层 / 锚点 / 伪标签
  HERO   blue #0F4D92      = 我方策略臂（proposed method）
  GRAY1..4 单调明度灰阶    = human benchmark / 参考 / 基线 / 近黑锚
  VIOLET #9A4D8E            = synthetic 对照系列专用（唯一彩色特例）
字体 Arial（回退 DejaVu）；spines 0.8；网格 #DADADA。
"""
import matplotlib.pyplot as plt

# --- 语义 token（单一真源；各图脚本从这里取色，禁止散落字面量） ---
PHYS_EDGE = "#0E7490"
PHYS_FILL = "#DAFFFF"
SEM_EDGE  = "#C2410C"
SEM_FILL  = "#FFE3DA"
FOCAL_FILL = "#FFD9C7"          # 焦点强调 tint（fig2 hub）
HERO      = "#0F4D92"           # proposed-strategy 曲线（fig4 entropy 臂）

GRAY_1 = "#A6A6A6"               # 浅 —— UCF 独立基准
GRAY_2 = "#767676"               # 中 —— NTU60 / random 基线
GRAY_3 = "#4D4D4D"               # 深 —— NTU120
GRAY_4 = "#272727"               # 近黑 —— PanAf
NEUTRAL_LT = "#CFCECE"
GRAY_EDGE  = "#6B7280"
GRAY_FILL  = "#F3F4F6"
GRAY_LINE  = "#9CA3AF"           # 流程图软箭头/虚线
IFACE_FILL, IFACE_EDGE = "#EFEFEF", "#9CA3AF"

VIOLET     = "#9A4D8E"           # synthetic-offset
INK        = "#111111"
ARROW      = "#374151"
NOTE       = "#4B5563"
GRID_GRAY  = "#DADADA"

def apply_style():
    """所有 PSD 图脚本顶端统一调用（替代各图散落的 rcParams.update）。"""
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "text.color": INK,
        "axes.edgecolor": GRAY_3,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    })
