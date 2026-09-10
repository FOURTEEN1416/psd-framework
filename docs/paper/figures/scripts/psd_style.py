# -*- coding: utf-8 -*-
r"""psd_style.py — PSD 论文图统一样式真源（R26 起=NS 九色板, 2026-09-10）。

R26 配色交割：用户指定参考图（微信图片_2026-09-10_204128_415.png，NS 单细胞九色板）
为全论文唯一配色来源。色板：
  #e53a46 红 / #ee726d 珊瑚 / #fedb65 黄 / #86bb4a 绿 / #75d3e3 天青 /
  #59abdd 中蓝 / #b67bb2 兰紫 / #f5b5b0 浅粉 / #fbcdb5 杏
配套工具箱 figure_style_guide 纪律：数据色走 token 禁硬编码；多系列=前 n 色+
不同 marker；浅填充(PALETTE_LIGHT)+深边(PALETTE)；全论文统一。

跨图色彩语义铁律（同色=同意图，全六图一致）：
  PHYS   蓝 #59abdd       = 物理层 / Φ 动态流 / 无监督提案
  SEM    红 #e53a46       = 语义层 / 锚点 / 伪标签 / proposed 策略臂(HERO)
  GRAY1..4 单调明度灰阶    = human benchmark / 参考 / 基线 / 近黑锚
  NS 其余五色              = 多系列散点专用（fig5 一系一色，marker 冗余编码）
字体 Arial（回退 DejaVu）；spines 0.8。
"""
import matplotlib.pyplot as plt

# --- NS 九色板（参考图真源，勿改值） ---
NS_RED     = "#E53A46"
NS_CORAL   = "#EE726D"
NS_YELLOW  = "#FEDB65"
NS_GREEN   = "#86BB4A"
NS_SKY     = "#75D3E3"
NS_BLUE    = "#59ABDD"
NS_ORCHID  = "#B67BB2"
NS_PINK    = "#F5B5B0"
NS_APRICOT = "#FBCDB5"

# --- 语义 token（单一真源；各图脚本从这里取色，禁止散落字面量） ---
PHYS_EDGE = NS_BLUE                      # 物理层边线/主色
PHYS_TEXT = "#2E77AE"                    # 浅蓝填充上的标题深蓝字（R26 judge: 白底蓝弱）
PHYS_FILL = "#E4F1FB"                    # 物理层浅填充（NS_BLUE 提亮）
SEM_EDGE  = NS_RED                       # 语义层边线/主色
SEM_FILL  = "#FCE9E7"                    # 语义层浅填充（NS_RED 提亮）
FOCAL_FILL = "#F9D2CE"                   # 焦点强调 tint（fig2 hub 侧盒）
HERO      = NS_RED                       # proposed-strategy 曲线（fig4 entropy 臂）

GRAY_1 = "#A6A6A6"                        # 浅 —— 图内次级注记/线
GRAY_2 = "#767676"                        # 中 —— random 基线
GRAY_3 = "#4D4D4D"                        # 深 —— 轴 spines/深色文字
GRAY_4 = "#272727"                        # 近黑 —— hub/强调文字
NEUTRAL_LT = "#CFCECE"
GRAY_EDGE  = "#8A94A6"                    # 流程图灰盒边（蓝灰调，融入 NS 板）
GRAY_FILL  = "#F2F4F7"                    # 流程图灰盒填充
GRAY_LINE  = "#9AA5B1"                    # 流程图软箭头/虚线（蓝灰）
IFACE_FILL, IFACE_EDGE = "#EEF1F5", "#9AA5B1"

# fig5 多系列映射（一系一色 + marker 形状冗余编码）
S_PUBV1   = NS_BLUE                      # public-real v1 (o)
S_PUBV2   = NS_RED                       # public-real v2 (s)
S_NTU60   = NS_GREEN                     # human benchmark NTU60 (D)
S_NTU120  = NS_ORCHID                    # human benchmark NTU120 (v)
S_UCF     = NS_SKY                       # independent benchmark UCF101 (+)
S_PANAF   = NS_CORAL                     # animal public benchmark PanAf500 (X)
S_SYNTH   = NS_YELLOW                    # synthetic-offset (^) —— 白底浅，配深描边
S_SYNTH_EDGE = "#C9A227"                 # 黄系列专用描边（保可见性）

INK   = "#22313F"                        # 主文字（深蓝黑，与 NS 板同调）
ARROW = "#5D6D7E"                        # 流程箭头
NOTE  = "#5D6D7E"                        # 次要标注
GRID_GRAY = "#E5E8EC"                    # 网格（浅蓝灰）

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
