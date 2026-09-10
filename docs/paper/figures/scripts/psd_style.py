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
from matplotlib.patches import FancyBboxPatch

# --- R27 现代扁平设计语言（用户两轮判丑后确认：问题在骨架不在色值） ---
SHADOW = "#1F2937"
SHADOW_ALPHA = 0.13

def card(ax, x, y, w, h, fill="#FFFFFF", edge=None, lw=1.1, rounding=1.6,
         dx=0.5, dy=-0.8, shadow=True, z=3, hatch=None):
    """软阴影+大圆角扁平卡片（数据坐标）。edge=None 即无边纯色块。"""
    if shadow:
        ax.add_patch(FancyBboxPatch((x + dx, y + dy), w, h,
                                     boxstyle=f"round,pad=0,rounding_size={rounding}",
                                     facecolor=SHADOW, edgecolor="none",
                                     alpha=SHADOW_ALPHA, zorder=z - 1))
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       facecolor=fill, edgecolor="none" if edge is None else edge,
                       linewidth=lw if edge is not None else 0, zorder=z,
                       hatch=hatch)
    ax.add_patch(p)
    return p

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
# R28 用户令：一切可见颜色直接取 NS 九色板，结构色（文字/箭头/轴）用黑灰，禁自创蓝灰系。
PHYS_EDGE = NS_BLUE                      # 物理层边线/主色
PHYS_TEXT = "#2E77AE"                    # 浅蓝填充上的标题深蓝字（R26 judge: 白底蓝弱）
PHYS_FILL = "#E4F1FB"                    # 物理层浅填充（NS_BLUE 提亮）
SEM_EDGE  = NS_RED                       # 语义层边线/主色
SEM_FILL  = "#FCE9E7"                    # 语义层浅填充（NS_RED 提亮）
FOCAL_FILL = "#F9D2CE"                   # 焦点强调 tint（fig2 hub 侧盒）
HERO      = NS_RED                       # proposed-strategy 曲线（fig4 entropy 臂）
HUMAN_EDGE  = NS_ORCHID                  # 人工环节=兰紫（R28: 替代灰盒）
HUMAN_FILL  = "#EDE0EC"                  # 兰紫提亮
IFACE_FILL, IFACE_EDGE = NS_PINK, "#D89B93"   # 接口带=粉（R28: 替代灰带）

GRAY_1 = "#A6A6A6"                        # 浅 —— 图内次级注记/线
GRAY_2 = "#767676"                        # 中 —— 结构虚线/参照
GRAY_3 = "#4D4D4D"                        # 深 —— 轴 spines
GRAY_4 = "#272727"                        # 近黑 —— hub/强调
NEUTRAL_LT = "#CFCECE"
GRAY_EDGE  = "#9E9E9E"                    # 中性灰边（null 参照专用）
GRAY_FILL  = "#E3E3E3"                    # 中性灰填充（null 参照专用）
GRAY_LINE  = "#666666"                    # 流程图软箭头/虚线（R28: 深灰非蓝灰）

# fig5 多系列映射（一系一色 + marker 形状冗余编码）
S_PUBV1   = NS_BLUE                      # public-real v1 (o)
S_PUBV2   = NS_RED                       # public-real v2 (s)
S_NTU60   = NS_GREEN                     # human benchmark NTU60 (D)
S_NTU120  = NS_ORCHID                    # human benchmark NTU120 (v)
S_UCF     = NS_SKY                       # independent benchmark UCF101 (+)
S_PANAF   = NS_CORAL                     # animal public benchmark PanAf500 (X)
S_SYNTH   = NS_YELLOW                    # synthetic-offset (^) —— 白底浅，配深描边
S_SYNTH_EDGE = "#C9A227"                 # 黄系列专用描边（保可见性）

INK   = "#1A1A1A"                        # 主文字=黑（R28: 参考图文字为黑，非蓝黑）
ARROW = "#333333"                        # 流程箭头=深灰黑
NOTE  = "#555555"                        # 次要标注
GRID_GRAY = "#E8E8E8"                    # 网格（中性浅灰）

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

# --- R29 顶刊方法图设计语言（Nature/Cell 柔色满块+层级文字+粗流线） ---
# 模块=柔色圆角块（无边）+深系文字；容器=极浅身体+满色头带（白字标题）；
# 数据流=粗圆头流线（lw>=2.2）。色相全部来自 NS 板（明度适配印刷对比度）。
PHYS_DEEP  = "#1F6FA8"                    # NS_BLUE 深化：头带/文字/流线
SEM_DEEP   = "#B02128"                    # NS_RED 深化
HUMAN_DEEP = "#7E4678"                    # NS_ORCHID 深化
PHYS_TINT  = "#C9E3F7"                    # 模块柔色填充（NS_BLUE 提亮一档）
SEM_TINT   = "#F8C9C6"
HUMAN_TINT = "#E4CFE2"
BODY_TINT  = "#F4F8FC"                    # 容器身体（近白微蓝）

def flowline(ax, x1, y1, x2, y2, color="#333333", lw=2.2, ls="-",
             head=13, z=2, connectionstyle=None):
    """顶刊式粗流线箭头（大箭头头；FancyArrowPatch 不收 capstyle，端形由 arrowstyle 决定）。"""
    from matplotlib.patches import FancyArrowPatch
    kw = dict(arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
              linestyle=ls, zorder=z)
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))

def module(ax, x, y, w, h, tint, title, sub=None, deep="#333333", z=4,
           title_fs=6.2, sub_fs=5.3, rounding=1.6):
    """顶刊式模块：无边柔色圆角块 + 深系标题(+灰副文)。返回文字基线信息。"""
    card(ax, x, y, w, h, fill=tint, edge=None, rounding=rounding,
         dx=0.4, dy=-0.7, z=z)
    if sub:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=deep, zorder=z+2)
        ax.text(x + w/2, y + h*0.26, sub, ha="center", va="center",
                fontsize=sub_fs, color="#444444", zorder=z+2)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=deep, zorder=z+2)
