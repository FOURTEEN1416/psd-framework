# -*- coding: utf-8 -*-
r"""psd_style.py — PSD 论文图统一样式真源（R30 起=海洋清风八色板, 2026-09-11）。

R30 配色交割：用户指定「海洋清风」科研配色（微信截图 8 色值）
为全论文唯一配色来源，取代 R26 NS 九色板。色板：
  #BFDDD2 浅绿 mint / #53999D 青绿 teal / #4098AC 深青 cyan / #7CC0CE 浅青 sky
  #DCC992 卡其 khaki  / #ECB66B 金橙 gold / #EC9E59 橙 orange / #EC8E5A 深橙 ember

R30 设计语言（diagram-design 编辑级规范 + figures4papers 房子风格联合裁决）：
  - 阴影全面废除（"Shadows are out. Borders are in."）——card() 的 dx/dy/shadow
    参数保留签名但不再生效，历史脚本零改动即去阴影；
  - 结构色纪律不变：文字=墨黑 INK，箭头=深灰黑，中性阶=灰（禁自创彩色结构色）；
  - 深化文字色=同色相明度压缩（HSL 降 L 保 H），白字对比 ≥5:1（R29 先例延续）；
  - 焦点强调（HERO/DEEP 满块）每图 ≤2 处（diagram-design focal rule）；
  - 数据图=figures4papers 房子风格：top/right spine 关闭、无框图例、dpi600。

跨图色彩语义铁律（同色=同意图，全六图一致；冷=物理/自动，暖=语义/焦点）：
  PHYS  深青 #4098AC 系   = 物理层 / Φ 动态流 / 无监督提案
  SEM   深橙 #EC8E5A 系   = 语义层 / 锚点 / 伪标签 / proposed 策略臂(HERO)
  HUMAN 青绿 #53999D 系   = human-in-the-loop / 人工环节
  IFACE 浅绿 #BFDDD2 系   = 层间接口桥
  GRAY1..4 单调明度灰阶   = human benchmark 参考 / 基线 / 近黑锚
字体 Arial（回退 DejaVu）；spines 0.8。
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# --- R30 海洋清风八色板（用户参考图真源，勿改值） ---
OCEAN_MINT   = "#BFDDD2"   # 浅绿
OCEAN_TEAL   = "#53999D"   # 青绿
OCEAN_CYAN   = "#4098AC"   # 深青
OCEAN_SKY    = "#7CC0CE"   # 浅青
OCEAN_KHAKI  = "#DCC992"   # 卡其
OCEAN_GOLD   = "#ECB66B"   # 金橙
OCEAN_ORANGE = "#EC9E59"   # 橙
OCEAN_EMBER  = "#EC8E5A"   # 深橙

# --- 旧 NS 板别名（deprecated, R30 起映射到海洋清风最近色，仅为未迁移引用兜底） ---
NS_RED     = OCEAN_EMBER
NS_CORAL   = OCEAN_ORANGE
NS_YELLOW  = OCEAN_GOLD
NS_GREEN   = OCEAN_TEAL
NS_SKY     = OCEAN_SKY
NS_BLUE    = OCEAN_CYAN
NS_ORCHID  = OCEAN_TEAL
NS_PINK    = OCEAN_MINT
NS_APRICOT = OCEAN_KHAKI

def _lighten(hex_color, f):
    """向白混合 f 比例（印刷浅填充派生：色相不出八色板）。"""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        round(r + (255 - r) * f), round(g + (255 - g) * f), round(b + (255 - b) * f))

# --- R30 语义 token（单一真源；各图脚本从这里取色，禁止散落字面量） ---
PHYS_EDGE = OCEAN_CYAN                   # 物理层边线/主色
PHYS_TEXT = PHYS_DEEP_COLOR = "#2D7686"  # 物理层深化字/线色（#4098AC 同相降L, 白底对比 5.9:1）
SEM_EDGE  = OCEAN_EMBER                  # 语义层边线/主色
SEM_TEXT = SEM_DEEP_COLOR = "#AF501D"    # 语义层深化字/线色（#EC8E5A 同相降L, 白字对比 5.3:1）
FOCAL_FILL = _lighten(OCEAN_EMBER, 0.55) # 焦点强调 tint
HERO      = OCEAN_EMBER                  # proposed-strategy 曲线（fig4 entropy 臂）
HUMAN_EDGE  = OCEAN_TEAL                 # 人工环节=青绿
HUMAN_TEXT = HUMAN_DEEP_COLOR = "#38686B"  # 青绿深化（#53999D 同相降L, 白字对比 6.2:1）
IFACE_FILL, IFACE_EDGE = OCEAN_MINT, OCEAN_TEAL   # 接口带=浅绿桥（R30）

GRAY_1 = "#A6A6A6"                        # 浅 —— 图内次级注记/线
GRAY_2 = "#767676"                        # 中 —— 结构虚线/参照
GRAY_3 = "#4D4D4D"                        # 深 —— 轴 spines
GRAY_4 = "#272727"                        # 近黑 —— hub/强调
NEUTRAL_LT = "#CFCECE"
GRAY_EDGE  = "#9E9E9E"                    # 中性灰边（null 参照专用）
GRAY_FILL  = "#E3E3E3"                    # 中性灰填充（null 参照专用）
GRAY_LINE  = "#666666"                    # 流程图软箭头/虚线

# R29 深化/柔色三件套 token 名保留（值换海洋清风深化系），v10/v12/GA v10 脚本零改动换装
PHYS_DEEP  = PHYS_DEEP_COLOR
SEM_DEEP   = SEM_DEEP_COLOR
HUMAN_DEEP = HUMAN_DEEP_COLOR
PHYS_TINT  = _lighten(OCEAN_SKY, 0.72)   # 容器柔色填充（浅青派生 #DEEFF3）
SEM_TINT   = _lighten(OCEAN_EMBER, 0.78) # （深橙派生 #FAE3D6）
HUMAN_TINT = OCEAN_MINT                  # 青绿系浅填充=板内浅绿原色
BODY_TINT  = "#F4FAFB"                   # 容器身体（近白微青）
# fig3 时间轴轨道填充（比容器 tint 深半档，配彩色描边）
PHYS_FILL  = _lighten(OCEAN_SKY, 0.62)   # #D6ECF0
SEM_FILL   = _lighten(OCEAN_EMBER, 0.70) # #F6D3C2

# fig5 多系列映射（一系一色 + marker 形状冗余编码；八色板内取色）
S_PUBV1   = OCEAN_CYAN                   # public-real v1 (o)
S_PUBV2   = OCEAN_EMBER                  # public-real v2 (s) —— proposed 臂
S_NTU60   = OCEAN_TEAL                   # human benchmark NTU60 (D)
S_NTU120  = OCEAN_SKY                    # human benchmark NTU120 (v)
S_UCF     = OCEAN_GOLD                   # independent benchmark UCF101 (+)
S_PANAF   = OCEAN_ORANGE                 # animal public benchmark PanAf500 (X)
S_SYNTH   = OCEAN_MINT                   # synthetic-offset (^) —— 白底浅，配深描边
S_SYNTH_EDGE = OCEAN_TEAL                # 浅绿系列专用描边（保可见性）

INK   = "#1A1A1A"                        # 主文字=黑
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
        "axes.spines.top": False,        # figures4papers 房子风格（R30）
        "axes.spines.right": False,
        "legend.frameon": False,
    })

def card(ax, x, y, w, h, fill="#FFFFFF", edge=None, lw=1.0, rounding=1.2,
         dx=0.5, dy=-0.8, shadow=False, z=3, hatch=None):
    """R30 扁平卡片：无阴影（diagram-design 铁律），edge 给定即描边。
    dx/dy/shadow 参数保留仅为历史脚本签名兼容，R30 起不再产生任何效果。"""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={rounding}",
                       facecolor=fill, edgecolor="none" if edge is None else edge,
                       linewidth=lw if edge is not None else 0, zorder=z,
                       hatch=hatch)
    ax.add_patch(p)
    return p

def flowline(ax, x1, y1, x2, y2, color="#333333", lw=1.6, ls="-",
             head=11, z=2, connectionstyle=None):
    """R30 编辑级流线：1.6pt 细线 + 适中箭头（顶刊方法图线宽惯例，替代 R29 粗流线）。"""
    from matplotlib.patches import FancyArrowPatch
    kw = dict(arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
              linestyle=ls, zorder=z)
    if connectionstyle:
        kw["connectionstyle"] = connectionstyle
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), **kw))

def module(ax, x, y, w, h, tint, title, sub=None, deep=None, z=4,
           title_fs=6.2, sub_fs=5.3, rounding=1.2):
    """R30 模块：无边柔色圆角块 + 墨黑粗标题(+灰副文)（diagram-design: node name=ink）。"""
    card(ax, x, y, w, h, fill=tint, edge=None, rounding=rounding, z=z)
    tc = INK if deep is None else deep
    if sub:
        ax.text(x + w/2, y + h*0.62, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=tc, zorder=z+2)
        ax.text(x + w/2, y + h*0.26, sub, ha="center", va="center",
                fontsize=sub_fs, color="#444444", zorder=z+2)
    else:
        ax.text(x + w/2, y + h/2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=tc, zorder=z+2)
