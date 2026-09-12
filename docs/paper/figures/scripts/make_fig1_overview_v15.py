# -*- coding: utf-8 -*-
r"""fig1 v15 — PSD 框架总览（正典 · 风格方向 C「分阶色标 + 制图精度」+ 尺度修复）。

承 v13 零交叉拓扑（顶出→左高右低分高平移、演化带回线纯垂直、焦点深块 ≤2），
本版叠加方向 C 的制图层，**图内全部文字逐字承 v13，零新增零删除零改写**：

1. 尺度修复（wt/figB 诊断头号根因）：去 `bbox_inches="tight"`，固定画布
   3.42x2.60in，MediaBox 严格 246.24x187.20pt=版心宽；v13 的 tight 墨迹框
   245.22x186.77pt 被 LaTeX 放大 1.004 倍，本版装入系数 1.000，字号所见即所得，
   版面占用 187.20pt 与 v13 的 187.46pt 逐点持平（不引发浮动回退）。
2. 分阶色标（tone ladder）：面板=最浅柔色底+0.6pt 同相发丝边框 →
   卡片=深一阶同相浅底(mix 0.58)+0.55pt 更深同相边框(mix 0.30) →
   焦点=DEEP 满块白字。层级来自明度阶梯与描边，不再靠白卡贴纸感。
3. 字重阶梯：面板标题 bold 6.8 → 卡片标题 regular 5.9-6.3（DEEP 色）→
   焦点块 bold 白字。v13 通篇 bold 的"同重"问题消除。
4. 笔触统一：全部主流线 1.6pt/head11（含接口主箭，v13 为 1.8）；
   虚线仅保留给 Taxonomy 约束回路（1.3pt/head9，次级语义）。
5. 净空规整：左右两栏卡片统一 x=4/25（w=19）与 x=66（w=40），
   上下左右边距全部 2 单位；两条顶部长横箭终点 52.5 精确触桥缘。
6. 字号下限 5.5pt：v13 桥标签 5.2 -> 5.5。

数据/文本零改动清单（对照 v13 逐字）：面板标题×2、卡片标题×9、桥标签
"embeddings"/"+ proposals"、底部条 "Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:
only $\Omega$ retrains"。
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]  # worktree 隔离：禁硬编码主仓绝对路径
OUT = ROOT / "docs" / "paper" / "figures"
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PE, SE_ = psd_style.PHYS_EDGE, psd_style.SEM_EDGE


def _mix(hex_color, f):
    """向白混合 f 比例（本脚本局部助手，不写入 psd_style）：同相明度阶梯派生。"""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        round(r + (255 - r) * f), round(g + (255 - g) * f), round(b + (255 - b) * f))


# ---- 分阶色标 token（同相明度阶梯：面板浅 -> 卡片中 -> 焦点深）----
P_PANEL      = psd_style.PHYS_TINT            # 物理面板底（最浅）
P_PANEL_EDGE = _mix(PE, 0.50)
P_CARD       = _mix(PE, 0.58)                 # 物理卡片底（深一阶）
P_CARD_EDGE  = _mix(PE, 0.30)
S_PANEL      = psd_style.SEM_TINT
S_PANEL_EDGE = _mix(SE_, 0.50)
S_CARD       = _mix(SE_, 0.58)
S_CARD_EDGE  = _mix(SE_, 0.30)

# ---- 固定画布：宽 3.42in == 版心 246.24pt ----
FIG_W, FIG_H = 3.42, 2.60
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 110); ax.set_ylim(0, 84)
ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)


def wcard(x, y, w, h, title, fill, edge, deep, title_fs=6.3, z=5):
    """方向 C 模块卡：同相浅底 + 0.55pt 更深同相描边 + DEEP 色 regular 标题。"""
    psd_style.card(ax, x, y, w, h, fill=fill, edge=edge, lw=0.55,
                   rounding=1.2, z=z)
    ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=title_fs,
            fontweight="normal", color=deep, zorder=z + 2, linespacing=1.15)


def deepblock(x, y, w, h, title, deep, z=5, title_fs=6.6):
    """DEEP 满色白字模块（焦点，每图 ≤2 处）。"""
    psd_style.card(ax, x, y, w, h, fill=deep, edge=None, rounding=1.2, z=z)
    ax.text(x + w/2, y + h/2, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color="white", zorder=z + 2, linespacing=1.15)


# ---- 层容器（最浅柔色底 + 0.6pt 同相发丝边框，DEEP 标题内嵌顶部） ----
psd_style.card(ax, 2, 14, 44, 62, fill=P_PANEL, edge=P_PANEL_EDGE, lw=0.6,
               rounding=2.0, z=2)
ax.text(24, 73.6, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=6.8, fontweight="bold", color=PD, zorder=6)
psd_style.card(ax, 64, 14, 44, 62, fill=S_PANEL, edge=S_PANEL_EDGE, lw=0.6,
               rounding=2.0, z=2)
ax.text(86, 73.6, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
        fontsize=6.8, fontweight="bold", color=SD_, zorder=6)

# ---- 中央接口桥（浅绿 IFACE，语义锁）+ 唯一接口主箭 + 横排边标签 ----
psd_style.card(ax, 52.5, 14, 7, 62, fill=psd_style.IFACE_FILL, edge=None,
               rounding=1.2, z=3)
psd_style.flowline(ax, 46.8, 44.8, 63.4, 44.8, color="#333333", lw=1.6, head=11, z=4)
ax.text(56.0, 50.4, "embeddings", ha="center", va="center", fontsize=5.5,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
ax.text(56.0, 47.6, "+ proposals", ha="center", va="center", fontsize=5.5,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

# ---- 物理层（Φ）：入口卡 -> 双轨道卡 -> 分高平移出接口桥 ----
# 净空规整：两栏卡片统一 x=4/25（w=19），面板左右边距均 2 单位
wcard(4.0, 17.0, 40, 5.6, "Unlabeled streams $(T$, 24, 3)",
      P_CARD, P_CARD_EDGE, PD, title_fs=6.0)
wcard(4.0, 32.5, 19, 9.5, "SSL\npretraining", P_CARD, P_CARD_EDGE, PD, title_fs=6.3)
wcard(25.0, 32.5, 19, 9.5, "Motion words\nquantization", P_CARD, P_CARD_EDGE, PD,
      title_fs=6.0)
wcard(4.0, 50.5, 19, 7.5, "Dynamics\nembeddings", P_CARD, P_CARD_EDGE, PD, title_fs=6.3)
wcard(25.0, 50.5, 19, 7.5, "Behavior\nproposals", P_CARD, P_CARD_EDGE, PD, title_fs=6.3)
# 主流线统一 1.6pt；顶出路由承 v13：左卡走高线 y=68.8、右卡走低线 y=65.4（零交叉）
psd_style.flowline(ax, 13.5, 22.6, 13.5, 32.3, color=PD)
psd_style.flowline(ax, 34.5, 22.6, 34.5, 32.3, color=PD)
psd_style.flowline(ax, 13.5, 42.2, 13.5, 50.3, color=PD)
psd_style.flowline(ax, 34.5, 42.2, 34.5, 50.3, color=PD)
psd_style.flowline(ax, 13.5, 58.2, 13.5, 68.8, color=PD)
psd_style.flowline(ax, 13.5, 68.8, 52.5, 68.8, color=PD)   # 52.5 = 精确触桥左缘
psd_style.flowline(ax, 34.5, 58.2, 34.5, 65.4, color=PD)
psd_style.flowline(ax, 34.5, 65.4, 52.5, 65.4, color=PD)

# ---- 语义层（Ω）：入口卡 -> 白卡×3 -> 出口焦点深块（净空统一 x=66, w=40） ----
wcard(66, 17.0, 40, 5.6, "Rule-engine seeds (budget $B$)",
      S_CARD, S_CARD_EDGE, SD_, title_fs=6.0)
wcard(66, 26.5, 40, 5.6, "Anchor learning", S_CARD, S_CARD_EDGE, SD_, title_fs=6.3)
wcard(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels",
      S_CARD, S_CARD_EDGE, SD_, title_fs=6.3)
wcard(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)",
      S_CARD, S_CARD_EDGE, SD_, title_fs=6.0)
deepblock(66, 60.5, 40, 5.6, r"Classification under $\mathcal{Y}$", deep=SD_,
          title_fs=6.6)
psd_style.flowline(ax, 86, 22.8, 86, 26.3, color=SD_)
psd_style.flowline(ax, 86, 32.3, 86, 36.3, color=SD_)
psd_style.flowline(ax, 86, 45.7, 86, 49.3, color=SD_)
psd_style.flowline(ax, 86, 56.7, 86, 60.3, color=SD_)

# ---- 演化标注带（焦点深块 #2 + 纯垂直虚线回 seeds；虚线=约束回路专用线型） ----
psd_style.card(ax, 24, 2.0, 62, 6.0, fill=SD_, edge=None, rounding=1.2, z=5)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.6, color="white",
        fontweight="bold", zorder=7)
psd_style.flowline(ax, 84, 8.3, 84, 16.6, color=SD_, lw=1.3, ls=(0, (3, 2)), head=9)

# ---- 出图（尺度修复：无 tight，MediaBox 严格等于画布） ----
pdf_path = OUT / "fig1_framework_overview.pdf"
fig.savefig(pdf_path)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600)
fig.savefig(PREVIEW / "preview_fig1.png", dpi=150)
print("fig1 v15 (direction C tone-ladder + fixed-canvas) saved")

# ---- 门禁（G1 印刷稳健 + G2 标签碰撞 + G5 排印 + G6 标记落位 + G4 文件级） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig1-v15", {"physics": psd_style.PHYS_EDGE, "semantic": psd_style.SEM_EDGE},
    redundancy="tone ladder + position + labels")
gates.gate_labels(fig, ax, list(ax.texts), name="fig1-v15")
gates.gate_typography(fig, min_pt=5.5, name="fig1-v15")
gates.gate_marks_in_axes(fig, name="fig1-v15")
gates.gate_pdf(pdf_path, max_w_pt=246.3)
