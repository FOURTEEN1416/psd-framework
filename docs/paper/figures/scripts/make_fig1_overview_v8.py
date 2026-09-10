# -*- coding: utf-8 -*-
r"""fig1 v8 — PSD 框架总览（R27 现代扁平重设计：软阴影卡片+大圆角+无边容器；版式承 v7）。

R27 设计语言（用户两轮判丑后骨架级重做，坐标/文字/箭头拓扑与 v7 完全一致）：
- 层容器 = 浅色纯色块（无边）+ 软阴影；
- 内盒 = 白色扁平卡片 + 细彩边 + 软阴影（flat card）；
- 箭头加粗 1.25 / 箭头头 8.5；标题 8.0pt。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
C_PHYS_FILL, C_PHYS_EDGE = psd_style.PHYS_FILL, psd_style.PHYS_EDGE
C_SEM_FILL, C_SEM_EDGE = psd_style.SEM_FILL, psd_style.SEM_EDGE
C_IFACE_FILL, C_IFACE_EDGE = psd_style.IFACE_FILL, psd_style.IFACE_EDGE
C_TEXT, C_ARROW = psd_style.INK, psd_style.ARROW

fig, ax = plt.subplots(figsize=(3.42, 2.62))
ax.set_xlim(0, 110); ax.set_ylim(0, 84)
ax.axis("off")

def box(x, y, w, h, text, fill, edge, fs=6.8, weight="normal", z=5):
    psd_style.card(ax, x, y, w, h, fill=fill, edge=edge, lw=1.1,
                   rounding=1.5, dx=0.45, dy=-0.75, z=z)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, fontweight=weight, linespacing=1.25, zorder=z + 2)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.25, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=8.5, color=color, lw=lw, linestyle=ls))

# ---- 层容器（无边浅色块 + 阴影） ----
psd_style.card(ax, 2, 14, 44, 62, fill=C_PHYS_FILL, edge=None, rounding=2.2, dx=0.7, dy=-1.1, z=2)
ax.text(24, 79.8, r"Physics layer $\Phi$ (frozen)", ha="center",
        fontsize=8.0, fontweight="bold", color=psd_style.PHYS_TEXT, zorder=6)
psd_style.card(ax, 64, 14, 44, 62, fill=C_SEM_FILL, edge=None, rounding=2.2, dx=0.7, dy=-1.1, z=2)
ax.text(86, 79.8, r"Semantic layer $\Omega$ (revisable)", ha="center",
        fontsize=8.0, fontweight="bold", color=C_SEM_EDGE, zorder=6)

# ---- 中央接口带（圆角灰条 + 阴影） ----
psd_style.card(ax, 52.5, 14, 7, 62, fill=C_IFACE_FILL, edge=C_IFACE_EDGE, lw=0.8,
               rounding=1.8, dx=0.45, dy=-0.75, z=3)
ax.text(56, 45, "embeddings + proposals", ha="center", va="center",
        fontsize=6.0, color=C_ARROW, rotation=90, zorder=5)

# ---- 物理层（四行拉开: 16.5 / 30 / 48 / 输出 58+62, 填满 14-76）----
box(4.0, 16.5, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", "white", C_PHYS_EDGE, fs=5.9)
box(4.5, 28.5, 19.5, 9.5, "SSL\npretraining", "white", C_PHYS_EDGE, fs=6.4)
box(25, 28.5, 19.5, 9.5, "Motion words\nquantization", "white", C_PHYS_EDGE, fs=6.0)
box(4.8, 41.5, 19.0, 7.5, "Dynamics\nembeddings", "white", C_PHYS_EDGE, fs=6.4)
box(24.0, 41.5, 19.0, 7.5, "Behavior\nproposals", "white", C_PHYS_EDGE, fs=6.4)
arrow(14.25, 22.1, 14.25, 28.5)
arrow(34.75, 22.1, 34.75, 28.5)
arrow(14.25, 38.0, 14.4, 41.5)
arrow(34.75, 38.0, 33.5, 41.5)
arrow(14.5, 49.0, 14.5, 61.0); arrow(14.5, 61.0, 52.5, 61.0)
arrow(33.5, 49.0, 33.5, 67.0); arrow(33.5, 67.0, 52.5, 67.0)
arrow(59.5, 39.0, 66, 39.0)  # 唯一正向接口箭头: interface -> clustering (Sec 3.3.2)

# ---- 语义层 ----
box(65.5, 16.5, 41.5, 5.6, "Rule-engine seeds (budget $B$)", "white", C_SEM_EDGE, fs=5.9)
box(66, 26.5, 40, 5.6, "Anchor learning", "white", C_SEM_EDGE, fs=6.4)
box(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", "white", C_SEM_EDGE, fs=6.4)
box(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)", "white", C_SEM_EDGE, fs=6.0)
box(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", "white", C_TEXT, fs=6.8, weight="bold")
arrow(86, 22.1, 86, 26.5)
arrow(86, 32.1, 86, 36.5)
arrow(86, 45.5, 86, 49.5)
arrow(86, 56.5, 86, 60.5)

# ---- 演化标注带（虚线箭头落 Rule-engine seeds 盒底缘）----
psd_style.card(ax, 24, 2.0, 62, 6.0, fill="white", edge=C_SEM_EDGE, lw=1.2,
               rounding=1.5, dx=0.45, dy=-0.75, z=5)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.6, color=C_SEM_EDGE, fontweight="bold", zorder=7)
arrow(84, 8.0, 86.2, 16.4, color=C_SEM_EDGE, lw=1.25, ls=(0, (3, 2)))

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig1_framework_overview.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig1 v8 (R27 flat-card redesign: shadow containers + white cards) saved")
