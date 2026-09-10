# -*- coding: utf-8 -*-
r"""fig1 v7 — PSD 框架总览（R25 审美统一：psd_style 单一真源取色 + Arial 字族；版式承 v6）。

v5 judge 四处修复: ①删除接口带->聚类间隙中向左鼓出的反向弧箭头(被读作
未说明的 Ω→Φ 反馈, 与"单向通信"spec 矛盾; 正向 L arrow(59.5,39,66,39) 保留);
②物理层四行模块拉开纵向间距填满容器(原顶部 1/3 悬空失衡);
③Dynamics/Behavior 两盒加宽 17->18.5(文字贴边); ④演化带虚线箭头终点
落在 Rule-engine seeds 盒底缘(原落容器边框含糊)。
继承 v5: 印刷尺寸 3.42in 1:1、落地字号、配色。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
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

def box(x, y, w, h, text, fill, edge, fs=6.8, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.0",
                                facecolor=fill, edgecolor=edge, linewidth=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, fontweight=weight, linespacing=1.25, zorder=3)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=0.9, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, color=color, lw=lw, linestyle=ls))

# ---- 层容器 ----
ax.add_patch(FancyBboxPatch((2, 14), 44, 62, boxstyle="round,pad=0.3,rounding_size=1.2",
                            facecolor="white", edgecolor=C_PHYS_EDGE, linewidth=1.4))
ax.text(24, 79.8, r"Physics layer $\Phi$ (frozen)", ha="center",
        fontsize=7.6, fontweight="bold", color=C_PHYS_EDGE)
ax.add_patch(FancyBboxPatch((64, 14), 44, 62, boxstyle="round,pad=0.3,rounding_size=1.2",
                            facecolor="white", edgecolor=C_SEM_EDGE, linewidth=1.4))
ax.text(86, 79.8, r"Semantic layer $\Omega$ (revisable)", ha="center",
        fontsize=7.6, fontweight="bold", color=C_SEM_EDGE)

# ---- 中央接口带 ----
ax.add_patch(Rectangle((52.5, 14), 7, 62, facecolor=C_IFACE_FILL, edgecolor=C_IFACE_EDGE, linewidth=0.7))
ax.text(56, 45, "embeddings + proposals", ha="center", va="center",
        fontsize=6.0, color=C_ARROW, rotation=90)

# ---- 物理层（四行拉开: 16.5 / 30 / 48 / 输出 58+62, 填满 14-76）----
box(4.0, 16.5, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", C_PHYS_FILL, C_PHYS_EDGE, fs=5.9)
box(4.5, 28.5, 19.5, 9.5, "SSL\npretraining", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
box(25, 28.5, 19.5, 9.5, "Motion words\nquantization", C_PHYS_FILL, C_PHYS_EDGE, fs=6.0)
box(4.8, 41.5, 19.0, 7.5, "Dynamics\nembeddings", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
box(24.0, 41.5, 19.0, 7.5, "Behavior\nproposals", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
arrow(14.25, 22.1, 14.25, 28.5)
arrow(34.75, 22.1, 34.75, 28.5)
arrow(14.25, 38.0, 14.4, 41.5)
arrow(34.75, 38.0, 33.5, 41.5)
arrow(14.5, 49.0, 14.5, 61.0); arrow(14.5, 61.0, 52.5, 61.0, lw=1.0)
arrow(33.5, 49.0, 33.5, 67.0); arrow(33.5, 67.0, 52.5, 67.0, lw=1.0)
arrow(59.5, 39.0, 66, 39.0, lw=1.0)  # 唯一正向接口箭头: interface -> clustering (Sec 3.3.2)

# ---- 语义层 ----
box(65.5, 16.5, 41.5, 5.6, "Rule-engine seeds (budget $B$)", C_SEM_FILL, C_SEM_EDGE, fs=5.9)
box(66, 26.5, 40, 5.6, "Anchor learning", C_SEM_FILL, C_SEM_EDGE, fs=6.4)
box(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", C_SEM_FILL, C_SEM_EDGE, fs=6.4)
box(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)", C_SEM_FILL, C_SEM_EDGE, fs=6.0)
box(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", "white", C_TEXT, fs=6.8)
arrow(86, 22.1, 86, 26.5)
arrow(86, 32.1, 86, 36.5)
arrow(86, 45.5, 86, 49.5)
arrow(86, 56.5, 86, 60.5)

# ---- 演化标注带（虚线箭头落 Rule-engine seeds 盒底缘）----
box(24, 2.0, 62, 6.0, "", "white", C_SEM_EDGE)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.6, color=C_SEM_EDGE, fontweight="bold")
arrow(84, 8.0, 86.2, 16.4, color=C_SEM_EDGE, lw=1.0, ls=(0, (3, 2)))

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig1_framework_overview.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig1 v7 (R24 fixes: no reverse arc, rebalanced, widened boxes, taxonomy arrow anchored) saved")
