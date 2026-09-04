# -*- coding: utf-8 -*-
r"""fig1 v5 — PSD 框架总览（期刊印刷尺寸设计版, 2026-09-05）。

R8 视觉验收教训: v4 画布 11in 缩到 8.6cm 栏宽后落地字号 2.6pt 不可读。
本版按最终印刷尺寸设计: figsize=(3.42, 2.55)in = cas-sc \linewidth 1:1,
字号即落地字号（6.2-8.5pt）; 标签缩短, 细节移入 caption。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

C_PHYS_FILL, C_PHYS_EDGE = "#DAFFFF", "#0E7490"
C_SEM_FILL, C_SEM_EDGE = "#FFE3DA", "#C2410C"
C_IFACE_FILL, C_IFACE_EDGE = "#EFEFEF", "#9CA3AF"
C_TEXT, C_ARROW = "#111111", "#374151"

# 1:1 印刷尺寸: 8.6cm = 3.39in, 取 3.42 留边
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
        fontsize=6.0, color="#374151", rotation=90)

# ---- 物理层 ----
box(4.5, 16.5, 39, 5.6, "Unlabeled streams $(T$, 24, 3)", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
box(4.5, 27.5, 19.5, 9.5, "SSL\npretraining", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
box(25, 27.5, 19.5, 9.5, "Motion words\nquantization", C_PHYS_FILL, C_PHYS_EDGE, fs=6.0)
box(6, 42.5, 17, 7.5, "Dynamics\nembeddings", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
box(25, 42.5, 17, 7.5, "Behavior\nproposals", C_PHYS_FILL, C_PHYS_EDGE, fs=6.4)
arrow(14.25, 22.1, 14.25, 27.5)
arrow(34.75, 22.1, 34.75, 27.5)
arrow(14.25, 37.0, 14.4, 42.5)
arrow(34.75, 37.0, 33.5, 42.5)
arrow(14.5, 50.0, 14.5, 54.0); arrow(14.5, 54.0, 52.5, 54.0, lw=1.0)
arrow(33.5, 50.0, 33.5, 57.0); arrow(33.5, 57.0, 52.5, 57.0, lw=1.0)
arrow(59.5, 55.5, 66, 55.5, lw=1.0)

# ---- 语义层 ----
box(66, 16.5, 40, 5.6, "Rule-engine seeds (budget $B$)", C_SEM_FILL, C_SEM_EDGE, fs=6.4)
box(66, 26.5, 40, 5.6, "Anchor learning", C_SEM_FILL, C_SEM_EDGE, fs=6.4)
box(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", C_SEM_FILL, C_SEM_EDGE, fs=6.4)
box(66, 49.5, 40, 7.0, "Semi-supervised\nself-training", C_SEM_FILL, C_SEM_EDGE, fs=6.4)
box(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", "white", C_TEXT, fs=6.8)
arrow(86, 22.1, 86, 26.5)
arrow(86, 32.1, 86, 36.5)
arrow(86, 45.5, 86, 49.5)
arrow(86, 56.5, 86, 60.5)
ax.annotate("", xy=(68.0, 43.5), xytext=(68.0, 38.0),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=0.8,
                            connectionstyle="arc3,rad=-1.1"))

# ---- 演化标注带 ----
box(24, 2.0, 62, 6.0, "", "white", C_SEM_EDGE)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.6, color=C_SEM_EDGE, fontweight="bold")
arrow(84, 8.0, 96, 14, color=C_SEM_EDGE, lw=1.0, ls=(0, (3, 2)))

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig1_framework_overview.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig1 v5 (print-size) saved")
