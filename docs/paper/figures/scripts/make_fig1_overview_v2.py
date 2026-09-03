# -*- coding: utf-8 -*-
r"""fig1 v4 — PSD 框架总览 hero 图（期刊规范, 2026-09-04 定稿版）."""
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

fig, ax = plt.subplots(figsize=(11.0, 6.6))
ax.set_xlim(0, 110); ax.set_ylim(0, 66)
ax.axis("off")

def box(x, y, w, h, text, fill, edge, fs=9, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, fontweight=weight, linespacing=1.35)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=lw, linestyle=ls))

# ---- 层容器 ----
ax.add_patch(FancyBboxPatch((2, 10), 44, 48, boxstyle="round,pad=0.3,rounding_size=1.6",
                            facecolor="white", edgecolor=C_PHYS_EDGE, linewidth=2.0))
ax.text(24, 54.6, r"Physics layer $\Phi$" + "  (frozen after training)", ha="center",
        fontsize=11.5, fontweight="bold", color=C_PHYS_EDGE)
ax.add_patch(FancyBboxPatch((64, 10), 44, 48, boxstyle="round,pad=0.3,rounding_size=1.6",
                            facecolor="white", edgecolor=C_SEM_EDGE, linewidth=2.0))
ax.text(86, 54.6, r"Semantic layer $\Omega$" + "  (revisable)", ha="center",
        fontsize=11.5, fontweight="bold", color=C_SEM_EDGE)

# ---- 中央接口带 ----
ax.add_patch(Rectangle((52.5, 10), 7, 48, facecolor=C_IFACE_FILL, edgecolor=C_IFACE_EDGE, linewidth=1.0))
ax.text(56, 34, "interface:  embeddings + proposals only", ha="center", va="center",
        fontsize=8.4, color="#374151", rotation=90, linespacing=1.5)

# ---- 物理层（streams 底部 → 双分支 → 产物 → 接口带） ----
box(4.5, 12.5, 39, 5.4, "Unlabeled skeleton streams  ($T$, 24, 3)", C_PHYS_FILL, C_PHYS_EDGE)
box(6, 23, 17, 7.6, "Self-supervised\npretraining\n(AimCLR-adapted)", C_PHYS_FILL, C_PHYS_EDGE, fs=8.4)
box(25, 23, 17, 7.6, "Motion-word\nquantization\n(SMQ)", C_PHYS_FILL, C_PHYS_EDGE, fs=8.4)
box(6, 35.5, 17, 7.0, "Dynamics\nembeddings", C_PHYS_FILL, C_PHYS_EDGE, fs=9)
box(25, 35.5, 17, 7.0, "Behavior\nproposals", C_PHYS_FILL, C_PHYS_EDGE, fs=9)
arrow(14.5, 17.9, 14.5, 23)     # streams -> pretraining（垂直，语义清晰）
arrow(33.5, 17.9, 33.5, 23)     # streams -> SMQ
arrow(14.5, 30.6, 14.5, 35.5)   # pretraining -> embeddings
arrow(33.5, 30.6, 33.5, 35.5)   # SMQ -> proposals
# 产物 -> 接口带（绕过 proposals 框上方走线）
arrow(14.5, 42.5, 14.5, 45.5); arrow(14.5, 45.5, 52.5, 45.5, lw=1.8)
arrow(33.5, 42.5, 33.5, 47.5); arrow(33.5, 47.5, 52.5, 47.5, lw=1.8)
ax.text(38, 48.6, "embeddings + proposals", fontsize=8.2, style="italic", color=C_TEXT)
# 出接口带 -> 聚类框（垂直到位）
arrow(59.5, 46.5, 61.5, 46.5, lw=1.8); arrow(61.5, 46.5, 61.5, 34.2, lw=1.8); arrow(61.5, 34.2, 66, 34.2, lw=1.8)

# ---- 语义层 ----
box(66, 12.5, 40, 5.4, "Rule-engine seeds  (small budget)", C_SEM_FILL, C_SEM_EDGE)
box(66, 21.5, 40, 5.4, "Anchor learning", C_SEM_FILL, C_SEM_EDGE)
box(66, 30.5, 40, 7.4, "Prototype clustering +\nconfidence-filtered pseudo-labeling", C_SEM_FILL, C_SEM_EDGE, fs=8.4)
box(66, 41.5, 40, 5.4, "Semi-supervised self-training", C_SEM_FILL, C_SEM_EDGE)
box(70, 61, 34, 4.6, r"Behavior classification under $\mathcal{Y}$", "white", C_TEXT, fs=9.5)
arrow(86, 17.9, 86, 21.5)
arrow(86, 26.9, 86, 30.5)
arrow(86, 37.9, 86, 41.5)
arrow(86, 46.9, 86, 61)
# 聚类自迭代环（框左缘小弧）
ax.annotate("", xy=(68.2, 37.2), xytext=(68.2, 31.2),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                            connectionstyle="arc3,rad=-1.1"))

# ---- 演化标注带（文字居中不溢出） ----
box(28, 1.0, 56, 4.6, "", "white", C_SEM_EDGE)
ax.text(56, 3.3, r"Taxonomy evolves $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains, physics stays frozen",
        ha="center", va="center", fontsize=8.8, color=C_SEM_EDGE, fontweight="bold")
arrow(84, 5.6, 96, 10, color=C_SEM_EDGE, lw=1.8, ls=(0, (4, 2)))

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "fig1_framework_overview.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600, bbox_inches="tight")
print("fig1 v4 saved")
