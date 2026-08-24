# -*- coding: utf-8 -*-
"""fig2 语义层迭代闭环细节图（落点 §3.3）生成脚本。

规格来源: docs/paper/figure-specs.md §fig2
  - 主闭环顺时针：C(判定) → L(伪标签池) → T(更新分类器) → R(重估原型) → C
  - τ 阈值分叉是视觉焦点（κ ≥ τ / κ < τ 两条出边加粗着色）
  - AL 节点旁标注人工预算 100–200 clips；物理编码器全程冻结注记
  - 白底 / ≤6 色 / 矢量 PDF 导出
输出: docs/paper/figures/fig2_pseudo_label_loop.pdf（矢量）+ .png（预览）
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CYAN_FILL = "#DAFFFF"
ORANGE_FILL = "#FFE3DA"
INK = "#000000"
EDGE_DARK = "#333333"
CYAN_DARK = "#0E7490"
ORANGE_DARK = "#C2410C"
NOTE_GRAY = "#888888"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "text.color": INK,
})

fig, ax = plt.subplots(figsize=(10.5, 8.0))
ax.set_xlim(0, 100)
ax.set_ylim(-4, 100)
ax.set_aspect("auto")
fig.patch.set_facecolor("white")
ax.axis("off")


def box(cx, cy, w, h, text, fill, edge, lw=1.2, fs=9.0, bold=False):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    ))
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=4)


def arrow(p0, p1, color=EDGE_DARK, lw=1.5, ls="solid", rad=0.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=15, linewidth=lw,
        color=color, linestyle=ls, zorder=2,
        connectionstyle=f"arc3,rad={rad}",
    ))


# ================= 种子链（左上；种子/验证=淡青，处理=淡橙）=================
box(14, 90, 24, 9, "Seed anchors A\n(rule-engine coarse labels)", CYAN_FILL, CYAN_DARK)
box(14, 74, 24, 9, "Initialize prototypes", ORANGE_FILL, ORANGE_DARK)
arrow((14, 85.1), (14, 78.9))                       # S -> P

# ================= 判定菱形 C（白底加粗描边，τ 分叉焦点）=================
dcx, dcy, dw, dh = 52, 74, 18, 10.5
diamond = [(dcx, dcy + dh), (dcx + dw, dcy), (dcx, dcy - dh), (dcx - dw, dcy)]
ax.add_patch(Polygon(diamond, closed=True, facecolor="white", edgecolor=INK,
                     linewidth=2.2, zorder=3))
ax.text(dcx, dcy, "Assign proposals to\nnearest prototype\n(confidence κ)",
        ha="center", va="center", fontsize=8.6, zorder=4)

arrow((26.4, 74), (33.4, 74), lw=1.8)               # P -> C

# ================= 主闭环（顺时针）=================
box(84, 50, 20, 12, "Pseudo-labeled\npool", ORANGE_FILL, ORANGE_DARK)      # L 右
box(66, 16, 26, 11, "Update classifier Ω\n(seeds ∪ pool)", ORANGE_FILL, ORANGE_DARK)   # T 右下
box(34, 16, 26, 11, "Re-estimate\nprototypes", ORANGE_FILL, ORANGE_DARK)   # R 左下

# C --κ≥τ--> L（加粗深橙 = 主通路）
arrow((61.5, 68.0), (80.0, 56.6), lw=2.6, color=ORANGE_DARK, rad=-0.15)
ax.text(76.8, 65.8, "κ ≥ τ", fontsize=11.5, fontweight="bold", color=ORANGE_DARK)

arrow((81.0, 43.6), (69.5, 22.0), rad=-0.12)        # L -> T
arrow((52.4, 16), (47.6, 16))                        # T -> R
arrow((32.0, 21.9), (40.2, 65.6), rad=0.28)          # R -> C（闭环回边）

# ================= 活动学习支路（κ < τ，左下，淡青）=================
box(14, 40, 24, 13, "Active-learning queue\n(human annotation)", CYAN_FILL, CYAN_DARK)
box(14, 57, 20, 8, "New verified seeds", CYAN_FILL, CYAN_DARK)

# C --κ<τ--> AL（加粗深青 = 低置信分流）
arrow((35.6, 68.6), (17.8, 47.0), lw=2.6, color=CYAN_DARK, rad=0.22)
ax.text(19.5, 64.5, "κ < τ", fontsize=11.5, fontweight="bold", color=CYAN_DARK)

arrow((14, 46.9), (14, 52.7))                        # AL -> S2
arrow((14, 61.3), (14, 69.0))                        # S2 -> P

# 预算标注（AL 节点旁）
ax.text(14, 30.8, "budget: 100–200 clips", ha="center", va="center",
        fontsize=8.8, style="italic", color=CYAN_DARK)

# 冻结注记
ax.text(50, 3.2, "Physics encoder Φ stays frozen throughout the loop.",
        ha="center", va="center", fontsize=8.8, color=EDGE_DARK, style="italic")

# ================= FIGURE_SOURCE 注记 =================
ax.text(99.5, -3.6,
        "FIGURE_SOURCE: scripts/make_fig2_pseudo_label_loop.py · PSD-Framework · W17 · 2026-08-24",
        ha="right", va="top", fontsize=6, color=NOTE_GRAY)

fig.savefig(OUT_DIR / "fig2_pseudo_label_loop.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT_DIR / "fig2_pseudo_label_loop.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("fig2 written:", OUT_DIR / "fig2_pseudo_label_loop.pdf")
