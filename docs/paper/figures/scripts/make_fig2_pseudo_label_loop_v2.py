# -*- coding: utf-8 -*-
r"""fig2 v2 — 语义层迭代闭环（期刊规范, 2026-09-04）.

修复清单（对照缺陷清单）:
1. κ<τ 分支箭头: 菱形→AL queue→new seeds（不跳过 queue）
2. 判断节点纳入环内（assign 每轮发生）
3. Initialize prototypes 拆双职责: 初始化入环, verified seeds 回流指向 prototype re-estimate
4. 颜色语义与 fig1 对齐: 青色=人工/种子侧, 橙色=自动学习侧（fig1 同款色板）
5. budget 标注改新口径: warm-start 20-clip usable, expansion ≤200 clips
6. 图内不再重复 caption 的 frozen 句（放 caption）
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")
C_PHYS_FILL, C_PHYS_EDGE = "#DAFFFF", "#0E7490"   # 人工/种子侧（青）
C_SEM_FILL, C_SEM_EDGE = "#FFE3DA", "#C2410C"     # 自动学习侧（橙）
C_TEXT, C_ARROW = "#111111", "#374151"

fig, ax = plt.subplots(figsize=(9.6, 7.0))
ax.set_xlim(0, 96); ax.set_ylim(0, 70)
ax.axis("off")

def box(x, y, w, h, text, fill, edge, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            color=C_TEXT, linespacing=1.35)

def arrow(x1, y1, x2, y2, color=C_ARROW, lw=1.7, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))

def diamond(cx, cy, w, h, text):
    ax.add_patch(plt.Polygon([(cx, cy+h/2), (cx+w/2, cy), (cx, cy-h/2), (cx-w/2, cy)],
                             facecolor="white", edgecolor=C_ARROW, linewidth=1.5))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.6, color=C_TEXT, linespacing=1.3)

# ---- 环路主体（顺时针） ----
box(20, 52, 26, 6.5, "Seed anchors $A$\n(rule-engine coarse labels)", C_PHYS_FILL, C_PHYS_EDGE)
box(38, 60.5, 22, 5.4, "Initialize\nprototypes", C_SEM_FILL, C_SEM_EDGE, fs=8.6)
diamond(66, 55, 30, 15, "Assign proposals to\nnearest prototype\n(confidence $\kappa$)")
box(74, 32, 18, 7.0, "Pseudo-labeled\npool", C_SEM_FILL, C_SEM_EDGE)
box(58, 14, 20, 6.6, "Update classifier $\Omega$\n(seeds $\cup$ pool)", C_SEM_FILL, C_SEM_EDGE, fs=8.6)
box(30, 14, 20, 6.6, "Re-estimate\nprototypes", C_SEM_FILL, C_SEM_EDGE, fs=8.6)

arrow(33, 55.2, 38, 62.0)                 # seeds -> init
arrow(60, 62.5, 60, 58.5)                 # init -> diamond
arrow(80, 50.5, 83, 39)                   # κ>=τ -> pool（右分支）
ax.text(84.5, 44, r"$\kappa \geq \tau$", fontsize=10.5, color=C_SEM_EDGE, fontweight="bold")
arrow(83, 32, 74, 20.5, rad=-0.15)        # pool -> update
arrow(58, 17.3, 50, 17.3)                 # update -> re-estimate
arrow(40, 20.6, 66, 49.2, rad=0.25)       # re-estimate -> diamond（闭环回判断）

# ---- 左侧人工回路（κ<τ: diamond -> AL queue -> verified seeds -> re-estimate） ----
box(6, 40, 22, 7.6, "Active-learning queue\n(human annotation,\nbudget 100-200 clips)", C_PHYS_FILL, C_PHYS_EDGE, fs=8.4)
box(2, 24, 18, 6.0, "New verified\nseeds", C_PHYS_FILL, C_PHYS_EDGE, fs=8.8)
arrow(51, 50.5, 28, 44, rad=0.2)          # κ<τ -> queue（直达 queue, 修复#1）
ax.text(34, 45.5, r"$\kappa < \tau$", fontsize=10.5, color=C_PHYS_EDGE, fontweight="bold")
arrow(17, 40, 11, 30)                     # queue -> verified seeds
arrow(20, 30, 34, 17.5, rad=0.2)          # verified seeds -> re-estimate（修复#3: 回流到 re-estimate）

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight")
print("fig2 v2 saved")
