# -*- coding: utf-8 -*-
"""fig1 框架总览图（hero figure，落点 §1 尾）生成脚本。

规格来源: docs/paper/figure-specs.md §fig1
  - 左=物理层 Φ（淡青 #DAFFFF 系），右=语义层 Ω（淡橙 #FFE3DA 系）
  - 中央灰色窄带（#DADADA）强调"只通过 embeddings + proposals 通信"
  - "Y → Y′ only Ω retrains" 虚线箭头视觉显性化
  - 白底 / ≤6 色 / 无 3D、阴影、渐变 / 矢量 PDF 导出
输出: docs/paper/figures/fig1_framework_overview.pdf（矢量）+ .png（预览）
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 配色（figure-specs: 淡青/淡橙/灰/黑，共 4 色系）----
CYAN_FILL = "#DAFFFF"      # 物理层模块
ORANGE_FILL = "#FFE3DA"    # 语义层模块
BAND_GRAY = "#DADADA"      # 接口带
INK = "#000000"            # 文字/边框主色
EDGE_DARK = "#333333"      # 箭头
CYAN_DARK = "#0E7490"      # 青系深色描边
ORANGE_DARK = "#C2410C"    # 橙系深色描边（兼作 Y→Y′ 高亮）
NOTE_GRAY = "#888888"      # FIGURE_SOURCE 小注

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "text.color": INK,
})

fig, ax = plt.subplots(figsize=(12.5, 7.0))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_aspect("auto")
fig.patch.set_facecolor("white")
ax.axis("off")


def box(x, y, w, h, text, fill, edge, lw=1.2, fs=9.0, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=3,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", zorder=4)


def arrow(p0, p1, color=EDGE_DARK, lw=1.4, style="-|>", ls="solid",
          rad=0.0, zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14, linewidth=lw,
        color=color, linestyle=ls, zorder=zorder,
        connectionstyle=f"arc3,rad={rad}",
    ))


# ================= 容器：物理层 Φ / 语义层 Ω =================
ax.add_patch(FancyBboxPatch((2, 12), 39, 76, boxstyle="round,pad=0.6,rounding_size=1.5",
                            facecolor="none", edgecolor=CYAN_DARK, linewidth=1.8, zorder=1))
ax.text(21.5, 84.5, "Physics Layer Φ  (frozen after training)",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

ax.add_patch(FancyBboxPatch((59, 12), 39, 76, boxstyle="round,pad=0.6,rounding_size=1.5",
                            facecolor="none", edgecolor=ORANGE_DARK, linewidth=1.8, zorder=1))
ax.text(78.5, 84.5, "Semantic Layer Ω  (revisable)",
        ha="center", va="center", fontsize=11.5, fontweight="bold", color=INK)

# 中央接口窄带
ax.add_patch(FancyBboxPatch((44, 12), 12, 76, boxstyle="round,pad=0.4,rounding_size=1.0",
                            facecolor=BAND_GRAY, edgecolor="#AAAAAA", linewidth=1.0, zorder=0))
ax.text(50, 50, "communication interface:\nembeddings + proposals only",
        ha="center", va="center", fontsize=8.5, rotation=90, color=INK)

# ================= 物理层内部 =================
box(5.5, 64, 15.5, 11, "Self-supervised\npretraining\n(AimCLR-adapted)", CYAN_FILL, CYAN_DARK)
box(22.5, 64, 15.5, 11, "Motion-word\nquantization\n(SMQ)", CYAN_FILL, CYAN_DARK)
box(5.5, 46, 15.5, 10, "Dynamics\nembeddings", CYAN_FILL, CYAN_DARK)
box(22.5, 46, 15.5, 10, "Behavior\nproposals", CYAN_FILL, CYAN_DARK)
box(5.5, 16, 32.5, 10, "Unlabeled skeleton streams  (T, 24, 3)", CYAN_FILL, CYAN_DARK, fs=9.5)

arrow((13.2, 26.4), (13.2, 63.6))          # A -> B
arrow((30.2, 26.4), (30.2, 63.6))          # A -> D
arrow((13.2, 63.6), (13.2, 56.4))          # B -> C
arrow((30.2, 63.6), (30.2, 56.4))          # D -> E

# ================= 语义层内部 =================
box(62.5, 13, 32.5, 9, "Rule-engine seeds  (100–200 clips)", ORANGE_FILL, ORANGE_DARK)
box(62.5, 28, 32.5, 9, "Anchor learning", ORANGE_FILL, ORANGE_DARK)
box(62.5, 45, 32.5, 13, "Prototype clustering +\npseudo-labeling loop", ORANGE_FILL, ORANGE_DARK, fs=9.5)
box(62.5, 66, 32.5, 10, "Semi-supervised self-training\n+ active learning", ORANGE_FILL, ORANGE_DARK)

arrow((78.7, 22.4), (78.7, 27.6))           # F -> G
arrow((78.7, 37.4), (78.7, 44.6))           # G -> H
arrow((78.7, 58.4), (78.7, 65.6))           # H -> I

# ================= 接口通信：embeddings / proposals -> H =================
arrow((21.2, 51), (62.1, 51), lw=1.8)       # C -- embeddings --> H
ax.text(41.5, 53.0, "embeddings", ha="center", va="bottom", fontsize=8.5, style="italic")
arrow((38.2, 49), (62.1, 47.5), lw=1.8)     # E -- proposals --> H
ax.text(41.0, 43.6, "proposals", ha="center", va="top", fontsize=8.5, style="italic")

# ================= 输出节点 J =================
box(36, 91.5, 28, 7, "Behavior classification under taxonomy Y", "white", INK, lw=1.6, fs=9.5)
arrow((78.7, 76.4), (78.7, 88))
arrow((78.7, 88), (60.5, 88))
arrow((60.5, 88), (60.5, 91.1))

# ================= Y → Y′ 演化路径（虚线高亮，叙事核心）=================
box(33, 1.5, 34, 7, "Taxonomy evolves   Y → Y′", "white", ORANGE_DARK, lw=2.0, fs=9.5, bold=True)
arrow((55, 8.9), (80, 11.6), color=ORANGE_DARK, lw=2.2, ls=(0, (6, 3)))
ax.text(70.5, 7.2, "only Ω retrains — physics layer stays frozen",
        ha="center", va="center", fontsize=8.8, color=ORANGE_DARK, fontweight="bold")

# ================= FIGURE_SOURCE 注记 =================
ax.text(99, -1.2, "FIGURE_SOURCE: scripts/make_fig1_overview.py · PSD-Framework · W17 · 2026-08-24",
        ha="right", va="top", fontsize=6, color=NOTE_GRAY)

fig.savefig(OUT_DIR / "fig1_framework_overview.pdf", bbox_inches="tight", pad_inches=0.08)
fig.savefig(OUT_DIR / "fig1_framework_overview.png", dpi=200, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)
print("fig1 written:", OUT_DIR / "fig1_framework_overview.pdf")
