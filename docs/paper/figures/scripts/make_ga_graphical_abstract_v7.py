# -*- coding: utf-8 -*-
"""graphical abstract v6 — Elsevier GA 规格（submission-package-draft.md §3）。

规格: ≤531×131 pt ≈ 7.375×1.82 in,印刷尺寸 1:1 铁律(figsize 直接落)，
  PNG 600dpi(≥250dpi 线) + PDF 矢量。R27 现代扁平重设计：双层框/微条全部
  卡片化（大圆角+软阴影，GA 坐标 1 单位≈5.3pt 故阴影偏移取小）。
  三元素不变——① 双层框(蓝=物理冻结 Φ / 红=语义可修订 Ω)
  ② Y→Y′ 演化虚线自环只穿语义层
  ③ 右侧保留率微条:NTU60 90.7(绿,墨字) / NTU120 88.9(兰紫,白字) /
    canine 28.9(红描边斜纹,条外红字+chance 注)。
数据来源: 正文 04-experiments.tex E9/E9b/E7 段(2026-09-07 10-seed 终口径)。
输出: docs/paper/figures/fig_ga_graphical_abstract.pdf + .png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
C_PHYS_FILL, C_PHYS_EDGE = psd_style.PHYS_FILL, psd_style.PHYS_EDGE
C_SEM_FILL, C_SEM_EDGE = psd_style.SEM_FILL, psd_style.SEM_EDGE
C_TEXT, C_ARROW, C_NOTE = psd_style.INK, psd_style.ARROW, "#888888"

plt.rcParams["mathtext.fontset"] = "dejavusans"  # mathtext 与 Arial 最兼容的字形集

# ---- 画布:印刷尺寸 1:1 ----
fig, ax = plt.subplots(figsize=(7.375, 1.82))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 24.7)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

def box(x, y, w, h, fill, edge):
    psd_style.card(ax, x, y, w, h, fill=fill, edge=edge, lw=1.5,
                   rounding=2.2, dx=0.35, dy=-0.6, z=3)

# ---- ① 双层框 ----
box(2, 4, 29, 16.5, C_PHYS_FILL, C_PHYS_EDGE)
ax.text(16.5, 17.3, r"Physics layer $\Phi$", ha="center", va="center",
        fontsize=9, fontweight="bold", color=psd_style.PHYS_TEXT, zorder=6)
ax.text(16.5, 13.9, "(frozen after pretraining)", ha="center", va="center",
        fontsize=6.3, style="italic", color=C_TEXT, zorder=6)
ax.text(16.5, 8.3, "self-supervised dynamics  ·  behavior proposals",
        ha="center", va="center", fontsize=6.0, color=C_TEXT, zorder=6)

box(40, 4, 29, 16.5, C_SEM_FILL, C_SEM_EDGE)
ax.text(54.5, 17.3, r"Semantic layer $\Omega$", ha="center", va="center",
        fontsize=9, fontweight="bold", color=C_SEM_EDGE, zorder=6)
ax.text(54.5, 13.9, "(revisable)", ha="center", va="center",
        fontsize=6.3, style="italic", color=C_TEXT, zorder=6)

# 接口箭头(物理→语义,只过 embeddings+proposals)
ax.add_patch(FancyArrowPatch((31.6, 12.2), (39.4, 12.2), arrowstyle="-|>",
                             mutation_scale=12, color=C_ARROW, lw=1.7))
ax.text(35.5, 14.6, "embeddings\n+ proposals", ha="center", va="center",
        fontsize=5.5, color=C_ARROW, linespacing=1.25)

# ---- ② 演化箭头:只穿语义层(框内虚线自环) ----
ax.add_patch(FancyArrowPatch((46, 7.2), (63, 7.2), arrowstyle="-|>",
                             mutation_scale=11, color=C_SEM_EDGE, lw=1.6,
                             linestyle=(0, (4, 2.2)), zorder=6,
                             connectionstyle="arc3,rad=0.32"))
ax.text(54.5, 11.0, r"taxonomy evolves $\mathcal{Y}\to\mathcal{Y}'$: only $\Omega$ retrains",
        ha="center", va="center", fontsize=5.8, color=C_SEM_EDGE, fontweight="bold")

# ---- ③ 保留率微条(圆角卡片条+阴影) ----
PANEL_X0, BAR_X0, BAR_MAX = 71.5, 81.5, 17.5   # 100% → 17.5 单位
rows = [
    ("NTU60", 90.7, False, psd_style.S_NTU60, psd_style.INK),      # 绿条浅, 白字对比不足->墨色
    ("NTU120", 88.9, False, psd_style.S_NTU120, "white"),
    ("canine", 28.9, True, psd_style.GRAY_FILL, C_SEM_EDGE),   # E7 v1 @13%-label 预算: 保留率 28.9%, 绝对精度近随机
]
ax.text((PANEL_X0 + 99) / 2, 21.0, "budget retention", ha="center", va="center",
        fontsize=6.3, fontweight="bold", color=C_TEXT)
ys = (15.6, 10.8, 6.0)
BAR_H = 3.0
for (label, val, is_boundary, bcol, txtcol), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.2, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=5.8, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=C_SEM_EDGE, lw=1.0,
                       rounding=0.8, dx=0.25, dy=-0.45, z=4, hatch="///")
        ax.text(BAR_X0 + w + 1.0, y + BAR_H / 2, f"{val:.1f}%",
                ha="left", va="center", fontsize=5.4, color=txtcol, zorder=6)
    else:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=None,
                       rounding=0.8, dx=0.25, dy=-0.45, z=4)
        ax.text(BAR_X0 + w - 0.8, y + BAR_H / 2, f"{val:.1f}%", ha="right", va="center",
                fontsize=5.6, color=txtcol, fontweight="bold", zorder=6)
ax.text((PANEL_X0 + 99) / 2, 2.2,
        "retention of own full-budget top-1 at 10% labels; canine: 9.8 vs 11.1% chance at 13% budget",
        ha="center", va="center", fontsize=4.8, color=C_NOTE)

pdf_path = OUT / "fig_ga_graphical_abstract.pdf"
png_path = OUT / "fig_ga_graphical_abstract.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

assert pdf_path.is_file() and png_path.is_file(), "GA outputs missing"
print("written:", pdf_path)
print("written:", png_path)
print("print size: 7.375 x 1.82 in (= 531 x 131 pt, Elsevier GA limit)")
