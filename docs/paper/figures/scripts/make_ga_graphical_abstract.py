# -*- coding: utf-8 -*-
"""graphical abstract — Elsevier GA 规格（submission-package-draft.md §3）。

规格: ≤531×131 pt ≈ 7.375×1.82 in,印刷尺寸 1:1 铁律(figsize 直接落)，
  PNG 600dpi(≥250dpi 线) + PDF 矢量。diagram-design 密度 4/10:三元素——
  ① 双层框(青=物理冻结 Φ / 橙=语义可修订 Ω,与 fig1 同谱系)
  ② Y→Y′ 演化箭头只穿语义层(框内虚线自环)
  ③ 右侧保留率微条:NTU60 90.6 / NTU120 88.9 (10% labels,各层自身全预算参照)
    + 犬科层如实边界(13% 预算绝对精度近随机,斜纹短条,不标保留率数)。
数据来源: 正文 04-experiments.tex E9/E9b/E7 段(2026-09-07 10-seed 终口径),零硬编码之外的数字。
输出: docs/paper/figures/fig_ga_graphical_abstract.pdf + .png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

C_PHYS_FILL, C_PHYS_EDGE = "#DAFFFF", "#0E7490"
C_SEM_FILL, C_SEM_EDGE = "#FFE3DA", "#C2410C"
C_TEXT, C_ARROW, C_NOTE = "#111111", "#374151", "#888888"
C_BAR_HUMAN = "#0E7490"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "text.color": C_TEXT,
    "mathtext.fontset": "dejavusans",
})

# ---- 画布:印刷尺寸 1:1 ----
fig, ax = plt.subplots(figsize=(7.375, 1.82))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 24.7)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

def box(x, y, w, h, fill, edge):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.0",
                                facecolor=fill, edgecolor=edge, linewidth=1.4))

# ---- ① 双层框 ----
box(2, 4, 29, 16.5, C_PHYS_FILL, C_PHYS_EDGE)
ax.text(16.5, 17.3, r"Physics layer $\Phi$", ha="center", va="center",
        fontsize=9, fontweight="bold", color=C_PHYS_EDGE)
ax.text(16.5, 13.9, "(frozen after pretraining)", ha="center", va="center",
        fontsize=6.3, style="italic", color=C_TEXT)
ax.text(16.5, 8.3, "self-supervised dynamics  ·  behavior proposals",
        ha="center", va="center", fontsize=6.0, color=C_TEXT)

box(40, 4, 29, 16.5, C_SEM_FILL, C_SEM_EDGE)
ax.text(54.5, 17.3, r"Semantic layer $\Omega$", ha="center", va="center",
        fontsize=9, fontweight="bold", color=C_SEM_EDGE)
ax.text(54.5, 13.9, "(revisable)", ha="center", va="center",
        fontsize=6.3, style="italic", color=C_TEXT)

# 接口箭头(物理→语义,只过 embeddings+proposals)
ax.add_patch(FancyArrowPatch((31.6, 12.2), (39.4, 12.2), arrowstyle="-|>",
                             mutation_scale=11, color=C_ARROW, lw=1.5))
ax.text(35.5, 14.6, "embeddings\n+ proposals", ha="center", va="center",
        fontsize=5.5, color=C_ARROW, linespacing=1.25)

# ---- ② 演化箭头:只穿语义层(框内虚线自环) ----
ax.add_patch(FancyArrowPatch((43.5, 8.3), (65.5, 8.3), arrowstyle="-|>",
                             mutation_scale=10, color=C_SEM_EDGE, lw=1.4,
                             linestyle=(0, (4, 2.2))))
ax.text(54.5, 10.6, r"taxonomy evolves $\mathcal{Y}\to\mathcal{Y}'$: only $\Omega$ retrains",
        ha="center", va="center", fontsize=5.8, color=C_SEM_EDGE, fontweight="bold")

# ---- ③ 保留率微条 ----
PANEL_X0, BAR_X0, BAR_MAX = 71.5, 81.5, 17.5   # 100% → 17.5 单位
rows = [
    ("NTU60", 90.6, False),
    ("NTU120", 88.9, False),
    ("canine", 28.9, True),   # E7 v1 @13.9% budget;绝对精度近随机→斜纹条+文字,不标保留率数
]
ax.text((PANEL_X0 + 99) / 2, 21.0, "budget retention", ha="center", va="center",
        fontsize=6.3, fontweight="bold", color=C_TEXT)
ys = (15.6, 10.8, 6.0)
BAR_H = 3.0
for (label, val, is_boundary), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.2, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=5.8, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        ax.add_patch(Rectangle((BAR_X0, y), w, BAR_H, facecolor="#F3F4F6",
                               edgecolor=C_SEM_EDGE, linewidth=0.9, hatch="///"))
        ax.text(BAR_X0 + w + 1.2, y + BAR_H / 2, "near chance @13% labels",
                ha="left", va="center", fontsize=5.2, color=C_SEM_EDGE)
    else:
        ax.add_patch(Rectangle((BAR_X0, y), w, BAR_H, facecolor=C_BAR_HUMAN,
                               edgecolor="none", alpha=0.85))
        ax.text(BAR_X0 + w - 0.8, y + BAR_H / 2, f"{val:.1f}%", ha="right", va="center",
                fontsize=5.6, color="white", fontweight="bold")
ax.text((PANEL_X0 + 99) / 2, 2.2,
        "of own full-budget top-1; canine boundary as measured",
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
