# -*- coding: utf-8 -*-
r"""GA 候选 C1（v14c1）— "全管线流"构图：输入 → Φ → Ω → 保留率。

候选菜单 a（任务包 A 第三轮）：产 2-3 套构图候选交用户挑选，勿擅自定稿。
本候选与 v13 的差异：补输入端（unlabeled streams 卡+波形微图标）与完整左→右
管线叙事，GA 独立阅读时因果链完整（v13 直接从 Φ 开始）。

冻结项：保留率 90.7/88.9/28.9、canine 9.8 vs 11.1% chance @13% budget、
语义色（NTU60 青绿 / NTU120 浅青 / canine 灰底橙描边斜纹）、531×131pt 规格。
输出为候选文件 fig_ga_graphical_abstract_c1.*，不覆盖在位 GA（用户裁决前）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
C_TEXT, C_ARROW = psd_style.INK, "#333333"
C_NOTE = "#666666"
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig, ax = plt.subplots(figsize=(7.375, 1.82))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 100); ax.set_ylim(0, 24.7); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---- 输入端白卡（中性灰描边 + 波形微图标） ----
psd_style.card(ax, 0.8, 5.6, 8.6, 13.6, fill="white", edge="#9E9E9E", lw=0.8,
               rounding=1.6, z=3)
ax.text(5.1, 17.0, "Unlabeled", ha="center", va="center", fontsize=5.6,
        fontweight="bold", color=C_TEXT, zorder=6)
ax.text(5.1, 14.9, "streams", ha="center", va="center", fontsize=5.6,
        fontweight="bold", color=C_TEXT, zorder=6)
ax.text(5.1, 12.2, "dog pose", ha="center", va="center", fontsize=4.9,
        color="#555555", zorder=6)
ax.text(5.1, 10.3, r"$T \times 24 \times 3$", ha="center", va="center",
        fontsize=4.9, color="#555555", zorder=6)
# 波形微图标（三条小幅正弦线, 物理青）
import numpy as np
_xs = np.linspace(2.6, 7.6, 40)
for k, yy in enumerate((8.2, 7.3, 6.4)):
    ax.plot(_xs, yy + 0.42 * np.sin(_xs * 3.1 + k * 1.9), color=psd_style.PHYS_EDGE,
            lw=0.7, zorder=5)
psd_style.flowline(ax, 9.6, 12.4, 11.0, 12.4, color=C_ARROW, lw=1.8, head=11)

# ---- Φ / Ω 双柔色块 ----
psd_style.card(ax, 11.4, 4, 24, 16.5, fill=psd_style.PHYS_TINT, edge=None,
               rounding=2.4, z=3)
ax.text(23.4, 16.9, r"Physics layer $\Phi$", ha="center", va="center",
        fontsize=8.6, fontweight="bold", color=PD, zorder=6)
ax.text(23.4, 13.7, "(frozen after pretraining)", ha="center", va="center",
        fontsize=5.9, style="italic", color="#444444", zorder=6)
ax.text(23.4, 10.6, "self-supervised dynamics", ha="center", va="center",
        fontsize=5.8, color="#333333", zorder=6)
ax.text(23.4, 7.9, "behavior proposals", ha="center", va="center",
        fontsize=5.8, color="#333333", zorder=6)

psd_style.card(ax, 38.4, 4, 24, 16.5, fill=psd_style.SEM_TINT, edge=None,
               rounding=2.4, z=3)
ax.text(50.4, 16.9, r"Semantic layer $\Omega$", ha="center", va="center",
        fontsize=8.6, fontweight="bold", color=SD_, zorder=6)
ax.text(50.4, 13.7, "(revisable)", ha="center", va="center",
        fontsize=5.9, style="italic", color="#444444", zorder=6)
ax.text(50.4, 10.6, r"taxonomy evolves $\mathcal{Y}\to\mathcal{Y}'$",
        ha="center", va="center", fontsize=5.6, color=SD_, fontweight="bold", zorder=6)
ax.text(50.4, 8.6, r"only $\Omega$ retrains", ha="center", va="center",
        fontsize=5.6, color=SD_, fontweight="bold", zorder=6)
ax.text(50.4, 6.1, "anchor-guided pseudo-labeling \u00b7 self-training",
        ha="center", va="center", fontsize=5.2, color="#333333", zorder=6)

psd_style.flowline(ax, 35.7, 12.2, 38.0, 12.2, color=C_ARROW, lw=2.2, head=13)
ax.text(36.85, 14.6, "embeddings\n+ proposals", ha="center", va="center",
        fontsize=4.7, color=C_NOTE, zorder=6, linespacing=1.25)

# ---- 保留率面板（水平条 + 左侧直标） ----
PANEL_X0, BAR_X0, BAR_MAX = 65.0, 72.6, 19.4
rows = [
    ("NTU60", 90.7, False, psd_style.S_NTU60, psd_style.INK),
    ("NTU120", 88.9, False, psd_style.S_NTU120, psd_style.INK),
    ("canine", 28.9, True, psd_style.GRAY_FILL, SD_),
]
ax.text((PANEL_X0 + 99) / 2, 21.0, "budget retention", ha="center", va="center",
        fontsize=6.3, fontweight="bold", color=C_TEXT)
ys = (15.6, 10.8, 6.0)
BAR_H = 3.0
for y in ys:  # 0-100% 基准轨
    psd_style.card(ax, BAR_X0, y, BAR_MAX, BAR_H, fill="#F2F2F2", edge=None,
                   rounding=0.8, z=3)
ax.text(BAR_X0 + BAR_MAX - 0.4, ys[0] + BAR_H + 1.1, "100%", ha="right",
        va="bottom", fontsize=4.8, color=C_NOTE, zorder=6)
for (label, val, is_boundary, bcol, txtcol), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.2, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=5.8, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=SD_, lw=1.0,
                       rounding=0.8, z=4, hatch="///")
        ax.text(BAR_X0 + w + 1.0, y + BAR_H / 2, f"{val:.1f}%",
                ha="left", va="center", fontsize=5.4, color=txtcol, zorder=6)
    else:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=None,
                       rounding=0.8, z=4)
        ax.text(BAR_X0 + w - 0.8, y + BAR_H / 2, f"{val:.1f}%", ha="right",
                va="center", fontsize=5.5, color=txtcol, fontweight="bold", zorder=6)
ax.text((PANEL_X0 + 99) / 2, 3.0, "retention of own full-budget top-1 at 10% labels;",
        ha="center", va="center", fontsize=5.1, color=C_NOTE)
ax.text((PANEL_X0 + 99) / 2, 1.0, "canine: 9.8 vs 11.1% chance at 13% budget",
        ha="center", va="center", fontsize=5.1, color=C_NOTE)

pdf_path = OUT / "fig_ga_graphical_abstract_c1.pdf"
png_path = OUT / "fig_ga_graphical_abstract_c1.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0)
plt.close(fig)

assert pdf_path.is_file() and png_path.is_file(), "GA c1 outputs missing"

# ---- 出图门禁（G1 印刷稳健性 + G4 尺寸/零位图） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "GA-c1", {"NTU60": psd_style.S_NTU60, "NTU120": psd_style.S_NTU120,
              "canine": psd_style.GRAY_FILL},
    redundancy="direct text labels + hatch")
gates.gate_pdf(pdf_path, max_w_pt=531, max_h_pt=131)
print("written:", pdf_path)
print("written:", png_path)
