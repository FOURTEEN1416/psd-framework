# -*- coding: utf-8 -*-
r"""GA v15 — 正典定稿版（用户裁决 2026-09-12 采纳候选 C3 "双行"构图）。

由 v14c3 候选转正（输出改正典文件名）；相对 v13 的差异：
1. 保留率条从右侧窄条（17.5 单位）改为底部全宽横条（~66 单位）——条长
   放大近 4 倍，28.9% vs 90.7% 的比例读数显著更诚实可读；
2. Φ/Ω 上行只承担解耦叙事，双块加宽至 36 单位，行距从容；
3. 接口箭头带 "embeddings + proposals" 单行边标签；演化回线（dashed）
   从 Ω 底绕回 Φ 底，"only Ω retrains" 直标其上。

冻结项：保留率 90.7/88.9/28.9、canine 9.8 vs 11.1% chance @13% budget、
语义色、531×131pt。输出正典 fig_ga_graphical_abstract.pdf/png（覆盖 v13 版）。
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
PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
C_TEXT, C_ARROW = psd_style.INK, "#333333"
C_NOTE = "#666666"
plt.rcParams["mathtext.fontset"] = "dejavusans"

fig, ax = plt.subplots(figsize=(7.375, 1.82))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 100); ax.set_ylim(0, 24.7); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---- 上行：Φ / Ω 宽双块 ----
psd_style.card(ax, 3.0, 12.6, 36, 11.2, fill=psd_style.PHYS_TINT, edge=None,
               rounding=2.2, z=3)
ax.text(21.0, 20.8, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=8.4, fontweight="bold", color=PD, zorder=6)
ax.text(21.0, 17.4, "self-supervised dynamics \u00b7 behavior proposals",
        ha="center", va="center", fontsize=5.8, color="#333333", zorder=6)
ax.text(21.0, 14.6, "trained once, reused across taxonomies",
        ha="center", va="center", fontsize=5.4, style="italic",
        color="#555555", zorder=6)
psd_style.card(ax, 48.0, 12.6, 36, 11.2, fill=psd_style.SEM_TINT, edge=None,
               rounding=2.2, z=3)
ax.text(66.0, 20.8, r"Semantic layer $\Omega$  (revisable)", ha="center",
        va="center", fontsize=8.4, fontweight="bold", color=SD_, zorder=6)
ax.text(66.0, 17.4, "anchor-guided pseudo-labeling \u00b7 self-training",
        ha="center", va="center", fontsize=5.8, color="#333333", zorder=6)
ax.text(66.0, 14.6, r"taxonomy evolves $\mathcal{Y}\to\mathcal{Y}'$",
        ha="center", va="center", fontsize=5.4, style="italic",
        color=SD_, fontweight="bold", zorder=6)

# 接口箭头 + 单行边标签
psd_style.flowline(ax, 39.4, 18.2, 47.6, 18.2, color=C_ARROW, lw=2.2, head=13)
ax.text(43.5, 21.2, "embeddings", ha="center", va="center", fontsize=4.9,
        color=C_NOTE, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(43.5, 19.7, "+ proposals", ha="center", va="center", fontsize=4.9,
        color=C_NOTE, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# 演化回线（Ω 底 → 下行绕 → Φ 底, dashed）+ 直标
ax.plot([66.0, 66.0], [12.4, 11.4], color="#999999", lw=1.0,
        ls=(0, (4, 3)), zorder=2)
ax.plot([66.0, 21.0], [11.4, 11.4], color="#999999", lw=1.0, ls=(0, (4, 3)),
        zorder=2)
ax.add_patch(FancyArrowPatch((21.0, 11.4), (21.0, 12.4), arrowstyle="-|>",
             mutation_scale=8, color="#999999", lw=1.0, linestyle=(0, (4, 3)),
             zorder=2))
ax.text(43.5, 10.2, r"only $\Omega$ retrains", ha="center", va="center",
        fontsize=5.4, color=SD_, fontweight="bold", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# ---- 下行：全宽保留率横条 ----
ax.text(50.0, 8.6, "budget retention \u00b7 own full-budget top-1 kept at "
        "10% labels (canine: 9.8 vs 11.1% chance at 13% budget)",
        ha="center", va="center", fontsize=5.2, color=C_NOTE)
BAR_X0, BAR_MAX, BAR_H = 20.0, 66.0, 2.4
rows = [
    ("NTU60", 90.7, False, psd_style.S_NTU60, psd_style.INK),
    ("NTU120", 88.9, False, psd_style.S_NTU120, psd_style.INK),
    ("canine", 28.9, True, psd_style.GRAY_FILL, SD_),
]
ys = (5.6, 3.0, 0.4)
for y in ys:  # 0-100% 基准轨
    psd_style.card(ax, BAR_X0, y, BAR_MAX, BAR_H, fill="#F2F2F2", edge=None,
                   rounding=0.6, z=3)
ax.text(BAR_X0 + BAR_MAX + 0.6, ys[0] + BAR_H + 0.5, "100%", ha="left",
        va="bottom", fontsize=4.6, color=C_NOTE, zorder=6)
for (label, val, is_boundary, bcol, txtcol), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.4, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=5.6, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=SD_, lw=0.9,
                       rounding=0.6, z=4, hatch="///")
        ax.text(BAR_X0 + w + 1.0, y + BAR_H / 2, f"{val:.1f}%", ha="left",
                va="center", fontsize=5.2, color=txtcol, zorder=6)
    else:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=None,
                       rounding=0.6, z=4)
        ax.text(BAR_X0 + w - 0.7, y + BAR_H / 2, f"{val:.1f}%", ha="right",
                va="center", fontsize=5.2, color=txtcol, fontweight="bold",
                zorder=6)

pdf_path = OUT / "fig_ga_graphical_abstract.pdf"
png_path = OUT / "fig_ga_graphical_abstract.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0)
plt.close(fig)

assert pdf_path.is_file() and png_path.is_file(), "GA v15 outputs missing"

import make_common_gates as gates
gates.gate_print_robustness(
    "GA-v15", {"NTU60": psd_style.S_NTU60, "NTU120": psd_style.S_NTU120,
              "canine": psd_style.GRAY_FILL},
    redundancy="direct text labels + hatch")
gates.gate_pdf(pdf_path, max_w_pt=531, max_h_pt=131)
print("written:", pdf_path)
print("written:", png_path)
