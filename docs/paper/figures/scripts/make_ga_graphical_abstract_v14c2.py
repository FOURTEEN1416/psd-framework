# -*- coding: utf-8 -*-
r"""GA 候选 C2（v14c2）— "三幕叙事"构图：PROBLEM → APPROACH → RESULT。

候选菜单 a（任务包 A 第三轮）。与 v13/C1 的差异：GA 按顶刊图形摘要常见的
"问题-方法-结果"三幕弧组织，每幕配 mono 小眉题（eyebrow, tracked 大写），
方法幕只保留 Φ/Ω 最小对（更窄的双块），结果幕保留率条独占右 1/3。

冻结项：保留率 90.7/88.9/28.9、canine 9.8 vs 11.1% chance @13% budget、
语义色、531×131pt。输出候选文件 *_c2.*，不覆盖在位 GA。
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

def eyebrow(x, y, text):
    ax.text(x, y, text, fontsize=4.8, color="#8A8A8A", ha="left", va="center",
            fontweight="bold", zorder=6)
    ax.plot([x, x + 6.0], [y + 1.5, y + 1.5], color="#D8D8D8", lw=0.6, zorder=2)

# ---- 幕一 PROBLEM（文字为主, 不加第三深色块保 focal 纪律） ----
eyebrow(2.0, 21.4, "PROBLEM")
ax.text(2.0, 17.2, "K9 action data:", ha="left", va="center", fontsize=7.2,
        fontweight="bold", color=C_TEXT, zorder=6)
ax.text(2.0, 14.2, "labels are scarce and costly", ha="left", va="center",
        fontsize=6.4, color=C_TEXT, zorder=6)
ax.text(2.0, 10.6, "annotation is the bottleneck:", ha="left", va="center",
        fontsize=5.6, color=C_NOTE, zorder=6)
ax.text(2.0, 8.2, "full supervision does not", ha="left", va="center",
        fontsize=5.6, color=C_NOTE, zorder=6)
ax.text(2.0, 5.8, "transfer across domains", ha="left", va="center",
        fontsize=5.6, color=C_NOTE, zorder=6)
# 竖分隔发丝线
ax.plot([26.5, 26.5], [3.6, 21.4], color="#E2E2E2", lw=0.7, zorder=2)

# ---- 幕二 APPROACH：Φ/Ω 窄双块 + 接口箭头 ----
eyebrow(29.0, 21.4, "APPROACH")
psd_style.card(ax, 29.0, 4.4, 15.5, 14.6, fill=psd_style.PHYS_TINT, edge=None,
               rounding=2.0, z=3)
ax.text(36.75, 16.3, r"Physics $\Phi$", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=PD, zorder=6)
ax.text(36.75, 13.3, "frozen", ha="center", va="center", fontsize=5.4,
        style="italic", color="#444444", zorder=6)
ax.text(36.75, 10.2, "self-supervised", ha="center", va="center", fontsize=5.4,
        color="#333333", zorder=6)
ax.text(36.75, 7.8, "dynamics +", ha="center", va="center", fontsize=5.4,
        color="#333333", zorder=6)
ax.text(36.75, 5.6, "proposals", ha="center", va="center", fontsize=5.4,
        color="#333333", zorder=6)
psd_style.card(ax, 46.5, 4.4, 15.5, 14.6, fill=psd_style.SEM_TINT, edge=None,
               rounding=2.0, z=3)
ax.text(54.25, 16.3, r"Semantic $\Omega$", ha="center", va="center", fontsize=7.0,
        fontweight="bold", color=SD_, zorder=6)
ax.text(54.25, 13.3, "revisable", ha="center", va="center", fontsize=5.4,
        style="italic", color="#444444", zorder=6)
ax.text(54.25, 10.2, r"$\mathcal{Y}\to\mathcal{Y}'$:", ha="center", va="center",
        fontsize=5.4, color=SD_, fontweight="bold", zorder=6)
ax.text(54.25, 7.8, r"only $\Omega$ retrains", ha="center", va="center",
        fontsize=5.4, color=SD_, fontweight="bold", zorder=6)
ax.text(54.25, 5.6, "anchor pseudo-labels", ha="center", va="center",
        fontsize=5.4, color="#333333", zorder=6)
psd_style.flowline(ax, 44.7, 11.7, 46.1, 11.7, color=C_ARROW, lw=1.9, head=12)

# ---- 幕三 RESULT：保留率 ----
eyebrow(66.0, 21.4, "RESULT")
ax.text(82.5, 17.6, "budget retention", ha="center", va="center", fontsize=6.2,
        fontweight="bold", color=C_TEXT)
BAR_X0, BAR_MAX, BAR_H = 72.6, 19.4, 3.0
rows = [
    ("NTU60", 90.7, False, psd_style.S_NTU60, psd_style.INK),
    ("NTU120", 88.9, False, psd_style.S_NTU120, psd_style.INK),
    ("canine", 28.9, True, psd_style.GRAY_FILL, SD_),
]
ys = (12.6, 8.4, 4.2)
for y in ys:
    psd_style.card(ax, BAR_X0, y, BAR_MAX, BAR_H, fill="#F2F2F2", edge=None,
                   rounding=0.8, z=3)
ax.text(BAR_X0 + BAR_MAX - 0.4, ys[0] + BAR_H + 1.0, "100%", ha="right",
        va="bottom", fontsize=4.8, color=C_NOTE, zorder=6)
for (label, val, is_boundary, bcol, txtcol), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.2, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=5.6, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=SD_, lw=1.0,
                       rounding=0.8, z=4, hatch="///")
        ax.text(BAR_X0 + w + 1.0, y + BAR_H / 2, f"{val:.1f}%", ha="left",
                va="center", fontsize=5.2, color=txtcol, zorder=6)
    else:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=None,
                       rounding=0.8, z=4)
        ax.text(BAR_X0 + w - 0.8, y + BAR_H / 2, f"{val:.1f}%", ha="right",
                va="center", fontsize=5.4, color=txtcol, fontweight="bold", zorder=6)
ax.text(82.5, 2.4, "own full-budget top-1 kept at 10% labels;",
        ha="center", va="center", fontsize=5.0, color=C_NOTE)
ax.text(82.5, 0.9, "canine: 9.8 vs 11.1% chance at 13% budget",
        ha="center", va="center", fontsize=5.0, color=C_NOTE)

pdf_path = OUT / "fig_ga_graphical_abstract_c2.pdf"
png_path = OUT / "fig_ga_graphical_abstract_c2.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0)
plt.close(fig)

assert pdf_path.is_file() and png_path.is_file(), "GA c2 outputs missing"

import make_common_gates as gates
gates.gate_print_robustness(
    "GA-c2", {"NTU60": psd_style.S_NTU60, "NTU120": psd_style.S_NTU120,
              "canine": psd_style.GRAY_FILL},
    redundancy="direct text labels + hatch")
gates.gate_pdf(pdf_path, max_w_pt=531, max_h_pt=131)
print("written:", pdf_path)
print("written:", png_path)
