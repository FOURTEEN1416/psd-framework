# -*- coding: utf-8 -*-
r"""GA v17 — 按 Elsevier 官方规格重制。

**为什么重做**：Guide for Authors 逐字要求
    "Ensure the image is 531 x 1328 pixels (h x w) or proportionally more,
     and is readable at a size of 5 x 13 cm."
→ 宽度:高度 = 1328:531 = 2.50 : 1。

v16 的画布是 7.375 x 1.82 in（宽高比 4.05:1），既不符合比例、像素密度也不足。
v17 改为 **1328 x 531 px @ 300 dpi = 4.4267 x 1.77 in = 318.7 x 127.4 pt**（比例 2.50:1），
并重新排布版式（横向空间变窄，纵向变高）。

**内容口径**：只用正文中确有的数字，避免 graphical abstract 与论文口径不一致：
    - 82.0% top-1 from a 20-clip marginal budget（synthetic-offset tier）
    - 90.7% of full-budget accuracy at 10% of the labels（NTU60）
    - >=3x lower transition cost at matched accuracy
（v16 曾标 "canine 28.9%"，那是 9.8/33.93 的派生值，正文并无此数字，故本版不采用。）
"""
from pathlib import Path
import sys, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PT_, ST_ = psd_style.PHYS_TINT, psd_style.SEM_TINT
INK = psd_style.INK
NOTE = "#666666"
ARROW = "#333333"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# 1328 x 531 px @ 300 dpi
W_IN, H_IN = 1328.0 / 300.0, 531.0 / 300.0
fig, ax = plt.subplots(figsize=(W_IN, H_IN))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---------------- 标题 ----------------
ax.text(1.2, 95.5, r"PSD: physics$-$semantics decoupled framework for animal behavior recognition",
        ha="left", va="center", fontsize=8.2, fontweight="bold", color=INK)
ax.text(1.2, 88.6, r"under evolving evaluation criteria $-$ a taxonomy change retrains the semantic layer alone",
        ha="left", va="center", fontsize=6.2, color=NOTE)

# ---------------- 双块：Φ (frozen) / Ω (revisable) ----------------
psd_style.card(ax, 2.0, 52.0, 43.0, 31.0, fill=PT_, edge=None, rounding=2.4, z=3)
ax.text(23.5, 77.5, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=8.0, fontweight="bold", color=PD, zorder=6)
ax.text(23.5, 69.5, "self-supervised dynamics", ha="center", va="center",
        fontsize=6.2, color="#333333", zorder=6)
ax.text(23.5, 63.5, "behavior proposals", ha="center", va="center",
        fontsize=6.2, color="#333333", zorder=6)
ax.text(23.5, 56.5, "trained once per species family", ha="center", va="center",
        fontsize=5.8, style="italic", color="#555555", zorder=6)

psd_style.card(ax, 55.0, 52.0, 43.0, 31.0, fill=ST_, edge=None, rounding=2.4, z=3)
ax.text(76.5, 77.5, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
        fontsize=8.0, fontweight="bold", color=SD_, zorder=6)
ax.text(76.5, 69.5, "anchor-guided pseudo-labeling", ha="center", va="center",
        fontsize=6.2, color="#333333", zorder=6)
ax.text(76.5, 63.5, "iterated self-training", ha="center", va="center",
        fontsize=6.2, color="#333333", zorder=6)
ax.text(76.5, 56.5, r"grows rule-engine seeds to full taxonomy", ha="center", va="center",
        fontsize=5.8, style="italic", color="#555555", zorder=6)

# ---------------- 接口箭头 ----------------
psd_style.flowline(ax, 45.6, 67.5, 54.4, 67.5, color=ARROW, lw=1.2, head=11)
ax.text(50.0, 74.0, "embeddings", ha="center", va="center", fontsize=5.8, color=NOTE, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(50.0, 61.0, "+ proposals", ha="center", va="center", fontsize=5.8, color=NOTE, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# 演化回线：Ω 底 -> 横折 -> Φ 底，Φ 端以 × 终止（明示 Φ 不被重训）
ax.plot([76.5, 76.5], [52.0, 46.5], color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=2)
ax.plot([76.5, 23.5], [46.5, 46.5], color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=2)
ax.plot([23.5, 23.5], [46.5, 50.2], color="#999999", lw=0.7, ls=(0, (4, 3)), zorder=2)
ax.text(23.5, 50.2, r"$\times$", ha="center", va="center", fontsize=7.0, color="#999999", zorder=3)
ax.add_patch(FancyArrowPatch((76.5, 52.0), (76.5, 48.4), arrowstyle="-|>",
                             mutation_scale=9, color=ARROW, lw=1.0, zorder=2))
ax.text(50.0, 43.0, r"only $\Omega$ retrains", ha="center", va="center",
        fontsize=6.4, color=SD_, fontweight="bold", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# ---------------- 三个关键结果 ----------------
res = [
    (16.5, "82.0%", "top-1 from a 20-clip budget", "under distribution shift"),
    (50.0, "90.7%", "of full-budget accuracy", "at 10% of the labels"),
    (83.5, r"$\geq$3$\times$", "lower transition cost", "at matched accuracy"),
]
for x, big, l1, l2 in res:
    psd_style.card(ax, x - 15.0, 3.0, 30.0, 32.0, fill="#F2F2F2", edge=None, rounding=2.0, z=3)
    ax.text(x, 27.0, big, ha="center", va="center", fontsize=13.0, fontweight="bold",
            color=SD_, zorder=6)
    ax.text(x, 16.5, l1, ha="center", va="center", fontsize=6.2, color=INK, zorder=6)
    ax.text(x, 10.5, l2, ha="center", va="center", fontsize=5.8, color=NOTE, zorder=6)

# ---------------- 闸门 ----------------
out_pdf = OUT / "fig_ga_graphical_abstract.pdf"
out_png = OUT / "fig_ga_graphical_abstract.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=300)

print("saved:", out_pdf)
print("figsize (in): %.4f x %.4f  -> %.1f x %.1f pt" % (W_IN, H_IN, W_IN * 72, H_IN * 72))
print("aspect w:h = %.3f  (target 2.50)" % (W_IN / H_IN))
print("pixels @300dpi = %d x %d  (target 1328 x 531)" % (W_IN * 300, H_IN * 300))
plt.close(fig)
