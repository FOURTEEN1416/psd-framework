# -*- coding: utf-8 -*-
"""fig4 v3 — 主动学习效率曲线（2026-09-10 R24 图表专项修正）。

v1 judge 实锤: ①线性 x 轴画 20-200 的 10 倍指数预算域, 20/50/100 间距被非线性压缩,
刻度像素比与线性/对数理论比均不吻合(轴比例失真观感); ②同预算处橙方压青圆;
③"random-guess baseline 4.5%" 文字骑在虚线上。
v2 修正: log x 轴(与 fig5 预算轴惯例一致) + 双系列 ±4% log 域 dodge + 基线文字抬升。
数据管线与配色继承 v1: 读 reports/p05-al-efficiency-{short,warmstart-short}.json, 不硬编码。
输出: fig4_al_efficiency.pdf/png(600dpi)。
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "reports" / "p05-al-efficiency-short-2026-08-24.json"
WDATA = ROOT / "reports" / "p05-al-efficiency-warmstart-short-2026-08-25.json"
OUT_DIR = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
ENT_COLOR = psd_style.HERO     # uncertainty sampling = proposed arm (Nature: blue=hero)
RND_COLOR = psd_style.NS_BLUE  # random selection = NS 蓝（R28: 板内取色, entropy红 vs random蓝）
NOTE_GRAY = "#666666"
GRID_GRAY = psd_style.GRID_GRAY


payload = json.load(open(DATA, encoding="utf-8"))
wpay = json.loads(WDATA.read_text(encoding="utf-8"))
BUDGETS = [20, 50, 100, 200]
DODGE = {0: 1.0, 1: 1.0}   # R24r3: 弃 dodge——位移使点到刻度间距与 log 理论间距失配(被读作轴失真); 重叠改由空心标记消解
SERIES = {
    "entropy": dict(color=ENT_COLOR, marker="o", ls="solid",
                    mfc=ENT_COLOR, label="Uncertainty sampling (softmax entropy)"),
    "random": dict(color=RND_COLOR, marker="s", ls=(0, (5, 2.5)),
                   mfc="none", label="Random selection"),
}
CHANCE_PCT = 4.5

def panel(ax, curves, title, note, note_xy):
    xs = {i: np.array([b * DODGE[i] for b in BUDGETS]) for i in (0, 1)}
    for i, (arm, style) in enumerate(SERIES.items()):
        means = [curves[arm][str(b)]["mean"] * 100.0 for b in BUDGETS]
        stds = [curves[arm][str(b)]["std"] * 100.0 for b in BUDGETS]
        ax.errorbar(xs[i], means, yerr=stds, color=style["color"],
                    marker=style["marker"], markersize=6.2, markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                    markeredgewidth=1.4, linewidth=1.6, linestyle=style["ls"], elinewidth=1.1, capsize=4,
                    capthick=1.1, label=style["label"], zorder=3)
    ax.axhline(CHANCE_PCT, color=NOTE_GRAY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlim(17, 245)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Best validation accuracy (%)", fontsize=7)
    ax.set_title(title, loc="left", fontsize=7)
    ax.grid(True, color=GRID_GRAY, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=6)
    if title.startswith("(a)"):
        ax.text(240, 9.5, "random-guess baseline 4.5%", ha="right", va="center",
                fontsize=6.2, color=NOTE_GRAY,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(note_xy[0], note_xy[1], note, fontsize=6.2, color=NOTE_GRAY,
            style="italic", ha=note_xy[2], va="center", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))

fig, axes = plt.subplots(2, 1, figsize=(3.42, 4.05), sharex=True)
fig.patch.set_facecolor("white")
panel(axes[0], payload["curves"], "(a) cold-start scorers",
      "random exceeds uncertainty\nfor budgets $\\geq$ 100 (cold-start)",
      (65, 27, "left"))
axes[0].legend(loc="upper left", frameon=False, fontsize=6.2)
panel(axes[1], wpay["curves"], "(b) warm-started in-domain scorers",
      "random leads by 4.2\u20135.0 pp for $b \\geq$ 50",
      (238, 14, "right"))
axes[1].set_xlabel("Annotation budget (labeled clips, log scale)", fontsize=7)
axes[1].legend(loc="center left", frameon=False, fontsize=6.2, bbox_to_anchor=(0.03, 0.42))
fig.subplots_adjust(hspace=0.35)

pdf_path = OUT_DIR / "fig4_al_efficiency.pdf"
png_path = OUT_DIR / "fig4_al_efficiency.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.10)
print("written:", pdf_path, "|", png_path)
for arm in SERIES:
    row = ", ".join(f"b={b}: {payload['curves'][arm][str(b)]['mean']*100:.1f}" for b in BUDGETS)
    print(f"(a) {arm:>8}: {row}")
    row = ", ".join(f"b={b}: {wpay['curves'][arm][str(b)]['mean']*100:.1f}" for b in BUDGETS)
    print(f"(b) {arm:>8}: {row}")
neg = all(payload["curves"]["random"][str(b)]["mean"] >= payload["curves"]["entropy"][str(b)]["mean"] for b in (100, 200))
print(f"negative-result fact preserved: {neg}")
