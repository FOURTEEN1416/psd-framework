# -*- coding: utf-8 -*-
"""fig4 v6 — 主动学习效率曲线（R43b 顶刊级冲刺 · scientific-visualization 笔触精修）。

相对 v5 的顶刊化精修（拓扑/数据/图例位置承 v5 不动）：
1. 网格改 y-only 且减淡（0.8→0.45）——去"matplotlib 默认全网格"观感（数据墨水比）；
2. 笔触系统性调细一档：轴线 0.8→0.65、marker 6.2→5.6、marker 描边 1.4→1.0、
   连线 1.6→1.4、误差棒 1.1→0.9/caps 4→3/capthick 1.1→0.9、机会线 1.2→0.9、
   ticks 0.65/2.4（Nature 系细线惯例）；
3. 图级单行共享图例、负结果注记、0-100 诚实轴承 v5。

数据：reports/p05-al-efficiency-{short-2026-08-24, warmstart-short-2026-08-25}.json。
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
ENT_COLOR = psd_style.HERO     # uncertainty sampling = proposed arm（深橙=HERO 臂）
RND_COLOR = psd_style.NS_BLUE  # random selection = 深青（冷青=random 臂）
NOTE_GRAY = "#666666"
GRID_GRAY = psd_style.GRID_GRAY


payload = json.load(open(DATA, encoding="utf-8"))
wpay = json.loads(WDATA.read_text(encoding="utf-8"))
BUDGETS = [20, 50, 100, 200]
SERIES = {
    "entropy": dict(color=ENT_COLOR, marker="o", ls="solid",
                    mfc=ENT_COLOR, label="Uncertainty sampling (softmax entropy)"),
    "random": dict(color=RND_COLOR, marker="s", ls=(0, (5, 2.5)),
                   mfc="none", label="Random selection"),
}
CHANCE_PCT = 4.5

def panel(ax, curves, title, note, note_xy):
    xs = {i: np.array([float(b) for b in BUDGETS]) for i in (0, 1)}
    for i, (arm, style) in enumerate(SERIES.items()):
        means = [curves[arm][str(b)]["mean"] * 100.0 for b in BUDGETS]
        stds = [curves[arm][str(b)]["std"] * 100.0 for b in BUDGETS]
        ax.errorbar(xs[i], means, yerr=stds, color=style["color"],
                    marker=style["marker"], markersize=5.6, markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                    markeredgewidth=1.0, linewidth=1.4, linestyle=style["ls"], elinewidth=0.9, capsize=3,
                    capthick=0.9, zorder=3)
    ax.axhline(CHANCE_PCT, color=NOTE_GRAY, linewidth=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlim(17, 245)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Best validation accuracy (%)", fontsize=7)
    ax.set_title(title, loc="left", fontsize=7)
    ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.65)
    ax.tick_params(labelsize=6, width=0.65, length=2.4)
    if title.startswith("(a)"):
        ax.text(240, 9.5, "random-guess baseline 4.5%", ha="right", va="center",
                fontsize=6.2, color=NOTE_GRAY,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax.text(note_xy[0], note_xy[1], note, fontsize=6.2, color=NOTE_GRAY,
            style="italic", ha=note_xy[2], va="center", zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0))

fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.66), sharex=True)
fig.patch.set_facecolor("white")
panel(axes[0], payload["curves"], "(a) cold-start scorers",
      "random exceeds uncertainty\nfor budgets $\\geq$ 100 (cold-start)",
      (65, 27, "left"))
panel(axes[1], wpay["curves"], "(b) warm-started in-domain scorers",
      "random leads by 4.2\u20135.0 pp for $b \\geq$ 50",
      (238, 14, "right"))
axes[1].set_xlabel("Annotation budget (labeled clips, log scale)", fontsize=7)

# 图级单行共享图例（面板 (a) 上方图外）——消 P17 遮挡 + 去双面板冗余（承 v5）
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=st["color"], marker=st["marker"], mfc=st["mfc"],
                  mec=st["color"], markeredgewidth=1.0, markersize=5.6,
                  linestyle=st["ls"], linewidth=1.4)
           for st in SERIES.values()]
labels = [st["label"] for st in SERIES.values()]
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
           ncol=2, frameon=False, fontsize=6.0, handlelength=2.6,
           columnspacing=1.2, borderaxespad=0.0)
fig.subplots_adjust(hspace=0.32, top=0.90)

pdf_path = OUT_DIR / "fig4_al_efficiency.pdf"
png_path = OUT_DIR / "fig4_al_efficiency.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
print("written:", pdf_path, "|", png_path)
for arm in SERIES:
    row = ", ".join(f"b={b}: {payload['curves'][arm][str(b)]['mean']*100:.1f}" for b in BUDGETS)
    print(f"(a) {arm:>8}: {row}")
    row = ", ".join(f"b={b}: {wpay['curves'][arm][str(b)]['mean']*100:.1f}" for b in BUDGETS)
    print(f"(b) {arm:>8}: {row}")
neg = all(payload["curves"]["random"][str(b)]["mean"] >= payload["curves"]["entropy"][str(b)]["mean"] for b in (100, 200))
print(f"negative-result fact preserved: {neg}")
