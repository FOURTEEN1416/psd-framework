# -*- coding: utf-8 -*-
"""fig4 v5 — 主动学习效率曲线（R43 全图重绘 · 数模竞赛·科研工具箱/paper-figure
P17 + composition-patterns 模式二；数据管线与配色承 v4，零硬编码）。

相对 v4 的重绘点：
1. 图例提升为图级单行共享（fig.legend, ncol=2, 面板 (a) 上方图外）——两面板系列
   完全相同，v4 双面板各挂一份图例纯属重复，且 (a) 图例压住 b=20 数据点
   （十八坑 P17 实锤违例），此处一次根治；
2. 面板内不再挂图例，注释文字位置微调避让；
3. 画布高 4.05→3.85in（图例区外置后回收）。

数据：reports/p05-al-efficiency-short-2026-08-24.json（冷启动）+
p05-al-efficiency-warmstart-short-2026-08-25.json（热启动），不硬编码。
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
                    marker=style["marker"], markersize=6.2, markerfacecolor=style["mfc"], markeredgecolor=style["color"],
                    markeredgewidth=1.4, linewidth=1.6, linestyle=style["ls"], elinewidth=1.1, capsize=4,
                    capthick=1.1, zorder=3)
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

fig, axes = plt.subplots(2, 1, figsize=(3.42, 3.66), sharex=True)
fig.patch.set_facecolor("white")
panel(axes[0], payload["curves"], "(a) cold-start scorers",
      "random exceeds uncertainty\nfor budgets $\\geq$ 100 (cold-start)",
      (65, 27, "left"))
panel(axes[1], wpay["curves"], "(b) warm-started in-domain scorers",
      "random leads by 4.2\u20135.0 pp for $b \\geq$ 50",
      (238, 14, "right"))
axes[1].set_xlabel("Annotation budget (labeled clips, log scale)", fontsize=7)

# 图级单行共享图例（面板 (a) 上方图外）——消 P17 遮挡 + 去双面板冗余
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=st["color"], marker=st["marker"], mfc=st["mfc"],
                  mec=st["color"], markeredgewidth=1.4, markersize=6.2,
                  linestyle=st["ls"], linewidth=1.6)
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
