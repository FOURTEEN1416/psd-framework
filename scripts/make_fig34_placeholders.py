# -*- coding: utf-8 -*-
"""fig3 / fig4 占位图生成脚本。

规格来源: docs/paper/figure-specs.md §fig3/fig4
  - fig3（SMQ 分割边界 vs GT 定性对比）与 fig4（主动学习效率曲线）
    均依赖 P0.2 / P0.5 实验数据，本窗口只产出占位 PDF。
  - 正式规格待数据形态确认后由后续窗口在 figure-specs.md 补齐并重绘。
输出: docs/paper/figures/fig3_placeholder.pdf / fig4_placeholder.pdf
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INK = "#000000"
NOTE_GRAY = "#888888"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
})

PLACEHOLDERS = [
    ("fig3_placeholder.pdf",
     "Figure 3 (placeholder)",
     "Qualitative comparison: SMQ segmentation boundaries vs. ground truth\n"
     "(with zoom-in panels)",
     "Awaiting P0.2 experiment data — spec to be finalized in figure-specs.md §fig3"),
    ("fig4_placeholder.pdf",
     "Figure 4 (placeholder)",
     "Active-learning efficiency curve: annotation budget (x) vs. accuracy (y),\n"
     "uncertainty sampling vs. random, target line y = 85%",
     "Awaiting P0.5 experiment data — spec to be finalized in figure-specs.md §fig4"),
]

for fname, title, desc, note in PLACEHOLDERS:
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.add_patch(plt.Rectangle((2, 2), 96, 96, fill=False,
                               edgecolor=NOTE_GRAY, linewidth=1.0, linestyle=(0, (4, 3))))
    ax.text(50, 66, title, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(50, 48, desc, ha="center", va="center", fontsize=10.5, color=INK)
    ax.text(50, 30, note, ha="center", va="center", fontsize=9.5, style="italic", color=NOTE_GRAY)
    ax.text(98, -1.5, f"FIGURE_SOURCE: scripts/make_fig34_placeholders.py · PSD-Framework · W17 · 2026-08-24",
            ha="right", va="top", fontsize=6, color=NOTE_GRAY)

    fig.savefig(OUT_DIR / fname, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print("written:", OUT_DIR / fname)
