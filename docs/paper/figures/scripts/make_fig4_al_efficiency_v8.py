# -*- coding: utf-8 -*-
"""fig4 v8 — 主动学习效率曲线（正典 · 风格方向 **C+ 强化版**）。

承 v7 全部构图（双面板 sharex、图级单行共享图例、面板内负结果注记、
机会线 4.5%、0-100 诚实轴、两系列 marker/线型冗余编码），**图内全部文字与
数据口径逐字承 v6，零新增零删除零改写**；负结果事实（cold-start 下
random ≥ entropy @ b=100/200）原样保留并打印断言。

━━ 尺度修复 + 高度上限合规（wt/figB 诊断）━━
v6 设计画布即 3.42x3.66in，但保存用 `bbox_inches="tight"` → MediaBox=墨迹框
227.28x260.92pt，被 LaTeX 按 \\linewidth 装入时放大 246.24/227.28 = **1.083 倍**，
实排高度 3.93in —— **顶破任务书「fig4 高 ≤3.66in 防浮动超页」上限 7.3%**，
这是 tight 拉伸在本图造成的直接规格违规（D4 裁决项据此判为"构图职能内修复"）。
本版固定画布 3.42x3.66in，MediaBox 严格 246.24x263.52pt=3.66in=版心宽，
装入系数 1.000，实排高度精确 3.66in（= 设计高度，合规上限内）。
显式版面边距替代 tight 兜底（bottom/top/left/right 全定量）。

C+ 相对 v7 的收紧项（本图高度受「fig4 高 ≤3.66in 防浮动超页」硬上限约束，
**不可加高**，故 C+ 的留白增益改走"面板间距 + 内部净空"而非画布尺寸）：
1. **笔触更细**：spines 0.65 -> 0.4；tick width 0.65 -> 0.4、length 2.4 -> 1.8；
   网格 0.45 -> 0.35；曲线 1.4 -> 1.0；marker 5.6 -> 5.2、edge 1.0 -> 0.7；
   误差棒 elinewidth 0.9 -> 0.6、capsize 3 -> 2.2、capthick 0.9 -> 0.6；
   机会线 0.9 -> 0.6。
2. **字号阶梯更大**：轴名 7 -> 7.2；面板标题 7 -> 7.5；刻度 6 -> 6.2；
   图例 6.0 -> 6.2；面板内注记 6.2（保持，受 G3 误差棒净空约束）。下限 5.5pt。
3. **留白**：hspace 0.32 -> 0.40（两面板更分得开）、bottom 0.085 -> 0.095、
   top 0.90 -> 0.905——在 3.66in 硬上限内做内部呼吸。
4. 负结果事实与 caption 文字保持逐字不变（见文末断言）。

数据：reports/p05-al-efficiency-{short-2026-08-24, warmstart-short-2026-08-25}.json（逐位承 v6/v7）。
caption 约束逐词核对（05-ablation.tex \\label{fig:al-efficiency}）：
panel (a) cold-start / panel (b) warm-started 标题、两系列图例、
"random leads by 4.2--5.0 pp"、机会线注记——全部保持。
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
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)

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

PANEL_ERRS = {"a": [], "b": []}   # (x, y_lo, y_hi, arm) 供 G3

def panel(ax, curves, title, note, note_xy, tag):
    xs = np.array([float(b) for b in BUDGETS])
    for arm, style in SERIES.items():
        means = [curves[arm][str(b)]["mean"] * 100.0 for b in BUDGETS]
        stds = [curves[arm][str(b)]["std"] * 100.0 for b in BUDGETS]
        ax.errorbar(xs, means, yerr=stds, color=style["color"],
                    marker=style["marker"], markersize=5.2, markerfacecolor=style["mfc"],
                    markeredgecolor=style["color"], markeredgewidth=0.7, linewidth=1.0,
                    linestyle=style["ls"], elinewidth=0.6, capsize=2.2,
                    capthick=0.6, zorder=3)
        for b, m, s in zip(xs, means, stds):
            PANEL_ERRS[tag].append((b, m - s, m + s, arm))
    ax.axhline(CHANCE_PCT, color=NOTE_GRAY, linewidth=0.6, linestyle=(0, (4, 3)), zorder=2)
    ax.set_xscale("log")
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.minorticks_off()
    ax.set_xlim(17, 245)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Best validation accuracy (%)", fontsize=7.2)
    ax.set_title(title, loc="left", fontsize=7.5)
    ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.35, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.4)
    ax.tick_params(labelsize=6.2, width=0.4, length=1.8)
    texts = []
    if title.startswith("(a)"):
        texts.append(ax.text(240, 9.5, "random-guess baseline 4.5%", ha="right", va="center",
                             fontsize=6.2, color=NOTE_GRAY,
                             bbox=dict(facecolor="white", edgecolor="none", pad=1.0)))
    texts.append(ax.text(note_xy[0], note_xy[1], note, fontsize=6.2, color=NOTE_GRAY,
                         style="italic", ha=note_xy[2], va="center", zorder=5,
                         bbox=dict(facecolor="white", edgecolor="none", alpha=1.0, pad=1.0)))
    return texts


# ---- 固定画布：宽 3.42in == 版心 246.24pt；高 3.66in == 规格上限 ----
FIG_W, FIG_H = 3.42, 3.66
fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H), sharex=True)
fig.patch.set_facecolor("white")
texts_a = panel(axes[0], payload["curves"], "(a) cold-start scorers",
                "random exceeds uncertainty\nfor budgets $\\geq$ 100 (cold-start)",
                (65, 27, "left"), "a")
texts_b = panel(axes[1], wpay["curves"], "(b) warm-started in-domain scorers",
                "random leads by 4.2\u20135.0 pp for $b \\geq$ 50",
                (238, 14, "right"), "b")
axes[1].set_xlabel("Annotation budget (labeled clips, log scale)", fontsize=7.2)

# 图级单行共享图例（面板 (a) 上方图外）——承 v5/v6
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=st["color"], marker=st["marker"], mfc=st["mfc"],
                  mec=st["color"], markeredgewidth=0.7, markersize=5.2,
                  linestyle=st["ls"], linewidth=1.0)
           for st in SERIES.values()]
labels = [st["label"] for st in SERIES.values()]
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
           ncol=2, frameon=False, fontsize=6.2, handlelength=2.6,
           columnspacing=1.2, borderaxespad=0.0)
# 去 tight 后的显式版面（legend 在上、xlabel 在下均留位）
fig.subplots_adjust(left=0.125, right=0.985, top=0.905, bottom=0.095, hspace=0.40)

# ---- 门禁（G1 + G3 双面板 + G5 + G6 + G4 高度上限） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig4-v8", {"uncertainty (proposed)": ENT_COLOR, "random": RND_COLOR},
    redundancy="marker shape (o/s) + linestyle (solid/dashed) + legend")
gates.gate_whisker_texts(fig, axes[0], texts_a, PANEL_ERRS["a"], name="fig4-v8-a")
gates.gate_whisker_texts(fig, axes[1], texts_b, PANEL_ERRS["b"], name="fig4-v8-b")

# ---- 出图（尺度修复：无 tight，MediaBox 严格等于画布） ----
pdf_path = OUT_DIR / "fig4_al_efficiency.pdf"
png_path = OUT_DIR / "fig4_al_efficiency.png"
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(PREVIEW / "preview_fig4.png", dpi=150)

gates.gate_typography(fig, min_pt=5.5, name="fig4-v8")
gates.gate_marks_in_axes(fig, name="fig4-v8")
gates.gate_pdf(pdf_path, max_w_pt=246.3, max_h_pt=263.6)

print("written:", pdf_path, "|", png_path)
for arm in SERIES:
    row = ", ".join(f"b={b}: {payload['curves'][arm][str(b)]['mean']*100:.1f}" for b in BUDGETS)
    print(f"(a) {arm:>8}: {row}")
    row = ", ".join(f"b={b}: {wpay['curves'][arm][str(b)]['mean']*100:.1f}" for b in BUDGETS)
    print(f"(b) {arm:>8}: {row}")
neg = all(payload["curves"]["random"][str(b)]["mean"] >= payload["curves"]["entropy"][str(b)]["mean"] for b in (100, 200))
print(f"negative-result fact preserved: {neg}")
assert neg, "负结果事实丢失：cold-start random >= entropy @ b>=100 不再成立"
print("fig4 v8 (direction C+ intensified, height 3.66in compliant) saved")
