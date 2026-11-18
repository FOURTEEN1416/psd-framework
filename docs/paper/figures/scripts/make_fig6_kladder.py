# -*- coding: utf-8 -*-
"""fig6 — Reinforcement E3b 标注预算阶梯（k-ladder）折线图。

数据真源: runs/p1_sprint/k_ladder/summary.json（canonical 合并版, 2026-11-16）。
只读 ladder[].arms[*].top1_overall / top1_overall_std（种子 42-44 均值±s.d., ddof=0）,
×100 转百分比; 数值与正文 tab:kladder / E3b 段逐位同源, 不新增统计量。

风格对齐 fig5_budget_retention v9（方向 C+ 正典）:
  - psd_style 统一样式; 固定画布宽 3.42in = 版心 246.24pt（不用 bbox_inches="tight",
    MediaBox == 画布, 装入零缩放）; top/right spine 关闭; 极浅 y 网格; 字号下限 5.5pt。
  - 三系列色取 psd_style 语义 token: decoupled rebuild = proposed 臂 = OCEAN_EMBER;
    coupled / frozen head = k 无关参照臂 = 中性灰 / 青绿（HUMAN 系）。
  - ±2.3pp parity band 画在 coupled 水平参照线周围（k* 判定口径:
    acc(coupled)-acc(decoupled) <= 2.3pp, summary.json config.parity_band_pp）。
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[4]  # worktree 隔离: scripts->figures->paper->docs->repo
OUT = ROOT / "docs" / "paper" / "figures"
SRC = ROOT / "runs" / "p1_sprint" / "k_ladder" / "summary.json"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK

C_DECOUPLED = psd_style.SEM_EDGE    # #EC8E5A 深橙 = proposed 语义重建臂
C_COUPLED = psd_style.GRAY_2        # #767676 中性灰 = k 无关全重训参照
C_FROZEN = psd_style.HUMAN_EDGE     # #53999D 青绿 = k 无关冻结头参照
BAND_FILL = "#FAE3D6"               # parity 带极浅橙（= psd_style.SEM_TINT 同源）


def _darken(hex_color, f):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(int(r * f), int(g * f), int(b * f))


data = json.loads(SRC.read_text(encoding="utf-8"))
ladder = {int(run["k"]): run["arms"] for run in data["ladder"]}
ks = sorted(ladder)                       # [10, 25, 50, 100]
assert ks == [10, 25, 50, 100], ks

series = {}
for arm in ("decoupled", "coupled", "frozen_head"):
    series[arm] = ([ladder[k][arm]["top1_overall"] * 100.0 for k in ks],
                   [ladder[k][arm]["top1_overall_std"] * 100.0 for k in ks])

dec_y, dec_e = series["decoupled"]
cpl_y, cpl_e = series["coupled"]
frz_y, frz_e = series["frozen_head"]
parity = float(data["config"]["parity_band_pp"])   # 2.3

# ---- 固定画布: 宽 3.42in == 版心 246.24pt; 高 2.4in ----
FIG_W, FIG_H = 3.42, 2.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
fig.subplots_adjust(left=0.112, right=0.985, bottom=0.148, top=0.98)

# ---- ±2.3pp parity band（围绕 coupled 水平参照; 判定口径见 summary.json k_star_analysis）----
band_lo, band_hi = cpl_y[0] - parity, cpl_y[0] + parity
ax.axhspan(band_lo, band_hi, color=BAND_FILL, linewidth=0, zorder=0.6)

# ---- 三条线（先画参照水平线, 再画主臂橙线置顶）----
ax.errorbar(ks, cpl_y, yerr=cpl_e, color=C_COUPLED, marker="s", markersize=3.6,
            ls="--", linewidth=0.9, markerfacecolor=C_COUPLED, markeredgecolor="white",
            markeredgewidth=0.5, elinewidth=0.6, capsize=2.0, capthick=0.6, zorder=2.5)
ax.errorbar(ks, frz_y, yerr=frz_e, color=C_FROZEN, marker="^", markersize=4.0,
            ls=":", linewidth=1.0, markerfacecolor=C_FROZEN, markeredgecolor="white",
            markeredgewidth=0.5, elinewidth=0.6, capsize=2.0, capthick=0.6, zorder=2.6)
ax.errorbar(ks, dec_y, yerr=dec_e, color=C_DECOUPLED, marker="o", markersize=3.8,
            ls="-", linewidth=1.1, markerfacecolor=C_DECOUPLED, markeredgecolor="white",
            markeredgewidth=0.5, elinewidth=0.6, capsize=2.0, capthick=0.6, zorder=3)

# ---- 轴 / 刻度 ----
ax.set_xscale("log")
ax.set_xlim(8.7, 140)
ax.set_xticks(ks)
ax.set_xticklabels([str(k) for k in ks])
ax.minorticks_off()
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(38, 102)
ax.set_yticks([40, 50, 60, 70, 80, 90, 100])
ax.set_xlabel("Fresh labels per new class $k$", fontsize=7.2)
ax.set_ylabel("Top-1 (%)", fontsize=7.2)
ax.grid(True, axis="y", color="#EFEFEF", linewidth=0.35, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(0.4)
ax.tick_params(labelsize=5.8, width=0.4, length=1.8)

# ---- 直标（figures4papers 风格, 无框图例）----
label_texts = []
t1 = ax.text(11.0, 39.5, "decoupled rebuild", fontsize=5.8, ha="left", va="bottom",
             color=_darken(C_DECOUPLED, 0.6), zorder=4)
label_texts.append(t1)
t2 = ax.text(30.0, 74.0, "frozen head ($k$-indep.)", fontsize=5.8, ha="left", va="bottom",
             color=_darken(C_FROZEN, 0.6), zorder=4)
label_texts.append(t2)
t3 = ax.text(30.0, 99.4, "coupled ($k$-indep.)", fontsize=5.8, ha="left", va="bottom",
             color=_darken(C_COUPLED, 0.75), zorder=4)
label_texts.append(t3)
t4 = ax.text(13.0, 92.4, "$\\pm$ 2.3 pp parity band", fontsize=5.6, ha="left", va="top",
             color="#7F7F7F", zorder=4)
label_texts.append(t4)

# ---- 门禁（共享模块, 只读调用）----
import make_common_gates as gates
point_errs = ([(k, y - e, y + e, "decoupled") for k, y, e in zip(ks, dec_y, dec_e)]
              + [(k, y - e, y + e, "coupled") for k, y, e in zip(ks, cpl_y, cpl_e)]
              + [(k, y - e, y + e, "frozen") for k, y, e in zip(ks, frz_y, frz_e)])
gates.gate_print_robustness(
    "fig6-kladder",
    {"decoupled rebuild": C_DECOUPLED, "coupled": C_COUPLED, "frozen head": C_FROZEN},
    redundancy="3 distinct colors + 3 marker shapes + 3 line styles + direct labels")
gates.gate_labels(fig, ax, label_texts, name="fig6-kladder")
gates.gate_whisker_texts(fig, ax, label_texts, point_errs, name="fig6-kladder")
gates.gate_typography(fig, min_pt=5.5, name="fig6-kladder")
gates.gate_marks_in_axes(fig, name="fig6-kladder")

pdf_path = OUT / "fig6_kladder.pdf"
fig.savefig(pdf_path)
fig.savefig(OUT / "fig6_kladder.png", dpi=600)
gates.gate_pdf(pdf_path, max_w_pt=246.3)

for k in ks:
    print(f"k={k}: decoupled={ladder[k]['decoupled']['top1_overall']*100:.2f}% "
          f"coupled={ladder[k]['coupled']['top1_overall']*100:.2f}% "
          f"frozen={ladder[k]['frozen_head']['top1_overall']*100:.2f}%")
print(f"parity band: [{band_lo:.2f}, {band_hi:.2f}] (coupled ± {parity} pp)")
print("fig6 k-ladder saved")
