# -*- coding: utf-8 -*-
"""fig7 — 头对头预算曲线（NTU60 同一 harness 下 PSD / AimCLR / ST-GCN）分组柱状图。

数据真源: runs/p1_sprint/h2h_matrix/analysis.json（2026-11-17, matrix_mode=full）。
只读 budget_curve["0.05"|"0.10"|"0.20"] 三预算档 × psd/aimclr/stgcn 精确均值,
×100 转百分比; 误差棒仅取文件中实际存在的 std（psd_std 三档; AimCLR/ST-GCN 的
10% 五初始化种子 std 取自 aimclr_5seed_subset42/stgcn_5seed_subset42.std;
5%/20% 基线为单 run 无 std, 不画误差棒）。数值与正文 tab:h2h 逐位同源, 不新增统计量。

风格对齐 fig5_budget_retention v9: psd_style 统一样式; 固定画布宽 3.42in（不用
bbox_inches="tight"）; top/right spine 关闭; 无框图例; 字号下限 5.5pt。
系列色: PSD(proposed)=OCEAN_EMBER, AimCLR=OCEAN_CYAN, ST-GCN=中性灰;
hatch 作黑白打印冗余编码。
"""
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"
SRC = ROOT / "runs" / "p1_sprint" / "h2h_matrix" / "analysis.json"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
plt.rcParams["hatch.linewidth"] = 0.4

C_PSD = psd_style.SEM_EDGE     # #EC8E5A 深橙 = proposed
C_AIM = psd_style.PHYS_EDGE    # #4098AC 深青 = published SSL baseline
C_STGCN = psd_style.GRAY_2     # #767676 中性灰 = supervised reference

data = json.loads(SRC.read_text(encoding="utf-8"))
bc = data["budget_curve"]
budgets = ["0.05", "0.10", "0.20"]
xticklabels = ["5%", "10%", "20%"]

psd = [bc[b]["psd"] * 100.0 for b in budgets]
psd_e = [bc[b]["psd_std"] * 100.0 for b in budgets]
aim = [bc[b]["aimclr"] * 100.0 for b in budgets]
stg = [bc[b]["stgcn"] * 100.0 for b in budgets]
# 10% 档基线为五初始化种子均值, std 取自对应 subset42 块; 5%/20% 单 run 无 std
aim_e = [None, data["aimclr_5seed_subset42"]["std"] * 100.0, None]
stg_e = [None, data["stgcn_5seed_subset42"]["std"] * 100.0, None]

# ---- 固定画布: 宽 3.42in == 版心 246.24pt; 高 2.4in ----
FIG_W, FIG_H = 3.42, 2.4
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
fig.subplots_adjust(left=0.112, right=0.985, bottom=0.148, top=0.98)

x = list(range(len(budgets)))
W = 0.26
groups = [
    ("PSD semantic pipeline", psd, psd_e, C_PSD, ""),
    ("AimCLR linear (re-impl.)", aim, aim_e, C_AIM, "//"),
    ("Supervised ST-GCN", stg, stg_e, C_STGCN, "xx"),
]
label_texts = []
for gi, (name, vals, errs, col, hatch) in enumerate(groups):
    xs = [xi + (gi - 1) * W for xi in x]
    ax.bar(xs, vals, width=W, color=col, hatch=hatch, edgecolor="white",
           linewidth=0.5, zorder=2.5)
    for xi, v, e in zip(xs, vals, errs):
        if e is not None:
            ax.errorbar(xi, v, yerr=e, color=psd_style.INK, elinewidth=0.6,
                        capsize=2.0, capthick=0.6, ls="none", zorder=3)
        top = v + (e if e is not None else 0.0)
        t = ax.text(xi, top + 1.4, f"{v:.1f}", fontsize=5.5, ha="center", va="bottom",
                    color=psd_style.INK, zorder=4)
        label_texts.append(t)

# ---- 轴 / 刻度 ----
ax.set_xlim(-0.55, 2.55)
ax.set_xticks(x)
ax.set_xticklabels(xticklabels)
ax.set_ylim(0, 104)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Labeled budget", fontsize=7.2)
ax.set_ylabel("NTU60 xsub top-1 (%)", fontsize=7.2)
ax.grid(True, axis="y", color="#EFEFEF", linewidth=0.35, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(0.4)
ax.tick_params(labelsize=5.8, width=0.4, length=1.8)

# ---- 无框图例（左上空区; 5% 组最高标注 75.7, 图例底 ≈80 之上）----
leg = ax.legend([plt.Rectangle((0, 0), 1, 1, facecolor=g[3], hatch=g[4],
                               edgecolor="white", linewidth=0.5) for g in groups],
                [g[0] for g in groups], loc="upper left", fontsize=5.8,
                handlelength=1.1, handleheight=0.9, borderaxespad=0.2,
                labelspacing=0.35, frameon=False)

# ---- 门禁（共享模块, 只读调用）----
import make_common_gates as gates
point_errs = []
for gi, (name, vals, errs, col, hatch) in enumerate(groups):
    for xi, v, e in zip(x, vals, errs):
        if e is not None:
            point_errs.append((xi + (gi - 1) * W, v - e, v + e, name))
gates.gate_print_robustness(
    "fig7-h2h",
    {"PSD": C_PSD, "AimCLR": C_AIM, "ST-GCN": C_STGCN},
    redundancy="3 colors + 2 hatch patterns + value labels + legend")
gates.gate_labels(fig, ax, label_texts, name="fig7-h2h")
gates.gate_typography(fig, min_pt=5.5, name="fig7-h2h")
gates.gate_marks_in_axes(fig, name="fig7-h2h")

pdf_path = OUT / "fig7_h2h_budget.pdf"
fig.savefig(pdf_path)
fig.savefig(OUT / "fig7_h2h_budget.png", dpi=600)
gates.gate_pdf(pdf_path, max_w_pt=246.3)

for b, p, a, s in zip(xticklabels, psd, aim, stg):
    print(f"budget {b}: PSD={p:.2f} AimCLR={a:.2f} STGCN={s:.2f}")
print("fig7 h2h budget saved")
