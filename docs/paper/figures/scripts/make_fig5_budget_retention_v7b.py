# -*- coding: utf-8 -*-
"""fig5 v7b — 风格方向 B：高密度组合面板（Nature/Cell 式 small multiples）。

在 v6 数据逻辑逐位照搬前提下，把单面板按 caption 的 tier 分组拆成两个共享 x 轴的
面板：(a) 高保留区 human/animal/synthetic，(b) 低保留区 indep./public-real。
保留 v6 的 7 个 marker 形状、psd_style 色板与 _darken 局部深化助手；字号下限 5.5。

数据诚实性：两面板 ylim 放宽以装下本面板全部 whisker（选①），不再截断误差棒。
图级单一 y 轴名用 fig.supylabel；面板字母轴内上方。产物只写 reports/figB-proposal。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "reports" / "figB-proposal"
OUT.mkdir(parents=True, exist_ok=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
GRID_GRAY = psd_style.GRID_GRAY

def _darken(hex_color, f):
    """同相降L（psd_style 深化纪律的多相变体）：RGB 乘 f 保色相，供直标字色。"""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(int(r*f), int(g*f), int(b*f))


def j(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))

# ---- 各层点: (label_fraction_pct, retention_pct, err_pp, tier) — 逐位照抄 v6 ----
r16 = j("r16-endtoend-pseudo-2026-09-05.json")
r16ntu = j("r16-ntu-pseudo-10seed-2026-09-07.json")
p07 = j("p07-endtoend-ak-full12-2026-09-04.json")   # supervised full reference only
p12 = j("p12-akv2-replication-2026-09-04.json")     # supervised full reference only
p14 = j("p14-ntu-lowres-2026-09-04.json")           # supervised (c) reference only
w23 = j("p05-al-efficiency-warmstart-short-2026-08-25.json")

pts = []
v1_full = p07["agg"]["warm_spc-1"]["top1_mean"]
v1 = r16["summary"]["v1"]["warm_spc2"]
pts.append((13.9, 100 * v1["top1_mean"] / v1_full, 100 * v1["top1_std"] / v1_full,
            "public-real v1", "o", psd_style.S_PUBV1))
v2_full = p12["summary"]["warm_spc-1"]["top1_mean"]
for spc, n_anc, x_disp in ((2, 14, 5.5), (4, 28, 11.5)):
    s = r16["summary"]["v2"][f"warm_spc{spc}"]
    pts.append((x_disp, 100 * s["top1_mean"] / v2_full, 100 * s["top1_std"] / v2_full,
                "public-real v2", "s", psd_style.S_PUBV2))
tb = [r["top1"] for r in r16ntu["arms"]["b_selftrain_10pct"]]
c_ref = p14["arms"]["c_full_linear"]["top1"]
ret = 100 * np.mean(tb) / c_ref
err = 100 * np.std(tb, ddof=1) / c_ref
pts.append((10.0, ret, err, "human NTU60", "D", psd_style.S_NTU60))
p5b_ntu = j("p5b-ntu120-retention-2026-09-07.json")
p5b_ucf = j("p5b-ucf101-retention-2026-09-07.json")
p23_panaf = j("p23-panaf-retention-2026-09-07.json")
tb120 = p5b_ntu["b_arms"]
ret120 = 100 * np.mean(tb120) / p5b_ntu["full_ref"]
err120 = 100 * np.std(tb120, ddof=1) / p5b_ntu["full_ref"]
pts.append((16.5, ret120, err120, "human NTU120", "v", psd_style.S_NTU120))
tbucf = p5b_ucf["b_arms"]
retucf = 100 * np.mean(tbucf) / p5b_ucf["full_ref"]
errucf = 100 * np.std(tbucf, ddof=1) / p5b_ucf["full_ref"]
pts.append((8.8, retucf, errucf, "indep. UCF101", "P", psd_style.S_UCF))
tbpan = p23_panaf["b_arms"][:10]  # 决策口径(E9d/tab2/caption)=前 10 种子 42-51; 后 20 为诊断扩展
retpan = 100 * np.mean(tbpan) / p23_panaf["full_ref"]
errpan = 100 * np.std(tbpan, ddof=1) / p23_panaf["full_ref"]
pts.append((13.2, retpan, errpan, "animal PanAf500", "X", psd_style.S_PANAF))
syn_full = w23["curves"]["random"]["200"]["mean"]
syn = w23["curves"]["random"]["20"]
pts.append((8.0, 100 * syn["mean"] / syn_full, 100 * syn["std"] / syn_full,
            "synthetic-offset", "^", psd_style.S_SYNTH))

# ---- 双面板 sharex：按 caption 的 tier 分组 ----
PANEL_A = {"human NTU60", "human NTU120", "animal PanAf500", "synthetic-offset"}
PANEL_B = {"indep. UCF101", "public-real v2", "public-real v1"}

fig, (axa, axb) = plt.subplots(
    2, 1, figsize=(3.42, 3.05), sharex=True,
    gridspec_kw=dict(height_ratios=[1, 1], hspace=0.22))
fig.patch.set_facecolor("white")
axa.set_facecolor("white"); axb.set_facecolor("white")

# 直标位置（数据坐标；字号下限 5.5）。ylim 已放宽装下全部 whisker，标签偏移重算。
LABEL_POS = {
    "human NTU60":     ("NTU60",          10.0,  92.8, "center", psd_style.HUMAN_DEEP),
    "human NTU120":    ("NTU120",         17.15, 88.9, "left",   _darken(psd_style.S_NTU120, 0.42)),
    "animal PanAf500": ("PanAf500",       14.0,  84.8, "left",   _darken(psd_style.S_PANAF, 0.45)),
    "synthetic-offset":("synthetic-offset", 8.0, 79.5, "center", _darken(psd_style.S_SYNTH_EDGE, 0.55)),
    "indep. UCF101":   ("UCF101",          8.15,  66.6, "right",  _darken(psd_style.S_UCF, 0.45)),
    "public-real v2":  ("public-real v2",  6.6,   38.0, "left",   psd_style.SEM_DEEP),
    "public-real v1":  ("public-real v1", 14.9,   30.0, "left",   psd_style.PHYS_DEEP),
}

labels_a, labels_b = [], []
errs_a, errs_b = [], []
seen = set()
for frac, r, e, tier, mk, col in pts:
    ax = axa if tier in PANEL_A else axb
    edge = psd_style.S_SYNTH_EDGE if col == psd_style.S_SYNTH else "white"
    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=6.0, ls="none",
                markerfacecolor=col, markeredgecolor=edge, markeredgewidth=1.0,
                elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=3)
    if tier in PANEL_A:
        errs_a.append((frac, r - e, r + e, tier))
    else:
        errs_b.append((frac, r - e, r + e, tier))
    if tier in seen:
        continue
    seen.add(tier)
    text, tx, ty, ha, tcol = LABEL_POS[tier]
    t = ax.text(tx, ty, text, fontsize=5.6, color=tcol, ha=ha, va="center",
                fontweight="bold", zorder=5)
    (labels_a if tier in PANEL_A else labels_b).append(t)

# (a) 面板：去掉 100% 参考线（超出 ylim）；右上角放一条 5.5pt 灰字注记
axa.text(29.0, 104.5, "100% = own full budget", fontsize=5.5, color="#767676",
         ha="right", va="top", zorder=2)
# (b) 面板：无 100% 线

# ---- 轴系设置（两面板各自） ----
for ax, ylim in ((axa, (72, 106)), (axb, (4, 78))):
    ax.set_yscale("linear")
    ax.set_ylim(*ylim)
    ax.set_xscale("log")
    ax.set_xlim(4, 30)
    ax.set_xticks([5, 10, 20])
    ax.set_xticklabels(["5", "10", "20"])
    ax.minorticks_off()
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis="y", color="#EFEFEF", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(labelsize=5.6, width=0.6, length=2.2)

axa.set_yticks([75, 85, 95, 105])
axb.set_yticks([10, 30, 50, 70])
axa.tick_params(labelbottom=False)  # 仅下panel 显示 x 刻度标签与轴名
axb.set_xlabel("Annotation budget (% of the tier's full-labeled pool, log)", fontsize=6.5)

# 面板字母 a/b（轴内上方，粗体小写；左上 vs 右上注记不冲突）
axa.text(0.015, 1.015, "a", transform=axa.transAxes, fontsize=8,
         fontweight="bold", ha="left", va="bottom")
axb.text(0.015, 1.015, "b", transform=axb.transAxes, fontsize=8,
         fontweight="bold", ha="left", va="bottom")

# 图级单一 y 轴名（避免双面板各 set_ylabel 重印叠字）；左边距留足使其在 canvas 内
fig.subplots_adjust(left=0.20, right=0.97, top=0.95, bottom=0.14, hspace=0.22)
fig.supylabel("Retention of own full-budget top-1 (%)", fontsize=6.2, va="center")

# ---- 出图门禁（G1–G6） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig5-v7b", {"public-real v1": psd_style.S_PUBV1, "public-real v2": psd_style.S_PUBV2,
                 "human NTU60": psd_style.S_NTU60, "human NTU120": psd_style.S_NTU120,
                 "indep. UCF101": psd_style.S_UCF, "animal PanAf500": psd_style.S_PANAF,
                 "synthetic-offset": psd_style.S_SYNTH},
    redundancy="7 unique marker shapes + direct labels")
gates.gate_labels(fig, axa, labels_a, name="fig5-v7b-a")
gates.gate_labels(fig, axb, labels_b, name="fig5-v7b-b")
gates.gate_whisker_texts(fig, axa, labels_a, errs_a, name="fig5-v7b-a")
gates.gate_whisker_texts(fig, axb, labels_b, errs_b, name="fig5-v7b-b")
gates.gate_typography(fig, name="fig5-v7b")
gates.gate_marks_in_axes(fig, name="fig5-v7b")

pdf_path = OUT / "fig5_v7b.pdf"
png_path = OUT / "fig5_v7b.png"
fig.savefig(pdf_path)                                  # 不用 tight：MediaBox 宽=3.42in=246.24pt
fig.savefig(png_path, dpi=600)                        # 600dpi
gates.gate_pdf(pdf_path)

for frac, r, e, tier, _, _ in pts:
    print(f"{tier}: {frac:.1f}% -> {r:.1f}% ± {e:.1f}")
print("fig5 v7b saved")
