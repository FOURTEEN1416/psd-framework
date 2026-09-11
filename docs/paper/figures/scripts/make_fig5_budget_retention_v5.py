# -*- coding: utf-8 -*-
"""fig5 v5 — 预算-保留率跨层汇总图（R43 全图重绘 · 数模竞赛·科研工具箱/paper-figure
P4/P7/P17 + composition-patterns；点定义/dodge/PanAf 前 10 种子口径逐位承 v4）。

相对 v4 的重绘点：
1. x 轴域收紧 (0.7,120)→(4,30)：v4 右半幅（20–120%）完全空置、主簇被压在中部；
   收紧后 5.5–16.5% 点簇横向展开近一倍；刻度 [5,10,20]；
2. 图例压缩：2 列×4 行长标签 → 4 列×2 行短标签（human NTU60 等；
   benchmark 全称与 HRNet 2D 细节由 caption 承载），底部占高显著下降；
3. ylim (2,112)→(0,108)：保留率 0 起步（P4 诚实轴），顶部死区回收。

层间永不连线（三层口径禁混排）; marker 形状冗余编码层, 颜色随层（语义锁承 psd_style）。
数据直读 reports/ JSON, 零硬编码精度值。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
GRID_GRAY = psd_style.GRID_GRAY


def j(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))

# ---- 各层点: (label_fraction_pct, retention_pct, err_pp, tier) ----
r16 = j("r16-endtoend-pseudo-2026-09-05.json")
r16ntu = j("r16-ntu-pseudo-10seed-2026-09-07.json")
p07 = j("p07-endtoend-ak-full12-2026-09-04.json")   # supervised full reference only
p12 = j("p12-akv2-replication-2026-09-04.json")     # supervised full reference only
p14 = j("p14-ntu-lowres-2026-09-04.json")           # supervised (c) reference only
w23 = j("p05-al-efficiency-warmstart-short-2026-08-25.json")

pts = []
# v1 public-real: spc2 = 18/141 anchors (9 classes with train coverage)
# x dodged 12.8->13.9 to keep the v1 error whisker clear of the v2 spc4 marker (12.5->11.5)
v1_full = p07["agg"]["warm_spc-1"]["top1_mean"]
v1 = r16["summary"]["v1"]["warm_spc2"]
pts.append((13.9, 100 * v1["top1_mean"] / v1_full, 100 * v1["top1_std"] / v1_full,
            "public-real v1", "o", psd_style.S_PUBV1))
# v2: spc2 = 14/256 (5.5%), spc4 = 28/256 (10.9%); supervised full reference from p12
v2_full = p12["summary"]["warm_spc-1"]["top1_mean"]
for spc, n_anc, x_disp in ((2, 14, 5.5), (4, 28, 11.5)):
    s = r16["summary"]["v2"][f"warm_spc{spc}"]
    pts.append((x_disp, 100 * s["top1_mean"] / v2_full, 100 * s["top1_std"] / v2_full,
                "public-real v2", "s", psd_style.S_PUBV2))
# NTU E9 (corrected): 10% labels; supervised (c) reference from p14
tb = [r["top1"] for r in r16ntu["arms"]["b_selftrain_10pct"]]
c_ref = p14["arms"]["c_full_linear"]["top1"]
ret = 100 * np.mean(tb) / c_ref
err = 100 * np.std(tb, ddof=1) / c_ref
pts.append((10.0, ret, err, "human NTU60", "D", psd_style.S_NTU60))
# E9b NTU120 / E9c UCF101 / E9d PanAf500 (P5/P7, corrected protocol): 10% labels each;
# x dodged along the log axis so the four 10% markers and their error whiskers
# stay disjoint (adjacent marker edges clear each other; two judge rounds):
# UCF101 8.8, NTU60 10.0 (in situ), PanAf500 12.5, NTU120 16.5
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
tbpan = p23_panaf["b_arms"][:10]  # R24 交叉审: p23 文件被 Amendment 2 扩为 30 seeds 后覆盖,
# 决策口径(E9d/tab2/caption)是前 10 个种子 42-51 (53.2±9.1, ret 88.7); 后 20 为诊断扩展
retpan = 100 * np.mean(tbpan) / p23_panaf["full_ref"]
errpan = 100 * np.std(tbpan, ddof=1) / p23_panaf["full_ref"]
pts.append((13.2, retpan, errpan, "animal PanAf500", "X", psd_style.S_PANAF))
# synthetic-offset: 20 clips of the 220-clip offset pool; warm 82.0 vs full-budget(offset) 95.7
syn_full = w23["curves"]["random"]["200"]["mean"]  # offset-tier full-budget (same warm-start protocol), NOT base-tier 96.6%
syn = w23["curves"]["random"]["20"]  # warm-start b=20 arm mean (same JSON holds warm curves under random/entropy at b=20 identical)
pts.append((8.0, 100 * syn["mean"] / syn_full, 100 * syn["std"] / syn_full,
            "synthetic-offset", "^", psd_style.S_SYNTH))

fig, ax = plt.subplots(figsize=(3.42, 2.95))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
seen = set()
for frac, r, e, tier, mk, col in pts:
    edge = psd_style.S_SYNTH_EDGE if col == psd_style.S_SYNTH else "white"  # 黄系深描边保可见
    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=7.0, ls="none",
                markerfacecolor=col, markeredgecolor=edge, markeredgewidth=1.0,
                elinewidth=1.1, capsize=3.0, label=tier if tier not in seen else None, zorder=3)
    seen.add(tier)
ax.axhline(100, color=GRID_GRAY, linewidth=1.0, zorder=1)
ax.set_xscale("log")
ax.set_xlim(4, 30)
ax.set_xticks([5, 10, 20])
ax.set_xticklabels(["5", "10", "20"])
ax.minorticks_off()  # log 轴边界 minor(4/6/30) 会被 LogFormatter 打成 "4 × 10⁰" 混排, 全关
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Annotation budget (% of the tier's full-labeled pool, log)", fontsize=7)
ax.set_ylabel("Retention of own full-budget top-1 (%)", fontsize=7)
ax.grid(True, color=GRID_GRAY, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(labelsize=6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4, frameon=False,
          fontsize=5.3, handlelength=1.2, columnspacing=0.9, handletextpad=0.4)
fig.savefig(OUT / "fig5_budget_retention.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig5_budget_retention.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
for frac, r, e, tier, _, _ in pts:
    print(f"{tier}: {frac:.1f}% -> {r:.1f}% ± {e:.1f}")
print("fig5 v5 saved")
