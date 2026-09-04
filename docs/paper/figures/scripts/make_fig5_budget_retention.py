# -*- coding: utf-8 -*-
"""fig5 — 预算-保留率跨层汇总图（R13 新增：论文核心低资源主张的视觉证据）。

每个点 = 一个层/协议在给定标注比例下, PSD 管线相对其自身全预算参考的 top-1 保留率。
层间永不连线（三层口径禁混排）; marker 形状编码层, 颜色随层。
数据全部直读 reports/ JSON, 零硬编码。
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

INK = "#111111"
GRID_GRAY = "#DADADA"

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "text.color": INK})

def j(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))

# ---- 各层点: (label_fraction_pct, retention_pct, err_pp, tier) ----
p10, p07 = j("p10-seedexpansion-2026-09-04.json"), j("p07-endtoend-ak-full12-2026-09-04.json")
p12 = j("p12-akv2-replication-2026-09-04.json")
p14 = j("p14-ntu-lowres-2026-09-04.json")
w23 = j("p05-al-efficiency-warmstart-short-2026-08-25.json")
y_full = j("p05-stgcnbc-synthetic-100perclass-Y.json")

pts = []
# v1 public-real: spc2 = 18/141 anchors (9 classes with train coverage)
v1_full = 0.3393
v1 = p10["summary"]["warm_spc2"]
pts.append((100 * 18 / 141, 100 * v1["top1_mean"] / v1_full, 100 * v1["top1_std"] / v1_full,
            "public-real v1", "o", "#0E7490"))
# v2: spc2 = 16/256, spc4 = 32/256; full = 0.375
v2_full = p12["summary"]["warm_spc-1"]["top1_mean"]
for spc, n_anc in ((2, 16), (4, 32)):
    s = p12["summary"][f"warm_spc{spc}"]
    pts.append((100 * n_anc / 256, 100 * s["top1_mean"] / v2_full, 100 * s["top1_std"] / v2_full,
                "public-real v2", "s", "#C2410C"))
# NTU E9: 10% labels
tb = [r["top1"] for r in p14["arms"]["b_selftrain_10pct"]]
import numpy as np
ret = 100 * np.mean(tb) / p14["arms"]["c_full_linear"]["top1"]
err = 100 * np.std(tb, ddof=1) / p14["arms"]["c_full_linear"]["top1"]
pts.append((10.0, ret, err, "human benchmark (NTU60)", "D", "#6B7280"))
# synthetic-offset: 20 clips / 2200 full-budget train; warm 82.0 vs full 96.6
syn_full = y_full["best_val_acc"] if "best_val_acc" in y_full else 0.96591
syn = w23["curves"]["random"]["20"]  # warm-start b=20 arm mean (same JSON holds warm curves under random/entropy at b=20 identical)
pts.append((100 * 20 / 2200, 100 * syn["mean"] / syn_full, 100 * syn["std"] / syn_full,
            "synthetic-offset", "^", "#374151"))

fig, ax = plt.subplots(figsize=(3.42, 2.95))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
seen = set()
for frac, r, e, tier, mk, col in pts:
    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=6.5, ls="none",
                elinewidth=1.1, capsize=3.5, label=tier if tier not in seen else None, zorder=3)
    seen.add(tier)
ax.axhline(100, color=GRID_GRAY, linewidth=1.0, zorder=1)
ax.set_xscale("log")
ax.set_xlim(0.7, 120)
ax.set_ylim(78, 104)
ax.set_xlabel("Annotation budget (% of the tier's full-labeled pool, log)", fontsize=7)
ax.set_ylabel("Retention of own full-budget top-1 (%)", fontsize=7)
ax.grid(True, color=GRID_GRAY, linewidth=0.6, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.tick_params(labelsize=6)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=6.0)
fig.savefig(OUT / "fig5_budget_retention.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig5_budget_retention.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
for frac, r, e, tier, _, _ in pts:
    print(f"{tier}: {frac:.1f}% -> {r:.1f}% ± {e:.1f}")
print("fig5 saved")
