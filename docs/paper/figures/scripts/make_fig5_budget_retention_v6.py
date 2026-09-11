# -*- coding: utf-8 -*-
"""fig5 v6 — 预算-保留率跨层汇总图（R43b 顶刊级冲刺 · paper-figure P8"直标优于远端图例"）。

相对 v5 的顶刊化精修：
1. **直接标注替换远端图例**（Nature 系首选）：7 系列名称直标点旁（同相降L深化字色，
   psd_style 深化纪律），关联零歧义；顺带消解 critique 清单"类目数 7>5-6"WARN；
   标签用 caption 同名 tier 词（NTU60/NTU120/UCF101/PanAf500/synthetic-offset/
   public-real v1/v2），caption-图对应增强，零 caption 改动；
2. 标签落位经手工几何预计算（避开自家/邻点误差棒竖线）+ **程序化碰撞检测**
   （渲染后两两 bbox 相交断言 + 出轴断言，R24"judge 定量指控须程序化复核"教训落地）；
3. 网格 y-only 减淡(0.45)、轴线 0.65、ticks 0.65/2.4（顶刊细笔触）；
4. marker 7→6.4、误差棒 1.1→0.9/caps 3→2.6；100% 基准线右端加 "own full budget" 注记；
5. 图例条移除后画布 2.95→2.75in。

点定义/dodge/PanAf 前 10 种子口径/JSON 直读逐位承 v4-v5，零硬编码。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import itertools

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

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

# ---- 各层点: (label_fraction_pct, retention_pct, err_pp, tier) ----
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

fig, ax = plt.subplots(figsize=(3.42, 2.75))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

# 直标配置: tier -> (dx_data_or_None, dy, ha, 字色)  几何预计算避开误差棒竖线与邻点
LABEL_CONF = {
    "public-real v1":  (None,          16.0, "left",   psd_style.PHYS_DEEP),
    "public-real v2":  (None,          16.0, "center", psd_style.SEM_DEEP),
    "human NTU60":     ("NTU60",       94.4, "center", psd_style.HUMAN_DEEP),
    "human NTU120":    ("NTU120",      None, "left",   _darken(psd_style.S_NTU120, 0.42)),
    "indep. UCF101":   ("UCF101",      None, "right",  _darken(psd_style.S_UCF, 0.45)),
    "animal PanAf500": ("PanAf500",    84.8, "left",   _darken(psd_style.S_PANAF, 0.45)),
    "synthetic-offset":(None,          78.8, "center", _darken(psd_style.S_SYNTH_EDGE, 0.55)),
}
# 特例坐标 ( tier -> (x_text_override, y_text_override) )
LABEL_XY = {
    "public-real v1": (14.9, 16.0),    # whisker 右侧绕行(±22pp 长whisker 竖线穿过居中位)
    "human NTU120":   (17.15, None),   # 点右同高
    "indep. UCF101":  (8.15,  None),   # 点左同高(避开 8.8 处 whisker 竖线)
    "animal PanAf500":(14.00, 84.8),   # 右下(同高会撞 NTU120 点; 84.8 从其 whisker 前方绕行)
}

seen = set()
label_texts = []
point_errs = []   # (x, y_low, y_high, tier) 供 whisker 相交检查
for frac, r, e, tier, mk, col in pts:
    edge = psd_style.S_SYNTH_EDGE if col == psd_style.S_SYNTH else "white"
    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=6.4, ls="none",
                markerfacecolor=col, markeredgecolor=edge, markeredgewidth=1.0,
                elinewidth=0.9, capsize=2.6, zorder=3)
    point_errs.append((frac, r - e, r + e, tier))
    if tier in seen:
        continue
    seen.add(tier)
    text_ovr, y_ovr, ha, tcol = LABEL_CONF[tier]
    text = text_ovr if text_ovr is not None else tier
    x_ovr, y2_ovr = LABEL_XY.get(tier, (None, None))
    tx = x_ovr if x_ovr is not None else frac
    ty = y2_ovr if y2_ovr is not None else (y_ovr if y_ovr is not None else r)
    t = ax.text(tx, ty, text, fontsize=5.8, color=tcol, ha=ha, va="center",
                fontweight="bold", zorder=5)
    label_texts.append(t)

ax.axhline(100, color=GRID_GRAY, linewidth=0.9, zorder=1)
ax.text(29.2, 102.0, "own full budget", fontsize=5.2, color="#767676",
        ha="right", va="bottom", zorder=2)
ax.set_xscale("log")
ax.set_xlim(4, 30)
ax.set_xticks([5, 10, 20])
ax.set_xticklabels(["5", "10", "20"])
ax.minorticks_off()  # log 轴边界 minor 会被 LogFormatter 打成 "4 × 10⁰" 混排, 全关
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Annotation budget (% of the tier's full-labeled pool, log)", fontsize=7)
ax.set_ylabel("Retention of own full-budget top-1 (%)", fontsize=7)
ax.grid(True, axis="y", color=GRID_GRAY, linewidth=0.45, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(0.65)
ax.tick_params(labelsize=6, width=0.65, length=2.4)

# ---- 出图门禁（任务包 A 第三轮：统一收敛到共享门禁模块 make_common_gates） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig5", {"public-real v1": psd_style.S_PUBV1, "public-real v2": psd_style.S_PUBV2,
             "human NTU60": psd_style.S_NTU60, "human NTU120": psd_style.S_NTU120,
             "indep. UCF101": psd_style.S_UCF, "animal PanAf500": psd_style.S_PANAF,
             "synthetic-offset": psd_style.S_SYNTH},
    redundancy="7 unique marker shapes + direct labels")
gates.gate_labels(fig, ax, label_texts, name="fig5")
gates.gate_whisker_texts(fig, ax, label_texts, point_errs, name="fig5")

fig.savefig(OUT / "fig5_budget_retention.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig5_budget_retention.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
gates.gate_pdf(OUT / "fig5_budget_retention.pdf")
for frac, r, e, tier, _, _ in pts:
    print(f"{tier}: {frac:.1f}% -> {r:.1f}% ± {e:.1f}")
print("fig5 v6 saved")
