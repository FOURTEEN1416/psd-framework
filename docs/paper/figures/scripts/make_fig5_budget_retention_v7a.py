# -*- coding: utf-8 -*-
"""fig5 v7a — 风格方向 A「编辑极简 · 单色骨架 + 单强调轴」提案样板（收尾修复）。

数据逻辑逐位承 v6（pts 构造 / JSON 直读 / 100*mean/ref 口径 / dodge x 位移 /
PanAf 前 10 种子口径），零硬编码、零改点。仅重做呈现：灰阶阶梯 + 单一暖强调
(public-real v2 = S_PUBV2)，全墨直标 + 小色点色键，去网格改三条极浅参考线。
本次收尾（wt/figB 2026-09-12）：右缘三标签改左对齐列 + 色键点一律左外侧；
NTU60 上移让出 diamond 顶 ≥2px；synthetic-offset 下移到 whisker 下帽以下 +
色键点左外侧；PanAf500 右端留 ≥0.4 余量；补 G5/G6 共享门禁 + 本地文字×标记自检。
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "reports" / "figB-proposal"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
GRID_GRAY = psd_style.GRID_GRAY

def j(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))

# ---- 各层点: (label_fraction_pct, retention_pct, err_pp, tier, marker, color) ----
# （pts 构造逐行照抄 v6；颜色 token 暂留原值，呈现色由 COLORMAP 重映射）
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

# ---- 风格方向 A：灰阶阶梯(6 档) + 单一暖强调(public-real v2) ----
COLORMAP = {
    "public-real v1":   psd_style.GRAY_4,       # #272727 近黑
    "public-real v2":   psd_style.S_PUBV2,      # #EC8E5A 唯一暖强调
    "human NTU60":      psd_style.GRAY_2,       # #767676
    "human NTU120":     psd_style.NEUTRAL_LT,   # #CFCECE
    "indep. UCF101":    psd_style.GRAY_FILL,    # #E3E3E3
    "animal PanAf500":  psd_style.GRAY_1,       # #A6A6A6
    "synthetic-offset": psd_style.GRAY_3,       # #4D4D4D
}

fig, ax = plt.subplots(figsize=(3.42, 2.80))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")

point_errs = []
for frac, r, e, tier, mk, _ in pts:
    col = COLORMAP[tier]
    mec = "white" if tier == "public-real v2" else INK
    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=6.2, ls="none",
                markerfacecolor=col, markeredgecolor=mec, markeredgewidth=0.9,
                elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=3)
    point_errs.append((frac, r - e, r + e, tier))

# ---- 参考线：去网格，仅 y=0/50 极浅 + 100% 基线 hairline ----
ax.axhline(0,   color="#EFEFEF", linewidth=0.4, zorder=0)
ax.axhline(50,  color="#EFEFEF", linewidth=0.4, zorder=0)
ax.axhline(100, color=GRID_GRAY, linewidth=0.9, zorder=1)

# 100% 基线注记（修复 v6 的 5.2 违规项 -> 5.5）
note_t = ax.text(29.2, 102.0, "own full budget", fontsize=5.5, color="#767676",
                 ha="right", va="bottom", zorder=2)

ax.set_xscale("log")
ax.set_xlim(4, 30)
ax.set_xticks([5, 10, 20])
ax.set_xticklabels(["5", "10", "20"])
ax.minorticks_off()
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Annotation budget (% of the tier's full-labeled pool, log)", fontsize=7)
ax.set_ylabel("Retention of own full-budget top-1 (%)", fontsize=7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(0.6)
ax.tick_params(labelsize=5.6, width=0.6, length=2.2)

# ---- 直标（全墨 INK / fontsize 5.6 / normal） ----
# 右缘三标签改左对齐列（RAIL_ANCHOR 即引线终点=marker 坐标，y 取数据点 y 台阶）
RAIL_ANCHOR = {
    "public-real v1":  (13.9, 28.94),
    "public-real v2":  (11.5, 44.16),
    "indep. UCF101":   (8.8,  66.63),
}
DIRECT = {
    "human NTU60":      (10.0,  94.4,  "center", "NTU60"),
    "human NTU120":     (17.6,  88.87, "left",   "NTU120"),
    "animal PanAf500":  (14.00, 84.8,  "left",   "PanAf500"),
    "synthetic-offset": (8.0,   78.8,  "center", "synthetic-offset"),
}
label_texts = []
rail_texts = []
TXT2TIER = {}          # 显示文字 -> 全 tier 名（DIRECT 用短名，COLORMAP 用全名）
for tier, (mx, my) in RAIL_ANCHOR.items():
    t = ax.text(20.0, my, tier, fontsize=5.6, color=INK, ha="left",
                va="center", fontweight="normal", zorder=5)
    label_texts.append(t); rail_texts.append(t)
    TXT2TIER[tier] = tier
for tier, (tx, ty, ha, txt) in DIRECT.items():
    t = ax.text(tx, ty, txt, fontsize=5.6, color=INK, ha=ha,
                va="center", fontweight="normal", zorder=5)
    label_texts.append(t)
    TXT2TIER[txt] = tier

# ---- 几何精修（先 draw 取 bbox，renderer 实测后调坐标） ----
DPI = float(fig.dpi)
MK_R_PX = 6.2 * DPI / 72 / 2          # errorbar marker 包围圆半径(px)
DOT_S = 4.5                            # 色键点面积(pt^2) ~ 直径 2.4pt
DOT_R_PX = float(np.sqrt(DOT_S / np.pi) * DPI / 72)
EDGE_DATA = 0.18                       # 色键点边到文字 bbox 的间隙(数据单位，缺陷1)

fig.canvas.draw()
ren = fig.canvas.get_renderer()
inv = ax.transData.inverted()
axbb = ax.get_window_extent(ren)
XL = ax.get_xlim()
px_per_log = axbb.width / (np.log10(XL[1]) - np.log10(XL[0]))

def edge_px(x):
    return EDGE_DATA * px_per_log / (x * np.log(10))

def shift_y_px(t, dpx):
    x0, y0 = t.get_position()
    d = ax.transData.inverted().transform(
        (ax.transData.transform((x0, y0))[0], ax.transData.transform((x0, y0))[1] + dpx))
    t.set_y(d[1])

# (1) RAIL 三标签左对齐到共同右锚 RX，使最长右端 = 29.6(≥0.4 余量)
right_edges = [inv.transform((t.get_window_extent(ren).x1,
                               t.get_window_extent(ren).y0))[0] for t in rail_texts]
shift = 29.6 - max(right_edges)
for t in rail_texts:
    t.set_x(t.get_position()[0] + shift)

fig.canvas.draw(); ren = fig.canvas.get_renderer(); axbb = ax.get_window_extent(ren)

# (2) NTU60 上移：label bbox 底 ≥ diamond 顶 + 2px
t60 = [t for t in label_texts if t.get_text() == "NTU60"][0]
mc = ax.transData.transform((10.0, 90.70))
diamond_top_px = mc[1] + MK_R_PX
bb = t60.get_window_extent(ren)
if bb.y0 < diamond_top_px + 2:
    shift_y_px(t60, diamond_top_px + 2 - bb.y0)

# (3) synthetic-offset 下移：label bbox 顶 ≤ whisker 下帽 - 2px
cap_y = 85.74 - 4.51
cap_py = ax.transData.transform((8.0, cap_y))[1]
ts = [t for t in label_texts if t.get_text() == "synthetic-offset"][0]
bb = ts.get_window_extent(ren)
if bb.y1 > cap_py - 2:
    shift_y_px(ts, -(bb.y1 - (cap_py - 2)))

fig.canvas.draw(); ren = fig.canvas.get_renderer(); axbb = ax.get_window_extent(ren)

# (4) 所有标签右端留 ≥0.4(PanAf 缺陷4 通用护栏)；越界左移
for t in label_texts:
    bb = t.get_window_extent(ren)
    rdata = inv.transform((bb.x1, bb.y0))[0]
    if rdata > 29.6:
        t.set_x(t.get_position()[0] - (rdata - 29.6))

fig.canvas.draw(); ren = fig.canvas.get_renderer()

# ---- 色键点（紧贴文字 bbox 外侧，不与文字/标记压叠） ----
dot_list = []   # (x_data, y_data, name)
for t in label_texts:
    bb = t.get_window_extent(ren)
    tier = TXT2TIER[t.get_text()]
    if tier in RAIL_ANCHOR:                 # 左对齐 -> 色键点放文字 bbox 左外侧
        cx = bb.x0 - (edge_px(inv.transform((bb.x0, bb.y0))[0]) + DOT_R_PX)
        cy = (bb.y0 + bb.y1) / 2
    elif tier == "NTU60":                    # 居中 -> 色键点放文字 bbox 正上方
        cx = (bb.x0 + bb.x1) / 2
        cy = bb.y1 + (edge_px(inv.transform(((bb.x0+bb.x1)/2, bb.y0))[0]) + DOT_R_PX)
    else:                                    # 左对齐 DIRECT -> 文字 bbox 左外侧
        cx = bb.x0 - (edge_px(inv.transform((bb.x0, bb.y0))[0]) + DOT_R_PX)
        cy = (bb.y0 + bb.y1) / 2
    d = inv.transform((cx, cy))
    dot_list.append((d[0], d[1], tier))
    ax.scatter([d[0]], [d[1]], marker="o", s=DOT_S, color=COLORMAP[tier],
               edgecolors=INK, linewidths=0.3, zorder=6)

# ---- 右缘引线：标签左端 -> 对应 marker（zorder=2，不盖 marker） ----
for t in rail_texts:
    tier = t.get_text()
    mx, my = RAIL_ANCHOR[tier]
    tx, ty = t.get_position()
    ax.plot([tx, mx], [ty, my], color="#BFBFBF", linewidth=0.5,
            zorder=2, solid_capstyle="round")

# ---- 本地自检：文字×标记 压叠（不进共享模块，仅本脚本内） ----
# 参与方：7 直标 + "own full budget" 注记 的 bbox，逐一与 8 marker + 7 色键点 的 bbox 相交
fig.canvas.draw(); ren = fig.canvas.get_renderer()
mark_bboxes = []
for frac, r, e, tier, mk, _ in pts:
    px = ax.transData.transform((frac, r))
    mark_bboxes.append(("marker:" + tier,
                        Bbox.from_extents(px[0] - MK_R_PX, px[1] - MK_R_PX,
                                          px[0] + MK_R_PX, px[1] + MK_R_PX)))
dot_bboxes = []
for dx, dy, nm in dot_list:
    px = ax.transData.transform((dx, dy))
    dot_bboxes.append(("dot:" + nm,
                       Bbox.from_extents(px[0] - DOT_R_PX, px[1] - DOT_R_PX,
                                         px[0] + DOT_R_PX, px[1] + DOT_R_PX)))
texts_chk = label_texts + [note_t]
hits = 0
for t in texts_chk:
    tb = t.get_window_extent(ren)
    for nm, mb in mark_bboxes + dot_bboxes:
        if tb.overlaps(mb):
            ix = min(tb.x1, mb.x1) - max(tb.x0, mb.x0)
            iy = min(tb.y1, mb.y1) - max(tb.y0, mb.y0)
            if ix * iy > 0:
                hits += 1
                print(f"  SELFCHECK-HIT: {t.get_text()!r} x {nm} (ovl {ix*iy:.1f} px^2)")
print(f"[LOCAL self-check] texts={len(texts_chk)} vs marks={len(mark_bboxes)}+dots="
      f"{len(dot_bboxes)} -> {'PASS' if hits == 0 else 'FAIL'} ({hits} hits)")
if hits:
    raise SystemExit("fig5-v7a: local text×mark self-check FAILED")

# ---- 出图门禁（共享模块 make_common_gates） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig5-v7a", {k: COLORMAP[k] for k in COLORMAP},
    redundancy="7 unique marker shapes + direct labels + color-key dots")
gates.gate_labels(fig, ax, label_texts, name="fig5-v7a")
gates.gate_whisker_texts(fig, ax, label_texts, point_errs, name="fig5-v7a")
gates.gate_typography(fig, min_pt=5.5, name="fig5-v7a")
gates.gate_marks_in_axes(fig, name="fig5-v7a")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig5_v7a.pdf", bbox_inches="tight", pad_inches=0.02)
fig.savefig(OUT / "fig5_v7a.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
gates.gate_pdf(OUT / "fig5_v7a.pdf")
for frac, r, e, tier, _, _ in pts:
    print(f"{tier}: {frac:.1f}% -> {r:.1f}% ± {e:.1f}")
print("fig5 v7a saved")
