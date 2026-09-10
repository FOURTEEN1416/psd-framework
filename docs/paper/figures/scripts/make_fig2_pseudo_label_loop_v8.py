# -*- coding: utf-8 -*-
r"""fig2 v8 — 伪标签循环双行管线（R25 审美统一：psd_style 取色 + Arial 字族；版式承 v7）。

v6 的"左环+右链"在 Assign 左下与 reest 绕行折线两处反复产生箭头重影/穿盒遮挡
(三轮 judge 均实锤)。v7 改经典双行管线: 上排=自动环(线性四站+顶部回环弧
next round), 下排=人工分支+中心 hub。所有边都是短直线/一条顶弧, 无交叉无遮挡。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
PAPER = "#FFFFFF"
INK = psd_style.INK
ACCENT = psd_style.SEM_EDGE
ACCENT_TINT = psd_style.SEM_FILL
FOCAL_FILL = psd_style.FOCAL_FILL
GRAY_EDGE = psd_style.GRAY_EDGE
GRAY_FILL = psd_style.GRAY_FILL
SOFT = psd_style.GRAY_LINE
C_TEXT = psd_style.INK

fig, ax = plt.subplots(figsize=(3.42, 2.95))
ax.set_xlim(0, 100); ax.set_ylim(0, 90)
ax.axis("off")

ST_W, ST_H = 19.0, 8.2
HUB_X, HUB_Y = 61.0, 40.0
HUB_W, HUB_H = 17.0, 10.0

ROW1_Y = 66.0
S = {
    "assign":  (11.0, ROW1_Y),
    "pool":    (36.5, ROW1_Y),
    "upd":     (62.0, ROW1_Y),
    "reest":   (87.5, ROW1_Y),
    "al":      (11.0, 40.0),
    "verified":(36.5, 40.0),
}
NAMES = {
    "assign": "Assign proposals", "pool": "Label pool",
    "upd": "Update $\\Omega$", "reest": "Re-estimate $P$",
    "al": "AL queue", "verified": "Verified seeds",
}
SUBS = {
    "assign": "conf. $\\kappa$", "pool": "conf. pass",
    "upd": "seeds $\\cup$ pool", "reest": "each round",
    "al": "human, $\\leq$B", "verified": "queue output",
}
FOCAL = "pool"
MANUAL = {"al", "verified"}

def edge_pt(c, target, hw=ST_W/2, hh=ST_H/2):
    v = np.array([target[0]-c[0], target[1]-c[1]], float)
    L = np.linalg.norm(v); v = v/L
    t = np.inf
    if abs(v[0]) > 1e-9: t = min(t, hw/abs(v[0]))
    if abs(v[1]) > 1e-9: t = min(t, hh/abs(v[1]))
    return (c[0]+v[0]*t, c[1]+v[1]*t)

def seg(a, b, label=None, lp=0.5, ldy=2.2, curve=0.0, ls="solid", lw=1.15, color=SOFT,
        a_box=True, b_box=True, a_hw=None):
    ca = S[a] if a_box else a
    cb = S[b] if b_box else b
    kw_a = dict(hw=ST_W/2, hh=ST_H/2)
    st = edge_pt(ca, cb, **kw_a) if a_box else a
    en = edge_pt(cb, ca, **kw_a) if b_box else b
    ap = dict(arrowstyle="-|>", mutation_scale=7.5, color=color, lw=lw, linestyle=ls)
    if curve:
        ap["connectionstyle"] = f"arc3,rad={curve}"
    ax.add_patch(FancyArrowPatch(st, en, **ap, zorder=2))
    if label:
        mx = st[0]+(en[0]-st[0])*lp; my = st[1]+(en[1]-st[1])*lp+ldy
        ax.text(mx, my, label, fontsize=5.8, color=psd_style.NOTE, ha="center", va="center", zorder=5)

# ---- 上排自动环(线性) ----
seg("assign", "pool", label=r"$\kappa \geq \tau$", ldy=5.2)
seg("pool", "upd")
seg("upd", "reest")
# 回环: reest -> assign, 单一顶部大弧(在 boxes 上方 y~78, 不与任何元素交)
# 三段折线回环: reest 顶 -> 上 -> 左 -> assign 顶(箭头)
ax.plot([87.5, 87.5], [70.3, 82.0], color=SOFT, lw=1.15, zorder=2)
ax.plot([87.5, 11.0], [82.0, 82.0], color=SOFT, lw=1.15, zorder=2)
ax.add_patch(FancyArrowPatch((11.0, 82.0), (11.0, 70.5), arrowstyle="-|>",
             mutation_scale=7.5, color=SOFT, lw=1.15, zorder=2))
ax.text(49.25, 84.0, "next round", fontsize=5.8, color=psd_style.NOTE, ha="center", va="center")

# ---- 下排人工分支 ----
seg("assign", "al")
ax.text(13.2, 53.0, r"$\kappa < \tau$", fontsize=5.8, color=psd_style.NOTE, ha="left", va="center")
seg("al", "verified")
seg("verified", (HUB_X - HUB_W/2 - 0.3, HUB_Y), ls="solid", lw=1.15, a_box=True, b_box=False)

# ---- hub + 写回辐条(虚线) ----
ax.add_patch(FancyBboxPatch((HUB_X-HUB_W/2, HUB_Y-HUB_H/2), HUB_W, HUB_H,
                            boxstyle="round,pad=0.12,rounding_size=1.0",
                            facecolor=INK, edgecolor=INK, linewidth=1.0, zorder=6))
ax.text(HUB_X, HUB_Y+2.8, "Prototypes $P$", ha="center", va="center", fontsize=5.6,
        color=PAPER, fontweight="bold", zorder=7)
ax.text(HUB_X, HUB_Y+0.0, "classifier $\\Omega$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=7)
ax.text(HUB_X, HUB_Y-2.8, "anchors $A$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=7)
for src in ("upd", "reest"):
    c = S[src]
    st = (c[0], c[1]-ST_H/2-0.2)
    tgt = edge_pt((HUB_X, HUB_Y), c, HUB_W/2, HUB_H/2)
    ax.add_patch(FancyArrowPatch(st, tgt, arrowstyle="-|>", mutation_scale=7,
                 color=SOFT, lw=0.9, linestyle=(0, (4, 3)), zorder=2))
ax.text(63.6, 55.5, "write-back", fontsize=5.4, color=SOFT, ha="left", va="center")

# ---- 站盒(最后画, 盖住线头) ----
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    fill = FOCAL_FILL if k == FOCAL else (GRAY_FILL if manual else ACCENT_TINT)
    edge = GRAY_EDGE if manual else ACCENT
    wk = 21.5 if k == "assign" else ST_W
    ax.add_patch(FancyBboxPatch((xk-wk/2, yk-ST_H/2), wk, ST_H,
                                boxstyle="round,pad=0.1,rounding_size=0.8",
                                facecolor=fill, edgecolor=edge,
                                linewidth=1.4 if k == FOCAL else 0.9, zorder=4))
    ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
            color=C_TEXT, fontweight="bold" if k == FOCAL else "normal", zorder=5)
    ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
            color=psd_style.NOTE, zorder=5)

# ---- 图例(底部左角) ----
LX = 2.0; LY = 24.0
for i, (ec, fc, txt) in enumerate([
        (ACCENT, ACCENT_TINT, "automated step"),
        (GRAY_EDGE, GRAY_FILL, "human-in-the-loop step")]):
    y = LY + (1-i)*5.5
    ax.add_patch(FancyBboxPatch((LX, y-1.4), 4.6, 2.8, boxstyle="round,pad=0.05,rounding_size=0.3",
                                facecolor=fc, edgecolor=ec, lw=0.9, zorder=8))
    ax.text(LX+6.4, y, txt, fontsize=5.6, color=C_TEXT, va="center")
ax.plot([62, 67.5], [24.6, 24.6], color=SOFT, lw=0.9, linestyle=(0, (4, 3)))
ax.text(69.0, 24.6, "write-back to state", fontsize=5.6, color=C_TEXT, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig2 v7 (two-row pipeline, no ring crossings) saved")
