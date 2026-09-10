# -*- coding: utf-8 -*-
r"""fig2 v13 — 伪标签循环双行管线（R30 海洋清风编辑级：白卡加家族色细描边+
无阴影+细流线；拓扑承 v11/v12 不变）。automated=深橙系，human=青绿系，hub=墨黑。"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
PAPER = "#FFFFFF"
INK = psd_style.INK
ACCENT = psd_style.SEM_DEEP            # automated 深橙(深化字色)
ACCENT_MID = psd_style.SEM_EDGE        # automated 描边(板内深橙原色)
ACCENT_TINT = psd_style.SEM_TINT       # 焦点块柔橙填充
HUMAN = psd_style.HUMAN_DEEP           # human 青绿深化
HUMAN_MID = psd_style.HUMAN_EDGE       # human 描边(板内青绿原色)
SOFT = "#555555"                        # 流线深灰

fig, ax = plt.subplots(figsize=(3.42, 2.95))
ax.set_xlim(0, 100); ax.set_ylim(17.5, 90)
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

def seg(a, b, label=None, lp=0.5, ldy=2.2, ls="solid", lw=1.6, color=SOFT,
        a_box=True, b_box=True):
    ca = S[a] if a_box else a
    cb = S[b] if b_box else b
    kw_a = dict(hw=ST_W/2, hh=ST_H/2)
    st = edge_pt(ca, cb, **kw_a) if a_box else a
    en = edge_pt(cb, ca, **kw_a) if b_box else b
    psd_style.flowline(ax, st[0], st[1], en[0], en[1], color=color, lw=lw,
                       ls=ls, head=10, z=2)
    if label:
        mx = st[0]+(en[0]-st[0])*lp; my = st[1]+(en[1]-st[1])*lp+ldy
        ax.text(mx, my, label, fontsize=5.8, color="#444444", ha="center", va="center", zorder=5)

# ---- 上排自动环(线性) ----
seg("assign", "pool", label=r"$\kappa \geq \tau$", ldy=7.0)
seg("pool", "upd")
seg("upd", "reest")
# 三段折线回环: reest 顶 -> 上 -> 左 -> assign 顶(箭头)
ax.plot([87.5, 87.5], [70.3, 82.0], color=SOFT, lw=1.6, zorder=2, solid_capstyle="round")
ax.plot([87.5, 11.0], [82.0, 82.0], color=SOFT, lw=1.6, zorder=2, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((11.0, 82.0), (11.0, 70.5), arrowstyle="-|>",
             mutation_scale=10, color=SOFT, lw=1.6, zorder=2))
ax.text(49.25, 84.0, "next round", fontsize=5.8, color="#444444", ha="center", va="center")

# ---- 下排人工分支 ----
seg("assign", "al")
ax.text(13.2, 53.0, r"$\kappa < \tau$", fontsize=5.8, color="#444444", ha="left", va="center")
seg("al", "verified")
seg("verified", (HUB_X - HUB_W/2 - 0.3, HUB_Y), a_box=True, b_box=False)

# ---- hub(墨黑满块白字, 扁平) + 写回辐条(虚流线) ----
psd_style.card(ax, HUB_X-HUB_W/2, HUB_Y-HUB_H/2, HUB_W, HUB_H, fill=INK, edge=None,
               rounding=1.2, z=6)
ax.text(HUB_X, HUB_Y+2.8, "Prototypes $P$", ha="center", va="center", fontsize=5.6,
        color=PAPER, fontweight="bold", zorder=8)
ax.text(HUB_X, HUB_Y+0.0, "classifier $\\Omega$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=8)
ax.text(HUB_X, HUB_Y-2.8, "anchors $A$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=8)
for src in ("upd", "reest"):
    c = S[src]
    st = (c[0], c[1]-ST_H/2-0.2)
    tgt = edge_pt((HUB_X, HUB_Y), c, HUB_W/2, HUB_H/2)
    psd_style.flowline(ax, st[0], st[1], tgt[0], tgt[1], color="#777777",
                       lw=1.1, ls=(0, (4, 3)), head=8, z=2)
ax.text(63.6, 55.5, "write-back", fontsize=5.4, color="#666666", ha="left", va="center")

# ---- 站盒(白卡+家族色细描边 DEEP 层级文字; 焦点=深橙满块白字; 最后画盖线头) ----
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    deep = HUMAN if manual else ACCENT
    mid = HUMAN_MID if manual else ACCENT_MID
    wk = 21.5 if k == "assign" else ST_W
    if k == FOCAL:
        psd_style.card(ax, xk-wk/2, yk-ST_H/2, wk, ST_H, fill=ACCENT, edge=None,
                       rounding=1.2, z=4)
        ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
                color="white", fontweight="bold", zorder=6)
        ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
                color="white", alpha=0.9, zorder=6)
    else:
        psd_style.card(ax, xk-wk/2, yk-ST_H/2, wk, ST_H, fill="white", edge=mid,
                       lw=0.9, rounding=1.2, z=4)
        ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
                color=deep, fontweight="bold", zorder=6)
        ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
                color="#555555", zorder=6)

# ---- 图例(底部左角) ----
LX = 2.0; LY = 24.0
for i, (ec, txt) in enumerate([
        (ACCENT_MID, "automated step"),
        (HUMAN_MID, "human-in-the-loop step")]):
    y = LY + (1-i)*5.5
    psd_style.card(ax, LX, y-1.4, 4.6, 2.8, fill="white", edge=ec, lw=1.0,
                   rounding=0.5, z=8)
    ax.text(LX+6.4, y, txt, fontsize=5.6, color=INK, va="center")
ax.plot([62, 67.5], [24.6, 24.6], color="#777777", lw=1.1, linestyle=(0, (4, 3)))
ax.text(69.0, 24.6, "write-back to state", fontsize=5.6, color=INK, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig2 v13 (R30 ocean editorial: bordered white stations + flat focal/hub) saved")
