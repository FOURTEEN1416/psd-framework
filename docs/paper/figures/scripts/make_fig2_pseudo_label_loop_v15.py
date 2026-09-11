# -*- coding: utf-8 -*-
r"""fig2 v15 — 伪标签循环双行管线（R43 全图重绘 · 数模竞赛·科研工具箱/diagram-design 主参考）。
automated=深橙系，human=青绿系，hub=墨黑（语义锁承 psd_style）。

相对 v14 的重绘点（拓扑不变，caption 的 top/bottom row 措辞保真）：
1. write-back 两条对角虚线改正交圆角肘线：Update Ω 垂直下入 hub 顶；
   Re-estimate P 垂直下行→肘弯左行入 hub 右缘（diagram-design 连线可追踪性）；
2. "next round" 回路轨道从 y=82 压至 y=69.5 贴盒（画布高度 2.95→2.35in，
   顶部死区回收，单位物理尺度不变）；
3. 图例重构：底部单行三键（automated / human-in-the-loop / write-back to state），
   删 v14 图内+图例双处 write-back 重复（spokes 旁保留一处标注）；
4. 六个站盒宽度统一 19.5（v14 Assign 独宽 21.5）。
"""
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
HUMAN = psd_style.HUMAN_DEEP           # human 青绿深化
HUMAN_MID = psd_style.HUMAN_EDGE       # human 描边(板内青绿原色)
SOFT = "#555555"                        # 流线深灰

fig, ax = plt.subplots(figsize=(3.42, 2.35))
ax.set_xlim(0, 100); ax.set_ylim(25, 74)
ax.axis("off")

ST_W, ST_H = 19.5, 8.2
HUB_X, HUB_Y = 61.0, 40.0
HUB_W, HUB_H = 17.0, 10.0

ROW1_Y = 62.0
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
    st = edge_pt(ca, cb) if a_box else a
    en = edge_pt(cb, ca) if b_box else b
    psd_style.flowline(ax, st[0], st[1], en[0], en[1], color=color, lw=lw,
                       ls=ls, head=10, z=2)
    if label:
        mx = st[0]+(en[0]-st[0])*lp; my = st[1]+(en[1]-st[1])*lp+ldy
        ax.text(mx, my, label, fontsize=5.8, color="#444444", ha="center", va="center", zorder=5)

# ---- 上排自动环(线性) + 贴盒回路轨道 ----
seg("assign", "pool", label=r"$\kappa \geq \tau$", ldy=2.6)
seg("pool", "upd")
seg("upd", "reest")
# 三段贴盒折线回环: reest 顶 -> 上 -> 左 -> assign 顶(箭头)
RAIL_Y = 69.5
ax.plot([87.5, 87.5], [66.3, RAIL_Y], color=SOFT, lw=1.6, zorder=2, solid_capstyle="round")
ax.plot([87.5, 11.0], [RAIL_Y, RAIL_Y], color=SOFT, lw=1.6, zorder=2, solid_capstyle="round")
ax.add_patch(FancyArrowPatch((11.0, RAIL_Y), (11.0, 66.5), arrowstyle="-|>",
             mutation_scale=10, color=SOFT, lw=1.6, zorder=2))
ax.text(49.25, 71.6, "next round", fontsize=5.8, color="#444444", ha="center", va="center")

# ---- 下排人工分支 ----
seg("assign", "al")
ax.text(13.4, 51.0, r"$\kappa < \tau$", fontsize=5.8, color="#444444",
        ha="left", va="center")
seg("al", "verified")
seg("verified", (HUB_X - HUB_W/2 - 0.3, HUB_Y), a_box=True, b_box=False)

# ---- write-back 正交肘线（虚流线）: upd 垂直下入 hub 顶; reest 垂下→左肘入 hub 右缘 ----
psd_style.flowline(ax, 62.0, 57.7, 62.0, HUB_Y + HUB_H/2 + 0.3, color="#777777",
                   lw=1.1, ls=(0, (4, 3)), head=8, z=2)
ax.plot([87.5, 87.5], [57.7, HUB_Y], color="#777777", lw=1.1, ls=(0, (4, 3)),
        zorder=2, solid_capstyle="round")
psd_style.flowline(ax, 87.5, HUB_Y, HUB_X + HUB_W/2 + 0.3, HUB_Y, color="#777777",
                   lw=1.1, ls=(0, (4, 3)), head=8, z=2)
ax.text(64.0, 51.5, "write-back", fontsize=5.4, color="#666666", ha="left", va="center")

# ---- hub(墨黑满块白字, 扁平) ----
psd_style.card(ax, HUB_X-HUB_W/2, HUB_Y-HUB_H/2, HUB_W, HUB_H, fill=INK, edge=None,
               rounding=1.2, z=6)
ax.text(HUB_X, HUB_Y+2.8, "Prototypes $P$", ha="center", va="center", fontsize=5.6,
        color=PAPER, fontweight="bold", zorder=8)
ax.text(HUB_X, HUB_Y+0.0, "classifier $\\Omega$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=8)
ax.text(HUB_X, HUB_Y-2.8, "anchors $A$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=8)

# ---- 站盒(白卡+家族色细描边; 焦点=深橙满块白字) ----
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    deep = HUMAN if manual else ACCENT
    mid = HUMAN_MID if manual else ACCENT_MID
    if k == FOCAL:
        psd_style.card(ax, xk-ST_W/2, yk-ST_H/2, ST_W, ST_H, fill=ACCENT, edge=None,
                       rounding=1.2, z=4)
        ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
                color="white", fontweight="bold", zorder=6)
        ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
                color="white", alpha=0.9, zorder=6)
    else:
        psd_style.card(ax, xk-ST_W/2, yk-ST_H/2, ST_W, ST_H, fill="white", edge=mid,
                       lw=0.9, rounding=1.2, z=4)
        ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
                color=deep, fontweight="bold", zorder=6)
        ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
                color="#555555", zorder=6)

# ---- 图例(底部单行三键) ----
LY = 30.0
psd_style.card(ax, 2.5, LY-1.3, 4.6, 2.6, fill="white", edge=ACCENT_MID, lw=1.0,
               rounding=0.5, z=8)
ax.text(8.9, LY, "automated step", fontsize=5.6, color=INK, va="center")
psd_style.card(ax, 26.0, LY-1.3, 4.6, 2.6, fill="white", edge=HUMAN_MID, lw=1.0,
               rounding=0.5, z=8)
ax.text(32.4, LY, "human-in-the-loop step", fontsize=5.6, color=INK, va="center")
ax.plot([62.0, 67.5], [LY, LY], color="#777777", lw=1.1, linestyle=(0, (4, 3)))
ax.text(69.0, LY, "write-back to state", fontsize=5.6, color=INK, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig2 v15 (redraw: orthogonal write-back elbows + compact rail + one-row legend) saved")
