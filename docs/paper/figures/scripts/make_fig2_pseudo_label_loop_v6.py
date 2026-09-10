# -*- coding: utf-8 -*-
r"""fig2 v6 — 伪标签飞轮（拓扑修正版, 2026-09-10 R24 图表专项）。

v5 judge 实锤问题: κ<τ 低置信分叉被画在 Re-estimate→AL queue 弧上——分叉语义
只存在于 Assign proposals 判定节点, re-estimate 站不产生待分诊 proposal。
v6 结构: 自动环(左, 逆时针 4 站: Assign→Pool→UpdateΩ→Re-est→Assign) +
人工分支(右, 2 站链: Assign -κ<τ-> AL queue -> Verified seeds -写回 A-> hub)。
继承 v5: 印刷尺寸 figsize、人工灰/自动橙、hub 深色、删除冗余辐条文字。
输出: fig2_pseudo_label_loop.pdf/png(600dpi)。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

PAPER = "#FFFFFF"
INK = "#1F2937"
ACCENT = "#C2410C"          # 橙 = 自动学习步
ACCENT_TINT = "#FFE3DA"
GRAY_EDGE = "#6B7280"       # 灰 = 人工环节
GRAY_FILL = "#F3F4F6"
SOFT = "#9CA3AF"
C_TEXT = "#111111"

fig, ax = plt.subplots(figsize=(3.42, 3.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 84)
ax.axis("off")

ST_W, ST_H = 23.0, 7.4
HUB_X, HUB_Y = 50.0, 41.5
HUB_W, HUB_H = 18.5, 9.8

# 站坐标: 左环 4 站(逆时针自动环), 右链 2 站(人工分支)
S = {
    "assign":   (50.0, 68.0),   # 顶
    "pool":     (21.5, 55.0),
    "upd":      (21.5, 30.0),
    "reest":    (50.0, 17.0),   # 底
    "al":       (78.5, 55.0),
    "verified": (78.5, 30.0),
}
FOCAL = "pool"
MANUAL = {"al", "verified"}
SUBS = {
    "assign": "conf. $\\kappa$", "pool": "accepted proposals",
    "upd": "seeds $\\cup$ pool", "reest": "each round",
    "al": "human, $\\leq$B clips", "verified": "queue output",
}
NAMES = {
    "assign": "Assign proposals", "pool": "Label pool",
    "upd": "Update $\\Omega$", "reest": "Re-estimate $P$",
    "al": "AL queue", "verified": "Verified seeds",
}

def edge_point(c0, c1, hw=ST_W/2, hh=ST_H/2):
    """从盒中心 c0 朝 c1 方向的盒缘出射点。"""
    v = np.array([c1[0]-c0[0], c1[1]-c0[1]], float)
    L = np.linalg.norm(v); v /= L
    t = np.inf
    if abs(v[0]) > 1e-9: t = min(t, hw/abs(v[0]))
    if abs(v[1]) > 1e-9: t = min(t, hh/abs(v[1]))
    return (c0[0]+v[0]*t, c0[1]+v[1]*t)

def arrow(a, b, label=None, lx=0, ly=0, ls="solid", lw=1.1, color=SOFT, curve=0.0):
    st = edge_point(S[a], S[b] if isinstance(b, str) else b) if isinstance(a, str) else a
    en = edge_point(S[b], S[a] if isinstance(a, str) else a) if isinstance(b, str) else b
    ap = dict(arrowstyle="-|>", mutation_scale=7, color=color, lw=lw, linestyle=ls)
    if curve:
        ap["connectionstyle"] = f"arc3,rad={curve}"
    ax.add_patch(FancyArrowPatch(st, en, **ap, zorder=2))
    if label:
        mx = (st[0]+en[0])/2 + lx; my = (st[1]+en[1])/2 + ly
        ax.text(mx, my, label, fontsize=5.8, color="#4B5563", ha="center", va="center")

# ---- 自动环(左, 逆时针) ----
arrow("assign", "pool",  label=r"$\kappa \geq \tau$", lx=-3.5, ly=1.5, curve=-0.18)
arrow("pool", "upd", curve=-0.18)
arrow("upd", "reest", curve=-0.18)
arrow("reest", (7.0, 42.5)); arrow((7.0, 42.5), "assign", label="next round", lx=-2.0, ly=0)
# ---- 人工分支(右): 分叉在 Assign 判定节点 ----
arrow("assign", "al", label=r"$\kappa < \tau$", lx=3.5, ly=1.5, curve=0.18)
arrow("al", "verified", curve=0.18)
# verified seeds 写回 hub 的 A (向心虚线)
st = edge_point(S["verified"], (HUB_X, HUB_Y))
d = np.array([HUB_X-st[0], HUB_Y-st[1]], float); L=np.linalg.norm(d); d/=L
hv = HUB_W/2 if abs(d[0]*L) > 0 else 0
# hub 边缘点(盒形)
t = np.inf
if abs(d[0]) > 1e-9: t = min(t, (HUB_W/2+1.0)/abs(d[0]))
if abs(d[1]) > 1e-9: t = min(t, (HUB_H/2+1.0)/abs(d[1]))
en = (HUB_X - d[0]*t, HUB_Y - d[1]*t)
ax.add_patch(FancyArrowPatch(st, en, arrowstyle="-|>", mutation_scale=7,
             color=SOFT, lw=0.9, linestyle=(0, (4, 3)), zorder=2))
# ---- 自动环写回辐条: UpdateΩ 和 Re-estimate P 向 hub 虚线 ----
for a in ("upd", "reest"):
    st = edge_point(S[a], (HUB_X, HUB_Y))
    d = np.array([HUB_X-st[0], HUB_Y-st[1]], float); L=np.linalg.norm(d); d/=L
    t = np.inf
    if abs(d[0]) > 1e-9: t = min(t, (HUB_W/2+1.0)/abs(d[0]))
    if abs(d[1]) > 1e-9: t = min(t, (HUB_H/2+1.0)/abs(d[1]))
    en = (HUB_X - d[0]*t, HUB_Y - d[1]*t)
    ax.add_patch(FancyArrowPatch(st, en, arrowstyle="-|>", mutation_scale=7,
                 color=SOFT, lw=0.9, linestyle=(0, (4, 3)), zorder=2))

# ---- hub(累积状态) ----
ax.add_patch(FancyBboxPatch((HUB_X-HUB_W/2, HUB_Y-HUB_H/2), HUB_W, HUB_H,
                            boxstyle="round,pad=0.12,rounding_size=1.0",
                            facecolor=INK, edgecolor=INK, linewidth=1.0, zorder=6))
ax.text(HUB_X, HUB_Y+2.7, "Prototypes $P$", ha="center", va="center", fontsize=5.7,
        color=PAPER, fontweight="bold", zorder=7)
ax.text(HUB_X, HUB_Y+0.0, "classifier $\\Omega$", ha="center", va="center", fontsize=5.7,
        color=PAPER, zorder=7)
ax.text(HUB_X, HUB_Y-2.7, "anchors $A$", ha="center", va="center", fontsize=5.7,
        color=PAPER, zorder=7)

# ---- 站盒 ----
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    fill = "#FFD9C7" if k == FOCAL else (GRAY_FILL if manual else ACCENT_TINT)
    edge = ACCENT if not manual else GRAY_EDGE
    ax.add_patch(FancyBboxPatch((xk-ST_W/2, yk-ST_H/2), ST_W, ST_H,
                                boxstyle="round,pad=0.1,rounding_size=0.8",
                                facecolor=fill, edgecolor=edge,
                                linewidth=1.4 if k == FOCAL else 0.9, zorder=4))
    ax.text(xk, yk+1.0, NAMES[k], ha="center", va="center", fontsize=6.0,
            color=C_TEXT, fontweight="bold" if k == FOCAL else "normal", zorder=5)
    ax.text(xk, yk-1.7, SUBS[k], ha="center", va="center", fontsize=5.6,
            color="#4B5563", zorder=5)

# ---- 图例(描边小方块, 对应盒描边; 线型样本单独一组) ----
LX = 3.0
for i, (ec, fc, txt) in enumerate([
        (ACCENT, ACCENT_TINT, "automated step"),
        (GRAY_EDGE, GRAY_FILL, "human-in-the-loop step")]):
    y = 80.0 - i*4.2
    ax.add_patch(FancyBboxPatch((LX, y-1.1), 5.2, 2.2, boxstyle="round,pad=0.05,rounding_size=0.3",
                                facecolor=fc, edgecolor=ec, lw=1.0, zorder=8))
    ax.text(LX+7.0, y, txt, fontsize=5.8, color=C_TEXT, va="center")
ax.plot([LX, LX+5.2], [71.6, 71.6], color=SOFT, lw=0.9, linestyle=(0, (4, 3)))
ax.text(LX+7.0, 71.6, "write-back to state", fontsize=5.8, color=C_TEXT, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig2 v7 (topology-corrected dual-loop flywheel) saved")
