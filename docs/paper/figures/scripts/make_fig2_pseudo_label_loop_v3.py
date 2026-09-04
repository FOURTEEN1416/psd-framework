# -*- coding: utf-8 -*-
r"""fig2 v3 — 语义层迭代闭环（diagram-design Loop/flywheel 类型规范重绘, 2026-09-04）.

规范: diagram-design references/type-loop.md —
  环形主流程(实线圆弧, 顺时针) + 中心 hub(深色填充=累积状态: prototypes P / classifier Ω)
  + 虚线 write-back 辐条(向心) + 至多1个 focal station(κ≥τ 分叉判断).
几何: §2 参数化公式(等角 360/N, 环半径 R, 圆弧 A R R 0 0 1, 辐条真半径不交叉).
色板: 对齐论文 fig1(青=人工/种子侧, 橙=自动学习侧) + hub 深墨.
预算: 6 stations + 1 hub(规范 5-8+1).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

# ---- token（对齐 fig1 色板; hub 用深墨=#1F2937） ----
PAPER = "#FFFFFF"
INK = "#1F2937"
MUTED = "#6B7280"       # 环形流
SOFT = "#9CA3AF"        # write-back 辐条
ACCENT = "#C2410C"      # focal（分叉判断）
ACCENT_TINT = "#FFE3DA"
C_TEXT = "#111111"
C_PHYS_EDGE = "#6B7280"  # 中性灰描边（人工环节; 青保留给 fig1 物理层, 跨图语义不冲突）
C_PHYS_FILL = "#F3F4F6"
C_SEM_FILL = "#FFE3DA"
C_SEM_EDGE = "#C2410C"

# ---- 几何参数（type-loop §2） ----
N = 6
CX, CY = 50.0, 33.0
R = 25.0
ST_W, ST_H = 21.0, 8.6
HUB_W, HUB_H = 17.0, 9.5
MARKER_OVERRHANG = 0.6

STATIONS = [
    # (name, sublabel, spoke_label or None, focal, manual)
    ("Assign proposals", "nearest prototype, conf. $\\kappa$", None, True,  False),
    ("Pseudo-labeled pool", "$\\kappa \\geq \\tau$ join training", "HIGH-CONF", False, False),
    ("Update classifier $\\Omega$", "seeds $\\cup$ pool", "POOL", False, False),
    ("Re-estimate prototypes", "each round", "RE-EST.", False, False),
    ("Active-learning queue", "human annotation, budget 100-200", "LOW-CONF $\\kappa<\\tau$", False, True),
    ("New verified seeds", "queue output", "VERIFIED", False, True),
]

fig, ax = plt.subplots(figsize=(12.4, 9.0))
ax.set_xlim(0, 100); ax.set_ylim(0, 66)
ax.axis("off")

# ---- 环形圆弧（§2.2: 圆弧段, 站盒之间） ----
# 站盒中心
centers = []
for k in range(N):
    theta = np.deg2rad(-90 + k * 360.0 / N)
    centers.append((CX + R * np.cos(theta), CY + R * np.sin(theta)))

def box_intersect(cx, cy, r, bx, by, hw, hh):
    """圆与站盒边交点, 返回按极角排序的入口/出口点(type-loop §2.2)."""
    cands = []
    # 垂直边 x = bx ± hw
    for xe in (bx - hw, bx + hw):
        dx = xe - cx
        if abs(dx) <= r:
            dy = np.sqrt(max(r*r - dx*dx, 0.0))
            for ye in (by - hh, by + hh):  # 近似: 交点取边上 y 范围内
                pass
            for sign in (1, -1):
                y = cy + sign * dy
                if by - hh <= y <= by + hh:
                    cands.append((xe, y))
    # 水平边 y = by ± hh
    for ye in (by - hh, by + hh):
        dy = ye - cy
        if abs(dy) <= r:
            dx = np.sqrt(max(r*r - dy*dy, 0.0))
            for sign in (1, -1):
                x = cx + sign * dx
                if bx - hw <= x <= bx + hw:
                    cands.append((x, ye))
    # 去重并按极角排序
    uniq = []
    for p in cands:
        if not any(abs(p[0]-q[0]) < 1e-6 and abs(p[1]-q[1]) < 1e-6 for q in uniq):
            uniq.append(p)
    uniq.sort(key=lambda p: np.arctan2(p[1]-cy, p[0]-cx))
    return uniq

# 画环形流: 从 station k 的出口到 k+1 的入口（顺时针圆弧）
arc_segs = []
for k in range(N):
    j = (k + 1) % N
    bxk, byk = centers[k]
    bxj, byj = centers[j]
    int_k = box_intersect(CX, CY, R, bxk, byk, ST_W/2, ST_H/2)
    int_j = box_intersect(CX, CY, R, bxj, byj, ST_W/2, ST_H/2)
    if len(int_k) < 2 or len(int_j) < 2:
        continue
    # 出口 = k 盒上极角"刚好在 theta_k 顺时针之后"的交点; 入口 = j 盒上"刚好在 theta_j 之前"
    th_k = np.arctan2(byk - CY, bxk - CX)
    th_j = np.arctan2(byj - CY, bxj - CX)
    def after(theta0, p):
        d = (np.arctan2(p[1]-CY, p[0]-CX) - theta0) % (2*np.pi)
        return d
    exit_p = min(int_k, key=lambda p: after(th_k, p))
    entry_p = min(int_j, key=lambda p: after(th_j, p))
    arc_segs.append((exit_p, entry_p))

for (exit_p, entry_p) in arc_segs:
    a0 = np.arctan2(exit_p[1]-CY, exit_p[0]-CX)
    a1 = np.arctan2(entry_p[1]-CY, entry_p[0]-CX)
    da = (a1 - a0) % (2*np.pi)
    ts = np.linspace(a0, a0 + da, 24)
    xs = CX + R * np.cos(ts); ys = CY + R * np.sin(ts)
    ax.plot(xs[:-2], ys[:-2], color=MUTED, lw=1.7, solid_capstyle="butt", zorder=2)
    # 末段带箭头
    ax.annotate("", xy=(xs[-1], ys[-1]), xytext=(xs[-3], ys[-3]),
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.7), zorder=2)

# ---- write-back 辐条（§2.3: 真半径虚线向心, 止于 hub 外 marker_gap） ----
HUB_HALF = (HUB_W/2, HUB_H/2)
ST_HALF = (ST_W/2, ST_H/2)
def box_distance(v, hw, hh):
    terms = []
    if abs(v[0]) > 1e-9: terms.append(hw / abs(v[0]))
    if abs(v[1]) > 1e-9: terms.append(hh / abs(v[1]))
    return min(terms)

for k, (name, sub, spoke_label, focal, manual) in enumerate(STATIONS):
    cxk, cyk = centers[k]
    u = np.array([cxk - CX, cyk - CY]); u = u / np.linalg.norm(u)
    d_st = box_distance(u, *ST_HALF)
    d_hub = box_distance(u, *HUB_HALF)
    start = (cxk - u[0]*d_st, cyk - u[1]*d_st)
    gap = 1.2
    end = (CX + u[0]*(d_hub + gap), CY + u[1]*(d_hub + gap))
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=1.3, linestyle=(0, (5, 4))))
    if spoke_label:
        lm = start[0] * 0.45 + end[0] * 0.55 + u[1] * 2.8
        lm_y = start[1] * 0.45 + end[1] * 0.55 - u[0] * 2.8
        ax.text(lm, lm_y, spoke_label, fontsize=7.0, color="#6B7280", ha="center", va="center",
                fontweight="bold", zorder=8,
                bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.4))

# ---- hub（中心: 累积状态） ----
ax.add_patch(FancyBboxPatch((CX - HUB_W/2, CY - HUB_H/2), HUB_W, HUB_H,
                            boxstyle="round,pad=0.15,rounding_size=1.2",
                            facecolor=INK, edgecolor=INK, linewidth=1.5))
ax.text(CX, CY + 1.6, "Prototypes $P$ +", ha="center", va="center", fontsize=8.6,
        color=PAPER, fontweight="bold")
ax.text(CX, CY - 1.8, "classifier $\\Omega$  (state)", ha="center", va="center", fontsize=8.6,
        color=PAPER)

# ---- 站盒（最后画, 盖住弧线溢出） ----
for k, (name, sub, spoke_label, focal, manual) in enumerate(STATIONS):
    cxk, cyk = centers[k]
    fill = ACCENT_TINT if focal else (C_PHYS_FILL if manual else C_SEM_FILL)
    edge = "#C2410C" if focal else (C_PHYS_EDGE if manual else C_SEM_EDGE)
    ax.add_patch(FancyBboxPatch((cxk - ST_W/2, cyk - ST_H/2), ST_W, ST_H,
                                boxstyle="round,pad=0.12,rounding_size=1.0",
                                facecolor=fill, edgecolor=edge, linewidth=1.8 if focal else 1.4, zorder=4))
    ax.text(cxk, cyk + 1.1, name, ha="center", va="center", fontsize=8.6,
            color=ACCENT if focal else C_TEXT, fontweight="bold" if focal else "normal", zorder=5)
    ax.text(cxk, cyk - 2.0, sub, ha="center", va="center", fontsize=7.2,
            color="#4B5563", zorder=5)

# ---- 图例（侧注: 颜色语义） ----
ax.plot([1.5, 4.5], [64.0, 64.0], color=C_PHYS_EDGE, lw=2.0)
ax.text(5.2, 64.0, "human-in-the-loop step", fontsize=7.4, color=C_TEXT, va="center")
ax.plot([1.5, 4.5], [61.8, 61.8], color="#C2410C", lw=2.0)
ax.text(5.2, 61.8, "automated learning step", fontsize=7.4, color=C_TEXT, va="center")
ax.plot([1.5, 4.5], [59.6, 59.6], color=SOFT, lw=1.4, linestyle=(0, (5, 4)))
ax.text(5.2, 59.6, "write-back to shared state", fontsize=7.4, color=C_TEXT, va="center")

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight")
print("fig2 v3 (Loop spec) saved")
