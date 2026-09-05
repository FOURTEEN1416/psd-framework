# -*- coding: utf-8 -*-
r"""fig2 v5 — 伪标签飞轮（Loop 规范, 期刊印刷尺寸设计版, 2026-09-05）。

v3 修复继承: 6 站环形 + 深色 hub(累积状态) + 虚线向心辐条 + focal 站。
v5 新增: ①按最终印刷尺寸设计 figsize=(3.42,3.05)in（v3 落地字号 2.0pt 不可读）;
②人工侧改中性灰（青保留给 fig1 物理层, 跨图语义统一）; ③辐条几何经计算可见
（对角可见段长 8.8 数据单位 ≈ 0.3in）。标签缩短, 细节入 caption。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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

N = 6
CX, CY = 50.0, 42.0
R = 25.0
ST_W, ST_H = 23.0, 7.4
HUB_W, HUB_H = 18.5, 9.8

STATIONS = [
    # (name, sublabel, spoke_label or None, focal, manual)
    ("Assign proposals", "conf. $\\kappa$", None, False, False),
    ("Label pool", "$\\kappa \\geq \\tau$", None, True, False),
    ("Update $\\Omega$", "seeds $\\cup$ pool", "$\\Omega$", False, False),
    ("Re-estimate $P$", "each round", "$P$", False, False),
    ("AL queue", "human, $\\leq$B clips", None, False, True),
    ("Verified seeds", "queue output", "$A$", False, True),
]

fig, ax = plt.subplots(figsize=(3.42, 3.05))
ax.set_xlim(0, 100); ax.set_ylim(0, 84)
ax.axis("off")

centers = []
for k in range(N):
    theta = np.deg2rad(-90 + k * 360.0 / N)
    centers.append((CX + R * np.cos(theta), CY + R * np.sin(theta)))

# ---- 环形圆弧（顺时针实线, 站盒之间） ----
for k in range(N):
    p0, p1 = centers[k], centers[(k + 1) % N]
    a0 = np.rad2deg(np.arctan2(p0[1] - CY, p0[0] - CX))
    a1 = np.rad2deg(np.arctan2(p1[1] - CY, p1[0] - CX))
    if a1 < a0:
        a1 += 360
    pad = 14.5
    arc = FancyBboxPatch((0, 0), 0.1, 0.1)  # placeholder unused
    from matplotlib.patches import Arc, FancyArrowPatch
    t0, t1 = a0 + pad, a1 - pad
    mid = (t0 + t1) / 2
    # 弧 + 末端箭头
    ax.add_patch(Arc((CX, CY), 2 * R, 2 * R, theta1=t0, theta2=t1,
                     color=SOFT, lw=1.1, zorder=1))
    x1 = CX + R * np.cos(np.deg2rad(t1)); y1 = CY + R * np.sin(np.deg2rad(t1))
    xt = CX + R * np.cos(np.deg2rad(t1 - 4)); yt = CY + R * np.sin(np.deg2rad(t1 - 4))
    ax.add_patch(FancyArrowPatch((xt, yt), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=7, color=SOFT, lw=1.1, zorder=1))
    if k == 0:   # Assign -> Pool 弧: 高置信路由
        am = np.deg2rad((t0 + t1) / 2)
        ax.text(CX + (R + 4.2) * np.cos(am), CY + (R + 4.2) * np.sin(am),
                "$\\kappa \geq \\tau$", fontsize=5.8, color="#4B5563", ha="center", va="center")
    if k == 3:   # Re-estimate -> AL queue 弧: 低置信路由
        am = np.deg2rad((t0 + t1) / 2)
        ax.text(CX + (R + 4.2) * np.cos(am), CY + (R + 4.2) * np.sin(am),
                "$\\kappa < \\tau$", fontsize=5.8, color="#4B5563", ha="center", va="center")

# ---- write-back 辐条（真半径虚线向心） ----
HUB_HALF = (HUB_W / 2, HUB_H / 2)
ST_HALF = (ST_W / 2, ST_H / 2)
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
    start = (cxk - u[0] * d_st, cyk - u[1] * d_st)
    gap = 1.0
    end = (CX + u[0] * (d_hub + gap), CY + u[1] * (d_hub + gap))
    if spoke_label:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=SOFT, lw=0.9, linestyle=(0, (4, 3))))
    # R18 judge 三轮：浮动 Ω/P/A 辐条标签反复与站盒角碰撞，且与 hub 文字、
    # 站名（Update Ω / Re-estimate P）冗余——按 diagram-design 删除优先原则移除文字，保留写回箭头。

# ---- hub（累积状态） ----
ax.add_patch(FancyBboxPatch((CX - HUB_W / 2, CY - HUB_H / 2), HUB_W, HUB_H,
                            boxstyle="round,pad=0.12,rounding_size=1.0",
                            facecolor=INK, edgecolor=INK, linewidth=1.0, zorder=6))
ax.text(CX, CY + 2.7, "Prototypes $P$", ha="center", va="center", fontsize=5.7,
        color=PAPER, fontweight="bold", zorder=7)
ax.text(CX, CY + 0.0, "classifier $\\Omega$", ha="center", va="center", fontsize=5.7,
        color=PAPER, zorder=7)
ax.text(CX, CY - 2.7, "anchors $A$", ha="center", va="center", fontsize=5.7,
        color=PAPER, zorder=7)

# ---- 站盒 ----
for k, (name, sub, spoke_label, focal, manual) in enumerate(STATIONS):
    cxk, cyk = centers[k]
    fill = ACCENT_TINT if focal else (GRAY_FILL if manual else ACCENT_TINT)
    edge = ACCENT if focal else (GRAY_EDGE if manual else ACCENT)
    if focal:
        fill, edge = "#FFD9C7", ACCENT
    ax.add_patch(FancyBboxPatch((cxk - ST_W / 2, cyk - ST_H / 2), ST_W, ST_H,
                                boxstyle="round,pad=0.1,rounding_size=0.8",
                                facecolor=fill, edgecolor=edge,
                                linewidth=1.4 if focal else 0.9, zorder=4))
    ax.text(cxk, cyk + 1.0, name, ha="center", va="center", fontsize=6.0,
            color=C_TEXT, fontweight="bold" if focal else "normal", zorder=5)
    ax.text(cxk, cyk - 1.7, sub, ha="center", va="center", fontsize=5.6,
            color="#4B5563", zorder=5)

# ---- 图例 ----
ax.plot([2, 6], [80.5, 80.5], color=GRAY_EDGE, lw=1.6)
ax.text(7.5, 80.5, "human step", fontsize=5.8, color=C_TEXT, va="center")
ax.plot([2, 6], [76.5, 76.5], color=ACCENT, lw=1.6)
ax.text(7.5, 76.5, "automated step", fontsize=5.8, color=C_TEXT, va="center")
ax.plot([2, 6], [72.5, 72.5], color=SOFT, lw=1.0, linestyle=(0, (4, 3)))
ax.text(7.5, 72.5, "write-back to state", fontsize=5.8, color=C_TEXT, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig2_pseudo_label_loop.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig2 v5 (print-size, Loop spec) saved")
