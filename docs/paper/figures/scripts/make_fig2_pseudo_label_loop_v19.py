# -*- coding: utf-8 -*-
r"""fig2 v19 — 伪标签循环双行管线（正典 · 风格方向 **C+ 强化版**）。

承 v18 全部拓扑（零交叉自动环 + 贴盒回路轨道 + 正交 write-back 肘线 +
κ≥τ 段标签置段下方），**图内全部文字逐字承 v18，零新增零删除零改写**。

C+ 相对 v18 的收紧项：
1. **笔触更细**：主流线 1.6 -> 1.2；write-back 虚线 1.1 -> 0.8；回路轨道
   1.6 -> 1.2；chip 圆环 0.8 -> 0.5；站卡描边 0.9 -> **去边**。
2. **字号阶梯更大**：站名 5.9 -> 6.2；hub 三行 5.6 -> 6.2；图例 5.6 -> 5.8；
   站注 5.5 -> 5.6；流注保持 5.8。下限仍 5.5pt。
3. **留白 +14%**：画布 2.36 -> 2.70in，坐标域 25..74 -> 22..78（56 单位，
   与原 20.76 单位/in 基本持平），两排垂直间距由 22 -> 26 单位。
4. **描边弱化**：站卡去 0.9pt 家族色描边，改为**仅靠色阶**（同相 mix 0.66，
   更浅一档）；图例键同样去边。
5. **焦点块由 2 减到 1**：v18 同时把 Label pool（SEM 满块）与 hub（INK 满块）
   标为焦点。**caption 明文 "The dark hub carries the shared state (P, Ω, A)"**
   ——hub 深块为 caption 保真所需而保留；Label pool 降为普通同相浅底卡，
   使全图唯一深块 = hub（视觉锚点与 caption 叙述一致）。
6. **caption 逐词保真**（03-method.tex \label{fig:pseudoloop}）：top row =
   automated(orange) / bottom row = human(teal) / dark hub / (P, Ω, A) 三行 /
   κ≥τ 上行与 κ<τ 下分 / next round 沿顶部回 / AL queue budget B /
   "the dashed write-back spokes" 虚线 / 图例三键——全部保持。

尺度修复保持：MediaBox 严格等于画布（246.24 × 194.40pt）。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
PAPER = "#FFFFFF"
INK = psd_style.INK
ACCENT = psd_style.SEM_DEEP
ACCENT_MID = psd_style.SEM_EDGE
HUMAN = psd_style.HUMAN_DEEP
HUMAN_MID = psd_style.HUMAN_EDGE
SOFT = "#555555"


def _mix(hex_color, f):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        round(r + (255 - r) * f), round(g + (255 - g) * f), round(b + (255 - b) * f))


# C+ 分阶色标：站卡更浅一档（mix 0.66）且无描边
A_CARD = _mix(ACCENT_MID, 0.66); H_CARD = _mix(HUMAN_MID, 0.66)

# ---- 画布：C+ 高度 +0.34in（2.36 -> 2.70） ----
FIG_W, FIG_H = 3.42, 2.70
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 100); ax.set_ylim(22, 78)
ax.axis("off")

ST_W, ST_H = 20.5, 8.6
HUB_X, HUB_Y = 62.0, 38.5
HUB_W, HUB_H = 18.0, 11.0

ROW1_Y = 64.5
S = {
    "assign":  (11.0, ROW1_Y),
    "pool":    (36.5, ROW1_Y),
    "upd":     (62.0, ROW1_Y),
    "reest":   (87.5, ROW1_Y),
    "al":      (11.0, 38.5),
    "verified":(36.5, 38.5),
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
MANUAL = {"al", "verified"}


def edge_pt(c, target, hw=ST_W/2, hh=ST_H/2):
    v = np.array([target[0]-c[0], target[1]-c[1]], float)
    L = np.linalg.norm(v); v = v/L
    t = np.inf
    if abs(v[0]) > 1e-9: t = min(t, hw/abs(v[0]))
    if abs(v[1]) > 1e-9: t = min(t, hh/abs(v[1]))
    return (c[0]+v[0]*t, c[1]+v[1]*t)


def seg(a, b, label=None, lp=0.5, ldy=2.4, lfs=5.8, ls="solid", lw=1.2,
        color=SOFT, a_box=True, b_box=True):
    ca = S[a] if a_box else a
    cb = S[b] if b_box else b
    st = edge_pt(ca, cb) if a_box else a
    en = edge_pt(cb, ca) if b_box else b
    psd_style.flowline(ax, st[0], st[1], en[0], en[1], color=color, lw=lw,
                       ls=ls, head=9, z=2)
    if label:
        mx = st[0]+(en[0]-st[0])*lp; my = st[1]+(en[1]-st[1])*lp+ldy
        ax.text(mx, my, label, fontsize=lfs, color="#444444", ha="center",
                va="center", zorder=5)


# ---- 上排自动环（线性）+ 贴盒回路轨道 ----
# C+ 净空修正：κ≥τ 由"段下方"（y-2.4 = 62.1，与两卡站注 62.5 同基线而连读成串）
# 改为"段上方"（y+2.6 = 67.1）——重排后该处为卡间空档，且 chip 圆环（x≈28）
# 与文字（x≈23.8）水平错开，无碰撞。
seg("assign", "pool", label=r"$\kappa \geq \tau$", ldy=+4.5, lfs=5.6)
seg("pool", "upd")
seg("upd", "reest")
RAIL_Y = 73.0
ax.plot([87.5, 87.5], [69.0, RAIL_Y], color=SOFT, lw=1.2, zorder=2,
        solid_capstyle="round")
ax.plot([87.5, 11.0], [RAIL_Y, RAIL_Y], color=SOFT, lw=1.2, zorder=2,
        solid_capstyle="round")
ax.add_patch(FancyArrowPatch((11.0, RAIL_Y), (11.0, 69.2), arrowstyle="-|>",
             mutation_scale=9, color=SOFT, lw=1.2, zorder=2))
ax.text(49.25, 75.2, "next round", fontsize=5.8, color="#444444",
        ha="center", va="center")

# ---- 下排人工分支 ----
seg("assign", "al")
ax.text(13.4, 51.5, r"$\kappa < \tau$", fontsize=5.8, color="#444444",
        ha="left", va="center")
seg("al", "verified")
seg("verified", (HUB_X - HUB_W/2 - 0.3, HUB_Y), a_box=True, b_box=False)

# ---- write-back 正交肘线（C+ 虚线 0.8） ----
psd_style.flowline(ax, 62.0, 60.0, 62.0, HUB_Y + HUB_H/2 + 0.3, color="#777777",
                   lw=0.8, ls=(0, (4, 3)), head=7, z=2)
ax.plot([87.5, 87.5], [60.0, HUB_Y], color="#777777", lw=0.8, ls=(0, (4, 3)),
        zorder=2, solid_capstyle="round")
psd_style.flowline(ax, 87.5, HUB_Y, HUB_X + HUB_W/2 + 0.3, HUB_Y, color="#777777",
                   lw=0.8, ls=(0, (4, 3)), head=7, z=2)
ax.text(65.0, 52.0, "write-back", fontsize=5.6, color="#666666", ha="left",
        va="center")

# ---- hub（INK 墨黑满块白字；**全图唯一焦点深块**，caption 明文 "dark hub"） ----
psd_style.card(ax, HUB_X-HUB_W/2, HUB_Y-HUB_H/2, HUB_W, HUB_H, fill=INK,
               edge=None, rounding=1.6, z=6)
ax.text(HUB_X, HUB_Y+3.2, "Prototypes $P$", ha="center", va="center",
        fontsize=6.2, color=PAPER, fontweight="bold", zorder=8)
ax.text(HUB_X, HUB_Y+0.0, "classifier $\\Omega$", ha="center", va="center",
        fontsize=6.2, color=PAPER, zorder=8)
ax.text(HUB_X, HUB_Y-3.2, "anchors $A$", ha="center", va="center",
        fontsize=6.2, color=PAPER, zorder=8)

# ---- 站盒（C+：同相浅底、**无描边**；全部常规卡，焦点只留 hub） ----
name_txt, sub_txt = {}, {}
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    deep = HUMAN if manual else ACCENT
    fill = H_CARD if manual else A_CARD
    psd_style.card(ax, xk-ST_W/2, yk-ST_H/2, ST_W, ST_H, fill=fill, edge=None,
                   rounding=1.6, z=4)
    name_txt[k] = ax.text(xk, yk+1.4, NAMES[k], ha="center", va="center",
                          fontsize=6.0, color=deep, fontweight="normal", zorder=6)
    sub_txt[k] = ax.text(xk, yk-2.1, SUBS[k], ha="center", va="center",
                         fontsize=5.6, color="#555555", zorder=6)

# ---- 自动环步骤号 chip ①-④（角标；C+ 环 0.5pt） ----
for i, k in enumerate(("assign", "pool", "upd", "reest"), 1):
    xk, yk = S[k]
    cx, cy = xk - ST_W/2 + 1.3, yk - ST_H/2 - 1.3
    ax.add_patch(Circle((cx, cy), 1.5, facecolor="white", edgecolor=ACCENT_MID,
                        lw=0.5, zorder=7))
    ax.text(cx, cy, str(i), ha="center", va="center", fontsize=5.5,
            color=ACCENT, fontweight="bold", zorder=8)

# ---- 图例（底部单行三键；C+ 去键描边、字号 5.8） ----
LY = 26.5
psd_style.card(ax, 2.5, LY-1.4, 4.8, 2.8, fill=A_CARD, edge=None,
               rounding=0.6, z=8)
ax.text(9.0, LY, "automated step", fontsize=5.8, color=INK, va="center")
psd_style.card(ax, 26.0, LY-1.4, 4.8, 2.8, fill=H_CARD, edge=None,
               rounding=0.6, z=8)
ax.text(32.5, LY, "human-in-the-loop step", fontsize=5.8, color=INK, va="center")
ax.plot([62.0, 67.5], [LY, LY], color="#777777", lw=0.8, linestyle=(0, (4, 3)))
ax.text(69.0, LY, "write-back to state", fontsize=5.8, color=INK, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

pdf_path = OUT / "fig2_pseudo_label_loop.pdf"
fig.savefig(pdf_path)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600)
fig.savefig(PREVIEW / "preview_fig2.png", dpi=150)
print("fig2 v19 (direction C+ intensified) saved")

import make_common_gates as gates
gates.gate_print_robustness(
    "fig2-v19", {"automated": psd_style.SEM_EDGE, "human": psd_style.HUMAN_EDGE},
    redundancy="card tint + legend keys + step chips")
gates.gate_labels(fig, ax, list(ax.texts), name="fig2-v19")
gates.gate_typography(fig, min_pt=5.5, name="fig2-v19")
gates.gate_marks_in_axes(fig, name="fig2-v19")
gates.gate_pdf(pdf_path, max_w_pt=246.3)

# ---- 本地净空自检（不进共享模块）：站卡内文字必须落在卡框内且净空 >= 2px ----
# G2 只断言"标签在 axes 内"，不覆盖"标签在自己所属卡片内"；v18 的 5.9pt 站名
# 实测净空仅 1.7px（临界），本版加宽卡到 20.5 单位 + 站名 6.0pt 后复验。
fig.canvas.draw()
ren = fig.canvas.get_renderer()
bad = []
for k, (xk, yk) in S.items():
    # 卡框四角 -> 显示像素（避免数据单位与 pt 的量纲混用）
    cx0, cy0 = ax.transData.transform((xk - ST_W/2, yk - ST_H/2))
    cx1, cy1 = ax.transData.transform((xk + ST_W/2, yk + ST_H/2))
    mins = []
    for t in (name_txt[k], sub_txt[k]):
        b = t.get_window_extent(ren)
        cl = min(b.x0 - cx0, cx1 - b.x1, b.y0 - cy0, cy1 - b.y1)   # 已是像素
        mins.append(cl)
        if cl < 2.0:
            bad.append((k, t.get_text()[:18], round(float(cl), 2)))
    print("[local clearance] card %-9s name+sub inside box, min clearance = "
          "%.2f px" % (k, min(mins)))
assert not bad, f"card-internal clearance < 2px: {bad}"
print("[local clearance] fig2-v19: 6 station cards -> PASS")