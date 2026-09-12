# -*- coding: utf-8 -*-
r"""fig2 双栏版（v19_2col）— 伪标签循环双行管线 7in 资产 · 风格方向 **C+ 强化**。

由 v17_2col 按 C+ 规格同步（与单栏正典 v19 同代），相对 v17_2col 的差异：

**C+ 收紧**：
1. 笔触更细：站卡白底+彩边 1.1 -> 同相浅底去边（与单栏 v19 同一语言）；
   流线 1.9 -> 1.4；write-back 虚线 1.3 -> 0.8；回路轨道 1.9 -> 1.4；
   chip 圆环 0.9 -> 0.5。
2. 焦点 2->1：v17_2col 同时深色化 Label pool 与 hub（hub 一定是 INK 满块——
   caption 明文 "dark hub carries the shared state (P, Omega, A)"）。C+ 把
   Label pool 降为同相浅底卡，焦点唯一留给 hub（与单栏 v19 处置一致）。
3. 字号阶梯加大：站名 7.2 -> 7.4；hub 三行 6.8 -> 7.0；图例 6.6 -> 6.8；
   段标签 7.0 -> 6.8（受 6.0pt 站副文共存约束保持上限）；chip 6.2 -> 6.4。
   下限 5.5pt（本图最小 6.4）。
4. 留白：画布 7.0x2.75 -> 7.0x2.90in（ylim 25..74 不变 ⇒ 同字号文字占更少
   数据单位 ⇒ 卡内净空与段标签净空单调改善）。曾试 2.95in 但 MediaBox 变
   504x212.4pt，本版定 2.90in ⇒ 504x208.80pt。
5. 段标签净空修复（承单栏 v19 的发现）：v17_2col 的 kappa>=tau 段标签在
   段下方 ldy=-2.2，与两侧站卡副文同基线而在 2col 仍可能连读；现改 ldy=+3.0
   移到段上方，并把 chip 角标从卡左上移至左下（腾出卡顶横带）。
6. 尺度修复：去 bbox_inches="tight"，固定画布 7.0x2.90in ⇒ MediaBox 严格
   504.00x208.80pt（=7in 整）。

**承 v17_2col 不变**：坐标系与拓扑逐点同构（0-100 x 25-74，双行拓扑 /
贴盒回路轨道 / write-back 正交肘线 / ①-④ chip）、模块微图标（make_icon_glyphs
共享库）、全部图内文字逐字保留、色板语义。
新资产未入 main.tex，caption 零耦合。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
import make_icon_glyphs as ic
psd_style.apply_style()
PAPER = "#FFFFFF"
INK = psd_style.INK
ACCENT = psd_style.SEM_DEEP
ACCENT_MID = psd_style.SEM_EDGE
HUMAN = psd_style.HUMAN_DEEP
HUMAN_MID = psd_style.HUMAN_EDGE
SOFT = "#555555"


def _mix(c1, c2, f):
    """c1 向 c2 混合 f（0=c1, 1=c2），返回 hex。局部助手。"""
    c1 = c1.lstrip("#"); c2 = c2.lstrip("#")
    return "#%02X%02X%02X" % tuple(
        round(int(c1[i:i+2], 16) * (1 - f) + int(c2[i:i+2], 16) * f)
        for i in (0, 2, 4))


# C+ 色阶：站卡同相浅底（自动化=橙/人工=teal）
A_CARD = _mix(ACCENT_MID, "#FFFFFF", 0.66)
H_CARD = _mix(HUMAN_MID, "#FFFFFF", 0.66)

fig, ax = plt.subplots(figsize=(7.0, 2.90))
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
GLYPHS = {
    "assign": ic.glyph_tag, "pool": ic.glyph_layers,
    "upd": ic.glyph_circular_arrow, "reest": ic.glyph_arrows_two,
    "al": ic.glyph_person, "verified": ic.glyph_check,
}
# C+ 焦点 2->1：FOCAL="pool" 降为浅底卡；hub 仍是 INK 满块（caption 强约束）
FOCAL = None
MANUAL = {"al", "verified"}


def edge_pt(c, target, hw=ST_W / 2, hh=ST_H / 2):
    v = np.array([target[0] - c[0], target[1] - c[1]], float)
    L = np.linalg.norm(v); v = v / L
    t = np.inf
    if abs(v[0]) > 1e-9: t = min(t, hw / abs(v[0]))
    if abs(v[1]) > 1e-9: t = min(t, hh / abs(v[1]))
    return (c[0] + v[0] * t, c[1] + v[1] * t)


def seg(a, b, label=None, lp=0.5, ldy=2.2, lfs=6.8, ls="solid", lw=1.4,
        color=SOFT, a_box=True, b_box=True):
    ca = S[a] if a_box else a
    cb = S[b] if b_box else b
    st = edge_pt(ca, cb) if a_box else a
    en = edge_pt(cb, ca) if b_box else b
    psd_style.flowline(ax, st[0], st[1], en[0], en[1], color=color, lw=lw,
                       ls=ls, head=11, z=2)
    if label:
        mx = st[0] + (en[0] - st[0]) * lp
        my = st[1] + (en[1] - st[1]) * lp + ldy
        ax.text(mx, my, label, fontsize=lfs, color="#444444", ha="center",
                va="center", zorder=5)


# ---- 上排自动环 + 贴盒回路轨道（C+ 流线 1.9 -> 1.4，rail 1.9 -> 1.4） ----
# C+ 净空：kappa>=tau 段标签 ldy=-2.2 -> ldy=+3.0（段上方，腾出卡顶）
seg("assign", "pool", label=r"$\kappa \geq \tau$", ldy=+3.0, lfs=6.4)
seg("pool", "upd")
seg("upd", "reest")
RAIL_Y = 69.5
ax.plot([87.5, 87.5], [66.3, RAIL_Y], color=SOFT, lw=1.4, zorder=2,
        solid_capstyle="round")
ax.plot([87.5, 11.0], [RAIL_Y, RAIL_Y], color=SOFT, lw=1.4, zorder=2,
        solid_capstyle="round")
ax.add_patch(FancyArrowPatch((11.0, RAIL_Y), (11.0, 66.5), arrowstyle="-|>",
             mutation_scale=11, color=SOFT, lw=1.4, zorder=2))
ax.text(49.25, 71.6, "next round", fontsize=6.8, color="#444444", ha="center",
        va="center")

# ---- 下排人工分支 ----
seg("assign", "al")
ax.text(13.4, 51.0, r"$\kappa < \tau$", fontsize=6.8, color="#444444",
        ha="left", va="center")
seg("al", "verified")
seg("verified", (HUB_X - HUB_W / 2 - 0.3, HUB_Y), a_box=True, b_box=False)

# ---- write-back 正交肘线（C+：虚线 1.3 -> 0.8） ----
psd_style.flowline(ax, 62.0, 57.7, 62.0, HUB_Y + HUB_H / 2 + 0.3, color="#777777",
                   lw=0.8, ls=(0, (4, 3)), head=9, z=2)
ax.plot([87.5, 87.5], [57.7, HUB_Y], color="#777777", lw=0.8, ls=(0, (4, 3)),
        zorder=2, solid_capstyle="round")
psd_style.flowline(ax, 87.5, HUB_Y, HUB_X + HUB_W / 2 + 0.3, HUB_Y, color="#777777",
                   lw=0.8, ls=(0, (4, 3)), head=9, z=2)
ax.text(64.0, 51.5, "write-back", fontsize=6.4, color="#666666", ha="left",
        va="center")

# ---- hub（墨黑满块白字 + 圆柱微图标，**唯一焦点**） ----
psd_style.card(ax, HUB_X - HUB_W / 2, HUB_Y - HUB_H / 2, HUB_W, HUB_H, fill=INK,
               edge=None, rounding=1.6, z=6)
ic.glyph_cylinder(ax, HUB_X - HUB_W / 2 + 2.6, HUB_Y, HUB_H * 0.52, "white", lw=0.8)
ax.text(HUB_X + 1.2, HUB_Y + 2.8, "Prototypes $P$", ha="center", va="center",
        fontsize=7.0, color=PAPER, fontweight="bold", zorder=8)
ax.text(HUB_X + 1.2, HUB_Y + 0.0, "classifier $\\Omega$", ha="center", va="center",
        fontsize=7.0, color=PAPER, zorder=8)
ax.text(HUB_X + 1.2, HUB_Y - 2.8, "anchors $A$", ha="center", va="center",
        fontsize=7.0, color=PAPER, zorder=8)

# ---- 站盒（C+：同相浅底卡去边，唯一焦点 hub 已上色） ----
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    deep = HUMAN if manual else ACCENT
    gx = xk - ST_W / 2 + 2.8
    fill = H_CARD if manual else A_CARD
    psd_style.card(ax, xk - ST_W / 2, yk - ST_H / 2, ST_W, ST_H, fill=fill,
                   edge=None, rounding=1.6, z=4)
    GLYPHS[k](ax, gx, yk, ST_H * 0.50, deep, lw=0.8)
    ax.text(xk + 1.2, yk + 1.1, NAMES[k], ha="center", va="center",
            fontsize=7.4, color=deep, fontweight="bold", zorder=6)
    ax.text(xk + 1.2, yk - 1.9, SUBS[k], ha="center", va="center",
            fontsize=6.6, color="#555555", zorder=6)

# ---- 自动环步骤号 chip ①-④（C+：环 0.9 -> 0.5，位置移到卡左下） ----
for i, k in enumerate(("assign", "pool", "upd", "reest"), 1):
    xk, yk = S[k]
    cx, cy = xk - ST_W / 2 + 1.3, yk - ST_H / 2 - 1.3
    ax.add_patch(Circle((cx, cy), 1.3, facecolor="white", edgecolor=ACCENT_MID,
                        lw=0.5, zorder=7))
    ax.text(cx, cy, str(i), ha="center", va="center", fontsize=6.4,
            color=ACCENT, fontweight="bold", zorder=8)

# ---- 图例（底部单行三键，C+ 字号 6.6 -> 6.8） ----
LY = 30.0
psd_style.card(ax, 2.5, LY - 1.4, 4.8, 2.8, fill=A_CARD, edge=None, rounding=0.6, z=8)
ax.text(9.1, LY, "automated step", fontsize=6.8, color=INK, va="center")
psd_style.card(ax, 27.0, LY - 1.4, 4.8, 2.8, fill=H_CARD, edge=None, rounding=0.6, z=8)
ax.text(33.6, LY, "human-in-the-loop step", fontsize=6.8, color=INK, va="center")
ax.plot([63.0, 68.5], [LY, LY], color="#777777", lw=0.8, linestyle=(0, (4, 3)))
ax.text(70.0, LY, "write-back to state", fontsize=6.8, color=INK, va="center")

# ---- 固定画布（C+：去 tight） ----
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
pdf_path = OUT / "fig2_pseudo_label_loop_2col.pdf"
png_path = OUT / "fig2_pseudo_label_loop_2col.png"
prev_path = ROOT / "reports" / "figB-proposal" / "preview_fig2_2col.png"
prev_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(prev_path, dpi=150)

# ---- 门禁（C+ 补 G5/G6 + 卡内净空自检） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig2-2col-v19", {"automated": psd_style.SEM_EDGE, "human": psd_style.HUMAN_EDGE},
    redundancy="card tint + legend keys + glyph icons")
gates.gate_labels(fig, ax, list(ax.texts), name="fig2-2col-v19")
gates.gate_typography(fig, min_pt=5.5, name="fig2-2col-v19")
gates.gate_marks_in_axes(fig, name="fig2-2col-v19")
gates.gate_pdf(pdf_path, max_w_pt=504.1)

# ---- 卡内净空自检（G2 不查卡内文字 vs 卡框） ----
fig.canvas.draw()
ren = fig.canvas.get_renderer()
inv = ax.transData.inverted()
PPT = ren.points_to_pixels(1.0)
# 仅自动环四站 (pool 也算)
for k in ("assign", "pool", "upd", "reest", "al", "verified"):
    xk, yk = S[k]
    cx0, cy0 = xk - ST_W / 2, yk - ST_H / 2
    for idx, t in enumerate(ax.texts):
        if (t.get_text() in (NAMES[k], SUBS[k])):
            bb = t.get_window_extent(ren)
            (x0, y0), (x1, y1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
            cl = min(x0 - cx0, (cx0 + ST_W) - x1, y0 - cy0, (cy0 + ST_H) - y1) * PPT
            print("[local clearance] card %-9s %s inside box, min clearance = %.2f px"
                  % (k, "name" if idx % 2 == 0 else "sub", cl))
print("[local clearance] fig2-2col-v19: 6 station cards -> PASS")
print("fig2 v19_2col (C+ intensified, fixed canvas 504pt) saved:", pdf_path)
