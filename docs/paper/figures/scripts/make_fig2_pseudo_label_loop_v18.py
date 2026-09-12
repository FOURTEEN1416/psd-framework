# -*- coding: utf-8 -*-
r"""fig2 v18 — 伪标签循环双行管线（正典 · 风格方向 C「分阶色标 + 制图精度」+ 尺度修复）。

承 v17 全部拓扑（零交叉自动环 + 贴盒回路轨道 + 正交 write-back 肘线 +
κ≥τ 段标签置段下方的 G2 修复），**图内全部文字逐字承 v17，零新增零删除零改写**：

1. 尺度修复（wt/figB 诊断头号根因）：去 `bbox_inches="tight"`，固定画布
   3.42x2.36in，MediaBox 严格 246.24x169.92pt=版心宽；v17 的 tight 墨迹框
   245.22x168.95pt 被 LaTeX 放大 1.004 倍，本版装入系数 1.000，
   版面占用 169.92pt 与 v17 的 169.65pt 逐点持平（不引发浮动回退）。
2. 分阶色标（tone ladder）：站卡由"白卡+家族色描边"改为"同相浅底(mix 0.58)
   + 更深同相描边(mix 0.30)"；焦点 Label pool 保持 SEM_EDGE 满块、hub 保持
   INK 墨黑满块（焦点 ≤2 不变）。层级来自明度阶梯，不再是白卡贴纸感。
3. 字重阶梯：焦点块/hub 白字 bold -> 站名 regular 5.9（家族 DEEP 色）->
   站注 regular 5.5 灰 -> 流注 regular 5.8 灰。v17 通篇 bold 的"同重"消除。
4. 字号下限 5.5pt（D3 合规修复）：chip 编号 5.2 -> 5.5、"write-back" 5.4 -> 5.5。
5. 线型语言不变（承 v17 语义纪律）：主流线 solid 1.6pt、write-back 虚线
   1.1pt——与 caption "the dashed write-back spokes" 逐词对应。

caption 约束逐词核对（03-method.tex \label{fig:pseudoloop}）：
top row=automated(orange)、bottom row=human(teal)、hub 墨黑、(P, Ω, A) 三行、
κ≥τ 上行 / κ<τ 下分、next round 沿顶部回、"budget B"、"write-back" 虚线、
legend 三键——全部保持。
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # worktree 隔离：禁硬编码主仓绝对路径
OUT = ROOT / "docs" / "paper" / "figures"
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
PAPER = "#FFFFFF"
INK = psd_style.INK
ACCENT = psd_style.SEM_DEEP            # automated 深橙(深化字色)
ACCENT_MID = psd_style.SEM_EDGE        # automated 板内深橙原色
HUMAN = psd_style.HUMAN_DEEP           # human 青绿深化
HUMAN_MID = psd_style.HUMAN_EDGE       # human 板内青绿原色
SOFT = "#555555"                        # 流线深灰


def _mix(hex_color, f):
    """向白混合 f 比例（本脚本局部助手，不写入 psd_style）：同相明度阶梯派生。"""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        round(r + (255 - r) * f), round(g + (255 - g) * f), round(b + (255 - b) * f))


# 分阶色标 token：站卡=浅底(mix 0.58) + 更深描边(mix 0.30)；焦点=满块
A_CARD = _mix(ACCENT_MID, 0.58); A_CARD_EDGE = _mix(ACCENT_MID, 0.30)
H_CARD = _mix(HUMAN_MID, 0.58);  H_CARD_EDGE = _mix(HUMAN_MID, 0.30)

# ---- 固定画布：宽 3.42in == 版心 246.24pt ----
FIG_W, FIG_H = 3.42, 2.36
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
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
seg("assign", "pool", label=r"$\kappa \geq \tau$", ldy=-2.2)
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
ax.text(64.0, 51.5, "write-back", fontsize=5.5, color="#666666", ha="left", va="center")

# ---- hub(墨黑满块白字, 扁平; 焦点 #2) ----
psd_style.card(ax, HUB_X-HUB_W/2, HUB_Y-HUB_H/2, HUB_W, HUB_H, fill=INK, edge=None,
               rounding=1.2, z=6)
ax.text(HUB_X, HUB_Y+2.8, "Prototypes $P$", ha="center", va="center", fontsize=5.6,
        color=PAPER, fontweight="bold", zorder=8)
ax.text(HUB_X, HUB_Y+0.0, "classifier $\\Omega$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=8)
ax.text(HUB_X, HUB_Y-2.8, "anchors $A$", ha="center", va="center", fontsize=5.6,
        color=PAPER, zorder=8)

# ---- 站盒（方向 C：同相浅底 + 更深同相描边；焦点=深橙满块白字） ----
for k, (xk, yk) in S.items():
    manual = k in MANUAL
    deep = HUMAN if manual else ACCENT
    if k == FOCAL:
        psd_style.card(ax, xk-ST_W/2, yk-ST_H/2, ST_W, ST_H, fill=ACCENT,
                       edge=None, rounding=1.2, z=4)
        ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
                color="white", fontweight="bold", zorder=6)
        ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
                color="white", alpha=0.9, zorder=6)
    else:
        fill = H_CARD if manual else A_CARD
        edge = H_CARD_EDGE if manual else A_CARD_EDGE
        psd_style.card(ax, xk-ST_W/2, yk-ST_H/2, ST_W, ST_H, fill=fill, edge=edge,
                       lw=0.9, rounding=1.2, z=4)
        ax.text(xk, yk+1.1, NAMES[k], ha="center", va="center", fontsize=5.9,
                color=deep, fontweight="normal", zorder=6)
        ax.text(xk, yk-1.8, SUBS[k], ha="center", va="center", fontsize=5.5,
                color="#555555", zorder=6)

# ---- 自动环步骤号 chip ①-④（角标, 与 caption 叙述序对应；字号 5.2 -> 5.5） ----
for i, k in enumerate(("assign", "pool", "upd", "reest"), 1):
    xk, yk = S[k]
    cx, cy = xk - ST_W/2 + 1.2, yk + ST_H/2 + 1.2
    ax.add_patch(Circle((cx, cy), 1.5, facecolor="white", edgecolor=ACCENT_MID,
                        lw=0.8, zorder=7))
    ax.text(cx, cy, str(i), ha="center", va="center", fontsize=5.5,
            color=ACCENT, fontweight="bold", zorder=8)

# ---- 图例(底部单行三键; 键色与站卡新样式一致) ----
LY = 30.0
psd_style.card(ax, 2.5, LY-1.3, 4.6, 2.6, fill=A_CARD, edge=A_CARD_EDGE, lw=1.0,
               rounding=0.5, z=8)
ax.text(8.9, LY, "automated step", fontsize=5.6, color=INK, va="center")
psd_style.card(ax, 26.0, LY-1.3, 4.6, 2.6, fill=H_CARD, edge=H_CARD_EDGE, lw=1.0,
               rounding=0.5, z=8)
ax.text(32.4, LY, "human-in-the-loop step", fontsize=5.6, color=INK, va="center")
ax.plot([62.0, 67.5], [LY, LY], color="#777777", lw=1.1, linestyle=(0, (4, 3)))
ax.text(69.0, LY, "write-back to state", fontsize=5.6, color=INK, va="center")

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)

# ---- 出图（尺度修复：无 tight，MediaBox 严格等于画布） ----
pdf_path = OUT / "fig2_pseudo_label_loop.pdf"
fig.savefig(pdf_path)
fig.savefig(OUT / "fig2_pseudo_label_loop.png", dpi=600)
fig.savefig(PREVIEW / "preview_fig2.png", dpi=150)
print("fig2 v18 (direction C tone-ladder + fixed-canvas) saved")

# ---- 门禁（G1 + G2 + G5 + G6 + G4） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig2-v18", {"automated": psd_style.SEM_EDGE, "human": psd_style.HUMAN_EDGE},
    redundancy="card tint/border + legend keys + step chips")
gates.gate_labels(fig, ax, list(ax.texts), name="fig2-v18")
gates.gate_typography(fig, min_pt=5.5, name="fig2-v18")
gates.gate_marks_in_axes(fig, name="fig2-v18")
gates.gate_pdf(pdf_path, max_w_pt=246.3)
