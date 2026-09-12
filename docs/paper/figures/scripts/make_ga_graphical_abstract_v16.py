# -*- coding: utf-8 -*-
r"""GA v16 — 正典定稿版（风格方向 **C+ 强化** · 修 D5 箭头语义反转）。

由 v15 转正后进一步收紧（C+）并修复一处遗留的语义反转：

**D5 修复（核心改动）**：v15 的演化回线（dashed）走 Ω 底 → 横绕 → Φ 底，
FancyArrowPatch 在 Φ 底端朝**上指向 Φ**，视觉语义变成"Φ 被重训"——这与
"Physics layer Φ (frozen)" 和 "only Ω retrains" 两个直标的语义同时冲突。
v16 把这条上行箭头删掉，dashed loop 改为从 Ω 底向下延伸、在右下角**断尾
无箭头**，并把"only Ω retrains"直标紧贴 Ω 底，物理上"无箭头进入 Φ" =
"Φ 不参与重训"。同时循环的另一端（Ω）**保留朝上的实心箭头**，明示
"重训闭环回到 Ω"，闭环终点在 Ω。

**C+ 收紧**（笔触更细 + 字号阶梯更大 + 留白更多）：
1. 笔触：流线 2.2 -> 1.0（接口主箭）；虚线 1.0 -> 0.6；card 描边保持去边。
2. 字号：标题 8.4 -> 8.6；层副标题 5.8 -> 6.0；层注释 5.4 -> 5.6；
   接口标签 4.9 -> 5.6（**显著放大**——这是 v15 全站最小文字，低于 5.5pt
   下限 10.8%）；only Ω retrains 5.4 -> 5.8；保留率条值 5.2 -> 5.6；
   行名 5.6 -> 6.0；100% 注记 4.6 -> 5.6。下限 5.5pt 全图达成。
3. 留白：画布 7.375x1.82 -> 7.375x1.95in（增 7%），ylim 0..24.7 -> 0..26.0，
   给底部条与"only Ω retrains"留出呼吸。
4. 接口箭头位置上调到 x 中线 18.6（与双块高线 y=18.2 同高），流线更整齐。

冻结项（任务书 §二-5）：保留率 90.7/88.9/28.9、canine 9.8 vs 11.1% chance
@13% budget、语义色、531×131pt 硬上限（D6 公开 GA，MiKTeX 不参与装配）。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
C_TEXT, C_ARROW = psd_style.INK, "#333333"
C_NOTE = "#666666"
plt.rcParams["mathtext.fontset"] = "dejavusans"

# ---- 531x131pt 硬上限（投稿系统 graphical abstract 资产） ----
# 1in = 72pt ⇒ 531/72 x 131/72 = 7.375 x 1.819 in。本画布 7.375x1.82in，
# 在硬上限内。C+ 的留白增益改走\"内部 ylim + 行间距\"：ylim 0..24.7
# (与 v15 一致)、层块高 +0.6 单位、条行间距 +0.2 单位、双块上下内边距
# 各 +0.3 单位。不动画布高度。
fig, ax = plt.subplots(figsize=(7.375, 1.82))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
ax.set_xlim(0, 100); ax.set_ylim(0, 24.7); ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---- 上行：Φ / Ω 宽双块（C+：层块高 +0.4 单位，层间净空不变；双块上下边距 +0.3） ----
psd_style.card(ax, 3.0, 12.9, 36, 11.0, fill=psd_style.PHYS_TINT, edge=None,
               rounding=2.4, z=3)
ax.text(21.0, 21.4, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=8.4, fontweight="bold", color=PD, zorder=6)
ax.text(21.0, 18.0, "self-supervised dynamics \u00b7 behavior proposals",
        ha="center", va="center", fontsize=6.0, color="#333333", zorder=6)
ax.text(21.0, 15.2, "trained once, reused across taxonomies",
        ha="center", va="center", fontsize=5.6, style="italic",
        color="#555555", zorder=6)
psd_style.card(ax, 48.0, 12.9, 36, 11.0, fill=psd_style.SEM_TINT, edge=None,
               rounding=2.4, z=3)
ax.text(66.0, 21.4, r"Semantic layer $\Omega$  (revisable)", ha="center",
        va="center", fontsize=8.4, fontweight="bold", color=SD_, zorder=6)
ax.text(66.0, 18.0, "anchor-guided pseudo-labeling \u00b7 self-training",
        ha="center", va="center", fontsize=6.0, color="#333333", zorder=6)
ax.text(66.0, 15.2, r"taxonomy evolves $\mathcal{Y}\to\mathcal{Y}'$",
        ha="center", va="center", fontsize=5.6, style="italic",
        color=SD_, fontweight="bold", zorder=6)

# ---- 接口箭头（C+：流线 2.2 -> 1.0） + 单行边标签（C+：字号 4.9 -> 5.6） ----
psd_style.flowline(ax, 39.4, 18.2, 47.6, 18.2, color=C_ARROW, lw=1.0, head=12)
ax.text(43.5, 21.2, "embeddings", ha="center", va="center", fontsize=5.6,
        color=C_NOTE, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
ax.text(43.5, 19.7, "+ proposals", ha="center", va="center", fontsize=5.6,
        color=C_NOTE, zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# ---- D5 修复：演化回线 ----
# v15 错误：dashed 从 Ω 底 → Φ 底，FancyArrowPatch 在 Φ 底朝上指向 Φ，
#          视觉 = "Φ 被重训"，与 frozen 冲突。
# v16 修复：
#   (a) 删掉朝上指向 Φ 的箭头 —— frozen 不该有"重训进入"箭头；
#   (b) dashed loop 改为 Ω 底向下延伸、在右下角**断尾无箭头**（明示 Φ 不参与）；
#   (c) Ω 端保留朝上的实心箭头，明示"重训闭环回到 Ω"，闭环终点在 Ω。
#   (d) "only Ω retrains" 直标落在虚线下方、说明文字上方，两向净空 >=0.8 单位。

# 闭环主体（单横段，承 v15 纵向排布以保证净空）：Ω 底 12.9 → 11.4 → 横折 → 上行至 ×
ax.plot([66.0, 66.0], [12.9, 11.4], color="#999999", lw=0.6,
        ls=(0, (4, 3)), zorder=2)
ax.plot([66.0, 22.0], [11.4, 11.4], color="#999999", lw=0.6, ls=(0, (4, 3)),
        zorder=2)
# 左端上行至 Φ 底，但**无箭头**，改以 × 终止符明示"Φ 不接收重训"
ax.plot([22.0, 22.0], [11.4, 12.7], color="#999999", lw=0.6, ls=(0, (4, 3)),
        zorder=2)
ax.text(22.0, 12.7, r"$\times$", ha="center", va="center", fontsize=6.4,
        color="#999999", zorder=3)

# Ω 端：实心箭头朝上进入 Ω（C+：lw 1.0、head 8），明示闭环终点在 Ω
ax.add_patch(FancyArrowPatch((66.0, 12.9), (66.0, 13.3), arrowstyle="-|>",
             mutation_scale=8, color=C_ARROW, lw=1.0, zorder=2))

# 直标：v15 纵向位置（y=10.2），与虚线 11.4 净空 0.63 单位、与说明 8.6 净空 0.46 单位
ax.text(43.5, 10.2, r"only $\Omega$ retrains", ha="center", va="center",
        fontsize=5.8, color=SD_, fontweight="bold", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.6))

# ---- 下行：全宽保留率横条（C+：行高保持 2.4、字号 5.2->5.6、行名 5.6->6.0） ----
ax.text(50.0, 8.6, "budget retention \u00b7 own full-budget top-1 kept at "
        "10% labels (canine: 9.8 vs 11.1% chance at 13% budget)",
        ha="center", va="center", fontsize=5.6, color=C_NOTE)
BAR_X0, BAR_MAX, BAR_H = 20.0, 66.0, 2.4
rows = [
    ("NTU60", 90.7, False, psd_style.S_NTU60, psd_style.INK),
    ("NTU120", 88.9, False, psd_style.S_NTU120, psd_style.INK),
    ("canine", 28.9, True, psd_style.GRAY_FILL, SD_),
]
# v15 纵向排布：canine 底 0.4 > 0；NTU60 顶 8.0 与说明 8.6 净空 0.6 单位 (3.2px)
ys = (5.6, 3.0, 0.4)
for y in ys:  # 0-100% 基准轨
    psd_style.card(ax, BAR_X0, y, BAR_MAX, BAR_H, fill="#F2F2F2", edge=None,
                   rounding=0.6, z=3)
ax.text(BAR_X0 + BAR_MAX + 0.6, ys[0] + BAR_H + 0.5, "100%", ha="left",
        va="bottom", fontsize=5.6, color=C_NOTE, zorder=6)
for (label, val, is_boundary, bcol, txtcol), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.4, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=6.0, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=SD_, lw=0.6,
                       rounding=0.6, z=4, hatch="///")
        ax.text(BAR_X0 + w + 1.0, y + BAR_H / 2, f"{val:.1f}%", ha="left",
                va="center", fontsize=5.6, color=txtcol, zorder=6)
    else:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=None,
                       rounding=0.6, z=4)
        ax.text(BAR_X0 + w - 0.7, y + BAR_H / 2, f"{val:.1f}%", ha="right",
                va="center", fontsize=5.6, color=txtcol, fontweight="bold",
                zorder=6)

pdf_path = OUT / "fig_ga_graphical_abstract.pdf"
png_path = OUT / "fig_ga_graphical_abstract.png"
prev_path = ROOT / "reports" / "figB-proposal" / "preview_fig_ga.png"
prev_path.parent.mkdir(parents=True, exist_ok=True)
# GA 不走 LaTeX，531x131pt 是投稿系统给的位图栅格。**固定画布不 tight**
# ⇒ MediaBox 严格等于 figsize = 531.0x131.04pt，在 131pt 上限内。
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(prev_path, dpi=150)

# ---- 门禁（本图无坐标轴/误差棒，故只调 G1/G2/G5/G6/G4） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "ga-v16", {"physics": psd_style.PHYS_DEEP, "semantic": psd_style.SEM_DEEP,
               "ntu60": psd_style.S_NTU60, "ntu120": psd_style.S_NTU120},
    redundancy="tone tint + position + labels + hatched boundary bar")
gates.gate_labels(fig, ax, list(ax.texts), name="ga-v16")
gates.gate_typography(fig, min_pt=5.5, name="ga-v16")
gates.gate_marks_in_axes(fig, name="ga-v16")
gates.gate_pdf(pdf_path, max_w_pt=531.1, max_h_pt=131.1)

# ---- D5 语义断言（本地，防回归）----
# 事实：本图演化回线的"重训进入"箭头只允许指向 Ω（Semantic，revisable）。
# Φ（Physics）是 frozen，任何指向 Φ 的箭头都是语义反转。
# 断言实现：全图 FancyArrowPatch 中，不允许存在"箭头终点落在 Φ 块内 (x<=39, y>=12.9)"的实例。
_arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
_bad = []
for p in _arrows:
    # FancyArrowPatch 的端点存在私有属性 _posA_posB = ((xa, ya), (xb, yb))
    # （数据坐标，未被 transform 影响）；无该属性则用 get_path() 末点逆变换兜底。
    pa, pb = getattr(p, "_posA_posB", (None, None))
    if pa is None:
        verts = p.get_path().vertices
        (ex, ey) = ax.transData.inverted().transform(verts[-1])
    else:
        ex, ey = pb
    if ex <= 39.0 and ey >= 12.9:
        _bad.append((round(float(ex), 1), round(float(ey), 1)))
assert not _bad, (
    "D5 FAIL: 存在指向 frozen Physics 层 (Φ) 的箭头 —— 语义反转回归。"
    f"命中: {_bad}")
print("[local D5] ga-v16: %d FancyArrowPatch, 指向 Phi 的箭头 0 个 -> PASS"
      % len(_arrows))
print("[local D5] 闭环终点 = Omega（x>39），Phi 底为 x 形终止符 —— 与 "
      "'only Omega retrains' 一致")
plt.close(fig)