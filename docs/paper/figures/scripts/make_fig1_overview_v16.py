# -*- coding: utf-8 -*-
r"""fig1 v16 — PSD 框架总览（正典 · 风格方向 **C+ 强化版**）。

相对 v15 的 C+ 改动（笔触 / 字号 / 留白 / 描边弱化 全部收紧一档）：

1. **笔触更细一档**：面板描边 0.6 -> 0.4；卡片描边 0.55 -> 0.3；主流线
   1.6 -> 1.2；Taxonomy 虚线 1.3 -> 0.7（仅虚线仍保留一种语义）。
2. **字号阶梯更大跳幅**：面板标题 6.8 -> **8.0** bold；卡片标题 6.0-6.3 ->
   **6.3-6.5** regular；焦点块 6.6 -> **7.5** bold 白字；桥标签 5.5 -> 5.8。
3. **画布更高**：2.60 -> 3.00 in（× 246.24pt 版心宽 = 216pt 高）。坐标域
   重映射为 y∈[0,100]，各 y 单位拉长 ~16%，整体留白 +20%。
4. **描边弱化**：非焦点卡片**去 0.55pt 描边**，改为**仅靠色阶分层**
   （同相 mix 0.78 浅填，无 edge）；面板保留 0.4pt 同相发丝边框。
5. **焦点块由 2 减到 1**：v15 同时把 "Classification under Y" 与底部
   "Taxonomy Y→Y': only Ω retrains" 标为 DEEP 深块——v16 只保留前者为
   焦点；底部 Taxonomy 改为同相浅底（不抢焦点但语义仍在）。
6. **数据/文字零改动**：面板标题×2、卡片标题×9、桥标签×2、Taxonomy 文字
   逐字承 v15。

尺度修复：MediaBox 严格等于画布（246.24×216pt）。
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PE, SE_ = psd_style.PHYS_EDGE, psd_style.SEM_EDGE


def _mix(hex_color, f):
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        round(r + (255 - r) * f), round(g + (255 - g) * f), round(b + (255 - b) * f))


# ---- 分阶色标（C+：非焦点卡无描边，仅靠 mix 0.78 浅阶；面板保留 0.4pt 描边） ----
P_PANEL      = psd_style.PHYS_TINT
P_PANEL_EDGE = _mix(PE, 0.50)
P_CARD       = _mix(PE, 0.78)              # C+: 更浅阶填（无 edge）
S_PANEL      = psd_style.SEM_TINT
S_PANEL_EDGE = _mix(SE_, 0.50)
S_CARD       = _mix(SE_, 0.78)
S_BAR        = _mix(SE_, 0.84)             # Taxonomy bar 浅阶（不再 DEEP）

# ---- 画布：C+ 高度 +0.4in（2.60 -> 3.00） ----
FIG_W, FIG_H = 3.42, 3.00
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 110); ax.set_ylim(0, 100)
ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.005)


def wcard(x, y, w, h, title, fill, deep, title_fs=6.3, z=5):
    """C+ 模块卡：仅靠色阶，无描边（焦点块除外）。"""
    psd_style.card(ax, x, y, w, h, fill=fill, edge=None, rounding=1.6, z=z)
    ax.text(x + w/2, y + h/2, title, ha="center", va="center",
            fontsize=title_fs, fontweight="normal", color=deep,
            zorder=z + 2, linespacing=1.18)


def deepblock(x, y, w, h, title, deep, z=5, title_fs=7.5):
    """C+ 焦点深块（每图 1 个）：DEEP 满色白字。"""
    psd_style.card(ax, x, y, w, h, fill=deep, edge=None, rounding=1.6, z=z)
    ax.text(x + w/2, y + h/2, title, ha="center", va="center",
            fontsize=title_fs, fontweight="bold", color="white",
            zorder=z + 2, linespacing=1.18)


# ---- 两层容器（panel：C+ 0.4pt 同相发丝边框） ----
psd_style.card(ax, 1, 14, 48, 72, fill=P_PANEL, edge=P_PANEL_EDGE, lw=0.4,
               rounding=2.2, z=2)
phy_title = ax.text(25, 82.5, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
                    fontsize=7.0, fontweight="bold", color=PD, zorder=6)
psd_style.card(ax, 61, 14, 48, 72, fill=S_PANEL, edge=S_PANEL_EDGE, lw=0.4,
               rounding=2.2, z=2)
sem_title = ax.text(85, 82.5, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
                    fontsize=7.0, fontweight="bold", color=SD_, zorder=6)

# ---- 中央接口桥（仅浅绿 IFACE 填，无 edge） ----
psd_style.card(ax, 52.0, 14, 6.0, 72, fill=psd_style.IFACE_FILL, edge=None,
               rounding=1.4, z=3)
psd_style.flowline(ax, 47.0, 44, 63.0, 44, color="#333333", lw=1.2, head=10, z=4)
ax.text(55, 50, "embeddings", ha="center", va="center", fontsize=5.8,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
ax.text(55, 47, "+ proposals", ha="center", va="center", fontsize=5.8,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

# ---- 物理层（Φ）：入口卡 -> 双轨道卡 -> 顶出到接口桥（零交叉拓扑承 v13） ----
# C+: 卡内留白 +20%；卡片宽 20（原来 19）；padding 2.5 单位
wcard(4.0, 17, 42, 5.5, "Unlabeled streams $(T$, 24, 3)",
      P_CARD, PD, title_fs=6.0)
wcard(4.0, 32, 20.5, 11, "SSL\npretraining", P_CARD, PD, title_fs=6.4)
wcard(25.5, 32, 20.5, 11, "Motion words\nquantization", P_CARD, PD, title_fs=6.3)
wcard(4.0, 52, 20.5, 9, "Dynamics\nembeddings", P_CARD, PD, title_fs=6.4)
wcard(25.5, 52, 20.5, 9, "Behavior\nproposals", P_CARD, PD, title_fs=6.4)
# C+ 主流线 1.2
psd_style.flowline(ax, 14, 22.5, 14, 32, color=PD, lw=1.2)
psd_style.flowline(ax, 35.75, 22.5, 35.75, 32, color=PD, lw=1.2)
psd_style.flowline(ax, 14, 43, 14, 52, color=PD, lw=1.2)
psd_style.flowline(ax, 35.75, 43, 35.75, 52, color=PD, lw=1.2)
psd_style.flowline(ax, 14, 61, 14, 73, color=PD, lw=1.2)
psd_style.flowline(ax, 14, 73, 52, 73, color=PD, lw=1.2)
psd_style.flowline(ax, 35.75, 61, 35.75, 69, color=PD, lw=1.2)
psd_style.flowline(ax, 35.75, 69, 52, 69, color=PD, lw=1.2)

# ---- 语义层（Ω）：入口卡 + 内部链卡 + 唯一焦点深块（Classification under Y） ----
wcard(64, 17, 42, 5.5, "Rule-engine seeds (budget $B$)", S_CARD, SD_, title_fs=6.0)
wcard(64, 27, 42, 5.5, "Anchor learning", S_CARD, SD_, title_fs=6.4)
wcard(64, 37, 42, 10, "Prototype clustering +\npseudo-labels",
      S_CARD, SD_, title_fs=6.4)
wcard(64, 51, 42, 8, "Semi-supervised\nself-training (warm start)",
      S_CARD, SD_, title_fs=6.0)
deepblock(64, 64, 42, 10, r"Classification under $\mathcal{Y}$", deep=SD_,
          title_fs=7.5)
psd_style.flowline(ax, 85, 22.5, 85, 27, color=SD_, lw=1.2)
psd_style.flowline(ax, 85, 32.5, 85, 37, color=SD_, lw=1.2)
psd_style.flowline(ax, 85, 47, 85, 51, color=SD_, lw=1.2)
psd_style.flowline(ax, 85, 59, 85, 64, color=SD_, lw=1.2)

# ---- 演化标注带（C+：不再是 DEEP，改浅填 S_BAR，留焦点给 Classification） ----
psd_style.card(ax, 22, 4.0, 66, 6.5, fill=S_BAR, edge=None, rounding=1.4, z=5)
ax.text(55, 7.25, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.0, color=SD_,
        fontweight="normal", zorder=7)
# C+ 虚线 0.7：仅 Taxonomy 回路
psd_style.flowline(ax, 85, 10.5, 85, 17, color=SD_, lw=0.7, ls=(0, (3, 2)),
                   head=8, z=4)

pdf_path = OUT / "fig1_framework_overview.pdf"
fig.savefig(pdf_path)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600)
fig.savefig(PREVIEW / "preview_fig1.png", dpi=150)
print("fig1 v16 (direction C+ intensified) saved")

import make_common_gates as gates
gates.gate_print_robustness(
    "fig1-v16", {"physics": psd_style.PHYS_EDGE, "semantic": psd_style.SEM_EDGE},
    redundancy="tone ladder + position + labels")
gates.gate_labels(fig, ax, list(ax.texts), name="fig1-v16")
gates.gate_typography(fig, min_pt=5.5, name="fig1-v16")
gates.gate_marks_in_axes(fig, name="fig1-v16")
gates.gate_pdf(pdf_path, max_w_pt=246.3)

# ---- 本地净空自检（不进共享模块）：容器内文字必须与容器边框留 >= 半个字号 ----
# G2 只断言"标签在 axes 内"，不覆盖"标签在自己所属面板内"——v15 的 6.8pt
# 面板标题实测已越出面板边框两侧各 ~0.9 单位（小于半个字号 1.22），本版修正。
fig.canvas.draw()
ren = fig.canvas.get_renderer()
inv = ax.transData.inverted()


def _xspan(t):
    b = t.get_window_extent(ren)
    (x0, _), (x1, _) = inv.transform([[b.x0, b.y0], [b.x1, b.y1]])
    return x0, x1


checks = [
    ("Physics layer title", phy_title, 1.0, 49.0),
    ("Semantic layer title", sem_title, 61.0, 109.0),
]
bad = []
for label, t, bx0, bx1 in checks:
    x0, x1 = _xspan(t)
    half = t.get_fontsize() / 2.0 / 72.0 * (110.0 / FIG_W)   # 半个字号（数据单位）
    left, right = x0 - bx0, bx1 - x1
    ok = left >= half and right >= half
    print("[local clearance] %-22s span %.2f..%.2f | panel %.0f..%.0f | "
          "clear L=%.2f R=%.2f (need >=%.2f) -> %s"
          % (label, x0, x1, bx0, bx1, left, right, half, "PASS" if ok else "FAIL"))
    if not ok:
        bad.append(label)
assert not bad, f"panel-title clearance failed: {bad}"
print("[local clearance] fig1-v16: 2 panel titles -> PASS")