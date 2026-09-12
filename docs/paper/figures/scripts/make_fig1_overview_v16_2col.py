# -*- coding: utf-8 -*-
r"""fig1 双栏版（v16_2col）— PSD 框架总览 7in 资产 · 风格方向 **C+ 强化**。

由 v14_2col 按 C+ 规格同步（与单栏正典 v16 同代），相对 v14_2col 的差异：

**C+ 收紧**：
1. 笔触更细：层容器发丝边 0.7→0.4；模块卡去边（白底+彩边 1.1/1.2 → 同相浅底
   mix 0.78 无边，与单栏 v16 同一色阶语言）；流线 1.9→1.4；接口主箭 2.2→1.4；
   演化虚线 1.5→0.9。
2. 焦点块 2→1：v14_2col 同时深色化 deepblock 与 Taxonomy 演化带；C+ 把
   Taxonomy 带降为浅阶（S_BAR 同相 mix 0.82 + 文字深化色），唯一焦点留给
   Classification under Y（与单栏 v16 处置一致）。
3. 字号阶梯更大：面板标题 8.6→8.8；副文 6.2→6.4；微图标卡副文 6.2→6.4；
   演化带文字 8.0→8.2。下限 5.5pt（本图最小 6.4）。
4. 尺度修复：去 bbox_inches="tight"，固定画布 7.0x2.95in ⇒ MediaBox 严格
   504.00x212.40pt（=7in 整）。

**承 v14_2col 不变**：坐标系与拓扑逐点同构（110x84 零交叉正交路由）、
模块微图标（make_icon_glyphs 共享库）、全部图内文字逐字保留、色板语义。
新资产未入 main.tex，caption 零耦合。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "docs" / "paper" / "figures"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
import make_icon_glyphs as ic
psd_style.apply_style()

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PE, SE_ = psd_style.PHYS_EDGE, psd_style.SEM_EDGE
PT, ST_ = psd_style.PHYS_TINT, psd_style.SEM_TINT


def _mix(c1, c2, f):
    """c1 向 c2 混合 f（0=c1, 1=c2），返回 hex。局部助手，不动 psd_style。"""
    c1 = c1.lstrip("#"); c2 = c2.lstrip("#")
    return "#%02X%02X%02X" % tuple(
        round(int(c1[i:i+2], 16) * (1 - f) + int(c2[i:i+2], 16) * f)
        for i in (0, 2, 4))


# C+ 色阶：模块卡同相浅底（在层容器 tint 之上再向白抬一阶）
P_CARD = _mix(PT, "#FFFFFF", 0.62)
S_CARD = _mix(ST_, "#FFFFFF", 0.62)
# Taxonomy 演化带浅阶（原 SD_ 满块 -> C+ 浅阶，焦点纪律 2->1）
S_BAR = _mix(SD_, "#FFFFFF", 0.82)

fig, ax = plt.subplots(figsize=(7.0, 2.95))
ax.set_xlim(0, 110); ax.set_ylim(0, 84)
ax.axis("off")


def wcard(x, y, w, h, title, sub=None, deep="#333333", fill="#FFFFFF", z=5,
          title_fs=7.6, sub_fs=6.4, glyph=None, gcolor=None):
    """C+ 模块卡：同相浅底、无边（仅靠色阶与图标色分层），图标置卡内顶部。"""
    psd_style.card(ax, x, y, w, h, fill=fill, edge=None, rounding=1.6, z=z)
    if glyph is not None and h >= 8:
        gs = min(w, h) * 0.30
        glyph(ax, x + w / 2, y + h - 1.2 - gs / 2, gs, gcolor or deep)
        ty, sy = y + h * 0.24, y + h * 0.06
    elif glyph is not None:  # 矮卡：图标居左，文字右移
        gs = min(w, h) * 0.55
        glyph(ax, x + 4.2, y + h / 2, gs, gcolor or deep)
        ty, sy = y + h * 0.63, y + h * 0.25
        x = x + 5.6
    else:
        ty, sy = y + h * 0.63, y + h * 0.25
    if sub:
        ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=deep, zorder=z + 2, linespacing=1.15)
        ax.text(x + w / 2, sy, sub, ha="center", va="center", fontsize=sub_fs,
                color="#555555", zorder=z + 2)
    else:
        ax.text(x + w / 2, y + h / 2 if glyph is None else ty,
                title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=deep, zorder=z + 2, linespacing=1.15)


def deepblock(x, y, w, h, title, deep="#333333", z=5, title_fs=8.0, glyph=None):
    """DEEP 满色白字焦点模块（唯一焦点；微图标白描边居左）。"""
    psd_style.card(ax, x, y, w, h, fill=deep, edge=None, rounding=1.6, z=z)
    if glyph is not None:
        glyph(ax, x + 3.6, y + h / 2, h * 0.56, "white", lw=0.8)
        x = x + 4.0
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color="white", zorder=z + 2, linespacing=1.15)


# ---- 层容器（C+ 发丝边 0.7 -> 0.4） ----
psd_style.card(ax, 2, 14, 44, 62, fill=PT, edge="#C0D6DB", lw=0.4, rounding=2.4, z=2)
ax.text(24, 73.6, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=PD, zorder=6)
psd_style.card(ax, 64, 14, 44, 62, fill=ST_, edge="#E7CABB", lw=0.4, rounding=2.4, z=2)
ax.text(86, 73.6, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=SD_, zorder=6)

# ---- 中央接口桥 + 唯一接口箭头 + 横排边标签（C+：主箭 2.2 -> 1.4） ----
psd_style.card(ax, 52.5, 14, 7, 62, fill=psd_style.IFACE_FILL, edge=None,
               rounding=1.6, z=3)
psd_style.flowline(ax, 46.8, 44.8, 63.4, 44.8, color="#333333", lw=1.4, head=12, z=4)
ax.text(55.2, 50.6, "embeddings", ha="center", va="center", fontsize=6.4,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9))
ax.text(55.2, 47.4, "+ proposals", ha="center", va="center", fontsize=6.4,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9))

# ---- 物理层（C+：同相浅底卡去边；流线 1.9 -> 1.4） ----
wcard(4.0, 17.0, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", deep=PD,
      fill=P_CARD, title_fs=7.0, glyph=ic.glyph_stack, gcolor=PD)
wcard(4.5, 32.5, 19.5, 9.5, "SSL\npretraining", deep=PD, fill=P_CARD, title_fs=7.4,
      glyph=ic.glyph_layers, gcolor=PD)
wcard(25, 32.5, 19.5, 9.5, "Motion words\nquantization", deep=PD, fill=P_CARD,
      title_fs=7.0, glyph=ic.glyph_grid_dots, gcolor=PD)
wcard(4.8, 50.5, 19.0, 7.5, "Dynamics\nembeddings", deep=PD, fill=P_CARD,
      title_fs=7.2, glyph=ic.glyph_wave, gcolor=PD)
wcard(24.0, 50.5, 19.0, 7.5, "Behavior\nproposals", deep=PD, fill=P_CARD,
      title_fs=7.2, glyph=ic.glyph_brackets, gcolor=PD)
psd_style.flowline(ax, 14.25, 22.6, 14.25, 32.3, color=PD, lw=1.4)
psd_style.flowline(ax, 34.75, 22.6, 34.75, 32.3, color=PD, lw=1.4)
psd_style.flowline(ax, 14.25, 42.2, 14.25, 50.3, color=PD, lw=1.4)
psd_style.flowline(ax, 34.75, 42.2, 34.75, 50.3, color=PD, lw=1.4)
psd_style.flowline(ax, 14.3, 58.2, 14.3, 68.8, color=PD, lw=1.4)
psd_style.flowline(ax, 14.3, 68.8, 52.3, 68.8, color=PD, lw=1.4)
psd_style.flowline(ax, 33.5, 58.2, 33.5, 65.4, color=PD, lw=1.4)
psd_style.flowline(ax, 33.5, 65.4, 52.3, 65.4, color=PD, lw=1.4)

# ---- 语义层（C+：同相浅底卡去边） ----
wcard(65.5, 17.0, 41.5, 5.6, "Rule-engine seeds (budget $B$)", deep=SD_,
      fill=S_CARD, title_fs=7.0, glyph=ic.glyph_doc_lines, gcolor=SD_)
wcard(66, 26.5, 40, 5.6, "Anchor learning", deep=SD_, fill=S_CARD, title_fs=7.6,
      glyph=ic.glyph_anchor_pin, gcolor=SD_)
wcard(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", deep=SD_,
      fill=S_CARD, title_fs=7.6, glyph=ic.glyph_cluster, gcolor=SD_)
wcard(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)", deep=SD_,
      fill=S_CARD, title_fs=7.0, glyph=ic.glyph_circular_arrow, gcolor=SD_)
deepblock(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", deep=SD_,
          title_fs=7.8, glyph=ic.glyph_tag)
psd_style.flowline(ax, 86, 22.8, 86, 26.3, color=SD_, lw=1.4)
psd_style.flowline(ax, 86, 32.3, 86, 36.3, color=SD_, lw=1.4)
psd_style.flowline(ax, 86, 45.7, 86, 49.3, color=SD_, lw=1.4)
psd_style.flowline(ax, 86, 56.7, 86, 60.3, color=SD_, lw=1.4)

# ---- 演化标注带（C+：SD_ 满块 -> 浅阶 S_BAR，焦点 2 -> 1） ----
psd_style.card(ax, 24, 2.0, 62, 6.0, fill=S_BAR, edge=None, rounding=1.6, z=5)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=8.2, color=SD_,
        fontweight="bold", zorder=7)
psd_style.flowline(ax, 84, 8.3, 84, 16.6, color=SD_, lw=0.9, ls=(0, (3, 2)), head=10)

# ---- 固定画布（C+：去 tight，MediaBox 严格 = figsize = 504x212.4pt） ----
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
pdf_path = OUT / "fig1_framework_overview_2col.pdf"
png_path = OUT / "fig1_framework_overview_2col.png"
prev_path = ROOT / "reports" / "figB-proposal" / "preview_fig1_2col.png"
prev_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(prev_path, dpi=150)

# ---- 门禁（C+ 补 G5/G6，与单栏正典同口径） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig1-2col-v16", {"physics": psd_style.PHYS_EDGE, "semantic": psd_style.SEM_EDGE},
    redundancy="tone ladder + position + labels + glyph icons")
gates.gate_labels(fig, ax, list(ax.texts), name="fig1-2col-v16")
gates.gate_typography(fig, min_pt=5.5, name="fig1-2col-v16")
gates.gate_marks_in_axes(fig, name="fig1-2col-v16")
gates.gate_pdf(pdf_path, max_w_pt=504.1)

# ---- 本地净空自检（承单栏 v16：面板标题 vs 面板边框，G2 覆盖不到的类别） ----
fig.canvas.draw()
ren = fig.canvas.get_renderer()
inv = ax.transData.inverted()
for name, t, x0, x1 in (
        ("Physics layer title", ax.texts[0], 2, 46),
        ("Semantic layer title", ax.texts[1], 64, 108)):
    bb = t.get_window_extent(ren)
    (bx0, by0), (bx1, by1) = inv.transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    cl = min(bx0 - x0, x1 - bx1)
    print("[local clearance] %-22s span %.2f..%.2f | panel %d..%d | clear L=%.2f R=%.2f"
          % (name, min(bx0, bx1), max(bx0, bx1), x0, x1,
             min(bx0, bx1) - x0, x1 - max(bx0, bx1)))
    assert cl >= 1.56, f"{name} 与面板边框净空不足（{cl:.2f} < 1.56 单位）"
print("[local clearance] fig1-2col-v16: 2 panel titles -> PASS")

print("fig1 v16_2col (C+ intensified, fixed canvas 504pt) saved:", pdf_path)
