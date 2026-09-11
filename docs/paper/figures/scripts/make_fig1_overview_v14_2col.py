# -*- coding: utf-8 -*-
r"""fig1 双栏版（v14_2col）— PSD 框架总览 7in 资产（任务包 A 第三轮 · 候选 c）。

新资产 fig1_framework_overview_2col.pdf/png（单栏版 v13 不动，是否采用由用户裁决）。
相对 v13：坐标系与拓扑逐点同构（110×84 零交叉正交路由 / 焦点深块 ≤2 /
演化带垂直虚线），物理画布 3.42→7.0in，解锁：
1. 模块微图标（make_icon_glyphs 共享库，描边极简，家族深化色）——fig-redraw
   报告 §七"有意不做清单"项由双栏版面正式解锁；
2. 字号整体上调一档（卡片标题 6.3→7.6pt，副文 5.2→6.2pt），印刷有效字号充裕；
3. 层容器发丝边框与接口桥横排标签承 v13 原样。
色板/语义/拓扑/caption 兼容性：与 v13 同构，caption 措辞零耦合（新资产未入 main.tex）。
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

fig, ax = plt.subplots(figsize=(7.0, 2.95))
ax.set_xlim(0, 110); ax.set_ylim(0, 84)
ax.axis("off")

def wcard(x, y, w, h, title, sub=None, deep="#333333", edge=None, z=5,
          title_fs=7.6, sub_fs=6.2, glyph=None, gcolor=None):
    """白色模块卡（微图标版）：图标置卡内顶部，标题/副文下排。"""
    psd_style.card(ax, x, y, w, h, fill="white", edge=edge,
                   lw=1.1 if edge is not None else 1.2,
                   rounding=1.6, z=z)
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
    """DEEP 满色白字焦点模块（微图标白描边居左）。"""
    psd_style.card(ax, x, y, w, h, fill=deep, edge=None, rounding=1.6, z=z)
    if glyph is not None:
        glyph(ax, x + 3.6, y + h / 2, h * 0.56, "white", lw=0.8)
        x = x + 4.0
    ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color="white", zorder=z + 2, linespacing=1.15)

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PE, SE_ = psd_style.PHYS_EDGE, psd_style.SEM_EDGE
PT, ST_ = psd_style.PHYS_TINT, psd_style.SEM_TINT

# ---- 层容器（同相发丝边框, 承 v13） ----
psd_style.card(ax, 2, 14, 44, 62, fill=PT, edge="#C0D6DB", lw=0.7, rounding=2.4, z=2)
ax.text(24, 73.6, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=8.6, fontweight="bold", color=PD, zorder=6)
psd_style.card(ax, 64, 14, 44, 62, fill=ST_, edge="#E7CABB", lw=0.7, rounding=2.4, z=2)
ax.text(86, 73.6, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
        fontsize=8.6, fontweight="bold", color=SD_, zorder=6)

# ---- 中央接口桥 + 唯一接口箭头 + 横排边标签 ----
psd_style.card(ax, 52.5, 14, 7, 62, fill=psd_style.IFACE_FILL, edge=None,
               rounding=1.6, z=3)
psd_style.flowline(ax, 46.8, 44.8, 63.4, 44.8, color="#333333", lw=2.2, head=13, z=4)
ax.text(55.2, 50.6, "embeddings", ha="center", va="center", fontsize=6.2,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9))
ax.text(55.2, 47.4, "+ proposals", ha="center", va="center", fontsize=6.2,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.9))

# ---- 物理层 ----
wcard(4.0, 17.0, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", deep=PD,
      edge=PE, title_fs=7.0, glyph=ic.glyph_stack, gcolor=PD)
wcard(4.5, 32.5, 19.5, 9.5, "SSL\npretraining", deep=PD, title_fs=7.4,
      glyph=ic.glyph_layers, gcolor=PD)
wcard(25, 32.5, 19.5, 9.5, "Motion words\nquantization", deep=PD, title_fs=7.0,
      glyph=ic.glyph_grid_dots, gcolor=PD)
wcard(4.8, 50.5, 19.0, 7.5, "Dynamics\nembeddings", deep=PD, title_fs=7.2,
      glyph=ic.glyph_wave, gcolor=PD)
wcard(24.0, 50.5, 19.0, 7.5, "Behavior\nproposals", deep=PD, title_fs=7.2,
      glyph=ic.glyph_brackets, gcolor=PD)
psd_style.flowline(ax, 14.25, 22.6, 14.25, 32.3, color=PD, lw=1.9)
psd_style.flowline(ax, 34.75, 22.6, 34.75, 32.3, color=PD, lw=1.9)
psd_style.flowline(ax, 14.25, 42.2, 14.25, 50.3, color=PD, lw=1.9)
psd_style.flowline(ax, 34.75, 42.2, 34.75, 50.3, color=PD, lw=1.9)
psd_style.flowline(ax, 14.3, 58.2, 14.3, 68.8, color=PD, lw=1.9)
psd_style.flowline(ax, 14.3, 68.8, 52.3, 68.8, color=PD, lw=1.9)
psd_style.flowline(ax, 33.5, 58.2, 33.5, 65.4, color=PD, lw=1.9)
psd_style.flowline(ax, 33.5, 65.4, 52.3, 65.4, color=PD, lw=1.9)

# ---- 语义层 ----
wcard(65.5, 17.0, 41.5, 5.6, "Rule-engine seeds (budget $B$)", deep=SD_,
      edge=SE_, title_fs=7.0, glyph=ic.glyph_doc_lines, gcolor=SD_)
wcard(66, 26.5, 40, 5.6, "Anchor learning", deep=SD_, title_fs=7.6,
      glyph=ic.glyph_anchor_pin, gcolor=SD_)
wcard(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", deep=SD_,
      title_fs=7.6, glyph=ic.glyph_cluster, gcolor=SD_)
wcard(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)", deep=SD_,
      title_fs=7.0, glyph=ic.glyph_circular_arrow, gcolor=SD_)
deepblock(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", deep=SD_,
          title_fs=7.8, glyph=ic.glyph_tag)
psd_style.flowline(ax, 86, 22.8, 86, 26.3, color=SD_, lw=1.9)
psd_style.flowline(ax, 86, 32.3, 86, 36.3, color=SD_, lw=1.9)
psd_style.flowline(ax, 86, 45.7, 86, 49.3, color=SD_, lw=1.9)
psd_style.flowline(ax, 86, 56.7, 86, 60.3, color=SD_, lw=1.9)

# ---- 演化标注带 + 纯垂直虚线回 seeds ----
psd_style.card(ax, 24, 2.0, 62, 6.0, fill=SD_, edge=None, rounding=1.6, z=5)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=8.0, color="white",
        fontweight="bold", zorder=7)
psd_style.flowline(ax, 84, 8.3, 84, 16.6, color=SD_, lw=1.5, ls=(0, (3, 2)), head=11)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
pdf_path = OUT / "fig1_framework_overview_2col.pdf"
png_path = OUT / "fig1_framework_overview_2col.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.01)

# ---- 出图门禁 ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig1-2col", {"physics": psd_style.PHYS_EDGE, "semantic": psd_style.SEM_EDGE},
    redundancy="container tint + position + labels")
gates.gate_labels(fig, ax, list(ax.texts), name="fig1-2col")
gates.gate_pdf(pdf_path, max_w_pt=514)  # 7.0in=504pt + pad 余量
print("fig1 v14_2col (7in + module micro-icons) saved:", pdf_path)
