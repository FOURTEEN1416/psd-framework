# -*- coding: utf-8 -*-
r"""fig1 v13 — PSD 框架总览（R43b 顶刊级冲刺 · 承 v12 零交叉拓扑，容器加同相发丝边框）。

相对 v11 的重绘点（构图/工艺改，色板与语义锁死承 psd_style 真源）：
1. 消连线交叉：物理层两轨道输出改"顶出→左低右高分高平移"正交路由——
   Dynamics embeddings 走上线 y=68.8、Behavior proposals 走下线 y=65.4，
   左横线越过右竖线顶端（65.4<68.8），全图零交叉（diagram-design 连线规则 3）；
2. 接口桥竖排文字（diagram-design 反模式）废除 → 唯一接口箭头（y=44.8）+
   桥上横排白底两行标签 "embeddings / + proposals"（studio-pro edge-label-first）；
3. 深色满块 4→2（出口 Classification + 演化带为焦点；入口两块降为白卡+家族色描边），
   恢复 psd_style "焦点强调每图 ≤2 处"纪律；
4. 演化带回线改纯垂直虚线（x=84），消 v11 斜虚线。
R43b: 双层容器加同相降饱和发丝边框（diagram-design "borders are in"），
容器边界从纯色块获得 0.6pt 精修轮廓；拓扑/焦点数承 v12 不变。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"D:\Desktop\psd-framework\docs\paper\figures")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()

fig, ax = plt.subplots(figsize=(3.42, 2.60))
ax.set_xlim(0, 110); ax.set_ylim(0, 84)
ax.axis("off")

def wcard(x, y, w, h, title, sub=None, deep="#333333", edge=None, z=5,
          title_fs=6.3, sub_fs=5.2):
    """白色模块卡（可选家族色描边）+ DEEP 标题(+灰副文)。R30: card 无阴影。"""
    psd_style.card(ax, x, y, w, h, fill="white", edge=edge,
                   lw=0.9 if edge is not None else 1.0,
                   rounding=1.2, z=z)
    if sub:
        ax.text(x+w/2, y+h*0.63, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=deep, zorder=z+2, linespacing=1.15)
        ax.text(x+w/2, y+h*0.25, sub, ha="center", va="center", fontsize=sub_fs,
                color="#555555", zorder=z+2)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=deep, zorder=z+2, linespacing=1.15)

def deepblock(x, y, w, h, title, deep="#333333", z=5, title_fs=6.3):
    """DEEP 满色白字模块（焦点，每图 ≤2 处）。"""
    psd_style.card(ax, x, y, w, h, fill=deep, edge=None, rounding=1.2, z=z)
    ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=title_fs,
            fontweight="bold", color="white", zorder=z+2, linespacing=1.15)

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PE, SE_ = psd_style.PHYS_EDGE, psd_style.SEM_EDGE
PT, ST_ = psd_style.PHYS_TINT, psd_style.SEM_TINT

# ---- 层容器（极浅柔色满块 + DEEP 标题内嵌顶部） ----
psd_style.card(ax, 2, 14, 44, 62, fill=PT, edge="#C0D6DB", lw=0.6, rounding=2.0, z=2)
ax.text(24, 73.6, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color=PD, zorder=6)
psd_style.card(ax, 64, 14, 44, 62, fill=ST_, edge="#E7CABB", lw=0.6, rounding=2.0, z=2)
ax.text(86, 73.6, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color=SD_, zorder=6)

# ---- 中央接口桥（浅绿 IFACE，语义锁）+ 唯一接口箭头 + 横排边标签 ----
psd_style.card(ax, 52.5, 14, 7, 62, fill=psd_style.IFACE_FILL, edge=None,
               rounding=1.2, z=3)
psd_style.flowline(ax, 46.8, 44.8, 63.4, 44.8, color="#333333", lw=1.8, head=11, z=4)
ax.text(55.2, 50.4, "embeddings", ha="center", va="center", fontsize=5.2,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))
ax.text(55.2, 47.6, "+ proposals", ha="center", va="center", fontsize=5.2,
        color="#333333", zorder=6,
        bbox=dict(facecolor="white", edgecolor="none", pad=0.8))

# ---- 物理层（入口白卡描边 → 双轨道白卡 → 分高平移出接口桥） ----
wcard(4.0, 17.0, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", deep=PD,
      edge=PE, title_fs=5.9)
wcard(4.5, 32.5, 19.5, 9.5, "SSL\npretraining", deep=PD, title_fs=6.3)
wcard(25, 32.5, 19.5, 9.5, "Motion words\nquantization", deep=PD, title_fs=6.0)
wcard(4.8, 50.5, 19.0, 7.5, "Dynamics\nembeddings", deep=PD, title_fs=6.3)
wcard(24.0, 50.5, 19.0, 7.5, "Behavior\nproposals", deep=PD, title_fs=6.3)
psd_style.flowline(ax, 14.25, 22.6, 14.25, 32.3, color=PD)
psd_style.flowline(ax, 34.75, 22.6, 34.75, 32.3, color=PD)
psd_style.flowline(ax, 14.25, 42.2, 14.25, 50.3, color=PD)
psd_style.flowline(ax, 34.75, 42.2, 34.75, 50.3, color=PD)
# 输出：左卡上行走高线 y=68.8，右卡上行走低线 y=65.4（左横线越过右竖线顶端→零交叉）
psd_style.flowline(ax, 14.3, 58.2, 14.3, 68.8, color=PD)
psd_style.flowline(ax, 14.3, 68.8, 52.3, 68.8, color=PD)
psd_style.flowline(ax, 33.5, 58.2, 33.5, 65.4, color=PD)
psd_style.flowline(ax, 33.5, 65.4, 52.3, 65.4, color=PD)

# ---- 语义层（入口白卡描边 → 白卡×3 → 出口焦点深块） ----
wcard(65.5, 17.0, 41.5, 5.6, "Rule-engine seeds (budget $B$)", deep=SD_,
      edge=SE_, title_fs=5.9)
wcard(66, 26.5, 40, 5.6, "Anchor learning", deep=SD_, title_fs=6.3)
wcard(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", deep=SD_, title_fs=6.3)
wcard(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)", deep=SD_, title_fs=6.0)
deepblock(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", deep=SD_, title_fs=6.6)
psd_style.flowline(ax, 86, 22.8, 86, 26.3, color=SD_)
psd_style.flowline(ax, 86, 32.3, 86, 36.3, color=SD_)
psd_style.flowline(ax, 86, 45.7, 86, 49.3, color=SD_)
psd_style.flowline(ax, 86, 56.7, 86, 60.3, color=SD_)

# ---- 演化标注带（焦点深块 #2 + 纯垂直虚线回 seeds） ----
psd_style.card(ax, 24, 2.0, 62, 6.0, fill=SD_, edge=None, rounding=1.2, z=5)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.6, color="white",
        fontweight="bold", zorder=7)
psd_style.flowline(ax, 84, 8.3, 84, 16.6, color=SD_, lw=1.3, ls=(0, (3, 2)), head=9)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig1_framework_overview.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig1 v13 (top-journal polish: hairline container borders) saved")
