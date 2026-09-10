# -*- coding: utf-8 -*-
r"""fig1 v10 — PSD 框架总览（R29 顶刊方法图语言：柔色满块容器+白卡 DEEP 层级文字+
粗圆头流线；坐标拓扑承 v9）。

顶刊（Nature/Cell 方法图）要素对照：
- 容器=柔色满块（NS 板提亮一档，无边，大圆角，软阴影）；
- 模块=白色扁平卡（无边感），标题 DEEP 系粗体+灰副文两级文字；
- 入口/出口模块=DEEP 满色白字（视觉锚点：Unlabeled streams / Rule-engine seeds /
  Classification）；
- 数据流=粗圆头流线 lw2.2（物理段蓝/语义段红/接口深灰黑）。
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

fig, ax = plt.subplots(figsize=(3.42, 2.62))
ax.set_xlim(0, 110); ax.set_ylim(0, 84)
ax.axis("off")

def wcard(x, y, w, h, title, sub=None, deep="#333333", z=5,
          title_fs=6.3, sub_fs=5.2):
    """白色模块卡 + DEEP 标题(+灰副文)。"""
    psd_style.card(ax, x, y, w, h, fill="white", edge=None,
                   rounding=1.6, dx=0.45, dy=-0.75, z=z)
    if sub:
        ax.text(x+w/2, y+h*0.63, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=deep, zorder=z+2, linespacing=1.15)
        ax.text(x+w/2, y+h*0.25, sub, ha="center", va="center", fontsize=sub_fs,
                color="#555555", zorder=z+2)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=deep, zorder=z+2, linespacing=1.15)

def deepblock(x, y, w, h, title, sub=None, deep="#333333", z=5,
              title_fs=6.3, sub_fs=5.2):
    """DEEP 满色白字模块（入口/出口锚点）。"""
    psd_style.card(ax, x, y, w, h, fill=deep, edge=None,
                   rounding=1.6, dx=0.45, dy=-0.75, z=z)
    tc = "white"
    if sub:
        ax.text(x+w/2, y+h*0.63, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=tc, zorder=z+2, linespacing=1.15)
        ax.text(x+w/2, y+h*0.25, sub, ha="center", va="center", fontsize=sub_fs,
                color="white", alpha=0.85, zorder=z+2)
    else:
        ax.text(x+w/2, y+h/2, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=tc, zorder=z+2, linespacing=1.15)

PD, SD_ = psd_style.PHYS_DEEP, psd_style.SEM_DEEP
PT, ST_ = psd_style.PHYS_TINT, psd_style.SEM_TINT

# ---- 层容器（柔色满块 + 阴影 + DEEP 标题内嵌顶部） ----
psd_style.card(ax, 2, 14, 44, 62, fill=PT, edge=None, rounding=2.4, dx=0.7, dy=-1.1, z=2)
ax.text(24, 73.0, r"Physics layer $\Phi$  (frozen)", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color=PD, zorder=6)
psd_style.card(ax, 64, 14, 44, 62, fill=ST_, edge=None, rounding=2.4, dx=0.7, dy=-1.1, z=2)
ax.text(86, 73.0, r"Semantic layer $\Omega$  (revisable)", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color=SD_, zorder=6)

# ---- 中央接口带（白卡窄条，深灰字） ----
psd_style.card(ax, 52.5, 14, 7, 62, fill="white", edge=None,
               rounding=1.8, dx=0.45, dy=-0.75, z=3)
ax.text(56, 45, "embeddings + proposals", ha="center", va="center",
        fontsize=5.8, color="#333333", rotation=90, zorder=5)

# ---- 物理层（入口 DEEP 块 → 白卡 → 输出白卡 → 接口带） ----
deepblock(4.0, 17.0, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", deep=PD, title_fs=5.9)
wcard(4.5, 32.5, 19.5, 9.5, "SSL\npretraining", deep=PD, title_fs=6.3)
wcard(25, 32.5, 19.5, 9.5, "Motion words\nquantization", deep=PD, title_fs=6.0)
wcard(4.8, 50.5, 19.0, 7.5, "Dynamics\nembeddings", deep=PD, title_fs=6.3)
wcard(24.0, 50.5, 19.0, 7.5, "Behavior\nproposals", deep=PD, title_fs=6.3)
psd_style.flowline(ax, 14.25, 22.6, 14.25, 32.3, color=PD)
psd_style.flowline(ax, 34.75, 22.6, 34.75, 32.3, color=PD)
psd_style.flowline(ax, 14.25, 42.2, 14.4, 50.3, color=PD)
psd_style.flowline(ax, 34.75, 42.2, 33.5, 50.3, color=PD)
psd_style.flowline(ax, 14.5, 58.2, 14.5, 63.0, color=PD)
psd_style.flowline(ax, 14.5, 63.0, 52.3, 63.0, color=PD)
psd_style.flowline(ax, 33.5, 58.2, 33.5, 68.3, color=PD)
psd_style.flowline(ax, 33.5, 68.3, 52.3, 68.3, color=PD)   # R29: 降线避让容器标题
psd_style.flowline(ax, 59.7, 39.0, 65.8, 39.0, color="#333333")  # 唯一正向接口箭头

# ---- 语义层（入口 DEEP 块 → 白卡×3 → 出口 DEEP 块） ----
deepblock(65.5, 17.0, 41.5, 5.6, "Rule-engine seeds (budget $B$)", deep=SD_, title_fs=5.9)
wcard(66, 26.5, 40, 5.6, "Anchor learning", deep=SD_, title_fs=6.3)
wcard(66, 36.5, 40, 9.0, "Prototype clustering +\npseudo-labels", deep=SD_, title_fs=6.3)
wcard(66, 49.5, 40, 7.0, "Semi-supervised\nself-training (warm start)", deep=SD_, title_fs=6.0)
deepblock(70, 60.5, 34, 5.6, r"Classification under $\mathcal{Y}$", deep=SD_, title_fs=6.6)
psd_style.flowline(ax, 86, 22.8, 86, 26.3, color=SD_)
psd_style.flowline(ax, 86, 32.3, 86, 36.3, color=SD_)
psd_style.flowline(ax, 86, 45.7, 86, 49.3, color=SD_)
psd_style.flowline(ax, 86, 56.7, 86, 60.3, color=SD_)

# ---- 演化标注带（DEEP 满色白字 + 虚流线落 seeds 块） ----
psd_style.card(ax, 24, 2.0, 62, 6.0, fill=SD_, edge=None,
               rounding=1.6, dx=0.45, dy=-0.75, z=5)
ax.text(55, 5.0, r"Taxonomy $\mathcal{Y} \to \mathcal{Y}'$:  only $\Omega$ retrains",
        ha="center", va="center", fontsize=6.6, color="white",
        fontweight="bold", zorder=7)
psd_style.flowline(ax, 84, 8.3, 86.0, 16.6, color=SD_, lw=1.6, ls=(0, (3, 2)), head=10)

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
fig.savefig(OUT / "fig1_framework_overview.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "fig1_framework_overview.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
print("fig1 v10 (R29 journal-style: tint containers + white cards + deep text + flowlines) saved")
