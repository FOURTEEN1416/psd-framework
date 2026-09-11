# -*- coding: utf-8 -*-
"""graphical abstract v13 — Elsevier GA 规格（R43b 顶刊级冲刺 · 承 v12 构图，保留率条加基准轨）。
diagram-design remove test + 密度预算；三段式构图与保留率条承 v11）。

规格: ≤531×131 pt ≈ 7.375×1.82 in,印刷尺寸 1:1 铁律(figsize 直接落)，
  PNG 600dpi + PDF 矢量。

相对 v11 的重绘点：
1. 删 Ω 块内装饰性弧线自环（remove test：不承载信息；"taxonomy evolves 𝒴→𝒴′"
   文字行已完整表达）；
2. Ω 块补机制行 "anchor-guided pseudo-labeling · self-training"（源忠实于
   fig1 caption/方法节），双块密度平衡至 ~4/10；
3. 双块文字垂直重排消中空死区；脚注 4.6→5.1pt 并精简。
保留率三条数值 90.7/88.9/28.9 与语义色（NTU60 青绿/NTU120 浅青/canine 灰底橙描边
斜纹）锁死不动。数据来源: 正文 04-experiments.tex E9/E9b/E7 段(10-seed 终口径)。
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# ---- 画布:印刷尺寸 1:1 ----
fig, ax = plt.subplots(figsize=(7.375, 1.82))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, 100)
ax.set_ylim(0, 24.7)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

# ---- ① 双柔色块(DEEP 标题 + 深灰内容, 无边) ----
psd_style.card(ax, 2, 4, 29, 16.5, fill=psd_style.PHYS_TINT, edge=None,
               rounding=2.4, z=3)
ax.text(16.5, 16.9, r"Physics layer $\Phi$", ha="center", va="center",
        fontsize=9, fontweight="bold", color=PD, zorder=6)
ax.text(16.5, 13.7, "(frozen after pretraining)", ha="center", va="center",
        fontsize=6.3, style="italic", color="#444444", zorder=6)
ax.text(16.5, 10.6, "self-supervised dynamics", ha="center", va="center",
        fontsize=6.0, color="#333333", zorder=6)
ax.text(16.5, 7.9, "behavior proposals", ha="center", va="center",
        fontsize=6.0, color="#333333", zorder=6)

psd_style.card(ax, 40, 4, 29, 16.5, fill=psd_style.SEM_TINT, edge=None,
               rounding=2.4, z=3)
ax.text(54.5, 16.9, r"Semantic layer $\Omega$", ha="center", va="center",
        fontsize=9, fontweight="bold", color=SD_, zorder=6)
ax.text(54.5, 13.7, "(revisable)", ha="center", va="center",
        fontsize=6.3, style="italic", color="#444444", zorder=6)
ax.text(54.5, 10.6, r"taxonomy evolves $\mathcal{Y}\to\mathcal{Y}'$: only $\Omega$ retrains",
        ha="center", va="center", fontsize=5.8, color=SD_, fontweight="bold", zorder=6)
ax.text(54.5, 7.9, "anchor-guided pseudo-labeling \u00b7 self-training",
        ha="center", va="center", fontsize=6.0, color="#333333", zorder=6)

# 接口箭头(粗流线, 物理→语义)
psd_style.flowline(ax, 31.8, 12.2, 39.2, 12.2, color=C_ARROW, lw=2.4, head=14)

# ---- ② 保留率微条(圆角柔色条) ----
PANEL_X0, BAR_X0, BAR_MAX = 71.5, 81.5, 17.5   # 100% → 17.5 单位
rows = [
    ("NTU60", 90.7, False, psd_style.S_NTU60, psd_style.INK),      # 绿条浅, 墨字
    ("NTU120", 88.9, False, psd_style.S_NTU120, psd_style.INK),  # 浅青条墨字(白字对比不足)
    ("canine", 28.9, True, psd_style.GRAY_FILL, SD_),   # E7 v1 @13%-label 预算: 保留率 28.9%, 绝对精度近随机
]
ax.text((PANEL_X0 + 99) / 2, 21.0, "budget retention", ha="center", va="center",
        fontsize=6.3, fontweight="bold", color=C_TEXT)
ys = (15.6, 10.8, 6.0)
for y in ys:  # 0-100% 基准轨（浅灰, 在彩条之下; 条高与下方彩条同为 3.0）
    psd_style.card(ax, BAR_X0, y, BAR_MAX, 3.0, fill="#F2F2F2", edge=None,
                   rounding=0.8, z=3)
ax.text(BAR_X0 + BAR_MAX - 0.4, ys[0] + 3.0 + 1.1, "100%", ha="right", va="bottom",
        fontsize=4.8, color=C_NOTE, zorder=6)
BAR_H = 3.0
for (label, val, is_boundary, bcol, txtcol), y in zip(rows, ys):
    ax.text(BAR_X0 - 1.2, y + BAR_H / 2, label, ha="right", va="center",
            fontsize=5.8, color=C_TEXT)
    w = BAR_MAX * val / 100.0
    if is_boundary:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=SD_, lw=1.0,
                       rounding=0.8, z=4, hatch="///")
        ax.text(BAR_X0 + w + 1.0, y + BAR_H / 2, f"{val:.1f}%",
                ha="left", va="center", fontsize=5.4, color=txtcol, zorder=6)
    else:
        psd_style.card(ax, BAR_X0, y, w, BAR_H, fill=bcol, edge=None,
                       rounding=0.8, z=4)
        ax.text(BAR_X0 + w - 0.8, y + BAR_H / 2, f"{val:.1f}%", ha="right", va="center",
                fontsize=5.6, color=txtcol, fontweight="bold", zorder=6)
ax.text((PANEL_X0 + 99) / 2, 3.0,
        "retention of own full-budget top-1 at 10% labels;",
        ha="center", va="center", fontsize=5.1, color=C_NOTE)
ax.text((PANEL_X0 + 99) / 2, 1.0,
        "canine: 9.8 vs 11.1% chance at 13% budget",
        ha="center", va="center", fontsize=5.1, color=C_NOTE)

pdf_path = OUT / "fig_ga_graphical_abstract.pdf"
png_path = OUT / "fig_ga_graphical_abstract.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0)
plt.close(fig)

assert pdf_path.is_file() and png_path.is_file(), "GA outputs missing"

# ---- 出图门禁（任务包 A 第三轮内建：G1 + G4(531x131pt)） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "GA", {"NTU60": psd_style.S_NTU60, "NTU120": psd_style.S_NTU120,
           "canine": psd_style.GRAY_FILL},
    redundancy="direct text labels + hatch")
gates.gate_pdf(pdf_path, max_w_pt=531, max_h_pt=131)

print("written:", pdf_path)
print("written:", png_path)
print("GA v13 (top-journal polish: retention rail + 100% tick) saved")
print("print size: 7.375 x 1.82 in (= 531 x 131 pt, Elsevier GA limit)")
