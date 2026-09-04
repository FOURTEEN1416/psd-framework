# -*- coding: utf-8 -*-
"""fig3 v2 — SMQ 分割边界定性图（期刊印刷尺寸设计版, 2026-09-05）。

v1 教训（R8 视觉验收）: 10.8in 画布缩到 8.6cm 栏宽落地 2.6pt 不可读; zoom 列在
印刷尺寸下无增益。v2: figsize=(3.42, 3.45)in 1:1 印刷, 字号即落地字号;
布局 = Episode 1/2 双时间轴（上=种子伪 GT 橙, 下=SMQ 预测青）+ 底部 4-episode
聚合柱状。窄段标签动态宽度门保留（文字放不下即隐藏, 防碰撞）。
数据来源不变: reports/p02-smq-iou-eC-seeds-recheck.json（E-C 端到端 K=8, epoch-30）。
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "reports" / "p02-smq-iou-eC-seeds-recheck.json"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
assert DATA.is_file(), f"data missing: {DATA}"

INK = "#000000"
CYAN_FILL, CYAN_DARK = "#DAFFFF", "#0E7490"
ORANGE_FILL, ORANGE_DARK = "#FFE3DA", "#C2410C"
BAND_GRAY, BASE_EDGE = "#DADADA", "#AAAAAA"
NOTE_GRAY, GRID_GRAY = "#888888", "#DADADA"

plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "text.color": INK})

with open(DATA, encoding="utf-8") as f:
    payload = json.load(f)
episodes = {ep["id"]: ep for ep in payload["episodes"]}
EP_MAIN = [1, 4]
BAR_H = 0.62
Y_GT, Y_PRED = 1.0, 0.0

# 印刷尺寸下 1 数据帧 ≈ 3.42in*frac/T; 标签宽≈len*0.045in → 换算帧数门限按轴宽动态算
def draw_track(ax, segments, y_center, facecolor, edgecolor, labels=None, t_total=1.0):
    ax_w_in = 3.05  # 轴可视宽近似 (figure 3.42 - 标签/边距)
    frames_per_in = t_total / ax_w_in
    for i, (s, e) in enumerate(segments):
        ax.broken_barh([(s, e - s)], (y_center - BAR_H / 2, BAR_H),
                       facecolors=facecolor, edgecolors=edgecolor, linewidth=0.6)
        if labels and i < len(labels) and labels[i]:
            need = len(labels[i]) * 0.042 * frames_per_in + 30  # 5.5pt 字宽估算
            if (e - s) >= max(40, need):
                ax.text((s + e) / 2, y_center, labels[i], ha="center", va="center",
                        fontsize=5.5, color=INK, clip_on=True)


def style_timeline(ax, ep, tag):
    t = ep["T"]
    ax.set_xlim(0, t)
    ax.set_ylim(-0.72, 1.78)
    ax.set_yticks([Y_PRED, Y_GT])
    ax.set_yticklabels(["SMQ pred.", "Pseudo-GT"], fontsize=6)
    ax.set_xticks([0, t // 2, t])
    ax.tick_params(labelsize=5.5)
    ax.set_title(f"Episode {tag} \u00b7 T={t} \u00b7 IoU {ep['mean_matched_iou']:.3f} "
                 f"(baseline {ep['random_baseline_iou']:.3f})", loc="left", fontsize=6.5, pad=3)
    ax.grid(True, axis="x", color=GRID_GRAY, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.7)


fig = plt.figure(figsize=(3.42, 3.45))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.15], hspace=0.62)

for row, eid in enumerate(EP_MAIN):
    ep = episodes[eid]
    ax = fig.add_subplot(gs[row, 0])
    draw_track(ax, ep["gt_segments"], Y_GT, ORANGE_FILL, ORANGE_DARK,
               labels=ep.get("gt_labels"), t_total=ep["T"])
    draw_track(ax, ep["pred_segments"], Y_PRED, CYAN_FILL, CYAN_DARK, t_total=ep["T"])
    style_timeline(ax, ep, eid)

axi = fig.add_subplot(gs[2, 0])
ids = sorted(episodes)
iou_pred = [episodes[i]["mean_matched_iou"] for i in ids]
iou_base = [episodes[i]["random_baseline_iou"] for i in ids]
xs = range(len(ids))
bw = 0.36
bars_p = axi.bar([x - bw / 2 for x in xs], iou_pred, width=bw,
                 facecolor=CYAN_FILL, edgecolor=CYAN_DARK, linewidth=0.9,
                 label="SMQ (E-C, K=8)")
bars_b = axi.bar([x + bw / 2 for x in xs], iou_base, width=bw,
                 facecolor=BAND_GRAY, edgecolor=BASE_EDGE, linewidth=0.9, hatch="///",
                 label="Random baseline")
for rect, v in zip(bars_p, iou_pred):
    axi.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=5.5, color=INK)
for rect, v in zip(bars_b, iou_base):
    axi.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=5.5, color=NOTE_GRAY)
agg = payload["aggregate"]
n_win = sum(p > b for p, b in zip(iou_pred, iou_base))
axi.text(0.02, 0.96,
         f"agg {agg['mean_matched_iou']:.4f}\u00b1{agg['std']:.4f} \u00b7 {n_win}/4 > baseline",
         transform=axi.transAxes, ha="left", va="top", fontsize=6, color=INK)
axi.set_xticks(list(xs))
axi.set_xticklabels([f"Ep {i}" for i in ids], fontsize=6)
axi.set_ylabel("Matched IoU", fontsize=6.5)
axi.tick_params(labelsize=5.5)
axi.set_ylim(0, 0.72)
axi.grid(True, axis="y", color=GRID_GRAY, linewidth=0.5)
axi.set_axisbelow(True)
for side in ("top", "right"):
    axi.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axi.spines[side].set_linewidth(0.7)
axi.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6)

pdf_path = OUT_DIR / "fig3_segmentation_qualitative.pdf"
png_path = OUT_DIR / "fig3_segmentation_qualitative.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
print("written:", pdf_path)
for i in ids:
    ep = episodes[i]
    print(f"ep{i}: T={ep['T']} IoU={ep['mean_matched_iou']} baseline={ep['random_baseline_iou']}")
print(f"aggregate: {agg['mean_matched_iou']} \u00b1 {agg['std']} | {n_win}/4")
