# -*- coding: utf-8 -*-
"""fig3 SMQ 分割边界定性可视化（预测边界 vs 种子伪 GT 边界对照）生成脚本。

数据来源: reports/p02-smq-iou-eC-seeds-recheck.json（P0.2 冲刺定稿 E-C 端到端 K=8,
  epoch-30 checkpoint；argmax 平局 bug 修复后的复核版，聚合值与原版一致 0.4577±0.0488）
  - 每个episode自带 gt_segments / gt_labels / pred_segments 边界序列——直接读 JSON 绘制，
    无需重跑推理（任务书降级预案未触发）。
设计（figure-specs §fig3 + 任务书 Step2）:
  - 两个episode的 上下两条时间轴 对照：上=种子伪 GT（淡橙，复用 fig1 配色族），
    下=SMQ 预测段（淡青）；右侧局部放大窗展示边界对齐细节（含类别文字标注）。
  - 底部全 4 episodes 的 per-episode matched-IoU 柱状面板（vs 随机基线）——
    「4/4 全超基线」的事实本身有展示价值，同时避免只挑好看episode的樱桃采摘嫌疑。
规格: 白底 / #DADADA 网格 / 色盲安全青橙对 / 无图内标题 / PDF 矢量 + PNG 600dpi。
输出: docs/paper/figures/fig3_segmentation_qualitative.pdf + .png(600dpi)
"""

from pathlib import Path

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle

# ---- 路径（脚本位于 <root>/docs/paper/figures/scripts/）----
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "reports" / "p02-smq-iou-eC-seeds-recheck.json"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
assert DATA.is_file(), f"data missing: {DATA}"

# ---- 配色（与 make_fig1_overview.py 同族，≤6 色）----
INK = "#000000"
CYAN_FILL = "#DAFFFF"
CYAN_DARK = "#0E7490"
ORANGE_FILL = "#FFE3DA"
ORANGE_DARK = "#C2410C"
BAND_GRAY = "#DADADA"
BASE_EDGE = "#AAAAAA"
NOTE_GRAY = "#888888"
GRID_GRAY = "#DADADA"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "text.color": INK,
})

with open(DATA, encoding="utf-8") as f:
    payload = json.load(f)

episodes = {ep["id"]: ep for ep in payload["episodes"]}
EP_MAIN = [1, 4]                       # 主展示：最难(受分辨率压制) + 最优，非单挑最好
ZOOM_WIN = {1: (650, 1020), 4: (980, 1260)}   # 边界对齐好/坏并存的代表性窗口

BAR_H = 0.62
Y_GT, Y_PRED = 1.0, 0.0


def draw_track(ax, segments, y_center, facecolor, edgecolor,
               labels=None, label_fs=6.0, label_min_dur=30, x_clip=None):
    """x_clip=(x0,x1) 时只画与窗口相交的段——防止 zoom 轴外的段/文字溢出到画布空白区。"""
    for i, (s, e) in enumerate(segments):
        if x_clip is not None and (e <= x_clip[0] or s >= x_clip[1]):
            continue
        ax.broken_barh([(s, e - s)], (y_center - BAR_H / 2, BAR_H),
                       facecolors=facecolor, edgecolors=edgecolor, linewidth=0.8)
        if labels and (e - s) >= label_min_dur:
            # zoom 窗口内只标注完全落在窗内的段，避免文字被轴缘截断
            fully_inside = x_clip is None or (s >= x_clip[0] and e <= x_clip[1])
            if fully_inside:
                ax.text((s + e) / 2, y_center, labels[i], ha="center", va="center",
                        fontsize=label_fs, color=INK, clip_on=True)


def style_timeline(ax, ep, tag):
    t = ep["T"]
    ax.set_xlim(0, t)
    ax.set_ylim(-0.72, 1.78)
    ax.set_yticks([Y_PRED, Y_GT])
    ax.set_yticklabels(["SMQ prediction", "Seed pseudo-GT"], fontsize=8)
    ax.set_xlabel("Frame index", fontsize=8.5)
    ax.set_title(
        f"Episode {tag} \u00b7 T={t} \u00b7 matched-IoU "
        f"{ep['mean_matched_iou']:.3f} (random baseline {ep['random_baseline_iou']:.3f})",
        loc="left", fontsize=9, pad=4)
    ax.grid(True, axis="x", color=GRID_GRAY, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


# ---- 版面：左列两条主时间轴 + 右列对应放大窗 + 底部跨列 IoU 面板 ----
fig = plt.figure(figsize=(10.8, 7.0))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 2, width_ratios=[3.3, 1.0],
                      height_ratios=[1.0, 1.0, 1.05],
                      hspace=0.78, wspace=0.18)

axes_main = {}
for row, eid in enumerate(EP_MAIN):
    ep = episodes[eid]
    ax = fig.add_subplot(gs[row, 0])
    draw_track(ax, ep["gt_segments"], Y_GT, ORANGE_FILL, ORANGE_DARK,
               labels=ep.get("gt_labels"), label_fs=6.0, label_min_dur=95)
    draw_track(ax, ep["pred_segments"], Y_PRED, CYAN_FILL, CYAN_DARK)
    style_timeline(ax, ep, eid)
    axes_main[eid] = ax

    # 放大窗（只画窗口内相交段，避免轴外文字溢出）
    w0, w1 = ZOOM_WIN[eid]
    axz = fig.add_subplot(gs[row, 1])
    draw_track(axz, ep["gt_segments"], Y_GT, ORANGE_FILL, ORANGE_DARK,
               labels=ep.get("gt_labels"), label_fs=6.0, label_min_dur=26,
               x_clip=(w0, w1))
    draw_track(axz, ep["pred_segments"], Y_PRED, CYAN_FILL, CYAN_DARK,
               x_clip=(w0, w1))
    axz.set_xlim(w0, w1)
    axz.set_ylim(-0.72, 1.78)
    axz.set_yticks([])
    axz.set_xlabel("zoom-in", fontsize=8)
    axz.grid(True, axis="x", color=GRID_GRAY, linewidth=0.6)
    axz.set_axisbelow(True)
    for spine in axz.spines.values():
        spine.set_linewidth(0.9)

    # 主图上的放大框 + 连线
    ax.add_patch(Rectangle((w0, -0.66), w1 - w0, 2.38, fill=False,
                           edgecolor=NOTE_GRAY, linewidth=0.9, linestyle=(0, (4, 3))))
    for y_corner in (1.72, -0.66):
        con = ConnectionPatch(xyA=(w0, y_corner), coordsA=ax.transData,
                              xyB=(w0, y_corner), coordsB=axz.transData,
                              color=NOTE_GRAY, linewidth=0.7, linestyle=":")
        fig.add_artist(con)

# ---- 底部：全部 4 episodes 的 per-episode IoU vs 随机基线 ----
axi = fig.add_subplot(gs[2, :])
ids = sorted(episodes)
iou_pred = [episodes[i]["mean_matched_iou"] for i in ids]
iou_base = [episodes[i]["random_baseline_iou"] for i in ids]
xs = range(len(ids))
bw = 0.36
bars_p = axi.bar([x - bw / 2 for x in xs], iou_pred, width=bw,
                 facecolor=CYAN_FILL, edgecolor=CYAN_DARK, linewidth=1.2,
                 label="SMQ (E-C, end-to-end K=8)")
bars_b = axi.bar([x + bw / 2 for x in xs], iou_base, width=bw,
                 facecolor=BAND_GRAY, edgecolor=BASE_EDGE, linewidth=1.2, hatch="///",
                 label="Random segmentation baseline")
for rect, v in zip(bars_p, iou_pred):
    axi.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=7.5, color=INK)
for rect, v in zip(bars_b, iou_base):
    axi.text(rect.get_x() + rect.get_width() / 2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=7.5, color=NOTE_GRAY)
agg = payload["aggregate"]
n_win = sum(p > b for p, b in zip(iou_pred, iou_base))
axi.text(0.985, 0.96,
         f"aggregate {agg['mean_matched_iou']:.4f} \u00b1 {agg['std']:.4f}"
         f" \u00b7 {n_win}/{len(ids)} episodes > random baseline",
         transform=axi.transAxes, ha="right", va="top", fontsize=8.5, color=INK)
axi.set_xticks(list(xs))
axi.set_xticklabels([f"Episode {i}" for i in ids], fontsize=9)
axi.set_ylabel("Matched IoU\n(seed pseudo-GT protocol)", fontsize=9)
axi.set_ylim(0, 0.72)
axi.grid(True, axis="y", color=GRID_GRAY, linewidth=0.8)
axi.set_axisbelow(True)
for spine in axi.spines.values():
    spine.set_linewidth(0.9)
axi.legend(loc="upper left", frameon=False, fontsize=8.5)

# ---- 口径注记 + 溯源小注：放画布下方负坐标区（bbox tight 自动收入），左右分行防重叠 ----
fig.text(0.01, -0.042,
         "Public-real layer (InterPet4D smal_npy) \u00b7 GT = W6 rule-engine seed pseudo-labels "
         "(\u03ba\u22650.8, \u22650.5 s; weak reference, not human annotation) \u00b7 "
         "band text = seed behavior class \u00b7 SMQ epoch-30 checkpoint",
         fontsize=6, color=NOTE_GRAY, ha="left", va="top")
fig.text(0.99, -0.016,
         "FIGURE_SOURCE: docs/paper/figures/scripts/make_fig3_segmentation_qualitative.py "
         "\u00b7 PSD-Framework \u00b7 W22 \u00b7 2026-08-25",
         fontsize=6, color=NOTE_GRAY, ha="right", va="top")

pdf_path = OUT_DIR / "fig3_segmentation_qualitative.pdf"
png_path = OUT_DIR / "fig3_segmentation_qualitative.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.10)
plt.close(fig)

# ---- 当次运行验证输出 ----
print("written:", pdf_path)
print("written:", png_path)
print("--- plotted numbers (from recheck JSON) ---")
for i in ids:
    ep = episodes[i]
    print(f"ep{i}: T={ep['T']} n_gt={ep['n_gt_segments']} "
          f"IoU={ep['mean_matched_iou']} baseline={ep['random_baseline_iou']}")
print(f"aggregate: {agg['mean_matched_iou']} ± {agg['std']} | episodes>baseline: {n_win}/4")
