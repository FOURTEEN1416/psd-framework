# -*- coding: utf-8 -*-
"""fig3 v10 — SMQ 分割边界定性图（正典 · 风格方向 **C+ 强化版**）。

承 v9 全部构图（双 episode 时间轴 + 四 episode IoU 聚合面板、面板字母 a/b/c、
灰底轨、直标值、网格），**图内全部文字与数据口径逐字承 v9，零新增零删除**。
数据源仍绑定 reports/p02-smq-iou-eC-seeds-recheck.json（aggregate
0.4577±0.0488 与 caption 逐位一致）。

C+ 相对 v9 的收紧项：
1. **笔触更细**：spines 0.65 -> 0.4；tick width 0.65 -> 0.4、length 2.4 -> 1.8；
   网格 0.40/0.45 -> 0.35；段条描边 0.5 -> 0.4；柱描边 0.9 -> 0.5。
2. **字号阶梯更大**：episode 标题 6.5 -> 7.2；面板字母 a/b/c 8 -> 9；
   段内标签 5.5 -> 5.8；轴名 6.5 -> 7.0；刻度 5.5 -> 5.8；柱值直标 5.5 -> 5.8；
   聚合注记 6.0 -> 6.2；图例 5.8 -> 6.0。下限仍 5.5pt。
   （注：v9 的 `set_yticklabels(fontsize=6)` 被随后的 `tick_params(labelsize=5.5)`
   覆盖，实际渲染为 5.5——本版改为先 tick_params 后 set_yticklabels，使字号生效。）
   （注：agg 注记与右上图例在 6.0pt 时已是共存临界，二者字号保持 v9 值，
   C+ 的字号增益用在有净空处——标题 7.2 / 面板字母 9 / 刻度 6.5 / 轴名 7.0。）
3. **留白 +9%**：画布 3.03 -> 3.30in，hspace 0.62 -> 0.75，面板间呼吸更大。
   **轴宽保持不变**（left/right 不动的 x 方向口径），故段内标签门控口径
   （ax_w_in=3.05）与标签集合与 v8/v9 完全一致。
4. **标注带净空**：柱值直标 0.012/0.018 -> 0.016/0.024；聚合注记留 0.008 边距。

caption 约束逐词核对（04-experiments.tex \\label{fig:segqual}）：
upper band = seed pseudo-GT(orange) / lower band = SMQ pred(teal) /
lower panel = 四 episode IoU 聚合 vs random-cut null / agg 0.4577±0.0488 /
4/4 episodes above baseline——全部保持。
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "reports" / "p02-smq-iou-eC-seeds-recheck.json"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)
assert DATA.is_file(), f"data missing: {DATA}"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
CYAN_FILL, CYAN_DARK = psd_style.PHYS_FILL, psd_style.PHYS_EDGE
ORANGE_FILL, ORANGE_DARK = psd_style.SEM_FILL, psd_style.SEM_EDGE
BAND_GRAY, BASE_EDGE = psd_style.GRAY_FILL, psd_style.GRAY_EDGE
RAIL_GRAY = "#ECECEC"
NOTE_GRAY, GRID_GRAY = "#666666", psd_style.GRID_GRAY

with open(DATA, encoding="utf-8") as f:
    payload = json.load(f)
episodes = {ep["id"]: ep for ep in payload["episodes"]}
EP_MAIN = [1, 4]
BAR_H = 0.66
Y_GT, Y_PRED = 1.0, 0.0

SEG_LABELS = []          # 收集段内标签（本地净空自检用：记录 (text_obj, s, e)）


def draw_track(ax, segments, y_center, facecolor, edgecolor, labels=None, t_total=1.0):
    """标签宽度估算口径承 v8/v9（ax_w_in=3.05 不变 → 标签集合完全一致）。"""
    ax_w_in = 3.05
    frames_per_in = t_total / ax_w_in
    ax.broken_barh([(0, t_total)], (y_center - BAR_H / 2, BAR_H),
                   facecolors=RAIL_GRAY, edgecolors="none", linewidth=0, zorder=1)
    for i, (s, e) in enumerate(segments):
        ax.broken_barh([(s, e - s)], (y_center - BAR_H / 2, BAR_H),
                       facecolors=facecolor, edgecolors=edgecolor, linewidth=0.4,
                       zorder=2)
        if labels and i < len(labels) and labels[i]:
            need = len(labels[i]) * 0.042 * frames_per_in + 30
            if (e - s) >= max(40, need):
                t = ax.text((s + e) / 2, y_center, labels[i], ha="center",
                            va="center", fontsize=5.8, color=INK, clip_on=True,
                            zorder=3)
                SEG_LABELS.append((t, s, e, ax))


def style_timeline(ax, ep, tag, show_xlabel):
    t = ep["T"]
    ax.set_xlim(0, t)
    ax.set_ylim(-0.75, 1.82)
    ax.set_yticks([Y_PRED, Y_GT])
    ax.set_xticks([0, t // 2, t])
    # C+：先 tick_params（统一 width/length/labelsize），再 set_*ticklabels 覆盖字号
    ax.tick_params(labelsize=5.8, width=0.4, length=1.8)
    ax.set_yticklabels(["SMQ pred.", "Pseudo-GT"], fontsize=6.5)
    for lab, colr in zip(ax.get_yticklabels(), (CYAN_DARK, ORANGE_DARK)):
        lab.set_color(colr); lab.set_fontweight("bold")
    if show_xlabel:
        ax.set_xlabel("frame index", fontsize=5.8)
    ax.set_title(f"Episode {tag} \u00b7 T={t} \u00b7 IoU {ep['mean_matched_iou']:.3f}",
                 loc="left", fontsize=7.2, pad=3.5)
    ax.grid(True, axis="x", color=GRID_GRAY, linewidth=0.35)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.4)


# ---- 画布：宽 3.42in == 版心 246.24pt；高 3.03 -> 3.30in（C+ 留白） ----
FIG_W, FIG_H = 3.42, 3.30
fig = plt.figure(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.12], hspace=0.75,
                      left=0.185, right=0.972, top=0.935, bottom=0.055)

PANEL_LETTERS = ["a", "b", "c"]
for row, eid in enumerate(EP_MAIN):
    ep = episodes[eid]
    ax = fig.add_subplot(gs[row, 0])
    draw_track(ax, ep["gt_segments"], Y_GT, ORANGE_FILL, ORANGE_DARK,
               labels=ep.get("gt_labels"), t_total=ep["T"])
    draw_track(ax, ep["pred_segments"], Y_PRED, CYAN_FILL, CYAN_DARK, t_total=ep["T"])
    style_timeline(ax, ep, eid, show_xlabel=(row == 1))
    ax.text(-0.16, 1.12, PANEL_LETTERS[row], transform=ax.transAxes,
            fontsize=9, fontweight="bold", ha="left", va="bottom", color=INK)

axi = fig.add_subplot(gs[2, 0])
ids = sorted(episodes)
iou_pred = [episodes[i]["mean_matched_iou"] for i in ids]
iou_base = [episodes[i]["random_baseline_iou"] for i in ids]
xs = range(len(ids))
bw = 0.36
bars_p = axi.bar([x - bw / 2 for x in xs], iou_pred, width=bw,
                 facecolor=CYAN_FILL, edgecolor=CYAN_DARK, linewidth=0.5,
                 label="SMQ (E-C, K=8)")
bars_b = axi.bar([x + bw / 2 for x in xs], iou_base, width=bw,
                 facecolor=BAND_GRAY, edgecolor=BASE_EDGE, linewidth=0.5,
                 hatch="///", label="Random baseline")
for rect, v in zip(bars_p, iou_pred):
    axi.text(rect.get_x() + rect.get_width() / 2, v + 0.016, f"{v:.3f}",
             ha="center", va="bottom", fontsize=5.8, color=INK)
for rect, v in zip(bars_b, iou_base):
    axi.text(rect.get_x() + rect.get_width() / 2 + 0.05, v + 0.024, f"{v:.3f}",
             ha="center", va="bottom", fontsize=5.8, color=NOTE_GRAY,
             bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
agg = payload["aggregate"]
n_win = sum(p > b for p, b in zip(iou_pred, iou_base))
axi.text(0.005, 0.99,
         f"agg {agg['mean_matched_iou']:.4f}\u00b1{agg['std']:.4f} \u00b7 {n_win}/4 > baseline",
         transform=axi.transAxes, ha="left", va="top", fontsize=6.0, color=INK)
axi.set_xticks(list(xs))
axi.set_xticklabels([f"Ep {i}" for i in ids], fontsize=6.5)
axi.set_ylabel("Matched IoU", fontsize=7.0)
axi.tick_params(labelsize=5.8, width=0.4, length=1.8)
axi.set_ylim(0, 0.95)   # C+ 顶部余量：给 agg 注记与图例留净空
axi.grid(True, axis="y", color=GRID_GRAY, linewidth=0.35)
axi.set_axisbelow(True)
for side in ("top", "right"):
    axi.spines[side].set_visible(False)
for side in ("left", "bottom"):
    axi.spines[side].set_linewidth(0.4)
# C+ 净空：agg 注记横贯面板顶（x 0.005..0.61），图例下移一档避让
axi.legend(loc="upper right", bbox_to_anchor=(1.0, 0.90), ncol=2,
           frameon=False, fontsize=5.8)
axi.text(-0.16, 1.12, PANEL_LETTERS[2], transform=axi.transAxes,
         fontsize=9, fontweight="bold", ha="left", va="bottom", color=INK)

# ---- 出图（尺度修复：无 tight，MediaBox 严格等于画布） ----
pdf_path = OUT_DIR / "fig3_segmentation_qualitative.pdf"
png_path = OUT_DIR / "fig3_segmentation_qualitative.png"
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(PREVIEW / "preview_fig3.png", dpi=150)
print("fig3 v10 (direction C+ intensified) saved")
for i in ids:
    ep = episodes[i]
    print(f"ep{i}: T={ep['T']} IoU={ep['mean_matched_iou']} baseline={ep['random_baseline_iou']}")
print(f"aggregate: {agg['mean_matched_iou']} \u00b1 {agg['std']} | {n_win}/4")

# ---- 门禁（G1 + G5 + G6 + G4；时间轴无误差棒故无 G3） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig3-v10", {"pseudo-GT band": ORANGE_DARK, "SMQ pred band": CYAN_DARK},
    redundancy="upper/lower band position + colored axis labels + bar hatch")
gates.gate_typography(fig, min_pt=5.5, name="fig3-v10")
gates.gate_marks_in_axes(fig, name="fig3-v10")
gates.gate_pdf(pdf_path, max_w_pt=246.3)

# ---- 本地净空自检（不进共享模块）：段内标签必须完全落在其所属段条范围内 ----
# clip_on=True 会把溢出部分裁掉（视觉缺陷且不会被 G2/G5 发现），故显式断言。
fig.canvas.draw()
ren = fig.canvas.get_renderer()
bad = []
for t, s, e, axref in SEG_LABELS:
    b = t.get_window_extent(ren)
    (x0, _), (x1, _) = axref.transData.inverted().transform([[b.x0, b.y0], [b.x1, b.y1]])
    x0, x1 = min(x0, x1), max(x0, x1)
    ok = x0 >= s - 0.5 and x1 <= e + 0.5
    if not ok:
        bad.append((t.get_text(), round(float(x0), 1), round(float(x1), 1), s, e))
print(f"[local clearance] fig3-v10: {len(SEG_LABELS)} in-segment labels, "
      "all inside own segment -> " + ("PASS" if not bad else f"FAIL {bad}"))
assert not bad, f"in-segment label overflow: {bad}"