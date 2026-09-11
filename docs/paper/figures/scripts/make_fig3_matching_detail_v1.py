# -*- coding: utf-8 -*-
r"""fig3b v1 — SMQ 分割 Hungarian 匹配对细节图（任务包 A 第三轮 · 用户裁决批准）。

方案（BOARD 09-11 DECISION 已登记）：
- 数据源 reports/p02-smq-iou-eC-seeds-recheck.json（与 fig3 同源，只读）；
- 逐 episode 计算 gt×pred IoU 矩阵，scipy Hungarian（linear_sum_assignment，
  代价=1-IoU）求最优匹配对；仅可视化衍生物，零新统计结论；
- 协议对齐门：复算 mean matched IoU（Σ匹配IoU/n_gt）与 JSON 既有
  mean_matched_iou 逐位核对——|Δ|>1e-3 如实双列打印，>0.05 硬失败（不静默选边）；
- 形态：与 fig3 一致的双泳道（上 pseudo-GT 橙带 / 下 SMQ pred 青带），
  匹配对以连接线相连（透明度 ∝ IoU，IoU=0 的强制配位不画线），
  未匹配段如实悬空可见；面板题用 JSON 既有数值。

输出为独立新资产 fig3_matching_detail.pdf/png，不进 main.tex（是否引入另行裁决）。
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "reports" / "p02-smq-iou-eC-seeds-recheck.json"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
assert DATA.is_file(), f"data missing: {DATA}"

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
CYAN_FILL, CYAN_DARK = psd_style.PHYS_FILL, psd_style.PHYS_EDGE
ORANGE_FILL, ORANGE_DARK = psd_style.SEM_FILL, psd_style.SEM_EDGE
RAIL_GRAY = "#ECECEC"
NOTE_GRAY, GRID_GRAY = "#666666", psd_style.GRID_GRAY

payload = json.loads(DATA.read_text(encoding="utf-8"))
episodes = {ep["id"]: ep for ep in payload["episodes"]}
EP_MAIN = [1, 4]  # 与 fig3 主面板一致

BAR_H = 0.66
Y_PRED, Y_GT = 0.0, 1.0


def seg_iou(a, b):
    lo = max(a[0], b[0]); hi = min(a[1], b[1])
    inter = max(0, hi - lo)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def hungarian_match(gt, pred):
    """Hungarian 最优匹配（代价 1-IoU）。返回 (匹配对列表[(gi,pi,iou)], 复算均值)。"""
    M = np.array([[seg_iou(g, p) for p in pred] for g in gt])
    rows, cols = linear_sum_assignment(1.0 - M)
    pairs = [(int(r), int(c), float(M[r, c])) for r, c in zip(rows, cols)]
    mean_iou = float(M[rows, cols].sum() / len(gt)) if len(gt) else 0.0
    return pairs, mean_iou


def draw_episode(ax, ep, tag, show_xlabel):
    gt, pred = ep["gt_segments"], ep["pred_segments"]
    t_total = ep["T"]
    pairs, mean_recalc = hungarian_match(gt, pred)
    stored = ep["mean_matched_iou"]
    diff = abs(mean_recalc - stored)
    status = "OK" if diff <= 1e-3 else ("WARN" if diff <= 0.05 else "FAIL")
    print(f"[protocol-gate] ep{tag}: stored={stored:.6f} recalc={mean_recalc:.6f} "
          f"|d|={diff:.6f} -> {status}")
    if status == "FAIL":
        raise SystemExit(f"ep{tag}: Hungarian recalc diverges from stored IoU")

    # 底轨 + 分段（与 fig3 同风格）
    for segs, y, fc, ec in ((gt, Y_GT, ORANGE_FILL, ORANGE_DARK),
                            (pred, Y_PRED, CYAN_FILL, CYAN_DARK)):
        ax.broken_barh([(0, t_total)], (y - BAR_H / 2, BAR_H),
                       facecolors=RAIL_GRAY, edgecolors="none", linewidth=0, zorder=1)
        for s, e in segs:
            ax.broken_barh([(s, e - s)], (y - BAR_H / 2, BAR_H),
                           facecolors=fc, edgecolors=ec, linewidth=0.5, zorder=2)
    # GT 标签（宽段才标, 与 fig3 同规则简化版）
    labels = ep.get("gt_labels") or []
    for i, (s, e) in enumerate(gt):
        if i < len(labels) and labels[i] and (e - s) >= 180:
            ax.text((s + e) / 2, Y_GT, labels[i], ha="center", va="center",
                    fontsize=5.2, color=INK, clip_on=True, zorder=3)
    # 匹配连接线（透明度 ∝ IoU; IoU=0 强制配位不画）
    n_linked = 0
    for gi, pi, iou in pairs:
        if iou <= 0:
            continue
        n_linked += 1
        gx = (gt[gi][0] + gt[gi][1]) / 2
        px = (pred[pi][0] + pred[pi][1]) / 2
        ax.plot([gx, px], [Y_GT - BAR_H / 2, Y_PRED + BAR_H / 2],
                color="#3A3A3A", lw=0.5, alpha=0.10 + 0.75 * iou, zorder=1)
    ax.set_xlim(0, t_total)
    ax.set_ylim(-0.75, 1.82)
    ax.set_yticks([Y_PRED, Y_GT])
    ax.set_yticklabels(["SMQ pred.", "Pseudo-GT"], fontsize=6)
    ax.set_xticks([0, t_total // 2, t_total])
    ax.tick_params(labelsize=5.5, width=0.65, length=2.4)
    if show_xlabel:
        ax.set_xlabel("frame index", fontsize=5.5)
    for lab, colr in zip(ax.get_yticklabels(), (CYAN_DARK, ORANGE_DARK)):
        lab.set_color(colr); lab.set_fontweight("bold")
    ax.set_title(
        f"Episode {tag} \u00b7 T={t_total} \u00b7 IoU {stored:.3f} \u00b7 "
        f"{len(gt)} GT \u00d7 {len(pred)} pred \u00b7 {n_linked} linked",
        loc="left", fontsize=6.5, pad=3)
    ax.grid(True, axis="x", color=GRID_GRAY, linewidth=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.65)
    return n_linked


fig = plt.figure(figsize=(3.42, 2.20))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.62)
PANEL_LETTERS = ["a", "b"]
for row, eid in enumerate(EP_MAIN):
    ax = fig.add_subplot(gs[row, 0])
    draw_episode(ax, episodes[eid], eid, show_xlabel=(row == 1))
    ax.text(-0.13, 1.10, PANEL_LETTERS[row], transform=ax.transAxes,
            fontsize=8, fontweight="bold", ha="left", va="bottom", color=INK)

fig.text(0.995, 0.995, "link opacity $\\propto$ matched IoU; IoU=0 forced pairs unlinked",
         ha="right", va="top", fontsize=5.0, color=NOTE_GRAY)

pdf_path = OUT_DIR / "fig3_matching_detail.pdf"
png_path = OUT_DIR / "fig3_matching_detail.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.01)
plt.close(fig)

# ---- 出图门禁 ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig3b", {"pseudo-GT band": ORANGE_DARK, "SMQ pred band": CYAN_DARK},
    redundancy="upper/lower band position + colored axis labels")
gates.gate_pdf(pdf_path)
print("fig3b v1 (Hungarian matching detail) saved:", pdf_path)
