# -*- coding: utf-8 -*-
"""make_next_paper_figures — 下一篇论文 4 张主图（数据全部来自实测 summary）。

产出 docs/next_paper/figures/fig{1,2,3,4}_*.pdf+.png。
数据源:
  fig1 四层冻结梯度      : runs/p1_sprint/labelfree_*/summary.json（published E9）
  fig2 三 tier 双用主图  : labelfree_*/summary.json + adapt_t4_*/summary_{same10,aug4,aug8_lr2e2}.json
  fig3 预算-分配率曲线   : runs/p1_sprint/adapt_t4_ntu120/summary_{ref5,split5,same10,ref15,dual15}.json
  fig4 增广剂量响应      : runs/p1_sprint/adapt_t4_ntu120/summary_{same10,aug4_lr2e2,aug8_lr2e2,aug12_lr2e2}.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "p1_sprint"
OUT = REPO / "docs" / "next_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "figure.dpi": 150,
})
C_FROZEN = "#8a8a8a"
C_DUAL = "#1f6fb4"
C_BAND = "#d62728"
# 白色软衬底：标签与判定线/曲线交叠时保证数字可读（不改变任何数据）
HALO = dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.85)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[fig] {name}.pdf/.png")


# ---------- fig1: 四层冻结梯度 ----------
def fig1():
    tiers = ["NTU60", "PanAf500", "NTU120", "UCF101"]
    vals = [90.7, 82.5, 74.96, 51.09]
    refs = [74.45, 64.0, 58.37, 19.87]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.axhspan(85, 90, color=C_BAND, alpha=0.06)
    ax.axhspan(90, 100, color=C_BAND, alpha=0.12)
    ax.axhline(90, color=C_BAND, lw=0.8, ls=":")
    ax.axhline(85, color=C_BAND, lw=0.8, ls="--")
    x = np.arange(4)
    ax.bar(x - 0.18, vals, 0.36, label="Budget retention (10%)", color=C_DUAL)
    ax.bar(x + 0.18, refs, 0.36, label="Full-label linear reference", color=C_FROZEN)
    for xi, v in zip(x, vals):
        ax.text(xi - 0.18, v + 1.2, f"{v:.2f}",
                ha="center", fontsize=8.5, bbox=HALO)
    for xi, v in zip(x, refs):
        ax.text(xi + 0.18, v + 1.2, f"{v:.2f}",
                ha="center", fontsize=8.5, color="#555555", bbox=HALO)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(0, 104)
    ax.set_ylabel("Accuracy / retention (%)")
    ax.set_title("Four-tier budget-retention gradient (frozen physical layer)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              fontsize=8, frameon=False)
    ax.text(3.42, 86.4, "PARTIAL", fontsize=7.5, color=C_BAND, ha="right")
    ax.text(3.42, 92.8, "CONFIRMS", fontsize=7.5, color=C_BAND, ha="right")
    save(fig, "fig1_four_tier_gradient")


# ---------- 预算-分配率曲线（输出 fig3） ----------
def fig2():
    budgets = [5, 10, 15]
    frozen = [64.61, 74.96, 79.04]
    dual = [67.51, 78.87, 83.60]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(budgets, frozen, "o-", color=C_FROZEN, label="Semantic-only (frozen)")
    ax.plot(budgets, dual, "s-", color=C_DUAL, label="Dual-use (recycled)")
    for b, f, d in zip(budgets, frozen, dual):
        ax.annotate(f"{f:.2f}", (b, f), textcoords="offset points", xytext=(0, -18),
                    ha="center", fontsize=8, color="#555555")
        ax.annotate(f"{d:.2f}", (b, d), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8)
        # 增益标注置于两曲线之间的空隙，并按文字宽度处的曲线高度夹取，避免压线
        lo = max(np.interp(b - 0.5, budgets, frozen), np.interp(b + 0.5, budgets, frozen))
        hi = min(np.interp(b - 0.5, budgets, dual), np.interp(b + 0.5, budgets, dual))
        y_gap = float(np.clip((f + d) / 2, lo + 0.7, hi - 1.2))
        ax.annotate(f"+{d-f:.2f}", (b, y_gap), ha="center", fontsize=7.5,
                    color="#b22222")
    ax.set_xticks(budgets)
    ax.set_xlabel("Anchor budget (% of training clips)")
    ax.set_ylabel("NTU120 retention (%)")
    ax.set_title("Budget-allocation curve: dual-use widens with budget")
    ax.set_ylim(60, 90)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    save(fig, "fig3_budget_allocation")


# ---------- 增广剂量响应（输出 fig4） ----------
def fig3():
    doses = [1, 4, 8, 12]
    ret = [82.73, 83.41, 85.16, 82.47]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(doses, ret, "s-", color=C_DUAL)
    for d, r in zip(doses, ret):
        ax.annotate(f"{r:.2f}", (d, r), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8)
    ax.axhline(85, color=C_BAND, lw=0.8, ls="--")
    ax.axhspan(85, 87, color=C_BAND, alpha=0.06)
    ax.annotate("PARTIAL band", (11.6, 86.0), fontsize=7.5, color=C_BAND, ha="right")
    ax.set_xticks(doses)
    ax.set_xlabel("Anchor-augmentation dose V (views per anchor)")
    ax.set_ylabel("NTU120 retention (%)")
    ax.set_title("Augmentation dose-response peaks at V=8 (lr 2e-2)")
    ax.set_ylim(81, 87.5)
    save(fig, "fig4_aug_dose_response")


# ---------- 三 tier 双用主图（输出 fig2） ----------
def fig4():
    tiers = ["NTU120", "PanAf500", "UCF101"]
    frozen = [74.96, 82.5, 51.09]
    dual = [85.16, 92.21, 71.21]
    verdicts = ["PARTIAL", "CONFIRMS", "FAILS"]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    x = np.arange(3)
    ax.axhline(85, color=C_BAND, lw=0.8, ls="--")
    ax.axhline(90, color=C_BAND, lw=0.8, ls=":")
    ax.bar(x - 0.18, frozen, 0.36, label="Frozen physical layer", color=C_FROZEN)
    ax.bar(x + 0.18, dual, 0.36, label="Anchor recycling (best)", color=C_DUAL)
    for xi, f, d, v in zip(x, frozen, dual, verdicts):
        ax.text(xi - 0.18, f + 1.0, f"{f:.2f}", ha="center", fontsize=8.5, bbox=HALO)
        ax.text(xi + 0.18, d + 1.0, f"{d:.2f}", ha="center", fontsize=8.5, bbox=HALO)
        ax.text(xi, max(f, d) + 4.2, f"{v}", ha="center", fontsize=8, color=C_BAND,
                bbox=HALO)
        ax.annotate(f"+{d-f:.2f}", (xi + 0.18, d - 3.0), ha="center", va="top",
                    fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylim(40, 100)
    ax.set_ylabel("Budget retention (%)")
    ax.set_title("Anchor recycling across weak-reference tiers (10% budget)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2, fontsize=8,
              frameon=False)
    save(fig, "fig2_tier_main_results")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"[done] figures -> {OUT}")
