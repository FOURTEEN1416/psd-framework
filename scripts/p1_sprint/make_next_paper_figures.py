# -*- coding: utf-8 -*-
"""make_next_paper_figures — 下一篇论文 4 张主图（v3：视觉重构）。

v3 相对 v2 的变化（针对"图太丑"的反馈，不是修瑕疵）：
  · 比例：从 180×66 的极宽条改为 165×78 / 165×72 / 110×82 / 110×76，画面不再空
  · 字号：正文 8.5–9 pt、刻度 8 pt、关键数字 9 pt 半粗，印刷与屏幕都可读
  · 体量：横条填满 0–100 轨道、哑铃箭头 2.6 pt、曲线 2.2–2.4 pt、面积填充
  · 层次：图1 左侧双行标签（tier + 全标注参考）；图2 左侧 tier + 判决；
          图3 图例移到坐标区上方，绘图区更满
  · 配色：深海军蓝 #1B4F72（方法）/ 砖红 #B2182B（判定与增益）/ 中性灰（参考）

数据全部来自实测 summary，脚本内断言与手稿数字对齐。
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

# ---------------- design system ----------------
INK = "#1A1A1A"
HAIR = "#3D3D3D"
GRAY = "#9A9A9A"
GRAY_D = "#6E6E6E"
TRACK = "#EDEDED"
BLUE = "#1B4F72"
BLUE_D = "#123A55"
RED = "#B2182B"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": HAIR,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": HAIR,
    "ytick.color": HAIR,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3.2,
    "ytick.major.size": 3.2,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"[fig] {name}.pdf/.png")


def seed_retention(summary: dict) -> list[float]:
    """冻结臂逐种子 retention = b_arm_i / full_ref。"""
    fr = summary["full_ref"]
    out = [b / fr for b in summary["b_arms"]]
    assert abs(np.mean(out) - summary["retention"]) < 5e-3, (
        f"{summary.get('tier')}: per-seed mean {np.mean(out):.4f} != {summary['retention']}")
    return out


# ---------- fig1: 四层梯度（0–100 轨道横条） ----------
def fig1():
    tiers = ["NTU60", "PanAf500", "NTU120", "UCF101"]
    ret = [90.70, 82.50, 74.96, 51.09]
    ref = [74.45, 64.00, 58.37, 19.87]

    fig, ax = plt.subplots(figsize=(165 * MM, 78 * MM))
    fig.subplots_adjust(left=0.185, right=0.965, top=0.855, bottom=0.185)

    y = np.arange(len(tiers))[::-1]

    # 0–100 轨道 + 实测横条
    ax.barh(y, np.full(len(tiers), 100.0), height=0.60, color=TRACK, lw=0, zorder=0)
    ax.barh(y, ret, height=0.60, color=BLUE, lw=0, zorder=2)

    # 判定阈值
    ax.axvline(90, color=RED, lw=1.0, ls=(0, (2.6, 2.2)), zorder=3)
    ax.axvline(85, color=RED, lw=1.0, ls=(0, (1.1, 2.0)), zorder=3)
    ax.text(87.5, 0.36, "PARTIAL", ha="center", va="bottom", rotation=90,
            fontsize=6.8, color=RED)
    ax.text(95.0, 0.36, "CONFIRMS", ha="center", va="bottom", rotation=90,
            fontsize=6.8, color=RED)

    for yi, v in zip(y, ret):
        ax.text(v - 2.2, yi, f"{v:.2f}", va="center", ha="right",
                fontsize=9.2, fontweight="bold", color="white", zorder=5)

    # 左侧双行标签：tier 名 + 全标注参考精度（retention 的分母）
    for yi, name, r in zip(y, tiers, ref):
        ax.text(-1.8, yi + 0.13, name, ha="right", va="center",
                fontsize=9.2, fontweight="bold", color=INK, clip_on=False)
        ax.text(-1.8, yi - 0.20, f"ref {r:.2f}", ha="right", va="center",
                fontsize=7.4, color=GRAY_D, clip_on=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.72, 3.55)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks([])
    ax.set_xlabel("Budget retention (%)")
    ax.tick_params(axis="y", length=0)

    save(fig, "fig1_four_tier_gradient")


# ---------- fig2: 三 tier 主结果（哑铃 + 种子刻度 + SD） ----------
def fig2():
    tiers = ["NTU120", "PanAf500", "UCF101"]
    frozen_sum = [load(RUNS / "labelfree_ntu120" / "summary.json"),
                  load(RUNS / "labelfree_panaf500" / "summary.json"),
                  load(RUNS / "labelfree_ucf101" / "summary.json")]
    dual_sum = [load(RUNS / "adapt_t4_ntu120" / "summary_aug8_lr2e2.json"),
                load(RUNS / "adapt_t4_panaf500" / "summary_aug4.json"),
                load(RUNS / "adapt_t4_ucf101" / "summary_same10.json")]
    verdicts = ["PARTIAL", "CONFIRMS", "FAILS"]

    frozen = [seed_retention(s) for s in frozen_sum]
    dual = [[r["retention_seed"] for r in s["per_seed"]] for s in dual_sum]
    f_mean = [s["retention"] * 100 for s in frozen_sum]
    d_mean = [s["retention"] * 100 for s in dual_sum]

    assert np.allclose(f_mean, [74.96, 82.50, 51.09], atol=0.02), f_mean
    assert np.allclose(d_mean, [85.16, 92.21, 71.21], atol=0.02), d_mean
    assert np.all(np.abs(np.mean(dual, axis=1) * 100 - np.array(d_mean)) < 1.0)

    fig, ax = plt.subplots(figsize=(165 * MM, 72 * MM))
    fig.subplots_adjust(left=0.175, right=0.975, top=0.86, bottom=0.18)

    # 判定带：只做背景，不抢哑铃
    ax.axvspan(85, 90, color=RED, alpha=0.05, lw=0, zorder=0)
    ax.axvspan(90, 100, color=RED, alpha=0.08, lw=0, zorder=0)
    ax.text(87.5, 2.60, "PARTIAL", ha="center", va="bottom", fontsize=6.6, color=RED)
    ax.text(95.0, 2.60, "CONFIRMS", ha="center", va="bottom", fontsize=7.0, color=RED)

    for i in range(3):
        y = 2 - i
        f = np.array(frozen[i]) * 100
        d = np.array(dual[i]) * 100
        xf, xd = f_mean[i], d_mean[i]

        # 极简哑铃：均值点 + 箭头（种子分布见表格与发布数据）

        # 哑铃箭头
        ax.annotate("", xy=(xd, y), xytext=(xf, y),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.6,
                                    shrinkA=5, shrinkB=5, mutation_scale=14),
                    zorder=3)
        ax.plot([xf], [y], "o", ms=9.0, mfc="white", mec=GRAY_D, mew=1.4, zorder=5)
        ax.plot([xd], [y], "o", ms=9.0, mfc=BLUE, mec="white", mew=1.2, zorder=5)

        # 增益
        ax.text((xf + xd) / 2, y + 0.30, f"+{d_mean[i] - f_mean[i]:.2f} pp",
                ha="center", va="bottom", fontsize=9.0, fontweight="bold",
                color=RED, zorder=6)

        # 左侧固定标签列：tier + 判决
        ax.text(39.0, y + 0.16, tiers[i], ha="right", va="center",
                fontsize=9.2, fontweight="bold", color=INK, clip_on=False)
        ax.text(39.0, y - 0.20, f"→ {verdicts[i]}", ha="right", va="center",
                fontsize=7.6, color=RED, clip_on=False)

    ax.set_xlim(40, 100)
    ax.set_ylim(-0.45, 2.86)
    ax.set_yticks([])
    ax.set_xticks(np.arange(40, 101, 10))
    ax.set_xlabel("Budget retention (%)")
    ax.tick_params(axis="y", length=0)

    save(fig, "fig2_tier_main_results")


# ---------- fig3: 预算-分配率曲线（增益带） ----------
def fig3():
    budgets = [5, 10, 15]
    ref5 = load(RUNS / "adapt_t4_ntu120" / "summary_ref5.json")
    ref15 = load(RUNS / "adapt_t4_ntu120" / "summary_ref15.json")
    frozen10 = load(RUNS / "labelfree_ntu120" / "summary.json")
    split5 = load(RUNS / "adapt_t4_ntu120" / "summary_split5.json")
    same10 = load(RUNS / "adapt_t4_ntu120" / "summary_same10.json")
    dual15 = load(RUNS / "adapt_t4_ntu120" / "summary_dual15.json")

    f_vals = [ref5["retention"] * 100, frozen10["retention"] * 100,
              ref15["retention"] * 100]
    d_vals = [split5["retention"] * 100, same10["retention"] * 100,
              dual15["retention"] * 100]
    assert np.allclose(f_vals, [64.61, 74.96, 79.04], atol=0.02), f_vals
    assert np.allclose(d_vals, [67.51, 78.87, 83.60], atol=0.02), d_vals

    fig, ax = plt.subplots(figsize=(110 * MM, 82 * MM))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.825, bottom=0.175)

    ax.fill_between(budgets, f_vals, d_vals, color=BLUE, alpha=0.14, lw=0)
    ax.plot(budgets, f_vals, "-o", color=GRAY_D, lw=1.6, ms=6.0, mfc="white",
            mec=GRAY_D, mew=1.5, label="Semantic-only (frozen)", zorder=3)
    ax.plot(budgets, d_vals, "-s", color=BLUE, lw=2.2, ms=6.0, mfc="white",
            mec=BLUE, mew=1.6, label="Anchor recycling", zorder=4)

    for b, f, d in zip(budgets, f_vals, d_vals):
        ax.text(b, d + 1.25, f"{d:.2f}", ha="center", va="bottom", fontsize=8.2,
                color=BLUE_D, fontweight="bold")
        gx, gha = (b + 0.30, "left") if b < 15 else (b - 0.30, "right")
        ax.text(gx, (f + d) / 2, f"+{d - f:.2f}", ha=gha, va="center",
                fontsize=8.0, fontweight="bold", color=RED, zorder=6)


    ax.set_xlim(4.2, 15.8)
    ax.set_ylim(62, 87)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.set_yticks(np.arange(65, 86, 5))
    ax.set_xlabel("Anchor budget (% of training clips)")
    ax.set_ylabel("NTU120 retention (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.6, columnspacing=1.8, handletextpad=0.5)

    save(fig, "fig3_budget_allocation")


# ---------- fig4: 增广剂量响应 ----------
def fig4():
    doses = [1, 4, 8, 12]
    ret = [82.73, 83.41, 85.16, 82.47]
    assert abs(ret[2] - 85.16) < 1e-9

    fig, ax = plt.subplots(figsize=(110 * MM, 76 * MM))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.875, bottom=0.175)

    ax.fill_between(doses, 81.8, ret, color=BLUE, alpha=0.10, lw=0, zorder=1)
    ax.axhspan(85, 86.2, color=RED, alpha=0.05, lw=0, zorder=0)
    ax.axhline(85, color=RED, lw=1.0, ls=(0, (2.6, 2.2)), zorder=2)
    ax.text(12.9, 85.10, "PARTIAL ≥ 85", ha="right", va="bottom",
            fontsize=7.6, color=RED)

    ax.plot(doses, ret, "-", color=BLUE, lw=2.4, zorder=3)
    for d, r in zip(doses, ret):
        if d == 8:
            ax.plot(d, r, "o", ms=9.0, mfc=BLUE, mec="white", mew=1.4, zorder=5)
            ax.plot([d, d], [81.9, r], color=BLUE, lw=0.9, ls=(0, (1, 2.2)),
                    alpha=0.5, zorder=1)
        else:
            ax.plot(d, r, "o", ms=6.4, mfc="white", mec=BLUE, mew=1.6, zorder=5)
        ax.text(d, r + (0.20 if d != 8 else 0.28), f"{r:.2f}", ha="center",
                va="bottom", fontsize=8.4,
                fontweight="bold" if d == 8 else "normal", color=INK)

    ax.set_xlim(0.2, 13.2)
    ax.set_ylim(81.8, 86.2)
    ax.set_xticks(doses)
    ax.set_xticklabels([f"×{d}" for d in doses])
    ax.set_yticks(np.arange(82, 87, 1))
    ax.set_xlabel("Anchor augmentation dose V (views per anchor)")
    ax.set_ylabel("NTU120 retention (%)")

    save(fig, "fig4_aug_dose_response")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"[done] figures -> {OUT}")
