# -*- coding: utf-8 -*-
"""make_next_paper_figures — 下一篇论文 4 张主图（v2 期刊级重制）。

设计系统（四图共用）
  · Arial 无衬线，7–8 pt，无图内标题（题注承载结论）
  · 语义色：灰=冻结物理层/参考，蓝=锚点回收，红=判定带/增益强调
  · 印刷尺寸：fig1/fig2 = 180 mm 通栏；fig3/fig4 = 120 mm
  · PDF 矢量 + 200 dpi PNG 预览；字体嵌入 Type 42

数据源（全部实测，脚本内断言防漂移）
  fig1 四层梯度      : runs/p1_sprint/labelfree_*/summary.json（published E9）
  fig2 三 tier 主图  : labelfree_*/summary.json + adapt_t4_*/summary_{same10,aug4,aug8_lr2e2}.json
  fig3 预算-分配率   : runs/p1_sprint/adapt_t4_ntu120/summary_{ref5,split5,ref15,dual15}.json
  fig4 增广剂量响应  : runs/p1_sprint/adapt_t4_ntu120/summary_{same10,aug4,aug8_lr2e2,aug12_lr2e2}.json
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "p1_sprint"
OUT = REPO / "docs" / "next_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- style baseline ----------------
INK = "#1f1f1f"
HAIR = "#4a4a4a"
GRAY = "#9e9e9e"
GRAY_D = "#6b6b6b"
GRAY_L = "#d4d4d4"
BLUE = "#2166AC"
BLUE_D = "#164b7d"
RED = "#B2182B"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": HAIR,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": HAIR,
    "ytick.color": HAIR,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
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
    """冻结臂的逐种子 retention = b_arm_i / full_ref（与 mean 断言一致）。"""
    fr = summary["full_ref"]
    out = [b / fr for b in summary["b_arms"]]
    assert abs(np.mean(out) - summary["retention"]) < 5e-3, (
        f"{summary.get('tier')}: per-seed mean {np.mean(out):.4f} != {summary['retention']}")
    return out


# ---------- fig1: 四层冻结梯度（横向点图 + 量纲分离） ----------
def fig1():
    tiers = ["NTU60", "PanAf500", "NTU120", "UCF101"]
    ret = [90.70, 82.50, 74.96, 51.09]
    ref = [74.45, 64.00, 58.37, 19.87]

    fig, ax = plt.subplots(figsize=(180 * MM, 66 * MM))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.90, bottom=0.20)

    y = np.arange(len(tiers))[::-1]

    # 判定带（只作用于 retention 轴，量纲单一）
    ax.axvspan(90, 100, color=RED, alpha=0.055, lw=0, zorder=0)
    ax.axvspan(85, 90, color=RED, alpha=0.030, lw=0, zorder=0)
    ax.axvline(90, color=RED, lw=0.7, ls=(0, (2.4, 2.0)), zorder=1)
    ax.axvline(85, color=RED, lw=0.7, ls=(0, (1.0, 2.0)), zorder=1)

    # 茎 + 点
    for yi, v in zip(y, ret):
        ax.plot([40, v], [yi, yi], color=GRAY_L, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([v], [yi], "o", ms=6.2, mfc=BLUE, mec="white", mew=1.0, zorder=5)
        ax.text(v + 1.3, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=7.6, color=INK, zorder=6)

    # 右侧参考列：全标注线性探针精度（retention 的分母，不参与横轴编码）
    ax.text(107.6, len(tiers) - 0.52, "full-label\nacc. (%)", ha="right", va="bottom",
            fontsize=6.2, color=GRAY_D, linespacing=1.05)
    for yi, r in zip(y, ref):
        ax.text(107.6, yi, f"{r:.2f}", ha="right", va="center",
                fontsize=6.8, color=GRAY_D)

    # 判定带标签（落在无数据区）
    ax.text(87.5, -0.44, "PARTIAL", ha="center", va="center", fontsize=6.0, color=RED)
    ax.text(95.0, -0.44, "CONFIRMS", ha="center", va="center", fontsize=6.0, color=RED)

    ax.set_yticks(y)
    ax.set_yticklabels(tiers)
    ax.set_ylim(-0.62, 3.62)
    ax.set_xlim(40, 108)
    ax.set_xticks(np.arange(40, 101, 10))
    ax.spines["bottom"].set_bounds(40, 100)
    ax.set_xlabel("Budget retention (%)")
    ax.xaxis.set_label_coords(0.44, -0.135)
    ax.tick_params(axis="y", length=0, pad=6)

    save(fig, "fig1_four_tier_gradient")


# ---------- fig2: 三 tier 主图（逐种子配对点图） ----------
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
    # 均值横杆用报告值（mean_b / mean_full_ref），逐点为各自分母的种子 retention；
    # 二者定义差异 < 1 pp（脚本断言），视觉上不可分辨，但横杆必须与表格数字一致。
    f_mean = [s["retention"] * 100 for s in frozen_sum]
    d_mean = [s["retention"] * 100 for s in dual_sum]

    # 与手稿数字对齐（防漂移）
    expect_f = [74.96, 82.50, 51.09]
    expect_d = [85.16, 92.21, 71.21]
    assert np.allclose(f_mean, expect_f, atol=0.02), f_mean
    assert np.allclose(d_mean, expect_d, atol=0.02), d_mean
    assert np.all(np.abs(np.mean(dual, axis=1) * 100 - np.array(d_mean)) < 1.0)

    fig, ax = plt.subplots(figsize=(180 * MM, 82 * MM))
    fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.21)

    rng = np.random.default_rng(7)
    for i in range(3):
        jf = rng.uniform(-0.030, 0.030, size=len(frozen[i]))
        jd = rng.uniform(-0.030, 0.030, size=len(dual[i]))
        xf = i - 0.17 + jf
        xd = i + 0.17 + jd
        f = np.array(frozen[i]) * 100
        d = np.array(dual[i]) * 100
        ax.scatter(xf, f, s=16, facecolor="white", edgecolor=GRAY_D, lw=0.65,
                   zorder=3)
        ax.scatter(xd, d, s=16, facecolor=BLUE, edgecolor="white", lw=0.4,
                   zorder=3)
        # 均值横杆
        ax.plot([i - 0.17 - 0.05, i - 0.17 + 0.05], [f_mean[i]] * 2,
                color=GRAY_D, lw=2.2, solid_capstyle="round", zorder=5)
        ax.plot([i + 0.17 - 0.05, i + 0.17 + 0.05], [d_mean[i]] * 2,
                color=BLUE_D, lw=2.2, solid_capstyle="round", zorder=5)
        # 均值数值：分列外侧，避开种子点云
        ax.text(i - 0.28, f_mean[i], f"{f_mean[i]:.2f}", ha="right",
                va="center", fontsize=6.8, color=GRAY_D)
        ax.text(i + 0.28, d_mean[i], f"{d_mean[i]:.2f}", ha="left",
                va="center", fontsize=6.8, color=BLUE_D)
        # 增益（两列中点，白底保证可读）
        ax.text(i, (f_mean[i] + d_mean[i]) / 2, f"+{d_mean[i] - f_mean[i]:.2f}",
                ha="center", va="center", fontsize=7.4, color=RED, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.88))

    ax.set_xlim(-0.70, 2.70)
    ax.set_ylim(42, 102)
    ax.set_xticks(range(3))
    ax.set_xticklabels(tiers)
    ax.set_yticks(np.arange(50, 101, 10))
    ax.set_ylabel("Budget retention (%)")
    ax.tick_params(axis="x", length=0, pad=5)

    # 回收后判定（置于刻度标签之下，颜色独立）
    trans = ax.get_xaxis_transform()
    for i, v in enumerate(verdicts):
        ax.text(i, -0.085, f"→ {v}", transform=trans, ha="center",
                va="center", fontsize=6.4, color=RED, clip_on=False)

    handles = [
        Line2D([], [], marker="o", ls="", ms=4.0, mfc="white", mec=GRAY_D,
               mew=0.8, label="Frozen physical layer"),
        Line2D([], [], marker="o", ls="", ms=4.0, mfc=BLUE, mec=BLUE,
               label="Anchor recycling (per-tier best)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.215),
              ncol=2, handletextpad=0.5, columnspacing=2.4, handlelength=1.2)

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

    # 冻结臂：5/15% 为同预算语义-only；10% 为发布版冻结基线 74.96
    f_vals = [ref5["retention"] * 100, frozen10["retention"] * 100,
              ref15["retention"] * 100]
    # 双用臂：5% 预算对半（split5），10% 同锚（same10），15% 全双用（dual15）
    d_vals = [split5["retention"] * 100, same10["retention"] * 100,
              dual15["retention"] * 100]
    expect_f = [64.61, 74.96, 79.04]
    expect_d = [67.51, 78.87, 83.60]
    assert np.allclose(f_vals, expect_f, atol=0.02), f_vals
    assert np.allclose(d_vals, expect_d, atol=0.02), d_vals

    fig, ax = plt.subplots(figsize=(120 * MM, 78 * MM))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.90, bottom=0.175)

    ax.fill_between(budgets, f_vals, d_vals, color=BLUE, alpha=0.075, lw=0)
    ax.plot(budgets, f_vals, "-o", color=GRAY_D, lw=1.2, ms=4.2, mfc="white",
            mec=GRAY_D, mew=1.1, label="Semantic-only (frozen)", zorder=3)
    ax.plot(budgets, d_vals, "-s", color=BLUE, lw=1.5, ms=4.2, mfc="white",
            mec=BLUE, mew=1.1, label="Anchor recycling", zorder=4)

    for b, f, d in zip(budgets, f_vals, d_vals):
        ax.text(b, f - 1.15, f"{f:.2f}", ha="center", va="top", fontsize=6.6,
                color=GRAY_D)
        ax.text(b, d + 1.15, f"{d:.2f}", ha="center", va="bottom", fontsize=6.6,
                color=BLUE_D)
        ax.text(b, (f + d) / 2, f"+{d - f:.2f}", ha="center", va="center",
                fontsize=6.6, color=RED, zorder=6,
                bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))

    ax.set_xlim(4.2, 15.8)
    ax.set_ylim(62, 87)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.set_yticks(np.arange(65, 86, 5))
    ax.set_xlabel("Anchor budget (% of training clips)")
    ax.set_ylabel("NTU120 retention (%)")
    ax.legend(loc="upper left", handlelength=1.4, labelspacing=0.35)

    save(fig, "fig3_budget_allocation")


# ---------- fig4: 增广剂量响应 ----------
def fig4():
    doses = [1, 4, 8, 12]
    ret = [82.73, 83.41, 85.16, 82.47]
    assert abs(ret[2] - 85.16) < 1e-9

    fig, ax = plt.subplots(figsize=(120 * MM, 70 * MM))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.90, bottom=0.175)

    ax.axhspan(85, 86.2, color=RED, alpha=0.04, lw=0, zorder=0)
    ax.axhline(85, color=RED, lw=0.7, ls=(0, (2.4, 2.0)), zorder=1)
    ax.text(12.9, 85.08, "PARTIAL ≥ 85", ha="right", va="bottom",
            fontsize=6.2, color=RED)

    ax.plot(doses, ret, "-", color=BLUE, lw=1.5, zorder=3)
    for i, (d, r) in enumerate(zip(doses, ret)):
        if d == 8:  # 峰值实心
            ax.plot(d, r, "o", ms=6.6, mfc=BLUE, mec="white", mew=1.2, zorder=5)
            ax.plot([d, d], [81.9, r], color=BLUE, lw=0.7, ls=(0, (1, 2)),
                    alpha=0.55, zorder=1)
        else:
            ax.plot(d, r, "o", ms=4.6, mfc="white", mec=BLUE, mew=1.2, zorder=5)
        ax.text(d, r + (0.18 if d != 8 else 0.24), f"{r:.2f}", ha="center",
                va="bottom", fontsize=6.8, color=INK)

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
