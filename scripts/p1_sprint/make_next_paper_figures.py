# -*- coding: utf-8 -*-
"""make_next_paper_figures — 下一篇论文 4 张主图（v4：Okabe-Ito + 多图型）。

v4 依据工具箱 `paper-figure-palette`（journal-submission 场景 = okabe-ito）
与 `fig-visualization-advisor`（图型多样化）重做：
  · 配色：Okabe-Ito 色盲友好亮色系（天蓝 #56B4E9 / 蓝 #0072B2 / 橙 #E69F00 /
          绿 #009E73 / 朱红 #D55E00 / 中性灰 #B5B5B5），弃深海军蓝
  · 图型：fig1 横条（0–100 轨道）· fig2 三分面斜率图（逐种子轨迹）·
          fig3 折线+增益带 · fig4 箱线+散点+均值趋势（分布图）
  · 每个分面独立坐标 + 分面字母；无图内标题

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

# ---------------- Summer-beach palette (A题提交 论文同款) ----------------
# 取自 D:/Desktop/A题提交.zip 论文源码：summer-beach-v6 填充色 + 同色相墨色
INK = "#2B2B2E"
HAIR = "#3D3D3D"
NEU = "#B6B6B2"
NEU_D = "#57575A"
TRACK = "#F2F4F7"
CYAN = "#6FCDFD"
CYAN2 = "#2BC3F1"
TEAL = "#19667F"
TEAL_D = "#307697"
MINT = "#9FE698"
GREEN = "#438040"
CREAM = "#FEE6A9"
PEACH = "#FEBDAA"
CORAL = "#FEB8B5"
CORAL2 = "#FDA574"
WARM = "#A3490D"
WARM2 = "#BE4E30"
WARM3 = "#A35B31"

MM = 1 / 25.4
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 7.8,
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
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.pad": 3.5,
    "ytick.major.pad": 3.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

VERDICT_COLOR = {"CONFIRMS": GREEN, "PARTIAL": WARM3, "FAILS": WARM2}
PALE = "#E6F7FE"


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"[fig] {name}.pdf/.png")


def seed_retention(summary: dict) -> list[float]:
    fr = summary["full_ref"]
    out = [b / fr for b in summary["b_arms"]]
    assert abs(np.mean(out) - summary["retention"]) < 5e-3
    return out


# ---------- fig1: 四层梯度（0–100 轨道横条，亮天蓝） ----------
def fig1():
    tiers = ["NTU60", "PanAf500", "NTU120", "UCF101"]
    ret = [90.70, 82.50, 74.96, 51.09]
    ref = [74.45, 64.00, 58.37, 19.87]

    fig, ax = plt.subplots(figsize=(165 * MM, 78 * MM))
    fig.subplots_adjust(left=0.185, right=0.965, top=0.86, bottom=0.185)

    y = np.arange(len(tiers))[::-1]
    ax.barh(y, np.full(len(tiers), 100.0), height=0.58, color=TRACK, lw=0, zorder=0)
    from matplotlib.colors import LinearSegmentedColormap
    bar_cmap = LinearSegmentedColormap.from_list("sb", ["#C9EEFD", CYAN2])
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    for yi, v in zip(y, ret):
        ax.imshow(grad, extent=(0, v, yi - 0.29, yi + 0.29), aspect="auto",
                  cmap=bar_cmap, interpolation="bicubic", zorder=2)
    ax.barh(y, ret, height=0.58, color="none", edgecolor=TEAL_D, lw=0.8, zorder=2.5)

    ax.axvline(90, color=WARM, lw=1.0, ls=(0, (2.6, 2.2)), zorder=3)
    ax.axvline(85, color=WARM, lw=1.0, ls=(0, (1.0, 2.0)), zorder=3)
    ax.text(87.5, 0.36, "PARTIAL", ha="center", va="bottom", rotation=90,
            fontsize=7.0, color=WARM)
    ax.text(95.0, 0.36, "CONFIRMS", ha="center", va="bottom", rotation=90,
            fontsize=7.0, color=WARM)

    for yi, v in zip(y, ret):
        ax.text(v - 2.0, yi, f"{v:.2f}", va="center", ha="right",
                fontsize=9.4, fontweight="bold", color=INK, zorder=5)

    for yi, name, r in zip(y, tiers, ref):
        ax.text(-1.8, yi + 0.13, name, ha="right", va="center",
                fontsize=9.4, fontweight="bold", color=INK, clip_on=False)
        ax.text(-1.8, yi - 0.20, f"ref {r:.2f}", ha="right", va="center",
                fontsize=7.4, color=NEU_D, clip_on=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.72, 3.55)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks([])
    ax.set_xlabel("Budget retention (%)")
    ax.tick_params(axis="y", length=0)

    save(fig, "fig1_four_tier_gradient")


# ---------- fig2: 三 tier 主结果（三分面斜率图） ----------
def fig2():
    tiers = ["NTU120", "PanAf500", "UCF101"]
    letters = ["a", "b", "c"]
    frozen_sum = [load(RUNS / "labelfree_ntu120" / "summary.json"),
                  load(RUNS / "labelfree_panaf500" / "summary.json"),
                  load(RUNS / "labelfree_ucf101" / "summary.json")]
    dual_sum = [load(RUNS / "adapt_t4_ntu120" / "summary_aug8_lr2e2.json"),
                load(RUNS / "adapt_t4_panaf500" / "summary_aug4.json"),
                load(RUNS / "adapt_t4_ucf101" / "summary_same10.json")]
    verdicts = ["PARTIAL", "CONFIRMS", "FAILS"]
    ylims = [(68, 89), (54, 103), (34, 102)]

    frozen = [seed_retention(s) for s in frozen_sum]
    dual = [[r["retention_seed"] for r in s["per_seed"]] for s in dual_sum]
    f_mean = [s["retention"] * 100 for s in frozen_sum]
    d_mean = [s["retention"] * 100 for s in dual_sum]
    assert np.allclose(f_mean, [74.96, 82.50, 51.09], atol=0.02)
    assert np.allclose(d_mean, [85.16, 92.21, 71.21], atol=0.02)

    fig, axes = plt.subplots(1, 3, figsize=(165 * MM, 82 * MM),
                             gridspec_kw={"wspace": 0.30})
    fig.subplots_adjust(left=0.085, right=0.985, top=0.86, bottom=0.20)

    yticks = [[70, 75, 80, 85], [60, 70, 80, 90, 100], [40, 50, 60, 70, 80, 90, 100]]
    for ax, letter, name, f, d, fm, dm, vd, yl, yt in zip(
            axes, letters, tiers, frozen, dual, f_mean, d_mean, verdicts, ylims, yticks):
        f = np.array(f) * 100
        d = np.array(d) * 100
        for a, b in zip(f, d):  # 逐种子轨迹
            ax.plot([0, 1], [a, b], color="#DCDCDC", lw=0.7, alpha=0.85, zorder=1)
        ax.fill_between([0, 1], [fm, fm], [fm, dm], color=CYAN, alpha=0.35,
                        lw=0, zorder=2)
        ax.plot([0, 1], [fm, dm], color=TEAL, lw=2.4, zorder=3, solid_capstyle="round")
        ax.plot([0], [fm], "o", ms=6.6, mfc="white", mec=NEU_D, mew=1.3, zorder=5)
        ax.plot([1], [dm], "o", ms=6.6, mfc=TEAL, mec="white", mew=1.0, zorder=5)

        ax.set_title(f"{letter}   {name}", loc="left", fontsize=9.2,
                     fontweight="bold", pad=7)
        ax.text(1.0, 1.035, f"→ {vd}", transform=ax.transAxes, ha="right",
                va="bottom", fontsize=7.4, color=VERDICT_COLOR[vd])
        ax.text(0.5, -0.125, f"+{dm - fm:.2f} pp", transform=ax.transAxes,
                ha="center", va="top", fontsize=9.2, fontweight="bold", color=WARM2)
        ax.set_xlim(-0.34, 1.34)
        ax.set_ylim(*yl)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Frozen", "Recycled"])
        ax.set_yticks(yt)

    axes[0].set_ylabel("Budget retention (%)")

    save(fig, "fig2_tier_main_results")


# ---------- fig3: 预算-分配率曲线（折线 + 增益带） ----------
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
    assert np.allclose(f_vals, [64.61, 74.96, 79.04], atol=0.02)
    assert np.allclose(d_vals, [67.51, 78.87, 83.60], atol=0.02)

    fig, ax = plt.subplots(figsize=(110 * MM, 82 * MM))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.825, bottom=0.175)

    ax.fill_between(budgets, f_vals, d_vals, color=CYAN, alpha=0.28, lw=0)
    ax.plot(budgets, f_vals, "-o", color=NEU, lw=2.0, ms=6.2, mfc="white",
            mec=NEU, mew=1.6, label="Semantic-only (frozen)", zorder=3)
    ax.plot(budgets, d_vals, "-s", color=TEAL, lw=2.6, ms=6.2, mfc="white",
            mec=TEAL, mew=1.6, label="Anchor recycling", zorder=4)

    for b, f, d in zip(budgets, f_vals, d_vals):
        ax.text(b, d + 1.3, f"{d:.2f}", ha="center", va="bottom", fontsize=8.4,
                color=TEAL, fontweight="bold")
        gx, gha = (b + 0.30, "left") if b < 15 else (b - 0.30, "right")
        ax.text(gx, (f + d) / 2, f"+{d - f:.2f}", ha=gha, va="center",
                fontsize=8.2, fontweight="bold", color=WARM2, zorder=6)

    ax.set_xlim(4.2, 15.8)
    ax.set_ylim(62, 87)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b}%" for b in budgets])
    ax.set_yticks(np.arange(65, 86, 5))
    ax.set_xlabel("Anchor budget (% of training clips)")
    ax.set_ylabel("NTU120 retention (%)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handlelength=1.7, columnspacing=2.0, handletextpad=0.5)

    save(fig, "fig3_budget_allocation")


# ---------- fig4: 增广剂量响应（箱线 + 散点 + 均值趋势；×12 为单种子探针） ----------
def fig4():
    tags = ["sweep_lr2e2", "aug4_lr2e2", "aug8_lr2e2", "aug12_lr2e2"]
    labels = ["×1", "×4", "×8", "×12"]
    sums = [load(RUNS / "adapt_t4_ntu120" / f"summary_{t}.json") for t in tags]
    data = [np.array([r["retention_seed"] for r in s["per_seed"]]) * 100 for s in sums]
    means = [s["retention"] * 100 for s in sums]
    assert [len(d) for d in data] == [10, 10, 10, 1], [len(d) for d in data]
    assert np.allclose(means, [82.73, 83.41, 85.16, 82.47], atol=0.02), means

    fig, ax = plt.subplots(figsize=(110 * MM, 78 * MM))
    fig.subplots_adjust(left=0.145, right=0.975, top=0.875, bottom=0.175)

    pos = np.arange(4)
    ax.boxplot(data[:3], positions=pos[:3], widths=0.50, patch_artist=True,
               showfliers=False, whiskerprops=dict(color=TEAL_D, lw=1.0),
               capprops=dict(color=TEAL_D, lw=1.0),
               medianprops=dict(color=TEAL, lw=1.6),
               boxprops=dict(facecolor=PALE, edgecolor=TEAL_D, lw=1.1))
    rng = np.random.default_rng(11)
    for i in range(3):
        d = data[i]
        jit = rng.uniform(-0.11, 0.11, size=len(d))
        ax.scatter(i + jit, d, s=15, color=CYAN2, alpha=0.95, lw=0.5,
                   edgecolor=TEAL_D, zorder=4)
    # ×12：单种子探针，用空心菱形表示，不做箱线
    ax.plot([3], [means[3]], "D", ms=7.2, mfc="white", mec=WARM2, mew=1.6, zorder=6)
    ax.text(3, means[3] - 1.0, "n=1", ha="center", va="top", fontsize=6.6, color=NEU_D)

    ax.plot(pos, means, "-o", color=WARM2, lw=2.2, ms=5.2, mfc="white",
            mec=WARM2, mew=1.6, zorder=5, label="Mean")

    ax.axhline(85, color=WARM, lw=1.0, ls=(0, (2.6, 2.2)), zorder=1)
    ax.text(3.42, 85.12, "PARTIAL ≥ 85", ha="right", va="bottom",
            fontsize=7.6, color=WARM)
    ax.text(2, means[2] + 1.0, f"{means[2]:.2f}", ha="center", va="bottom",
            fontsize=8.6, fontweight="bold", color=INK)

    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(79.0, 88.6)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(80, 89, 2))
    ax.set_xlabel("Anchor augmentation dose V (views per anchor)")
    ax.set_ylabel("NTU120 retention (%)")

    save(fig, "fig4_aug_dose_response")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print(f"[done] figures -> {OUT}")
