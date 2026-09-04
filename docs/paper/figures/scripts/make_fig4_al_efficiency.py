# -*- coding: utf-8 -*-
"""fig4 主动学习效率曲线（负结果如实呈现）生成脚本。

数据来源: reports/p05-al-efficiency-short-2026-08-24.json（W14 归档，合成层短预算协议）
  - curves 字段: {entropy, random} × 预算 {20,50,100,200} × mean/std(3 seeds)
  - 负结果事实: 预算 b>=100 时 random 曲线高于 entropy 曲线，本图不隐藏不美化，
    并在图内以浅灰注释显式指出（caption 叙述见 FIGURE_SOURCE.md）。
规格来源: docs/paper/experiment-skeleton.md §图表规范 + docs/paper/figure-specs.md §fig4
  - 白底 / #DADADA 网格 / 色盲安全青橙对（复用 fig1 配色族）/ 无图内标题 / PDF 矢量 + PNG 600dpi
  - 规格偏差登记: figure-specs §fig4 待定项中的 "目标线 y=85%" 不绘制——该验收线属真实 K9 层口径,
    本数据为合成层协议(meta.layer_note 明确禁止外推)，只画任务书指定的 4.5% 随机猜测基线。
风格一致性: 复用 scripts/make_fig1_overview.py 的配色与 rcParams（DejaVu Sans / pdf.fonttype 42）。
输出: docs/paper/figures/fig4_al_efficiency.pdf + fig4_al_efficiency.png(600dpi)
"""

from pathlib import Path

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 路径（脚本位于 <root>/docs/paper/figures/scripts/）----
ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "reports" / "p05-al-efficiency-short-2026-08-24.json"
OUT_DIR = ROOT / "docs" / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
assert DATA.is_file(), f"data missing: {DATA}"

# ---- 配色（与 make_fig1_overview.py 同族，≤6 色）----
INK = "#000000"
CYAN_DARK = "#0E7490"    # entropy 臂（蓝系深色）
ORANGE_DARK = "#C2410C"  # random 臂（橙系深色）
NOTE_GRAY = "#888888"
GRID_GRAY = "#DADADA"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})

# ---- 读数（直接来自归档 JSON，不在脚本里硬编码任何实验数字）----
with open(DATA, encoding="utf-8") as f:
    payload = json.load(f)

BUDGETS = [20, 50, 100, 200]
SERIES = {
    "entropy": dict(color=CYAN_DARK, marker="o", ls="solid",
                    label="Uncertainty sampling (softmax entropy)"),
    "random": dict(color=ORANGE_DARK, marker="s", ls=(0, (5, 2.5)),
                   label="Random selection"),
}
stats = {}
for arm in SERIES:
    stats[arm] = {
        b: (payload["curves"][arm][str(b)]["mean"] * 100.0,
            payload["curves"][arm][str(b)]["std"] * 100.0)
        for b in BUDGETS
    }

CHANCE_PCT = 4.5  # 22 类随机猜测基线（K9 实验，HANDOVER §7）

# ---- 绘图 ----
fig, axes = plt.subplots(2, 1, figsize=(3.42, 4.45), sharex=True)
ax = axes[0]
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for arm, style in SERIES.items():
    means = [stats[arm][b][0] for b in BUDGETS]
    stds = [stats[arm][b][1] for b in BUDGETS]
    ax.errorbar(BUDGETS, means, yerr=stds,
                color=style["color"], marker=style["marker"],
                markersize=6, linewidth=1.8, linestyle=style["ls"],
                elinewidth=1.1, capsize=4, capthick=1.1,
                label=style["label"], zorder=3)

# 随机猜测基线（22 类 -> 4.5%）
ax.axhline(CHANCE_PCT, color=NOTE_GRAY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
ax.text(203, 2.2, "random-guess baseline 4.5% (22 classes)",
        ha="right", va="center", fontsize=6.2, color=NOTE_GRAY,
        bbox=dict(facecolor="white", edgecolor="none", pad=1.0))

# 负结果如实标注（不隐藏不美化）
ax.annotate("random exceeds uncertainty\nfor budgets \u2265 100\n(cold-start protocol)",
            xy=(200, stats["random"][200][0]), xytext=(112, 46),
            fontsize=6.2, color=NOTE_GRAY, style="italic", ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=NOTE_GRAY, linewidth=0.9,
                            shrinkA=2, shrinkB=4))

ax.set_ylabel("Best validation accuracy (%)", fontsize=7)
ax.set_title("(a) cold-start scorers", loc="left", fontsize=7)
ax.set_xticks(BUDGETS)
ax.set_xlim(8, 212)
ax.set_ylim(0, 100)
ax.grid(True, color=GRID_GRAY, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    ax.spines[side].set_linewidth(1.0)
ax.tick_params(labelsize=6)
ax.legend(loc="upper left", frameon=False, fontsize=6.2)

# ---- panel (b): warm-started in-domain scorers (stronger negative evidence, same protocol) ----
WDATA = ROOT / "reports" / "p05-al-efficiency-warmstart-short-2026-08-25.json"
assert WDATA.is_file(), f"data missing: {WDATA}"
wpay = json.loads(WDATA.read_text(encoding="utf-8"))
axb = axes[1]
for arm, style in SERIES.items():
    means = [wpay["curves"][arm][str(b)]["mean"] * 100.0 for b in BUDGETS]
    stds = [wpay["curves"][arm][str(b)]["std"] * 100.0 for b in BUDGETS]
    axb.errorbar(BUDGETS, means, yerr=stds,
                 color=style["color"], marker=style["marker"],
                 markersize=6, linewidth=1.8, linestyle=style["ls"],
                 elinewidth=1.1, capsize=4, capthick=1.1,
                 label=style["label"], zorder=3)
axb.axhline(CHANCE_PCT, color=NOTE_GRAY, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
axb.annotate("random leads by 4.2–5.0 pp\nfor $b \\geq 50$\n(b=20: shared initial set)",
             xy=(200, wpay["curves"]["random"]["200"]["mean"] * 100.0), xytext=(60, 55),
             fontsize=6.2, color=NOTE_GRAY, style="italic", ha="left", va="center",
             arrowprops=dict(arrowstyle="-", color=NOTE_GRAY, linewidth=0.9,
                             shrinkA=2, shrinkB=4))
axb.set_xlabel("Annotation budget (labeled clips)", fontsize=7)
axb.set_ylabel("Best validation accuracy (%)", fontsize=7)
axb.set_title("(b) warm-started in-domain scorers", loc="left", fontsize=7)
axb.set_xticks(BUDGETS)
axb.set_ylim(0, 100)
axb.grid(True, color=GRID_GRAY, linewidth=0.8, zorder=0)
axb.set_axisbelow(True)
for side in ('top', 'right'):
    axb.spines[side].set_visible(False)
for side in ('left', 'bottom'):
    axb.spines[side].set_linewidth(1.0)
axb.tick_params(labelsize=6)
axb.legend(loc="lower right", frameon=False, fontsize=6.2)
fig.subplots_adjust(hspace=0.35)

pdf_path = OUT_DIR / "fig4_al_efficiency.pdf"
png_path = OUT_DIR / "fig4_al_efficiency.png"
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.10)
fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.10)
plt.close(fig)

# ---- 当次运行验证输出 ----
print("written:", pdf_path)
print("written:", png_path)
print("--- plotted numbers (from JSON, %) ---")
for arm in SERIES:
    row = ", ".join(f"b={b}: {stats[arm][b][0]:.1f}\u00b1{stats[arm][b][1]:.1f}" for b in BUDGETS)
    print(f"{arm:>8}: {row}")
neg = all(stats["random"][b][0] >= stats["entropy"][b][0] for b in (100, 200))
print(f"negative-result fact preserved (random >= entropy at b>=100): {neg}")
