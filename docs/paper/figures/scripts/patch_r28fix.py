# -*- coding: utf-8 -*-
r"""R28 修复轮补丁（judge 7 项指控：6 实 1 半实）：
fig1 v9: 标题缩字防贴边 / 物理层五卡上移均衡 / 演化箭头落卡缘消 T
fig2 v11: 底部死白裁掉 / kappa>=tau 标签抬高
GA v7: 自环弧补 zorder(被卡片盖住=真消失) / 脚注补全句首
"""
from pathlib import Path
SD = Path(r"D:\Desktop\psd-framework\docs\paper\figures\scripts")

def patch(src, dst, pairs, label):
    t = (SD / src).read_text(encoding="utf-8")
    for old, new, cnt in pairs:
        n = t.count(old)
        assert n == cnt, f"{label}: expect {cnt}, got {n} for {old[:70]!r}"
        t = t.replace(old, new)
    (SD / dst).write_text(t, encoding="utf-8")
    print(f"OK {label} -> {dst}")

# ---------------- fig1 v8 -> v9 ----------------
patch("make_fig1_overview_v8.py", "make_fig1_overview_v9.py", [
    # 标题 8.0->7.0（8pt 时 "Semantic layer Ω (revisable)" 宽于容器 44u，贴 canvas 右缘）
    ('fontsize=8.0, fontweight="bold", color=psd_style.PHYS_TEXT, zorder=6)',
     'fontsize=7.0, fontweight="bold", color=psd_style.PHYS_TEXT, zorder=6)', 1),
    ('fontsize=8.0, fontweight="bold", color=C_SEM_EDGE, zorder=6)',
     'fontsize=7.0, fontweight="bold", color=C_SEM_EDGE, zorder=6)', 1),
    # 物理层五卡上移均衡（judge: 无边色块放大顶部空 1/3）
    ('box(4.0, 16.5, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", "white", C_PHYS_EDGE, fs=5.9)\n'
     'box(4.5, 28.5, 19.5, 9.5, "SSL\\npretraining", "white", C_PHYS_EDGE, fs=6.4)\n'
     'box(25, 28.5, 19.5, 9.5, "Motion words\\nquantization", "white", C_PHYS_EDGE, fs=6.0)\n'
     'box(4.8, 41.5, 19.0, 7.5, "Dynamics\\nembeddings", "white", C_PHYS_EDGE, fs=6.4)\n'
     'box(24.0, 41.5, 19.0, 7.5, "Behavior\\nproposals", "white", C_PHYS_EDGE, fs=6.4)\n'
     'arrow(14.25, 22.1, 14.25, 28.5)\n'
     'arrow(34.75, 22.1, 34.75, 28.5)\n'
     'arrow(14.25, 38.0, 14.4, 41.5)\n'
     'arrow(34.75, 38.0, 33.5, 41.5)\n'
     'arrow(14.5, 49.0, 14.5, 61.0); arrow(14.5, 61.0, 52.5, 61.0)\n'
     'arrow(33.5, 49.0, 33.5, 67.0); arrow(33.5, 67.0, 52.5, 67.0)',
     'box(4.0, 17.0, 40.5, 5.6, "Unlabeled streams $(T$, 24, 3)", "white", C_PHYS_EDGE, fs=5.9)\n'
     'box(4.5, 32.5, 19.5, 9.5, "SSL\\npretraining", "white", C_PHYS_EDGE, fs=6.4)\n'
     'box(25, 32.5, 19.5, 9.5, "Motion words\\nquantization", "white", C_PHYS_EDGE, fs=6.0)\n'
     'box(4.8, 50.5, 19.0, 7.5, "Dynamics\\nembeddings", "white", C_PHYS_EDGE, fs=6.4)\n'
     'box(24.0, 50.5, 19.0, 7.5, "Behavior\\nproposals", "white", C_PHYS_EDGE, fs=6.4)\n'
     'arrow(14.25, 22.6, 14.25, 32.5)\n'
     'arrow(34.75, 22.6, 34.75, 32.5)\n'
     'arrow(14.25, 42.0, 14.4, 50.5)\n'
     'arrow(34.75, 42.0, 33.5, 50.5)\n'
     'arrow(14.5, 58.0, 14.5, 65.0); arrow(14.5, 65.0, 52.5, 65.0)\n'
     'arrow(33.5, 58.0, 33.5, 71.0); arrow(33.5, 71.0, 52.5, 71.0)', 1),
    # 演化虚线箭头: 终点从容器底缘(16.4, 箭头头叠边框成 T)改落 seeds 卡底缘上方
    ('arrow(84, 8.0, 86.2, 16.4, color=C_SEM_EDGE, lw=1.25, ls=(0, (3, 2)))',
     'arrow(84, 8.0, 86.2, 16.9, color=C_SEM_EDGE, lw=1.25, ls=(0, (3, 2)))', 1),
], "fig1")

# ---------------- fig2 v10 -> v11 ----------------
patch("make_fig2_pseudo_label_loop_v10.py", "make_fig2_pseudo_label_loop_v11.py", [
    ('ax.set_xlim(0, 100); ax.set_ylim(0, 90)',
     'ax.set_xlim(0, 100); ax.set_ylim(17.5, 90)   # R28: 裁底部死白', 1),
    ('seg("assign", "pool", label=r"$\\kappa \\geq \\tau$", ldy=5.2)',
     'seg("assign", "pool", label=r"$\\kappa \\geq \\tau$", ldy=7.0)   # R28: 抬离卡顶缘', 1),
], "fig2")

# ---------------- GA v6 -> v7 ----------------
patch("make_ga_graphical_abstract_v6.py", "make_ga_graphical_abstract_v7.py", [
    # 真 bug: 弧无 zorder(默认2)被 card(z=3) 填充盖没
    ('ax.add_patch(FancyArrowPatch((46, 7.2), (63, 7.2), arrowstyle="-|>",\n'
     '                             mutation_scale=11, color=C_SEM_EDGE, lw=1.6,\n'
     '                             linestyle=(0, (4, 2.2)),\n'
     '                             connectionstyle="arc3,rad=0.32"))',
     'ax.add_patch(FancyArrowPatch((46, 7.2), (63, 7.2), arrowstyle="-|>",\n'
     '                             mutation_scale=11, color=C_SEM_EDGE, lw=1.6,\n'
     '                             linestyle=(0, (4, 2.2)), zorder=6,\n'
     '                             connectionstyle="arc3,rad=0.32"))', 1),
    ('"of own full-budget top-1 at 10% labels; canine: 9.8 vs 11.1% chance at 13% budget"',
     '"retention of own full-budget top-1 at 10% labels; canine: 9.8 vs 11.1% chance at 13% budget"', 1),
], "GA")
print("ALL R28-FIX PATCHES APPLIED")
