# -*- coding: utf-8 -*-
r"""R26 NS 九色板交割补丁：fig3/fig5/GA 版本递进 + caption 颜色词同步。
fig1/fig2/fig4 全走 psd_style token，重跑即生效，不升版。"""
from pathlib import Path

SD = Path(r"D:\Desktop\psd-framework\docs\paper\figures\scripts")
LTX = Path(r"D:\Desktop\psd-framework\docs\paper\latex")

def patch(src, dst, pairs, label):
    t = (SD / src).read_text(encoding="utf-8")
    for old, new, cnt in pairs:
        n = t.count(old)
        assert n == cnt, f"{label}: expect {cnt}, got {n} for {old[:70]!r}"
        t = t.replace(old, new)
    (SD / dst).write_text(t, encoding="utf-8")
    print(f"OK {label} -> {dst}")

def tex_patch(path, old, new, cnt=1):
    t = path.read_text(encoding="utf-8")
    n = t.count(old)
    assert n == cnt, f"{path.name}: expect {cnt}, got {n} for {old[:70]!r}"
    path.write_text(t.replace(old, new), encoding="utf-8")
    print(f"OK tex {path.name}: {old[:50]!r}")

# ---- fig3 v4 -> v5: 基线带/注记色独立（不随网格 token 变浅） ----
patch("make_fig3_segmentation_qualitative_v4.py", "make_fig3_segmentation_qualitative_v5.py", [
    ('BAND_GRAY, BASE_EDGE = psd_style.GRID_GRAY, "#AAAAAA"',
     'BAND_GRAY, BASE_EDGE = "#D7DCE2", "#AEB6BF"   # 基线带蓝灰调（R26 与 NS 板同调）', 1),
    ('NOTE_GRAY, GRID_GRAY = "#888888", psd_style.GRID_GRAY',
     'NOTE_GRAY, GRID_GRAY = "#7F8C8D", psd_style.GRID_GRAY', 1),
], "fig3")

# ---- fig5 v3 -> v4: 七系列 NS 全色板 + 黄系列深描边 ----
patch("make_fig5_budget_retention_v3.py", "make_fig5_budget_retention_v4.py", [
    ('"public-real v1", "o", psd_style.PHYS_EDGE', '"public-real v1", "o", psd_style.S_PUBV1', 1),
    ('"public-real v2", "s", psd_style.SEM_EDGE', '"public-real v2", "s", psd_style.S_PUBV2', 1),
    ('"human benchmark (NTU60)", "D", psd_style.GRAY_2', '"human benchmark (NTU60)", "D", psd_style.S_NTU60', 1),
    ('"human benchmark (NTU120, HRNet 2D)", "v", psd_style.GRAY_3',
     '"human benchmark (NTU120, HRNet 2D)", "v", psd_style.S_NTU120', 1),
    ('"independent benchmark (UCF101, HRNet 2D)", "P", psd_style.GRAY_1',
     '"independent benchmark (UCF101, HRNet 2D)", "P", psd_style.S_UCF', 1),
    ('"animal public benchmark (PanAf500)", "X", psd_style.GRAY_4',
     '"animal public benchmark (PanAf500)", "X", psd_style.S_PANAF', 1),
    ('"synthetic-offset", "^", psd_style.VIOLET', '"synthetic-offset", "^", psd_style.S_SYNTH', 1),
    ('ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=7.0, ls="none",\n'
     '                markerfacecolor=col, markeredgecolor="white", markeredgewidth=1.0,',
     'edge = psd_style.S_SYNTH_EDGE if col == psd_style.S_SYNTH else "white"  # 黄系深描边保可见\n'
     '    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=7.0, ls="none",\n'
     '                markerfacecolor=col, markeredgecolor=edge, markeredgewidth=1.0,', 1),
], "fig5")

# ---- GA v3 -> v4: 人基准条与 fig5 同步取色 ----
patch("make_ga_graphical_abstract_v3.py", "make_ga_graphical_abstract_v4.py", [
    ('("NTU60", 90.7, False, psd_style.GRAY_2),', '("NTU60", 90.7, False, psd_style.S_NTU60),', 1),
    ('("NTU120", 88.9, False, psd_style.GRAY_3),', '("NTU120", 88.9, False, psd_style.S_NTU120),', 1),
], "GA")

# ---- caption 颜色词同步（橙->红 / 青->蓝） ----
tex_patch(LTX / "sections" / "04-experiments.tex",
          "seed pseudo-ground-truth segments (orange; rule-engine seeds",
          "seed pseudo-ground-truth segments (red; rule-engine seeds")
tex_patch(LTX / "sections" / "04-experiments.tex",
          "lower band: SMQ predictions (cyan; end-to-end K=8",
          "lower band: SMQ predictions (blue; end-to-end K=8")
tex_patch(LTX / "sections" / "03-method.tex",
          "Gray boxes are human-in-the-loop steps, orange boxes automated steps.",
          "Gray boxes are human-in-the-loop steps, red boxes automated steps.")
print("ALL R26 PATCHES APPLIED")
