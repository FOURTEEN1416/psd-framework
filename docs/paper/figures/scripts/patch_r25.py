# -*- coding: utf-8 -*-
r"""R25 审美/配色/结构升级补丁：六图脚本版本递进 + 色板/字体统一 + 两处 caption-图失配修复。

原则：布局/坐标/数据口径零改动（R24 八轮 judge 成果保留），只动 颜色语义、字体族、
色阶规范。版式承前版本。运行后必须执行渲染并核对 mtime（R24 教训：脚本写了≠跑了）。
"""
from pathlib import Path
import sys

SD = Path(r"D:\Desktop\psd-framework\docs\paper\figures\scripts")
LTX = Path(r"D:\Desktop\psd-framework\docs\paper\latex")

IMP = ('import sys, os\n'
       'sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))\n'
       'import psd_style\n'
       'psd_style.apply_style()\n')

def patch(src, dst, pairs, label):
    t = (SD / src).read_text(encoding="utf-8")
    for old, new, cnt in pairs:
        n = t.count(old)
        assert n == cnt, f"{label}: expect {cnt}, got {n} for {old[:60]!r}"
        t = t.replace(old, new)
    (SD / dst).write_text(t, encoding="utf-8")
    print(f"OK {label} -> {dst}")

def tex_patch(path, old, new, cnt=1):
    t = path.read_text(encoding="utf-8")
    n = t.count(old)
    assert n == cnt, f"{path.name}: expect {cnt}, got {n} for {old[:60]!r}"
    path.write_text(t.replace(old, new), encoding="utf-8")
    print(f"OK tex {path.name}: {old[:48]!r}")

# ---------------- fig1 v6 -> v7 ----------------
patch("make_fig1_overview_v6.py", "make_fig1_overview_v7.py", [
    ('r"""fig1 v6 — PSD 框架总览（R24 图表精修, 2026-09-10）。',
     'r"""fig1 v7 — PSD 框架总览（R25 审美统一：psd_style 单一真源取色 + Arial 字族；版式承 v6）。', 1),
    ('C_PHYS_FILL, C_PHYS_EDGE = "#DAFFFF", "#0E7490"\n'
     'C_SEM_FILL, C_SEM_EDGE = "#FFE3DA", "#C2410C"\n'
     'C_IFACE_FILL, C_IFACE_EDGE = "#EFEFEF", "#9CA3AF"\n'
     'C_TEXT, C_ARROW = "#111111", "#374151"',
     IMP +
     'C_PHYS_FILL, C_PHYS_EDGE = psd_style.PHYS_FILL, psd_style.PHYS_EDGE\n'
     'C_SEM_FILL, C_SEM_EDGE = psd_style.SEM_FILL, psd_style.SEM_EDGE\n'
     'C_IFACE_FILL, C_IFACE_EDGE = psd_style.IFACE_FILL, psd_style.IFACE_EDGE\n'
     'C_TEXT, C_ARROW = psd_style.INK, psd_style.ARROW', 1),
    ('color="#374151", rotation=90', 'color=C_ARROW, rotation=90', 1),
], "fig1")

# ---------------- fig2 v7 -> v8 ----------------
patch("make_fig2_pseudo_label_loop_v7.py", "make_fig2_pseudo_label_loop_v8.py", [
    ('r"""fig2 v7 — 伪标签飞轮: 双行流程图布局(彻底弃圆环, R24 图表专项第四稿)。',
     'r"""fig2 v8 — 伪标签循环双行管线（R25 审美统一：psd_style 取色 + Arial 字族；版式承 v7）。', 1),
    ('PAPER = "#FFFFFF"\n'
     'INK = "#1F2937"\n'
     'ACCENT = "#C2410C"\n'
     'ACCENT_TINT = "#FFE3DA"\n'
     'FOCAL_FILL = "#FFD9C7"\n'
     'GRAY_EDGE = "#6B7280"\n'
     'GRAY_FILL = "#F3F4F6"\n'
     'SOFT = "#9CA3AF"\n'
     'C_TEXT = "#111111"',
     IMP +
     'PAPER = "#FFFFFF"\n'
     'INK = psd_style.INK\n'
     'ACCENT = psd_style.SEM_EDGE\n'
     'ACCENT_TINT = psd_style.SEM_FILL\n'
     'FOCAL_FILL = psd_style.FOCAL_FILL\n'
     'GRAY_EDGE = psd_style.GRAY_EDGE\n'
     'GRAY_FILL = psd_style.GRAY_FILL\n'
     'SOFT = psd_style.GRAY_LINE\n'
     'C_TEXT = psd_style.INK', 1),
    ('color="#4B5563"', 'color=psd_style.NOTE', 4),
], "fig2")

# ---------------- fig3 v3 -> v4 ----------------
patch("make_fig3_segmentation_qualitative_v3.py", "make_fig3_segmentation_qualitative_v4.py", [
    ('INK = "#000000"\n'
     'CYAN_FILL, CYAN_DARK = "#DAFFFF", "#0E7490"\n'
     'ORANGE_FILL, ORANGE_DARK = "#FFE3DA", "#C2410C"',
     IMP +
     'INK = psd_style.INK\n'
     'CYAN_FILL, CYAN_DARK = psd_style.PHYS_FILL, psd_style.PHYS_EDGE\n'
     'ORANGE_FILL, ORANGE_DARK = psd_style.SEM_FILL, psd_style.SEM_EDGE', 1),
    ('plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "text.color": INK})\n',
     '', 1),
    ('BAND_GRAY, BASE_EDGE = "#DADADA", "#AAAAAA"',
     'BAND_GRAY, BASE_EDGE = psd_style.GRID_GRAY, "#AAAAAA"', 1),
    ('NOTE_GRAY, GRID_GRAY = "#888888", "#DADADA"',
     'NOTE_GRAY, GRID_GRAY = "#888888", psd_style.GRID_GRAY', 1),
], "fig3")

# ---------------- fig4 v2 -> v3 (语义重组: entropy=hero 蓝, random=中性灰) ----------------
patch("make_fig4_al_efficiency_v2.py", "make_fig4_al_efficiency_v3.py", [
    ('INK = "#000000"\n'
     'CYAN_DARK = "#0E7490"\n'
     'ORANGE_DARK = "#C2410C"\n'
     'NOTE_GRAY = "#888888"\n'
     'GRID_GRAY = "#DADADA"',
     IMP +
     'INK = psd_style.INK\n'
     'ENT_COLOR = psd_style.HERO     # uncertainty sampling = proposed arm (Nature: blue=hero)\n'
     'RND_COLOR = psd_style.GRAY_2   # random selection = neutral baseline\n'
     'NOTE_GRAY = "#888888"\n'
     'GRID_GRAY = psd_style.GRID_GRAY', 1),
    ('plt.rcParams.update({\n'
     '    "pdf.fonttype": 42, "font.family": "DejaVu Sans",\n'
     '    "text.color": INK, "axes.edgecolor": INK, "axes.labelcolor": INK,\n'
     '    "xtick.color": INK, "ytick.color": INK,\n'
     '})\n', '', 1),
    ('"entropy": dict(color=CYAN_DARK, marker="o", ls="solid",\n'
     '                    mfc=CYAN_DARK, label="Uncertainty sampling (softmax entropy)"),\n'
     '    "random": dict(color=ORANGE_DARK, marker="s", ls=(0, (5, 2.5)),',
     '"entropy": dict(color=ENT_COLOR, marker="o", ls="solid",\n'
     '                    mfc=ENT_COLOR, label="Uncertainty sampling (softmax entropy)"),\n'
     '    "random": dict(color=RND_COLOR, marker="s", ls=(0, (5, 2.5)),', 1),
], "fig4")

# ---------------- fig5 v2 -> v3 (七系列: 灰阶单调化 + 紫归队 + token 化) ----------------
patch("make_fig5_budget_retention_v2.py", "make_fig5_budget_retention_v3.py", [
    ('INK = "#111111"\nGRID_GRAY = "#DADADA"',
     IMP + 'INK = psd_style.INK\nGRID_GRAY = psd_style.GRID_GRAY', 1),
    ('plt.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans", "text.color": INK})\n',
     '', 1),
    ('"public-real v1", "o", "#0E7490"', '"public-real v1", "o", psd_style.PHYS_EDGE', 1),
    ('"public-real v2", "s", "#C2410C"', '"public-real v2", "s", psd_style.SEM_EDGE', 1),
    ('"human benchmark (NTU60)", "D", "#6B7280"', '"human benchmark (NTU60)", "D", psd_style.GRAY_2', 1),
    ('"human benchmark (NTU120, HRNet 2D)", "v", "#4B5563"',
     '"human benchmark (NTU120, HRNet 2D)", "v", psd_style.GRAY_3', 1),
    ('"independent benchmark (UCF101, HRNet 2D)", "P", "#9CA3AF"',
     '"independent benchmark (UCF101, HRNet 2D)", "P", psd_style.GRAY_1', 1),
    ('"animal public benchmark (PanAf500)", "X", "#1F2937"',
     '"animal public benchmark (PanAf500)", "X", psd_style.GRAY_4', 1),
    ('"synthetic-offset", "^", "#6D28D9"', '"synthetic-offset", "^", psd_style.VIOLET', 1),
], "fig5")

# ---------------- GA v2 -> v3 ----------------
patch("make_ga_graphical_abstract_v2.py", "make_ga_graphical_abstract_v3.py", [
    ('C_PHYS_FILL, C_PHYS_EDGE = "#DAFFFF", "#0E7490"\n'
     'C_SEM_FILL, C_SEM_EDGE = "#FFE3DA", "#C2410C"\n'
     'C_TEXT, C_ARROW, C_NOTE = "#111111", "#374151", "#888888"\n'
     'C_BAR_HUMAN = "#6B7280"      # R24: 与 fig5 NTU60 同灰系(青专属 Φ 层语义)',
     IMP +
     'C_PHYS_FILL, C_PHYS_EDGE = psd_style.PHYS_FILL, psd_style.PHYS_EDGE\n'
     'C_SEM_FILL, C_SEM_EDGE = psd_style.SEM_FILL, psd_style.SEM_EDGE\n'
     'C_TEXT, C_ARROW, C_NOTE = psd_style.INK, psd_style.ARROW, "#888888"', 1),
    ('plt.rcParams.update({\n'
     '    "pdf.fonttype": 42,\n'
     '    "font.family": "DejaVu Sans",\n'
     '    "text.color": C_TEXT,\n'
     '    "mathtext.fontset": "dejavusans",\n'
     '})',
     'plt.rcParams["mathtext.fontset"] = "dejavusans"  # mathtext 与 Arial 最兼容的字形集', 1),
    ('("NTU60", 90.7, False, "#6B7280"),', '("NTU60", 90.7, False, psd_style.GRAY_2),', 1),
    ('("NTU120", 88.9, False, "#4B5563"),', '("NTU120", 88.9, False, psd_style.GRAY_3),', 1),
    ('("canine", 28.9, True, "#F3F4F6"),   # E7 v1 @13%-label 预算: 保留率 28.9%, 绝对精度近随机',
     '("canine", 28.9, True, psd_style.GRAY_FILL),   # E7 v1 @13%-label 预算: 保留率 28.9%, 绝对精度近随机', 1),
], "GA")

# ---------------- caption 修复两处（图已变、文字残留） ----------------
tex_patch(
    LTX / "sections" / "03-method.tex",
    "The anchor-guided pseudo-labeling loop as a two-ring flywheel sharing the Assign node.",
    "The anchor-guided pseudo-labeling loop as a two-row pipeline around a shared state hub.",
)
tex_patch(
    LTX / "sections" / "05-ablation-analysis.tex",
    "; arms dodge $\\pm$4.5\\% at shared budgets",
    "",
)
print("ALL PATCHES APPLIED")
