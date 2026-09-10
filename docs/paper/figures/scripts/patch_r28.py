# -*- coding: utf-8 -*-
r"""R28 补丁：一切颜色直取 NS 九色板（用户令），人工盒=兰紫、fig4 基线=蓝、fig3 参照=中性灰。"""
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

# ---- fig2 v9 -> v10: 人工盒/图例 兰紫化 ----
patch("make_fig2_pseudo_label_loop_v9.py", "make_fig2_pseudo_label_loop_v10.py", [
    ('GRAY_EDGE = psd_style.GRAY_EDGE\nGRAY_FILL = psd_style.GRAY_FILL',
     'GRAY_EDGE = psd_style.HUMAN_EDGE\nGRAY_FILL = psd_style.HUMAN_FILL', 1),
    ('r"""fig2 v9 — 伪标签循环双行管线（R27 现代扁平重设计：白卡片+软阴影；拓扑承 v8）。',
     'r"""fig2 v10 — 伪标签循环双行管线（R28: 人工环节兰紫化, 一切颜色直取 NS 九色板）。', 1),
], "fig2")

# ---- fig3 v5 -> v6: null 参照带中性灰 ----
patch("make_fig3_segmentation_qualitative_v5.py", "make_fig3_segmentation_qualitative_v6.py", [
    ('BAND_GRAY, BASE_EDGE = "#D7DCE2", "#AEB6BF"   # 基线带蓝灰调（R26 与 NS 板同调）',
     'BAND_GRAY, BASE_EDGE = psd_style.GRAY_FILL, psd_style.GRAY_EDGE   # R28: null 参照=中性灰', 1),
    ('NOTE_GRAY, GRID_GRAY = "#7F8C8D", psd_style.GRID_GRAY',
     'NOTE_GRAY, GRID_GRAY = "#666666", psd_style.GRID_GRAY', 1),
], "fig3")

# ---- fig4 v3 -> v4: random 基线=蓝（entropy 红 vs random 蓝, 全板内） ----
patch("make_fig4_al_efficiency_v3.py", "make_fig4_al_efficiency_v4.py", [
    ('RND_COLOR = psd_style.GRAY_2   # random selection = neutral baseline',
     'RND_COLOR = psd_style.NS_BLUE  # random selection = NS 蓝（R28: 板内取色, entropy红 vs random蓝）', 1),
    ('NOTE_GRAY = "#888888"',
     'NOTE_GRAY = "#666666"', 1),
], "fig4")

# ---- caption 颜色词同步: fig2 人工盒 gray->orchid ----
tex_patch(LTX / "sections" / "03-method.tex",
          "Gray boxes are human-in-the-loop steps, red boxes automated steps.",
          "Orchid boxes are human-in-the-loop steps, red boxes automated steps.")
print("ALL R28 PATCHES APPLIED")
