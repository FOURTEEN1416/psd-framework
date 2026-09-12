# -*- coding: utf-8 -*-
"""fig5 v8 — 预算-保留率跨层汇总图 · 正典（风格方向 C「分阶色标 + 制图精度」+ 尺度修复）。

承 v8 前身 v7c（方向 C 提案样板，路径 reports/figB-proposal/fig5_v7c.*，留档不删）：
本文件=方向 C 通过后的**正典生产版**，覆盖 docs/paper/figures/fig5_budget_retention.*。

━━ 数据零改动（逐位承 v6 / v7c）━━
JSON 直读、100*mean/ref 口径、dodge 后 x 位移 13.9/5.5/11.5/10.0/16.5/8.8/13.2/8.0、
PanAf 前 10 种子决策口径 —— 一行未改，不新增任何统计量。

━━ 尺度修复（wt/figB 诊断出的头号根因）━━
v6 用 `bbox_inches="tight"` 保存 → PDF MediaBox = **墨迹框**（220.21×177.29pt）而非版面框，
LaTeX 按 `\\linewidth` 装入时被额外放大 246.24/220.21 = **1.118 倍**，导致：
  ① 同一声明字号在六图间实际渲染值相差最多 11.8%（fig2 5.22pt vs fig5 5.81pt）；
  ② 版式高度失控。
本版改为**固定画布**：figsize 宽恒 3.42in=246.24pt（=版心宽），MediaBox 严格等于画布，
装入时不发生任何缩放（系数 1.000），字号所见即所得。
高度取 2.75in ⇒ 版面占用 198.0pt，与 v6 的 246.24×(177.29/220.21)=198.2pt **逐点持平**，
不引起浮动体位置回退。

━━ 方向 C 的制图手艺（本图）━━
1. tonal ring：每点先画同色相浅阶环（向白混合 0.62），再压实心 marker → 点有"厚度"；
2. 引线锚定：仅 2 个远点系列（public-real v1/v2）走正交引线轨，其余 5 系列就近直标；
3. 预注册参考带：y∈[85,90] 极浅强调色横带 + 右端灰字注记（数值 85/90 取自正文既有预注册
   口径 04-experiments.tex "in the pre-registered PARTIAL band (85--90%)"，非新统计量）；
4. 笔触整体细一档；网格极浅（#EFEFEF/0.4，仅 y 向 [0..100] 刻度位）；
5. 字号下限 5.5pt（脚本内无低于 5.5 的 fontsize）。

━━ 门禁 ━━
G1 印刷稳健性 / G2 标签碰撞 / G3 误差棒穿线 / G5 全图排印 / G6 标记落位 / G4 文件级，
另含**本地引线洁净度断言**（引线穿字 / 引线相交 / 引线近 marker / 引线穿误差棒，四者零命中）。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[4]  # worktree 隔离：禁硬编码主仓绝对路径
OUT = ROOT / "docs" / "paper" / "figures"
PREVIEW = ROOT / "reports" / "figB-proposal"
PREVIEW.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import psd_style
psd_style.apply_style()
INK = psd_style.INK
GRID_GRAY = psd_style.GRID_GRAY


def _darken(hex_color, f):
    """同相降L（psd_style 深化纪律）：RGB 乘 f 保色相，供直标字色。"""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(int(r*f), int(g*f), int(b*f))


def _mix_white(hex_color, f):
    """向白混合 f 比例（本脚本局部助手，不写入 psd_style）：派生 tonal ring 浅阶 / 极浅带。"""
    r = int(hex_color[1:3], 16); g = int(hex_color[3:5], 16); b = int(hex_color[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        round(r + (255 - r) * f), round(g + (255 - g) * f), round(b + (255 - b) * f))


def j(name):
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


# ---- 各层点: (label_fraction_pct, retention_pct, err_pp, tier, marker, color) —— 逐位承 v6 ----
r16 = j("r16-endtoend-pseudo-2026-09-05.json")
r16ntu = j("r16-ntu-pseudo-10seed-2026-09-07.json")
p07 = j("p07-endtoend-ak-full12-2026-09-04.json")   # supervised full reference only
p12 = j("p12-akv2-replication-2026-09-04.json")     # supervised full reference only
p14 = j("p14-ntu-lowres-2026-09-04.json")           # supervised (c) reference only
w23 = j("p05-al-efficiency-warmstart-short-2026-08-25.json")

pts = []
v1_full = p07["agg"]["warm_spc-1"]["top1_mean"]
v1 = r16["summary"]["v1"]["warm_spc2"]
pts.append((13.9, 100 * v1["top1_mean"] / v1_full, 100 * v1["top1_std"] / v1_full,
            "public-real v1", "o", psd_style.S_PUBV1))
v2_full = p12["summary"]["warm_spc-1"]["top1_mean"]
for spc, n_anc, x_disp in ((2, 14, 5.5), (4, 28, 11.5)):
    s = r16["summary"]["v2"][f"warm_spc{spc}"]
    pts.append((x_disp, 100 * s["top1_mean"] / v2_full, 100 * s["top1_std"] / v2_full,
                "public-real v2", "s", psd_style.S_PUBV2))
tb = [r["top1"] for r in r16ntu["arms"]["b_selftrain_10pct"]]
c_ref = p14["arms"]["c_full_linear"]["top1"]
ret = 100 * np.mean(tb) / c_ref
err = 100 * np.std(tb, ddof=1) / c_ref
pts.append((10.0, ret, err, "human NTU60", "D", psd_style.S_NTU60))
p5b_ntu = j("p5b-ntu120-retention-2026-09-07.json")
p5b_ucf = j("p5b-ucf101-retention-2026-09-07.json")
p23_panaf = j("p23-panaf-retention-2026-09-07.json")
tb120 = p5b_ntu["b_arms"]
ret120 = 100 * np.mean(tb120) / p5b_ntu["full_ref"]
err120 = 100 * np.std(tb120, ddof=1) / p5b_ntu["full_ref"]
pts.append((16.5, ret120, err120, "human NTU120", "v", psd_style.S_NTU120))
tbucf = p5b_ucf["b_arms"]
retucf = 100 * np.mean(tbucf) / p5b_ucf["full_ref"]
errucf = 100 * np.std(tbucf, ddof=1) / p5b_ucf["full_ref"]
pts.append((8.8, retucf, errucf, "indep. UCF101", "P", psd_style.S_UCF))
tbpan = p23_panaf["b_arms"][:10]  # 决策口径(E9d/tab2/caption)=前 10 种子 42-51; 后 20 为诊断扩展
retpan = 100 * np.mean(tbpan) / p23_panaf["full_ref"]
errpan = 100 * np.std(tbpan, ddof=1) / p23_panaf["full_ref"]
pts.append((13.2, retpan, errpan, "animal PanAf500", "X", psd_style.S_PANAF))
syn_full = w23["curves"]["random"]["200"]["mean"]
syn = w23["curves"]["random"]["20"]
pts.append((8.0, 100 * syn["mean"] / syn_full, 100 * syn["std"] / syn_full,
            "synthetic-offset", "^", psd_style.S_SYNTH))

# ---- 固定画布：宽 3.42in == 版心 246.24pt；高 2.75in ⇒ 版面占用与 v6 持平 ----
FIG_W, FIG_H = 3.42, 2.75
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white"); ax.set_facecolor("white")
# 去 tight 后由本行显式定版心（保证 ylabel/ticklabel/xlabel 全落在画布内且四周留白均匀）
fig.subplots_adjust(left=0.113, right=0.982, bottom=0.135, top=0.985)

# ---- 信息层：预注册 PARTIAL 判定带 85-90%（数值 85/90 来自正文既有预注册口径，
#      见 04-experiments.tex 原文 "88.9% retention, in the pre-registered PARTIAL band
#      (85--90%)"；此为参考区间，非新统计量，硬编码并注明出处） ----
ax.axhspan(85, 90, color=_mix_white(psd_style.OCEAN_EMBER, 0.90),
           linewidth=0, zorder=0.6)

# ---- 点：先 tonal ring（同色相浅阶），再实心 marker ---- （zorder 2.5 / 3，照 v6 口径） ----
RING_F = 0.62
MARKER_SIZE = 6.0
RING_SIZE = 8.6
point_errs = []        # (x, y_low, y_high, tier) 供 G3 / 本地引线自检
for frac, r, e, tier, mk, col in pts:
    light = _mix_white(col, RING_F)
    ax.plot(frac, r, marker=mk, markersize=RING_SIZE, markerfacecolor=light,
            markeredgecolor="none", linestyle="none", zorder=2.5)
    edge = psd_style.S_SYNTH_EDGE if col == psd_style.S_SYNTH else "white"
    ax.errorbar(frac, r, yerr=e, marker=mk, color=col, markersize=MARKER_SIZE,
                ls="none", markerfacecolor=col, markeredgecolor=edge,
                markeredgewidth=0.9, elinewidth=0.8, capsize=2.4, capthick=0.8,
                zorder=3)
    point_errs.append((frac, r - e, r + e, tier))

# ---- 轴 / 刻度 / 笔触（逐位承 v6 口径，字号下限 5.5；置于标签之前，transform 就绪） ----
ax.set_xscale("log")
ax.set_xlim(4, 30)
ax.set_xticks([5, 10, 20])
ax.set_xticklabels(["5", "10", "20"])
ax.minorticks_off()
from matplotlib.ticker import NullFormatter
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(0, 108)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.set_xlabel("Annotation budget (% of the tier's full-labeled pool, log)", fontsize=6.8)
ax.set_ylabel("Retention of own full-budget top-1 (%)", fontsize=6.8)
ax.grid(True, axis="y", color="#EFEFEF", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_linewidth(0.6)
ax.tick_params(labelsize=5.6, width=0.6, length=2.2)

# ---- 标签系统（方向 C 内核：仅 v1/v2 走引线轨，其余 5 系列就近直标） ----
DIRECT = {
    "human NTU60":     ("human NTU60",     10.0, 94.5,  "center", "bottom", _darken(psd_style.S_NTU60, 0.6)),
    "human NTU120":    ("human NTU120",    17.3, 91.0,  "left",   "center", _darken(psd_style.S_NTU120, 0.45)),
    "animal PanAf500": ("animal PanAf500", 14.3, 80.0,  "left",   "center", _darken(psd_style.S_PANAF, 0.5)),
    "synthetic-offset":("synthetic-offset", 8.0, 78.0,  "center", "top",    _darken(psd_style.S_SYNTH, 0.55)),
    "indep. UCF101":   ("indep. UCF101",    9.6, 66.63, "left",   "center", _darken(psd_style.S_UCF, 0.5)),
}
LEADER_SPEC = {
    "public-real v1": dict(text="public-real v1", x=21.0, y=28.94, ha="left", va="center",
                          tcol=_darken(psd_style.S_PUBV1, 0.6),
                          path=[(13.9, 28.94), (21.0, 28.94)]),
    "public-real v2": dict(text="public-real v2", x=4.8, y=15.0, ha="left", va="center",
                          tcol=_darken(psd_style.S_PUBV2, 0.6),
                          path=[(5.5, 35.01), (4.6, 35.01), (4.6, 15.0)]),
}

label_texts = []
leaders = []   # 本地断言用：name / data_path / 所连 text 对象
for tier, (text, x, y, ha, va, tcol) in DIRECT.items():
    t = ax.text(x, y, text, fontsize=5.6, color=tcol, ha=ha, va=va,
                fontweight="normal", zorder=4)
    label_texts.append(t)
for tier, spec in LEADER_SPEC.items():
    ax.plot([p[0] for p in spec["path"]], [p[1] for p in spec["path"]],
            color="#BFBFBF", linewidth=0.5, solid_capstyle="butt", zorder=2)
    t = ax.text(spec["x"], spec["y"], spec["text"], fontsize=5.6, color=spec["tcol"],
                ha=spec["ha"], va=spec["va"], fontweight="normal", zorder=4)
    label_texts.append(t)
    leaders.append({"name": tier, "data_path": spec["path"], "text": t})

# ---- 100% 基线 + 注记（注记置于基线右端**下方**，va="top" 使文字整体在 100 线之下，
#      避免基线从字中穿过——v7c 原坐标 (29.6,98.5) 配 va="bottom" 会让线切字）----
ax.axhline(100, color=GRID_GRAY, linewidth=0.9, zorder=1)
t_own = ax.text(29.6, 99.4, "own full budget", fontsize=5.5, color="#767676",
                ha="right", va="top", zorder=4)
label_texts.append(t_own)

# ---- 预注册带注记（5.5pt 灰字，**左端内嵌两行**）----
# 几何实测：x=4.40 起，左侧可用窗口止于 x=8 处 synthetic-offset 的 marker 外缘
# （marker 6pt / tonal ring 8.6pt ⇒ 环左缘在数据 x=7.68），文字宽上限 0.80in。
# 故拆两行：0.479in / 0.784in ⇒ 文字块 x∈[4.40,7.49]、y∈[84.4,90.6]，
# 与环留 3.7px(≈2.7pt) 净空、与 v2 点列（y≤52.6）无重叠；同时消解右上角
# human NTU120 / animal PanAf500 / 带注记 三点拥挤，把空置的左上区变成有归属的注记。
t_band = ax.text(4.40, 87.5, "pre-registered\nPARTIAL band 85-90%", fontsize=5.5,
                 color="#7F7F7F", ha="left", va="center", zorder=4, linespacing=1.25)
label_texts.append(t_band)


# ---- 本地断言：引线系统洁净度（不进共享模块） ----
def _ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])


def _seg_seg(p1, p2, p3, p4):
    d1, d2 = _ccw(p3, p4, p1), _ccw(p3, p4, p2)
    d3, d4 = _ccw(p1, p2, p3), _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _seg_rect_hit(p1, p2, rect):
    if (rect.x0 <= p1[0] <= rect.x1 and rect.y0 <= p1[1] <= rect.y1) or \
       (rect.x0 <= p2[0] <= rect.x1 and rect.y0 <= p2[1] <= rect.y1):
        return True
    for a, b in [((rect.x0, rect.y0), (rect.x1, rect.y0)),
                 ((rect.x1, rect.y0), (rect.x1, rect.y1)),
                 ((rect.x1, rect.y1), (rect.x0, rect.y1)),
                 ((rect.x0, rect.y1), (rect.x0, rect.y0))]:
        if _seg_seg(p1, p2, a, b):
            return True
    return False


def _assert_leaders_clean(fig, ax, leaders, point_errs):
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    dpi = fig.dpi
    r_px = 6.0 / 72.0 * dpi / 2.0 * 1.15   # 实心 marker 像素半径 + 裕量
    texts = [t for t in fig.findobj(Text)
             if t.get_visible() and t.get_text().strip()]
    boxes = {id(t): t.get_window_extent(ren) for t in texts}
    for L in leaders:
        L["disp"] = [ax.transData.transform(p) for p in L["data_path"]]
    n_text = n_lead = n_mk = n_err = 0
    for L in leaders:
        dp = L["disp"]; own = L["text"]
        for i in range(len(dp) - 1):
            s, e = dp[i], dp[i + 1]
            for t in texts:
                if id(t) == id(own):
                    continue
                if _seg_rect_hit(s, e, boxes[id(t)]):
                    n_text += 1
                    print(f"  LEADER-CROSS-TEXT: {L['name']} seg{i} x {t.get_text()[:24]!r}")
            for K in leaders:
                if K is L:
                    continue
                for a in range(len(dp) - 1):
                    for b in range(len(K["disp"]) - 1):
                        if _seg_seg(dp[a], dp[a + 1], K["disp"][b], K["disp"][b + 1]):
                            n_lead += 1
                            print(f"  LEADER-CROSS-LEADER: {L['name']} x {K['name']}")
        tgt = L["data_path"][0]
        for (x, y, e, _) in point_errs:
            if abs(x - tgt[0]) < 1e-6 and abs(y - tgt[1]) < 1e-6:
                continue   # 自己的 marker，跳过
            mc = ax.transData.transform((x, y))
            if min(np.linalg.norm(np.array(dp[i]) - mc) for i in range(len(dp))) < r_px:
                n_mk += 1
                print(f"  LEADER-NEAR-MARKER: {L['name']} x marker({x},{y})")
        for (x, lo, hi, _) in point_errs:
            if abs(x - tgt[0]) < 1e-6:
                continue
            p0 = ax.transData.transform((x, lo)); p1 = ax.transData.transform((x, hi))
            for i in range(len(dp) - 1):
                lx0, lx1 = dp[i][0], dp[i + 1][0]
                if min(lx0, lx1) - 2 <= p0[0] <= max(lx0, lx1) + 2:
                    ly0, ly1 = dp[i][1], dp[i + 1][1]
                    sy = (min(ly0, ly1), max(ly0, ly1))
                    ey = (min(p0[1], p1[1]), max(p0[1], p1[1]))
                    if not (sy[1] < ey[0] - 3 or sy[0] > ey[1] + 3):
                        n_err += 1
                        print(f"  LEADER-CROSS-ERRBAR: {L['name']} x errbar({x})")
    total = n_text + n_lead + n_mk + n_err
    print(f"[LOCAL leader-clean] text={n_text} leaderXleader={n_lead} "
          f"nearMarker={n_mk} crossErrbar={n_err} -> "
          f"{'PASS' if total == 0 else 'FAIL'}")
    if total != 0:
        raise SystemExit("leader-clean assertion FAILED")


# ---- 出图门禁（共享模块 make_common_gates，只读调用，绝不改动） ----
import make_common_gates as gates
gates.gate_print_robustness(
    "fig5-v8",
    {"public-real v1": psd_style.S_PUBV1, "public-real v2": psd_style.S_PUBV2,
     "human NTU60": psd_style.S_NTU60, "human NTU120": psd_style.S_NTU120,
     "indep. UCF101": psd_style.S_UCF, "animal PanAf500": psd_style.S_PANAF,
     "synthetic-offset": psd_style.S_SYNTH},
    redundancy="7 unique marker shapes + direct labels + 2 leader lines")
gates.gate_labels(fig, ax, label_texts, name="fig5-v8")
gates.gate_whisker_texts(fig, ax, label_texts, point_errs, name="fig5-v8")
_assert_leaders_clean(fig, ax, leaders, point_errs)
gates.gate_typography(fig, min_pt=5.5, name="fig5-v8")
gates.gate_marks_in_axes(fig, name="fig5-v8")

pdf_path = OUT / "fig5_budget_retention.pdf"
png_path = OUT / "fig5_budget_retention.png"
# 尺度修复：不用 bbox_inches="tight"，MediaBox 严格等于 figsize（3.42in = 246.24pt）
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=600)
fig.savefig(PREVIEW / "preview_fig5.png", dpi=150)
gates.gate_pdf(pdf_path, max_w_pt=246.3)

for frac, r, e, tier, _, _ in pts:
    print(f"{tier}: {frac:.1f}% -> {r:.1f}% ± {e:.1f}")
print("fig5 v8 (direction C, fixed-canvas scale fix) saved")
