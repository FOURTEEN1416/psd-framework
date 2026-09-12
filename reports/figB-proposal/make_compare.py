# -*- coding: utf-8 -*-
r"""make_compare.py — figB 提案对比图生成（wt/figB，2026-09-12）。

产出两张拼版（只写 reports/figB-proposal/，只读正典 PNG，零写入正典）：
  1. fig5 四联对比：在位正典 v6 + 三方向样板 v7a/v7b/v7c
  2. 诊断证据图：四处缺陷的 600dpi 原生裁剪（fig2 / fig3 / GA / fig5）

用法: D:\Desktop\psd-framework\.venv\Scripts\python.exe make_compare.py
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]          # worktree 根
CANON = ROOT / "docs" / "paper" / "figures"
PROP = ROOT / "reports" / "figB-proposal"

FONT_R = Path(r"C:\Windows\Fonts\msyh.ttc")      # 微软雅黑（含 CJK 字形）
FONT_B = Path(r"C:\Windows\Fonts\msyhbd.ttc")

INK = (26, 26, 26)
SUB = (85, 85, 85)
BG = (255, 255, 255)
BAR = (244, 246, 248)
RULE = (222, 226, 230)


def font(path, size):
    try:
        return ImageFont.truetype(str(path), size)
    except Exception:
        return ImageFont.load_default()


def fit(im, w):
    """等比缩放到宽 w。"""
    h = round(im.height * w / im.width)
    return im.resize((w, h), Image.LANCZOS)


def panel(im, title, note, w, title_h=76):
    """给一张图套上标题条，返回统一宽度的卡片。"""
    body = fit(im, w)
    card = Image.new("RGB", (w, title_h + body.height), BG)
    d = ImageDraw.Draw(card)
    d.rectangle([0, 0, w - 1, title_h - 1], fill=BAR)
    d.line([0, title_h - 1, w, title_h - 1], fill=RULE, width=2)
    d.text((14, 12), title, font=font(FONT_B, 26), fill=INK)
    d.text((14, 45), note, font=font(FONT_R, 19), fill=SUB)
    card.paste(body, (0, title_h))
    return card


# ─────────────────────────── 1. fig5 四联对比 ───────────────────────────
CARD_W = 900
items = [
    (CANON / "fig5_budget_retention.png", "在位正典  fig5 v6",
     "单面板 · 7 色彩字直标 · 下部三分之一空置"),
    (PROP / "fig5_v7a.png", "方向 A  v7a · 编辑极简",
     "灰阶骨架 + 单一暖强调 · 全墨注记 + 色键点 · 去网格"),
    (PROP / "fig5_v7b.png", "方向 B  v7b · 高密度组合面板",
     "按 caption 分组拆双面板 sharex · 死区清零 · 误差棒不截断"),
    (PROP / "fig5_v7c.png", "方向 C  v7c · 分阶色标 + 引线注记",
     "tonal ring 分层环 + 预注册带 85–90% + 正交引线轨"),
]
cards = [panel(Image.open(p).convert("RGB"), t, n, CARD_W) for p, t, n in items]
row_h = max(c.height for c in cards) + 24
pad, gap = 26, 26
W = pad * 2 + CARD_W * 2 + gap
H = pad * 2 + row_h * 2 + 64
sheet = Image.new("RGB", (W, H), BG)
d = ImageDraw.Draw(sheet)
d.text((pad, 18), "fig5 预算–保留率散点 · 在位 vs 三方向样板（同一数据口径，仅风格层不同）",
       font=font(FONT_B, 30), fill=INK)
for i, c in enumerate(cards):
    x = pad + (i % 2) * (CARD_W + gap)
    y = 64 + pad + (i // 2) * row_h
    sheet.paste(c, (x, y))
    d.rectangle([x, y, x + c.width - 1, y + c.height - 1], outline=RULE, width=1)
out1 = PROP / "figB-fig5-fourway.png"
sheet.save(out1)
print("written:", out1, sheet.size)

# ─────────────────────────── 2. 诊断证据图 ───────────────────────────
EV = [
    (CANON / "fig2_pseudo_label_loop.png", (0, 113, 900, 660),
     "fig2 · 卡片宽度不随文字长度自适应",
     "「Assign proposals」压出卡片左/右边界；κ≥τ 骑在卡边上且不在箭头上；② 号章压在深色块上对比不足"),
    (CANON / "fig3_segmentation_qualitative.png", (0, 1075, 1947, 1290),
     "fig3 · 轴名归属歧义 + 面板 c 顶部拥挤",
     "「frame index」悬在 b/c 之间，读作 c 的 x 轴名而 c 的 x 是 Episodes；agg 注记与图例并排挤在轴顶"),
    (CANON / "fig_ga_graphical_abstract.png", (600, 420, 3400, 860),
     "GA · 演化回线指向物理层 Φ",
     "虚线箭头末端指进 Φ（冻结层），与「only Ω retrains」语义相反；回线与块内斜体行重复表达同一件事"),
    (CANON / "fig5_budget_retention.png", (0, 300, 1835, 1478),
     "fig5 · 空区 + 彩虹粗体注记",
     "y≈0–25 与右侧大片空置；7 个粗体彩字各自为政；金橙 #ECB66B 字在白底对比不足"),
]
ew = 1280
ecards = []
for src, box, t, n in EV:
    im = Image.open(src).convert("RGB").crop(box)
    c = panel(im, t, n, ew, title_h=88)
    ecards.append(c)
tot_h = pad * 2 + 56 + sum(c.height + 18 for c in ecards)
esheet = Image.new("RGB", (ew + pad * 2, tot_h), BG)
ed = ImageDraw.Draw(esheet)
ed.text((pad, 16), "六图审美诊断 · 600dpi 原生裁剪证据", font=font(FONT_B, 30), fill=INK)
y = 16 + 46
for c in ecards:
    esheet.paste(c, (pad, y))
    ed.rectangle([pad, y, pad + c.width - 1, y + c.height - 1], outline=RULE, width=1)
    y += c.height + 18
out2 = PROP / "figB-diagnosis-evidence.png"
esheet.save(out2)
print("written:", out2, esheet.size)
