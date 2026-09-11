# -*- coding: utf-8 -*-
r"""make_icon_glyphs.py — 模块微图标共享库（任务包 A 第三轮 · 双栏版配套）。

双栏 7in 版面解锁卡片内微图标（单栏 3.42in 下卡片仅 0.6in 宽会退化为噪点，
fig-redraw-2026-09-11 §七遗留项；本轮由 2col 资产解锁）。

全部图标为描边极简风格（stroke-only, lw≈0.7），色调用模块家族深化色
（psd_style DEEP 系），不引入新色板；每个 glyph 以 (cx, cy) 为中心、s 为
外接尺寸绘制，zorder=6（卡片 z=3-5 之上）。
"""
import numpy as np
from matplotlib.patches import Circle, Arc, FancyArrowPatch, Polygon

Z = 6


def glyph_layers(ax, cx, cy, s, color, lw=0.7):
    """三层堆叠横条（SSL 预训练 / 网络层）。"""
    for i, dy in enumerate((0.30, 0.0, -0.30)):
        w = s * (1.0 - 0.18 * abs(i - 1))
        ax.plot([cx - w / 2, cx + w / 2], [cy + s * dy, cy + s * dy],
                color=color, lw=lw, solid_capstyle="round", zorder=Z)


def glyph_grid_dots(ax, cx, cy, s, color, lw=0.7):
    """3×3 点阵（运动词量化 / 码本）。"""
    off = s * 0.32
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            ax.add_patch(Circle((cx + i * off, cy + j * off), s * 0.055,
                                facecolor=color, edgecolor="none", zorder=Z))


def glyph_wave(ax, cx, cy, s, color, lw=0.7):
    """正弦波（动态嵌入 / 流）。"""
    xs = np.linspace(cx - s / 2, cx + s / 2, 36)
    ax.plot(xs, cy + s * 0.28 * np.sin((xs - cx) / s * 2 * np.pi * 1.6),
            color=color, lw=lw, solid_capstyle="round", zorder=Z)


def glyph_brackets(ax, cx, cy, s, color, lw=0.7):
    """括号夹点（行为提案区间 [· ·]）。"""
    h = s * 0.42
    for side in (-1, 1):
        bx = cx + side * s * 0.42
        ax.plot([bx - side * s * 0.10, bx, bx, bx - side * s * 0.10],
                [cy + h * 0.75, cy + h * 0.75, cy - h * 0.75, cy - h * 0.75],
                color=color, lw=lw, solid_capstyle="round", zorder=Z)
    for i in (-1, 1):
        ax.add_patch(Circle((cx + i * s * 0.12, cy), s * 0.05,
                            facecolor=color, edgecolor="none", zorder=Z))


def glyph_stack(ax, cx, cy, s, color, lw=0.7):
    """错位叠放的矩形帧（视频流 / unlabeled streams）。"""
    for i in range(3):
        dx = (i - 1) * s * 0.14
        dy = (1 - i) * s * 0.16
        ax.plot([cx - s * 0.38 + dx, cx + s * 0.38 + dx, cx + s * 0.38 + dx,
                 cx - s * 0.38 + dx, cx - s * 0.38 + dx],
                [cy - s * 0.22 + dy, cy - s * 0.22 + dy, cy + s * 0.22 + dy,
                 cy + s * 0.22 + dy, cy - s * 0.22 + dy],
                color=color, lw=lw, zorder=Z)


def glyph_doc_lines(ax, cx, cy, s, color, lw=0.7):
    """文档行（规则引擎种子 / 规则文本）。"""
    w, h = s * 0.62, s * 0.78
    ax.plot([cx - w / 2, cx + w / 2, cx + w / 2, cx - w / 2, cx - w / 2],
            [cy - h / 2, cy - h / 2, cy + h / 2, cy + h / 2, cy - h / 2],
            color=color, lw=lw, zorder=Z)
    for k, yy in enumerate((0.24, 0.02, -0.20)):
        ax.plot([cx - w * 0.28, cx + w * (0.28 - 0.16 * k)], [cy + s * yy] * 2,
                color=color, lw=lw * 0.9, solid_capstyle="round", zorder=Z)


def glyph_anchor_pin(ax, cx, cy, s, color, lw=0.7):
    """锚点定位针（anchor learning）。"""
    ax.add_patch(Circle((cx, cy + s * 0.10), s * 0.26, facecolor="none",
                        edgecolor=color, lw=lw, zorder=Z))
    ax.plot([cx, cx], [cy - s * 0.10, cy - s * 0.42], color=color, lw=lw,
            solid_capstyle="round", zorder=Z)
    ax.add_patch(Circle((cx, cy + s * 0.10), s * 0.06, facecolor=color,
                        edgecolor="none", zorder=Z))


def glyph_cluster(ax, cx, cy, s, color, lw=0.7):
    """椭圆圈住三点之二（原型聚类）。"""
    from matplotlib.patches import Ellipse
    pts = [(cx - s * 0.22, cy + s * 0.06), (cx + s * 0.06, cy + s * 0.22),
           (cx + s * 0.26, cy - s * 0.16)]
    for px, py in pts:
        ax.add_patch(Circle((px, py), s * 0.065, facecolor=color,
                            edgecolor="none", zorder=Z + 1))
    ax.add_patch(Ellipse((cx - s * 0.07, cy + s * 0.13), s * 0.72, s * 0.52,
                         angle=18, facecolor="none", edgecolor=color,
                         lw=lw, zorder=Z))


def glyph_circular_arrow(ax, cx, cy, s, color, lw=0.7):
    """单环自转箭头（自训练 / 循环更新）。"""
    ax.add_patch(Arc((cx, cy), s * 0.82, s * 0.82, theta1=40, theta2=320,
                     color=color, lw=lw, zorder=Z))
    th = np.deg2rad(40)
    tipx, tipy = cx + s * 0.41 * np.cos(th), cy + s * 0.41 * np.sin(th)
    ax.add_patch(Polygon([(tipx, tipy), (tipx - s * 0.13, tipy + s * 0.02),
                          (tipx - s * 0.04, tipy - s * 0.13)],
                         closed=True, facecolor=color, edgecolor="none",
                         zorder=Z))


def glyph_tag(ax, cx, cy, s, color, lw=0.7):
    """标签牌（分类 / 指派）。"""
    w, h = s * 0.66, s * 0.44
    ax.plot([cx - w / 2, cx + w * 0.18, cx + w / 2, cx + w / 2, cx - w / 2,
             cx - w / 2],
            [cy - h / 2, cy - h / 2, cy, cy + h / 2, cy + h / 2, cy - h / 2],
            color=color, lw=lw, zorder=Z)
    ax.add_patch(Circle((cx - w * 0.26, cy), s * 0.05, facecolor=color,
                        edgecolor="none", zorder=Z))


def glyph_check(ax, cx, cy, s, color, lw=0.9):
    """对勾（人工核验通过）。"""
    ax.plot([cx - s * 0.34, cx - s * 0.06, cx + s * 0.38],
            [cy - s * 0.02, cy - s * 0.30, cy + s * 0.32],
            color=color, lw=lw, solid_capstyle="round", zorder=Z)


def glyph_person(ax, cx, cy, s, color, lw=0.7):
    """人形（human-in-the-loop）。"""
    ax.add_patch(Circle((cx, cy + s * 0.26), s * 0.14, facecolor="none",
                        edgecolor=color, lw=lw, zorder=Z))
    ax.plot([cx, cx], [cy + s * 0.08, cy - s * 0.24], color=color, lw=lw,
            solid_capstyle="round", zorder=Z)
    ax.plot([cx - s * 0.24, cx + s * 0.24], [cy - s * 0.04] * 2, color=color,
            lw=lw, solid_capstyle="round", zorder=Z)
    ax.plot([cx, cx - s * 0.18], [cy - s * 0.24, cy - s * 0.44], color=color,
            lw=lw, solid_capstyle="round", zorder=Z)
    ax.plot([cx, cx + s * 0.18], [cy - s * 0.24, cy - s * 0.44], color=color,
            lw=lw, solid_capstyle="round", zorder=Z)


def glyph_cylinder(ax, cx, cy, s, color, lw=0.7):
    """数据库圆柱（状态 hub / 共享存储）。"""
    from matplotlib.patches import Ellipse
    w, h = s * 0.62, s * 0.20
    top = cy + s * 0.26
    ax.add_patch(Ellipse((cx, top), w, h, facecolor="none", edgecolor=color,
                         lw=lw, zorder=Z + 1))
    ax.plot([cx - w / 2, cx - w / 2], [top, cy - s * 0.26], color=color,
            lw=lw, zorder=Z)
    ax.plot([cx + w / 2, cx + w / 2], [top, cy - s * 0.26], color=color,
            lw=lw, zorder=Z)
    ax.add_patch(Arc((cx, cy - s * 0.26), w, h, theta1=180, theta2=360,
                     color=color, lw=lw, zorder=Z))


def glyph_arrows_two(ax, cx, cy, s, color, lw=0.7):
    """双向对置箭头（重新估计 / 迭代）。"""
    for i, (y0, d) in enumerate(((0.14, 1), (-0.14, -1))):
        ax.plot([cx - s * 0.34 * d, cx + s * 0.34 * d], [cy + s * y0] * 2,
                color=color, lw=lw, solid_capstyle="round", zorder=Z)
        tipx = cx + s * 0.34 * d
        ax.add_patch(Polygon([(tipx, cy + s * y0),
                              (tipx - s * 0.12 * d, cy + s * y0 + s * 0.07),
                              (tipx - s * 0.12 * d, cy + s * y0 - s * 0.07)],
                             closed=True, facecolor=color, edgecolor="none",
                             zorder=Z))
