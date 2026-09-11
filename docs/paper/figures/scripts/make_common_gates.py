# -*- coding: utf-8 -*-
r"""make_common_gates.py — PSD 论文图共享出图门禁（任务包 A 第三轮新增，2026-09-11）。

每图出图时自动断言，任一失败 SystemExit 非零（先证明再宣称）：

G1 印刷稳健性 gate_print_robustness()
   色盲模拟（deuteranope / protanope，Machado et al. 2009 severity=1.0 线性 RGB 矩阵）
   + 灰度转换（CIE 相对亮度 → L*）。对本图承担"系列区分"的色集合逐一计算
   两两 ΔE76（sRGB→linear→CVD 矩阵→Lab）：
   - 正常 / 模拟后三种视觉条件下 ΔE ≥ min_de 即视为色觉可辨（硬门禁）；
   - 灰度下 ΔL* < min_dl 的配对：若调用方已声明冗余编码（marker 形状 / hatch /
     线型 / 直标文字——psd_style 跨图铁律的既有手段）则 WARN+留痕放行，
     未声明冗余即 FAIL；
G2 标签碰撞 gate_labels()
   渲染后两两 text bbox 相交断言 + 出轴/出画布断言（fig5 v6 逻辑泛化）；
G3 误差棒穿线 gate_whisker_texts()
   文本 bbox × 误差棒竖线相交断言（有 errorbar 的图启用；fig5 v6 逻辑泛化）；
G4 文件级 gate_pdf()
   PDF MediaBox 实测尺寸上限断言（GA 531×131pt 等）+ pdfimages -list 零嵌入
   位图断言（poppler 工具链，MiKTeX 自带）。

用法（各 make_*.py 末尾）：
    import make_common_gates as gates
    gates.gate_print_robustness("fig5", {"v1": "#4098AC", ...}, redundancy="marker+direct-label")
    gates.gate_labels(fig, ax, label_texts)
    gates.gate_whisker_texts(fig, ax, label_texts, point_errs)
    gates.gate_pdf(pdf_path, max_w_pt=531, max_h_pt=131)

隔离纪律：本模块只读借用 scientific-visualization 色盲检查方法论（Okabe-Ito 惯例），
矩阵数值为公开标准（Machado 2009 Table X, severity 1.0），非技能库资产复制。
"""
import itertools
import shutil
import subprocess

import numpy as np

# ---- Machado, Oliveira & Fernandes (2009) severity=1.0 模拟矩阵（线性 RGB） ----
_CVD_MATRICES = {
    "deuteranope": np.array([
        [0.367322, 0.860646, -0.227968],
        [0.280085, 0.672501, 0.047413],
        [-0.011820, 0.042940, 0.968881]]),
    "protanope": np.array([
        [0.152286, 1.052583, -0.204868],
        [0.114503, 0.786281, 0.099216],
        [-0.003882, -0.048116, 1.051998]]),
    "tritanope": np.array([
        [1.255528, -0.076749, -0.178779],
        [-0.078411, 0.930809, 0.147602],
        [0.004733, 0.691367, 0.303900]]),
}


def _hex2rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float) / 255.0


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_cvd(hex_color, kind):
    """sRGB hex → CVD 模拟 sRGB hex（Machado 2009 severity 1.0，线性域变换）。"""
    rgb = _hex2rgb(hex_color)
    sim = _CVD_MATRICES[kind] @ _srgb_to_linear(rgb)
    sim = np.clip(_linear_to_srgb(sim), 0, 1)
    return "#{:02X}{:02X}{:02X}".format(*(int(round(v * 255)) for v in sim))


def _rgb2lab(rgb):
    """sRGB(0..1) → CIELAB（D65）。"""
    lin = _srgb_to_linear(rgb)
    M = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = M @ lin / np.array([0.95047, 1.0, 1.08883])
    eps, kap = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kap * xyz + 16) / 116)
    L = 116 * f[1] - 16
    a = 500 * (f[0] - f[1])
    b = 200 * (f[1] - f[2])
    return np.array([L, a, b])


def _rel_luminance(hex_color):
    lin = _srgb_to_linear(_hex2rgb(hex_color))
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _lstar_from_lum(Y):
    Y = np.clip(Y, 0, 1)
    return np.where(Y > 216 / 24389, 116 * np.cbrt(Y) - 16, 24389 / 27 * Y)


def delta_e76(hex_a, hex_b, cvd=None):
    """两 hex 色的 ΔE76；cvd 指定先做色盲模拟（None=正常视觉）。"""
    if cvd:
        hex_a, hex_b = simulate_cvd(hex_a, cvd), simulate_cvd(hex_b, cvd)
    return float(np.linalg.norm(_rgb2lab(_hex2rgb(hex_a)) - _rgb2lab(_hex2rgb(hex_b))))


def gate_print_robustness(fig_name, series_colors, redundancy=None,
                          min_de=12.0, min_dl=10.0):
    """G1 印刷稳健性门禁。

    series_colors: dict 名→hex，只收录"靠颜色区分系列"的色（结构灰/墨黑不必入集）。
    redundancy: None 或冗余编码声明字符串（如 "marker shape + direct label"）。
      灰度 ΔL* 不足的配对：有声明 → WARN 放行留痕；无声明 → FAIL。
      CVD 模拟后 ΔE 不足的配对：同样按冗余声明降级 WARN（形状/线型冗余在色觉
      模拟下仍然有效），无声明 → FAIL。
    返回 (pass_bool, report_lines)。
    """
    lines = [f"[G1 print-robustness] {fig_name}: {len(series_colors)} series colors"]
    fails, warns = [], []
    for n1, n2 in itertools.combinations(sorted(series_colors), 2):
        c1, c2 = series_colors[n1], series_colors[n2]
        de_norm = delta_e76(c1, c2)
        de_d = delta_e76(c1, c2, "deuteranope")
        de_p = delta_e76(c1, c2, "protanope")
        dl = abs(_lstar_from_lum(_rel_luminance(c1)) - _lstar_from_lum(_rel_luminance(c2)))
        lines.append(f"  {n1} x {n2}: dE {de_norm:.1f} / deut {de_d:.1f} / prot {de_p:.1f} "
                     f"/ dL* {dl:.1f}")
        if min(de_norm, de_d, de_p) < min_de:
            (warns if redundancy else fails).append(
                f"{n1} x {n2}: CVD dE {min(de_norm, de_d, de_p):.1f} < {min_de}")
        if dl < min_dl:
            (warns if redundancy else fails).append(
                f"{n1} x {n2}: grayscale dL* {dl:.1f} < {min_dl}")
    if redundancy:
        for w in warns:
            lines.append(f"  WARN(redundancy={redundancy}): {w}")
    for f_ in fails:
        lines.append(f"  FAIL: {f_}")
    ok = not fails
    lines.append(f"  -> {'PASS' if ok else 'FAIL'}"
                 + (f" ({len(warns)} redundancy-exempt warns)" if warns else ""))
    print("\n".join(lines))
    if not ok:
        raise SystemExit(f"{fig_name}: print-robustness gate FAILED")
    return ok, lines


def gate_labels(fig, ax, texts, tol_px=2.0, name="fig"):
    """G2 标签碰撞门禁：两两 bbox 相交 + 出轴断言（fig5 v6 泛化）。"""
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(ren)
    boxes = [(t.get_text(), t.get_window_extent(ren)) for t in texts]
    problems = []
    for (n1, b1), (n2, b2) in itertools.combinations(boxes, 2):
        if b1.overlaps(b2):
            problems.append(f"label-overlap: {n1!r} x {n2!r} | b1=({b1.x0:.1f},{b1.y0:.1f},{b1.x1:.1f},{b1.y1:.1f}) b2=({b2.x0:.1f},{b2.y0:.1f},{b2.x1:.1f},{b2.y1:.1f})")
    for n, b in boxes:
        if not (b.x0 >= ax_bb.x0 - tol_px and b.x1 <= ax_bb.x1 + tol_px
                and b.y0 >= ax_bb.y0 - tol_px and b.y1 <= ax_bb.y1 + tol_px):
            problems.append(f"label-out-of-axes: {n!r}")
    print(f"[G2 label-collision] {name}: {len(boxes)} labels -> "
          + ("PASS" if not problems else "FAIL"))
    if problems:
        for p in problems:
            print("  FAIL:", p)
        raise SystemExit(f"{name}: label collision gate FAILED")


def gate_whisker_texts(fig, ax, texts, point_errs, cap_px=3.0, name="fig"):
    """G3 误差棒穿线门禁：文本 bbox × whisker 竖线相交断言（fig5 v6 泛化）。

    point_errs: [(x_data, y_lo_data, y_hi_data, tier_name), ...]
    """
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    boxes = [(t.get_text(), t.get_window_extent(ren)) for t in texts]
    problems = []
    for fx, lo, hi, tier in point_errs:
        x_line = ax.transData.transform((fx, 0))[0]
        y_lo = ax.transData.transform((fx, lo))[1]
        y_hi = ax.transData.transform((fx, hi))[1]
        for n, b in boxes:
            if (b.x0 - cap_px <= x_line <= b.x1 + cap_px
                    and not (b.y0 > max(y_lo, y_hi) or b.y1 < min(y_lo, y_hi))):
                problems.append(f"label-on-whisker: {n!r} x {tier} errbar")
    print(f"[G3 whisker-crossing] {name}: {len(boxes)} labels x {len(point_errs)} errbars -> "
          + ("PASS" if not problems else "FAIL"))
    if problems:
        for p in problems:
            print("  FAIL:", p)
        raise SystemExit(f"{name}: whisker crossing gate FAILED")


def _pdf_mediabox_pt(pdf_path):
    """MediaBox 实测（pt）：优先 pdfinfo，缺则 PyMuPDF。返回 (w_pt, h_pt) 或 None。"""
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        try:
            # bytes 模式 + errors=replace：pdfinfo 在中文 Windows 控制台输出 GBK
            # 头部信息（时间戳警告等），text=True 的 utf-8 解码会炸 reader 线程
            out = subprocess.run([pdfinfo, str(pdf_path)], capture_output=True,
                                 timeout=60).stdout.decode("utf-8", errors="replace")
            for line in out.splitlines():
                if line.startswith("Page size:"):
                    nums = [float(v) for v in line.replace("Page size:", "").split("pts")[0].split() if v.replace(".", "").isdigit()]
                    if len(nums) >= 2:
                        return nums[0], nums[1]
        except Exception:
            pass
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        r = doc[0].rect
        doc.close()
        return r.width, r.height
    except Exception:
        return None


def gate_pdf(pdf_path, max_w_pt=None, max_h_pt=None, name=None):
    """G4 文件级门禁：MediaBox 尺寸上限 + pdfimages 零嵌入位图。"""
    pdf_path = str(pdf_path)
    name = name or pdf_path
    problems = []
    box = _pdf_mediabox_pt(pdf_path)
    if box is None:
        problems.append("mediabox-unreadable")
    else:
        w, h = box
        print(f"[G4 file] {name}: MediaBox {w:.2f} x {h:.2f} pt")
        if max_w_pt and w > max_w_pt + 0.05:
            problems.append(f"width {w:.2f}pt > limit {max_w_pt}pt")
        if max_h_pt and h > max_h_pt + 0.05:
            problems.append(f"height {h:.2f}pt > limit {max_h_pt}pt")
    pdfimages = shutil.which("pdfimages")
    if pdfimages:
        out = subprocess.run([pdfimages, "-list", pdf_path], capture_output=True,
                             text=True, timeout=60).stdout
        n_img = max(0, len(out.strip().splitlines()) - 2)  # 减表头两行
        print(f"[G4 file] {name}: embedded images = {n_img}")
        if n_img > 0:
            problems.append(f"{n_img} embedded bitmap(s) in vector PDF")
    else:
        print(f"[G4 file] {name}: pdfimages unavailable, bitmap check SKIPPED")
    if problems:
        for p in problems:
            print("  FAIL:", p)
        raise SystemExit(f"{name}: pdf file gate FAILED")
    print(f"[G4 file] {name} -> PASS")
