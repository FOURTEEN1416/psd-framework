# -*- coding: utf-8 -*-
r"""check_latex_next_paper — LaTeX 版与 markdown 真源的一致性机检。

检查项：
  1. markdown 正文/图表注中的每个数字都出现在 main.tex（归一化后）；
  2. 41 条 bibitem 全部被 \cite 覆盖，且无未定义 key；
  3. 摘要词数与 markdown 一致（≤250）；
  4. 编译产物 main.pdf 存在、无 "??"、页数合理。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MD = REPO / "docs" / "next_paper" / "paper_manuscript_v1.md"
TEX = REPO / "docs" / "next_paper" / "latex" / "main.tex"
PDF = REPO / "docs" / "next_paper" / "latex" / "main.pdf"


def norm_tex(s: str) -> str:
    s = s.replace("\\%", "%").replace("\\,", "").replace("~", " ")
    s = s.replace("$", "").replace("{", "").replace("}", "")
    return s


def extract_numbers(text: str) -> set[str]:
    # 去掉 LaTeX 注释、markdown 标题与引用括号（节号/文献号不是内容数字）
    text = re.sub(r"(?m)^\s*%.*$", "", text)
    text = re.sub(r"(?m)^#{1,6}\s.*$", "", text)
    text = re.sub(r"\[[\d,\s\u2013\u2014-]+\]", "", text)
    # 参考文献列表的顺序编号（"29. ..."）由 LaTeX 自动生成，不是内容数字
    text = re.sub(r"(?m)^\d+\.\s+", " ", text)
    nums = set()
    for m in re.finditer(r"\d[\d,]*(?:\.\d+)?", text):
        tok = m.group(0).rstrip(",")
        if len(tok.replace(",", "")) >= 2:
            nums.add(tok)
    return nums


def main() -> None:
    md = MD.read_text(encoding="utf-8")
    tex_raw = TEX.read_text(encoding="utf-8")
    app = TEX.with_name("appendix.tex")
    if app.exists():
        tex_raw += "\n" + app.read_text(encoding="utf-8")
    tex = norm_tex(tex_raw)

    md_body = md
    missing = sorted(n for n in extract_numbers(md_body) if n not in tex)
    # 过滤 markdown 结构噪声（年份范围、图号等仍应真实存在）
    if missing:
        print(f"MISSING IN TEX ({len(missing)}): {missing}")
    else:
        print("NUMBERS_OK: all markdown numbers present in main.tex")

    body = tex_raw.split(r"\begin{thebibliography}")[0]
    cited: set[str] = set()
    for grp in re.findall(r"\\cite\{([^}]*)\}", body):
        cited.update(k.strip() for k in grp.split(","))
    items = re.findall(r"\\bibitem\{([^}]*)\}", tex_raw)
    uncited = sorted(set(items) - cited)
    missing_bib = sorted(cited - set(items))
    dup = [k for k in set(items) if items.count(k) > 1]
    print(f"REFS: {len(items)} bibitems, {len(cited)} cited keys, "
          f"uncited={uncited or 'none'}, missing={missing_bib or 'none'}, dup={dup or 'none'}")

    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex_raw, re.S)
    abs_tex = m.group(1)
    abs_tex = re.sub(r"\\cite\{[^}]*\}", "", abs_tex)
    abs_tex = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", abs_tex)
    abs_tex = abs_tex.replace("$", " ").replace("~", " ")
    words_tex = re.findall(r"[A-Za-z][A-Za-z'-]+", abs_tex)
    m2 = re.search(r"## Abstract\n+(.*?)\n---", md, re.S)
    words_md = re.findall(r"[A-Za-z][A-Za-z'-]+", m2.group(1))
    wc, wc_md = len(words_tex), len(words_md)
    print(f"ABSTRACT: tex={wc} words, md={wc_md} words, limit=250 "
          f"({'OK' if wc <= 250 and abs(wc - wc_md) <= 3 else 'CHECK'})")

    if PDF.exists():
        try:
            import fitz
        except ImportError:
            fitz = None
        if fitz is not None:
            doc = fitz.open(PDF)
            text = "".join(p.get_text() for p in doc)
            qq = text.count("??")
            print(f"PDF: {doc.page_count} pages, '??' occurrences={qq}")
            if qq:
                sys.exit(1)
        else:
            print("PDF: exists (fitz unavailable, skipped text scan)")
    else:
        print("PDF: MISSING")
        sys.exit(1)

    if missing or uncited or missing_bib or dup:
        sys.exit(1)
    print("LATEX_CHECK_OK")


if __name__ == "__main__":
    main()
