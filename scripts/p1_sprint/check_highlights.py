# -*- coding: utf-8 -*-
"""check_highlights — 校验 highlights.tex 是否符合 Elsevier 要件。

规则：3–5 条，每条 ≤85 字符（含空格）；条目中的数字必须出现在主稿 md 中。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HL = REPO / "docs" / "next_paper" / "latex" / "highlights.tex"
MD = REPO / "docs" / "next_paper" / "paper_manuscript_v1.md"


def main() -> None:
    raw = HL.read_text(encoding="utf-8")
    items = [ln[len(r"\item"):].strip() for ln in raw.splitlines() if ln.startswith(r"\item")]
    print(f"items: {len(items)}")
    ok = 3 <= len(items) <= 5
    md = MD.read_text(encoding="utf-8")
    for i, s in enumerate(items, 1):
        n = len(s)
        nums = [x.rstrip(".,/") for x in re.findall(r"\d[\d.,/]*", s)]
        missing = [x for x in nums if x not in md]
        flag = "OK" if n <= 85 else "TOO LONG"
        print(f"  {i}: {n:>2} chars [{flag}] missing_in_md={missing or 'none'} | {s}")
        ok = ok and n <= 85 and not missing
    print("HIGHLIGHTS_OK" if ok else "HIGHLIGHTS_FAIL")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
