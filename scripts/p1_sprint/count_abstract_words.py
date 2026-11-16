"""Abstract word counter per main.tex:132-137 double-standard method.

Base method: strip \\textbf/\\emph shells keeping content; strip other commands and $;
split on whitespace; count tokens containing alphanumerics; math groups count 1.
Conservative: additionally split tokens on em/en dashes (-- and ---) before counting.
"""
import re
import sys

SRC = r"D:\Desktop\PR论文\psd-framework\docs\paper\latex\main.tex"


def extract_abstract(text):
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    return m.group(1).strip()


def strip_inline_math(s):
    # replace $...$ groups with a single placeholder token
    return re.sub(r"\$[^$]*\$", " MATHGROUP ", s)


def base_tokens(abstract):
    s = abstract
    # NOTE: no comment stripping here — the abstract contains escaped \% which is
    # content, not a comment; a naive %.* strip would delete the rest of the line.
    # unwrap \textbf{...} and \emph{...} keeping content (handle nesting-free cases iteratively)
    for _ in range(6):
        s2 = re.sub(r"\\(?:textbf|emph)\{([^{}]*)\}", r" \1 ", s)
        if s2 == s:
            break
        s = s2
    # remove remaining commands like \geq, \pm, \, etc. (keep math group content handled below)
    s = strip_inline_math(s)
    s = re.sub(r"\\[A-Za-z]+", " ", s)   # command names out
    s = s.replace("{", " ").replace("}", " ")
    s = s.replace("~", " ")
    toks = [t for t in s.split() if re.search(r"[A-Za-z0-9]", t)]
    return toks


def conservative_count(abstract):
    toks = base_tokens(abstract)
    n = 0
    for t in toks:
        # split only on em/en dashes (2+ hyphens); single hyphen is a regular hyphen
        parts = re.split(r"--+", t)
        n += sum(1 for p in parts if re.search(r"[A-Za-z0-9]", p))
    return n


def main():
    text = open(SRC, encoding="utf-8").read()
    ab = extract_abstract(text)
    toks = base_tokens(ab)
    print("base (whitespace, dashes intact):", len(toks))
    print("conservative (em/en-dash split):", conservative_count(ab))
    if "-v" in sys.argv:
        for t in toks:
            print(" ", t)


if __name__ == "__main__":
    main()
