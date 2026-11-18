# -*- coding: utf-8 -*-
"""paper_state_check — 论文返修的统一状态基线与一致性门禁（防"基线不一致/各轮各说各话"）。

解决的现实问题：多轮返修中，不同 agent/不同轮次各自测量（行号漂移、词数口径不一、
冷审读到旧状态、CHECKLIST 声明与工件不符、caption 计数三种说法）。本脚本把
"当前稿子的客观状态"变成一份机器可复算的快照，任何修改前后都必须跑同一把尺子。

用法:
  python paper_state_check.py --config paper_state_config.json --mode snapshot \
      --out state_baseline.json          # 修改前: 建立基线
  python paper_state_check.py --config paper_state_config.json --mode check \
      --baseline state_baseline.json     # 修改后: 对照基线做门禁（exit 1 = 违例）
  追加 --update-baseline: check 通过后把基线刷新为当前状态（用于有意的结构变更之后）。

检查项（全部由本脚本统一定义，禁止各 agent 自行 improvising 计数）:
  1) 编译日志健康: LaTeX 错误(!)行 / Overfull / Float too large / undefined 计数
  2) PDF 文本: 坏串(如 "??", 历史笔误)计数=0；关键结果数字在位（带数字边界）
  3) 页数（pdfinfo, 缺则用 pdftotext 换页符估算）
  4) 数字保全: strict 文件逐文件数字多重集不减；relaxed 文件关键数字在位
  5) em-dash(---) 密度: 逐文件上限（跳过 % 注释行）
  6) caption 清单: 逐条词数，不得超过基线+容差（防再次膨胀）
  7) 摘要词数 ≤ 上限（优先调用项目既有计数器，保持口径一致）
  8) main.aux 的 label→编号映射（跨文档写死编号时核对用）
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def numbers_multiset(text: str) -> dict:
    out: dict = {}
    for n in re.findall(r"[0-9]+(?:\.[0-9]+)?", text):
        out[n] = out.get(n, 0) + 1
    return out


def emdash_count(text: str) -> int:
    """统计正文行（跳过 % 注释行）中的 --- 出现次数。"""
    total = 0
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        total += line.count("---")
    return total


def strip_latex(s: str) -> str:
    s = re.sub(r"\\[A-Za-z]+\*?(\[[^\]]*\])?", " ", s)  # \command[\opt] → 空格
    s = s.replace("\\", " ")
    s = re.sub(r"[{}]", " ", s)
    s = s.replace("~", " ").replace("--", " ").replace("---", " ")
    return re.sub(r"\s+", " ", s).strip()


def find_captions(tex: str) -> list:
    """返回 [(前 60 字符摘要, 词数)]，用括号配对提取 \\caption{...}。"""
    out = []
    i = 0
    while True:
        i = tex.find("\\caption{", i)
        if i < 0:
            break
        j = i + len("\\caption{")
        depth = 1
        k = j
        while k < len(tex) and depth > 0:
            ch = tex[k]
            if ch == "\\":
                k += 2
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        body = tex[j:k - 1]
        words = len(strip_latex(body).split())
        out.append({"preview": strip_latex(body)[:60], "words": words})
        i = k
    return out


def abstract_words(tex: str) -> int:
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        return -1
    return len(strip_latex(m.group(1)).split())


def run(cmd: str, cwd: Path = None) -> str:
    return subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None,
                          capture_output=True).stdout.decode("utf-8", "replace")


def pdf_text(pdf: Path, tmp: Path) -> str:
    txt = tmp / (pdf.stem + "_pstate.txt")
    run(f'pdftotext -layout "{pdf}" "{txt}"')
    if txt.exists():
        return read_text(txt)
    return ""


def page_count(pdf: Path, tmp: Path) -> int:
    info = run(f'pdfinfo "{pdf}"')
    m = re.search(r"Pages:\s+(\d+)", info)
    if m:
        return int(m.group(1))
    t = pdf_text(pdf, tmp)
    return t.count("\f") + 1 if t else -1


def bad_string_count(text: str, bad: str) -> int:
    """坏串计数：要求坏串前不是字母（避免 approximately 误伤 pprox 检测）。"""
    return len(re.findall(r"(?<![A-Za-z])" + re.escape(bad), text))


def key_number_ok(text: str, num: str) -> bool:
    pat = r"(?<![0-9.])" + re.escape(num) + r"(?![0-9.])"
    return re.search(pat, text) is not None


def collect_state(cfg: dict, root: Path, tmp: Path) -> dict:
    state: dict = {"docs": {}, "files": {}, "captions": {}, "labels": {}}
    for doc in cfg["docs"]:
        log = read_text(root / doc["log"]) if (root / doc["log"]).exists() else ""
        pdf = root / doc["pdf"]
        text = pdf_text(pdf, tmp) if pdf.exists() else ""
        state["docs"][doc["name"]] = {
            "pages": page_count(pdf, tmp) if pdf.exists() else -1,
            "errors": len([l for l in log.splitlines() if l.startswith("!")]),
            "overfull": log.count("Overfull"),
            "float_too_large": log.count("Float too large"),
            "undefined": log.count("undefined"),
            "bad_strings": {b: bad_string_count(text, b) for b in cfg.get("pdf_bad_strings", [])},
            "key_numbers_missing": [k for k in cfg.get("pdf_key_numbers", {}).get(doc["name"], [])
                                    if not key_number_ok(text, k)],
        }
    for f in cfg["numbers"]["strict"] + list(cfg["numbers"].get("relaxed", {}).keys()):
        p = root / f
        if not p.exists():
            state["files"][f] = {"missing": True}
            continue
        t = read_text(p)
        state["files"][f] = {"emdash": emdash_count(t),
                             "numbers": numbers_multiset(t) if f in cfg["numbers"]["strict"] else {}}
    for f in cfg.get("caption_files", []):
        p = root / f
        if p.exists():
            state["captions"][f] = find_captions(read_text(p))
    ab = cfg.get("abstract", {})
    if ab.get("file"):
        ap = root / ab["file"]
        if ap.exists():
            repo = root.parent.parent.parent
            if ab.get("external_counter") and (repo / ab["external_counter"]).exists():
                r = subprocess.run([sys.executable, "-I", str(repo / ab["external_counter"])],
                                   capture_output=True, cwd=str(repo))
                nums = re.findall(r"(\d+)\s*$", r.stdout.decode("utf-8", "replace"), re.M)
                state["abstract"] = {"external": [int(x) for x in nums[-2:]]}
            else:
                state["abstract"] = {"words": abstract_words(read_text(ap))}
    aux = root / "main.aux"
    if aux.exists():
        state["labels"] = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", read_text(aux)))
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["snapshot", "check"], required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", default=None, help="snapshot 输出路径")
    ap.add_argument("--update-baseline", action="store_true",
                    help="check 通过后刷新基线（有意的结构变更之后用）")
    a = ap.parse_args()

    cfg = json.loads(read_text(Path(a.config)))
    root = (Path(a.config).resolve().parent.parent.parent / cfg.get("latex_root", "docs/paper/latex"))
    if not root.exists():
        root = Path(cfg.get("abs_latex_root", root))
    tmp = root / ".pstate_tmp"
    tmp.mkdir(exist_ok=True)

    if a.mode == "snapshot":
        state = collect_state(cfg, root, tmp)
        Path(a.baseline).write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[snapshot] written {a.baseline}: "
              f"{sum(1 for _ in state['files'])} files, "
              f"{sum(len(v) for v in state['captions'].values())} captions, "
              f"{len(state['labels'])} labels")
        return

    base = json.loads(read_text(Path(a.baseline)))
    cur = collect_state(cfg, root, tmp)
    problems: list = []

    for doc, b in base["docs"].items():
        c = cur["docs"].get(doc, {})
        for k, limit in (("errors", 0), ("float_too_large", 0), ("undefined", 0)):
            if c.get(k, 0) > limit:
                problems.append(f"{doc}.{k}={c.get(k)} (应为 ≤{limit})")
        if c.get("overfull", 0) > b.get("overfull", 0):
            problems.append(f"{doc}.overfull {b.get('overfull')}->{c.get('overfull')} (不得增加)")
        for bad, bn in b.get("bad_strings", {}).items():
            if c.get("bad_strings", {}).get(bad, 0) > 0:
                problems.append(f"{doc}.pdf 坏串 '{bad}' 出现 {c['bad_strings'][bad]} 次")
        for k in b.get("key_numbers_missing", []):
            pass  # 基线里缺失的关键数字由 relaxed/required 侧接管
        for k in cfg.get("pdf_key_numbers", {}).get(doc, []):
            if k in cur["docs"][doc]["key_numbers_missing"]:
                problems.append(f"{doc}.pdf 关键数字 {k} 不在 PDF 文本中")
        if b.get("pages", 0) > 0 and c.get("pages", 0) > 0 and abs(c["pages"] - b["pages"]) > 2:
            problems.append(f"{doc}.pages {b['pages']}->{c['pages']} (波动 >2 页需说明)")

    for f, b in base["files"].items():
        c = cur["files"].get(f, {})
        if b.get("missing"):
            continue
        cap = cfg.get("emdash_caps", {}).get(f)
        if cap is not None and c.get("emdash", 0) > cap:
            problems.append(f"{f}: em-dash {c.get('emdash')} > 上限 {cap}")
        if f in cfg["numbers"]["strict"]:
            for n, need in b.get("numbers", {}).items():
                if c.get("numbers", {}).get(n, 0) < need:
                    problems.append(f"{f}: 数字 {n} {need}->{c.get('numbers', {}).get(n, 0)} (不得减少)")
        req = cfg["numbers"].get("relaxed", {}).get(f, [])
        curtext = read_text(root / f) if (root / f).exists() else ""
        for n in req:
            if not key_number_ok(curtext, n):
                problems.append(f"{f}: 关键数字 {n} 不在文件中")

    for f, caps in base.get("captions", {}).items():
        curcaps = cur.get("captions", {}).get(f, [])
        if len(curcaps) != len(caps):
            problems.append(f"{f}: caption 数量 {len(caps)}->{len(curcaps)}")
            continue
        for i, (bc, cc) in enumerate(zip(caps, curcaps)):
            if cc["words"] > bc["words"] + 10:
                problems.append(f"{f}: 第{i+1}条 caption 词数 {bc['words']}->{cc['words']} (膨胀 >10 词)")

    ab = cfg.get("abstract", {})
    if ab.get("file") and "abstract" in base:
        bc, cc = base["abstract"], cur.get("abstract", {})
        vals = []
        if "external" in bc:
            vals += [v for v in bc["external"] if v > 0]
        if "external" in cc:
            vals += [v for v in cc["external"] if v > 0]
        if "words" in bc:
            vals.append(bc["words"])
        if "words" in cc:
            vals.append(cc["words"])
        for v in vals:
            if v > ab.get("max_words", 250):
                problems.append(f"摘要词数 {v} 超过 {ab.get('max_words', 250)}")

    print(json.dumps({"violations": problems, "state": cur}, ensure_ascii=False, indent=1)[:4000])
    if problems:
        print("STATE_CHECK_FAIL: " + str(len(problems)) + " 项违例")
        sys.exit(1)
    print("STATE_CHECK_OK")
    if a.update_baseline:
        Path(a.baseline).write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[baseline] refreshed -> {a.baseline}")


if __name__ == "__main__":
    main()
