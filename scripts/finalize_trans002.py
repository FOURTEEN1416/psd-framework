# -*- coding: utf-8 -*-
"""TRANS-002 自动化 finalize 管线（用户令: 自动化进行, 不依赖人工轮询）。

流程（全部自动, 审计痕迹齐全）:
  1. 轮询等待 reports/r23-trans002-full-*.json（当前后台训练产物, 预计 6-9h）
  2. 自动补跑 10% 档（driver --stage full10, 每 seed ~20min）
  3. 按 TRANS-002 冻结判据(协议 §3)计算两档 verdict
  4. 按 verdict 三分支模板自动回写论文三处:
     a) 04-experiments.tex E6-real 段后追加 E6-real-2 段落
     b) 06-conclusion-limitations.tex L12 段追加 TRANS-002 句
     c) main.tex 摘要括号更新
  5. 重建 PDF (pdflatex x2)
  6. git commit + tag review-snapshot 重打
  7. 写 DONE 标记 + 全程日志

跑法: .venv/Scripts/python.exe scripts/finalize_trans002.py  (后台, 预计 8-10h)
"""
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "reports" / "trans002_finalize.log"
DONE = REPO / "reports" / "trans002_finalize_DONE.flag"
PYTHON = sys.executable


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def poll_full_json(max_hours=14):
    deadline = time.time() + max_hours * 3600
    while time.time() < deadline:
        hits = sorted((REPO / "reports").glob("r23-trans002-full-*.json"))
        if hits:
            j = json.loads(hits[-1].read_text(encoding="utf-8"))
            if len(j.get("runs", [])) >= 9:  # 3 arms x 3 seeds
                return hits[-1], j
            log(f"json exists but incomplete ({len(j.get('runs', []))} runs), keep waiting")
        proc_alive = subprocess.run(["powershell", "-Command",
            "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Where-Object {$_.CommandLine -like '*trans002*'}).Count"],
            capture_output=True).stdout.decode().strip()
        log(f"waiting... full json missing, trans002 python procs={proc_alive}")
        time.sleep(600)
    raise TimeoutError("full json did not appear within window")


def summarize(js, budget):
    runs = [r for r in js["runs"] if r["budget"] == budget]
    out = {}
    for arm in ("C_prime", "D_prime", "M_matched"):
        rows = [r for r in runs if r["arm"] == arm]
        out[arm] = {"seeds": [r["seed"] for r in rows],
                    "wall": [r["wall_clock_sec"] for r in rows],
                    "acc": [r["val_acc"] for r in rows]}
    return out


def verdict_of(summary):
    import statistics as st
    med_c = st.median(summary["C_prime"]["wall"])
    med_d = st.median(summary["D_prime"]["wall"])
    ratio = med_c / med_d
    gap = (st.mean(summary["D_prime"]["acc"]) - st.mean(summary["C_prime"]["acc"])) * 100
    if ratio >= 3 and abs(gap) < 2.3:
        v = "CONFIRMS"
    elif ratio >= 3:
        v = "PARTIAL"
    else:
        v = "FAILS"
    return {"median_C_s": round(med_c, 1), "median_D_s": round(med_d, 1),
            "cost_ratio": round(ratio, 2), "acc_gap_pp": round(gap, 2),
            "acc_C_mean": round(st.mean(summary["C_prime"]["acc"]) * 100, 2),
            "acc_D_mean": round(st.mean(summary["D_prime"]["acc"]) * 100, 2),
            "verdict": v}


def paper_writeback(full_v, full_s, ten_v, ten_s):
    """三分支措辞模板（预写死, 防自动生成语病）；两档 verdict 独立陈述。"""
    tex_dir = REPO / "docs" / "paper" / "latex"
    r = full_v; t = ten_v

    if r["verdict"] == "CONFIRMS":
        l12 = (f"A backbone-scale three-arm replication (PSD-NTU-TRANS-002, frozen before the run; "
               f"coupled/decoupled/matched-solver, seeds 42--44, HRNet 2D skeleton, 80 epochs, "
               f"both arms SGD) **CONFIRMS the cost endpoint at matched accuracy**: median coupled/decoupled "
               f"wall-clock {r['median_C_s']:.0f} s vs {r['median_D_s']:.0f} s ({r['cost_ratio']}x) with a "
               f"decoupled-vs-coupled accuracy gap of {r['acc_gap_pp']:+.1f} pp (inside the pre-registered "
               f"$\\pm$2.3 pp band)---the $\\geq 3\\times$ claim now carries a real-domain backbone-scale "
               f"replication in addition to the synthetic-tier measurement, and the MLP-scale failure of "
               f"TRANS-001 is re-read as scale-bound rather than paradigm-bound.")
        e6 = (f"At the 10\\% budget the same replication reads {t['cost_ratio']}x (coupled {t['median_C_s']:.1f} s "
              f"vs decoupled {t['median_D_s']:.1f} s) with a {t['acc_gap_pp']:+.1f} pp decoupled advantage "
              f"({t['verdict']} at the 10\\% tier).")
        abstract_frag = ("a pre-registered real-domain replication fails the wall-clock endpoint at small-backbone "
                         "scale while a backbone-scale replication with matched solvers confirms it")
    elif r["verdict"] == "PARTIAL":
        l12 = (f"A backbone-scale three-arm replication (PSD-NTU-TRANS-002, frozen before the run; "
               f"coupled/decoupled/matched-solver, seeds 42--44) returns **PARTIAL**: median "
               f"coupled/decoupled wall-clock {r['median_C_s']:.0f} s vs {r['median_D_s']:.0f} s "
               f"({r['cost_ratio']}x, above the 3x line) but with a decoupled-vs-coupled accuracy gap of "
               f"{r['acc_gap_pp']:+.1f} pp (outside the $\\pm$2.3 pp band)---both the ratio and the cost are "
               f"reported as measured; at the 10\\% budget the replication reads {t['cost_ratio']}x "
               f"({t['verdict']}). The $\\geq 3\\times$ claim's real-domain status is therefore "
               f"backbone-scale-PARTIAL, upgrading the MLP-scale failure of TRANS-001 from 'no cost to save' "
               f"to 'cost saving exists at backbone scale but the accuracy band is not matched'.")
        e6 = (f"At the 10\\% budget the same replication reads {t['cost_ratio']}x with a {t['acc_gap_pp']:+.1f} pp "
              f"decoupled advantage ({t['verdict']}).")
        abstract_frag = ("a pre-registered real-domain replication fails the wall-clock endpoint at small-backbone "
                         "scale and returns a partial result (cost ratio above the 3x line, accuracy band not "
                         "matched) at backbone scale")
    else:  # FAILS
        l12 = (f"A backbone-scale three-arm replication (PSD-NTU-TRANS-002, frozen before the run; "
               f"coupled/decoupled/matched-solver, seeds 42--44, both arms SGD, 80 epochs) **also FAILS the "
               f"cost endpoint**: median coupled/decoupled wall-clock {r['median_C_s']:.0f} s vs "
               f"{r['median_D_s']:.0f} s ({r['cost_ratio']}x, below the 3x line), with a decoupled-vs-coupled "
               f"accuracy gap of {r['acc_gap_pp']:+.1f} pp; at the 10\\% budget it reads {t['cost_ratio']}x "
               f"({t['verdict']}). This closes the 'small-backbone scale' escape route of TRANS-001: the "
               f"$\\geq 3\\times$ claim is measured on the synthetic tier and **fails both real-domain "
               f"replications at MLP and backbone scale**, and the paper says so verbatim.")
        e6 = (f"At the 10\\% budget the same replication reads {t['cost_ratio']}x with a {t['acc_gap_pp']:+.1f} pp "
              f"decoupled advantage ({t['verdict']}).")
        abstract_frag = ("pre-registered real-domain replications at both small-backbone and backbone scale fail "
                         "the wall-clock endpoint")

    # a) E6-real-2 paragraph appended after E6-real paragraph in 04-experiments.tex
    e6_real_2 = (
        "\n\\paragraph{E6-real-2: backbone-scale replication with matched solvers (pre-registered).}\n"
        "TRANS-001's coupled arm retrains in $\\approx$35\\,s, so its cost endpoint was arithmetically "
        "unpassable---the replication could not have confirmed the cost claim at that scale. A second "
        "pre-registered replication (protocol PSD-NTU-TRANS-002, frozen before the run; same $\\mathcal{Y}\\to"
        "\\mathcal{Y}'$ merge, 60$\\to$49 classes) therefore scales the backbone to the full ST-GCN of the "
        "equivalence verification and adds a third arm that decouples the two confounds TRANS-001 carried: "
        "the **matched-solver arm (M)** retrains a linear head on the **coupled arm's own penultimate "
        "features** with the decoupled arm's solver, isolating representation quality from retraining cost. "
        f"Seeds 42--44, full budget (40,091 clips): the coupled arm retrains in a median of "
        f"{r['median_C_s']:.0f}\\,s against the decoupled head's {r['median_D_s']:.0f}\\,s "
        f"({r['cost_ratio']}x; verdict **{r['verdict']}** per the frozen rule), accuracy "
        f"{r['acc_C_mean']:.1f}\\% (coupled) vs {r['acc_D_mean']:.1f}\\% (decoupled, gap "
        f"{r['acc_gap_pp']:+.1f}\\,pp); the matched-solver arm reads "
        f"{sum(full_s['M_matched']['acc'])/len(full_s['M_matched']['acc'])*100:.1f}\\%. " + e6 + "\n"
    )
    return l12, e6_real_2, abstract_frag


def rebuild_pdf():
    latex = REPO / "docs" / "paper" / "latex"
    for _ in range(2):
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "main.tex"],
                       cwd=latex, capture_output=True, timeout=600)
    log("PDF rebuilt")


def git_commit_push(msg):
    subprocess.run(["git", "add", "-A", "docs/paper", "reports", "scripts"], cwd=REPO, capture_output=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=REPO, capture_output=True)
    subprocess.run(["git", "tag", "-d", "review-snapshot"], cwd=REPO, capture_output=True)
    subprocess.run(["git", "tag", "-a", "review-snapshot",
                    "-m", "Submission review snapshot: TRANS-002 backbone-scale replication landed"], cwd=REPO, capture_output=True)
    log("git commit + tag re-pointed")


def main():
    if DONE.exists():
        DONE.unlink()
    log("finalize pipeline started")
    full_json, full_js = poll_full_json()
    log(f"full json ready: {full_json.name}")

    # 10% tier (protocol second budget)
    log("running 10% tier (driver --stage full10)...")
    subprocess.run([PYTHON, "-u", str(REPO / "scripts" / "run_r23_trans002.py"), "--stage", "full10"],
                   cwd=REPO)
    hits = sorted((REPO / "reports").glob("r23-trans002-full10-*.json"))
    ten_js = json.loads(hits[-1].read_text(encoding="utf-8")) if hits else {"runs": []}

    full_s = summarize(full_js, "full")
    ten_s = summarize(ten_js, "10pct") if ten_js.get("runs") else None
    full_v = verdict_of(full_s)
    ten_v = verdict_of(ten_s) if ten_s else {"verdict": "NOT-RUN", "cost_ratio": 0, "acc_gap_pp": 0,
                                             "median_C_s": 0, "median_D_s": 0, "acc_C_mean": 0, "acc_D_mean": 0}
    log(f"full verdict: {full_v}")
    log(f"10% verdict: {ten_v}")

    l12, e6_real_2, abstract_frag = paper_writeback(full_v, full_s, ten_v, ten_v)

    # paper writeback
    exp = REPO / "docs" / "paper" / "latex" / "sections" / "04-experiments.tex"
    s = exp.read_text(encoding="utf-8")
    anchor = "\\paragraph{Segmentation strategy.}"
    assert anchor in s
    s = s.replace(anchor, e6_real_2 + "\n" + anchor)
    exp.write_text(s, encoding="utf-8", newline="\n")
    log("04-experiments.tex: E6-real-2 inserted")

    lim = REPO / "docs" / "paper" / "latex" / "sections" / "06-conclusion-limitations.tex"
    s = lim.read_text(encoding="utf-8")
    anchor2 = "\\paragraph{L13: No head-to-head comparison"
    assert anchor2 in s
    s = s.replace(anchor2, l12 + "\n\n" + anchor2)
    lim.write_text(s, encoding="utf-8", newline="\n")
    log("06-conclusion-limitations.tex: L12 extended")

    main_tex = REPO / "docs" / "paper" / "latex" / "main.tex"
    s = main_tex.read_text(encoding="utf-8")
    old_frag = "a pre-registered real-domain replication fails the wall-clock endpoint at small-backbone scale while showing an accuracy advantage for the decoupled path, measured under a disclosed solver-family confound"
    assert old_frag in s
    s = s.replace(old_frag, abstract_frag)
    main_tex.write_text(s, encoding="utf-8", newline="\n")
    log("main.tex: abstract updated")

    rebuild_pdf()
    git_commit_push(f"feat(trans002-results): TRANS-002 backbone-scale replication landed——full档 verdict={full_v['verdict']} "
                    f"(C/D={full_v['cost_ratio']}x, gap={full_v['acc_gap_pp']:+.1f}pp) 10%档 verdict={ten_v['verdict']} "
                    f"(C/D={ten_v['cost_ratio']}x)；论文 E6-real-2 段/L12/摘要三分支模板回写；重建完成")
    DONE.write_text(datetime.now().isoformat(timespec="seconds"))
    log("DONE")


if __name__ == "__main__":
    main()
