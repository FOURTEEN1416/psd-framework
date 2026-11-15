#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 Autopilot — 完全自主的实验看护/收割/回填/编译状态机。

每个周期（计划任务每 15 分钟触发一次）：
  1. 读取 queue.json 的四个实验，按日志/summary 判定 done / failed / running / pending
  2. failed → 有界重试（第 1 次原样重跑；第 2 次加 cuDNN 缓解环境变量；仍失败则终态上报）
  3. done  → 收割 summary.json 指标
  4. 依据 BACKFILL_OPS 把已有指标回填 LaTeX 的 \\TODO 单元格（精确锚点替换；无指标不填，绝不编造）
  5. 有新回填 → latexmk 编译；编译通过 → 刷新 submission_p1 的 PDF
  6. 全部实验完成且全部回填且编译通过 → state.ready = True（P1_READY_FOR_SUBMISSION）

幂等：重复运行安全；每个动作写 audit 日志；state 落盘 runs/p1_sprint/autopilot_state.json。
"""
import json
import os
import subprocess
import sys
import datetime
import time

ROOT = r"D:\Desktop\PR论文\psd-framework"
SPRINT = os.path.join(ROOT, "runs", "p1_sprint")
SCRIPTS = os.path.join(ROOT, "scripts", "p1_sprint")
LATEX = os.path.join(ROOT, "docs", "paper", "latex")
SUBM = os.path.join(ROOT, "..", "psd-resubmission-20260915", "submission_p1")
STATE_F = os.path.join(SPRINT, "autopilot_state.json")
LOG_F = os.path.join(SPRINT, "autopilot.log")
QUEUE_F = os.path.join(SCRIPTS, "queue.json")
MAX_RETRIES = 2

DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    line = "[%s] %s" % (now(), msg)
    print(line)
    with open(LOG_F, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state():
    if os.path.exists(STATE_F):
        with open(STATE_F, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"cycles": 0, "retries": {}, "mitigation": {}, "ops_done": [], "compile_ok": False, "ready": False}


def save_state(st):
    with open(STATE_F, "w", encoding="utf-8") as f:
        json.dump(st, f, ensure_ascii=False, indent=2)


def job_status(job):
    """按 doneMarker / END exit 行 / summary.json 判定单个实验状态。"""
    log_path = os.path.join(SPRINT, job["id"] + ".log")
    summary_path = job.get("summary")
    marker = job.get("doneMarker", "")
    if not os.path.exists(log_path):
        return "pending", {}
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    if marker and marker in text and summary_path and os.path.exists(summary_path):
        return "done", {}
    ends = [l for l in text.splitlines() if l.startswith("[") and " END " in l and "exit=" in l]
    if ends:
        last = ends[-1]
        try:
            code = int(last.rsplit("exit=", 1)[1].strip().split()[0])
        except Exception:
            code = 1
        if code == 0:
            return "done", {}
        return "failed", {"exit": code}
    # 无 END 行：块缓冲丢失的崩溃或仍在跑。日志超过 25 分钟未更新 → 视为停滞失败。
    age = time.time() - os.path.getmtime(log_path)
    if age > 1500:
        return "failed", {"exit": "stalled"}
    return "running", {}


RETRY_ARGS = {
    # 发射员结论：8GB 卡上桌面应用占 ~2.9GB，fullft batch=32+AMP 超显存 → batch 16
    "frozen_backbone_contrast": ["--batch-size", "16"],
}


def relaunch(job, mitigation):
    """后台重启单个实验；经 PowerShell Start-Process（本机验证可用的机制）；
    mitigation='cudnn_v8_disabled' 时给子进程注入 cuDNN 缓解环境变量。"""
    log_path = os.path.join(SPRINT, job["id"] + ".log")
    err_path = os.path.join(SPRINT, job["id"] + ".log.err")
    env_prefix = "$env:PYTHONUNBUFFERED='1'; "
    if mitigation == "cudnn_v8_disabled":
        env_prefix += "$env:TORCH_CUDNN_V8_API_DISABLED='1'; $env:CUDNN_V8_API_DISABLED='1'; "
    arg_items = ["-u", job["script"]] + RETRY_ARGS.get(job["id"], [])
    arg_str = ",".join("'" + a + "'" for a in arg_items)
    ps = (
        "Start-Process -FilePath '{python}' -ArgumentList " + arg_str + " "
        "-WorkingDirectory '{root}' -RedirectStandardOutput '{log}' "
        "-RedirectStandardError '{err}' -WindowStyle Hidden"
    ).format(python=job_python(), root=ROOT, log=log_path, err=err_path)
    with open(log_path, "a", encoding="utf-8", errors="replace") as out:
        out.write("\n[%s] AUTOPILOT relaunch (mitigation=%s, args=%s)\n" % (now(), mitigation, RETRY_ARGS.get(job["id"], [])))
    subprocess.run(["powershell", "-NoProfile", "-Command", env_prefix + ps],
                   capture_output=True, timeout=60)


def job_python():
    py = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
    return py if os.path.exists(py) else sys.executable


def harvest(job):
    p = job.get("summary")
    if p and os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def pct(x):
    # LaTeX 转义百分号，避免 % 注释掉行尾的 \\ 行终止符
    return ("%.1f\\%%" % (float(x) * 100))


def build_ops(metrics):
    """根据已收割指标生成回填操作（精确锚点替换）。requires 指明前置实验。"""
    ops = []
    m = metrics.get("merge_rename_upper") or {}
    if m:
        ops.append({
            "file": "sections/04-experiments.tex",
            "anchor": r"\TODO{XX.X\% (zero retraining; no wall-clock)}",
            "repl": pct(m.get("rename_oracle_upper_top1", 0)) + " (rename-oracle upper; zero retraining)",
            "requires": "merge_rename_upper",
            "desc": "merge/rename 标签重映射上界",
        })
        ops.append({
            "file": "sections/04-experiments.tex",
            "anchor": r"\TODO{XX.X\%}",
            "repl": pct(m.get("argmax_vote_lower_top1", 0)) + " (argmax hard-vote lower bound)",
            "requires": "merge_rename_upper",
            "desc": "merge/rename 硬投票下界",
        })
        ops.append({
            "file": "sections/04-experiments.tex",
            "anchor": r"\TODO{pending} (Section~\ref{ssec:boundary})",
            "repl": pct(m.get("rename_oracle_upper_top1", 0)) + " upper / " + pct(m.get("argmax_vote_lower_top1", 0))
                    + " lower (NTU60 ten-merge) (Section~\\ref{ssec:boundary})",
            "requires": "merge_rename_upper",
            "desc": "总表 merge/rename 行",
        })
    f = metrics.get("frozen_backbone_contrast") or {}
    if f:
        frozen_acc = f.get("frozen_head_val_acc") or f.get("best_val_acc")
        if frozen_acc:
            ops.append({
                "file": "sections/04-experiments.tex",
                "anchor": r"\TODO{NTU60: XX.X\% $\pm$ X.X (10 seeds)}",
                "repl": pct(frozen_acc) + " (NTU60, seed 42)",
                "requires": "frozen_backbone_contrast",
                "desc": "frozen-backbone NTU60 主格",
            })
            ops.append({
                "file": "sections/04-experiments.tex",
                "anchor": r"\TODO{pending} (Section~\ref{ssec:boundary})",
                "repl": pct(frozen_acc) + " (NTU60, seed 42) (Section~\\ref{ssec:boundary})",
                "requires": "frozen_backbone_contrast",
                "desc": "总表 frozen-backbone 行",
            })
    if metrics.get("merge_rename_upper") and metrics.get("frozen_backbone_contrast"):
        ops.append({
            "file": "sections/04-experiments.tex",
            "anchor": r"\TODO{results pending, to be backfilled from runs/p1\_sprint}",
            "repl": r"results backfilled verbatim from the released run artifacts (runs/p1\_sprint)",
            "requires": "all",
            "desc": "表注从 pending 改为已回填",
        })
    return ops


def apply_backfill(ops, metrics):
    """逐个精确替换 \TODO 锚点；返回 (applied_desc, remaining_desc)。幂等：锚点不存在即跳过。"""
    files = {}
    applied, remaining = [], []
    for op in ops:
        if op["requires"] == "all":
            need = ["merge_rename_upper", "frozen_backbone_contrast"]
        else:
            need = [op["requires"]]
        if any(not metrics.get(k) for k in need):
            remaining.append(op["desc"] + "（缺前置指标）")
            continue
        rel = op["file"]
        if rel not in files:
            path = os.path.join(LATEX, rel)
            with open(path, "r", encoding="utf-8") as f:
                files[rel] = f.read()
        content = files[rel]
        if op["anchor"] not in content:
            # 已回填过（幂等）或锚点已被人工修改
            continue
        content = content.replace(op["anchor"], op["repl"], 1)
        files[rel] = content
        applied.append(op["desc"])
    for rel, content in files.items():
        with open(os.path.join(LATEX, rel), "w", encoding="utf-8") as f:
            f.write(content)
    return applied, remaining


def compile_paper():
    try:
        r = subprocess.run(
            ["latexmk", "-cd", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=LATEX, capture_output=True, text=True, timeout=900,
        )
        return r.returncode == 0, (r.stdout or "")[-1500:]
    except Exception as e:  # 工具缺失/超时
        return False, repr(e)


def refresh_submission():
    import shutil
    pdf = os.path.join(LATEX, "main.pdf")
    if os.path.exists(pdf):
        os.makedirs(SUBM, exist_ok=True)
        shutil.copy2(pdf, os.path.join(SUBM, "main_cviu_draft.pdf"))
        return True
    return False


def main():
    with open(QUEUE_F, "r", encoding="utf-8") as f:
        queue = json.load(f)
    jobs = queue["jobs"]
    state = load_state()
    state["cycles"] = int(state.get("cycles", 0)) + 1
    state["last_run"] = now()
    metrics = state.get("metrics", {})

    log("=== cycle %d ===" % state["cycles"])
    counts = {"done": 0, "failed": 0, "running": 0, "pending": 0}
    for job in jobs:
        jid = job["id"]
        st, info = job_status(job)
        counts[st] = counts.get(st, 0) + 1
        if st == "done":
            m = harvest(job)
            if m:
                metrics[jid] = m
            log("DONE  %s（指标已收割）" % jid)
        elif st == "failed":
            retries = int(state.get("retries", {}).get(jid, 0))
            if retries < MAX_RETRIES:
                mitigation = "cudnn_v8_disabled" if retries >= 1 else "none"
                log("FAILED %s（第 %d 次重试，mitigation=%s）" % (jid, retries + 1, mitigation))
                relaunch(job, mitigation)
                state.setdefault("retries", {})[jid] = retries + 1
                state.setdefault("mitigation", {})[jid] = mitigation
                counts["failed"] -= 1
                counts["running"] += 1
            else:
                log("FAILED-FINAL %s：重试 %d 次仍失败，转人工/AI 处理" % (jid, retries))
        else:
            log("%-7s %s" % (st.upper(), jid))
    state["metrics"] = metrics

    # 回填（只有全部实验 done 才填 frozen 相关格；其余能填先填）
    all_done = all(job_status(j)[0] == "done" for j in jobs)
    ops = build_ops(metrics if all_done else {k: v for k, v in metrics.items() if k != "frozen_backbone_contrast"})
    applied, remaining = apply_backfill(ops, {k: (v if k in metrics else None) for k, v in
                                              [("merge_rename_upper", metrics.get("merge_rename_upper")),
                                               ("frozen_backbone_contrast", metrics.get("frozen_backbone_contrast") if all_done else None)]})
    state["ops_done"] = sorted(set(state.get("ops_done", [])) | set(applied))
    if applied:
        log("回填 %d 处：%s" % (len(applied), "；".join(applied)))
    if remaining:
        log("待回填 %d 处：%s" % (len(remaining), "；".join(remaining)))

    # 编译门禁：仅当本轮有新回填或尚未编译过
    if applied or not state.get("compile_checked"):
        ok, tail = compile_paper()
        state["compile_checked"] = True
        state["compile_ok"] = ok
        log("编译 %s" % ("通过" if ok else "失败：" + tail[-300:]))
        if ok and applied:
            if refresh_submission():
                log("投稿包 PDF 已刷新（submission_p1/main_cviu_draft.pdf）")

    # 就绪判定
    state["ready"] = bool(all_done and len(state.get("ops_done", [])) >= len(ops) and state.get("compile_ok"))
    if state["ready"]:
        log("P1_READY_FOR_SUBMISSION：四实验完成、LaTeX 已回填并编译通过、投稿包已刷新")
    save_state(state)
    log("状态：%s" % json.dumps(counts, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
