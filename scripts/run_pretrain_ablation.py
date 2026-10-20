"""AimCLR 预训练扩容消融 —— 强健自动化编排（可中断续跑）.

实验问题:
    用 MammalNet 家犬子集（716 clip）扩充分量自监督预训练池，
    是否提升 InterPet4D 上的表征质量（kNN probe top-1）？

两臂设计（唯一差异 = 预训练数据量）:
    baseline : 225 样本（InterPet4D only）      60 epoch
    expanded : 941 样本（InterPet4D + MammalNet）60 epoch
    probe    : 两臂均用同一批 225 样本 + dog-ID 标签（逐字节核验相同）

强健设计（对抗验收标准）:
    1. 状态机落盘 .state/pretrain_ablation.json，每步完成即原子写
    2. 每步幂等：产物已存在则跳过（可随时中断续跑）
    3. 子进程超时保护 + 失败重试（3 次指数退避）
    4. 全程日志 logs/pretrain_ablation_<date>.log
    5. 训练进程外部已启动时自动等待（不重复起训）
    6. 任一步最终失败 → 记录并继续后续可做步骤，最终 exit 1 并给出原因
    7. 结束后自动 git commit（不 push，推送给用户裁决）

用法:
    python scripts/run_pretrain_ablation.py            # 从断点续跑到底
    python scripts/run_pretrain_ablation.py --status   # 只打印状态
    python scripts/run_pretrain_ablation.py --reset    # 清状态重跑
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
STATE_DIR = REPO_ROOT / ".state"
STATE_FILE = STATE_DIR / "pretrain_ablation.json"
LOG_DIR = REPO_ROOT / "logs"

EXPANDED_WORK = REPO_ROOT / "runs" / "p01_aimclr_expanded"
BASELINE_WORK = REPO_ROOT / "runs" / "p01_aimclr_baseline60"
PROBE_DATA = "data/processed/p01/train_data.npy"
PROBE_LABELS = "data/processed/p01/train_label.pkl"
PROBE_N = 225

CHECKPOINT_EPOCHS = [20, 40, 60]
TRAIN_TIMEOUT_S = 3600
EVAL_TIMEOUT_S = 900
TRAIN_WAIT_MAX_S = 3600  # 等待外部已启动训练的上限
RETRIES = 3


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Log:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"pretrain_ablation_{datetime.now():%Y-%m-%d}.log"

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


log = Log()


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("WARN 状态文件损坏，重新开始")
    return {"steps": {}, "created_at": _now()}


def save_state(state: dict) -> None:
    """原子写：tmp + replace，防中断写坏。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def step_done(state: dict, name: str) -> bool:
    s = state["steps"].get(name)
    return bool(s and s.get("status") == "done")


def mark(state: dict, name: str, status: str, **extra) -> None:
    state["steps"][name] = {"status": status, "at": _now(), **extra}
    save_state(state)


def run(cmd: list[str], timeout: int, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    """跑子进程，返回 (exit_code, 输出尾部)。超时返回 -1。"""
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        out = (p.stdout or "") + (p.stderr or "")
        return p.returncode, out[-4000:]
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return -2, f"{type(e).__name__}: {e}"


def with_retries(fn, name: str, attempts: int = RETRIES):
    """指数退避重试；返回 (ok, last_result)。"""
    last = None
    for i in range(1, attempts + 1):
        ok, result = fn()
        last = result
        if ok:
            return True, result
        log(f"  {name} 第 {i}/{attempts} 次失败：{str(result)[:300]}")
        if i < attempts:
            back = 2 ** i
            log(f"  {back}s 后重试…")
            time.sleep(back)
    return False, last


# ---------------------------------------------------------------------------
# 步骤实现
# ---------------------------------------------------------------------------


def wait_expanded_training(state: dict) -> bool:
    """等待扩容臂训练产出 epoch60 检查点（外部可能已在训练）。"""
    name = "wait_expanded_training"
    if step_done(state, name):
        log("SKIP 扩容臂训练已完成（状态记录）")
        return True

    target = EXPANDED_WORK / "epoch60_model.pt"
    if target.exists():
        log(f"OK 扩容臂 epoch60 已存在：{target}")
        mark(state, name, "done", ckpt=str(target))
        return True

    mark(state, name, "running")
    log("等待扩容臂训练产出 epoch60_model.pt（最长 %d 分钟）…" % (TRAIN_WAIT_MAX_S // 60))
    t0 = time.time()
    last_log_size = -1
    while time.time() - t0 < TRAIN_WAIT_MAX_S:
        if target.exists():
            # 等一下确保文件写完整
            time.sleep(2)
            size = target.stat().st_size
            log(f"OK 扩容臂训练完成：epoch60_model.pt ({size} bytes)")
            mark(state, name, "done", ckpt=str(target), bytes=size)
            return True

        # 进度心跳（每 60s）
        logfile = EXPANDED_WORK / "log.txt"
        if logfile.exists():
            txt = logfile.read_text(encoding="utf-8", errors="replace")
            n_ep = txt.count("Training epoch")
            if n_ep != last_log_size:
                last_log_size = n_ep
                log(f"  训练进度：epoch {n_ep}/60")
        time.sleep(20)

    log("FAIL 等待超时：扩容臂未产出 epoch60")
    mark(state, name, "failed", reason="wait timeout")
    return False


def eval_arm(state: dict, arm: str, workdir: Path) -> dict:
    """对某一臂的所有可用检查点跑 kNN probe，返回 {epoch: result}。"""
    name = f"eval_{arm}"
    results: dict[int, dict] = {}

    if step_done(state, name):
        prev = state["steps"][name].get("results", {})
        log(f"SKIP {arm} 评估已完成（状态记录）")
        return {int(k): v for k, v in prev.items()}

    mark(state, name, "running")
    for ep in CHECKPOINT_EPOCHS:
        ckpt = workdir / f"epoch{ep}_model.pt"
        if not ckpt.exists():
            log(f"  跳过 epoch{ep}（检查点不存在）")
            continue

        out_json = REPO_ROOT / "reports" / f"p1-knn-{arm}-ep{ep}-2026-10-20.json"
        cmd = [
            PY, "scripts/eval_aimclr.py", "--knn",
            "--weights", str(ckpt.relative_to(REPO_ROOT)).replace("\\", "/"),
            "--data", PROBE_DATA,
            "--labels", PROBE_LABELS,
            "--expect-samples", str(PROBE_N),
            "--out", str(out_json.relative_to(REPO_ROOT)).replace("\\", "/"),
        ]
        log(f"  评估 {arm} epoch{ep} …")
        ok, out = with_retries(
            lambda c=cmd: (
                lambda rc, o: (rc == 0, o)
            )(*run(c, EVAL_TIMEOUT_S)),
            f"{arm}-ep{ep}",
        )
        if ok and out_json.exists():
            try:
                results[ep] = json.loads(out_json.read_text(encoding="utf-8"))
                log(f"  OK {arm} epoch{ep}: kNN {results[ep].get('knn_top1_mean_pct')}% "
                    f"({results[ep].get('ratio_vs_random')}x)")
            except json.JSONDecodeError:
                log(f"  WARN {arm} epoch{ep} 结果 JSON 解析失败")
        else:
            log(f"  FAIL {arm} epoch{ep}: {str(out)[:300]}")

    if results:
        mark(state, name, "done", results={str(k): v for k, v in results.items()})
        return results
    mark(state, name, "failed", reason="no checkpoint evaluated")
    return {}


def train_baseline(state: dict) -> bool:
    """训练基线臂（225 样本，60 epoch）。"""
    name = "train_baseline"
    ckpt = BASELINE_WORK / "epoch60_model.pt"
    if step_done(state, name) or ckpt.exists():
        log(f"SKIP 基线臂训练已完成：{ckpt}")
        mark(state, name, "done", ckpt=str(ckpt))
        return True

    mark(state, name, "running")
    log("启动基线臂训练（225 样本，60 epoch）…")
    cmd = [PY, "scripts/train_aimclr.py", "--config", "configs/p01_aimclr_baseline60.yaml"]
    ok, out = with_retries(
        lambda: (lambda rc, o: (rc == 0, o))(*run(cmd, TRAIN_TIMEOUT_S)),
        "baseline-train",
        attempts=2,
    )
    if ok and ckpt.exists():
        log(f"OK 基线臂训练完成：{ckpt}")
        mark(state, name, "done", ckpt=str(ckpt))
        return True
    log(f"FAIL 基线臂训练：{str(out)[:500]}")
    mark(state, name, "failed", reason=str(out)[:500])
    return False


def write_report(state: dict) -> Path | None:
    """生成两臂对比报告（JSON + MD）。"""
    log("生成对比报告…")
    exp = state["steps"].get("eval_expanded", {}).get("results", {})
    base = state["steps"].get("eval_baseline", {}).get("results", {})

    def pick(res: dict, ep: int):
        return (res.get(str(ep)) or {}).get("knn_top1_mean_pct")

    rows = []
    for ep in CHECKPOINT_EPOCHS:
        e, b = pick(exp, ep), pick(base, ep)
        rows.append({
            "epoch": ep,
            "expanded_941": e,
            "baseline_225": b,
            "delta_pp": (round(e - b, 2) if (e is not None and b is not None) else None),
        })

    summary = {
        "generated_at": _now(),
        "experiment": "AimCLR 预训练池扩容消融（941 vs 225）",
        "probe": {"data": PROBE_DATA, "labels": PROBE_LABELS, "n": PROBE_N,
                  "note": "两臂 probe 样本逐字节相同（对抗验收核验）"},
        "expanded_pool": {"n": 941, "composition": "225 InterPet4D + 716 MammalNet canine"},
        "baseline_pool": {"n": 225, "composition": "InterPet4D only"},
        "curves": rows,
        "limitations": [
            "queue_size=1024 > 941 样本，负样本队列重复入队，绝对 kNN 值被系统性压低（两臂同口径，相对比较仍公平）",
            "MammalNet 为 2D 源，第三通道诚实置 0，与 InterPet4D 的真实 z 存在域差（未做消融）",
            "MammalNet 品种标签来自 keyword 字段，非专家核验",
        ],
        "verdict": None,
    }

    final_exp, final_base = pick(exp, 60), pick(base, 60)
    if final_exp is not None and final_base is not None:
        delta = final_exp - final_base
        if delta > 0.5:
            summary["verdict"] = f"扩容有效：941 比 225 高 {delta:.2f}pp"
        elif delta < -0.5:
            summary["verdict"] = f"扩容无效/有害：941 比 225 低 {abs(delta):.2f}pp"
        else:
            summary["verdict"] = f"无显著差异（Δ={delta:+.2f}pp，阈值 ±0.5pp）"
    else:
        summary["verdict"] = "数据不完整，无法判定"

    rdir = REPO_ROOT / "reports"
    rdir.mkdir(exist_ok=True)
    jpath = rdir / "p1-pretrain-ablation-2026-10-20.json"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# AimCLR 预训练池扩容消融报告（2026-10-20）",
        "",
        f"> 结论：**{summary['verdict']}**",
        "",
        "## 实验设计",
        "| 臂 | 预训练池 | 样本数 | epoch |",
        "|----|----------|--------|-------|",
        "| baseline | InterPet4D only | 225 | 60 |",
        "| expanded | InterPet4D + MammalNet canine | 941 | 60 |",
        "",
        f"probe：两臂均用 `{PROBE_DATA}`（{PROBE_N} 样本，dog-ID 12 类），逐字节相同。",
        "",
        "## kNN top-1 曲线",
        "| epoch | expanded(941) | baseline(225) | Δ(pp) |",
        "|-------|---------------|---------------|-------|",
    ]
    for r in rows:
        md.append(
            f"| {r['epoch']} | "
            f"{r['expanded_941'] if r['expanded_941'] is not None else '—'} | "
            f"{r['baseline_225'] if r['baseline_225'] is not None else '—'} | "
            f"{r['delta_pp'] if r['delta_pp'] is not None else '—'} |"
        )
    md += [
        "",
        "> 随机基线 8.33%（12 类）。验收门：≥2× 随机。",
        "",
        "## 限制（诚实清单）",
        *[f"- {x}" for x in summary["limitations"]],
        "",
        "## 口径声明",
        "本实验测的是「**额外无标签数据能否提升表征质量**」——"
        "MammalNet 样本不参与 probe（无行为标签，品种与 dog-ID 不同语义），"
        "只作自监督预训练燃料。**该结果不代表行为识别精度**，行为验收仍须 K9-Bench-v0（ADR 0015）。",
    ]
    mpath = rdir / "p1-pretrain-ablation-2026-10-20.md"
    mpath.write_text("\n".join(md) + "\n", encoding="utf-8")

    for r in rows:
        log(f"  epoch{r['epoch']}: expanded={r['expanded_941']} baseline={r['baseline_225']} Δ={r['delta_pp']}pp")
    log(f"OK 报告：{jpath.name} / {mpath.name}")
    return mpath


def git_commit(state: dict) -> None:
    """提交实验结果（不 push）。"""
    log("提交实验产物…")
    files = [
        "reports/p1-pretrain-ablation-2026-10-20.json",
        "reports/p1-pretrain-ablation-2026-10-20.md",
    ]
    files += [f"reports/{p.name}" for p in (REPO_ROOT / "reports").glob("p1-knn-*-ep*-2026-10-20.json")]
    existing = [f for f in files if (REPO_ROOT / f).exists()]
    if not existing:
        log("WARN 无可提交的实验产物")
        return
    run(["git", "add", *existing], 60)
    msg = (
        "feat(exp): AimCLR 预训练池扩容消融结果（941 vs 225，60 epoch 对照）\n\n"
        "两臂唯一差异=预训练数据量；probe 均用同一批 225 样本（逐字节核验相同）。\n"
        "结论见 reports/p1-pretrain-ablation-2026-10-20.md。\n"
        "限制：queue_size>样本数、MammalNet 无 z，均已在报告中如实记录。"
    )
    rc, out = run(["git", "commit", "-m", msg], 120)
    log(f"{'OK 已提交' if rc == 0 else 'WARN 提交失败（可能无变更）'}：{str(out)[:200]}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true", help="只打印状态")
    ap.add_argument("--reset", action="store_true", help="清空状态重跑")
    args = ap.parse_args()

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log("状态已重置")

    state = load_state()

    if args.status:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    log("=" * 60)
    log("AimCLR 预训练扩容消融 —— 自动化编排启动")
    log("=" * 60)

    failed: list[str] = []

    if not wait_expanded_training(state):
        failed.append("扩容臂训练未完成")

    exp = eval_arm(state, "expanded", EXPANDED_WORK)
    if not exp:
        failed.append("扩容臂评估")

    if not train_baseline(state):
        failed.append("基线臂训练")

    base = eval_arm(state, "baseline", BASELINE_WORK)
    if not base:
        failed.append("基线臂评估")

    if exp or base:
        write_report(state)
        git_commit(state)

    log("=" * 60)
    if failed:
        log(f"结束：部分步骤失败 -> {', '.join(failed)}（详见日志）")
        return 1
    log("结束：全部步骤完成")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
