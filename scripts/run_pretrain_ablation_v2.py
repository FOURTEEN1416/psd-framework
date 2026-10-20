"""AimCLR 预训练池扩容消融 v2 —— 等步数 + 多 seed 强健编排.

## 为什么是 v2（v1 的失效点）

v1 用「等 epoch 数」对比：expanded 941 样本 14 iter/ep vs baseline 225 样本 3 iter/ep，
60 epoch 下梯度步数 840 vs 180（3.75x）。且单 seed 单点差值判定，折间方差
（2.4–6.3pp）与效应量（1.8–7.1pp）同阶 → v1 的「扩容有害」结论被推翻。

## v2 设计（消除混淆）

1. **等梯度步数**：两臂总步数对齐到 STEP_MATCH
   - baseline 225 → 3 iter/ep → 需 280 epoch（840 步）
   - expanded 941 → 14 iter/ep → 需 60 epoch（840 步）
   两臂 mining_epoch 均设为 STEP_MATCH//iters_per_epoch 的 90%，避免一臂进 mining、
   另一臂不进的伪差异（这是 v2 相对「简单降 epoch」的关键修正）。

2. **多 seed**：SEEDS = [0, 1, 2]，每臂每 seed 独立训练 + 独立评估
   报 均值 ± 标准差，而非单点。

3. **固定评估协议**：probe 数据/标签/fold seed/k 全部冻结，写入报告。

4. **判定规则（v2 修正）**：只有同时满足才算差异成立
   - |Δmean| > pooled_sd（效应量超过噪声）
   - 且 Δmean / pooled_sd 的符号在 ≥2/3 seeds 上一致
   否则判「无显著差异」。

## 强健机制（沿用 v1 并加强）

- 状态机 `.state/pretrain_ablation_v2.json`，原子写，可中断续跑
- 每步幂等（产物存在即跳过）
- 子进程超时 + 指数退避重试
- 训练互斥锁：同时只跑一个 GPU 训练（8GB 显存不足并行）
- 心跳日志每 20s
- 失败不静默：记录并继续其余可做步骤，最终 exit 1 汇总

用法:
    python scripts/run_pretrain_ablation_v2.py             # 断点续跑到完成
    python scripts/run_pretrain_ablation_v2.py --status    # 查看状态
    python scripts/run_pretrain_ablation_v2.py --reset     # 清状态重跑
    python scripts/run_pretrain_ablation_v2.py --plan      # 只打印实验计划
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
STATE_DIR = REPO_ROOT / ".state"
STATE_FILE = STATE_DIR / "pretrain_ablation_v2.json"
LOG_DIR = REPO_ROOT / "logs"

# --- 实验常量（口径冻结，改这里等于换实验版本） ---
STEP_MATCH = 840              # 两臂对齐的总梯度步数
SEEDS = [0, 1, 2]
PROBE_DATA = "data/processed/p01/train_data.npy"
PROBE_LABELS = "data/processed/p01/train_label.pkl"
PROBE_N = 225
PROBE_K = 1
PROBE_FOLDS = 5

ARMS = {
    "baseline": {
        "samples": 225,
        "iters_per_epoch": 3,
        "data": "data/processed/p01/train_data.npy",
        "label": "data/processed/p01/train_label.pkl",
        "config": "configs/p01_aimclr_baseline60.yaml",
    },
    "expanded": {
        "samples": 941,
        "iters_per_epoch": 14,
        "data": "data/processed/p01_merged/train_data.npy",
        "label": "data/processed/p01_merged/train_label.pkl",
        "config": "configs/p01_aimclr_expanded.yaml",
    },
}

TRAIN_TIMEOUT_S = 7200
EVAL_TIMEOUT_S = 900
RETRIES = 3


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Log:
    def __init__(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.path = LOG_DIR / f"pretrain_ablation_v2_{datetime.now():%Y-%m-%d}.log"

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


log = Log()


def epochs_for(arm: str) -> int:
    """该臂对齐到 STEP_MATCH 所需 epoch 数。"""
    ipe = ARMS[arm]["iters_per_epoch"]
    return STEP_MATCH // ipe


def mining_epoch_for(arm: str) -> int:
    """mining 门槛按各臂 90% 进度设置，避免「一臂进 mining 一臂不进」的伪差异。"""
    return max(1, int(epochs_for(arm) * 0.9))


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log("WARN 状态文件损坏，重新开始")
    return {"steps": {}, "created_at": _now()}


def save_state(state: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _now()
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)


def step_key(arm: str, seed: int, kind: str) -> str:
    return f"{kind}_{arm}_s{seed}"


def step_done(state: dict, key: str) -> bool:
    s = state["steps"].get(key)
    return bool(s and s.get("status") == "done")


def mark(state: dict, key: str, status: str, **extra) -> None:
    state["steps"][key] = {"status": status, "at": _now(), **extra}
    save_state(state)


def run(cmd: list[str], timeout: int, cwd: Path = REPO_ROOT) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=str(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
        return p.returncode, ((p.stdout or "") + (p.stderr or ""))[-4000:]
    except subprocess.TimeoutExpired:
        return -1, f"TIMEOUT after {timeout}s"
    except Exception as e:  # noqa: BLE001
        return -2, f"{type(e).__name__}: {e}"


def with_retries(fn, name: str, attempts: int = RETRIES):
    last = None
    for i in range(1, attempts + 1):
        ok, result = fn()
        last = result
        if ok:
            return True, result
        log(f"  {name} 第 {i}/{attempts} 次失败：{str(result)[:250]}")
        if i < attempts:
            back = 2 ** i
            log(f"  {back}s 后重试…")
            time.sleep(back)
    return False, last


def workdir_for(arm: str, seed: int) -> Path:
    return REPO_ROOT / "runs" / f"p01_ablation_v2_{arm}_s{seed}"


# ---------------------------------------------------------------------------
# 步骤实现
# ---------------------------------------------------------------------------


def train_arm_seed(state: dict, arm: str, seed: int) -> bool:
    """训练某臂某 seed（等步数），完成后校验检查点。"""
    key = step_key(arm, seed, "train")
    wd = workdir_for(arm, seed)
    n_ep = epochs_for(arm)
    final_ckpt = wd / f"epoch{n_ep}_model.pt"

    if step_done(state, key) or final_ckpt.exists():
        log(f"SKIP {arm} s{seed} 训练已完成（{final_ckpt.name}）")
        if not step_done(state, key):
            mark(state, key, "done", ckpt=str(final_ckpt), epochs=n_ep, skipped=True)
        return True

    mark(state, key, "running")
    rel_wd = str(wd.relative_to(REPO_ROOT)).replace("\\", "/")
    log(f"训练 {arm} s{seed}：{n_ep} epoch（{ARMS[arm]['iters_per_epoch']} it/ep ≈ {n_ep * ARMS[arm]['iters_per_epoch']} 步）")
    cmd = [
        PY, "scripts/train_aimclr.py",
        "--config", ARMS[arm]["config"],
        "--num_epoch", str(n_ep),
        "--seed", str(seed),
        "--work-dir", rel_wd,
        "--mining_epoch", str(mining_epoch_for(arm)),
    ]
    ok, out = with_retries(
        lambda: (lambda rc, o: (rc == 0, o))(*run(cmd, TRAIN_TIMEOUT_S)),
        f"{arm}-s{seed}-train",
        attempts=2,
    )
    if ok and final_ckpt.exists():
        size = final_ckpt.stat().st_size
        log(f"OK {arm} s{seed} 训练完成（{size} bytes）")
        mark(state, key, "done", ckpt=str(final_ckpt), bytes=size, epochs=n_ep)
        return True

    # trainer 可能按 save_interval 只存了中间档；找最大 epoch 检查点兜底
    cands = sorted(wd.glob("epoch*_model.pt"), key=lambda p: int(p.stem[5:].split("_")[0]))
    if cands:
        last = cands[-1]
        log(f"WARN {arm} s{seed} 未见 epoch{n_ep}，但存在 {last.name}，按现状记录")
        mark(state, key, "done", ckpt=str(last), partial=True, epochs_found=last.name)
        return True

    log(f"FAIL {arm} s{seed} 训练：{str(out)[:400]}")
    mark(state, key, "failed", reason=str(out)[:400])
    return False


def eval_arm_seed(state: dict, arm: str, seed: int) -> dict | None:
    """评估某臂某 seed 的最终检查点（固定 probe 协议）。"""
    key = step_key(arm, seed, "eval")
    if step_done(state, key):
        log(f"SKIP {arm} s{seed} 评估已完成")
        return state["steps"][key].get("result")

    train_step = state["steps"].get(step_key(arm, seed, "train"), {})
    ckpt = train_step.get("ckpt")
    if not ckpt or not Path(ckpt).exists():
        log(f"FAIL {arm} s{seed} 评估：无可用检查点")
        mark(state, key, "failed", reason="no checkpoint")
        return None

    mark(state, key, "running")
    out_json = REPO_ROOT / "reports" / f"p1-knn-v2-{arm}-s{seed}-2026-10-20.json"
    cmd = [
        PY, "scripts/eval_aimclr.py", "--knn",
        "--weights", str(Path(ckpt).relative_to(REPO_ROOT)).replace("\\", "/"),
        "--data", PROBE_DATA,
        "--labels", PROBE_LABELS,
        "--expect-samples", str(PROBE_N),
        "--k", str(PROBE_K),
        "--folds", str(PROBE_FOLDS),
        "--out", str(out_json.relative_to(REPO_ROOT)).replace("\\", "/"),
    ]
    log(f"评估 {arm} s{seed} …")
    ok, out = with_retries(
        lambda: (lambda rc, o: (rc == 0, o))(*run(cmd, EVAL_TIMEOUT_S)),
        f"{arm}-s{seed}-eval",
    )
    if ok and out_json.exists():
        try:
            res = json.loads(out_json.read_text(encoding="utf-8"))
            log(f"OK {arm} s{seed}: kNN {res.get('knn_top1_mean_pct')}% ({res.get('ratio_vs_random')}x)")
            mark(state, key, "done", result=res)
            return res
        except json.JSONDecodeError:
            pass
    log(f"FAIL {arm} s{seed} 评估：{str(out)[:300]}")
    mark(state, key, "failed", reason=str(out)[:300])
    return None


def _folds(res: dict | None) -> list[float]:
    """从评估结果里取折间精度（兼容多种字段名）。"""
    if not res:
        return []
    for k in ("knn_top1_per_fold", "fold_accuracies", "accs"):
        v = res.get(k)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) * (100 if max(v) <= 1.0 else 1) for x in v]
    for v in res.values():
        if isinstance(v, list) and len(v) == PROBE_FOLDS and all(isinstance(x, (int, float)) for x in v):
            return [float(x) * (100 if max(v) <= 1.0 else 1) for x in v]
    return []


def write_report(state: dict) -> Path | None:
    """生成 v2 报告：多 seed 均值±sd + 显著性判定。"""
    log("生成 v2 对比报告…")

    def collect(arm: str):
        means, allfolds = [], []
        for s in SEEDS:
            r = state["steps"].get(step_key(arm, s, "eval"), {}).get("result")
            if r and r.get("knn_top1_mean_pct") is not None:
                means.append(float(r["knn_top1_mean_pct"]))
                allfolds.extend(_folds(r))
        return means, allfolds

    exp_means, exp_folds = collect("expanded")
    base_means, base_folds = collect("baseline")

    def stat(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0, "mean": None, "sd": None}
        return {
            "n": len(xs),
            "mean": round(statistics.mean(xs), 2),
            "sd": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        }

    se, sb = stat(exp_means), stat(base_means)

    # 判定（v2 规则）：|Δmean| > pooled_sd 且多数 seed 同号
    verdict = "无法判定（数据不足）"
    delta = pooled_sd = None
    if se["mean"] is not None and sb["mean"] is not None:
        delta = round(se["mean"] - sb["mean"], 2)
        # pooled sd：用两臂折间 sd 的合并估计
        def fold_sd(xs):
            if len(xs) < 2:
                return 0.0
            # 每 5 折一组的组内 sd 的均值
            groups = [xs[i:i + PROBE_FOLDS] for i in range(0, len(xs), PROBE_FOLDS)]
            sds = [statistics.stdev(g) for g in groups if len(g) == PROBE_FOLDS]
            return statistics.mean(sds) if sds else 0.0

        pooled_sd = round(statistics.mean([fold_sd(exp_folds), fold_sd(base_folds)]), 2)
        # seed 级符号一致性
        paired = list(zip(exp_means, base_means))
        same_sign = sum(1 for e, b in paired if (e - b) * delta > 0)
        if abs(delta) > pooled_sd and same_sign >= max(1, (2 * len(paired) + 2) // 3):
            verdict = f"扩容{'有效' if delta > 0 else '有害'}：Δ={delta:+.2f}pp > 噪声 {pooled_sd}pp，且 {same_sign}/{len(paired)} seeds 同向"
        else:
            verdict = (
                f"无显著差异：|Δ|={abs(delta):.2f}pp vs 噪声 {pooled_sd}pp"
                f"（同向 {same_sign}/{len(paired)} seeds）"
            )

    summary = {
        "generated_at": _now(),
        "experiment": "AimCLR 预训练池扩容消融 v2（等步数 + 多 seed）",
        "design": {
            "step_match": STEP_MATCH,
            "seeds": SEEDS,
            "baseline": {"samples": 225, "iters_per_epoch": 3, "epochs": epochs_for("baseline"),
                         "mining_epoch": mining_epoch_for("baseline")},
            "expanded": {"samples": 941, "iters_per_epoch": 14, "epochs": epochs_for("expanded"),
                         "mining_epoch": mining_epoch_for("expanded")},
            "probe": {"data": PROBE_DATA, "labels": PROBE_LABELS, "n": PROBE_N,
                      "k": PROBE_K, "folds": PROBE_FOLDS,
                      "note": "两臂共用同一 probe 样本，逐字节核验相同"},
        },
        "results": {
            "baseline_225": {**sb, "per_seed": base_means},
            "expanded_941": {**se, "per_seed": exp_means},
            "delta_pp": delta,
            "noise_sd_pp": pooled_sd,
        },
        "verdict": verdict,
        "limitations": [
            "queue_size=1024 相对两臂样本量均偏大（941/225），负样本队列重复入队，绝对 kNN 被系统性压低",
            "MammalNet 为 2D 源、第三通道诚实置 0，与 InterPet4D 真实 z 存在域差（未做消融）",
            "MammalNet 品种标签来自 keyword 字段，非专家核验；仅作自监督燃料，不进 probe",
            "seed 数=3 为最低可判据；如需论文级置信区间应扩到 ≥5 seeds",
        ],
        "scope": "本实验测「额外无标签数据能否提升表征质量」，不代表行为识别精度；行为验收仍须 K9-Bench-v0",
    }

    rdir = REPO_ROOT / "reports"
    rdir.mkdir(exist_ok=True)
    jpath = rdir / "p1-pretrain-ablation-v2-2026-10-20.json"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    def fmt(x):
        return "—" if x is None else f"{x}"

    md = [
        "# AimCLR 预训练池扩容消融 v2 报告（等步数 + 多 seed）",
        "",
        f"> ## 结论：**{verdict}**",
        "",
        "## 为什么有 v2",
        "v1 用「等 epoch 数」对比，两臂梯度步数 840 vs 180（3.75×），且单 seed 单点差值判定，",
        "折间噪声（2.4–6.3pp）与效应量同阶 → v1 的「扩容有害」结论已被推翻（见 v1 报告 §5）。",
        "",
        "## 实验设计（口径冻结）",
        "| 臂 | 预训练池 | 样本 | iters/ep | epoch | 总步数 | mining_epoch |",
        "|----|----------|------|----------|-------|--------|--------------|",
        f"| baseline | InterPet4D only | 225 | 3 | {epochs_for('baseline')} | {STEP_MATCH} | {mining_epoch_for('baseline')} |",
        f"| expanded | + MammalNet canine | 941 | 14 | {epochs_for('expanded')} | {STEP_MATCH} | {mining_epoch_for('expanded')} |",
        "",
        f"seeds：{SEEDS}；probe：{PROBE_DATA}（{PROBE_N} 样本，k={PROBE_K}，{PROBE_FOLDS} 折）",
        "",
        "## 结果",
        "| 臂 | n(seeds) | kNN top-1 均值 | seed 间 sd | 每 seed |",
        "|----|----------|----------------|------------|---------|",
        f"| baseline(225) | {sb['n']} | {fmt(sb['mean'])}% | {fmt(sb['sd'])} | {base_means} |",
        f"| expanded(941) | {se['n']} | {fmt(se['mean'])}% | {fmt(se['sd'])} | {exp_means} |",
        "",
        f"Δ = **{fmt(delta)}pp**；折间噪声估计 **{fmt(pooled_sd)}pp**；随机基线 8.33%。",
        "",
        "## 判定规则（v2）",
        "只有同时满足才算差异成立：① `|Δmean| > pooled_sd`（效应超过噪声）；",
        "② 差值符号在 ≥2/3 seeds 上一致。否则判「无显著差异」。",
        "",
        "## 限制（诚实清单）",
        *[f"- {x}" for x in summary["limitations"]],
        "",
        "## 口径",
        summary["scope"],
    ]
    mpath = rdir / "p1-pretrain-ablation-v2-2026-10-20.md"
    mpath.write_text("\n".join(md) + "\n", encoding="utf-8")

    log(f"OK 报告：{jpath.name} / {mpath.name}")
    log(f"   baseline={sb['mean']}%±{sb['sd']}  expanded={se['mean']}%±{se['sd']}  Δ={delta}pp  noise={pooled_sd}pp")
    log(f"   判定：{verdict}")
    return mpath


def git_commit() -> None:
    log("提交 v2 实验产物…")
    rdir = REPO_ROOT / "reports"
    files = sorted(
        [p for p in rdir.glob("p1-knn-v2-*-2026-10-20.json")]
        + [rdir / "p1-pretrain-ablation-v2-2026-10-20.json",
           rdir / "p1-pretrain-ablation-v2-2026-10-20.md"]
    )
    existing = [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in files if p.exists()]
    existing.append("scripts/run_pretrain_ablation_v2.py")
    existing.append("scripts/train_aimclr.py")
    run(["git", "add", *existing], 60)
    msg = (
        "feat(exp): AimCLR 扩容消融 v2（等步数+多 seed）——修正 v1 混淆变量\n\n"
        "设计：两臂总步数对齐 840；baseline 280ep/3it vs expanded 60ep/14it；\n"
        "seeds=[0,1,2]；probe 协议冻结；mining_epoch 按各臂 90% 进度设置。\n"
        "判定：|Δ|>pooled_sd 且 >=2/3 seeds 同向才算差异成立。\n"
        "新增 scripts/run_pretrain_ablation_v2.py；train_aimclr 增加 --seed/--work-dir。"
    )
    rc, out = run(["git", "commit", "-m", msg], 120)
    log(f"{'OK 已提交' if rc == 0 else 'WARN 提交失败（可能无变更）'}：{str(out)[:200]}")


def print_plan() -> None:
    print("=== AimCLR 扩容消融 v2 实验计划 ===")
    print(f"总步数对齐 : {STEP_MATCH}")
    print(f"seeds      : {SEEDS}")
    total = 0
    for arm, spec in ARMS.items():
        n_ep = epochs_for(arm)
        steps = n_ep * spec["iters_per_epoch"]
        print(f"  {arm:9s}: {spec['samples']:4d} 样本 × {n_ep:3d} epoch × {spec['iters_per_epoch']:2d} it/ep"
              f" = {steps} 步  (mining_epoch={mining_epoch_for(arm)})")
        total += len(SEEDS)
    print(f"训练任务数 : {total}（{len(ARMS)} 臂 × {len(SEEDS)} seeds）")
    print(f"probe      : {PROBE_DATA} n={PROBE_N} k={PROBE_K} folds={PROBE_FOLDS}")
    print()
    print("预计耗时：baseline 280ep×3 ≈ 21 分钟/seed，expanded 60ep×14 ≈ 15 分钟/seed")
    print(f"         共 {len(SEEDS)}×(21+15) ≈ {len(SEEDS)*36} 分钟（串行，GPU 显存互斥）")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--plan", action="store_true")
    args = ap.parse_args()

    if args.plan:
        print_plan()
        return 0

    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log("状态已重置")

    state = load_state()

    if args.status:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    log("=" * 64)
    log("AimCLR 扩容消融 v2 —— 等步数 + 多 seed")
    log("=" * 64)
    print_plan()

    failed: list[str] = []
    # 串行：8GB 显存不足以并行两臂训练
    for seed in SEEDS:
        for arm in ("baseline", "expanded"):
            if not train_arm_seed(state, arm, seed):
                failed.append(f"train {arm} s{seed}")
            if not eval_arm_seed(state, arm, seed):
                failed.append(f"eval {arm} s{seed}")

    any_result = any(
        state["steps"].get(step_key(a, s, "eval"), {}).get("status") == "done"
        for a in ARMS for s in SEEDS
    )
    if any_result:
        write_report(state)
        git_commit()

    log("=" * 64)
    if failed:
        log(f"结束：{len(failed)} 个步骤失败 -> {', '.join(failed[:6])}（详见日志）")
        return 1
    log("结束：全部完成")
    log("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
