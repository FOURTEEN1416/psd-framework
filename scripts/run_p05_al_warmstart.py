"""P0.5-AL warm-start 协议实验入口 — W23 窗口.

预注册: docs/superpowers/plans/2026-08-25-w23-warmstart-al.md（协议字段禁止运行时更改）
用户裁决: A2（warm-start + 加噪偏移），2026-08-24

执行链:
    1. 合成层 AL 模拟(warm-start): 熵 vs 随机 × 预算 {20,50,100,200} × 3 seeds → val_acc 曲线
       —— 每预算点从 best.pt 加载权重微调；熵打分器 = 上一累计预算的域内微调模型
          （禁止原始 best.pt 直接跨域打分）
    2. 汇总 JSON 归档 reports/（含 per-seed 明细与 mean±std 误差棒）

与 W14 入口(run_p05_al_efficiency.py)分离的理由见预注册 §4:
    ① W18 排队任务零风险隔离 ② warm 协议必填项独立可审计 ③ 可比性声明物理保障

用法:
    # 冒烟（tiny 参数端到端验证管线，~1 分钟）
    python scripts/run_p05_al_warmstart.py --smoke

    # 50 epoch 短预算全量扫描（CPU，断点续跑粒度=单条轨迹）
    python scripts/run_p05_al_warmstart.py --config configs/p05_al_warmstart_short.yaml

    # full-budget 一键复跑（GPU 空闲后由 W18 队列执行）
    python scripts/run_p05_al_warmstart.py --config configs/p05_al_warmstart_full.yaml --fresh
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.synth_stgcn import make_synthetic_dataset  # noqa: E402
from psd.models.stgcn_bc import build_stgcn_bc  # noqa: E402
from psd.training.active_learning import ALSimulationRunner  # noqa: E402
from psd.training.train_stgcn_bc import TrainConfig  # noqa: E402

SMOKE_OVERRIDES = {
    "budgets": [4, 8],
    "seeds": [42],
    "pool_spc": 2,
    "val_spc": 1,
    "base_channels": 8,
    "num_stages": 2,
    "epochs": 2,
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cfg(cfg: dict, smoke: bool) -> dict:
    """应用冒烟覆盖（不改动配置文件本身）。"""
    cfg = json.loads(json.dumps(cfg))  # deep copy
    if not smoke:
        return cfg
    exp = cfg["experiment"]
    exp["budgets"] = SMOKE_OVERRIDES["budgets"]
    exp["seeds"] = SMOKE_OVERRIDES["seeds"]
    exp["name"] += "_smoke"
    cfg["pool"]["samples_per_class"] = SMOKE_OVERRIDES["pool_spc"]
    cfg["val"]["samples_per_class"] = SMOKE_OVERRIDES["val_spc"]
    cfg["model"].update(base_channels=SMOKE_OVERRIDES["base_channels"],
                        num_stages=SMOKE_OVERRIDES["num_stages"])
    cfg["train"].update(epochs=SMOKE_OVERRIDES["epochs"], device="cpu",
                        output_dir="runs/_tmp_al_smoke")
    cfg["output"]["report_json"] = "reports/p05-al-efficiency-warmstart-smoke.json"
    cfg["output"]["state_dir"] = "runs/_tmp_al_smoke/state_warmstart"
    # 冒烟用 tiny 随机权重代替真实 best.pt（架构不同无法加载）
    tiny = build_stgcn_bc(in_channels=3, num_classes=22,
                          base_channels=SMOKE_OVERRIDES["base_channels"],
                          num_stages=SMOKE_OVERRIDES["num_stages"])
    cfg["_smoke_init_sd"] = {k: v.tolist() for k, v in tiny.state_dict().items()}
    return cfg


def trajectory_state_path(state_dir: Path, strategy: str, seed: int) -> Path:
    return state_dir / f"state_{strategy}_{seed}.json"


def load_or_init_state(path: Path) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {"complete": False, "metrics": {}, "selections": {}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="忽略已有状态重跑全部轨迹")
    args = ap.parse_args()

    t0 = time.time()
    config_path = args.config or str(REPO_ROOT / "configs/p05_al_warmstart_short.yaml")
    with open(config_path, encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    cfg = build_cfg(raw_cfg, args.smoke)

    exp = cfg["experiment"]
    print("=" * 68)
    print(f"P0.5-AL warm-start 实验 [{exp['name']}] 层口径={exp['layer']} "
          f"(git {git_sha()}, {datetime.now().isoformat(timespec='seconds')})")
    print("=" * 68)

    device = str(cfg["train"].get("device", "cpu"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(exp["seeds"][0])

    # ---- 偏移数据（诊断选定 noise_std，新种子隔离生成） --------------------
    pool = make_synthetic_dataset(
        samples_per_class=cfg["pool"]["samples_per_class"], T=cfg["pool"]["T"],
        noise_std=cfg["pool"]["noise_std"], seed=cfg["pool"]["seed"],
    )
    val = make_synthetic_dataset(
        samples_per_class=cfg["val"]["samples_per_class"], T=cfg["val"]["T"],
        noise_std=cfg["val"]["noise_std"], seed=cfg["val"]["seed"],
    )
    assert len(pool) >= max(exp["budgets"]), f"池容量 {len(pool)} < 最大预算"
    print(f"[data] pool={len(pool)}(noise={cfg['pool']['noise_std']},s{cfg['pool']['seed']}) "
          f"val={len(val)}(noise={cfg['val']['noise_std']},s{cfg['val']['seed']}) "
          f"budgets={exp['budgets']} seeds={exp['seeds']}")

    def build_model():
        m = build_stgcn_bc(**{k: v for k, v in cfg["model"].items()})
        return m

    # ---- warm 初始化权重 ---------------------------------------------------
    if args.smoke:
        init_sd = {k: torch.tensor(v) for k, v in cfg.pop("_smoke_init_sd").items()}
        init_note = "tiny 随机权重（冒烟专用，架构匹配）"
    else:
        ckpt_path = REPO_ROOT / cfg["warmstart"]["ckpt"]
        assert ckpt_path.exists(), f"warm 初始化 ckpt 不存在: {ckpt_path}"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        init_sd = ckpt["model_state_dict"]
        init_note = (f"epoch={ckpt.get('epoch')} train域best_val_acc="
                     f"{ckpt.get('best_val_acc')} sha256前12={sha256_file(ckpt_path)[:12]}")
    print(f"[warm] init_from_ckpt: {init_note}")

    template_tc = TrainConfig(**{
        k: v for k, v in cfg["train"].items()
        if k in TrainConfig.__dataclass_fields__ and k != "device"
    })
    template_tc.use_amp = bool(cfg["train"].get("use_amp", False)) and device == "cuda"

    state_dir = REPO_ROOT / cfg["output"]["state_dir"]
    state_dir.mkdir(parents=True, exist_ok=True)

    # ---- 主曲线: 策略 × seed 轨迹 -----------------------------------------
    curves: dict = {s: {} for s in exp["strategies"]}
    for strategy in exp["strategies"]:
        for seed in exp["seeds"]:
            spath = trajectory_state_path(state_dir, strategy, seed)
            st = load_or_init_state(spath)
            if st.get("complete") and not args.fresh:
                curves[strategy][str(seed)] = {int(k): v for k, v in st["metrics"].items()}
                print(f"[skip] {strategy}/s{seed} 已完成（--fresh 重跑）")
                continue
            runner = ALSimulationRunner(
                build_model=build_model, pool_samples=pool, val_samples=val,
                train_config=template_tc, budgets=exp["budgets"], device=device,
                init_from_ckpt=init_sd,
            )
            metrics = runner.run_trajectory(strategy=strategy, seed=seed)
            curves[strategy][str(seed)] = metrics
            st.update(complete=True,
                      metrics={str(k): v for k, v in metrics.items()},
                      selections={str(k): v for k, v in runner._selections.items()})
            with open(spath, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False, indent=2)
            print(f"[done] {strategy}/s{seed}: "
                  + " ".join(f"b{b}={metrics[b]:.4f}" for b in exp["budgets"])
                  + f" ({time.time()-t0:.0f}s)")

    # ---- 汇总（误差棒 = seed 间 std） -------------------------------------
    aggregate = {}
    for strategy in exp["strategies"]:
        aggregate[strategy] = {}
        for b in exp["budgets"]:
            vals = np.array([curves[strategy][str(s)][b] for s in exp["seeds"]], dtype=np.float64)
            aggregate[strategy][str(b)] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "per_seed": {s: float(curves[strategy][str(s)][b]) for s in exp["seeds"]},
            }
    print("\n[curve] mean±std (n_seeds=%d)" % len(exp["seeds"]))
    for b in exp["budgets"]:
        row = " | ".join(
            f"{s}: {aggregate[s][str(b)]['mean']:.4f}±{aggregate[s][str(b)]['std']:.4f}"
            for s in exp["strategies"]
        )
        delta = aggregate["entropy"][str(b)]["mean"] - aggregate["random"][str(b)]["mean"]
        print(f"  budget={b:>3}: {row}  Δ(熵−随机)={delta:+.4f}")

    report = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "window": "W23",
            "git_sha": git_sha(),
            "layer": exp["layer"],
            "layer_note": "合成层口径：GT oracle 来自合成生成器标签（AGENTS.md 三层铁律）；"
                          "结论不得外推至公开真实层或真实 K9 层",
            "preregistration": "docs/superpowers/plans/2026-08-25-w23-warmstart-al.md",
            "protocol": {
                "design": "配对增量式 AL（warm-start 版）：同 seed 两臂共享随机初始核(b=%d)，"
                          "增量由各臂策略以上一累计预算的**域内微调模型**打分选择；"
                          "每预算点从 best.pt 加载权重微调（优化器全新构建）" % exp["budgets"][0],
                "scorer_rule": "熵打分器 = 上一累计预算的域内微调模型，"
                               "禁止原始 best.pt 直接跨域打分",
                "comparability": "与 W14 冷启动曲线绝对数值不可直接互比"
                                 "（分布 0.10 vs 0.05、数据实例均不同）；仅允许定性方向性对比",
                "uncertainty_method": exp["uncertainty_method"],
                "curve_metric": "best_val_acc（22 类 top-1，固定 GT 验证集 n=%d）" % len(val),
                "error_bars": "mean±std across %d seeds" % len(exp["seeds"]),
            },
            "data_fingerprints": {
                "pool_gen": {k: cfg["pool"][k] for k in ("samples_per_class", "T", "noise_std", "seed")},
                "val_gen": {k: cfg["val"][k] for k in ("samples_per_class", "T", "noise_std", "seed")},
                "diagnosis_selected_noise_std": cfg["pool"]["noise_std"],
            },
            "warm_init": {
                "source": init_note,
                "optimizer_state_loaded": False,
            },
            "model": cfg["model"],
            "train": cfg["train"],
            "total_time_sec": round(time.time() - t0, 1),
        },
        "curves": {s: {str(b): aggregate[s][str(b)] for b in exp["budgets"]} for s in exp["strategies"]},
        "curves_per_seed": curves,
    }

    out_json = REPO_ROOT / cfg["output"]["report_json"]
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[output] 报告 JSON -> {out_json}")
    print(f"[time] 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
