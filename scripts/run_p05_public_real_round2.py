# -*- coding: utf-8 -*-
"""W40 数据飞轮效力验证 — 编排入口（预注册配置: configs/public_real_round2.yaml）.

四段式:
    build     从统一池构造 Phase A 适应集（含 DogSet 运动学先验门禁参考）
    adapt     AdaBN 域适应（Phase A，无监督，标签零接触）
    finetune  Phase B 头重训（与 round1 协议逐字段一致；round2 各臂 --init 指向适应后 ckpt）
    report    聚合全部结果 → reports/p05-public-real-round2-<date>.json

round1 基线复跑不经本脚本——直接只读执行既有 scripts/run_p05_public_real_finetune.py，
由 orchestrate 子进程调用，保证基线协议零改动保真。

用法示例:
    python scripts/run_p05_public_real_round2.py --stage build
    python scripts/run_p05_public_real_round2.py --stage adapt --tag r2_full
    python scripts/run_p05_public_real_round2.py --stage finetune \
        --round-name r2_full_seed42 --init runs/public_real_round2/adabn_backbone_r2_full.pt
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import yaml  # noqa: E402

from public_real_round2_lib import (  # noqa: E402
    adabn_adapt,
    build_adaptation_set,
    kinematic_gate_thresholds,
    kinematic_ratio,
    make_train_config,
)
from psd.models.stgcn_bc import STGCNBC  # noqa: E402
from run_c1_decouple import freeze_backbone, load_y_backbone  # noqa: E402
from run_p05_public_real_finetune import (  # noqa: E402  协议同一性: 直接复用基线函数
    load_dataset,
    per_class_val_acc,
)

OUT_DIR = REPO / "runs" / "public_real_round2"
CFG_PATH = REPO / "configs" / "public_real_round2.yaml"
POOL_PATH_DEFAULT = Path(r"D:\Desktop\psd-framework\runs\data_campaign\unified\real_expansion_pool_v1.pkl")
GPU_MEM_LIMIT_MIB = 2600   # 桌面基线适配（relay v2 同款阈值），低于此才允许点火


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_prereg() -> dict:
    return yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))


def gpu_free(limit_mib: int = GPU_MEM_LIMIT_MIB) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        used = int(proc.stdout.strip().splitlines()[0])
    except Exception as exc:  # noqa: BLE001 — 无卡环境放行给 CPU 冒烟
        return True, f"nvidia-smi 不可用({exc})，按 CPU 环境放行"
    return (used < limit_mib), f"GPU 已用 {used}MiB (门限 {limit_mib}MiB)"


# ---------------------------------------------------------------------------
# stage: build
# ---------------------------------------------------------------------------

def stage_build(args: argparse.Namespace) -> Path:
    prereg = load_prereg()
    pool_path = Path(args.pool)
    assert pool_path.exists(), f"统一池不存在: {pool_path}"
    print(f"[build] 统一池(只读): {pool_path} sha256={sha256_of(pool_path)[:16]}…")
    with open(pool_path, "rb") as f:
        pool = pickle_load(f)

    entries = pool["entries"]
    n_total_pool = len(entries)

    # DogSet 运动学先验参考分布（度量域 dims=3, fps=60 实测字段）
    mocap = [e for e in entries if e.get("usage_scope") == "kinematic_prior"]
    ref_ratios = [kinematic_ratio(np.asarray(e["keypoints"]), fps=60.0, dims=3) for e in mocap]
    thresholds = kinematic_gate_thresholds(ref_ratios)
    print(f"[build] DogSet 先验参考: n={len(ref_ratios)} band=({thresholds['lo']:.4f}, {thresholds['hi']:.4f})×{thresholds['factor']}")

    fps_assumptions = {"aptv2_c2_w26": 15.0}
    arrays, metas, report = build_adaptation_set(entries, gate_ref=thresholds,
                                                 fps_assumptions=fps_assumptions)
    print(f"[build] 适应集: n={report['n_total']} counts={report['counts']} "
          f"gate_excluded={report['gate'].get('excluded', [])[:5]}")

    # 变长分桶落盘（T 是唯一合法批维度分组）
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buckets: dict[int, list[int]] = {}
    for i, a in enumerate(arrays):
        buckets.setdefault(int(a.shape[0]), []).append(i)
    npz_payload = {}
    bucket_index = {}
    for t_len, idxs in sorted(buckets.items()):
        npz_payload[f"arr_t{t_len}"] = np.stack([arrays[i] for i in idxs])
        bucket_index[str(t_len)] = idxs
    npz_path = OUT_DIR / "adapt_set_w40.npz"
    np.savez_compressed(npz_path, **npz_payload)

    meta = {
        "schema": "w40.round2.adapt_set_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pool_path": str(pool_path),
        "pool_sha256_16": sha256_of(pool_path)[:16],
        "pool_n_entries": n_total_pool,
        "slot": report["slot"],
        "counts": report["counts"],
        "n_total": len(arrays),
        "bucket_index": bucket_index,
        "metas": metas,
        "gate_report": report["gate"],
        "dogset_reference": {
            "n_sequences": len(ref_ratios),
            "ratios_summary": {
                "min": float(np.min(ref_ratios)), "median": float(np.median(ref_ratios)),
                "max": float(np.max(ref_ratios))},
            "thresholds": thresholds,
            "fps_assumptions_echo": report["fps_assumptions_echo"],
        },
        "prereg_version": prereg["version"],
    }
    meta_path = OUT_DIR / "adapt_set_meta_w40.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build] {npz_path}")
    print(f"[build] {meta_path}")
    return npz_path


def pickle_load(fobj):
    import pickle
    return pickle.load(fobj)


# ---------------------------------------------------------------------------
# stage: adapt
# ---------------------------------------------------------------------------

def _load_adapt_arrays(sources: set[str] | None) -> tuple[list[np.ndarray], dict]:
    meta_path = OUT_DIR / "adapt_set_meta_w40.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    npz = np.load(OUT_DIR / "adapt_set_w40.npz")
    id2src = {m["sample_id"]: m["source_channel"] for m in meta["metas"]}
    id2meta = {m["sample_id"]: m for m in meta["metas"]}
    arrays: list[np.ndarray] = []
    kept_metas: dict = {"metas": [], "dropped_by_filter": 0}
    for key in sorted(npz.files):
        if not key.startswith("arr_t"):
            continue
        bucket = npz[key]
        idxs = meta["bucket_index"][key[5:]]
        for row, global_i in enumerate(idxs):
            sid = meta["metas"][global_i]["sample_id"]
            src = id2src[sid]
            if sources is not None and src not in sources:
                kept_metas["dropped_by_filter"] += 1
                continue
            arrays.append(bucket[row])
            kept_metas["metas"].append(id2meta[sid])
    return arrays, kept_metas


def stage_adapt(args: argparse.Namespace) -> Path:
    prereg = load_prereg()
    fwd = prereg["round2_treatment"]["phase_a_adabn"]["forward"]
    ok, msg = gpu_free()
    if not ok and not args.force:
        raise SystemExit(f"[adapt] GPU 门禁不通过: {msg}（--force 可越权，仅限协调者指令）")
    print(f"[adapt] GPU 门禁: {msg}")

    sources = set(args.sources.split(",")) if args.sources else None
    arrays, info = _load_adapt_arrays(sources)
    if getattr(args, "limit", None):
        arrays = arrays[: int(args.limit)]      # 冒烟专用——正式运行禁止
        print(f"[adapt] ⚠️ SMOKE 截取前 {len(arrays)} 条（产物仅供管线验证）")
    print(f"[adapt] 适应样本 n={len(arrays)} 来源过滤={sorted(sources) if sources else 'ALL'} "
          f"(过滤丢弃 {info['dropped_by_filter']})")

    seed = int(fwd["shuffle_seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = STGCNBC(in_channels=3, num_classes=int(prereg["round1_baseline"]["protocol_echo"]["num_classes"]))
    init_ckpt = REPO / prereg["round1_baseline"]["init_ckpt"]
    load_info = load_y_backbone(model, init_ckpt)
    print(f"[adapt] warm-init ← {init_ckpt.name} missing={len(load_info.missing)} unexpected={len(load_info.unexpected)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    summary = adabn_adapt(model, arrays,
                          batch_size=int(fwd["batch_size"]),
                          seed=seed,
                          passes=int(fwd.get("passes", 1)),
                          device=device)
    summary["wall_clock_sec"] = round(time.time() - t0, 1)
    summary["device"] = device
    summary["sources_filter"] = sorted(sources) if sources else "ALL"
    summary["n_bn_moved"] = int(sum(summary.pop("per_bn_moved")))
    summary["dropped_by_source_filter"] = info["dropped_by_filter"]

    tag = args.tag
    out_path = OUT_DIR / f"adabn_backbone_{tag}.pt"
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "adabn_summary": summary,
        "init_from": str(init_ckpt),
        "prereg_version": prereg["version"],
    }, out_path)
    (OUT_DIR / f"adabn_stats_{tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[adapt] {out_path}")
    print(f"[adapt] BN 移动数 {summary['n_bn_moved']}/{summary['n_bn_modules']} "
          f"前向样本 {summary['n_forward_samples']} 用时 {summary['wall_clock_sec']}s")
    return out_path


# ---------------------------------------------------------------------------
# stage: finetune（Phase B——协议与 round1 逐字段一致）
# ---------------------------------------------------------------------------

def stage_finetune(args: argparse.Namespace) -> Path:
    prereg = load_prereg()
    base = prereg["round1_baseline"]
    echo = dict(base["protocol_echo"])
    ok, msg = gpu_free()
    if not ok and not args.force:
        raise SystemExit(f"[finetune] GPU 门禁不通过: {msg}（--force 可越权，仅限协调者指令）")
    print(f"[finetune] GPU 门禁: {msg}")

    seed = args.seed if args.seed is not None else int(echo["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_pkl = Path(args.data_pkl) if args.data_pkl else Path(base["data_pkl"])
    train_s, val_s, dist = load_dataset(data_pkl)
    print(f"[finetune] {args.round_name} seed={seed} data={data_pkl.name} "
          f"train={len(train_s)} val={len(val_s)} dist={dist}")

    model = STGCNBC(in_channels=3, num_classes=int(echo["num_classes"]))
    init_path = Path(args.init)
    info = load_y_backbone(model, init_path)
    print(f"[finetune] init←{init_path} missing(head新建)={len(info.missing)} unexpected={len(info.unexpected)}")
    freeze_backbone(model)

    tc = make_train_config(echo)
    tc.seed = seed                                   # 诊断臂换种子时显式覆盖并回显
    tc.device = args.device
    tc.use_amp = bool(echo["use_amp"]) and tc.device != "cpu"
    if getattr(args, "smoke_epochs", None):
        tc.epochs = int(args.smoke_epochs)           # 冒烟专用——正式运行协议回显为准
        tc.patience = max(2, int(args.smoke_epochs))
        print(f"[finetune] ⚠️ SMOKE 预算 epochs={tc.epochs}（结果落 smoke 区，不入报告聚合）")

    smoke = bool(getattr(args, "smoke_epochs", None))
    tc.output_dir = str(OUT_DIR / ("smoke" if smoke else f"train_{args.round_name}"))

    from psd.training.train_stgcn_bc import STGCNBCTrainer
    trainer = STGCNBCTrainer(model, train_s, val_s, config=tc)
    fit_res = trainer.fit()

    dev = tc.device if tc.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    per_class = per_class_val_acc(model, val_s, dev)

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "round_name": args.round_name,
        "layer": "public_real",
        "protocol": "AK partialclass4 frozen-backbone head-retrain (W40 round2 harness, 协议回显见 prereg)",
        "config_echo": {
            "pkl": str(data_pkl), "init": str(init_path),
            "num_classes": echo["num_classes"], "epochs": echo["epochs"],
            "batch_size": echo["batch_size"], "patience": echo["patience"],
            "seed": seed, "use_amp": tc.use_amp, "device": dev,
            "n_train": len(train_s), "n_val": len(val_s), "class_dist": dist,
        },
        "summary": {**fit_res, "per_class_val_acc": per_class},
        "prereg_version": prereg["version"],
    }
    out = (OUT_DIR / ("smoke" if smoke else "results") / f"{args.round_name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[finetune] {out}")
    print(f"[finetune] best_val_acc={fit_res['best_val_acc']:.4f} per_class={per_class}")
    return out


# ---------------------------------------------------------------------------
# stage: orchestrate（round1 复跑走原脚本子进程；其余臂走本脚本）
# ---------------------------------------------------------------------------

ORCHESTRATION = [
    # (kind, round_name, init_tag_or_None, sources_or_None, seed)
    ("r1", "round1_rerun_seed42", None, None, 42),
    ("r2", "round2_full_seed42", "r2_full", None, 42),
    ("r2", "round2_w35only_seed42", "r2_w35only", "video_c1_w35", 42),
    ("r2", "round2_aptv2only_seed42", "r2_aptv2only", "aptv2_c2_w26", 42),
    ("r1", "round1_rerun_seed43", None, None, 43),
    ("r2", "round2_full_seed43", "r2_full", None, 43),
    ("r1", "round1_rerun_seed44", None, None, 44),
    ("r2", "round2_full_seed44", "r2_full", None, 44),
]


def stage_orchestrate(args: argparse.Namespace) -> None:
    py = sys.executable
    orig_script = REPO / "scripts" / "run_p05_public_real_finetune.py"
    force_extra = ["--force"] if args.force else []
    for kind, name, init_tag, sources, seed in ORCHESTRATION:
        out_json = OUT_DIR / "results" / f"{name}.json"
        if out_json.exists() and not args.force:
            print(f"[orch] 跳过已完成: {name}")
            continue
        if kind == "r1":
            cmd = [py, str(orig_script),
                   "--pkl", load_prereg()["round1_baseline"]["data_pkl"],
                   "--init", load_prereg()["round1_baseline"]["init_ckpt"],
                   "--seed", str(seed),
                   "--output-dir", str((OUT_DIR / f"train_{name}").relative_to(REPO)),
                   "--output-json", str(out_json.relative_to(REPO))]
            print(f"[orch] round1 复跑（原脚本只读执行）: {name}")
        else:
            # 组件臂需要各自的适应产物
            subprocess.run([py, str(Path(__file__)), "--stage", "adapt",
                            "--tag", init_tag, "--sources", sources or ""] + force_extra,
                           check=True)
            cmd = [py, str(Path(__file__)), "--stage", "finetune",
                   "--round-name", name,
                   "--init", str(OUT_DIR / f"adabn_backbone_{init_tag}.pt"),
                   "--seed", str(seed)] + force_extra
            print(f"[orch] round2 臂: {name} (sources={sources or 'ALL'})")
        subprocess.run(cmd, check=True, cwd=str(REPO))
    print("[orch] 全部臂完成 → --stage report")


# ---------------------------------------------------------------------------
# stage: report
# ---------------------------------------------------------------------------

def stage_report(args: argparse.Namespace) -> Path:
    prereg = load_prereg()
    base = prereg["round1_baseline"]
    results_dir = OUT_DIR / "results"
    runs = {}
    for p in sorted(results_dir.glob("*.json")):
        if p.stem.startswith("smoke"):
            continue                                  # 冒烟产物永不入报告
        runs[p.stem] = json.loads(p.read_text(encoding="utf-8"))
    assert runs, "没有任何结果——先跑 orchestrate"

    def acc(name): return runs[name]["summary"]["best_val_acc"]

    arch = float(base["archived_best_val_acc"])
    r1 = acc("round1_rerun_seed42")
    r2 = acc("round2_full_seed42")

    def interp(delta_pp: float, pc_r1: dict, pc_r2: dict) -> str:
        minority_worse = any(pc_r2[k] < pc_r1[k] - 1e-9 for k in ("jump", "stay"))
        if delta_pp >= 2.0 and not minority_worse:
            return "flywheel_positive"
        if abs(delta_pp) < 2.0:
            return "flywheel_neutral"
        return "flywheel_negative"

    diag = {}
    for seed in (43, 44):
        k1, k2 = f"round1_rerun_seed{seed}", f"round2_full_seed{seed}"
        if k1 in runs and k2 in runs:
            diag[f"seed{seed}"] = {"round1": acc(k1), "round2": acc(k2),
                                   "delta_pp": round((acc(k2) - acc(k1)) * 100, 2)}
    comp = {}
    for arm in ("round2_w35only_seed42", "round2_aptv2only_seed42"):
        if arm in runs:
            comp[arm] = {"best_val_acc": acc(arm),
                         "per_class": runs[arm]["summary"]["per_class_val_acc"]}

    delta_vs_archived_pp = round((r2 - arch) * 100, 2)
    delta_vs_rerun_pp = round((r2 - r1) * 100, 2)
    primary_interp = interp(delta_vs_rerun_pp,
                            runs["round1_rerun_seed42"]["summary"]["per_class_val_acc"],
                            runs["round2_full_seed42"]["summary"]["per_class_val_acc"])

    adapt_meta_path = OUT_DIR / "adapt_set_meta_w40.json"
    adapt_summary = {}
    if adapt_meta_path.exists():
        am = json.loads(adapt_meta_path.read_text(encoding="utf-8"))
        adapt_summary = {k: am[k] for k in
                         ("counts", "n_total", "gate_report", "dogset_reference", "pool_sha256_16")
                         if k in am}

    payload = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "W40 数据飞轮效力验证——round1 复跑 vs round2(+pretrain_geometric 槽位 AdaBN 增强)，同 seed 同协议",
        "prereg": {"config": "configs/public_real_round2.yaml", "version": prereg["version"],
                   "prereg_commit_note": "预注册先于实现（commit 996e9b6）"},
        "round1_baseline": {"archived_best_val_acc": arch,
                            "archived_source": base["archived_result"],
                            "rerun_best_val_acc": r1,
                            "rerun_per_class": runs["round1_rerun_seed42"]["summary"]["per_class_val_acc"],
                            "archived_vs_rerun_pp": round((r1 - arch) * 100, 2)},
        "round2_primary": {"best_val_acc": r2,
                           "per_class": runs["round2_full_seed42"]["summary"]["per_class_val_acc"],
                           "delta_vs_archived_pp": delta_vs_archived_pp,
                           "delta_vs_rerun_pp": delta_vs_rerun_pp,
                           "interpretation": primary_interp,
                           "verdict_text": {
                               "flywheel_positive": "数据飞轮持续供数主张获首个直接实证",
                               "flywheel_neutral": "供数未转化为精度——转化链条断点分析见报告 md",
                               "flywheel_negative": "扩展池增强反而有害——域差瓶颈分析见报告 md"}[primary_interp]},
        "diagnostics": {"seeds": diag, "component_arms": comp},
        "adapt_set_summary": adapt_summary,
        "all_runs": {k: {"best_val_acc": v["summary"]["best_val_acc"],
                          "per_class_val_acc": v["summary"].get("per_class_val_acc"),
                          "total_epochs_trained": v["summary"]["total_epochs_trained"],
                          "best_epoch": v["summary"]["best_epoch"]}
                     for k, v in runs.items()},
    }
    date_tag = args.date_tag or datetime.now().strftime("%Y-%m-%d")
    out = REPO / "reports" / f"p05-public-real-round2-{date_tag}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {out}")
    print(f"[report] 主对照: round1复跑={r1:.4f} round2={r2:.4f} Δ={delta_vs_rerun_pp:+.2f}pp "
          f"(vs 存档 {delta_vs_archived_pp:+.2f}pp) → {primary_interp}")
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True,
                    choices=["build", "adapt", "finetune", "orchestrate", "report"])
    ap.add_argument("--pool", default=str(POOL_PATH_DEFAULT))
    ap.add_argument("--tag", default="r2_full", help="adapt 产物命名（r2_full/r2_w35only/r2_aptv2only）")
    ap.add_argument("--sources", default="", help="adapt 来源过滤（逗号分隔，空=全量）")
    ap.add_argument("--round-name", default="")
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--data-pkl", default="")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--date-tag", default="")
    ap.add_argument("--limit", type=int, default=None,
                    help="adapt 冒烟专用截取（正式运行禁止使用）")
    ap.add_argument("--smoke-epochs", type=int, default=None,
                    help="finetune 冒烟专用预算（结果落 smoke 区不入报告）")
    args = ap.parse_args()

    if args.stage == "build":
        stage_build(args)
    elif args.stage == "adapt":
        stage_adapt(args)
    elif args.stage == "finetune":
        assert args.round_name and args.init, "--round-name 与 --init 必填"
        stage_finetune(args)
    elif args.stage == "orchestrate":
        stage_orchestrate(args)
    elif args.stage == "report":
        stage_report(args)


if __name__ == "__main__":
    main()
