# -*- coding: utf-8 -*-
"""P1 头对头升级 — 多初始化种子 / 多子集种子 / 多预算矩阵（NTU60 xsub）· v2 子进程隔离版.

v2 变更（2026-11-16 晚）: 首版单进程跑满矩阵时，本机 commit 内存达 90%，
第 3 格起连环 OOM（DefaultCPUAllocator / 页面文件太小 / CUDA unknown error）。
改为**每格独立子进程**执行（--run-one），父进程保持轻量（不 import torch），
单格内存随子进程退出即释放；每格失败自动重试 1 次；断点续跑（summary 已完成格跳过）。

三个方法完全同 harness（同数据 data/ntu60_frame50/xsub、同 40,091/16,487 划分、
同分层抽选公式 k=max(1,round(frac·n_c))）:
  - AimCLR 线性评测（官方 LE recipe, in-house pretext, joint stream）
  - 监督 ST-GCN（16 hidden channels, 0.84M 参数, 端到端 CE）
  - PSD 语义管线（E9 arm-b 同实现: 冻结 256d 特征 + run_selftrain + ScaledLR）

矩阵 20 格:
  A. init 种子方差 @10%/subset42: aimclr/stgcn × init_seed {1,2,3,4}（与已发表 i0 合计 5 值）
  B. 子集重抽稳健性 @10%: subset {43,44} × {psd(3 selftrain seeds), aimclr(i0), stgcn(i0)}
  C. 预算趋势: frac {0.05,0.20} × {psd, aimclr, stgcn}（subset42, i0; PSD 3 seeds）
诚实条款: 全部数字来自实跑；单格失败如实记录 failed 并继续；绝不编造。

用法:
    .venv/Scripts/python.exe -u scripts/p1_sprint/h2h_matrix.py --smoke   # 冒烟（进程内）
    .venv/Scripts/python.exe -u scripts/p1_sprint/h2h_matrix.py           # 父编排（推荐）
    .venv/Scripts/python.exe -u scripts/p1_sprint/h2h_matrix.py --run-one <key>
产出:
    runs/p1_sprint/h2h_matrix/summary.json（增量覆盖; 完成时含 doneMarker H2H_MATRIX_DONE）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
AIMCLR_ROOT = REPO / "external" / "AimCLR"
XSUB = REPO / "data" / "ntu60_frame50" / "xsub"
BUDGET_DIR = REPO / "runs" / "p1_sprint" / "ntu60_budget_data"
OUT_DIR = REPO / "runs" / "p1_sprint" / "h2h_matrix"
SUMMARY = OUT_DIR / "summary.json"
LAST_CELL = OUT_DIR / "last_cell.json"
WEIGHTS = "runs/ntu_phaseB/joint_pretext/epoch300_model.pt"
N_CLASS = 60
PY = str(REPO / ".venv" / "Scripts" / "python.exe")


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def stratified(labels: np.ndarray, seed: int, frac: float) -> np.ndarray:
    """与 E9/published_baseline 逐行同式分层抽选，仅 frac 参数化。返回 bool mask。"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in range(N_CLASS):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * frac)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    return mask


def _fp(idx: np.ndarray) -> str:
    return hashlib.sha256(idx.astype(np.int64).tobytes()).hexdigest()[:16]


def build_budget(frac: float, seed: int) -> dict:
    """生成（或指纹一致时复用）frac/seed 参数化的预算数据对，返回相对路径 dict。
    memmap 花式索引只读所选行页（~30MB），不会整体载入。"""
    tag = f"f{int(round(frac * 100)):02d}_s{seed}"
    out_dir = BUDGET_DIR / tag
    with open(XSUB / "train_label.pkl", "rb") as f:
        names, labels = pickle.load(f)
    labels_arr = np.asarray(labels, dtype=np.int64)
    idx = np.where(stratified(labels_arr, seed, frac))[0]
    meta = {"frac": frac, "seed": seed, "n_subset": int(len(idx)),
            "index_sha256_16": _fp(idx),
            "formula": "per-class k=max(1, round(frac*n_c)), default_rng(seed), E9-identical"}
    pos_out, lab_out, meta_path = (out_dir / "train_position.npy"), (out_dir / "train_label.pkl"), (out_dir / "meta.json")
    if pos_out.exists() and lab_out.exists() and meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("index_sha256_16") == meta["index_sha256_16"]:
            log(f"[budget:{tag}] reuse n={meta['n_subset']} fp={meta['index_sha256_16']}")
            return {"data": f"runs/p1_sprint/ntu60_budget_data/{tag}/train_position.npy",
                    "label": f"runs/p1_sprint/ntu60_budget_data/{tag}/train_label.pkl",
                    "n": meta["n_subset"], "fp": meta["index_sha256_16"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    full_pos = np.load(XSUB / "train_position.npy", mmap_mode="r")
    np.save(pos_out, np.asarray(full_pos[idx]))
    del full_pos
    with open(lab_out, "wb") as f:
        pickle.dump(([names[i] for i in idx], [int(labels_arr[i]) for i in idx]), f)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[budget:{tag}] built n={meta['n_subset']} fp={meta['index_sha256_16']}")
    return {"data": f"runs/p1_sprint/ntu60_budget_data/{tag}/train_position.npy",
            "label": f"runs/p1_sprint/ntu60_budget_data/{tag}/train_label.pkl",
            "n": meta["n_subset"], "fp": meta["index_sha256_16"]}


# ── 单格执行体（子进程内运行；重模块在函数内 import） ──

def aimclr_cell(tag: str, budget: dict, val: dict, init_seed_val: int,
                epochs: int = 100, batch: int = 128, device: int = 0) -> dict:
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))
    import published_baseline_ntu60 as pb  # noqa: E402
    from processor.processor import init_seed  # noqa: E402  (external/AimCLR)
    init_seed(init_seed_val)
    from processor.linear_evaluation import LE_Processor  # noqa: E402

    cfg = pb.linear_cfg(work_dir=f"runs/p1_sprint/h2h_matrix/{tag}", weights=WEIGHTS,
                        train={"data": budget["data"], "label": budget["label"]},
                        val=val, epochs=epochs, batch=batch, eval_interval=5, device=device)
    work_dir = REPO / cfg["work_dir"]
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = work_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    os.chdir(REPO)
    t0 = time.time()
    proc = LE_Processor(["--config", str(cfg_path)])
    proc.start()
    top1 = float(getattr(proc, "best_result", 0.0)) / 100.0
    log(f"[aimclr:{tag}] top1={top1 * 100:.2f}% ({time.time() - t0:.0f}s, init_seed={init_seed_val})")
    return {"top1": round(top1, 4), "init_seed": init_seed_val,
            "wall_clock_sec": round(time.time() - t0, 1)}


def stgcn_cell(tag: str, budget: dict, val: dict, init_seed_val: int,
               epochs: int = 80, batch: int = 128, device: int = 0) -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))
    import types

    import supervised_stgcn_ntu60 as sg  # noqa: E402
    args = types.SimpleNamespace(seed=init_seed_val, device=device, epochs=epochs, batch_size=batch,
                                 lr=0.1, steps=[50, 70], wd=1e-4, hidden_channels=16,
                                 eval_interval=5)
    r = sg.train_budget(tag, {"data": budget["data"], "label": budget["label"]}, val,
                        budget["n"], args, "runs/p1_sprint/h2h_matrix")
    log(f"[stgcn:{tag}] top1={r['top1'] * 100:.2f}% (init_seed={init_seed_val})")
    return {"top1": r["top1"], "best_epoch": r["best_epoch"], "init_seed": init_seed_val,
            "n_params": r["n_params"]}


def psd_cell(subset_seed: int, frac: float, selftrain_seeds=(42, 43, 44)) -> dict:
    """PSD 语义管线（E9 arm-b 同实现）: 冻结特征 + run_selftrain + ScaledLR 终头。"""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    import run_p14_ntu_lowres as p14  # noqa: E402
    from psd.training.tcl_selftrain import run_selftrain  # noqa: E402
    from run_p07_endtoend_ak import evaluate  # noqa: E402

    z = np.load(p14.NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    labels_str = np.array([str(int(v)) for v in tr_lab])
    anchor = stratified(tr_lab, subset_seed, frac)
    universe = ~anchor
    rows = []
    for s in selftrain_seeds:
        t0 = time.time()
        r = run_selftrain(tr_feat, labels_str, anchor, run_seed=s,
                          class_names=p14.CLASS_NAMES, head_cfg=p14.HEAD_CFG,
                          pool_universe_mask=universe, **p14.KW)
        train_mask = anchor.copy()
        train_mask[r["final_pool_idx"]] = True
        clf = p14.train_linear_head_scaled(tr_feat, tr_lab, train_mask, N_CLASS)
        ev = evaluate(clf, va_feat, va_lab, np.ones(len(va_lab), bool), p14.CLASS_NAMES)
        rows.append({"seed": s, "n_pool": int(len(r["final_pool_idx"])),
                     "rounds": len(r["rounds"]), "stop": r["stop_reason"],
                     "wall_clock_sec": round(time.time() - t0, 1), **ev})
        log(f"[psd:f{int(round(frac * 100)):02d}_s{subset_seed}] seed{s} top1={ev['top1'] * 100:.2f}% "
            f"pool={rows[-1]['n_pool']} rounds={rows[-1]['rounds']} stop={rows[-1]['stop']}")
    tops = [x["top1"] for x in rows]
    return {"frac": frac, "subset_seed": subset_seed, "selftrain_seeds": list(selftrain_seeds),
            "per_seed": rows, "mean_top1": round(float(np.mean(tops)), 4),
            "std_top1": round(float(np.std(tops, ddof=1)), 4) if len(tops) > 1 else 0.0,
            "n_anchor": int(anchor.sum())}


# ── 格注册表（顺序 = 执行顺序；预算文件由父进程预建，键见 BUDGET_KEYS） ──

BUDGET_KEYS = {"10_s42": (0.10, 42), "10_s43": (0.10, 43), "10_s44": (0.10, 44),
               "05_s42": (0.05, 42), "20_s42": (0.20, 42)}


def cell_specs() -> list[dict]:
    specs = []
    for s in (1, 2, 3, 4):
        specs.append({"key": f"aimclr_10_s42_i{s}", "kind": "aimclr",
                      "budget": "10_s42", "init_seed": s})
    for s in (1, 2, 3, 4):
        specs.append({"key": f"stgcn_10_s42_i{s}", "kind": "stgcn",
                      "budget": "10_s42", "init_seed": s})
    for sub in (43, 44):
        specs.append({"key": f"psd_10_s{sub}", "kind": "psd", "budget": f"10_s{sub}",
                      "subset_seed": sub, "frac": 0.10})
        specs.append({"key": f"aimclr_10_s{sub}_i0", "kind": "aimclr",
                      "budget": f"10_s{sub}", "init_seed": 0})
        specs.append({"key": f"stgcn_10_s{sub}_i0", "kind": "stgcn",
                      "budget": f"10_s{sub}", "init_seed": 0})
    for fk, frac in (("05", 0.05), ("20", 0.20)):
        specs.append({"key": f"psd_f{fk}_s42", "kind": "psd", "budget": f"{fk}_s42",
                      "subset_seed": 42, "frac": frac})
        specs.append({"key": f"aimclr_f{fk}_s42_i0", "kind": "aimclr",
                      "budget": f"{fk}_s42", "init_seed": 0})
        specs.append({"key": f"stgcn_f{fk}_s42_i0", "kind": "stgcn",
                      "budget": f"{fk}_s42", "init_seed": 0})
    return specs


def run_one(key: str) -> None:
    """子进程入口：执行单格并把结果写 LAST_CELL（父进程回收合并）。"""
    spec = next((s for s in cell_specs() if s["key"] == key), None)
    if spec is None:
        raise SystemExit(f"[run-one] 未知格: {key}")
    frac, seed = BUDGET_KEYS[spec["budget"]]
    budget = build_budget(frac, seed)
    val = {"data": "data/ntu60_frame50/xsub/val_position.npy",
           "label": "data/ntu60_frame50/xsub/val_label.pkl"}
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        if spec["kind"] == "aimclr":
            res = aimclr_cell(key, budget, val, spec["init_seed"])
        elif spec["kind"] == "stgcn":
            res = stgcn_cell(key, budget, val, spec["init_seed"])
        else:
            res = psd_cell(spec["subset_seed"], spec["frac"])
        LAST_CELL.write_text(json.dumps({"key": key, "ok": True, "result": res},
                                         ensure_ascii=False), encoding="utf-8")
    except Exception as e:  # 单格失败如实上抛给父进程记录
        LAST_CELL.write_text(json.dumps({"key": key, "ok": False, "error": repr(e)[:500]},
                                         ensure_ascii=False), encoding="utf-8")
        raise


def save_summary(state: dict) -> None:
    SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def base_state(mode: str) -> dict:
    state = {
        "experiment": "p1_sprint_h2h_matrix",
        "mode": mode,
        "date_start": datetime.now().isoformat(timespec="seconds"),
        "harness": {"data": "data/ntu60_frame50/xsub", "n_train": 40091, "n_val": 16487,
                    "weights": WEIGHTS, "runner": "v2 per-cell subprocess isolation",
                    "budget_rule": "per-class k=max(1, round(frac*n_c)), default_rng(subset_seed)"},
        "budgets": {},
        "cells": {},
        "failed": {},
    }
    if SUMMARY.exists():
        old = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if old.get("mode") == mode:
            for k in ("cells", "failed", "budgets", "harness"):
                if old.get(k):
                    state[k] = old[k]
            log(f"[resume] 载入已完成 {len(state['cells'])} 格 / 失败 {len(state['failed'])} 格")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="缩样管线验证（数字无意义，进程内）")
    ap.add_argument("--run-one", metavar="KEY", help="子进程模式：执行单格并写 last_cell.json")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.run_one:
        run_one(args.run_one)
        return

    if args.smoke:
        import run_p14_ntu_lowres as _p14  # noqa: F401  冒烟仅验管线
        val = {"data": "data/ntu60_frame50/xsub/val_position.npy",
               "label": "data/ntu60_frame50/xsub/val_label.pkl"}
        b10 = build_budget(0.10, 42)
        kw_small = dict(_p14.KW, max_iters=1, converge_change_rate=1.0)
        _p14.KW.clear(); _p14.KW.update(kw_small)
        state = base_state("smoke")
        try:
            state["cells"]["smoke_aimclr"] = aimclr_cell("smoke_aimclr", b10, val, 0, epochs=2, batch=64)
        except Exception as e:
            state["failed"]["smoke_aimclr"] = repr(e)[:500]
        try:
            state["cells"]["smoke_stgcn"] = stgcn_cell("smoke_stgcn", b10, val, 0, epochs=2, batch=64)
        except Exception as e:
            state["failed"]["smoke_stgcn"] = repr(e)[:500]
        try:
            state["cells"]["smoke_psd"] = psd_cell(42, 0.10, selftrain_seeds=(42,))
        except Exception as e:
            state["failed"]["smoke_psd"] = repr(e)[:500]
        state["doneMarker"] = "H2H_MATRIX_SMOKE_OK"
        state["date_end"] = datetime.now().isoformat(timespec="seconds")
        save_summary(state)
        log(f"[done-smoke] cells={len(state['cells'])} failed={len(state['failed'])}")
        return

    # ── 父编排：预算预建 → 逐格子进程执行 → 合并 → 重试 ≤1 → doneMarker ──
    state = base_state("full")
    for bk, (frac, seed) in BUDGET_KEYS.items():
        if bk in state.get("budgets", {}):
            continue
        state.setdefault("budgets", {})[bk] = build_budget(frac, seed)
        save_summary(state)

    for spec in cell_specs():
        key = spec["key"]
        if key in state["cells"]:
            log(f"[skip] {key} (已完成)")
            continue
        done = False
        for attempt in (1, 2):
            log(f"[launch] {key} (attempt {attempt})")
            cell_log = open(OUT_DIR / f"{key}.log", "w", encoding="utf-8")
            try:
                subprocess.run([PY, "-u", str(Path(__file__).resolve()), "--run-one", key],
                               cwd=str(REPO), timeout=5400, stdout=cell_log,
                               stderr=subprocess.STDOUT)
            except subprocess.TimeoutExpired:
                log(f"[timeout] {key}")
            finally:
                cell_log.close()
            try:
                lc = json.loads(LAST_CELL.read_text(encoding="utf-8"))
            except Exception:
                lc = {}
            if lc.get("key") == key and lc.get("ok"):
                state["cells"][key] = lc["result"]
                state["failed"].pop(key, None)
                done = True
                log(f"[cell-ok] {key}")
                break
            err = lc.get("error", "child crashed / no result") if lc.get("key") == key else "child crashed / stale last_cell"
            state["failed"][key] = err
            log(f"[cell-FAILED] {key} (attempt {attempt}): {err[:160]}")
            time.sleep(20)  # 给系统回收内存的间隙
        save_summary(state)
        if not done:
            log(f"[cell-GIVEUP] {key}：2 次尝试均失败，如实记录后继续")

    state["doneMarker"] = "H2H_MATRIX_DONE"
    state["date_end"] = datetime.now().isoformat(timespec="seconds")
    save_summary(state)
    log(f"[done] cells={len(state['cells'])} failed={len(state['failed'])} marker=H2H_MATRIX_DONE")


if __name__ == "__main__":
    main()
