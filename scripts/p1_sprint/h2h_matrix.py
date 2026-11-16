# -*- coding: utf-8 -*-
"""P1 头对头升级 — 多初始化种子 / 多子集种子 / 多预算矩阵（NTU60 xsub）.

目的: 把 tab:h2h 从「单次运行、描述性对比」升级为「多种子 + 显著性检验 + 预算趋势」。
三个方法完全同 harness（同数据 data/ntu60_frame50/xsub、同 40,091/16,487 划分、
同分层抽选公式 k=max(1,round(frac·n_c))）:
  - AimCLR 线性评测（官方 LE recipe, in-house pretext, joint stream）
  - 监督 ST-GCN（16 hidden channels, 0.84M 参数, 端到端 CE）
  - PSD 语义管线（E9 arm-b 同实现: 冻结 256d 特征 + run_selftrain + ScaledLR）

矩阵（全部单元格串行执行，增量落盘，崩溃可续跑）:
  A. 初始化种子方差 @10%/subset42: aimclr/stgcn × init_seed {1,2,3,4}
     （与已发表单次运行 init_seed=0 合计 5 值 → 与 PSD 10 自训练种子做 Welch t/Wilcoxon）
  B. 子集重抽稳健性 @10%: subset_seed {43,44} × {psd(3 selftrain seeds), aimclr(i0), stgcn(i0)}
     （与 subset42 已发表值合成 3 子集配对差 → 描述性）
  C. 预算趋势: frac {0.05, 0.20} × {psd, aimclr, stgcn}（subset42, i0; PSD 3 seeds）

复用: published_baseline_ntu60.linear_cfg / supervised_stgcn_ntu60.train_budget /
run_p14_ntu_lowres 的 ScaledLR·HEAD_CFG·KW·train_linear_head_scaled + run_selftrain。
诚实条款: 全部数字来自实跑；单元格失败如实记录 failed并继续；不编造。

用法:
    .venv/Scripts/python.exe -u scripts/p1_sprint/h2h_matrix.py --smoke
    .venv/Scripts/python.exe -u scripts/p1_sprint/h2h_matrix.py
产出:
    runs/p1_sprint/h2h_matrix/summary.json（增量覆盖写; 完成时含 doneMarker H2H_MATRIX_DONE）
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import types
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
WEIGHTS = "runs/ntu_phaseB/joint_pretext/epoch300_model.pt"
N_CLASS = 60

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(AIMCLR_ROOT))
sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))

import published_baseline_ntu60 as pb  # noqa: E402
import supervised_stgcn_ntu60 as sg  # noqa: E402
import run_p14_ntu_lowres as p14  # noqa: E402
from run_p07_endtoend_ak import evaluate  # noqa: E402


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
    """生成（或指纹一致时复用）frac/seed 参数化的预算数据对，返回相对路径 dict。"""
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
    with open(lab_out, "wb") as f:
        pickle.dump(([names[i] for i in idx], [int(labels_arr[i]) for i in idx]), f)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[budget:{tag}] built n={meta['n_subset']} fp={meta['index_sha256_16']}")
    return {"data": f"runs/p1_sprint/ntu60_budget_data/{tag}/train_position.npy",
            "label": f"runs/p1_sprint/ntu60_budget_data/{tag}/train_label.pkl",
            "n": meta["n_subset"], "fp": meta["index_sha256_16"]}


def aimclr_cell(tag: str, budget: dict, val: dict, init_seed_val: int,
                epochs: int = 100, batch: int = 128, device: int = 0) -> dict:
    """AimCLR 线性评测（官方 LE recipe）；与 pb.run_linear_eval 同链路，init_seed 参数化。"""
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


def stgcn_cell(tag: str, budget: dict, val: dict, init_seed: int,
               epochs: int = 80, batch: int = 128, device: int = 0) -> dict:
    """监督 ST-GCN（16ch 受控容量）；复用 sg.train_budget（同官方 Feeder_single 无增强口径）。"""
    args = types.SimpleNamespace(seed=init_seed, device=device, epochs=epochs, batch_size=batch,
                                 lr=0.1, steps=[50, 70], wd=1e-4, hidden_channels=16,
                                 eval_interval=5)
    r = sg.train_budget(tag, {"data": budget["data"], "label": budget["label"]}, val,
                        budget["n"], args, "runs/p1_sprint/h2h_matrix")
    log(f"[stgcn:{tag}] top1={r['top1'] * 100:.2f}% ({r['wall_clock_sec'] if 'wall_clock_sec' in r else ''}s, init_seed={init_seed})")
    return {"top1": r["top1"], "best_epoch": r["best_epoch"], "init_seed": init_seed,
            "n_params": r["n_params"]}


def psd_cell(subset_seed: int, frac: float, selftrain_seeds=(42, 43, 44)) -> dict:
    """PSD 语义管线（E9 arm-b 同实现）: 冻结特征 + run_selftrain + ScaledLR 终头。"""
    z = np.load(p14.NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    labels_str = np.array([str(int(v)) for v in tr_lab])
    anchor = stratified(tr_lab, subset_seed, frac)
    universe = ~anchor
    rows = []
    for s in selftrain_seeds:
        t0 = time.time()
        # 与 p14.main 逐参一致的直接调用
        from psd.training.tcl_selftrain import run_selftrain  # noqa: E402
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


def save_summary(state: dict) -> None:
    SUMMARY.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="缩样管线验证（数字无意义）")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    for rel in ("train_position.npy", "train_label.pkl", "val_position.npy", "val_label.pkl"):
        if not (XSUB / rel).exists():
            raise SystemExit(f"[h2h-matrix] NTU60 数据缺失: {XSUB / rel}")
    if not (REPO / WEIGHTS).exists():
        raise SystemExit(f"[h2h-matrix] weights 缺失: {WEIGHTS}")
    if not p14.NPZ.exists():
        raise SystemExit(f"[h2h-matrix] 特征转储缺失: {p14.NPZ}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val = {"data": "data/ntu60_frame50/xsub/val_position.npy",
           "label": "data/ntu60_frame50/xsub/val_label.pkl"}
    b10 = build_budget(0.10, 42)
    state = {
        "experiment": "p1_sprint_h2h_matrix",
        "mode": "smoke" if args.smoke else "full",
        "date_start": datetime.now().isoformat(timespec="seconds"),
        "harness": {"data": "data/ntu60_frame50/xsub", "n_train": 40091, "n_val": 16487,
                    "weights": WEIGHTS,
                    "budget_rule": "per-class k=max(1, round(frac*n_c)), default_rng(subset_seed)",
                    "subset42_fp": b10["fp"]},
        "cells": {},
        "failed": {},
    }
    if SUMMARY.exists():
        old = json.loads(SUMMARY.read_text(encoding="utf-8"))
        if old.get("mode") == state["mode"]:
            state["cells"] = old.get("cells", {})
            state["failed"] = old.get("failed", {})
            log(f"[resume] 载入已完成 {len(state['cells'])} 格")

    def run_cell(key: str, fn):
        if key in state["cells"]:
            log(f"[skip] {key} (已完成)")
            return
        t0 = time.time()
        try:
            state["cells"][key] = fn()
            state["cells"][key]["cell_wall_clock_sec"] = round(time.time() - t0, 1)
            log(f"[cell-ok] {key} ({state['cells'][key]['cell_wall_clock_sec']}s)")
        except Exception as e:  # 单格失败不拖垮矩阵，如实记录后继续
            state["failed"][key] = repr(e)[:500]
            log(f"[cell-FAILED] {key}: {e!r}")
        save_summary(state)

    if args.smoke:
        kw_small = dict(p14.KW, max_iters=1, converge_change_rate=1.0)
        p14.KW.clear(); p14.KW.update(kw_small)
        run_cell("smoke_aimclr", lambda: aimclr_cell("smoke_aimclr", b10, val, 0, epochs=2, batch=64))
        run_cell("smoke_stgcn", lambda: stgcn_cell("smoke_stgcn", b10, val, 0, epochs=2, batch=64))
        run_cell("smoke_psd", lambda: psd_cell(42, 0.10, selftrain_seeds=(42,)))
        state["doneMarker"] = "H2H_MATRIX_SMOKE_OK"
    else:
        # A. 初始化种子方差 @10% subset42（与已发表 i0 合计 5 值）
        for s in (1, 2, 3, 4):
            run_cell(f"aimclr_10_s42_i{s}", lambda s=s: aimclr_cell(f"aimclr_10_s42_i{s}", b10, val, s))
        for s in (1, 2, 3, 4):
            run_cell(f"stgcn_10_s42_i{s}", lambda s=s: stgcn_cell(f"stgcn_10_s42_i{s}", b10, val, s))
        # B. 子集重抽稳健性 @10%（init0; PSD 3 selftrain seeds）
        for sub in (43, 44):
            bsub = build_budget(0.10, sub)
            run_cell(f"psd_10_s{sub}", lambda sub=sub: psd_cell(sub, 0.10))
            run_cell(f"aimclr_10_s{sub}_i0", lambda sub=sub, b=bsub: aimclr_cell(f"aimclr_10_s{sub}_i0", b, val, 0))
            run_cell(f"stgcn_10_s{sub}_i0", lambda sub=sub, b=bsub: stgcn_cell(f"stgcn_10_s{sub}_i0", b, val, 0))
        # C. 预算趋势（subset42, init0; PSD 3 selftrain seeds）
        for frac in (0.05, 0.20):
            bf = build_budget(frac, 42)
            run_cell(f"psd_f{int(round(frac * 100)):02d}_s42", lambda fr=frac: psd_cell(42, fr))
            run_cell(f"aimclr_f{int(round(frac * 100)):02d}_s42_i0", lambda fr=frac, b=bf: aimclr_cell(f"aimclr_f{int(round(fr * 100)):02d}_s42_i0", b, val, 0))
            run_cell(f"stgcn_f{int(round(frac * 100)):02d}_s42_i0", lambda fr=frac, b=bf: stgcn_cell(f"stgcn_f{int(round(fr * 100)):02d}_s42_i0", b, val, 0))
        state["doneMarker"] = "H2H_MATRIX_DONE"

    state["date_end"] = datetime.now().isoformat(timespec="seconds")
    save_summary(state)
    log(f"[done] {SUMMARY} cells={len(state['cells'])} failed={len(state['failed'])} marker={state.get('doneMarker')}")


if __name__ == "__main__":
    main()
