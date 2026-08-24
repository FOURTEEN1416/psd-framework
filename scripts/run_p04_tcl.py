"""P0.4 TCL 迭代自训练主实验入口 — 伪标签迭代闭环（W10 窗口）。

执行链（dev-docs/handovers/W10-p04-tcl.md §3）：
  Step 1 Ω 头部训练（锚点种子 ∪ τ_c 过滤伪标签池；冻结 Φ 特征全走 W8 缓存）
  Step 2 迭代循环（Ω 重分配 → 新 κ → τ_c 入池 → 并入训练集 → 重训 Ω）
  Step 3 评估（池精度/覆盖率迭代曲线 + B-1/B-2/α 消融 + ≥3 seeds mean±std）
  Step 4 归档（结果 JSON + 附录 A 格式伪标签池导出 → P0.5 移交物）

用法：
    python scripts/run_p04_tcl.py --config configs/p04_tcl.yaml
    python scripts/run_p04_tcl.py --config configs/p04_tcl.yaml --smoke   # 快速冒烟

零 GPU 纪律：特征缓存按指纹命中即用，MISS 直接报错退出（交接 §2.3 勿重抽，
且规避与 W4 定时任务撞卡）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.p03_seed_consumer import (  # noqa: E402
    filter_segments,
    label_stats,
    load_seed_segments,
    split_clips,
)
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

METRIC_LAYER = "public_real_physics_prior"   # 口径锁字段（附录 A，禁止混池）


# ---------------------------------------------------------------- 段准备与缓存复用

def prepare_segments(cfg) -> tuple[list[dict], list[dict], dict]:
    """与 W8 完全一致的过滤 + clip 级不相交切分（seed=42 防泄漏不许改）。"""
    segs = load_seed_segments(REPO_ROOT / cfg["data"]["seeds_dir"])
    kept = filter_segments(
        segs,
        conf_min=float(cfg["filter"]["conf_min"]),
        min_duration_s=float(cfg["filter"]["min_duration_s"]),
    )
    all_clips = sorted({s["clip_id"] for s in kept})
    anchor_clips, eval_clips = split_clips(
        all_clips,
        eval_ratio=float(cfg["split"]["eval_ratio"]),
        seed=int(cfg["split"]["split_seed"]),
    )
    a_set, e_set = set(anchor_clips), set(eval_clips)
    anchor_segs = [s for s in kept if s["clip_id"] in a_set]
    eval_segs = [s for s in kept if s["clip_id"] in e_set]
    meta = {
        "n_raw_segments": len(segs),
        "n_filtered_segments": len(kept),
        "filtered_label_stats": label_stats(kept),
        "n_anchor_segments": len(anchor_segs),
        "n_eval_segments": len(eval_segs),
    }
    return anchor_segs, eval_segs, meta


def _segments_fingerprint(segments: list[dict]) -> str:
    """段集合指纹——与 scripts/run_p03_phasea.py 同一算法（内容寻址缓存契约）。"""
    payload = ";".join(
        f"{s['clip_id']}:{s['start_frame']}:{s['end_frame']}" for s in segments)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def load_cached_embeddings(cfg, segments_list: list[list[dict]]) -> list[np.ndarray]:
    """只读复用 W8 指纹缓存，按输入顺序返回各段列表的 embedding 数组。
    任何 MISS 即报错（禁重抽 GPU）。"""
    cache_dir = REPO_ROOT / cfg["data"]["cache_dir_p03"]
    embs = []
    for segs in segments_list:
        fp = _segments_fingerprint(segs)
        path = cache_dir / f"segment_embeddings_{fp[:16]}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"W8 特征缓存未命中: {path} —— 按 W10 交接 §2.3 禁止重复抽特征，"
                f"请核对 filter/split 配置是否与 P0.3 一致")
        with np.load(path, allow_pickle=False) as z:
            if str(z["fingerprint"]) != fp:
                raise ValueError(f"缓存指纹不匹配: {path}")
            embs.append(np.asarray(z["emb"], dtype=np.float32))
    return embs


# ---------------------------------------------------------------- 消融矩阵执行


def _calib_str(v) -> str:
    """YAML 会把未引号的 on/off 解析为布尔——统一归一为 'on'/'off' 字符串。"""
    return "on" if str(v).strip().lower() in ("on", "true") else "off"


def cell_kwargs(cell: dict, cfg) -> dict:
    """消融格 -> run_selftrain 关键字。calib off 行退回原型路自迭代（W8 κ 口径）。
    支持 cell 级 target_coverage 覆盖（用于 α × τ 消融网格）。"""
    calib_on = _calib_str(cell["calib"]) == "on"
    grid_src = cfg["experiment"]
    tau_grid_key = "tau_grid" if calib_on else "tau_grid_raw_margin"
    if tau_grid_key not in grid_src:
        grid_src = cfg                              # 原始 margin 网格允许放根层级
    #  cell 级 target_coverage 覆盖（优先使用 cell 自带，无则回退到配置默认）
    tc = cell.get("target_coverage",
                  cfg["experiment"]["tau_select_rule"]["target_coverage"])
    return dict(
        calib_method="softmax_temperature" if calib_on else "off",
        calib_target=float(cfg["experiment"]["calibration"]["target_median_prob_margin"]),
        tau_grid=[float(t) for t in grid_src[tau_grid_key]],
        tau_select=dict(
            rule=cfg["experiment"]["tau_select_rule"]["rule"],
            target_coverage=float(tc)),
        alpha=float(cell["alpha"]),
        standing_mode=str(cell["standing"]),
        subcluster_k=int(cfg["experiment"]["subcluster_k"]),
        subcluster_min_share=float(cfg["experiment"]["subcluster_min_share"]),
    )


def aggregate_cell(runs: list[dict]) -> dict:
    """≥3 seeds 聚合: 逐轮 mean±std（各 seed 停止轮数可不同，按可用 run 聚合）
    + 首末轮配对检验（统计协议）。"""
    n_rounds = max(len(r["rounds"]) for r in runs)
    rounds_out = []
    for ri in range(n_rounds):
        avail = [r["rounds"][ri] for r in runs if len(r["rounds"]) > ri]
        if not avail:
            continue
        rec = {"round": avail[0]["round"], "n_runs": len(avail)}
        for k in ("pool_size", "coverage", "precision"):
            vals = [x[k] for x in avail if x.get(k) is not None]
            if vals:
                arr = np.array(vals, dtype=np.float64)
                rec[f"{k}_mean"] = round(float(arr.mean()), 4)
                rec[f"{k}_std"] = round(float(arr.std(ddof=1)), 4) if len(arr) > 1 else 0.0
        rounds_out.append(rec)

    # 首末轮配对比较（同 seed 配对；n=3 如实披露）
    paired: dict = {}
    p_first, p_last, seeds_used = [], [], []
    for r in runs:
        pr = [x["precision"] for x in r["rounds"]
              if x.get("precision") is not None]
        if len(pr) >= 2:
            p_first.append(pr[0])
            p_last.append(pr[-1])
            seeds_used.append(r["run_seed"])
    if len(p_first) >= 2:
        d = np.array(p_last) - np.array(p_first)
        t_stat, t_p = stats.ttest_rel(p_last, p_first)
        paired = {
            "seeds": seeds_used,
            "delta_pp_mean": round(float(d.mean()) * 100, 2),
            "delta_pp_std": round(float(d.std(ddof=1)) * 100, 4) if len(d) > 1 else 0.0,
            "paired_t_stat": round(float(t_stat), 4),
            "paired_t_p": round(float(t_p), 4),
            "note": "n=3 seeds 配对 t 检验，统计功效有限，效应量以 Δpp 为主披露",
        }

    # standing 精度轨迹（B-2 治理焦点类）
    standing_traj = {}
    for ri in range(n_rounds):
        vals = []
        for r in runs:
            if ri < len(r["rounds"]):
                v = r["rounds"][ri].get("precision_by_pred_class", {}).get("standing")
                if v is not None:
                    vals.append(v)
        if vals:
            standing_traj[f"round_{ri}"] = {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else 0.0}

    return {
        "rounds_agg": rounds_out,
        "paired_first_vs_final": paired,
        "standing_precision_trajectory": standing_traj,
        "stop_reasons": [r["stop_reason"] for r in runs],
        "tau_operating_by_seed": {str(r["run_seed"]): round(float(r["tau_operating"]), 4)
                                  for r in runs},
    }


# ---------------------------------------------------------------- 池导出（附录 A 格式）

def export_pool(out_dir: Path, universe_segs: list[dict], result: dict,
                class_order: list[str], tag: str, run_seed: int,
                embedding_ref: str, iteration_round: str) -> Path:
    """最终清洗后伪标签池 -> JSONL（p03 报告附录 A 字段结构，逐条一行）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_pool = result["final_pool_idx"]
    pred_full = result["final_pred_full"]
    kappa_full = result["final_kappa_full"]
    proto_of_class = {c: i for i, c in enumerate(sorted(class_order))}
    path = out_dir / f"pseudo_pool_{tag}_seed{run_seed}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for i in idx_pool.tolist():
            lab = str(pred_full[i])
            entry = {
                "clip_id": universe_segs[i]["clip_id"],
                "start_frame": int(universe_segs[i]["start_frame"]),
                "end_frame": int(universe_segs[i]["end_frame"]),
                "pseudo_label": lab,
                "proto_idx": proto_of_class.get(lab, -1),   # 类对齐原型槽位（可追溯）
                "kappa_margin": round(float(kappa_full[i]), 6),
                "tau_pass": True,                            # 导出面=过筛面
                "embedding_ref": embedding_ref,
                "label_source": f"p04_tcl_{iteration_round}_{tag}",
                "metric_layer": METRIC_LAYER,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p04_tcl.yaml")
    ap.add_argument("--smoke", action="store_true", help="冒烟：截断 clip 数快速验证全链")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    t0 = time.time()
    print("=" * 64)
    print("P0.4 TCL 半监督自训练 — 伪标签迭代闭环（公开真实层口径）")
    print("=" * 64)

    # ---- Step 0: 段准备 + W8 缓存复用（零 GPU 前提）
    anchor_segs, eval_segs, meta = prepare_segments(cfg)
    print(f"[step0] 锚点侧种子 {len(anchor_segs)} 段 | 未标注宇宙(评估侧) "
          f"{len(eval_segs)} 段 | 过滤分布 {meta['filtered_label_stats']}")

    emb_anchor, emb_eval = load_cached_embeddings(cfg, [anchor_segs, eval_segs])

    if args.smoke:
        # 冒烟：全量缓存命中后按 clip 同步截断段与 embedding 行（不产生新指纹，不触 GPU）
        def _truncate(segs, emb, n_clips):
            keep_clips = set(sorted({s["clip_id"] for s in segs})[:n_clips])
            idx = [i for i, s in enumerate(segs) if s["clip_id"] in keep_clips]
            return [segs[i] for i in idx], emb[idx]
        anchor_segs, emb_anchor = _truncate(anchor_segs, emb_anchor, 24)
        eval_segs, emb_eval = _truncate(eval_segs, emb_eval, 12)
        meta["smoke"] = True

    emb_all = np.vstack([emb_anchor, emb_eval])
    truth_all = np.array([s["label"] for s in anchor_segs] +
                         [s["label"] for s in eval_segs])
    anchor_mask = np.zeros(len(emb_all), dtype=bool)
    anchor_mask[: len(anchor_segs)] = True
    class_names = sorted(set(truth_all.tolist()))
    # 全量段记录（与 emb_all 行序一一对应；池恒 ⊆ 评估侧尾部区段）
    universe_segs = [
        {"clip_id": s["clip_id"], "start_frame": s["start_frame"],
         "end_frame": s["end_frame"]}
        for s in list(anchor_segs) + list(eval_segs)]
    eval_cache_name = f"segment_embeddings_{_segments_fingerprint(eval_segs)[:16]}.npz"
    print(f"[step0] Φ 特征缓存命中: {emb_all.shape} | 类别 {class_names}")

    exp_cfg = cfg["experiment"]
    run_seeds = [int(s) for s in exp_cfg["run_seeds"]]
    ablation = cfg["ablation_matrix"]

    results: dict = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "metric_layer": METRIC_LAYER,
        "config_echo": {
            "filter": cfg["filter"], "split": cfg["split"],
            "head": exp_cfg["head"],
            "calibration": exp_cfg["calibration"],
            "tau_select_rule": exp_cfg["tau_select_rule"],
            "iteration": exp_cfg["iteration"],
            "ablation_matrix": ablation,
            "run_seeds": run_seeds,
            "smoke": bool(args.smoke),
        },
        "pipeline_meta": meta,
        "cells": {},
    }

    # ---- Step 1+2+3: 消融矩阵 × seeds
    main_runs: list[dict] | None = None

    def _is_main_cell(cell: dict, tc_val) -> bool:
        """主配置谓词: calib on / standing consensus / α=1.0 / τ_cov=0.35。
        参数化匹配，避免 tag 字符串脆弱比较（复核修复 2026-08-24）。"""
        return _calib_str(cell["calib"]) == "on" \
            and str(cell["standing"]) == "consensus" \
            and float(cell["alpha"]) == 1.0 \
            and float(tc_val) == 0.35

    for cell in ablation:
        calib_s = _calib_str(cell["calib"])
        # Include target_coverage in tag to avoid collision across τ-grid cells
        tc = cell.get("target_coverage", cfg["experiment"]["tau_select_rule"]["target_coverage"])
        tag = f"{calib_s}_{cell['standing']}_a{cell['alpha']}_tc{tc}"
        print(f"\n[cell] {tag}  (calib={calib_s}, standing={cell['standing']},"
              f" α={cell['alpha']}, τ_cov={tc})")
        kw = cell_kwargs(cell, cfg)
        runs = []
        for seed in run_seeds:
            r = run_selftrain(
                emb_all, truth_all, anchor_mask,
                run_seed=seed, class_names=class_names,
                head_cfg=exp_cfg["head"],
                max_iters=int(exp_cfg["iteration"]["max_iters"]),
                converge_change_rate=float(exp_cfg["iteration"]["converge_change_rate"]),
                precision_drop_patience=int(
                    exp_cfg["iteration"]["precision_drop_patience"]),
                **kw)
            r["run_seed"] = seed
            runs.append(r)
            traj = " -> ".join(str(x.get("precision")) for x in r["rounds"])
            print(f"  seed{seed}: τ*={r['tau_operating']:.3f} "
                  f"精度轨迹 [{traj}] 停止={r['stop_reason']}")
        results["cells"][tag] = aggregate_cell(runs)
        if _is_main_cell(cell, tc):
            main_runs = runs   # 主配置 runs 复用（P0.5 移交池取 seed42）

# ---- P0.5 移交物：主配置 seed42 最终池（附录 A 格式 JSONL）
    pool_paths = []
    if not args.smoke and main_runs is not None:
        r_main = next(r for r in main_runs if r["run_seed"] == 42)
        final_round = f"iter{len(r_main['rounds']) - 1}"
        pool_path = export_pool(
            REPO_ROOT / cfg["data"]["pool_out_dir"],
            universe_segs,
            r_main, class_names, tag="main_consensus_a1.0", run_seed=42,
            embedding_ref=eval_cache_name,
            iteration_round=final_round)
        pool_paths.append(str(pool_path.relative_to(REPO_ROOT)))
        results["handoff_p05"] = {
            "pool_path": str(pool_path.relative_to(REPO_ROOT)),
            "pool_size": int(len(r_main["final_pool_idx"])),
            "universe_size": int(len(eval_segs)),
            "label_source": f"p04_tcl_{final_round}_main_consensus_a1.0",
            "note": "主配置(calib on/consensus/α=1/τ_cov=0.35) seed42 最终轮过筛池；"
                    "真值口径仍为物理先验共识，P0.5 使用时应知悉",
        }
        print(f"\n[handoff] P0.5 移交池 -> {pool_path}")

    # ---- 验收判定（W10 交接 §5: 提升或如实报告不提升+根因）
    # 直接由主配置 runs 聚合（与 main_runs 同源，不依赖 tag 字符串解析）
    main_cell = aggregate_cell(main_runs) if main_runs is not None else None
    if main_cell and main_cell["paired_first_vs_final"]:
        pf = main_cell["paired_first_vs_final"]
        results["acceptance"] = {
            "precision_improved_vs_round0": bool(pf["delta_pp_mean"] > 0),
            "delta_pp_mean": pf["delta_pp_mean"],
            "paired_t_p": pf["paired_t_p"],
            "honest_note": "真值为物理先验伪标签共识（public_real_physics_prior），非人工标注",
        }
        print(f"\n[acceptance] Δpp={pf['delta_pp_mean']:+.2f} (p={pf['paired_t_p']})")

    out_json = REPO_ROOT / "reports" / ("p04-tcl-smoke-results.json" if args.smoke
                                        else "p04-tcl-results.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 结果 JSON -> {out_json} | 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
