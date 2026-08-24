"""P0.3 Phase B 主实验入口 v2 — 语义桥修复版（W13-C1 任务书）。

执行链（dev-docs/handovers/W13-C1-phaseb-fix.md）：
  Step 1 Phase A 复现（只读复用 run_p03_phasea：段消费 + 锚点聚类器）
  Step 2 双编码器特征：--encoder {aimclr,stgcnbc}（默认 stgcnbc）
         - stgcnbc: W12 checkpoint penultimate 特征（方案 B 主攻，真实段先居中）
         - aimclr : P0.1 冻结 Φ（方案 A 对照消融：量化纯对齐能救多少）
  Step 3 均值中心化对齐（任务书 §二）：μ_syn/μ_real 分别拟合各自减去后
         L2 归一；gate 度量在合成留出协议上进行
  Step 4 余弦最近语义原型映射（复用 jia_phaseB_mapper，映射逻辑零改动）
  Step 5 生产伪标签池 + 报告 JSON 归档

用法：
    python scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml            # 方案 B
    python scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml --encoder aimclr   # 方案 A 消融
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from psd.data.interpet4d import load_clip, resample_to_fixed_t  # noqa: E402
from psd.data.stgcn_bc_dataset import load_pyskl_pickle  # noqa: E402
from psd.training.jia_features import build_segment_view  # noqa: E402
from psd.training.jia_phaseB_mapper import (  # noqa: E402  映射逻辑零改动
    bridge_map_prototypes,
    build_pool_rows,
    build_semantic_prototypes,
    coverage_ratio,
    map_embeddings_to_22,
    split_synthetic_ref_probe,
)
from psd.training.stgcnbc_feature_extractor import (  # noqa: E402
    STGCNBCFeatureExtractor,
    apply_feature_alignment,
    fit_feature_alignment,
)
from psd.training.jia_prototype import PrototypeClusterer  # noqa: E402
import run_p03_phasea as phasea  # noqa: E402  只读复用装配函数


# ---------------------------------------------------------------- 特征装配

def encode_synthetic_aimclr(samples, encoder, *, target_t, conf_threshold, batch_size):
    """方案 A 路径：合成样本经冻结 Φ（build_segment_view 口径）。"""
    views = np.stack([
        build_segment_view(
            np.asarray(s["keypoints"], dtype=np.float32),
            np.ones((np.asarray(s["keypoints"]).shape[0], 24), dtype=np.float32),
            target_t=target_t, conf_threshold=conf_threshold)
        for s in samples])
    outs = []
    for i in range(0, len(views), batch_size):
        outs.append(np.asarray(encoder(views[i:i + batch_size]), dtype=np.float32))
    return np.vstack(outs)


def extract_real_segments_stgcnbc(segments, smal_root, extractor, *, target_t, batch_size):
    """方案 B 路径：真实段切片 → 重采样 → 居中 → penultimate 特征。"""
    cache: dict[str, dict] = {}
    mats = []
    for s in segments:
        cid = s["clip_id"]
        if cid not in cache:
            data = load_clip(smal_root / f"{cid}.npz")
            if data is None:
                raise KeyError(f"clip 未找到: {cid}")
            cache[cid] = data
        kp_seg = cache[cid]["kp_world"][s["start_frame"]: s["end_frame"] + 1]
        mats.append(resample_to_fixed_t(kp_seg, target_t=target_t))
    arr = np.stack(mats)
    outs = []
    for i in range(0, len(arr), batch_size):
        outs.append(extractor.extract(arr[i:i + batch_size]))
    return np.vstack(outs).astype(np.float64)


def extract_synthetic_stgcnbc(samples, extractor, *, batch_size):
    """方案 B 路径：合成样本（原生 T=30）→ penultimate 特征。"""
    arr = np.stack([np.asarray(s["keypoints"], dtype=np.float32) for s in samples])
    outs = []
    for i in range(0, len(arr), batch_size):
        outs.append(extractor.extract(arr[i:i + batch_size]))
    return np.vstack(outs).astype(np.float64)


def accuracy_vs_truth(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(pred) != len(truth):
        raise ValueError(f"预测数 {len(pred)} 与真值数 {len(truth)} 不一致")
    return float((np.asarray(pred) == np.asarray(truth)).mean())


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p03_jia_phaseb.yaml")
    ap.add_argument("--encoder", choices=["stgcnbc", "aimclr"], default="stgcnbc",
                    help="stgcnbc=方案B主攻 / aimclr=方案A对照消融")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("smoke", False)

    t0 = time.time()
    print("=" * 64)
    print(f"P0.3 JIA Phase B v2 — 语义桥={args.encoder}（C1 修复版）")
    print("  池口径: 公开真实层 | 验证口径: 合成层留出协议 + 中心化对齐")
    print("=" * 64)

    # ---- Step 1: Phase A 复现（只读复用，锚点聚类器供元数据）
    anchor_segs, eval_segs, meta = phasea.prepare_segments(cfg)
    cl = PrototypeClusterer(mode="class_mean", seed=42)
    all_segs_sorted = sorted(anchor_segs + eval_segs,
                             key=lambda s: (s["clip_id"], s["start_frame"], s["end_frame"]))
    key_a = {(s["clip_id"], s["start_frame"], s["end_frame"]): i
             for i, s in enumerate(anchor_segs)}
    key_e = {(s["clip_id"], s["start_frame"], s["end_frame"]): i
             for i, s in enumerate(eval_segs)}
    is_eval_row = [k in key_e for k in ((s["clip_id"], s["start_frame"], s["end_frame"])
                                        for s in all_segs_sorted)]
    labels_a = np.array([s["label"] for s in anchor_segs])

    # ---- Step 2: 双路特征
    batch_size = int(cfg["experiment"]["batch_size"])
    if args.encoder == "stgcnbc":
        ckpt_rel = cfg["bridge"]["stgcnbc_ckpt"]
        target_t = int(cfg["bridge"]["target_t"])
        extractor = STGCNBCFeatureExtractor.from_checkpoint(
            str(REPO_ROOT / ckpt_rel), device="cuda" if _cuda_ok() else "cpu")
        smal_root = Path(cfg["data"]["smal_npy_dir"])

        def embed_segments(segs):
            return extract_real_segments_stgcnbc(
                segs, smal_root, extractor, target_t=target_t, batch_size=batch_size)

        def embed_syn(samples):
            return extract_synthetic_stgcnbc(samples, extractor, batch_size=batch_size)
        print(f"[step2] ST-GCN+BC penultimate 桥（{ckpt_rel}, T={target_t}）")
    else:
        encoder = phasea.build_backbone_encoder(
            cfg["backbone"]["weights"], batch_size)
        cache_dir = REPO_ROOT / cfg["data"]["processed_dir"]

        def embed_segments(segs):
            return phasea.get_embeddings(cfg, segs, cache_dir,
                                         encoder=encoder).astype(np.float64)

        def embed_syn(samples):
            return encode_synthetic_aimclr(
                samples, encoder, target_t=int(cfg["backbone"]["target_t"]),
                conf_threshold=float(cfg["backbone"]["conf_threshold"]),
                batch_size=batch_size)
        print("[step2] AimCLR 冻结 Φ 桥（方案 A 消融）")

    samples = load_pyskl_pickle(str(REPO_ROOT / cfg["data"]["synthetic_pkl"]))
    syn_emb = embed_syn(samples)
    syn_labels = np.array([str(s["label_name"]) for s in samples])
    print(f"[step2] 合成特征 {syn_emb.shape} | 真实段特征抽取中…")

    # 真实段按池行序抽取一次（锚点/评估合并表）
    seg_by_key_a = {(s["clip_id"], s["start_frame"], s["end_frame"]): i
                    for i, s in enumerate(anchor_segs)}
    seg_by_key_e = {(s["clip_id"], s["start_frame"], s["end_frame"]): i
                    for i, s in enumerate(eval_segs)}
    emb_anchor = embed_segments(anchor_segs)
    emb_eval = embed_segments(eval_segs)
    emb_all = np.empty((len(all_segs_sorted), emb_anchor.shape[1]))
    kappa_all = np.zeros(len(all_segs_sorted))
    for r, s in enumerate(all_segs_sorted):
        k = (s["clip_id"], s["start_frame"], s["end_frame"])
        if k in key_a:
            emb_all[r] = emb_anchor[seg_by_key_a[k]]
        else:
            emb_all[r] = emb_eval[seg_by_key_e[k]]
    proto_idx_a, _, kappa_a = PrototypeClusterer(
        mode="class_mean", seed=42).fit(emb_anchor, labels_a).assign(emb_anchor)
    _, _, kappa_e = PrototypeClusterer(
        mode="class_mean", seed=42).fit(emb_anchor, labels_a).assign(emb_eval)
    for r, s in enumerate(all_segs_sorted):
        k = (s["clip_id"], s["start_frame"], s["end_frame"])
        kappa_all[r] = kappa_a[seg_by_key_a[k]] if k in key_a else kappa_e[seg_by_key_e[k]]
    proto_idx_all, _, _ = PrototypeClusterer(
        mode="class_mean", seed=42).fit(emb_anchor, labels_a).assign(emb_all)
    print(f"[step2] 真实段特征 {emb_all.shape}")

    results: dict = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "plan": {"encoder": args.encoder},
        "config_echo": {
            "synthetic_pkl": cfg["data"]["synthetic_pkl"],
            "bridge_ckpt": cfg.get("bridge", {}).get("stgcnbc_ckpt"),
            "filter": cfg["filter"], "split": cfg["split"],
            "probe_seed": int(cfg["phaseb"]["probe_seed"]),
        },
        "pipeline_meta": meta,
    }

    # ---- Step 3: 留出协议 gate（对齐统计只在合成 ref 半区拟合，防泄漏）
    probe_seed = int(cfg["phaseb"]["probe_seed"])
    ref_idx, probe_idx = split_synthetic_ref_probe(syn_labels, seed=probe_seed)
    stats_ref = fit_feature_alignment(syn_emb[ref_idx])
    syn_ref_al = apply_feature_alignment(syn_emb[ref_idx], stats_ref,
                                         use_std=args.encoder == "aimclr")
    syn_probe_al = apply_feature_alignment(syn_emb[probe_idx], stats_ref,
                                           use_std=args.encoder == "aimclr")
    protos_ref, names_ref = build_semantic_prototypes(syn_ref_al, syn_labels[ref_idx])
    pred_probe, _ = map_embeddings_to_22(syn_probe_al, protos_ref, names_ref)
    heldout_acc = accuracy_vs_truth(pred_probe, syn_labels[probe_idx])
    gate = heldout_acc >= float(cfg["phaseb"]["heldout_acc_gate"])
    print(f"[step3] 留出精度={heldout_acc:.4f}（gate ≥{cfg['phaseb']['heldout_acc_gate']} → "
          f"{'PASS' if gate else 'FAIL'}）")
    results["mapping_quality_synthetic_layer"] = {
        "protocol": f"stratified_half_split_seed{probe_seed}+mean_centering",
        "n_ref": int(len(ref_idx)), "n_probe": int(len(probe_idx)),
        "heldout_accuracy": round(heldout_acc, 4),
        "acc_gate": float(cfg["phaseb"]["heldout_acc_gate"]),
        "pass_gate": bool(gate),
    }

    # ---- Step 4: 生产池（μ_syn 全量 / μ_real 全量各自拟合——任务书 §二.B 口径）
    stats_syn_full = fit_feature_alignment(syn_emb)
    stats_real = fit_feature_alignment(emb_all)
    use_std = args.encoder == "aimclr"
    syn_full_al = apply_feature_alignment(syn_emb, stats_syn_full, use_std=use_std)
    real_al = apply_feature_alignment(emb_all, stats_real, use_std=use_std)
    protos_full, names_full = build_semantic_prototypes(syn_full_al, syn_labels)
    pred_all, sim_all = map_embeddings_to_22(real_al, protos_full, names_full)

    label_source = f"p03_phaseB_{args.encoder}_centered_seed{probe_seed}"
    out_dir = REPO_ROOT / cfg["data"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_pool_rows(
        segments=all_segs_sorted, seg_emb=real_al, proto_idx=proto_idx_all,
        kappa=kappa_all, semantic_protos=protos_full, semantic_names=names_full,
        embedding_ref="segment_embeddings_phaseB_pool.npz",
        label_source=label_source,
    )
    pool_path = out_dir / "pseudo_pool_phaseB_22class_seed42.jsonl"
    with open(pool_path, "w", encoding="utf-8") as f:
        for row in rows:
            row_out = {k: v for k, v in row.items()}
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
    np.savez_compressed(out_dir / "segment_embeddings_phaseB_pool.npz",
                        emb=real_al.astype(np.float32),
                        fingerprint=np.array(label_source))

    eval_tau = np.array([r["tau_pass"] for r, e in zip(rows, is_eval_row) if e])
    eval_coverage = coverage_ratio(eval_tau) if len(eval_tau) else 0.0
    dist: dict[str, int] = {}
    for r in rows:
        dist[r["pseudo_label"]] = dist.get(r["pseudo_label"], 0) + 1
    n_distinct = len(dist)
    min_classes = int(cfg["phaseb"].get("pool_min_distinct_classes", 6))
    dist_pass = n_distinct >= min_classes
    top = sorted(dist.items(), key=lambda kv: -kv[1])[:8]
    print(f"[step4] 池落盘 {len(rows)} 行 | 不同类别数={n_distinct}（≥{min_classes} → "
          f"{'PASS' if dist_pass else 'FAIL'}）| 覆盖率={eval_coverage:.4f}")
    print(f"[step4] 分布 top8: {top}")
    results["pool_public_real_layer"] = {
        "pool_path": str(pool_path.relative_to(REPO_ROOT)),
        "n_rows_total": len(rows),
        "eval_side_coverage": round(eval_coverage, 4),
        "phasea_coverage_reference_alpha1_tau0": 1.0,
        "coverage_pass_gate": bool(eval_coverage >= 1.0),
        "distinct_classes": n_distinct,
        "distinct_classes_gate": min_classes,
        "distribution_pass_gate": bool(dist_pass),
        "pool_label_distribution_top8": dict(top),
        "embedding_ref": "segment_embeddings_phaseB_pool.npz",
    }

    # ---- Step 4b: Phase A 原型桥接诊断（沿用，作对齐质量参照）
    pa_cl = PrototypeClusterer(mode="class_mean", seed=42).fit(emb_anchor, labels_a)
    bridge_map, _ = bridge_map_prototypes(
        pa_cl.prototypes, pa_cl.prototype_labels, protos_ref, names_ref)
    results["phaseA_bridge_diagnostic"] = {
        "prototype_to_22class_map": bridge_map,
        "note": "诊断量，不作为池标签来源",
    }

    results["acceptance"] = {
        "heldout_accuracy_ge_0p50": bool(gate),
        "coverage_ge_phasea_alpha1": bool(eval_coverage >= 1.0),
        "pool_distinct_classes_ge_6": bool(dist_pass),
        "all_gates_pass": bool(gate and eval_coverage >= 1.0 and dist_pass),
    }
    print("[acceptance]", json.dumps(results["acceptance"], ensure_ascii=False))

    suffix = "" if args.encoder == "stgcnbc" else f"_{args.encoder}"
    out_json = REPO_ROOT / "reports" / f"p03-jia-phaseb-results{suffix}.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 结果 JSON -> {out_json} | 总耗时 {time.time()-t0:.1f}s")


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
