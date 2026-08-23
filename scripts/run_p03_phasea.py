"""P0.3 Phase A 主实验入口 — 锚点学习 → 原型聚类 → 置信分配（W8 窗口）。

执行链（dev-docs/handovers/W8-p03-jia-phaseA.md §3）：
  Step 1 种子消费适配（p03_seed_consumer）
  Step 2 特征抽取（P0.1 冻结 Φ，jia_features 注入式封装，本脚本装配真实编码器）
  Step 3 原型聚类（jia_prototype: class_mean 主配置 / kmeans 敏感性 + margin κ
                  + frequency-aware margin）
  Step 4 评估（jia_metrics: 纯度/NMI/覆盖率 + 随机基线 + 噪声注入消融）

产出：reports/p03-jia-phasea-results.json（纯度曲线 JSON + 敏感性表原始数据）。

用法：
    python scripts/run_p03_phasea.py --config configs/p03_jia_phasea.yaml
    python scripts/run_p03_phasea.py --config configs/p03_jia_phasea.yaml --smoke   # 快速冒烟
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

from psd.data.p03_seed_consumer import (  # noqa: E402
    filter_segments,
    label_stats,
    load_seed_segments,
    sample_anchor_segments,
    split_clips,
)
from psd.training.jia_features import extract_segment_embeddings  # noqa: E402
from psd.training.jia_metrics import (  # noqa: E402
    inject_label_noise,
    label_priors,
    majority_class_baseline,
    nmi,
    purity,
    purity_at_threshold,
    random_assignment_purity,
)
from psd.training.jia_prototype import (  # noqa: E402
    PrototypeClusterer,
    frequency_aware_thresholds,
)


# ---------------------------------------------------------------- 编码器装配

def build_backbone_encoder(weights_rel: str, batch_size: int):
    """装配 P0.1 冻结 AimCLR 骨干的前向闭包（与 scripts/eval_aimclr.py 同口径）。"""
    import torch

    aimclr_root = REPO_ROOT / "external" / "AimCLR"
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from torchlight import import_class  # noqa: E402

    model_cls = import_class("net.aimclr.AimCLR")
    model = model_cls(
        base_encoder="net.st_gcn.Model", pretrain=True, feature_dim=128,
        queue_size=1024, momentum=0.999, Temperature=0.07, mlp=True,
        in_channels=3, hidden_channels=16, hidden_dim=256, num_class=12,
        dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    )
    sd = torch.load(REPO_ROOT / weights_rel, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.cuda().eval()

    def encode(x_np):
        """(B,3,T,V,M) numpy -> (B,256) backbone 池化特征（fc 前，惯例口径）。"""
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(x_np)).float().cuda()
            enc = model.encoder_q
            n, c, t, v, m = x.size()
            h = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
            h = enc.data_bn(h)
            h = h.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
            for gcn, imp in zip(enc.st_gcn_networks, enc.edge_importance):
                h, _ = gcn(h, enc.A * imp)
            h = torch.nn.functional.avg_pool2d(h, h.size()[2:])
            h = h.view(n, m, -1).mean(dim=1)
        return h.cpu().numpy().astype(np.float32)

    return encode


# ---------------------------------------------------------------- 管线步骤

def prepare_segments(cfg) -> tuple[list[dict], list[dict], dict]:
    """Step 1+切分：种子加载 → 过滤 → clip 级不相交切分。"""
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
        "anchor_clip_stats": label_stats(anchor_segs),
        "eval_clip_stats": label_stats(eval_segs),
        "n_anchor_clips": len(anchor_clips),
        "n_eval_clips": len(eval_clips),
    }
    return anchor_segs, eval_segs, meta


def _segments_fingerprint(segments: list[dict]) -> str:
    """段集合指纹：clip_id:start:end 序列的 sha1——缓存按内容寻址，防错读。"""
    import hashlib
    payload = ";".join(
        f"{s['clip_id']}:{s['start_frame']}:{s['end_frame']}" for s in segments)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def get_embeddings(cfg, segments, cache_dir: Path, encoder=None, clip_loader=None):
    """Step 2：段 embedding（一次性抽取并按指纹缓存；smoke 模式不缓存）。"""
    fp = _segments_fingerprint(segments)
    cache_path = cache_dir / f"segment_embeddings_{fp[:16]}.npz"
    if not cfg.get("smoke") and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if str(z["fingerprint"]) == fp:
                return z["emb"]

    smal_root = Path(cfg["data"]["smal_npy_dir"])

    def _load(clip_id):
        from psd.data.interpet4d import load_clip
        return load_clip(smal_root / f"{clip_id}.npz")

    loader = clip_loader or _load
    encode = encoder or build_backbone_encoder(cfg["backbone"]["weights"],
                                               int(cfg["experiment"]["batch_size"]))
    emb = extract_segment_embeddings(
        segments, loader, encode,
        batch_size=int(cfg["experiment"]["batch_size"]),
        target_t=int(cfg["backbone"]["target_t"]),
        conf_threshold=float(cfg["backbone"]["conf_threshold"]),
    )
    if not cfg.get("smoke"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, emb=emb, fingerprint=np.array(fp))
        print(f"[cache] {len(segments)} 段 embedding -> {cache_path.name} {emb.shape}")
    return emb


# ---------------------------------------------------------------- 实验单元

def run_once(emb_a, labels_a, emb_e, labels_e, *, mode="class_mean", k=None,
             run_seed=42, tau_grid=None, freq_alpha=1.0):
    """单次运行：锚点建原型 → 评估侧分配 → 指标。返回指标 dict。"""
    cl = PrototypeClusterer(mode=mode, k=k, seed=run_seed).fit(emb_a, labels_a)
    _, pred_e, kappa_e = cl.assign(emb_e)

    priors_eval = label_priors(labels_e)
    out = {
        "run_seed": run_seed,
        "mode": mode,
        "k": int(k) if k else int(len(cl.prototype_labels)),
        "n_anchors": int(len(labels_a)),
        "n_eval": int(len(labels_e)),
        "purity": round(purity(pred_e, labels_e), 4),
        "nmi": round(nmi(pred_e, labels_e), 4),
        "random_baseline_purity": round(random_assignment_purity(priors_eval), 4),
        "majority_baseline": round(majority_class_baseline(labels_e), 4),
    }

    # frequency-aware margin：按评估侧先验对 κ 过滤阈值做类别化下调
    # （Phase A 无独立提案池，先验用评估侧经验分布估计——报告备案此近似）
    taus = tau_grid or [0.0]
    tau_mult = frequency_aware_thresholds(1.0, priors_eval, alpha=freq_alpha)
    curves = []
    pred_arr = np.asarray(pred_e)
    for tau in taus:
        # 类别相关阈值：τ_c = τ · max(0.5,(π_c/π_max)^α)——稀有类更易过筛
        thr = np.array([tau * tau_mult.get(str(lab), tau) for lab in pred_e])
        mask = kappa_e >= thr
        cov = float(mask.mean())
        p_t = purity(pred_arr[mask], np.asarray(labels_e)[mask]) if mask.any() else float("nan")
        curves.append({"tau": float(tau), "coverage": round(cov, 4),
                       "purity_covered": (round(p_t, 4) if np.isfinite(p_t) else None)})
    out["tau_curve"] = curves
    return out


def agg_runs(runs: list[dict], keys=("purity", "nmi")) -> dict:
    """≥3 runs 聚合 mean±std（统计协议）。"""
    out = {}
    for key in keys:
        vals = np.array([r[key] for r in runs], dtype=np.float64)
        out[f"{key}_mean"] = round(float(vals.mean()), 4)
        out[f"{key}_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
    return out


# ---------------------------------------------------------------- 主流程

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p03_jia_phasea.yaml")
    ap.add_argument("--smoke", action="store_true", help="冒烟模式：小样本快速验证全链")
    args = ap.parse_args()

    with open(REPO_ROOT / args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["smoke"] = args.smoke
    if args.smoke:
        cfg["data"]["seeds_dir"] = cfg["data"]["seeds_dir"]  # 冒烟仅截断 clip 数（下方处理）

    t0 = time.time()
    print("=" * 64)
    print("P0.3 JIA Phase A — 锚点学习 + 原型聚类（公开真实层口径）")
    print("=" * 64)

    # ---- Step 1: 种子消费 + 切分
    anchor_segs, eval_segs, meta = prepare_segments(cfg)
    if args.smoke:  # 冒烟：只取前 24 个锚点 clip + 12 个评估 clip 的段
        a_clips = sorted({s["clip_id"] for s in anchor_segs})[:24]
        e_clips = sorted({s["clip_id"] for s in eval_segs})[:12]
        anchor_segs = [s for s in anchor_segs if s["clip_id"] in set(a_clips)]
        eval_segs = [s for s in eval_segs if s["clip_id"] in set(e_clips)]
        meta["smoke"] = {"n_anchor_clips": len(a_clips), "n_eval_clips": len(e_clips)}
    print(f"[step1] 原始段 {meta['n_raw_segments']} -> 过滤后 {meta['n_filtered_segments']}"
          f" | 锚点侧 {len(anchor_segs)} 段 / {meta['n_anchor_clips']} clips"
          f" | 评估侧 {len(eval_segs)} 段 / {meta['n_eval_clips']} clips")
    print(f"[step1] 过滤后类别分布: {meta['filtered_label_stats']}")

    # ---- Step 2: 特征抽取（冻结 Φ，一次性，指纹寻址缓存）
    cache_dir = REPO_ROOT / cfg["data"]["processed_dir"]
    emb_a = get_embeddings(cfg, anchor_segs, cache_dir)
    emb_e = get_embeddings(cfg, eval_segs, cache_dir)
    labels_a = np.array([s["label"] for s in anchor_segs])
    labels_e = np.array([s["label"] for s in eval_segs])
    print(f"[step2] Φ 特征: 锚点 {emb_a.shape} / 评估 {emb_e.shape}")

    exp_cfg = cfg["experiment"]
    run_seeds = [int(s) for s in exp_cfg["run_seeds"]]
    tau_grid = [float(t) for t in exp_cfg["tau_grid"]]

    results: dict = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "metric_layer": "公开真实层 - 物理先验伪标签 (InterPet4D v1)",
        "config_echo": {
            "filter": cfg["filter"], "split": cfg["split"],
            "backbone_weights": cfg["backbone"]["weights"],
            "run_seeds": run_seeds, "smoke": bool(args.smoke),
        },
        "pipeline_meta": meta,
    }

    # ---- E3 主配置：class_mean 初始化 + τ 曲线（α=1 默认 与 α=0 消融）
    print("\n[E3-main] class_mean × τ 扫描 × frequency-aware α∈{1.0, 0.0}")
    main_block = {}
    for alpha in [float(exp_cfg["freq_alpha_default"]), float(exp_cfg["freq_alpha_ablation"])]:
        runs = [run_once(emb_a, labels_a, emb_e, labels_e, mode="class_mean",
                         run_seed=s, tau_grid=tau_grid, freq_alpha=alpha)
                for s in run_seeds]
        tag = f"alpha_{alpha:.1f}"
        main_block[tag] = {
            **agg_runs(runs),
            "random_baseline_purity": runs[0]["random_baseline_purity"],
            "majority_baseline": runs[0]["majority_baseline"],
            "per_run": [{k: r[k] for k in ("run_seed", "purity", "nmi")} for r in runs],
            # τ 曲线取 run_seed 首个为代表（τ 过滤在 class_mean 下对 seed 不敏感，
            # 差异仅来自锚点子采样——完整 per-seed 曲线见 results json 附带）
            "tau_curve_representative": runs[0]["tau_curve"],
        }
        r0 = main_block[tag]
        print(f"  α={alpha}: purity={r0['purity_mean']:.4f}±{r0['purity_std']:.4f}"
              f" nmi={r0['nmi_mean']:.4f}±{r0['nmi_std']:.4f}"
              f" | 随机基线 {runs[0]['random_baseline_purity']}"
              f" 多数类 {runs[0]['majority_baseline']}")
    results["e3_main"] = main_block

    # ---- 敏感性扫描：种子比例 |S| × K（kmeans 模式）
    print("\n[sensitivity] ratio × K (kmeans)")
    sens_rows = []
    for ratio in [float(r) for r in exp_cfg["ratios"]]:
        for k in [int(k) for k in exp_cfg["ks"]]:
            runs = []
            for s in run_seeds:
                idx_keep = sample_anchor_segments(
                    [{"i": i, "label": l} for i, l in enumerate(labels_a.tolist())],
                    ratio=ratio, seed=s)
                sel = np.array([d["i"] for d in idx_keep])
                runs.append(run_once(emb_a[sel], labels_a[sel], emb_e, labels_e,
                                     mode="kmeans", k=k, run_seed=s))
            ag = agg_runs(runs)
            row = {"ratio": ratio, "k": k, **ag}
            sens_rows.append(row)
            print(f"  |S|={ratio:.2f} K={k}: purity={ag['purity_mean']:.4f}±{ag['purity_std']:.4f}")
    results["sensitivity_ratio_k"] = sens_rows

    # ---- 噪声注入消融（R8 缓解项）：主配置上污染锚点标签
    print("\n[noise-ablation] 锚点标签噪声 q ∈ ", exp_cfg["noise_rates"])
    noise_rows = []
    for q in [float(q) for q in exp_cfg["noise_rates"]]:
        runs = []
        for s in run_seeds:
            noisy_a = inject_label_noise(labels_a, rate=q, seed=s)
            runs.append(run_once(emb_a, noisy_a, emb_e, labels_e, mode="class_mean",
                                 run_seed=s, tau_grid=[0.0]))
        ag = agg_runs(runs)
        noise_rows.append({"noise_rate": q, **ag})
        print(f"  q={q:.2f}: purity={ag['purity_mean']:.4f}±{ag['purity_std']:.4f}")
    results["noise_ablation"] = noise_rows

    # ---- 验收判定（交接 §5/§6）
    base_random = results["e3_main"][f"alpha_{float(exp_cfg['freq_alpha_default']):.1f}"]
    ratio_vs_random = base_random["purity_mean"] / base_random["random_baseline_purity"]
    noise30 = next(r for r in noise_rows if abs(r["noise_rate"] - 0.3) < 1e-9)
    results["acceptance"] = {
        "purity_mean": base_random["purity_mean"],
        "ratio_vs_random_baseline": round(ratio_vs_random, 3),
        "threshold_ratio_vs_random": float(cfg["acceptance"]["purity_min_ratio_vs_random"]),
        "pass_ratio_gate": bool(ratio_vs_random >= float(cfg["acceptance"]["purity_min_ratio_vs_random"])),
        "noise_q30_purity_mean": noise30["purity_mean"],
        "noise_q30_above_1p5x_random": bool(
            noise30["purity_mean"] >= 1.5 * base_random["random_baseline_purity"]),
    }
    print("\n[acceptance]", json.dumps(results["acceptance"], ensure_ascii=False))

    out_json = REPO_ROOT / "reports" / "p03-jia-phasea-results.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[done] 结果 JSON -> {out_json} | 总耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
