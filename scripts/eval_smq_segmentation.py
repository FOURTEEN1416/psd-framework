"""P0.2 评估入口：SMQ 分割 IoU + 边界 F1 + 可视化。

协议（拼接式 episode，公开真实层口径——报告需披露）：
- 每 episode 由 ep_clips 个不同狗的 clip 拼接，GT 段 = 源 clip 区间
- 预测段 = VQ motion word 序列的同码连续段（≥min_seg_len）
- 指标：Hungarian 匹配 mean IoU / seg P/R@0.5 / 边界 F1@tol + 等段数随机基线对照

用法：
    .\\.venv\\Scripts\\python.exe scripts\\eval_smq_segmentation.py --iou \\
        [--ckpt runs/p02_smq/models/epoch-30.model] [--vis] [--out reports/p02-smq-iou.json]
"""
import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from psd.data.smq_input import build_episode, group_into_episodes, select_eval_clips  # noqa: E402
from psd.training.segment_iou import (  # noqa: E402
    boundary_f1,
    match_segments,
    random_baseline_mean_iou,
    segmentation_from_indices,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p02_smq.yaml")
    ap.add_argument("--iou", action="store_true", help="执行拼接式 episode IoU 评估")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--vis", action="store_true", help="输出前 N 个 episode 的可视化 PNG")
    ap.add_argument("--out", default="reports/p02-smq-iou.json")
    args = ap.parse_args()
    assert args.iou or args.vis, "至少指定 --iou 或 --vis"
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    feats_all = REPO_ROOT / cfg["work_root"] / "features_all"
    ckpt = Path(args.ckpt) if args.ckpt else REPO_ROOT / "runs/p02_smq/models" / f"epoch-{cfg['epochs']}.model"
    if not ckpt.exists():
        raise SystemExit(f"[p02-eval] checkpoint 不存在: {ckpt}（先跑 train_smq_segmentation.py）")

    # 重建与训练时一致的 eval 抽选（同 seed 确定性）
    all_names = [p.stem for p in sorted(feats_all.glob("*.npy"))]
    eval_names = select_eval_clips(all_names,
                                   total=cfg["eval_episodes"] * cfg["ep_clips"],
                                   seed=cfg["eval_seed"])
    groups = group_into_episodes(eval_names, clips_per_episode=cfg["ep_clips"])

    from psd.training.smq_runner import SMQSegmenter

    seg = SMQSegmenter(
        in_channels=cfg["in_channels"], filters=cfg["filters"],
        num_layers=cfg["num_layers"], latent_dim=cfg["latent_dim"],
        num_actions=cfg["num_actions"], num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=cfg["patch_size"],
        decay=cfg["decay"], kmeans=cfg["kmeans"], kmeans_metric=cfg["kmeans_metric"],
        sampling_quantile=cfg["sampling_quantile"],
        replacement_strategy=cfg["replacement_strategy"],
    )

    results = {"protocol": "concatenation-episode-IoU",
               "metric_layer": "public-real(InterPet4D smal_npy)",
               "ckpt": str(ckpt), "episodes": []}
    ious, f1s = [], []
    for k, group in enumerate(groups, 1):
        ep = build_episode(feats_all, group)
        data = ep["data"]
        indices = seg.infer_indices(data, ckpt)
        pred = segmentation_from_indices(indices, min_len=cfg["min_seg_len"])
        gt = [(s["start"], s["end"]) for s in ep["segments"]]
        gt_bounds = [s for _, s in gt][:-1]

        m = match_segments(pred, gt)
        p_b, r_b, f1_b = boundary_f1(pred, gt_bounds, tol=cfg["boundary_tol"])
        base = random_baseline_mean_iou(gt, total_len=data.shape[1],
                                        n_random_segs=max(len(pred), 2),
                                        n_sims=cfg["random_baseline_sims"],
                                        seed=cfg["eval_seed"] + k)
        ious.append(m["mean_matched_iou"]); f1s.append(f1_b)
        results["episodes"].append({
            "id": k, "clips": group, "T": int(data.shape[1]),
            "gt_segments": [list(s) for s in gt],
            "pred_segments": [list(s) for s in pred],
            "mean_matched_iou": round(m["mean_matched_iou"], 4),
            "seg_precision@0.5": round(m["seg_precision"], 4),
            "seg_recall@0.5": round(m["seg_recall"], 4),
            "boundary_f1": round(f1_b, 4), "boundary_p": round(p_b, 4),
            "boundary_r": round(r_b, 4),
            "random_baseline_iou": round(base, 4),
        })
        print(f"[p02-eval] ep{k}: IoU={m['mean_matched_iou']:.3f} "
              f"F1@{cfg['boundary_tol']}={f1_b:.3f} 随机基线={base:.3f} "
              f"(pred {len(pred)} 段 vs GT {len(gt)} 段)")

        if args.vis and k <= cfg["vis_episodes"]:
            gt_bounds_all = [s for _, s in gt]
            _plot(k, data, indices, gt_bounds_all, pred, cfg)

    results["aggregate"] = {
        "mean_matched_iou": round(float(np.mean(ious)), 4),
        "std": round(float(np.std(ious)), 4),
        "boundary_f1_mean": round(float(np.mean(f1s)), 4),
        "n_episodes": len(groups),
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[p02-eval] 汇总 IoU={results['aggregate']['mean_matched_iou']}±"
          f"{results['aggregate']['std']} F1={results['aggregate']['boundary_f1_mean']} → {out_path}")


def _plot(ep_id, data, indices, gt_bounds, pred, cfg) -> None:
    t = data.shape[1]
    speed = np.linalg.norm(np.diff(data[0].transpose(1, 0, 2), axis=0), axis=-1).mean(axis=-1)
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True,
                             gridspec_kw={"height_ratios": [1, 2]})
    axes[0].imshow(indices[None, :], aspect="auto", cmap="tab20", interpolation="nearest",
                   extent=[0, t, 0, 1])
    axes[0].set_yticks([])
    axes[0].set_title(f"P0.2 SMQ motion words — episode {ep_id}")
    for b in gt_bounds[:-1]:
        for ax in axes:
            ax.axvline(b, color="red", ls="--", lw=1)
    for s, _ in pred[1:]:
        axes[0].axvline(s, color="blue", ls=":", lw=0.8)
    axes[1].plot(speed, color="black", lw=0.6)
    axes[1].set_ylabel("骨架运动速度")
    axes[1].set_xlabel("帧")
    fig.legend(["GT 边界(红虚)", "预测边界(蓝点)"], loc="upper right")
    out = REPO_ROOT / "reports" / f"p02-vis-episode{ep_id}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[p02-eval] 可视化 → {out}")


if __name__ == "__main__":
    main()
