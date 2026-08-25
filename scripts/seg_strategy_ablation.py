"""tab3 −无监督分割第三臂：分割策略三臂消融（W34 设计落地，W38 执行）。

设计来源: docs/paper/experiment-skeleton.md tab3「−无监督分割」行入册的最小实验
（W34 排查定案: P0.2 既有对照系等段数随机切分 null 而非滑窗方法臂，本实验补齐第三臂）。

三臂对照（同 episode 同 GT 同匹配数学，唯一变量 = 边界放置策略）:
  A. smq      SMQ 运动词量化自适应边界（E-C checkpoint 只读推理）
  B. uniform  等段数均匀切分（段数 = max(len(pred_smq), 2)，与随机 null 的等段数控
              制完全对称；确定性、无 RNG）——本实验新增的方法臂
  B'. grid    附属变体 stride=patch_size 固定网格平铺（W34 设计"或"字第二选项，
              不参与判据主判定，仅作稳健性旁证）
  C. null     随机切分蒙特卡洛期望（直接引用 P0.2 random_baseline_mean_iou）

协议对齐: 原样复用 scripts/eval_smq_segmentation.py --gt-protocol seeds 基建
（select_eval_clips / group_into_episodes / build_seed_gt_episode 消费规则 /
segment_iou 匹配数学），seeds 规范逐项写入输出 JSON。
复现门: SMQ 臂须与 reports/p02-smq-iou-eC-seeds.json 逐位一致（推理确定性实证）。

判据（W34 预注册原文）: 均匀窗显著劣于 SMQ 且 ≥ 随机 null 才构成边界增益消融。

资源纪律: 进程级 CUDA_VISIBLE_DEVICES 置空 → 纯 CPU，不与 GPU 窗口（W33/W39）冲突。

用法:
    & .venv\\Scripts\\python.exe scripts\\seg_strategy_ablation.py `
        [--config configs/seg_ablation_p02.yaml]
        [--out reports/p02-seg-strategy-ablation-2026-08-25.json]
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # 纯 CPU 纪律：必须在 torch 导入前生效

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from psd.data.smq_input import (  # noqa: E402
    build_episode,
    build_seed_gt_episode,
    group_into_episodes,
    select_eval_clips,
)
from psd.training.segment_iou import (  # noqa: E402
    boundary_f1,
    match_segments,
    random_baseline_mean_iou,
    segmentation_from_indices,
)


# ---------------------------------------------------------------- 纯函数臂构造器

def uniform_cut_segments(total_len: int, n_segments: int) -> list[tuple[int, int]]:
    """等段数均匀切分 [0,total_len)：无缝无重叠，段长差 ≤1（array_split 语义）。

    前 total_len % n_segments 段各多 1 帧。与随机 null 的等段数控度对称：
    两臂段数同为 max(len(pred_smq), 2)，差异只剩边界放置策略本身。
    """
    if n_segments < 1 or n_segments > total_len:
        raise ValueError(f"n_segments 必须在 [1, total_len] 内: got {n_segments} (total={total_len})")
    base, rem = divmod(total_len, n_segments)
    segs: list[tuple[int, int]] = []
    pos = 0
    for i in range(n_segments):
        end = pos + base + (1 if i < rem else 0)
        segs.append((pos, end))
        pos = end
    return segs


def fixed_grid_segments(total_len: int, window: int) -> list[tuple[int, int]]:
    """固定网格平铺 stride=window：[(0,w),(w,2w),...]，尾段允许短于 window。"""
    if window < 1:
        raise ValueError(f"window 必须 ≥1: got {window}")
    segs: list[tuple[int, int]] = []
    pos = 0
    while pos < total_len:
        end = min(pos + window, total_len)
        segs.append((pos, end))
        pos = end
    return segs


def evaluate_arm(pred: list[tuple[int, int]], gt: list[tuple[int, int]],
                 gt_bounds: list[int], tol: int) -> dict:
    """单臂单 episode 指标封装（复用 segment_iou 匹配数学，键名对齐 P0.2 报告口径）。"""
    m = match_segments(pred, gt)
    _, _, f1_b = boundary_f1(pred, gt_bounds, tol)
    return {
        "mean_matched_iou": round(m["mean_matched_iou"], 4),
        "seg_precision@0.5": round(m["seg_precision"], 4),
        "seg_recall@0.5": round(m["seg_recall"], 4),
        "boundary_f1": round(f1_b, 4),
    }


def evaluate_criterion(smq_ious: list[float], uniform_ious: list[float],
                       null_ious: list[float]) -> dict:
    """W34 预注册判据操作化（三门风格沿 c1 报告 §9.3）:

    - 方向门 direction_gate_pass: uniform < smq 的 episode 数 ≥ ceil(2n/3)
    - 幅度门 magnitude_gate_pass: 均值差超出两臂 seed 间噪声量级（较大臂内 std）
    - uniform_worse_than_smq = 方向门 AND 幅度门
    - uniform_ge_random_null: uniform 聚合均值 ≥ null 聚合均值（并列算过）
    - boundary_gain_ablation_established = 上两者同时成立
    """
    assert len(smq_ious) == len(uniform_ious) == len(null_ious) >= 1, "三臂 episode 数必须一致"
    n = len(smq_ious)
    direction_wins = sum(1 for s, u in zip(smq_ious, uniform_ious) if u < s)
    direction_gate_pass = direction_wins >= math.ceil(2 * n / 3)

    mean_gap = float(np.mean(smq_ious)) - float(np.mean(uniform_ious))
    noise = max(float(np.std(smq_ious)), float(np.std(uniform_ious)))
    magnitude_gate_pass = mean_gap > noise

    uni_mean = float(np.mean(uniform_ious))
    null_mean = float(np.mean(null_ious))
    ge_null = uni_mean >= null_mean

    worse = direction_gate_pass and magnitude_gate_pass
    return {
        "n_episodes": n,
        "direction_wins": direction_wins,
        "direction_gate_pass": bool(direction_gate_pass),
        "mean_gap_smq_minus_uniform": round(mean_gap, 6),
        "noise_scale_max_std": round(noise, 6),
        "magnitude_gate_pass": bool(magnitude_gate_pass),
        "uniform_worse_than_smq": bool(worse),
        "uniform_mean": round(uni_mean, 6),
        "null_mean": round(null_mean, 6),
        "uniform_ge_random_null": bool(ge_null),
        "boundary_gain_ablation_established": bool(worse and ge_null),
    }


# ---------------------------------------------------------------- 三臂 runner

def _aggregate(vals: list[float]) -> dict:
    return {"mean_matched_iou": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4), "n_episodes": len(vals)}


def run(config_path: str, out_path: str) -> dict:
    cfg = yaml.safe_load((REPO_ROOT / config_path).read_text(encoding="utf-8"))

    # seeds 规范对齐：与 eval_smq_segmentation.py 同式全局播种
    seed = cfg["eval_seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    feats_all = REPO_ROOT / cfg["work_root"] / "features_all"
    ckpt = REPO_ROOT / cfg["ckpt"]
    if not ckpt.exists():
        raise SystemExit(f"[seg-ablation] checkpoint 不存在: {ckpt}")
    ref_path = REPO_ROOT / cfg["p02_reference_report"]
    ref = json.loads(ref_path.read_text(encoding="utf-8"))

    all_names = [p.stem for p in sorted(feats_all.glob("*.npy"))]
    eval_names = select_eval_clips(all_names,
                                   total=cfg["eval_episodes"] * cfg["ep_clips"],
                                   seed=seed)
    groups = group_into_episodes(eval_names, clips_per_episode=cfg["ep_clips"])

    from psd.training.smq_runner import SMQSegmenter  # noqa: E402 延迟导入保持纯函数可独立测试

    seg = SMQSegmenter(
        in_channels=cfg["in_channels"], filters=cfg["filters"],
        num_layers=cfg["num_layers"], latent_dim=cfg["latent_dim"],
        num_actions=cfg["num_actions"], num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=cfg["patch_size"],
        decay=cfg["decay"], kmeans=cfg["kmeans"],
        kmeans_metric=cfg["kmeans_metric"],
        sampling_quantile=cfg["sampling_quantile"],
        replacement_strategy=cfg["replacement_strategy"],
    )

    results: dict = {
        "experiment": "p02-seg-strategy-ablation",
        "design_source": ("docs/paper/experiment-skeleton.md tab3 −无监督分割行 "
                          "(W34 入册最小版) + NEXT-BATCH-plan.md W38"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "metric_layer": "public-real(InterPet4D smal_npy)",
        "gt_protocol": "seeds",
        "protocol": "seed-pseudo-gt-episode-IoU",
        "arms": {
            "smq": "E-C checkpoint 运动词量化自适应边界(只读推理)",
            "uniform": "等段数均匀切分(n=max(len(pred_smq),2),确定性)",
            "grid_supplementary": f"固定网格平铺 stride=patch_size({cfg['patch_size']})",
            "null": "等段数随机切分蒙特卡洛期望(直接引用 P0.2 random_baseline_mean_iou)",
        },
        "seeds_alignment": {
            "global_seeds_set_to": seed,
            "eval_clip_selection_seed": seed,
            "mc_null_seed_formula": "eval_seed + episode_id (同 P0.2 eval 协议)",
            "mc_null_sims": cfg["random_baseline_sims"],
            "uniform_and_grid_arm": "deterministic(无 RNG)",
            "seed_gt_consumption_rules": {"min_conf": 0.8, "min_duration_s": 0.5},
            "device_policy": "CPU-only(CUDA_VISIBLE_DEVICES 置空于 torch 导入前)",
        },
        "ckpt": str(ckpt),
        "config_echo": {"min_seg_len": cfg["min_seg_len"],
                        "boundary_tol": cfg["boundary_tol"],
                        "patch_size": cfg["patch_size"],
                        "short_run": "drop", "vocab_merge": None},
        "episodes": [],
    }

    ious: dict[str, list[float]] = {"smq": [], "uniform": [], "grid": [], "null": []}
    f1s: dict[str, list[float]] = {"smq": [], "uniform": [], "grid": []}

    for k, group in enumerate(groups, 1):
        ep = build_episode(feats_all, group)
        data = ep["data"]
        t_total = int(data.shape[1])

        # 臂 A：SMQ 自适应边界（与 eval 脚本默认路径逐步一致：drop / 无 vocab merge）
        indices = seg.infer_indices(data, ckpt)
        pred_smq = segmentation_from_indices(indices, min_len=cfg["min_seg_len"])
        n_ctrl = max(len(pred_smq), 2)

        # 种子伪 GT（消费规则 conf>=0.8 & dur>=0.5s，帧坐标以 features_all 为准零错位）
        seed_gt = build_seed_gt_episode(REPO_ROOT / cfg["seeds_dir"], feats_all, group)
        gt = [(s["start"], s["end"]) for s in seed_gt]
        gt_labels = [s["label"] for s in seed_gt]
        gt_bounds = [e for _, e in gt][:-1]

        # 臂 B / B' / C
        pred_uni = uniform_cut_segments(t_total, n_ctrl)
        pred_grid = fixed_grid_segments(t_total, cfg["patch_size"])
        null_iou = random_baseline_mean_iou(gt, total_len=t_total,
                                            n_random_segs=n_ctrl,
                                            n_sims=cfg["random_baseline_sims"],
                                            seed=seed + k)

        m_smq = evaluate_arm(pred_smq, gt, gt_bounds, cfg["boundary_tol"])
        m_uni = evaluate_arm(pred_uni, gt, gt_bounds, cfg["boundary_tol"])
        m_grid = evaluate_arm(pred_grid, gt, gt_bounds, cfg["boundary_tol"])

        ious["smq"].append(m_smq["mean_matched_iou"])
        ious["uniform"].append(m_uni["mean_matched_iou"])
        ious["grid"].append(m_grid["mean_matched_iou"])
        ious["null"].append(round(null_iou, 4))
        for arm, m in (("smq", m_smq), ("uniform", m_uni), ("grid", m_grid)):
            f1s[arm].append(m["boundary_f1"])

        results["episodes"].append({
            "id": k, "clips": group, "T": t_total,
            "n_gt_segments": len(gt), "gt_labels": gt_labels,
            "gt_segments": [list(s) for s in gt],
            "pred_smq_segments": [list(s) for s in pred_smq],
            "n_pred_control": n_ctrl,
            "arm_smq": m_smq, "arm_uniform": m_uni,
            "arm_grid_supplementary": m_grid,
            "null_random_mc_expected_iou": round(null_iou, 4),
        })
        print(f"[seg-ablation] ep{k}: T={t_total} 段控={n_ctrl} | "
              f"SMQ={m_smq['mean_matched_iou']:.3f} 均匀={m_uni['mean_matched_iou']:.3f} "
              f"网格={m_grid['mean_matched_iou']:.3f} null={null_iou:.3f}")

    # 复现门：SMQ 臂须逐位复现既有 eC-seeds 报告（推理确定性实证 → 协议对齐证明）
    ref_ious = [e["mean_matched_iou"] for e in ref["episodes"]]
    ref_nulls = [e["random_baseline_iou"] for e in ref["episodes"]]
    smq_match = ref_ious == ious["smq"]
    null_match = ref_nulls == ious["null"]
    results["reproduction_check_vs_p02_report"] = {
        "reference": str(cfg["p02_reference_report"]),
        "reference_smq_ious": ref_ious,
        "this_run_smq_ious": ious["smq"],
        "smq_bitwise_match": bool(smq_match),
        "null_bitwise_match": bool(null_match),
        "passed": bool(smq_match and null_match),
        "note": ("SMQ 臂逐位复现既有报告 = 同 episode 同协议同 ckpt 实证；"
                 "不一致则本实验协议对齐失败，结论作废"),
    }
    if not (smq_match and null_match):
        print("[seg-ablation] ⚠️ 复现门未过：SMQ/null 与既有报告不一致！")

    results["criterion"] = evaluate_criterion(ious["smq"], ious["uniform"], ious["null"])
    results["aggregate"] = {
        arm: _aggregate(ious[arm]) for arm in ("smq", "uniform", "grid", "null")
    }
    results["aggregate_boundary_f1"] = {
        arm: round(float(np.mean(v)), 4) for arm, v in f1s.items()
    }

    out = REPO_ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[seg-ablation] 判据: {results['criterion']}")
    print(f"[seg-ablation] 汇总 → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/seg_ablation_p02.yaml")
    ap.add_argument("--out", default="reports/p02-seg-strategy-ablation-2026-08-25.json")
    args = ap.parse_args()
    run(args.config, args.out)


if __name__ == "__main__":
    main()
