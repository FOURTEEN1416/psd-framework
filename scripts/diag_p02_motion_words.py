"""P0.2 Step A-1 分诊探针：运动词用量体检 + 输入健康检查（W7 救援）。

诊断四问：
  P1 输入特征是否退化——导出视图的零值占比 / 全局方差 / patch 间余弦相似度
     （若输入近恒定，单码重建即"正确感知"，根因在数据管线而非模型）
  P2 motion word 用量——checkpoint 在评估集上的逐帧码直方图 / 熵 / 唯一码数
     （W7 交接文档 Step A-1 官方分诊指标）
  P3 编码器 latent 坍缩探针——patch 级池化后成对余弦相似度
     （P0.1 E-探针同款手法；区分"输入退化"vs"编码器侧坍缩"）
  P4 码本状态——训练后码本行间差异 / EMA cluster_size（活码数）

输出：reports/p02-diag-motionwords.json（口径：公开真实层）+ stdout 摘要。
只读诊断：不改任何 owner 实现；checkpoint 与特征均为现有产物复用。

用法：
    \\.venv\\Scripts\\python.exe scripts\\diag_p02_motion_words.py \
        [--config configs/p02_smq.yaml] \
        [--ckpt runs/p02_smq/models/epoch-30.model] \
        [--out reports/p02-diag-motionwords.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402


def _cosine_matrix(mat: np.ndarray) -> np.ndarray:
    """(n, d) 行向量的成对余弦相似度矩阵。零向量行相似度记 0。"""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = np.divide(mat, np.maximum(norms, 1e-12))
    return unit @ unit.T


def _offdiag_stats(sim: np.ndarray) -> dict:
    n = sim.shape[0]
    if n < 2:
        return {"mean": 1.0, "min": 1.0, "max": 1.0}
    off = sim[~np.eye(n, dtype=bool)]
    return {"mean": round(float(off.mean()), 6),
            "min": round(float(off.min()), 6),
            "max": round(float(off.max()), 6)}


def _entropy(counts: np.ndarray) -> float:
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def probe_inputs(features_dir: Path, episode_groups: list[list[str]]) -> dict:
    """P1：导出特征健康度（eval 全部 clip + 首个训练 episode）。"""
    per_clip = []
    all_patches: list[np.ndarray] = []
    for g in episode_groups[:1]:          # 输入统计取第一个 episode 的 5 个 clip 即足够代表性
        for name in g:
            v = np.load(features_dir / f"{name}.npy")   # (3,T,24,1)
            c, t = v.shape[0], v.shape[1]
            flat = v.transpose(1, 0, 2, 3).reshape(t, -1)      # (T, C*24)
            zeros = float((np.abs(flat) < 1e-8).mean())
            # 帧 32 一切与 SMQ patch 网格对齐，看 patch 间是否近恒定
            p = min(t // 32, 20)
            patches = flat[: p * 32].reshape(p, 32, -1).mean(axis=1)  # (p, C*24)
            sim = _cosine_matrix(patches)
            per_clip.append({
                "clip": name, "T": int(t),
                "zero_fraction": round(zeros, 4),
                "std": round(float(flat.std()), 6),
                "frame_mean_abs": round(float(np.abs(flat).mean()), 6),
                "inter_patch_cos": _offdiag_stats(sim),
            })
            all_patches.append(patches)
    pooled = np.concatenate(all_patches, axis=0)
    cross = _cosine_matrix(pooled)
    return {
        "per_clip": per_clip,
        "pooled_patch_count": int(pooled.shape[0]),
        "cross_clip_inter_patch_cos": _offdiag_stats(cross),
        "verdict_inputs_degenerate": bool(cross[~np.eye(cross.shape[0], dtype=bool)].mean() > 0.999),
    }


def probe_model(cfg: dict, ckpt: Path, features_dir: Path,
                episode_groups: list[list[str]]) -> dict:
    """P2/P3/P4：运动词直方图、latent 坍缩探针、码本状态。"""
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

    K = cfg["num_actions"]
    W = cfg["patch_size"]

    episodes = []
    pooled_hist = Counter()
    latent_probe = None
    for k, group in enumerate(episode_groups, 1):
        ep = np.concatenate(
            [np.load(features_dir / f"{n}.npy") for n in group], axis=1)
        indices = seg.infer_indices(ep.astype(np.float32), ckpt)

        counts = np.bincount(indices, minlength=K)
        pooled_hist.update(indices.tolist())
        runs = []  # 码切换点（patch 网格）
        for i in range(1, len(indices)):
            if indices[i] != indices[i - 1]:
                runs.append(int(i))
        episodes.append({
            "episode": k, "T": int(len(indices)),
            "unique_codes": int((counts > 0).sum()),
            "histogram": {str(i): int(c) for i, c in enumerate(counts) if c > 0},
            "entropy_bits": round(_entropy(counts) / np.log(2), 4),
            "code_change_positions": runs,
        })

        # P3：latent 探针（仅首个 episode，取 forward 后残留的 self.latent）
        if k == 1:
            lat = seg.trainer.model.latent.detach().cpu().numpy()  # (N*M*V, Z, T)
            n_v = lat.shape[0]           # = V（N=M=1）
            z = lat.shape[1]
            t_total = lat.shape[2]
            p = min(t_total // W, 40)
            # (V, Z, T) -> patch 池化 (V, Z, P) -> 每 patch 一个 V*Z 向量
            lp = lat.reshape(n_v, z, p, W).mean(axis=3).transpose(2, 0, 1).reshape(p, -1)
            sim = _cosine_matrix(lp)
            latent_probe = {
                "shape_latent": list(lat.shape),
                "patches": int(p),
                "token_dim": int(n_v * z),
                "latent_std_per_dim_mean": round(float(lat.std(axis=(0, 2)).mean()), 8),
                "inter_patch_cos": _offdiag_stats(sim),
            }

    # P4：码本状态
    sd_keys = [k for k in seg.trainer.model.state_dict().keys()
               if "_embedding" in k or "cluster_size" in k]
    sd = seg.trainer.model.state_dict()
    emb = sd["vq._embedding"].cpu().numpy()              # (K, W, D)
    cs = sd.get("vq.cluster_size", None)
    code_means = emb.mean(axis=1)                        # (K, D)
    sim_codes = _cosine_matrix(code_means)
    off = sim_codes[~np.eye(K, dtype=bool)]
    codebook = {
        "state_keys": sd_keys,
        "embedding_shape": list(emb.shape),
        "code_pair_cos_offdiag": {"mean": round(float(off.mean()), 6),
                                  "min": round(float(off.min()), 6),
                                  "max": round(float(off.max()), 6)},
        "distinct_codes_l2>1e-4": int(sum(
            1 for i in range(K) if np.linalg.norm(
                emb[i] - emb[(i + 1) % K]) > 1e-4)),
        "ema_cluster_size": ([round(float(x), 4) for x in cs.flatten().tolist()]
                             if cs is not None else None),
    }

    hist_arr = np.array([pooled_hist.get(i, 0) for i in range(K)])
    return {
        "ckpt": str(ckpt),
        "episodes": episodes,
        "pooled_unique_codes": int((hist_arr > 0).sum()),
        "pooled_entropy_bits": round(_entropy(hist_arr) / np.log(2), 4),
        "latent_probe_ep1": latent_probe,
        "codebook": codebook,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p02_smq.yaml")
    ap.add_argument("--ckpt", default="runs/p02_smq/models/epoch-30.model")
    ap.add_argument("--out", default="reports/p02-diag-motionwords.json")
    args = ap.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))
    ckpt = REPO_ROOT / args.ckpt
    out_path = REPO_ROOT / args.out
    assert ckpt.exists(), f"checkpoint 不存在: {ckpt}"

    features_dir = REPO_ROOT / cfg["work_root"] / "features_all"
    assert features_dir.is_dir(), f"特征目录不存在: {features_dir}（先跑训练入口导出）"

    # 复现评估集划分（与 eval 脚本完全一致的确定性协议）
    from psd.data.smq_input import group_into_episodes, select_eval_clips

    names = [p.stem for p in sorted(features_dir.glob("*.npy"))]
    eval_names = select_eval_clips(names, total=cfg["eval_episodes"] * cfg["ep_clips"],
                                   seed=cfg["eval_seed"])
    groups = group_into_episodes(eval_names, clips_per_episode=cfg["ep_clips"])

    print("[diag] P1 输入健康探针 ...")
    inputs = probe_inputs(features_dir, groups)
    print(f"  判定 inputs_degenerate={inputs['verdict_inputs_degenerate']} "
          f"(cross-clip patch 余弦均值={inputs['cross_clip_inter_patch_cos']['mean']})")

    print("[diag] P2/P3/P4 模型探针（加载 checkpoint）...")
    model = probe_model(cfg, ckpt, features_dir, groups)

    result = {
        "purpose": "W7 Step A-1 运动词用量体检 + 输入/latent/码本三探针",
        "metric_layer": "public-real(InterPet4D smal_npy)",
        "protocol": "concatenation-episode(seed42, 同 eval 脚本)",
        "inputs": inputs,
        "model": model,
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    print("\n===== Step A-1 摘要 =====")
    print(f"输入退化判定: {inputs['verdict_inputs_degenerate']}")
    for ep in model["episodes"]:
        print(f"  ep{ep['episode']}: 唯一码={ep['unique_codes']}/{cfg['num_actions']} "
              f"H={ep['entropy_bits']}bits 切换点={ep['code_change_positions']}")
    print(f"合并唯一码={model['pooled_unique_codes']}/{cfg['num_actions']} "
          f"H={model['pooled_entropy_bits']}bits")
    if model["latent_probe_ep1"]:
        lp = model["latent_probe_ep1"]
        print(f"latent patch 间余弦: mean={lp['inter_patch_cos']['mean']} "
              f"max={lp['inter_patch_cos']['max']} (≈1 即编码器坍缩)")
    cb = model["codebook"]
    print(f"码本行间余弦 mean={cb['code_pair_cos_offdiag']['mean']} "
          f"distinct={cb['distinct_codes_l2>1e-4']}/{cfg['num_actions']}")
    print(f"[diag] 已写 {out_path}")


if __name__ == "__main__":
    main()
