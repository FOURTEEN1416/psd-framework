"""P0.2 训练入口：InterPet4D → SMQ 无监督分割训练（薄入口）。

流程：导出特征视图（跳过无效 clip）→ 确定性抽选 eval 集隔离 → 官方 Trainer 训练。
产物：data/processed/p02/（gitignore）、runs/p02_smq/models/epoch-*.model、train_log.txt。

用法：
    .\\.venv\\Scripts\\python.exe scripts\\train_smq_segmentation.py [--config configs/p02_smq.yaml]
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from psd.data.smq_input import (  # noqa: E402
    build_episode,
    chunk_names,
    export_smq_features,
    group_into_episodes,
    rotate_by_dog,
    select_eval_clips,
)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, s):
        for st in self._streams:
            st.write(s)

    def flush(self):
        for st in self._streams:
            st.flush()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/p02_smq.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load((REPO_ROOT / args.config).read_text(encoding="utf-8"))

    # 复现种子（沿用官方口径）
    import random

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])

    work = REPO_ROOT / cfg["work_root"]
    feats_all = work / "features_all"
    train_dir = work / "train_features"
    eval_dir = work / "eval_clips"

    # 1) 导出全部有效 clip 视图（幂等：协议指纹一致则复用，否则重导出）
    proto_fp = f"target_t={cfg['target_t']}"
    marker = feats_all / ".done"
    if not (marker.exists() and marker.read_text(encoding="utf-8") == proto_fp):
        report = export_smq_features(cfg["data_root"], feats_all, target_t=cfg["target_t"])
        print(f"[p02] 导出 {len(report['names'])} clips；剔除无效: {report['skipped']}")
        marker.write_text(proto_fp, encoding="utf-8")
    else:
        report = {"names": [p.stem for p in sorted(feats_all.glob("*.npy"))], "skipped": []}
        print(f"[p02] 复用已有特征目录 {feats_all}（{len(report['names'])} clips）")

    # 2) eval 集确定性抽选与隔离（同 seed 可复现）
    total_eval = cfg["eval_episodes"] * cfg["ep_clips"]
    eval_names = select_eval_clips(report["names"], total=total_eval, seed=cfg["eval_seed"])
    groups = group_into_episodes(eval_names, clips_per_episode=cfg["ep_clips"])
    import json

    (work / "eval_split.json").write_text(
        json.dumps({"eval_clips": eval_names,
                    "episodes": [list(g) for g in groups]}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"[p02] eval 隔离集 {total_eval} clips / {len(groups)} episodes（seed={cfg['eval_seed']}）")

    # 3) 训练集 = 全量 − eval 集，再按 dog 轮转拼成训练 episode（长序列适配 SMQ 的
    #    patch 级 VQ 与死码阈值：4 clip×128 帧=16 patch/样本，batch 12→192 patch/批）
    train_names = [n for n in report["names"] if n not in set(eval_names)]
    rot = rotate_by_dog(train_names, seed=cfg["seed"] % 100000)
    groups_train = chunk_names(rot, cfg["ep_clips_train"])
    ep_dir = work / "train_features_episodes"
    if ep_dir.exists():
        shutil.rmtree(ep_dir)
    ep_dir.mkdir(parents=True)
    for i, g in enumerate(groups_train):
        ep = build_episode(feats_all, g)
        np.save(ep_dir / f"ep{i:03d}.npy", ep["data"])
    import json

    (work / "train_episode_groups.json").write_text(
        json.dumps(groups_train, ensure_ascii=False, indent=1), encoding="utf-8")
    n_train = len(groups_train)
    t_desc = "native" if cfg["target_t"] is None else cfg["ep_clips_train"] * cfg["target_t"]
    print(f"[p02] 训练集 {n_train} episodes（每 {cfg['ep_clips_train']} clips 拼接，T={t_desc}）")

    # 4) 日志 tee 到 runs/
    runs_dir = REPO_ROOT / "runs" / "p02_smq"
    runs_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(runs_dir / "train_log.txt", "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log_f)

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
    save_dir = runs_dir / "models"
    seg.fit(
        features_path=ep_dir, save_dir=save_dir,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], learning_rate=cfg["lr"],
        commit_weight=cfg["commit_weight"], mse_loss_weight=cfg["mse_loss_weight"],
        joint_distance_recons=cfg["joint_distance_recons"], sample_rate=cfg["sample_rate"],
    )
    sys.stdout = sys.__stdout__
    log_f.close()
    final_ckpt = save_dir / f"epoch-{cfg['epochs']}.model"
    print(f"[p02] 训练完成。checkpoint: {final_ckpt}")
    assert final_ckpt.exists(), "末轮 checkpoint 未生成（epochs 需为 5 的倍数）"


if __name__ == "__main__":
    main()
