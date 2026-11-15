"""P1 冲刺实验 3 — SMQ 分割对下游识别的消融（有无 SMQ 两种管线）.

问题（PR-D-26-13140 改造方向 ③）: SMQ 时序分割在 PSD 框架中降级为辅助步骤后，
它对下游识别的真实贡献需要用受控消融量化，而不是只靠写作口径。

设计（同数据、同标签、同分类器，唯一差异 = 中间是否过 SMQ 分割）:
  臂 with_smq:    每个 clip 先经 SMQ 推理得到逐帧 motion-word 索引，再对
                  原始骨架序列按「同码连续段」聚合——每个段内骨架帧平均为一个
                  子序列单元；clip 表示 = 段单元特征的拼接池化（帧级→段级压缩）。
  臂 without_smq: 同一骨架序列直接均匀切分为等长子序列单元（段数与 with_smq
                  臂逐 clip 对齐，保证表示容量一致），池化方式相同。

  两侧单元特征用同一冻结 ST-GCN+BC 特征抽取器（psd/training/stgcnbc_feature_
  extractor，p07 同款），下游分类器同一 LogisticRegression。clip 级得分 =
  单元得分的均值聚合（与段数解耦，保证两臂可比）。

监督信号（诚实口径披露）:
  InterPet4D 无行为 GT；单元标签用规则种子伪 GT（data/seeds/rule_seeds，
  W6 §8 规则，conf>=0.8），在 clip 级聚合为「clip 主导类 = 置信度加权的
  众数标签」。因此本实验度量的是「SMQ 分割是否改善伪标签下的 clip 级
  行为可分性」，属公开真实层口径；数字不与有 GT 基准混报。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/smq_downstream_ablation.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/smq_downstream_ablation.py \
        --ckpt runs/p02_smq/models/epoch-30.model
产出:
    runs/p1_sprint/smq_downstream_ablation/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from psd.data.smq_input import clip_to_smq_view  # noqa: E402
from psd.data.interpet4d import load_clip, resample_to_fixed_t  # noqa: E402
from psd.training.segment_iou import segmentation_from_indices  # noqa: E402
from psd.training.stgcnbc_feature_extractor import STGCNBCFeatureExtractor  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "smq_downstream_ablation"
DEFAULT_CKPT = REPO_ROOT / "runs" / "p02_smq" / "models" / "epoch-30.model"
SEEDS_DIR = REPO_ROOT / "data" / "seeds" / "rule_seeds"
TARGET_T = 30  # ST-GCN+BC 训练口径（extractor 契约）


def load_config() -> dict:
    cfg_path = REPO_ROOT / "configs" / "p02_smq.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # 路径清洗（2026-09-19 迁仓）: 配置里的 k9 绝对路径指向旧根，
    # 仓迁到 D:/Desktop/PR论文 后实际数据在同盘新根 k9-training-system。
    # 本脚本只读消费，不改配置文件（避免动 P0.2 真源）。
    dr = Path(cfg["data_root"])
    if not dr.exists():
        alt = Path("D:/Desktop/PR论文/k9-training-system/data/interpet4d/smal_npy")
        if alt.exists():
            cfg["data_root"] = str(alt)
    return cfg


def list_clips(cfg: dict) -> list[str]:
    feats_dir = REPO_ROOT / cfg["work_root"] / "features_all"
    return sorted(p.stem for p in feats_dir.glob("*.npy"))


def clip_label_from_seeds(name: str, t_view: int) -> tuple[str, float] | None:
    """规则种子 → clip 主导伪标签（帧级众数，置信度≥0.8 加权）。

    返回 (label, mean_conf_of_dominant) 或 None（无种子/无高置信帧）。
    """
    p = SEEDS_DIR / f"{name}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    labels = d["frame_labels"]
    conf = d["frame_confidence"]
    # 对齐到视图帧长（种子帧数 = 原生帧数；视图可能重采样过）
    if len(labels) != t_view:
        idx = np.linspace(0, len(labels) - 1, t_view).round().astype(int)
        labels = labels[idx]
        conf = conf[idx]
    keep = conf >= 0.8
    if not keep.any():
        return None
    lab_k, conf_k = labels[keep], conf[keep].astype(np.float64)
    votes: dict[str, float] = {}
    counts: dict[str, int] = {}
    for l, c in zip(lab_k.tolist(), conf_k.tolist()):
        votes[l] = votes.get(l, 0.0) + float(c)
        counts[l] = counts.get(l, 0) + 1
    dom = max(votes, key=votes.get)
    return dom, votes[dom] / max(counts[dom], 1)


def load_clip_skeleton(name: str, cfg: dict) -> np.ndarray | None:
    """加载原始骨架并重采样到 TARGET_T → (T,24,3) float32；无效 clip 返回 None。"""
    data_root = cfg["data_root"]
    path = Path(data_root) / f"{name}.npz"
    if not path.exists():
        return None
    try:
        clip = load_clip(path)
    except Exception:
        return None
    kp = clip["kp_world"]
    if not np.isfinite(kp).all():
        return None
    return resample_to_fixed_t(kp, target_t=TARGET_T)


def smq_segment_boundaries(seg, view: np.ndarray, ckpt: Path,
                           min_seg_len: int) -> list[tuple[int, int]]:
    """SMQ 推理 → 同码连续段（原生帧长坐标系，重采样到视图帧长比例）。"""
    indices = seg.infer_indices(view, ckpt)          # (T_view,) motion words
    return segmentation_from_indices(indices, min_len=min_seg_len)


def unit_pool_features(extractor: STGCNBCFeatureExtractor,
                       kp: np.ndarray,
                       units: list[tuple[int, int]]) -> np.ndarray:
    """段/等分单元 → 单元级特征 (n_units, D)。

    每个单元内骨架帧平均为 (T_unit,24,3)（保持单元级时序形态前先重采样到
    TARGET_T，短单元上采样、长单元下采样），过冻结特征器。
    """
    feats = []
    for (s, e) in units:
        seg_kp = kp[s:e]
        if len(seg_kp) == 0:
            seg_kp = kp[max(0, s - 1):max(1, e)]
        seg_kp = resample_to_fixed_t(seg_kp, target_t=TARGET_T)
        feats.append(extractor.extract(seg_kp[None])[0])
    return np.vstack(feats)


def even_units(n_frames: int, n_units: int) -> list[tuple[int, int]]:
    """把 T 帧均匀切 n_units 段（与 SMQ 段数容量对齐的对照切分）。"""
    n_units = max(1, min(n_units, n_frames))
    bounds = np.linspace(0, n_frames, n_units + 1).round().astype(int)
    return [(int(bounds[i]), int(bounds[i + 1])) for i in range(n_units)
            if bounds[i + 1] > bounds[i]]


def aggregate_clip_scores(unit_probs: np.ndarray, unit_labels: list[str],
                          class_names: list[str]) -> np.ndarray:
    """单元后验按类名对齐求均值 → clip 级类别分布。"""
    idx = {c: i for i, c in enumerate(class_names)}
    out = np.zeros(len(class_names), dtype=np.float64)
    for p, l in zip(unit_probs, unit_labels):
        out[idx[l]] += 1.0
    out /= max(len(unit_labels), 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="最小 clip 数管线验证")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--limit", type=int, default=None, help="限制 clip 数（调试）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    cfg = load_config()
    ckpt = Path(args.ckpt)
    assert ckpt.exists(), f"SMQ checkpoint 不存在: {ckpt}"

    names = list_clips(cfg)
    if args.smoke:
        names = names[:12]
        print(f"[smoke] 仅用前 {len(names)} 个 clip 验证管线（数字无意义）")
    elif args.limit:
        names = names[:args.limit]
    print(f"[data] clips={len(names)}")

    # 1) 构建 clip 级伪标签（种子主导类）
    clip_items: list[dict] = []
    for name in names:
        kp = load_clip_skeleton(name, cfg)
        if kp is None:
            continue
        lab = clip_label_from_seeds(name, len(kp))
        if lab is None:
            continue
        clip_items.append({"name": name, "kp": kp, "label": lab[0]})
    label_vocab = sorted({it["label"] for it in clip_items})
    print(f"[labels] 有伪标签 clips={len(clip_items)} 类别={label_vocab}")
    assert len(clip_items) >= 4 * len(label_vocab), "有效样本过少，无法分层切分"

    # 2) 确定性 train/val 切分（按类分层，每类最后 1/3 作 val）
    rng = np.random.default_rng(args.seed)
    train_items, val_items = [], []
    by_cls: dict[str, list[dict]] = {}
    for it in clip_items:
        by_cls.setdefault(it["label"], []).append(it)
    for c, items in by_cls.items():
        order = rng.permutation(len(items))
        n_val = max(1, len(items) // 3)
        val_idx = set(order[:n_val].tolist())
        for i, it in enumerate(items):
            (val_items if i in val_idx else train_items).append(it)
    print(f"[split] train={len(train_items)} val={len(val_items)}")

    # 3) 冻结特征器（p07 同款，warm checkpoint）
    from run_p07_endtoend_ak import Y_CKPT  # noqa: E402  # 复用路径常量

    y_ckpt = Path(Y_CKPT) if Path(Y_CKPT).is_absolute() else REPO_ROOT / Y_CKPT
    assert y_ckpt.exists(), f"ST-GCN+BC warm ckpt 不存在: {y_ckpt}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = STGCNBCFeatureExtractor.from_checkpoint(y_ckpt, device=device,
                                                        num_classes=22)

    # 4) SMQ segmenter
    from psd.training.smq_runner import SMQSegmenter  # noqa: E402

    seg = SMQSegmenter(
        in_channels=cfg["in_channels"], filters=cfg["filters"],
        num_layers=cfg["num_layers"], latent_dim=cfg["latent_dim"],
        num_actions=cfg["num_actions"], num_joints=cfg["num_joints"],
        num_person=cfg["num_person"], patch_size=cfg["patch_size"],
        decay=cfg["decay"], kmeans=cfg["kmeans"], kmeans_metric=cfg["kmeans_metric"],
        sampling_quantile=cfg["sampling_quantile"],
        replacement_strategy=cfg["replacement_strategy"],
    )

    feats_dir = REPO_ROOT / cfg["work_root"] / "features_all"

    # 两臂共享 SMQ 段数：with_smq 用同码连续段，without_smq 用等分对齐段数
    def featurize_pair(items: list[dict]):
        out_w, out_wo, labels = [], [], []
        for it in items:
            kp = it["kp"]                     # (T,24,3)，已重采样到 TARGET_T
            view = clip_to_smq_view(kp, np.ones(kp.shape[:2], dtype=np.float32),
                                    target_t=None,
                                    normalize=cfg.get("norm_mode", "frame"))
            bounds = smq_segment_boundaries(seg, view, ckpt, cfg["min_seg_len"])
            units_w = bounds or [(0, len(kp))]
            units_wo = even_units(len(kp), max(1, len(units_w)))
            out_w.append(unit_pool_features(extractor, kp, units_w))
            out_wo.append(unit_pool_features(extractor, kp, units_wo))
            labels.append(it["label"])
        return out_w, out_wo, labels

    tr_w, tr_wo, tr_y = featurize_pair(train_items)
    va_w, va_wo, va_y = featurize_pair(val_items)

    # 5) 下游分类：单元级 LR → clip 级均值聚合
    from sklearn.linear_model import LogisticRegression

    def train_eval(tr_feats, va_feats, tr_y, va_y):
        Xtr = np.vstack(tr_feats)
        ytr = np.array([l for f, l in zip(tr_feats, tr_y) for _ in range(len(f))])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xtr, ytr)
        # clip 级聚合: 单元后验均值 → argmax
        correct = 0
        preds = []
        for f, y in zip(va_feats, va_y):
            probs = clf.predict_proba(f)
            agg = probs.mean(axis=0)
            pred = clf.classes_[int(agg.argmax())]
            preds.append(pred)
            correct += int(pred == y)
        return correct / max(len(va_y), 1), preds

    acc_w, _ = train_eval(tr_w, va_w, tr_y, va_y)
    acc_wo, _ = train_eval(tr_wo, va_wo, tr_y, va_y)

    out = {
        "experiment": "smq_downstream_ablation",
        "mode": "smoke" if args.smoke else "full",
        "ckpt": str(ckpt.relative_to(REPO_ROOT)),
        "n_clips_labeled": len(clip_items),
        "n_train": len(train_items), "n_val": len(val_items),
        "label_vocab": label_vocab,
        "val_acc_with_smq": round(acc_w, 4),
        "val_acc_without_smq": round(acc_wo, 4),
        "delta": round(acc_w - acc_wo, 4),
        "protocol_notes": [
            "clip 标签=规则种子伪 GT 主导类（conf>=0.8 帧众数），公开真实层口径",
            "with_smq 单元=SMQ 同码连续段; without_smq 单元=等长切分(段数对齐)",
            "单元特征=冻结 ST-GCN+BC (p07 warm) penultimate; 下游=单元级 LR + clip 级均值聚合",
        ],
        "duration_sec": round(time.time() - t0, 1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
