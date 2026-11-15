"""P1 补强 E3 — 合成层 split+addition 迁移 episode（Y 22 类 → Y′ 29 类）.

动机（冷审补强清单 E3 / 贡献 1 的假设转实证）: 在纯 merge/rename 迁移上，零重训
remap 与 matched-solver 对照优于 decoupled 重建（TRANS-002 实测）；PSD 的剩余价值
主张应落在**需要新监督的迁移**（类别拆分/新增）。本脚本在完全可控的合成层构造
split+addition episode，把该主张从假设变成可检验证据：

  episode（冻结定义）:
    Y  = 22 类（psd.data.synth_stgcn.ALL_BEHAVIORS_22）
    拆分: 3 个父类（sit/track/retrieve）各拆成 2 个子类（模板加固定小偏移）
          → 6 个新子类，父类在 Y′ 中不复存在（其旧样本不可用 → 子类必须新监督）
    新增: 4 个全新行为类（新模板）
    Y′ = 19 旧类 + 6 子类 + 4 新类 = 29 类

  四臂（同数据、同预算、同 seed、同 TrainConfig）:
    coupled        : AimCLRFinetune 从零端到端重训于 Y′ 全标注（29 类 × spc）
    decoupled      : 冻结 Y-era 预训练 encoder；语义层重建——旧 19 类用 remap 后的
                     既有标注（零新增成本），10 个新类只用 new-supervision-per-class
                     条新鲜标注（可选再加 rule-seeds-per_class 弱标注种子）
    frozen_head    : 冻结 Y-era 预训练 encoder + 线性头，Y′ 全标注（冻结路径上界参照）
    remap_zero_shot: Y-era 22 类头 + 标签 remap，零新标注——新类在 remap 下无目标，
                     原理上不可命中等价于随机撞（量化「remap 覆盖不了新类」）

  关键读数: remap(0 新标注) < decoupled(k 新标注) ≤ frozen_head(全量新标注) 的
  新类精度阶梯 + 各臂 wall-clock / 新增标注成本 → decoupling 残余价值集中在
  需要新监督的迁移。

复用声明: encoder/ckpt/训练器与 frozen_backbone_contrast.py 同源（AimCLRFinetune +
STGCNBCTrainer + epoch120 预训练权重）；合成 clip 生成复刻 psd/data/synth_stgcn
的时序扰动配方（sin + 逐帧高斯），仅模板参数化。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/split_addition_episode.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/split_addition_episode.py --seeds 42
产出:
    runs/p1_sprint/split_addition_episode/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from psd.data.synth_stgcn import (  # noqa: E402
    ALL_BEHAVIORS_22,
    NUM_JOINTS,
    _behavior_pose_template,
    make_synthetic_dataset,
)
from psd.models.aimclr_finetune import AimCLRFinetune, load_pretrained_encoder  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts" / "p1_sprint"))
from frozen_backbone_contrast import (  # noqa: E402  同仓 p1_sprint 复用
    PRETRAINED_CKPT,
    FrozenHeadOnly,
    build_arm,
)

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "split_addition_episode"

# ---------------------------------------------------------------------------
# 冻结 episode 定义
# ---------------------------------------------------------------------------
SPLIT_PARENTS = ["sit", "track", "retrieve"]
# 子类偏移: 在父类模板上叠加固定小偏移（幅度 0.06，介于 noise_std=0.05 量级——
# 可学但非平凡；两子类偏移方向相反保证彼此可分）
SPLIT_DELTA_A = np.array([0.06, 0, 0] * 24, dtype=np.float32).reshape(24, 3)
SPLIT_DELTA_B = -SPLIT_DELTA_A
CHILDREN = [f"{p}_split_a" for p in SPLIT_PARENTS] + [f"{p}_split_b" for p in SPLIT_PARENTS]

# 4 个新增行为: 基础站立模板 + 手工定义偏置（与既有 22 类模板均不同）
_ADDITION_OFFSETS: dict[str, np.ndarray] = {
    "play_bow": np.array([[0, 0, -0.25]] * 3 + [[0, 0, 0]] * 6 + [[0, 0, 0.05]] * 3
                         + [[0, 0, 0]] * 12, dtype=np.float32).reshape(24, 3),
    "roll_over": np.array([[-0.12, 0, -0.05]] * 12 + [[0.12, 0, -0.05]] * 12,
                          dtype=np.float32).reshape(24, 3),
    "shake_off": np.array([[0, 0, 0]] * 12 + [[0.10, 0, 0.02]] * 2 + [[-0.10, 0, 0.02]] * 2
                          + [[0, 0, 0]] * 8, dtype=np.float32).reshape(24, 3),
    "point_freeze": np.array([[0, 0.10, 0.20], [0, 0, 0], [0, 0, 0]]
                             + [[0, 0, 0]] * 21, dtype=np.float32).reshape(24, 3),
}
ADDITIONS = list(_ADDITION_OFFSETS.keys())

_OLD_KEEP = [c for c in ALL_BEHAVIORS_22 if c not in SPLIT_PARENTS]  # 19 旧类
YPRIME_NAMES = _OLD_KEEP + CHILDREN + ADDITIONS                       # 29 类
N_OLD, N_CHILD, N_ADD = len(_OLD_KEEP), len(CHILDREN), len(ADDITIONS)
N_YPRIME = len(YPRIME_NAMES)  # 29
assert N_YPRIME == 29 and N_OLD == 19 and N_CHILD == 6 and N_ADD == 4


def make_clip(template: np.ndarray, T: int, noise_std: float, seed: int) -> np.ndarray:
    """复刻 psd/data/synth_stgcn._generate_clip 的时序扰动配方（sin + 逐帧高斯），
    模板参数化（支持子类/新增类模板）。"""
    rng = np.random.default_rng(seed)
    t = np.arange(T, dtype=np.float32) / max(T - 1, 1)
    sin_phase = rng.uniform(0, 2 * np.pi)
    sin_wave = 0.05 * np.sin(2 * np.pi * t + sin_phase)
    kp = np.zeros((T, NUM_JOINTS, 3), dtype=np.float32)
    for f in range(T):
        kp[f] = template + sin_wave[f] * 0.1
        kp[f, :, 0] += rng.normal(0, noise_std, NUM_JOINTS)
        kp[f, :, 1] += rng.normal(0, noise_std, NUM_JOINTS)
        kp[f, :, 2] += rng.normal(0, noise_std, NUM_JOINTS)
    return kp


def _sample(name: str, label: int, kp: np.ndarray, T: int) -> dict:
    boundary = np.zeros(T, dtype=np.float32)
    boundary[:2] = 1.0
    boundary[-2:] = 1.0
    return {"keypoints": kp, "label": label, "label_name": name,
            "boundary": boundary, "frame_dir": f"ep_{name}_{label}"}


def build_episode(spc: int, vspc: int, seed: int, T: int = 30,
                  noise_std: float = 0.05) -> tuple[dict[str, list[dict]], list[dict]]:
    """返回 (train 按来源分组的样本 dict, Y′ val 全集)。

    train 组: old19（Y-era 既有样本 remap，零新增）/ children6 / additions4（新模板新样本）
    """
    y_era = make_synthetic_dataset(samples_per_class=spc, T=T, noise_std=noise_std, seed=seed)
    by_name: dict[str, list[dict]] = {}
    for s in y_era:
        by_name.setdefault(s["label_name"], []).append(s)

    train: dict[str, list[dict]] = {"old19": [], "parents3": [], "children6": [], "additions4": []}
    for i, name in enumerate(_OLD_KEEP):            # 旧类: 直接复用 Y-era 样本
        for s in by_name[name]:
            train["old19"].append(_sample(name, i, s["keypoints"], T))
    for name in SPLIT_PARENTS:                      # 父类样本: 仅供 remap 臂的 Y-era 头训练
        for s in by_name[name]:
            train["parents3"].append(_sample(name, ALL_BEHAVIORS_22.index(name),
                                             s["keypoints"], T))
    rng = np.random.default_rng(seed + 7)
    for j, parent in enumerate(SPLIT_PARENTS):      # 拆分子类: 新模板新样本
        base = _behavior_pose_template(parent)
        for tag, delta in (("split_a", SPLIT_DELTA_A), ("split_b", SPLIT_DELTA_B)):
            tpl = (base + delta).astype(np.float32)
            label = N_OLD + j + (0 if tag == "split_a" else len(SPLIT_PARENTS))
            for k in range(spc):
                kp = make_clip(tpl, T, noise_std, int(rng.integers(0, 2**31 - 1)))
                train["children6"].append(_sample(f"{parent}_{tag}", label, kp, T))
    for j, name in enumerate(ADDITIONS):            # 新增类: 新模板新样本
        tpl = (_behavior_pose_template("stand") + _ADDITION_OFFSETS[name]).astype(np.float32)
        label = N_OLD + N_CHILD + j
        for k in range(spc):
            kp = make_clip(tpl, T, noise_std, int(rng.integers(0, 2**31 - 1)))
            train["additions4"].append(_sample(name, label, kp, T))

    # val: 独立种子全类新 draw
    rng_v = np.random.default_rng(seed + 1000)
    val: list[dict] = []
    for i, name in enumerate(_OLD_KEEP):
        base = _behavior_pose_template(name)
        for _ in range(vspc):
            val.append(_sample(name, i, make_clip(base, T, noise_std,
                                                  int(rng_v.integers(0, 2**31 - 1))), T))
    for j, parent in enumerate(SPLIT_PARENTS):
        base = _behavior_pose_template(parent)
        for tag, delta in (("split_a", SPLIT_DELTA_A), ("split_b", SPLIT_DELTA_B)):
            tpl = (base + delta).astype(np.float32)
            label = N_OLD + j + (0 if tag == "split_a" else len(SPLIT_PARENTS))
            for _ in range(vspc):
                val.append(_sample(f"{parent}_{tag}", label, make_clip(tpl, T, noise_std,
                                                                       int(rng_v.integers(0, 2**31 - 1))), T))
    for j, name in enumerate(ADDITIONS):
        tpl = (_behavior_pose_template("stand") + _ADDITION_OFFSETS[name]).astype(np.float32)
        label = N_OLD + N_CHILD + j
        for _ in range(vspc):
            val.append(_sample(name, label, make_clip(tpl, T, noise_std,
                                                      int(rng_v.integers(0, 2**31 - 1))), T))
    return train, val


def evaluate_groups(model: torch.nn.Module, samples: list[dict], device: str,
                    batch_size: int = 64) -> dict:
    """整体 + 分组（old19/children6/additions4）top1。"""
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            x = torch.stack([torch.as_tensor(s["keypoints"], dtype=torch.float32)
                             for s in chunk]).to(device)
            cls_logits, _ = model(x)
            preds.append(cls_logits.argmax(dim=-1).cpu())
            ys.append(torch.tensor([int(s["label"]) for s in chunk]))
    pred = torch.cat(preds).numpy()
    y = torch.cat(ys).numpy()
    out = {"top1_overall": round(float(np.mean(pred == y)), 4)}
    for g, (lo, hi) in {"old19": (0, N_OLD), "children6": (N_OLD, N_OLD + N_CHILD),
                        "additions4": (N_OLD + N_CHILD, N_YPRIME)}.items():
        m = (y >= lo) & (y < hi)
        out[f"top1_{g}"] = round(float(np.mean(pred[m] == y[m])), 4) if m.any() else None
    return out


def count_trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_arm(arm: str, seed: int, cfg: TrainConfig, train_groups: dict[str, list[dict]],
            val: list[dict], new_supervision: int, rule_seeds: int,
            noise_std: float, T: int) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)

    if arm == "remap_zero_shot":
        # Y-era 22 类头（冻结 encoder，完整 Y-era 22 类监督）+ 标签 remap: 零新标注。
        # remap 空间 = 22 旧类 -> Y′ 的 19 个旧类位置（父类/子类/新增类无目标）。
        y_era_train = []
        for s in train_groups["old19"] + train_groups["parents3"]:
            s2 = dict(s)
            s2["label"] = ALL_BEHAVIORS_22.index(s["label_name"])
            y_era_train.append(s2)
        model = build_arm("frozen", seed, num_classes=22)
        # trainer 内部 val 只用于选 checkpoint: 必须落在 22 类头标签域内，
        # 故仅喂旧 19 类 val（子类/新增类 val 的 Y′ 标签 ≥22 会触发 CE device assert）；
        # 最终指标由下方自定义列收集在完整 Y′ val 上计算。
        val_in_domain = [s for s in val if int(s["label"]) < N_OLD]
        trainer = STGCNBCTrainer(model, y_era_train, val_in_domain, cfg)
        trainer.fit()
        # 评估: 22 维得分按 _OLD_KEEP 顺序取 19 个旧类列（remap 语义），新类无列不可命中
        cols = [ALL_BEHAVIORS_22.index(name) for name in _OLD_KEEP]
        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for i in range(0, len(val), 64):
                chunk = val[i:i + 64]
                x = torch.stack([torch.as_tensor(s["keypoints"], dtype=torch.float32)
                                 for s in chunk]).to(trainer.device)
                cls_logits, _ = model(x)
                old_scores = cls_logits[:, cols]
                preds.append(old_scores.argmax(dim=-1).cpu())
                ys.append(torch.tensor([int(s["label"]) for s in chunk]))
        pred = torch.cat(preds).numpy()
        y = torch.cat(ys).numpy()
        res = {"top1_overall": round(float(np.mean(pred == y)), 4)}
        for g, (lo, hi) in {"old19": (0, N_OLD), "children6": (N_OLD, N_OLD + N_CHILD),
                            "additions4": (N_OLD + N_CHILD, N_YPRIME)}.items():
            m = (y >= lo) & (y < hi)
            res[f"top1_{g}"] = round(float(np.mean(pred[m] == y[m])), 4) if m.any() else None
        n_trainable = count_trainable(model)
        new_labels_used = 0
        total_train = len(y_era_train)
    else:
        if arm == "coupled":
            model = build_arm("scratch", seed, num_classes=N_YPRIME)
            train_samples = train_groups["old19"] + train_groups["children6"] + train_groups["additions4"]
            new_labels_used = len(train_groups["children6"]) + len(train_groups["additions4"])
        elif arm == "frozen_head":
            model = build_arm("frozen", seed, num_classes=N_YPRIME)
            train_samples = train_groups["old19"] + train_groups["children6"] + train_groups["additions4"]
            new_labels_used = len(train_groups["children6"]) + len(train_groups["additions4"])
        elif arm == "decoupled":
            model = build_arm("frozen", seed, num_classes=N_YPRIME)
            # 旧 19 类: remap 后既有标注全量（零新增成本）；新 10 类: 只给 k 条新鲜标注
            rng = np.random.default_rng(seed + 99)
            new_labeled = []
            for s in train_groups["children6"] + train_groups["additions4"]:
                new_labeled.append(s)
            idx = rng.choice(len(new_labeled), size=min(new_supervision * (N_CHILD + N_ADD),
                                                        len(new_labeled)), replace=False)
            new_labeled = [new_labeled[i] for i in idx]
            # 可选: 规则种子通道（更高噪声的弱标注 clip，模拟规则引擎种子）
            rule_seeded = []
            if rule_seeds > 0:
                rng_r = np.random.default_rng(seed + 123)
                tpl_of = {}
                for j, parent in enumerate(SPLIT_PARENTS):
                    base = _behavior_pose_template(parent)
                    tpl_of[N_OLD + j] = (base + SPLIT_DELTA_A).astype(np.float32)
                    tpl_of[N_OLD + len(SPLIT_PARENTS) + j] = (base + SPLIT_DELTA_B).astype(np.float32)
                for j, name in enumerate(ADDITIONS):
                    tpl_of[N_OLD + N_CHILD + j] = (_behavior_pose_template("stand")
                                                   + _ADDITION_OFFSETS[name]).astype(np.float32)
                for lbl, tpl in tpl_of.items():
                    for _ in range(rule_seeds):
                        kp = make_clip(tpl, T, noise_std * 2.0,
                                       int(rng_r.integers(0, 2**31 - 1)))
                        rule_seeded.append(_sample(YPRIME_NAMES[lbl], lbl, kp, T))
            train_samples = train_groups["old19"] + new_labeled + rule_seeded
            new_labels_used = len(new_labeled)
        else:
            raise ValueError(arm)

        total_train = len(train_samples)
        trainer = STGCNBCTrainer(model, train_samples, val, cfg)
        trainer.fit()
        res = evaluate_groups(trainer.model, val, str(trainer.device))
        n_trainable = count_trainable(trainer.model)

    res.update({
        "arm": arm, "seed": seed,
        "trainable_params": int(n_trainable),
        "new_labels_used": int(new_labels_used),
        "total_train_clips": int(total_train),
        "duration_sec": round(time.time() - t0, 1),
    })
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="最小步数管线验证（数字无意义）")
    ap.add_argument("--seeds", default="42", help="逗号分隔，如 42 或 42,43,44")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples-per-class", type=int, default=50)
    ap.add_argument("--val-samples-per-class", type=int, default=10)
    ap.add_argument("--new-supervision-per-class", type=int, default=10,
                    help="decoupled 臂每个新类的新鲜标注数（新监督预算 k）")
    ap.add_argument("--rule-seeds-per-class", type=int, default=0,
                    help="decoupled 臂每个新类追加的规则种子数（高噪声弱标注；0=关）")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        epochs, spc, vspc, new_sup, rule_seeds = 2, 4, 2, 2, 0
        print("[smoke] 冒烟模式: epochs=2, spc=4（仅验证管线通，数字无意义）")
    else:
        epochs, spc, vspc = args.epochs, args.samples_per_class, args.val_samples_per_class
        new_sup, rule_seeds = args.new_supervision_per_class, args.rule_seeds_per_class

    assert PRETRAINED_CKPT.exists(), f"预训练 ckpt 缺失: {PRETRAINED_CKPT}"
    train_groups, val = build_episode(spc, vspc, seed=42)
    print(f"[episode] Y(22) -> Y'({N_YPRIME}): old19={len(train_groups['old19'])} "
          f"children6={len(train_groups['children6'])} additions4={len(train_groups['additions4'])} "
          f"val={len(val)} device={device}")

    cfg = TrainConfig(
        epochs=epochs, batch_size=args.batch_size, lr=args.lr,
        val_interval=1, save_interval=max(epochs, 1), early_stopping=False,
        use_amp=(device == "cuda"), device=device,
        output_dir=str(OUT_DIR / "trainer_artifacts"), seed=42,
    )

    arms = ["coupled", "decoupled", "frozen_head", "remap_zero_shot"]
    rows = []
    for seed in seeds:
        for arm in arms:
            row = run_arm(arm, seed, cfg, train_groups, val, new_sup, rule_seeds,
                          noise_std=0.05, T=30)
            rows.append(row)
            print(f"[s{seed}][{arm}] top1={row['top1_overall']} "
                  f"new_labels={row['new_labels_used']} ({row['duration_sec']}s)", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "split_addition_episode_synthetic",
        "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "episode": {
            "Y": "22 classes (psd.data.synth_stgcn.ALL_BEHAVIORS_22)",
            "split_parents": SPLIT_PARENTS,
            "children": CHILDREN,
            "additions": ADDITIONS,
            "Y_prime": YPRIME_NAMES,
            "split_delta": 0.06,
            "note": "父类样本在 Y' 中不可用（子类必须新监督）; 新增类无 Y-era 来源",
        },
        "config": {"epochs": epochs, "spc": spc, "vspc": vspc,
                   "new_supervision_per_class": new_sup,
                   "rule_seeds_per_class": rule_seeds,
                   "batch_size": args.batch_size, "lr": args.lr, "seeds": seeds},
        "pretrained_ckpt": str(PRETRAINED_CKPT.relative_to(REPO_ROOT)),
        "runs": rows,
        "protocol_notes": [
            "coupled/frozen_head 消费 Y' 全量标注; decoupled 旧类零新增+新类 k 条新监督",
            "remap_zero_shot: Y-era 22 类头 remap 到 19 旧类，新类原理上不可命名",
            "关键读数: 新类精度阶梯 remap < decoupled(k) ≤ frozen_head(全量) + 成本对照",
            "clip 生成复刻 psd/data/synth_stgcn 配方（sin+高斯），模板参数化",
        ],
    }
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")
    print("SPLIT_ADDITION_EPISODE_DONE")


if __name__ == "__main__":
    main()
