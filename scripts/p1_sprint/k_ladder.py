"""P1 补强 E3b — k 阶梯: decoupled 新标注预算梯度（回应「解耦残余价值」主张）.

动机（投稿最后一轮待办 A）: E3（split_addition_episode）在 k=10 新标注/类 时
decoupled(45.75%) 远逊 coupled(97.13%)。本脚本在完全相同的合成 split+addition
episode 上做新标注预算阶梯 k ∈ {10,25,50,100}/类，回答: decoupled 是否在某个
k* 达到 coupled 平价（±2.3pp 带）？这决定「解耦残余价值」主张是实证成立还是
如实降级为未来工作。

三臂定义（与 E3 同源；episode 数据固定 seed=42 构建一次，臂种子只变训练随机性，
与 split_addition_episode 协议一致）:
  coupled(k=full) : AimCLRFinetune 从零端到端重训于 Y' 全量标注 —— k 无关的
                    平价参照（coupled 全量重训）。
  frozen_head     : 冻结 Y-era 预训练 encoder + 线性头，Y' 全量标注 —— 冻结路径
                    上限参照（k 无关）。
  decoupled(k)    : 冻结 Y-era 预训练 encoder + 语义层重建——旧 19 类用 remap 后
                    既有标注（零新增成本），10 个新类各恰好 k 条新鲜标注（逐类
                    分层抽取、独立 rng 流；区别于 E3 的合并池近似抽样）。

阶梯读数: 每个 k 档给出三臂 val top1（整体 + old19/children6/additions4 分组）
与墙钟秒；k* 判定 = min{ k : acc(coupled) − acc(decoupled(k)) ≤ 2.3pp }；
若测试范围内无解则如实报 no_parity_within_tested_range。

与 E3 的差异（披露）: 标注池 spc 50→100（支持 k=100 档需每新类 ≥100 条可抽），
epochs/lr/batch/val 口径与 E3 一致；k=10 档不是 E3 复现（池不同），仅供趋势参照。

复用声明（不重复造轮子）: episode 构造/分组评估/常量全部 import 自
split_addition_episode.py；encoder 构建与预训练 ckpt 加载复用
frozen_backbone_contrast.build_arm；训练循环契约 = psd.training.STGCNBCTrainer。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/k_ladder.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/k_ladder.py --seeds 42
    .venv/Scripts/python.exe scripts/p1_sprint/k_ladder.py --seeds 42,43,44
产出:
    runs/p1_sprint/k_ladder/summary.json
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
sys.path.insert(0, str(REPO_ROOT / "scripts" / "p1_sprint"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import split_addition_episode as sae  # noqa: E402  复用 E3 episode 构造/评估/常量
from frozen_backbone_contrast import PRETRAINED_CKPT, build_arm  # noqa: E402
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "k_ladder"

K_VALUES_DEFAULT = [10, 25, 50, 100]   # 新标注预算阶梯（每新类新鲜标注数）
PARITY_BAND_PP = 2.3                    # coupled 平价带（百分点）
METRIC_KEYS = ("top1_overall", "top1_old19", "top1_children6", "top1_additions4")


# ---------------------------------------------------------------------------
# decoupled 臂的新鲜标注抽取: 每个新类恰好 k 条（分层、确定性）
# ---------------------------------------------------------------------------
def sample_new_per_class(train_groups: dict[str, list[dict]], k: int,
                         seed: int) -> list[dict]:
    """从 10 个新类（6 拆分子类 + 4 新增类）各抽恰好 k 条新鲜标注。

    逐类独立顺序消费同一 rng 流（seed+99，与 E3 decoupled 同源种子偏移），
    保证同 seed 同 k 下抽取可复现；take=min(k, 类池大小) 天然截断。
    """
    rng = np.random.default_rng(seed + 99)
    picked: list[dict] = []
    for group in ("children6", "additions4"):
        by_label: dict[int, list[dict]] = {}
        for s in train_groups[group]:
            by_label.setdefault(int(s["label"]), []).append(s)
        for lbl in sorted(by_label):
            pool = by_label[lbl]
            take = min(k, len(pool))
            idx = rng.choice(len(pool), size=take, replace=False)
            picked.extend(pool[i] for i in idx)
    return picked


def run_arm_kl(arm: str, k: int | None, seed: int, cfg: TrainConfig,
               train_groups: dict[str, list[dict]], val: list[dict]) -> dict:
    """单臂单档运行: 构建 → 训练 → 分组评估。k=None 表示 k 无关参照臂。

    与 sae.run_arm 的差异仅在 decoupled 抽样改为逐类恰好 k 条；coupled /
    frozen_head 逻辑与 E3 完全一致（全量标注）。
    """
    t0 = time.time()
    torch.manual_seed(seed)

    if arm == "coupled":
        model = build_arm("scratch", seed, num_classes=sae.N_YPRIME)
        train = (train_groups["old19"] + train_groups["children6"]
                 + train_groups["additions4"])
        new_labels_used = len(train_groups["children6"]) + len(train_groups["additions4"])
    elif arm == "frozen_head":
        model = build_arm("frozen", seed, num_classes=sae.N_YPRIME)
        train = (train_groups["old19"] + train_groups["children6"]
                 + train_groups["additions4"])
        new_labels_used = len(train_groups["children6"]) + len(train_groups["additions4"])
    elif arm == "decoupled":
        assert k is not None, "decoupled 臂必须给出预算档 k"
        model = build_arm("frozen", seed, num_classes=sae.N_YPRIME)
        new_samples = sample_new_per_class(train_groups, k, seed)
        train = train_groups["old19"] + new_samples
        new_labels_used = len(new_samples)
    else:
        raise ValueError(arm)

    total_train = len(train)
    trainer = STGCNBCTrainer(model, train, val, cfg)
    trainer.fit()
    res = sae.evaluate_groups(trainer.model, val, str(trainer.device))
    res.update({
        "arm": arm,
        "k": k,  # None = k 无关参照臂（coupled/frozen_head 全量标注）
        "seed": seed,
        "trainable_params": int(sae.count_trainable(trainer.model)),
        "new_labels_used": int(new_labels_used),
        "total_train_clips": int(total_train),
        "duration_sec": round(time.time() - t0, 1),
    })
    return res


# ---------------------------------------------------------------------------
# 聚合与 k* 判定
# ---------------------------------------------------------------------------
def _mean(xs: list[float]) -> float:
    return round(float(np.mean(xs)), 4)


def _std(xs: list[float]) -> float:
    return round(float(np.std(xs)), 4) if len(xs) > 1 else 0.0


def aggregate(rows: list[dict]) -> dict:
    """同 (arm, k) 多 seed 聚合: 指标 mean/std + 首个 seed 的成本字段。"""
    out: dict = {"n_seeds": len(rows),
                 "seeds": sorted({int(r["seed"]) for r in rows})}
    for key in METRIC_KEYS:
        vals = [float(r[key]) for r in rows if r.get(key) is not None]
        out[key] = _mean(vals) if vals else None
        out[f"{key}_std"] = _std(vals) if vals else None
    out["new_labels_used"] = int(np.mean([r["new_labels_used"] for r in rows]))
    out["total_train_clips"] = int(np.mean([r["total_train_clips"] for r in rows]))
    out["duration_sec"] = round(float(np.mean([r["duration_sec"] for r in rows])), 1)
    out["duration_sec_per_seed"] = {str(r["seed"]): r["duration_sec"] for r in rows}
    return out


def find_k_star(ladder: list[dict], band_pp: float) -> dict:
    """k* = 最小的 k 使 acc(coupled) − acc(decoupled) ≤ band（种子均值口径）。

    无解时如实返回 k_star=None + verdict=no_parity_within_tested_range。
    """
    margins = {}
    k_star = None
    for entry in ladder:
        c = entry["arms"]["coupled"]["top1_overall"]
        d = entry["arms"]["decoupled"]["top1_overall"]
        f = entry["arms"]["frozen_head"]["top1_overall"]
        if c is None or d is None:
            continue
        margin_pp = round((c - d) * 100.0, 2)
        margins[str(entry["k"])] = {
            "coupled_minus_decoupled_pp": margin_pp,
            "decoupled_minus_frozen_head_pp": (round((d - f) * 100.0, 2)
                                               if f is not None else None),
            "within_parity_band": bool(margin_pp <= band_pp),
        }
        if k_star is None and margin_pp <= band_pp:
            k_star = entry["k"]
    if k_star is not None:
        verdict = f"decoupled_reaches_coupled_parity_at_k={k_star}"
    else:
        verdict = "no_parity_within_tested_range"
    return {"parity_band_pp": band_pp, "margin_per_k": margins,
            "k_star": k_star, "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="最小步数管线验证（数字无意义）")
    ap.add_argument("--seeds", default="42", help="逗号分隔，如 42 或 42,43,44")
    ap.add_argument("--k-values", default=",".join(str(k) for k in K_VALUES_DEFAULT),
                    help="逗号分隔的新标注预算阶梯（每新类条数）")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--samples-per-class", type=int, default=100,
                    help="episode 标注池（每类；需 ≥ max(k) 以支持最高档）")
    ap.add_argument("--val-samples-per-class", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--parity-band-pp", type=float, default=PARITY_BAND_PP)
    ap.add_argument("--arms", default="coupled,decoupled,frozen_head",
                    help="逗号分隔: coupled,decoupled,frozen_head 的任意子集")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    k_values = sorted({int(k) for k in args.k_values.split(",") if k.strip()})
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in arms:
        assert a in ("coupled", "decoupled", "frozen_head"), f"未知臂: {a}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        epochs, spc, vspc = 2, 8, 2
        if max(k_values) > spc:
            k_values = [k for k in k_values if k <= spc] or [2]
            print(f"[smoke] k 阶梯截断到池内: {k_values}")
        print("[smoke] 冒烟模式: epochs=2, spc=8（仅验证管线通，数字无意义）")
    else:
        epochs, spc, vspc = args.epochs, args.samples_per_class, args.val_samples_per_class
        assert max(k_values) <= spc, (
            f"最高预算档 k={max(k_values)} 超过标注池 spc={spc}，"
            f"请提高 --samples-per-class 或降低 k 阶梯")

    assert PRETRAINED_CKPT.exists(), f"预训练 ckpt 缺失: {PRETRAINED_CKPT}"

    t_build = time.time()
    # episode 数据固定 seed=42 构建一次（与 E3 协议一致: 臂种子只变训练随机性）
    train_groups, val = sae.build_episode(spc, vspc, seed=42)
    build_sec = round(time.time() - t_build, 1)
    print(f"[episode] Y(22) -> Y'({sae.N_YPRIME}): "
          f"old19={len(train_groups['old19'])} "
          f"children6={len(train_groups['children6'])} "
          f"additions4={len(train_groups['additions4'])} "
          f"val={len(val)} spc={spc} k ladder={k_values} device={device} "
          f"(build {build_sec}s)", flush=True)

    rows: list[dict] = []
    ref_by_seed: dict[int, dict[str, dict]] = {}   # seed -> arm -> row（k 无关参照）
    for seed in seeds:
        cfg = TrainConfig(
            epochs=epochs, batch_size=args.batch_size, lr=args.lr,
            val_interval=1, save_interval=max(epochs, 1), early_stopping=False,
            use_amp=(device == "cuda"), device=device,
            output_dir=str(OUT_DIR / "trainer_artifacts"), seed=seed,
            samples_per_class=spc,
        )
        ref_by_seed[seed] = {}
        # 参照臂（coupled 全量重训 / frozen_head 全量）每 seed 各跑一次，k 无关
        for arm in ("coupled", "frozen_head"):
            if arm not in arms:
                continue
            row = run_arm_kl(arm, None, seed, cfg, train_groups, val)
            rows.append(row)
            ref_by_seed[seed][arm] = row
            print(f"[s{seed}][{arm}][full] top1={row['top1_overall']} "
                  f"new_labels={row['new_labels_used']} ({row['duration_sec']}s)",
                  flush=True)
        # decoupled 阶梯
        if "decoupled" in arms:
            for k in k_values:
                row = run_arm_kl("decoupled", k, seed, cfg, train_groups, val)
                rows.append(row)
                print(f"[s{seed}][decoupled][k={k}] top1={row['top1_overall']} "
                      f"new_labels={row['new_labels_used']} ({row['duration_sec']}s)",
                      flush=True)

    # ---- 每 k 每臂表: 参照臂数值按 seed 复用（k 无关，显式标注） ----
    ladder = []
    for k in k_values:
        entry: dict = {"k": k, "arms": {}}
        for arm in ("coupled", "decoupled", "frozen_head"):
            if arm not in arms:
                continue
            if arm == "decoupled":
                arm_rows = [r for r in rows if r["arm"] == "decoupled" and r["k"] == k]
            else:
                arm_rows = [ref_by_seed[s][arm] for s in seeds if arm in ref_by_seed[s]]
            agg = aggregate(arm_rows)
            agg["k_independent_reference"] = arm != "decoupled"
            entry["arms"][arm] = agg
        ladder.append(entry)

    k_star_analysis = (find_k_star(ladder, args.parity_band_pp)
                       if "decoupled" in arms and "coupled" in arms else None)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "experiment": "k_ladder_decoupled_budget",
        "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "episode": {
            "Y": "22 classes (psd.data.synth_stgcn.ALL_BEHAVIORS_22)",
            "split_parents": sae.SPLIT_PARENTS,
            "children": sae.CHILDREN,
            "additions": sae.ADDITIONS,
            "Y_prime": sae.YPRIME_NAMES,
            "split_delta": 0.06,
            "pool_samples_per_class": spc,
            "val_samples_per_class": vspc,
            "note": "与 E3 同一 episode 定义; 池 spc=100 以支持 k=100 档",
        },
        "config": {
            "epochs": epochs, "spc": spc, "vspc": vspc,
            "k_values": k_values, "arms": arms, "seeds": seeds,
            "batch_size": args.batch_size, "lr": args.lr,
            "parity_band_pp": args.parity_band_pp,
            "episode_data_seed": 42,
            "decoupled_sampling": "exact k per new class (stratified, rng seed+99)",
        },
        "pretrained_ckpt": str(PRETRAINED_CKPT.relative_to(REPO_ROOT)),
        "episode_build_sec": build_sec,
        "ladder": ladder,
        "k_star_analysis": k_star_analysis,
        "runs": rows,
        "protocol_notes": [
            "coupled/frozen_head 消费 Y' 全量标注（k 无关参照，每 seed 各一次，数值按 k 档复用）",
            "decoupled: 旧 19 类 remap 既有标注零新增 + 每新类恰好 k 条新鲜标注",
            "k* 判定: min{k: acc(coupled)-acc(decoupled(k)) <= 2.3pp}（种子均值）",
            "episode 数据固定 seed=42 构建一次; 臂种子只变训练随机性（与 E3 一致）",
            "与 E3 差异: 池 spc 50->100（k=100 档需要），k=10 档仅供趋势参照非 E3 复现",
            "clip 生成复刻 psd/data/synth_stgcn 配方（sin+高斯），模板参数化",
        ],
    }
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")
    if k_star_analysis:
        print(f"[k*] verdict={k_star_analysis['verdict']} "
              f"k_star={k_star_analysis['k_star']}")
    print("K_LADDER_DONE")


if __name__ == "__main__":
    main()
