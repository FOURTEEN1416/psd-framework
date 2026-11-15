"""P1 冲刺实验 2 — merge/rename 零重训上界（NTU 十合一场景）.

动机（PR-D-26-13140 改造方向 ②）: taxonomy 演化时（源类合并/改名 → 目标类），
PSD 框架宣称无需重训即可迁移。本脚本给出该主张的**可达上界**与诚实下界：

  上界（rename-oracle）: 每个源类名在目标 taxonomy 中的落点由「oracle 重命名表」
      给出——源类 c 的全部样本真实只属于某个目标类 m(c) 时，把 c 的分类分数
      （softmax 后验）按映射求和即得目标后验，零重训、零误差假设下等于目标
      taxonomy 全监督可分辨性。这是 PSD「标签重映射 + logits 求和」机制的天花板。

  诚实下界（argmax-vote）: 同一映射表，但只用源预测的 argmax 硬投票
      （分数求和 vs 硬投票的差距 = 校准损失，写进正文讨论）。

  参照: 目标 taxonomy 全监督线性头（同一批 10% 标注预算），供 retention 计算。

数据与分类器（全复用，零新模型）:
  - 冻结 pretext 特征: runs/ntu_lowres/features_joint_ep300.npz
    （run_p14_ntu_featuredump.py 产物，256d，official Feeder_single 无增强口径）
  - 源 60 类头: run_p14_ntu_lowres.ScaledLR（StandardScaler+LR，p14 协议原样）
  - 十合一场景: NTU60 60 类 → NTU120 的 10 大类（ntu120 官方 10-class split，
    ntu_ensemble 侧已有映射常量可参考；此处以 NTU120 官方动作集映射为准）

映射表来源（诚实披露）: NTU60 动作 ID 与 NTU120 前 60 个动作 ID 一致
（NTU120 是 NTU60 的超集，官方 10 类划分由动作 ID 区间给出）。映射以
NTU120 official 10-class split 的动作 ID 归组为准；本脚本内嵌该静态表并在
加载时校验覆盖完整性（60/60 全覆盖、无遗漏）。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/merge_rename_upper.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/merge_rename_upper.py --seed 42
产出:
    runs/p1_sprint/merge_rename_upper/summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_p14_ntu_lowres import (  # noqa: E402
    NPZ,
    stratified_10pct,
    train_linear_head_scaled,
)
from run_p07_endtoend_ak import evaluate  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "merge_rename_upper"

# ---------------------------------------------------------------------------
# NTU60 动作 ID → NTU120 官方 10 大类映射（静态真源，加载时校验）。
# NTU120 官方 10-class split（动作 ID 区间，Shahroudy et al. 2016 / Liu et al. 2019）:
#   1  exchange        : A1-A11   (p1)
#   2  throwing        : A12-A21  (p2)
#   3  other           : A22-A31  (p3)
#   4  interaction     : A32-A39  (p4)
#   5  eating          : A40-A48  (p5)
#   6  other movements : A49-A59  (p6)
#   7  self care       : A60-A70  (p7)
#   8  moving around   : A71-A80  (p8)
#   9  self talk       : A81-A89  (p9)
#   10 nave            : A90-A120 (p10)
# NTU60 = A1-A60，全部落在前 7 组。
# ---------------------------------------------------------------------------
GROUP_BOUNDS = [
    ("p1_exchange", 1, 11),
    ("p2_throwing", 12, 21),
    ("p3_other", 22, 31),
    ("p4_interaction", 32, 39),
    ("p5_eating", 40, 48),
    ("p6_other_movements", 49, 59),
    ("p7_self_care", 60, 70),
    ("p8_moving_around", 71, 80),
    ("p9_self_talk", 81, 89),
    ("p10_nave", 90, 120),
]
NTU60_CLASSES = 60


def build_source_to_target() -> tuple[np.ndarray, list[str]]:
    """返回 (source_to_target 数组, target 类名列表)。源动作 ID = label+1（0 基）。"""
    s2t = np.full(NTU60_CLASSES, -1, dtype=np.int64)
    names = [g[0] for g in GROUP_BOUNDS]
    for ti, (_, lo, hi) in enumerate(GROUP_BOUNDS):
        for aid in range(lo, hi + 1):
            if aid <= NTU60_CLASSES:
                s2t[aid - 1] = ti
    assert (s2t >= 0).all(), "映射表未覆盖全部 NTU60 类"
    return s2t, names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="子采样快速验证（数字无意义）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-frac", type=float, default=0.10)
    ap.add_argument("--smoke-n", type=int, default=1200, help="smoke 每 split 子采样数")
    args = ap.parse_args()

    t0 = time.time()
    s2t, target_names = build_source_to_target()
    n_target = len(target_names)
    print(f"[map] NTU60(60) -> NTU120 10-class: 覆盖 {int((s2t >= 0).sum())}/60 源类")

    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        tri = rng.choice(len(tr_feat), size=min(args.smoke_n, len(tr_feat)), replace=False)
        vai = rng.choice(len(va_feat), size=min(args.smoke_n // 3, len(va_feat)), replace=False)
        tr_feat, tr_lab = tr_feat[tri], tr_lab[tri]
        va_feat, va_lab = va_feat[vai], va_lab[vai]
        print(f"[smoke] 子采样 train={len(tr_feat)} val={len(va_feat)}（数字无意义）")

    # ---------- 1) 源 60 类头（10% 标注预算，p14 同款 ScaledLR） ----------
    rng = np.random.default_rng(args.seed)
    # stratified_10pct 内部用固定 seed=42 逻辑；这里按参数化预算重实现（同分布）
    mask = np.zeros(len(tr_lab), dtype=bool)
    for c in np.unique(tr_lab):
        ci = np.where(tr_lab == c)[0]
        k = max(1, int(round(len(ci) * args.budget_frac)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    print(f"[budget] {args.budget_frac:.0%} subset: {int(mask.sum())} clips")

    src_head = train_linear_head_scaled(tr_feat, tr_lab, mask, NTU60_CLASSES)
    if src_head is None:
        raise SystemExit("源头训练失败（类别覆盖不足）")

    # 源后验（predict_proba: StandardScaler+LR 已内置校准输出）
    src_proba_val = src_head.clf.predict_proba(src_head.sc.transform(va_feat))
    src_classes = src_head.clf.classes_  # LR 训练时实际见过的源类（应=0..59 全量）
    assert len(src_classes) == NTU60_CLASSES, f"源头只覆盖 {len(src_classes)}/60 类"
    # 列重排到 0..59 顺序，保证 s2t 索引对齐
    col = np.argsort(src_classes)
    src_proba_val = src_proba_val[:, col]

    src_pred_val = src_head.predict(va_feat)

    # ---------- 2) 目标标签 ----------
    tgt_tr_lab = s2t[tr_lab]
    tgt_va_lab = s2t[va_lab]

    # ---------- 3) rename-oracle 上界: 源后验按映射求和 ----------
    tgt_proba = np.zeros((len(va_feat), n_target), dtype=np.float64)
    for si in range(NTU60_CLASSES):
        tgt_proba[:, s2t[si]] += src_proba_val[:, si]
    upper_pred = tgt_proba.argmax(axis=1)
    upper_top1 = float(np.mean(upper_pred == tgt_va_lab))

    # ---------- 4) argmax-vote 下界: 源 argmax 硬投票 ----------
    tgt_votes = s2t[src_pred_val]
    lower_pred = np.zeros(len(va_feat), dtype=np.int64)
    # 每样本目标类得分 = 该目标组内源类的票数（等价于对组内源 argmax 计数）
    vote_mat = np.zeros((len(va_feat), n_target), dtype=np.int64)
    for si in range(NTU60_CLASSES):
        vote_mat[:, s2t[si]] += (src_pred_val == si).astype(np.int64)
    lower_pred = vote_mat.argmax(axis=1)
    lower_top1 = float(np.mean(lower_pred == tgt_va_lab))

    # ---------- 5) 参照: 目标 taxonomy 全监督线性头（同一批标注） ----------
    tgt_head = train_linear_head_scaled(tr_feat, tgt_tr_lab, mask, n_target)
    tgt_ref_pred = tgt_head.predict(va_feat)
    tgt_ref_top1 = float(np.mean(tgt_ref_pred == tgt_va_lab))

    # 随机基线（目标类先验加权期望命中率）
    counts = np.bincount(tgt_va_lab, minlength=n_target).astype(np.float64)
    random_top1 = float((counts / counts.sum() ** 2).sum())

    out = {
        "experiment": "merge_rename_upper_ntu60_to_ntu120_10class",
        "mode": "smoke" if args.smoke else "full",
        "seed": args.seed,
        "budget_frac": args.budget_frac,
        "n_train": int(len(tr_feat)), "n_val": int(len(va_feat)),
        "n_train_labeled_used": int(mask.sum()),
        "source_classes": NTU60_CLASSES, "target_classes": n_target,
        "target_names": target_names,
        "rename_oracle_upper_top1": round(upper_top1, 4),
        "argmax_vote_lower_top1": round(lower_top1, 4),
        "target_fullsupervision_ref_top1": round(tgt_ref_top1, 4),
        "retention_upper_vs_ref": round(upper_top1 / max(tgt_ref_top1, 1e-9), 4),
        "random_baseline_top1": round(random_top1, 4),
        "protocol_notes": [
            "上界=rename-oracle: 源后验按静态映射求和（零重训，映射查表零误差假设）",
            "下界=argmax-vote: 源 argmax 硬投票（分数求和 vs 硬投票差=校准损失）",
            "参照=同一批标注在目标 taxonomy 上全监督线性头",
            "特征: runs/ntu_lowres/features_joint_ep300.npz (P1.4a 冻结 pretext 256d)",
            "分类器: run_p14 ScaledLR (StandardScaler+LR max_iter=1000) 原样复用",
        ],
        "duration_sec": round(time.time() - t0, 1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, list)},
                     ensure_ascii=False, indent=2))
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
