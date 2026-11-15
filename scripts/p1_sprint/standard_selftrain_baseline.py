"""P1 冲刺实验 4 — 标准 confidence-based self-training 基线（NTU60, 10% 标注预算）.

问题（PR-D-26-13140 改造方向 ②的对照面）: PSD 框架的自训练使用
「锚点原型 + 头路共识 + 频率感知阈值（τ·prior^α）」等 PSD 特有机制。
审稿人需要一个**领域无关的标准基线**：FixMatch/Vanilla-ST 式的
confidence thresholding + 伪标签迭代，不含任何 PSD 专用组件。

标准基线协议（教科书口径）:
  1. 10% 分层种子（与 p14 同 mask 生成逻辑）训练初始线性头（256d 冻结特征）;
  2. 每轮: 对未标注池推理 → 取 max-softmax ≥ τ 的样本 → 用其 argmax 伪标签
     并入训练集 → 重训头;
  3. τ 固定 0.95（FixMatch 默认），另报 τ=0.9/0.8 敏感性;
  4. 停止: 无新样本过阈（自然收敛）或 max_iters;
  5. 评估: val top-1 / macro-F1（run_p07.evaluate 同口径）。

与 PSD 自训练的可比性: 同种子、同预算、同特征、同下游头，仅自训练机制不同。
该基线数字用于证明/量化 PSD 机制在标准方法之上的增量，属论文必备对照。

复用声明:
  - 特征 npz 与 ScaledLR: run_p14_ntu_lowres（协议一致性的关键——同一特征、
    同一头型，只换自训练机制）;
  - 分层 10% 采样: run_p14_ntu_lowres.stratified_10pct;
  - 评估: run_p07_endtoend_ak.evaluate。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/standard_selftrain_baseline.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/standard_selftrain_baseline.py --seed 42
产出:
    runs/p1_sprint/standard_selftrain_baseline/summary.json
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

from run_p07_endtoend_ak import evaluate  # noqa: E402
from run_p14_ntu_lowres import (  # noqa: E402
    CLASS_NAMES,
    NPZ,
    stratified_10pct,
    train_linear_head_scaled,
)

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "standard_selftrain_baseline"


def softmax_logits(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def standard_selftrain(tr_feat, tr_lab, va_feat, va_lab, anchor_mask, *,
                       tau: float, max_iters: int, n_cls: int, smoke: bool):
    """标准 confidence-based self-training 迭代。

    返回 dict: 每轮记录 + 最终头 + 最终 train_mask + 伪标签数。
    """
    train_mask = anchor_mask.copy()
    y_train = tr_lab.copy()          # 伪标签直接覆盖池样本标签（伪标签语义）
    rounds = []
    clf = None
    for it in range(1, max_iters + 1):
        clf = train_linear_head_scaled(tr_feat, y_train, train_mask, n_cls)
        if clf is None:
            break
        pool_idx = np.where(~train_mask)[0]
        # 池样本后验（StandardScaler+LR.predict_proba）
        proba = clf.clf.predict_proba(clf.sc.transform(tr_feat[pool_idx]))
        conf = proba.max(axis=1)
        pred = clf.clf.classes_[proba.argmax(axis=1)].astype(np.int64)
        accept = conf >= tau
        n_accept = int(accept.sum())
        rounds.append({
            "iter": it,
            "n_train": int(train_mask.sum()),
            "n_pool": int(len(pool_idx)),
            "n_accepted": n_accept,
            "accept_rate": round(n_accept / max(len(pool_idx), 1), 4),
            "mean_conf_accepted": round(float(conf[accept].mean()), 4) if n_accept else None,
        })
        print(f"  [iter {it}] train={train_mask.sum()} pool={len(pool_idx)} "
              f"accepted={n_accept} (tau={tau})")
        if n_accept == 0:
            rounds[-1]["stop"] = "no-sample-above-tau"
            break
        # 接受样本并入训练集（硬伪标签）
        new_idx = pool_idx[accept]
        train_mask[new_idx] = True
        y_train[new_idx] = pred[accept]
        if smoke and it >= 2:
            rounds[-1]["stop"] = "smoke-budget"
            break
    else:
        rounds[-1]["stop"] = "max-iters"
    return {"clf": clf, "train_mask": train_mask, "y_train": y_train,
            "rounds": rounds, "n_pseudo": int((train_mask & ~anchor_mask).sum())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="最小迭代/子采样验证管线")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-frac", type=float, default=0.10)
    ap.add_argument("--tau", type=float, default=0.95, help="FixMatch 默认 0.95")
    ap.add_argument("--tau-sensitivity", default="0.8,0.9",
                    help="附加敏感性 τ 列表（仅 full 模式跑）")
    ap.add_argument("--max-iters", type=int, default=10)
    args = ap.parse_args()

    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        tri = rng.choice(len(tr_feat), size=3000, replace=False)
        vai = rng.choice(len(va_feat), size=1500, replace=False)
        tr_feat, tr_lab = tr_feat[tri], tr_lab[tri]
        va_feat, va_lab = va_feat[vai], va_lab[vai]
        print(f"[smoke] 子采样 train={len(tr_feat)} val={len(va_feat)}（数字无意义）")

    n_cls = int(max(tr_lab.max(), va_lab.max()) + 1)
    print(f"[data] train={tr_feat.shape} val={va_feat.shape} classes={n_cls}")

    # 10% 分层种子（p14 同款函数；smoke 下按 budget-frac 等比）
    anchor = stratified_10pct(tr_lab, args.seed) if not args.smoke else _strat_frac(
        tr_lab, args.budget_frac, args.seed)
    print(f"[budget] {args.budget_frac:.0%} subset: {int(anchor.sum())} clips")

    # 参照臂: 10% 纯监督线性头（无自训练）
    clf_a = train_linear_head_scaled(tr_feat, tr_lab, anchor, n_cls)
    ev_a = evaluate(clf_a, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[ref] linear@10%: top1={ev_a['top1']} macroF1={ev_a['macro_f1']}")

    # 标准自训练基线（主 τ）
    print(f"[std-st] tau={args.tau} max_iters={args.max_iters}")
    r = standard_selftrain(tr_feat, tr_lab, va_feat, va_lab, anchor,
                           tau=args.tau, max_iters=args.max_iters,
                           n_cls=n_cls, smoke=args.smoke)
    ev_b = evaluate(r["clf"], va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[std-st] top1={ev_b['top1']} macroF1={ev_b['macro_f1']} "
          f"pseudo={r['n_pseudo']} stop={r['rounds'][-1]['stop']}")

    sensitivity = {}
    if not args.smoke:
        for tau_s in [float(x) for x in args.tau_sensitivity.split(",") if x.strip()]:
            rs = standard_selftrain(tr_feat, tr_lab, va_feat, va_lab, anchor,
                                    tau=tau_s, max_iters=args.max_iters,
                                    n_cls=n_cls, smoke=False)
            evs = evaluate(rs["clf"], va_feat, va_lab, np.ones(len(va_lab), bool),
                           CLASS_NAMES)
            sensitivity[str(tau_s)] = {
                "top1": evs["top1"], "macro_f1": evs["macro_f1"],
                "n_pseudo": rs["n_pseudo"], "stop": rs["rounds"][-1]["stop"]}
            print(f"[std-st tau={tau_s}] top1={evs['top1']} pseudo={rs['n_pseudo']}")

    out = {
        "experiment": "standard_selftrain_baseline_ntu60",
        "mode": "smoke" if args.smoke else "full",
        "seed": args.seed, "budget_frac": args.budget_frac,
        "tau": args.tau, "max_iters": args.max_iters,
        "n_train": int(len(tr_feat)), "n_val": int(len(va_feat)),
        "n_anchor": int(anchor.sum()),
        "ref_linear10_top1": ev_a["top1"], "ref_linear10_macro_f1": ev_a["macro_f1"],
        "std_st_top1": ev_b["top1"], "std_st_macro_f1": ev_b["macro_f1"],
        "std_st_n_pseudo": r["n_pseudo"], "std_st_stop": r["rounds"][-1]["stop"],
        "std_st_rounds": r["rounds"],
        "tau_sensitivity": sensitivity,
        "delta_vs_ref": round(ev_b["top1"] - ev_a["top1"], 4),
        "protocol_notes": [
            "标准 confidence-based self-training: max-softmax>=τ 硬伪标签迭代, "
            "τ=0.95 (FixMatch 默认), 无 PSD 专用组件（无原型/共识/频率阈值）",
            "特征/头/预算与 p14 完全一致（ScaledLR + 10% 分层种子），仅自训练机制不同",
            "停止: 无样本过阈 或 max_iters; precision_stop 不适用（无 oracle）",
        ],
        "duration_sec": round(time.time() - t0, 1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_path}")


def _strat_frac(labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """任意比例的分层采样（p14.stratified_10pct 的参数化版，smoke 用）。"""
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * frac)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    return mask


if __name__ == "__main__":
    main()
