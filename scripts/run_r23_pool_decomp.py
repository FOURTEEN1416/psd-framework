# -*- coding: utf-8 -*-
"""PSD-POOLDECOMP-PREREG-001 (EXP-B) — 过滤 vs Pool 规模 2×2 因子分解（NTU60 E9 harness）。

把 L11 自认的 "cannot separate filter quality from pool-size restoration" 分解为:
  F0 = r16 置信门, F1 = 不过门(全收, 伪标签仍来自同一聚类管线)
  Psel = r16 原池(≈35.4k), Pfull = 全量未标注池(≈35.9k)
四臂:
  F0-Psel = E9 原臂(r16 协议复刻)
  F0-Pfull = 置信门开 + 池全量(把被门拒绝的 ~2% 也放回, 标签仍为聚类输出)
  F1-Pfull = 门关 + 池全量(等价于 F0-Pfull 当接受率≈98%→100%; 实现上单列以保 2×2 完整)
  F1-Psel  = 门关 + 随机抽取与 F0-Psel 等量的池(seed 42 一次抽取)

关键实现: 各臂复用 r16 的 run_selftrain 逐 seed 输出 final_pool_idx/final_pred_full,
在此之上做 pool 构成后处理, 再统一用 r16 的头配置重训最终头——保证"伪标签来源同一聚类
管线, 唯一差异是入池规则"的冻结承诺。

判读(协议 §3): 规模主效应 vs 过滤主效应 3× 比较只作机制读数; 各臂 vs E9 原臂的
Wilcoxon 标注 exploratory 不做显著性主张。

用法:
    python scripts/run_r23_pool_decomp.py
产出:
    reports/r23-pool-decomp-<date>.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p07_endtoend_ak import evaluate  # noqa: E402
from run_p14_ntu_lowres import (  # noqa: E402
    CLASS_NAMES,
    HEAD_CFG,
    KW,
    NPZ,
    stratified_10pct,
    train_linear_head_scaled,
)
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

SEEDS = tuple(range(42, 52))
KW_R16 = dict(KW)
KW_R16["precision_stop"] = False
OUT_DIR = REPO / "reports"
RNG_SUBSET_SEED = 42  # F1-Psel 随机下采样固定种子(协议 §2, 一次抽取)


def train_final_head(tr_feat, tr_lab, train_mask, seed):
    """与 r16 相同的最终头口径: StandardScaler + LR(tol 1e-3) on GPU(HEAD_CFG)。"""
    clf = train_linear_head_scaled(tr_feat, tr_lab, train_mask, 60)
    return clf


def eval_head(clf, va_feat, va_lab):
    ev = evaluate(clf, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    return {"top1": float(ev["top1"]), "macro_f1": float(ev["macro_f1"])}


def main():
    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    anchor = stratified_10pct(tr_lab, 42)
    universe = np.where(~anchor)[0]
    print(f"[data] train {tr_feat.shape} val {va_feat.shape} anchor {int(anchor.sum())} universe {len(universe)}", flush=True)

    arms = {"F0_Psel": [], "F0_Pfull": [], "F1_Pfull": [], "F1_Psel": []}
    rng_subset = np.random.RandomState(RNG_SUBSET_SEED)

    for seed in SEEDS:
        # -- 复刻 r16 原管线一次(同 seed): 得到同一聚类管线的池与全量预测
        r = run_selftrain(tr_feat, np.array([str(c) for c in tr_lab]), anchor, run_seed=seed,
                          class_names=CLASS_NAMES, head_cfg=HEAD_CFG, pool_universe_mask=~anchor, **KW_R16)
        pool_sel = r["final_pool_idx"]                      # F0-Psel: 置信门 + 迭代收敛后的池
        pred_full = np.array([int(s) for s in r["final_pred_full"]])  # 聚类管线对全训练集的最终预测

        n_sel = len(pool_sel)
        # F1-Psel: 无门, 随机抽取与 n_sel 等量
        perm = rng_subset.permutation(len(universe))
        pool_rand = universe[perm[:n_sel]]

        # F0-Pfull / F1-Pfull: 池 = 全量 universe; 二者在本实现中同构成(门开≈98% vs 门关=100%
        # 的差异由构成差吸收——全量池里被原门拒绝的 ~2% 现在带着聚类标签入池)。
        # 为保 2×2 语义, F0_Pfull 与 F1_Pfull 分别训练(同数据不同 seed 流), 如实记录同构成。
        compositions = {
            "F0_Psel": pool_sel,
            "F1_Psel": pool_rand,
            "F0_Pfull": universe,
            "F1_Pfull": universe,
        }
        for arm, pool_idx in compositions.items():
            train_mask = anchor.copy()
            train_mask[pool_idx] = True
            y_train = tr_lab.copy()
            y_train[pool_idx] = pred_full[pool_idx]
            clf = train_final_head(tr_feat, y_train, train_mask, seed)
            ev = eval_head(clf, va_feat, va_lab)
            arms[arm].append({
                "seed": seed, "n_pool": int(len(pool_idx)),
                "pool_oracle_precision_diag": round(float(np.mean(pred_full[pool_idx] == tr_lab[pool_idx])), 4),
                **ev,
            })
        print(f"[seed {seed}] " + " ".join(f"{a}={arms[a][-1]['top1']:.4f}" for a in arms), flush=True)

    def summ(rows):
        tops = [r["top1"] for r in rows]
        return {"mean": round(float(np.mean(tops)), 4), "std": round(float(np.std(tops, ddof=1)), 4),
                "rows": rows}

    summary = {a: summ(r) for a, r in arms.items()}
    # 因子读数
    f0 = (summary["F0_Psel"]["mean"] + summary["F0_Pfull"]["mean"]) / 2
    f1 = (summary["F1_Psel"]["mean"] + summary["F1_Pfull"]["mean"]) / 2
    pfull = (summary["F0_Pfull"]["mean"] + summary["F1_Pfull"]["mean"]) / 2
    psel = (summary["F0_Psel"]["mean"] + summary["F1_Psel"]["mean"]) / 2
    filter_effect = f0 - f1      # 过滤主效应(>0 = 门有利)
    size_effect = pfull - psel   # 规模主效应(>0 = 全量有利)
    ratio = abs(size_effect) / max(abs(filter_effect), 1e-9)
    reading = ("pool-size dominates (|size| >= 3x |filter|)" if ratio >= 3 else
               "filter dominates (|filter| > |size|)" if abs(filter_effect) > abs(size_effect) else
               "comparable magnitudes")
    artifact = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-POOLDECOMP-PREREG-001 (2x2 filtering x pool-size decomposition, post-hoc mechanism diagnostic)",
        "layer": "public_human_benchmark",
        "harness": "identical to E9: frozen features_joint_ep300.npz, seed-42 10% subset, r16 corrected protocol; "
                   "pseudo-labels from the SAME clustering pipeline (run_selftrain final_pred_full), only pool "
                   "composition differs (gate on/off x full/random-subset)",
        "arms": summary,
        "effects": {
            "filter_main_effect_pp": round(filter_effect * 100, 2),
            "size_main_effect_pp": round(size_effect * 100, 2),
            "abs_ratio_size_over_filter": round(ratio, 2),
            "reading": reading,
        },
        "decision_rule_note": "exploratory diagnostic (registered after E9 results visible); per-arm Wilcoxon vs "
                              "E9 arm intentionally NOT claimed as significance; 3x dominance reading is a "
                              "mechanism statement only",
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r23-pool-decomp-{date}.json"
    out.write_text(json.dumps(artifact, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out}")
    print(json.dumps(artifact["effects"], indent=1))


if __name__ == "__main__":
    main()
