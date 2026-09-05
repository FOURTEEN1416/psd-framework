"""R16 协议修正重跑 — 端到端臂最终头改用伪标签（诚实低资源协议）。

背景（R16 对抗审稿 CRITICAL F1/F2）: p07/p08/p10/p12 的"端到端"臂把
run_selftrain 选出的池片段用**真实标签**重训线性头——最终分类器实际消费了
60-99%（v1）/ 96%（NTU）训练标签，"13% 标注达 94%"等主张与实现不符。
本脚本以修正协议重跑全部端到端臂:
  最终头 = 种子(真标签) ∪ 池(最终轮伪标签 final_pred_full) 上的线性头;
  停止规则 = 预算/收敛（precision_stop=False，oracle 精度仅作诊断记录）;
  全监督参照臂不变（纯监督 LR，合法）。
产出为论文端到端数字的新真源; 旧 p07/p10/p12 JSON 保留为历史归档（附勘误）。

用法:
    .venv/Scripts/python.exe scripts/run_r16_endtoend_pseudo.py [--tier v1|v2|all]
产出:
    reports/r16-endtoend-pseudo-<date>.json
    reports/r16-holm-<date>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p07_endtoend_ak import (  # noqa: E402
    HEAD_CFG,
    KW,
    extract_features,
    load_dataset,
)
from run_p08_aimclr_arm import build_aimclr_views, extract_aimclr_features  # noqa: E402
from run_p10_seedexpansion import paired_stats  # noqa: E402
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

SEEDS10 = tuple(range(42, 52))
PKL_V2 = REPO / "runs" / "public_real_dataset" / "full12v2_T30.pkl"
OUT_DIR = REPO / "reports"

KW_R16 = dict(KW)
KW_R16["precision_stop"] = False  # R16: 无 oracle 停止控制


def load_v2():
    with open(PKL_V2, "rb") as f:
        data = pickle.load(f)
    kp = np.stack([np.asarray(d["keypoints"], dtype=np.float32) for d in data])
    labels = np.array([int(d["label"]) for d in data])
    splits = np.array([d["split"] for d in data])
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    return kp, labels, labels_str, splits, class_names


def evaluate_str(clf, emb, labels_str, val_mask, class_names):
    """字符串标签口径的 val 评估（与 p07.evaluate 同数学，仅标签类型不同）。"""
    X = emb[val_mask]
    y = labels_str[val_mask]
    pred = clf.predict(X)
    top1 = float(np.mean(pred == y))
    f1s = []
    per_cls = {}
    for c in class_names:
        tp = np.sum((y == c) & (pred == c))
        fp = np.sum((y != c) & (pred == c))
        fn = np.sum((y == c) & (pred != c))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        sup = int(np.sum(y == c))
        if sup > 0:
            per_cls[c] = {"support": sup, "acc": round(float(np.mean(pred[y == c] == c)), 4)}
    return {"top1": round(top1, 4), "macro_f1": round(float(np.mean(f1s)), 4), "per_class": per_cls}


def run_one_pseudo(emb, labels_str, splits, class_names, arm, spc, seed):
    """修正协议端到端臂: 最终头 = 种子真标签 ∪ 池伪标签（final_pred_full）。"""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    if spc < 0:  # 全监督参照: 纯监督 LR（合法，不变）
        train_mask = (splits == "train")
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(emb[train_mask], labels_str[train_mask])
        ev = evaluate_str(clf, emb, labels_str, splits == "val", class_names)
        return {"arm": arm, "spc": -1, "seed": seed, "budget": "full",
                "n_seeds": int(train_mask.sum()), "n_pool": 0, "n_pseudo": 0,
                "tau_star": None, "rounds": 0, "stop": "full-supervision(no pseudo-label)",
                "pool_oracle_precision_diag": None, **ev}
    anchor_mask = _pick_seeds_str(labels_str, splits, spc, rng)
    universe = (splits == "train") & ~anchor_mask
    r = run_selftrain(emb, labels_str, anchor_mask, run_seed=seed, class_names=class_names,
                      head_cfg=HEAD_CFG, pool_universe_mask=universe, **KW_R16)
    pool_idx = r["final_pool_idx"]
    # ---- R16 核心修正: 池片段用最终轮伪标签，不用真标签 ----
    train_mask = anchor_mask.copy()
    train_mask[pool_idx] = True
    y_train = np.array([None] * len(labels_str), dtype=object)
    y_train[anchor_mask] = labels_str[anchor_mask]
    y_train[pool_idx] = r["final_pred_full"][pool_idx]
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(emb[train_mask], y_train[train_mask])
    ev = evaluate_str(clf, emb, labels_str, splits == "val", class_names)
    return {"arm": arm, "spc": spc, "seed": seed,
            "n_seeds": int(anchor_mask.sum()), "n_pool": int(len(pool_idx)),
            "n_pseudo": int(len(pool_idx)),
            "tau_star": round(float(r["tau_operating"]), 4),
            "rounds": len(r["rounds"]), "stop": r["stop_reason"],
            "pool_oracle_precision_diag": r["rounds"][-1].get("precision"), **ev}


def _pick_seeds_str(labels_str, splits, spc, rng):
    """train split 内每类 spc 个 clip 作种子（字符串标签版）。"""
    n = len(labels_str)
    mask = np.zeros(n, dtype=bool)
    train_idx = np.where(splits == "train")[0]
    by_cls = {}
    for i in train_idx:
        by_cls.setdefault(str(labels_str[i]), []).append(i)
    for c, idxs in by_cls.items():
        chosen = rng.choice(idxs, size=min(spc, len(idxs)), replace=False)
        mask[chosen] = True
    return mask


def holm(pairs: dict) -> dict:
    """Holm-Bonferroni: pairs = {name: raw_p} -> {name: {raw, adjusted}}。"""
    items = sorted(pairs.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running_max = {}, 0.0
    for k, (name, p) in enumerate(items):
        adj = min(1.0, max(running_max, (m - k) * p))
        running_max = adj
        out[name] = {"raw_p": round(p, 4), "holm_adjusted": round(adj, 4)}
    return out


def run_tier(name, kp, labels_str, splits, class_names, want_aimclr=True):
    print(f"[{name}] feat warm...")
    f_warm = extract_features(kp, "warm")
    feats = {"warm": f_warm}
    if want_aimclr:
        print(f"[{name}] feat aimclr...")
        feats["aimclr"] = extract_aimclr_features(build_aimclr_views(kp))
    results = {}
    for arm, emb in feats.items():
        for spc in (2, 4):
            rows = []
            for seed in SEEDS10:
                r = run_one_pseudo(emb, labels_str, splits, class_names, arm, spc, seed)
                rows.append(r)
                print(f"[{name} {arm} spc{spc}] seed{seed}: top1={r['top1']} mf1={r['macro_f1']} pool={r['n_pool']} stop={r['stop']}")
            results[f"{arm}_spc{spc}"] = rows
    # scratch: 每 seed 独立随机骨干
    rows = []
    for seed in SEEDS10:
        torch.manual_seed(seed)
        f_scr = extract_features(kp, "scratch")
        r = run_one_pseudo(f_scr, labels_str, splits, class_names, "scratch", 2, seed)
        rows.append(r)
        print(f"[{name} scratch spc2] seed{seed}: top1={r['top1']} mf1={r['macro_f1']} pool={r['n_pool']} stop={r['stop']}")
    results["scratch_spc2"] = rows
    # 全监督参照（确定性）
    results["warm_full"] = [run_one_pseudo(f_warm, labels_str, splits, class_names, "warm", -1, 42)]
    return results


def summarize(results):
    out = {}
    for k, rows in results.items():
        t = [r["top1"] for r in rows if "top1" in r]
        m = [r["macro_f1"] for r in rows if "macro_f1" in r]
        p = [r["pool_oracle_precision_diag"] for r in rows if r.get("pool_oracle_precision_diag") is not None]
        out[k] = {"n": len(t),
                  "top1_mean": round(float(np.mean(t)), 4),
                  "top1_std": round(float(np.std(t, ddof=1)), 4) if len(t) > 1 else 0.0,
                  "macro_f1_mean": round(float(np.mean(m)), 4),
                  "macro_f1_std": round(float(np.std(m, ddof=1)), 4) if len(m) > 1 else 0.0,
                  "pool_oracle_precision_mean_diag": round(float(np.mean(p)), 4) if p else None,
                  "n_pool_mean": round(float(np.mean([r["n_pool"] for r in rows])), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="all", choices=["v1", "v2", "all"])
    args = ap.parse_args()
    t0 = time.time()
    date = datetime.now().strftime("%Y-%m-%d")

    all_results, all_summ = {}, {}
    if args.tier in ("v1", "all"):
        data, kp, labels, splits = load_dataset()
        labels_str = np.array([str(d["psd_class"]) for d in data])
        class_names = sorted({str(d["psd_class"]) for d in data})
        res = run_tier("v1", kp, labels_str, splits, class_names)
        all_results["v1"] = res
        all_summ["v1"] = summarize(res)

    if args.tier in ("v2", "all"):
        kp2, lab2, ls2, sp2, cn2 = load_v2()
        res = run_tier("v2", kp2, ls2, sp2, cn2)
        all_results["v2"] = res
        all_summ["v2"] = summarize(res)

    # 先落盘原始 runs（防统计段异常丢结果）
    date = datetime.now().strftime("%Y-%m-%d")
    raw = OUT_DIR / f"r16-endtoend-pseudo-{date}.raw.json"
    raw.write_text(json.dumps({"summary": all_summ, "runs": all_results},
                              ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # R16 诊断臂: 头路锚点侧再校准（GT 无关）——检验门控失效是否为 κ 尺度错配所致
    diag = {}
    if args.tier in ("v1", "all"):
        from psd.training.tcl_selftrain import run_selftrain as _rs
        from run_p07_endtoend_ak import HEAD_CFG as _HC
        data, kp, labels, splits = load_dataset()
        labels_str = np.array([str(d["psd_class"]) for d in data])
        class_names = sorted({str(d["psd_class"]) for d in data})
        f_warm = extract_features(kp, "warm")
        rows = []
        for seed in (42, 43, 44):
            rng = np.random.default_rng(seed)
            anchor = _pick_seeds_str(labels_str, splits, 2, rng)
            universe = (splits == "train") & ~anchor
            kw = dict(KW_R16)
            kw["head_calib"] = True
            r = _rs(f_warm, labels_str, anchor, run_seed=seed, class_names=class_names,
                    head_cfg=_HC, pool_universe_mask=universe, **kw)
            pool_idx = r["final_pool_idx"]
            tm = anchor.copy(); tm[pool_idx] = True
            yt = np.array([None] * len(labels_str), dtype=object)
            yt[anchor] = labels_str[anchor]
            yt[pool_idx] = r["final_pred_full"][pool_idx]
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(f_warm[tm], yt[tm])
            ev = evaluate_str(clf, f_warm, labels_str, splits == "val", class_names)
            rows.append({"seed": seed, "n_pool": int(len(pool_idx)), "rounds": len(r["rounds"]),
                         "stop": r["stop_reason"],
                         "pool_oracle_precision": r["rounds"][-1].get("precision"), **ev})
            print(f"[diag head_calib] seed{seed}: top1={ev['top1']} pool_prec={rows[-1]['pool_oracle_precision']} pool={len(pool_idx)}")
        diag["v1_warm_spc2_head_calib"] = rows

    # 配对检验 + Holm
    tests, raw_ps = {}, {}
    for tier in all_results:
        R = all_results[tier]
        def top1(k): return np.array([r["top1"] for r in R[k]])
        def mf1(k): return np.array([r["macro_f1"] for r in R[k]])
        for spc in (2, 4):
            for metric, fn in (("top1", top1), ("macro_f1", mf1)):
                key = f"{tier}_warm_vs_aimclr_spc{spc}_{metric}"
                tests[key] = paired_stats(fn(f"warm_spc{spc}")[:10], fn(f"aimclr_spc{spc}")[:10])
                if "wilcoxon_p" in tests[key]:
                    raw_ps[key] = tests[key]["wilcoxon_p"]
        for metric, fn in (("top1", top1), ("macro_f1", mf1)):
            key = f"{tier}_warm_vs_scratch_spc2_{metric}"
            tests[key] = paired_stats(fn("warm_spc2")[:10], fn("scratch_spc2")[:10])
            if "wilcoxon_p" in tests[key]:
                raw_ps[key] = tests[key]["wilcoxon_p"]
    holm_out = holm(raw_ps)
    for k in tests:
        if k in holm_out:
            tests[k]["holm_adjusted"] = holm_out[k]["holm_adjusted"]

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "R16 corrected end-to-end: final head = seeds(true) + pool(PSEUDO from final round); "
                    "precision-drop stopping disabled (no oracle); full-supervision references unchanged",
        "correction_note": "Supersedes p07/p08/p10/p12 end-to-end arms (those trained the reported head on "
                           "TRUE labels of pool clips — a protocol error found in adversarial review R16).",
        "layer": "public_real",
        "summary": all_summ, "paired_tests": tests, "diagnostics": diag,
        "holm": {"families": "per-tier warm-vs-aimclr(spc2/spc4 × top1/mf1) + warm-vs-scratch(spc2 × top1/mf1)",
                 "adjusted": holm_out},
        "runs": {t: {k: v for k, v in R.items()} for t, R in all_results.items()},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = OUT_DIR / f"r16-endtoend-pseudo-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps(all_summ, indent=1))
    print(json.dumps({k: v for k, v in holm_out.items()}, indent=1))


if __name__ == "__main__":
    main()
