"""P1.4b NTU 低资源保留率协议 — PSD-NTU-PREREG-001 §3 终点执行。

三臂（冻结 pretext 256d 特征，官方 Feeder 口径，p14a 导出）:
  (c) 100% train 线性头 = 本探针参照天花板
  (a) 10% 分层子集线性头 = 纯预算缩减
  (b) 10% 子集 + PSD 语义管线（run_selftrain 锚点聚类伪标签，与 E7 同函数同参）
主终点: retention = top1(b)/top1(c)，判据冻结 ≥90% / 85-90% / <85%。
对照: TCL 发表保留率 82.7/88.6 = 93.3%（细调管线口径，仅比保留行为不比绝对值）。

披露: ①HEAD_CFG device cpu→cuda（36k 池算力适配，其余超参逐字同 E7）；
②10% 子集 seed42 固定（协议 §2），selftrain 随机性 3 seeds；③val 永不入池。

用法:
    .venv/Scripts/python.exe scripts/run_p14_ntu_lowres.py
产出:
    reports/p14-ntu-lowres-<date>.json / .md
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


class ScaledLR:
    """协议适配 #2: StandardScaler+LR(max_iter=1000)。NTU pretext 特征尺度方差大，
    原 E7 裸 LR(lbfgs,2000) 在 40k×60 上收敛爬行(>30min 未完成)；缩放为标准做法，
    与 device 适配一同披露。"""

    def __init__(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        self.sc = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)

    def fit(self, X, y):
        self.sc.fit(X, y)
        self.clf.fit(self.sc.transform(X), y)
        return self

    def predict(self, X):
        return self.clf.predict(self.sc.transform(X))


def train_linear_head_scaled(emb, labels, train_mask, n_cls):
    X = emb[train_mask]
    y = labels[train_mask]
    if len(np.unique(y)) < 2:
        return None
    return ScaledLR().fit(X, y)
from psd.training.tcl_selftrain import run_selftrain  # noqa: E402

NPZ = REPO / "runs" / "ntu_lowres" / "features_joint_ep300.npz"
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
BUDGET_FRAC = 0.10
CLASS_NAMES = [str(i) for i in range(60)]
HEAD_CFG = {"hidden_dim": 64, "epochs": 150, "lr": 0.001, "weight_decay": 0.0001,
            "batch_size": 128, "device": "cuda"}  # 唯一适配: device（36k 池），其余同 E7
KW = dict(calib_method="softmax_temperature", calib_target=0.10,
          tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
          tau_select={"rule": "quantile", "target_coverage": 0.35},
          alpha=1.0, standing_mode="consensus", subcluster_k=2, subcluster_min_share=0.05,
          max_iters=6, converge_change_rate=0.01, precision_drop_patience=2)


def stratified_10pct(labels, seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(labels), dtype=bool)
    for c in range(60):
        ci = np.where(labels == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    return mask


def main():
    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    labels_str = np.array([str(int(v)) for v in tr_lab])
    print(f"[ntu] train {tr_feat.shape} val {va_feat.shape}")

    # (c) 全预算参照
    clf_full = train_linear_head_scaled(tr_feat, tr_lab, np.ones(len(tr_lab), bool), 60)
    ev_full = evaluate(clf_full, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[c] full linear head: top1={ev_full['top1']} macroF1={ev_full['macro_f1']}")

    # 10% 子集（seed42 固定，协议冻结）
    anchor = stratified_10pct(tr_lab, 42)
    print(f"[budget] 10% subset: {int(anchor.sum())} clips")

    # (a) 纯线性头 @10%
    clf_a = train_linear_head_scaled(tr_feat, tr_lab, anchor, 60)
    ev_a = evaluate(clf_a, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[a] linear@10%: top1={ev_a['top1']} macroF1={ev_a['macro_f1']}")

    # (b) PSD selftrain @10% × 3 seeds
    universe = ~anchor  # 池宇宙 = train 非种子；val 不在 emb 空间内，天然隔离
    rows_b = []
    for seed in SEEDS:
        r = run_selftrain(tr_feat, labels_str, anchor, run_seed=seed, class_names=CLASS_NAMES,
                          head_cfg=HEAD_CFG, pool_universe_mask=universe, **KW)
        train_mask = anchor.copy()
        train_mask[r["final_pool_idx"]] = True
        clf_b = train_linear_head_scaled(tr_feat, tr_lab, train_mask, 60)
        ev_b = evaluate(clf_b, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
        rows_b.append({"seed": seed, "n_pool": int(len(r["final_pool_idx"])),
                       "rounds": len(r["rounds"]), "stop": r["stop_reason"], **ev_b})
        print(f"[b] selftrain seed{seed}: top1={ev_b['top1']} macroF1={ev_b['macro_f1']} pool={rows_b[-1]['n_pool']}")

    tb = [r["top1"] for r in rows_b]
    retention = float(np.mean(tb)) / ev_full["top1"]
    if retention >= 0.90:
        verdict = "GENERALIZES (retention >= 90%)"
    elif retention >= 0.85:
        verdict = "PARTIAL (85-90%)"
    else:
        verdict = "ANIMAL_DOMAIN_SPECIFIC (<85%, reported as boundary)"

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-NTU-PREREG-001",
        "layer": "public_human_benchmark",
        "disclosures": [
            "HEAD_CFG device cpu->cuda for 36k pool (adaptation #1; other hyperparams identical to E7)",
            "final linear head uses StandardScaler + LR max_iter=1000 tol=1e-3 (adaptation #2; unscaled default-tol lbfgs crawled >30 min on 40k x 60)",
            "10% subset fixed at seed 42 per protocol; selftrain stochasticity over 3 seeds",
            "retention axis only: our frozen-probe protocol vs TCL fine-tuned pipeline — absolute numbers not comparable",
        ],
        "arms": {"c_full_linear": ev_full, "a_linear_10pct": ev_a, "b_selftrain_10pct": rows_b},
        "endpoints": {
            "retention_b_over_c": round(retention, 4),
            "verdict": verdict,
            "tcl_published_retention": 0.933,
            "linear_only_retention_a_over_c": round(ev_a["top1"] / ev_full["top1"], 4),
        },
        "config_echo": {"npz": str(NPZ), "weights": "runs/ntu_phaseB/joint_pretext/epoch300_model.pt",
                        "n_train": int(len(tr_lab)), "n_val": int(len(va_lab)),
                        "n_budget": int(anchor.sum()), "seeds": list(SEEDS)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p14-ntu-lowres-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out} ({result['wall_clock_sec']}s)")
    print(json.dumps(result["endpoints"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
