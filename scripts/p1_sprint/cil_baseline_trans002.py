# -*- coding: utf-8 -*-
"""P1 大修 — CIL 风格最小基线（TRANS-002 60→49 merge，冻结特征同 harness）.

回应评估确认的 "与 CIL 零基线对比、新颖性停在机制组合级" 弱点：在 TRANS-002
episode 上实现一个 CIL 代表方法（LwF 蒸馏 + 旧类 exemplar 回放，iCaRL/LwF 风格
最小适配），与论文各臂同冻结特征、同评估口径对比。

设计（诚实披露适配自由度）：
  - 特征：PRETEXT_CKPT（NTU60 冻结编码器）penultimate 256-d，train 40,091 /
    val 16,487，标签为 Y' 49 类空间（dump 函数复用 run_r23_trans002 同一实现）。
  - 旧头（Y 时代）：Linear(256→60)，10% 旧标注子集（seed 42）CE 训练——与 E1
    source head 同口径。
  - CIL 新任务（49 类）：新头 Linear(256→49) 由旧头按 merge 映射行平均初始化；
    训练损失 = CE(新标注) + λ·T²·KL(teacher=旧头概率按 merge 组聚合, student)，T=2、
    λ=1（标准 LwF 配方）；10% 预算下旧类 10% 样本即 exemplar 回放集（标签按 merge
    映射改写）。full 预算 = 全部 40,091 样本的新标签。
  - 种子 42-44（E 系列惯例）；两档预算 {10pct, full} 与 decoupled 臂口径对齐。
  - 评估：val 16,487 的 49 类 top-1。

对照锚点（论文既有，不重跑）：zero-retraining bound 68.04%/67.97%（E1）、
decoupled@10% 67.6%、rebuild@full 75.3%、coupled 79.8%(3 seeds)/79.7%(10 seeds)、
matched-solver 82.6%。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/cil_baseline_trans002.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/cil_baseline_trans002.py
产出:
    runs/p1_sprint/cil_baseline_trans002/summary.json（冒烟为 summary_smoke.json）
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO))

import run_r23_trans002 as r23  # noqa: E402  （复用同一 dump 与 merge 映射实现）
from run_p14_ntu_lowres import stratified_10pct  # noqa: E402

RUN_DIR = REPO / "runs" / "p1_sprint" / "cil_baseline_trans002"
SEEDS = (42, 43, 44)
EPOCHS = 400
LR = 1e-2
T = 2.0
LAMBDA = 1.0

ANCHORS = {
    "e1_zero_retraining_bound": 0.6804,
    "e1_argmax_vote_lower": 0.6797,
    "decoupled_rebuild_10pct": 0.676,
    "decoupled_rebuild_full": 0.753,
    "coupled_three_seed": 0.798,
    "coupled_ten_seed": 0.797,
    "matched_solver_ten_seed": 0.826,
}


def train_linear_head(X, y, n_class, seed, epochs=EPOCHS, lr=LR, device="cpu"):
    """冻结特征上的线性头 CE 训练（确定性）。"""
    import torch

    torch.manual_seed(seed)
    clf = torch.nn.Linear(X.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    xt = torch.from_numpy(X).float().to(device)
    yt = torch.from_numpy(y).long().to(device)
    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(clf(xt), yt)
        loss.backward()
        opt.step()
    clf.eval()
    return clf


def predict(clf, X, device="cpu", batch=8192):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xt = torch.from_numpy(X[i:i + batch]).float().to(device)
            out.append(clf(xt).cpu().numpy())
    return np.vstack(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REPO / ".pstate_tmp"
    tmp.mkdir(exist_ok=True)

    # 1) 冻结编码器特征（与所有 PSD 臂同一 encoder、同一 dump 实现）
    tr_npz = tmp / "cil_train_feat_pretext.npz"
    va_npz = tmp / "cil_val_feat_pretext.npz"
    if not tr_npz.exists():
        r23.dump_penultimate(r23.PRETEXT_CKPT, "train", "full", tr_npz)
    if not va_npz.exists():
        r23.dump_penultimate(r23.PRETEXT_CKPT, "val", "full", va_npz)
    ztr = np.load(tr_npz)
    zva = np.load(va_npz)
    Xtr, ytr49 = ztr["feat"], ztr["label"].astype(int)
    Xva, yva49 = zva["feat"], zva["label"].astype(int)

    # 2) 旧 60 类标签（与 train_position.npy 同序）
    with open(r23.DATA / "train_label.pkl", "rb") as f:
        _, tr_label60 = pickle.load(f)
    tr_label60 = np.array(tr_label60)
    from run_r22_ntu_transition import NTU60_CLASSES  # noqa: E402
    y2yp = r23.build_merge_map(list(NTU60_CLASSES))
    assert all(int(y2yp[int(l)]) == int(y) for l, y in zip(tr_label60, ytr49)), \
        "npz 标签与 60 类旧标签顺序不一致"

    # 3) 旧头（Y 时代，10% 旧标注，seed 42）——LwF teacher / E1 source head 同口径
    results = {"budgets": {}, "anchors": ANCHORS}
    for budget in ("10pct", "full"):
        per_seed = {}
        for seed in SEEDS:
            mask = stratified_10pct(tr_label60, seed) if budget == "10pct" \
                else np.ones(len(tr_label60), dtype=bool)
            Xs, y60s, y49s = Xtr[mask], tr_label60[mask], ytr49[mask]

            old_head = train_linear_head(Xs, y60s, 60, seed, device=device)
            # merge 组聚合的 teacher 概率（49 维）
            p_old = torch.softmax(torch.from_numpy(predict(old_head, Xs, device)), dim=1).numpy()
            teacher49 = np.zeros((len(Xs), 49), dtype=np.float32)
            for o, n in y2yp.items():
                teacher49[:, n] += p_old[:, o]
            teacher49 = teacher49 / np.clip(teacher49.sum(axis=1, keepdims=True), 1e-9, None)

            # 新头：旧头按 merge 行平均初始化
            import torch
            torch.manual_seed(seed)
            new_head = torch.nn.Linear(256, 49).to(device)
            with torch.no_grad():
                w = old_head.weight.detach().cpu().numpy()
                b = old_head.bias.detach().cpu().numpy()
                for n in range(49):
                    rows = [o for o, nn in y2yp.items() if nn == n]
                    new_head.weight.data[n] = torch.from_numpy(w[rows].mean(axis=0))
                    new_head.bias.data[n] = float(b[rows].mean())

            opt = torch.optim.AdamW(new_head.parameters(), lr=LR, weight_decay=1e-4)
            xt = torch.from_numpy(Xs).float().to(device)
            yt = torch.from_numpy(y49s).long().to(device)
            tt = torch.from_numpy(teacher49).float().to(device)
            for _ in range(EPOCHS):
                new_head.train()
                opt.zero_grad()
                logits = new_head(xt)
                ce = torch.nn.functional.cross_entropy(logits, yt)
                kl = torch.nn.functional.kl_div(
                    torch.log_softmax(logits / T, dim=1),
                    torch.softmax(tt / T, dim=1), reduction="batchmean") * (T * T)
                (ce + LAMBDA * kl).backward()
                opt.step()
            new_head.eval()

            pred_va = predict(new_head, Xva, device).argmax(axis=1)
            top1 = float(np.mean(pred_va == yva49))
            per_seed[seed] = round(top1, 4)
            print(f"[cil:{budget}] seed{seed}: top1={top1 * 100:.2f}%", flush=True)
        tops = [v for v in per_seed.values()]
        results["budgets"][budget] = {
            "per_seed": per_seed,
            "mean": round(float(np.mean(tops)), 4),
            "std": round(float(np.std(tops, ddof=1)), 4) if len(tops) > 1 else 0.0,
        }

    result = {
        "doneMarker": f"P1_SPRINT_CIL_BASELINE_TRANS002_{'SMOKE_OK' if args.smoke else 'DONE'}",
        "experiment": "p1_sprint_cil_baseline_trans002",
        "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "method": "CIL-flavored minimal baseline (LwF distillation + old-class replay) "
                  "on TRANS-002 (60->49 merge), frozen 256-d pretext features, same "
                  "harness/eval as all arms",
        "adaptation_disclosure": "CIL protocols assume class-incremental arrival; the merge "
                                 "episode required an adaptation (logit-aggregation "
                                 "distillation targets + merge-relabeled replay) whose design "
                                 "freedom is exactly why the paper positions CIL as "
                                 "not-directly-comparable - reported honestly as a "
                                 "CIL-flavored reference, not a published-method reproduction",
        "recipe": {"epochs": EPOCHS, "lr": LR, "optimizer": "AdamW", "T": T, "lambda": LAMBDA,
                   "seeds": list(SEEDS), "budgets": ["10pct", "full"],
                   "init": "old-head rows averaged per merge group",
                   "features": "PRETEXT_CKPT penultimate 256-d (frozen)"},
        "results": results,
        "anchors": ANCHORS,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = RUN_DIR / ("summary_smoke.json" if args.smoke else "summary.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {out}")
    print(json.dumps(results["budgets"], indent=1))


if __name__ == "__main__":
    main()
