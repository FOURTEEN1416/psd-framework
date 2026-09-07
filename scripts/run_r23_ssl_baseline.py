# -*- coding: utf-8 -*-
"""PSD-SSL-BASE-001 — 外部 SSL 基线（Mean-Teacher 式池一致性蒸馏）对照 E9 NTU60 10% 预算。

协议 docs/paper/ssl-baseline-preregistration.md v1.0（跑前冻结）：
  (a) linear@10%   —— E9 对照臂（66.05%）
  (b) PSD pipeline —— E9 已有结果（67.53%±0.24, r16-ntu-pseudo-10seed JSON, 不重跑）
  (mt) Mean-Teacher —— 教师=种子标签线性头；学生=种子标签+池软伪标签（τ=0.95 置信掩码），
       池上 5 轮教师-学生互馈；特征=E9 冻结 pretext 256d；10 seeds 42-51。
判据（协议 §4）: 三方向均如实入文，方向不筛选。
产出: reports/r23-ssl-baseline-<date>.json
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

from run_p14_ntu_lowres import (  # noqa: E402
    CLASS_NAMES,
    NPZ,
    stratified_10pct,
    train_linear_head_scaled,
)
from run_p07_endtoend_ak import evaluate  # noqa: E402

SEEDS = tuple(range(42, 52))
TAU = 0.95
ROUNDS = 5
OUT_DIR = REPO / "reports"


def soft_head_train(feat, y_soft, sample_weight, seed):
    """线性头拟合软目标（跨熵 over 类别分布, 温度 1）——用 sklearn LogisticRegression 的
    替代: 直接最小化软目标交叉熵的小型 numpy 实现? 太慢。此处用 torch 单层线性 +
    KL(soft || softmax(logits)) 拟合, 100ep GPU。"""
    import torch
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_class = y_soft.shape[1]
    model = torch.nn.Linear(feat.shape[1], n_class).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xt = torch.from_numpy(feat).float().to(device)
    yt = torch.from_numpy(y_soft).float().to(device)
    wt = torch.from_numpy(sample_weight).float().to(device)
    for _ in range(100):
        model.train()
        opt.zero_grad()
        logp = torch.log_softmax(model(xt), dim=1)
        loss = -(yt * logp).sum(1)
        loss = (loss * wt).sum() / wt.sum().clamp_min(1.0)
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict(model, feat):
    import torch
    with torch.no_grad():
        xt = torch.from_numpy(feat).float().to(next(model.parameters()).device)
        return model(xt).argmax(1).cpu().numpy()


def main():
    t0 = time.time()
    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]
    anchor = stratified_10pct(tr_lab, 42)
    print(f"[data] train {tr_feat.shape} val {va_feat.shape} anchor {int(anchor.sum())}", flush=True)

    # (a) 对照臂: 种子标签线性头（与 E9 完全一致, 确定性）
    clf_a = train_linear_head_scaled(tr_feat, tr_lab, anchor, 60)
    ev_a = evaluate(clf_a, va_feat, va_lab, np.ones(len(va_lab), bool), CLASS_NAMES)
    print(f"[a] linear@10%: top1={ev_a['top1']}", flush=True)
    a_acc = float(ev_a["top1"])

    rows_mt = []
    for seed in SEEDS:
        import torch
        torch.manual_seed(seed)
        # 教师初始 = 种子标签线性头（软概率输出）
        clf_t = train_linear_head_scaled(tr_feat, tr_lab, anchor, 60)
        prob_t = clf_t.clf.predict_proba(clf_t.sc.transform(tr_feat))  # (N, C) 训练池上的软伪标签
        classes_t = clf_t.clf.classes_.astype(int)
        pool = np.where(~anchor)[0]
        pool_conf = prob_t[pool].max(1)
        pool_soft = prob_t[pool]  # (P, C_sorted)

        # 5 轮教师-学生互馈: 学生 = 种子硬标签 + 池软标签(τ 掩码)；教师 = 学生重置后的概率输出
        pool_mask = (pool_conf >= TAU).astype(np.float64)  # (P,)
        w_pool = pool_mask  # 样本权重=τ 掩码
        y_soft_pool = pool_soft  # (P, C)
        # 构造全训练集软目标: anchor=one-hot 硬标签; pool=当前软伪标签
        y_soft_full = np.zeros((len(tr_lab), 60), dtype=np.float64)
        y_soft_full[anchor, tr_lab[anchor]] = 1.0
        y_soft_full[pool] = y_soft_pool
        w_full = np.zeros(len(tr_lab))
        w_full[anchor] = 1.0
        w_full[pool] = w_pool

        for rnd in range(ROUNDS):
            model = soft_head_train(tr_feat, y_soft_full, w_full, seed + rnd)
            # 学生在池上的新概率 → 下一轮教师软标签
            with torch.no_grad():
                xt = torch.from_numpy(tr_feat[pool]).float().to(next(model.parameters()).device)
                new_prob = torch.softmax(model(xt), dim=1).cpu().numpy()
            # 列对齐: model 输出按 0..59 索引; pool_soft 列按 classes_t
            remap = np.zeros_like(new_prob)
            remap[:, classes_t] = new_prob[:, classes_t]
            y_soft_full[pool] = remap
        pred = predict(model, va_feat)
        acc = float(np.mean(pred == va_lab))
        n_conf = int(pool_mask.sum())
        rows_mt.append({"seed": seed, "top1": round(acc, 4), "n_pool_confident": n_conf})
        print(f"[mt] seed{seed}: top1={acc:.4f} confident_pool={n_conf}", flush=True)

    accs = [r["top1"] for r in rows_mt]
    mt_mean = float(np.mean(accs))
    print(f"[mt] mean={mt_mean:.4f} std={np.std(accs, ddof=1):.4f}", flush=True)

    # 判据（协议 §4）: 与 E9 b 臂逐 seed 配对（同 seed 同子集）
    e9 = json.load(open(OUT_DIR / "r16-ntu-pseudo-10seed-2026-09-07.json", encoding="utf-8"))
    b_seeds = {r["seed"]: r["top1"] for r in e9["arms"]["b_selftrain_10pct"]}
    pairs = [(b_seeds[r["seed"]], r["top1"]) for r in rows_mt]
    wins_b = sum(1 for bb, mm in pairs if bb > mm)
    from scipy import stats
    try:
        w, pw = stats.wilcoxon([bb - mm for bb, mm in pairs])
    except Exception:
        w, pw = None, None
    print(f"[verdict] b-mean={np.mean([p[0] for p in pairs]):.4f} mt-mean={mt_mean:.4f} "
          f"b wins {wins_b}/10 wilcoxon p={pw}", flush=True)

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-SSL-BASE-001 (external SSL baseline, Mean-Teacher-style pool distillation, frozen v1.0)",
        "arms": {
            "a_linear_10pct": a_acc,
            "b_psd_pipeline_ref": e9["endpoints"],
            "mt_mean": round(mt_mean, 4),
            "mt_std": round(float(np.std(accs, ddof=1)), 4),
            "mt_rows": rows_mt,
        },
        "verdict": {
            "b_wins": f"{wins_b}/10",
            "wilcoxon_p": (round(pw, 4) if pw is not None else None),
            "parity_band_1pp": bool(abs(np.mean([p[0] for p in pairs]) - mt_mean) < 0.01),
        },
        "config_echo": {"tau": TAU, "rounds": ROUNDS, "seeds": list(SEEDS),
                        "n_pool": int((~anchor).sum())},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r23-ssl-baseline-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
