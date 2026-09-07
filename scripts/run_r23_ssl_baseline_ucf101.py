# -*- coding: utf-8 -*-
"""PSD-SSL-BASE-001 Amendment 1 — Mean-Teacher 外部基线第二域（UCF101, E9c harness）。

特征口径: 复用 run_p5b_generic_retention 的确定性 pretext（train_pretext_mlp 内部固定
torch.manual_seed(42)）重训后 dump——与 E9c 10-seed 运行所用特征逐位一致（一致性与
(a) 臂 14.04% 自检核对）。臂: (a) linear@10% / (mt) Mean-Teacher τ=0.95 5 轮 10 seeds。
(b) PSD 臂不重跑（引用 p5b-ucf101-retention-2026-09-07.json 15.40%±1.89）。
判据: 协议 Amendment 1，三方向如实。
产出: reports/r23-ssl-baseline-ucf101-<date>.json
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

from run_p5b_generic_retention import (  # noqa: E402
    dump_features,
    load_dataset,
    prep_clip,
    train_pretext_mlp,
)

PKL = REPO / "data" / "pyskl" / "ucf101_hrnet.pkl"
SEEDS = tuple(range(42, 52))
TAU = 0.95
ROUNDS = 5
OUT_DIR = REPO / "reports"


class ScaledLR:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)

    def fit(self, X_, y_):
        self.sc.fit(X_, y_)
        self.clf.fit(self.sc.transform(X_), y_)
        return self

    def predict_proba(self, X_):
        return self.clf.predict_proba(self.sc.transform(X_))

    def predict(self, X_):
        return self.clf.predict(self.sc.transform(X_))


def soft_head_train(feat, y_soft, sample_weight, seed, n_class):
    import torch
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    rows = load_dataset(PKL, 10000)
    print("[prep] resampling clips...", flush=True)
    X = np.stack([prep_clip(r["kp"]) for r in rows])
    labels = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    tr, vm = splits == "train", splits == "val"
    n_cls = len(set(labels.tolist()))
    print(f"[data] X {X.shape} train={tr.sum()} val={vm.sum()} classes={n_cls}", flush=True)

    print("[pretext] joint-MLP 80ep (deterministic seed-42, matches E9c features)...", flush=True)
    model = train_pretext_mlp(X[tr], labels[tr], n_cls, epochs=80)
    feats = dump_features(model, X)
    print(f"[feat] {feats.shape}", flush=True)

    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    tr_labels = labels[tr_idx]
    mask = np.zeros(tr.sum(), dtype=bool)
    for c in sorted(set(tr_labels.tolist())):
        ci = np.where(tr_labels == c)[0]
        k = max(1, int(round(len(ci) * 0.10)))
        mask[rng.choice(ci, size=k, replace=False)] = True

    # (a) 对照臂——一致性自检: 应=14.04%（E9c 09-07 工件 linear_10pct）
    clf_a = ScaledLR().fit(feats[tr_idx[mask]], labels[tr_idx[mask]])
    acc_a = float(np.mean(clf_a.predict(feats[vm]) == labels[vm]))
    print(f"[a] linear@10%: {acc_a:.4f} (expect 0.1404)", flush=True)
    consistency_ok = abs(acc_a - 0.1404) < 0.005
    print(f"[check] consistency_ok={consistency_ok}", flush=True)

    rows_mt = []
    for seed in SEEDS:
        import torch
        torch.manual_seed(seed)
        clf_t = ScaledLR().fit(feats[tr_idx[mask]], labels[tr_idx[mask]])
        prob_t = clf_t.predict_proba(feats)
        classes_t = clf_t.clf.classes_.astype(int)
        pool_abs = np.where(~anchor_abs(tr, mask, tr_idx))[0] if False else None
        # 池=全体 train 中非 anchor 者（绝对索引）
        anchor_abs_mask = np.zeros(len(labels), dtype=bool)
        anchor_abs_mask[tr_idx[mask]] = True
        pool = np.where(tr & ~anchor_abs_mask)[0]
        pool_conf = prob_t[pool].max(1)
        pool_soft = prob_t[pool]

        y_soft_full = np.zeros((len(labels), n_cls), dtype=np.float64)
        y_soft_full[tr_idx[mask], labels[tr_idx[mask]]] = 1.0
        y_soft_full[pool] = pool_soft
        w_full = np.zeros(len(labels))
        w_full[tr_idx[mask]] = 1.0
        w_full[pool] = (pool_conf >= TAU).astype(np.float64)
        n_conf = int(w_full[pool].sum())

        for rnd in range(ROUNDS):
            m = soft_head_train(feats, y_soft_full, w_full, seed + rnd, n_cls)
            with torch.no_grad():
                xt = torch.from_numpy(feats[pool]).float().to(next(m.parameters()).device)
                new_prob = torch.softmax(m(xt), dim=1).cpu().numpy()
            y_soft_full[pool] = new_prob  # 模型输出即 0..n-1 排, 无需 remap
        pred = predict(m, feats[vm])
        acc = float(np.mean(pred == labels[vm]))
        rows_mt.append({"seed": seed, "top1": round(acc, 4), "n_pool_confident": n_conf})
        print(f"[mt] seed{seed}: top1={acc:.4f} confident_pool={n_conf}", flush=True)

    accs = [r["top1"] for r in rows_mt]
    mt_mean = float(np.mean(accs))
    b_ref = json.load(open(OUT_DIR / "p5b-ucf101-retention-2026-09-07.json", encoding="utf-8"))
    b_map = {42 + i: v for i, v in enumerate(b_ref["b_arms"])}
    pairs = [(b_map[r["seed"]], r["top1"]) for r in rows_mt]
    wins_b = sum(1 for bb, mm in pairs if bb > mm)
    from scipy import stats
    try:
        _, pw = stats.wilcoxon([bb - mm for bb, mm in pairs])
    except Exception:
        pw = None
    print(f"[verdict] b-mean={np.mean([p[0] for p in pairs]):.4f} mt-mean={mt_mean:.4f} "
          f"b wins {wins_b}/10 wilcoxon p={pw}", flush=True)

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-SSL-BASE-001 Amendment 1 (Mean-Teacher second domain, UCF101 E9c harness)",
        "consistency_check": {"a_linear": round(acc_a, 4), "expect": 0.1404, "ok": bool(consistency_ok)},
        "arms": {
            "a_linear_10pct": round(acc_a, 4),
            "b_psd_ref": {"mean": b_ref["retention"] * 0 + float(np.mean(b_ref["b_arms"])),
                          "std": round(float(np.std(b_ref["b_arms"], ddof=1)), 4)},
            "mt_mean": round(mt_mean, 4),
            "mt_std": round(float(np.std(accs, ddof=1)), 4),
            "mt_rows": rows_mt,
        },
        "verdict": {"b_wins": f"{wins_b}/10",
                    "wilcoxon_p": (round(pw, 4) if pw is not None else None)},
        "config_echo": {"tau": TAU, "rounds": ROUNDS, "seeds": list(SEEDS),
                        "n_used": int(len(labels)), "classes": n_cls},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r23-ssl-baseline-ucf101-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}", flush=True)


def anchor_abs(tr, mask, tr_idx):  # unused helper kept out of the loop
    return np.zeros(len(tr), dtype=bool)


if __name__ == "__main__":
    main()
