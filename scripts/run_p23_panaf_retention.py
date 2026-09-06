# -*- coding: utf-8 -*-
"""P7 PanAf500 保留率三臂 — 预注册协议 PSD-PANAF-PREREG-001 §2/§3 终点执行。

与 E9b/E9c（run_p5b_generic_retention.py）完全同构：joint-MLP pretext(80ep,
train split) → 冻结 256d 特征 → 三臂 (c)全监督线性参照 / (a)10% 线性 /
(b)10%+PSD 语义管线（修正协议 precision_stop=False）→ retention=top1(b)/top1(c)
@val。数据源 panaf500_T30.pkl（psd full12 格式，官方 split 400/25/75，9 类长尾）。

判据（预注册冻结）: ≥90 CONFIRMS / 85-90 PARTIAL / <85 FAILS——双向如实上报。

用法: .venv/Scripts/python.exe scripts/run_p23_panaf_retention.py
产出: reports/p23-panaf-retention-<date>.json
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT_DIR = REPO / "reports"
PKL = REPO / "runs" / "p7_asbar" / "panaf500_T30.pkl"
SEEDS = (42, 43, 44)
BUDGET_FRAC = 0.10
KW = dict(calib_method="softmax_temperature", calib_target=0.10,
          tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
          tau_select={"rule": "quantile", "target_coverage": 0.35},
          alpha=1.0, standing_mode="none", subcluster_k=2, subcluster_min_share=0.05,
          max_iters=6, converge_change_rate=0.01, precision_drop_patience=2,
          precision_stop=False, gate_mode="standing")


def prep_clip(kp: np.ndarray, T: int = 30) -> np.ndarray:
    """(T0,24,3) 已重采样；此处仅 center + dtype 对齐（与 p5b 同构）。"""
    kp = np.asarray(kp, dtype=np.float32)
    return kp - kp.mean(axis=(0, 1), keepdims=True)


def load_rows():
    rows = pickle.load(open(PKL, "rb"))
    X = np.stack([prep_clip(r["keypoints"]) for r in rows])
    labels = np.array([int(r["label"]) for r in rows])
    splits = np.array([r["split"] for r in rows])
    classes = sorted({str(r["psd_class"]) for r in rows})
    print(f"[panaf] {len(rows)} clips train={int((splits=='train').sum())} "
          f"val={int((splits=='val').sum())} test={int((splits=='test').sum())} classes={classes}")
    return X, labels, splits, classes


class JointMLP(torch.nn.Module):
    def __init__(self, in_dim, n_classes, hidden=512, feat_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, feat_dim), torch.nn.ReLU(),
        )
        self.head = torch.nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_feat=False):
        f = self.net(x)
        if return_feat:
            return self.head(f), f
        return self.head(f)


def train_pretext(X, y, n_classes, epochs=80, batch=64, device="cuda", lr=1e-3):
    torch.manual_seed(42)
    in_dim = int(np.prod(X.shape[1:]))
    model = JointMLP(in_dim, n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    xt = torch.from_numpy(X.reshape(len(X), -1)).to(device)
    yt = torch.from_numpy(y).long().to(device)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        tot, nb = 0.0, 0
        for s in range(0, len(xt), batch):
            idx = perm[s:s + batch]
            loss = ce(model(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 20 == 0:
            print(f"  [mlp] ep{ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    model.eval()
    return model


def dump_features(model, X):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 256):
            x = torch.from_numpy(X[s:s + 256].reshape(len(X[s:s + 256]), -1)).to(next(model.parameters()).device)
            _, f = model(x, return_feat=True)
            out.append(f.cpu().numpy())
    return np.vstack(out)


class ScaledLR:
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler(); self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)

    def fit(self, X_, y_):
        self.sc.fit(X_, y_); self.clf.fit(self.sc.transform(X_), y_); return self

    def predict(self, X_):
        return self.clf.predict(self.sc.transform(X_))


def main():
    import torch
    from psd.training.tcl_selftrain import run_selftrain

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, labels, splits, classes = load_rows()
    tr, vm = splits == "train", splits == "val"
    n_cls = len(set(labels.tolist()))

    print("[pretext] joint-MLP 80ep on train split...")
    model = train_pretext(X[tr], labels[tr], n_cls, device=device)
    feats = dump_features(model, X)

    labels_str = np.array([str(l) for l in labels])

    # (c) 全预算参照
    ref = float(np.mean(ScaledLR().fit(feats[tr], labels_str[tr]).predict(feats[vm]) == labels_str[vm]))
    print(f"[c] full ref: {ref:.4f}")

    # (a) 10% 线性（seed42 固定子集）
    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    mask_a = np.zeros(len(tr_idx), dtype=bool)
    for c in sorted(set(labels[tr_idx].tolist())):
        ci = np.where(labels[tr_idx] == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    acc_a = float(np.mean(ScaledLR().fit(feats[tr_idx[mask_a]], labels_str[tr_idx[mask_a]]).predict(feats[vm]) == labels_str[vm]))
    print(f"[a] 10% linear: {acc_a:.4f}")

    # (b) 10% + PSD 语义管线（修正协议）
    accs_b, pools = [], []
    for seed in SEEDS:
        anchor = np.zeros(len(labels_str), dtype=bool)
        rng2 = np.random.default_rng(seed)
        for c in classes:
            ci = np.where((labels_str == c) & tr)[0]
            k = max(1, int(round(len(ci) * BUDGET_FRAC)))
            anchor[rng2.choice(ci, size=k, replace=False)] = True
        universe = tr & ~anchor
        r = run_selftrain(feats, labels_str, anchor, run_seed=seed, class_names=classes,
                          head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                    "weight_decay": 1e-4, "batch_size": 128, "device": "cpu"},
                          pool_universe_mask=universe, **KW)
        pool_idx = r["final_pool_idx"]
        tm2 = anchor.copy(); tm2[pool_idx] = True
        y2 = np.array([None] * len(labels_str), dtype=object)
        y2[anchor] = labels_str[anchor]
        y2[pool_idx] = r["final_pred_full"][pool_idx]
        # 评分在字符串域（P5b 教训: str/int 错配恒 False）
        pred_b = ScaledLR().fit(feats[tm2], y2[tm2]).predict(feats[vm])
        acc_b = float(np.mean(pred_b == labels_str[vm]))
        accs_b.append(acc_b); pools.append(len(pool_idx))
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(pool_idx)} stop={r['stop_reason']}")

    ret = float(np.mean(accs_b)) / ref
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-PANAF-PREREG-001",
        "full_ref": round(ref, 4), "linear_10pct": round(acc_a, 4),
        "b_arms": [round(a, 4) for a in accs_b],
        "pools": pools, "linear_retention": round(acc_a / ref, 4),
        "retention": round(ret, 4), "verdict": verdict,
        "config_echo": {"n_clips": int(len(labels)), "classes": n_cls, "seeds": list(SEEDS),
                        "budget": BUDGET_FRAC, "longtail": "minority classes contribute 1 seed clip each"},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = OUT_DIR / f"p23-panaf-retention-{datetime.now():%Y-%m-%d}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
