# -*- coding: utf-8 -*-
"""P5-B NTU120 / P5-A UCF101 保留率 — 通用 HRNet 2D 骨架版（任意关节数）。

E9 协议镜像（三臂参照），但 pretext 改为 joint-level MLP（适配任意 V 关节）：
  - 输入 clip (T=30, V, 3)，flatten 全部关节坐标 → 3 层 MLP → 动作分类 100ep
  - 冻结 MLP penultimate 作为特征提取器
  - (c) 100% 线性参照 / (a) 10% 线性 / (b) 10%+PSD 语义管线（修正协议）
判据: retention ≥90% CONFIRMS / 85-90 PARTIAL / <85 FAILS
"""
from __future__ import annotations
import json, pickle, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
BUDGET_FRAC = 0.10
KW = dict(calib_method="softmax_temperature", calib_target=0.10,
          tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
          tau_select={"rule": "quantile", "target_coverage": 0.35},
          alpha=1.0, standing_mode="none", subcluster_k=2, subcluster_min_share=0.05,
          max_iters=6, converge_change_rate=0.01, precision_drop_patience=2,
          precision_stop=False, gate_mode="standing")


def prep_clip(kp_raw: np.ndarray, T: int = 30) -> np.ndarray:
    """kp_raw (T0,V,C) → (T,V,C) 重采样 + center。V 任意。"""
    kp = np.asarray(kp_raw, dtype=np.float32)
    T0, V = kp.shape[0], kp.shape[1]
    if T0 < T:
        idx = np.resize(np.arange(T0), T)
    else:
        idx = np.linspace(0, T0 - 1, T, dtype=int)
    kp = kp[idx]
    if kp.shape[-1] < 3:
        kp = np.concatenate([kp, np.ones((T, V, 1), np.float32)], axis=-1)
    # center by mean over (T,V)
    kp = kp - kp.mean(axis=(0, 1), keepdims=True)
    return kp.astype(np.float32)


def load_dataset(pkl_path: Path, max_n: int = 12000):
    print(f"[load] {pkl_path.name}...")
    d = pickle.load(open(pkl_path, "rb"))
    anns = d["annotations"]
    split = d["split"]
    train_keys = set()
    for k in ("xsub_train", "train1"):
        train_keys.update(split.get(k, []))
    val_keys = set()
    for k in ("xsub_val", "xsub_test", "test1"):
        val_keys.update(split.get(k, []))
    rows = []
    for a in anns[:max_n * 3]:
        if len(rows) >= max_n:
            break
        kp = np.asarray(a["keypoint"], dtype=np.float32)
        while kp.ndim > 3 and kp.shape[0] == 1:
            kp = kp[0]
        if kp.ndim == 4:
            kp = kp[:, 0]
        kps = np.asarray(a.get("keypoint_score", np.zeros(0)), dtype=np.float32)
        while kps.ndim > 2 and kps.shape[0] == 1:
            kps = kps[0]
        if kps.ndim == 2 and kps.shape == kp.shape[:2]:
            kps = kps[:, :, np.newaxis]
        else:
            kps = np.ones((*kp.shape[:2], 1), np.float32)
        kp3 = np.concatenate([kp, kps], axis=-1).astype(np.float32)
        key = str(a.get("frame_dir", len(rows)))
        sp = "train" if key in train_keys else ("val" if key in val_keys else None)
        if sp is None:
            continue
        rows.append({"kp": kp3, "label": int(a["label"]), "split": sp, "key": key})
    print(f"[load] {len(rows)} clips train={sum(1 for r in rows if r['split']=='train')} val={sum(1 for r in rows if r['split']=='val')}")
    return rows


class JointMLP(torch.nn.Module):
    """(B, T*V*3) → hidden → action logits。penultimate 256d。"""
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


def train_pretext_mlp(X, y, n_classes, epochs=80, batch=256, device="cuda", lr=1e-3):
    import torch
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
            logits = model(xt[idx])
            loss = ce(logits, yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 20 == 0:
            print(f"  [mlp] ep{ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    model.eval()
    return model


def dump_features(model, X):
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 512):
            x = torch.from_numpy(X[s:s + 512].reshape(len(X[s:s + 512]), -1)).to(next(model.parameters()).device)
            _, f = model(x, return_feat=True)
            out.append(f.cpu().numpy())
    return np.vstack(out)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="PYSKL pkl path")
    ap.add_argument("--name", required=True, help="experiment name (ntu120/ucf101)")
    ap.add_argument("--max", type=int, default=12000)
    ap.add_argument("--seeds", type=int, default=3, help="3 or 10 (E9-series expansion)")
    args = ap.parse_args()
    global SEEDS
    if args.seeds == 10:
        SEEDS = tuple(range(42, 52))
    import torch
    from psd.training.tcl_selftrain import run_selftrain
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_dataset(Path(args.pkl), args.max)

    print("[prep] resampling clips...")
    X = np.stack([prep_clip(r["kp"]) for r in rows])
    labels = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    tr, vm = splits == "train", splits == "val"
    n_cls = len(set(labels.tolist()))
    print(f"[data] X {X.shape} train={tr.sum()} val={vm.sum()} classes={n_cls}")

    print("[pretext] joint-MLP 80ep...")
    model = train_pretext_mlp(X[tr], labels[tr], n_cls, device=device)
    feats = dump_features(model, X)
    print(f"[feat] {feats.shape}")

    class ScaledLR:
        def __init__(self):
            from sklearn.linear_model import LogisticRegression
            self.sc = StandardScaler(); self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)
        def fit(self, X_, y_): self.sc.fit(X_, y_); self.clf.fit(self.sc.transform(X_), y_); return self
        def predict(self, X_): return self.clf.predict(self.sc.transform(X_))

    ref = float(np.mean(ScaledLR().fit(feats[tr], labels[tr]).predict(feats[vm]) == labels[vm]))
    print(f"[c] full ref: {ref:.4f}")

    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]; tr_labels = labels[tr_idx]
    mask_a = np.zeros(tr.sum(), dtype=bool)
    for c in sorted(set(tr_labels.tolist())):
        ci = np.where(tr_labels == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    acc_a = float(np.mean(ScaledLR().fit(feats[tr_idx[mask_a]], labels[tr_idx[mask_a]]).predict(feats[vm]) == labels[vm]))
    print(f"[a] 10% linear: {acc_a:.4f}")

    labels_str = np.array([str(l) for l in labels])
    class_names = [str(c) for c in sorted(set(labels.tolist()))]
    accs_b = []
    for seed in SEEDS:
        anchor = np.zeros(len(labels), dtype=bool)
        rng2 = np.random.default_rng(seed)
        for c in class_names:
            ci = np.where((labels_str == c) & tr)[0]
            k = max(1, int(round(len(ci) * BUDGET_FRAC)))
            anchor[rng2.choice(ci, size=k, replace=False)] = True
        universe = tr & ~anchor
        r = run_selftrain(feats, labels_str, anchor, run_seed=seed, class_names=class_names,
                          head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                    "weight_decay": 1e-4, "batch_size": 128, "device": "cpu"},
                          pool_universe_mask=universe, **KW)
        pool_idx = r["final_pool_idx"]
        tm2 = anchor.copy(); tm2[pool_idx] = True
        y2 = np.array([None] * len(labels_str), dtype=object)
        y2[anchor] = labels_str[anchor]
        y2[pool_idx] = r["final_pred_full"][pool_idx]
        # 评分必须在字符串域: predict 返回类名字符串, 与 labels_str 比较
        # (此前误与整数 labels 比较, str!=int 恒 False → b 臂伪 0.0)
        pred_b = ScaledLR().fit(feats[tm2], y2[tm2]).predict(feats[vm])
        acc_b = float(np.mean(pred_b == labels_str[vm]))
        accs_b.append(acc_b)
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(pool_idx)} stop={r['stop_reason']}")

    ret = float(np.mean(accs_b)) / ref
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": f"PSD-{args.name.upper()}-PREREG-001",
        "full_ref": round(ref, 4), "linear_10pct": round(acc_a, 4),
        "b_arms": [round(a, 4) for a in accs_b],
        "retention": round(ret, 4), "linear_retention": round(acc_a / ref, 4),
        "verdict": verdict,
        "config_echo": {"n_used": int(len(labels)), "classes": n_cls, "seeds": list(SEEDS)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p5b-{args.name}-retention-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
