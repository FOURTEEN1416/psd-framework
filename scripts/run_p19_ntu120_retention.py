# -*- coding: utf-8 -*-
"""P5-B NTU120 低资源保留率 — PSD-NTU120-PREREG-001 主终点执行。

三臂（镜像 E9，预注册 §2）:
  (c) 100% train 线性头参照（from-scratch ST-GCN pretext 150ep 冻结特征）
  (a) 10% 分层子集线性头
  (b) 10% + PSD 语义管线（修正协议，precision_stop=False）
主终点: retention = top1(b)/top1(c)，判据 ≥90% CONFIRMS / 85-90 PARTIAL / <85 FAILS。

用法: .venv/Scripts/python.exe scripts/run_p19_ntu120_retention.py [--pretext]
  --pretext 先跑 ST-GCN pretext（150ep）；无此参则用已有 checkpoint 跑三臂。
产出: reports/p19-ntu120-retention-<date>.json
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

PKL = REPO / "data" / "ntu120" / "ntu120_3danno.pkl"
CKPT = REPO / "runs" / "p19_ntu120" / "pretext_best.pt"
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
BUDGET_FRAC = 0.10
KW = dict(calib_method="softmax_temperature", calib_target=0.10,
          tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
          tau_select={"rule": "quantile", "target_coverage": 0.35},
          alpha=1.0, standing_mode="none", subcluster_k=2, subcluster_min_share=0.05,
          max_iters=6, converge_change_rate=0.01, precision_drop_patience=2,
          precision_stop=False, gate_mode="standing")


def load_ntu120():
    """PYSKL 格式: {'split': {'xsub_train': [keys]}, 'annotations': [{'keypoint':(T,1,17,2),...}]}"""
    print("[ntu120] loading pkl (2GB, ~1min)...")
    d = pickle.load(open(PKL, "rb"))
    anns = d["annotations"]
    split = d["split"]
    train_keys = set(split["xsub_train"])
    rows = []
    for a in anns:
        kp = a["keypoint"]  # (1,T,17,2) or (T,1,17,2)
        kp = np.asarray(kp, dtype=np.float32)
        if kp.ndim == 4 and kp.shape[0] == 1:
            kp = kp.transpose(1, 0, 2, 3)[0]  # (T,17,2)
        if kp.ndim == 4:
            kp = kp[:, 0]  # (T,17,2)
        label = int(a["label"])
        key = a.get("frame_dir", a.get("filename", str(len(rows))))
        rows.append({"key": key, "kp": kp, "label": label,
                     "split": "train" if key in train_keys else "val"})
    print(f"[ntu120] {len(rows)} clips, train={sum(1 for r in rows if r['split']=='train')}")
    return rows


def train_pretext(rows, epochs=150, batch=32, device="cuda"):
    """from-scratch 简化 pretext: ST-GCN 骨干重采样 T=30 训练（够产生冻结特征）。
    为控时间用轻量 ST-GCN（psd.models.stgcn_bc, in_ch=3: x,y,zero-conf）。"""
    import torch
    from psd.models.stgcn_bc import build_stgcn_bc
    from psd.training.stgcnbc_feature_extractor import center_keypoints

    torch.manual_seed(42)
    model = build_stgcn_bc(in_channels=3, num_classes=120).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()

    def prep(r):
        kp = np.asarray(r["kp"], dtype=np.float32)
        T = kp.shape[0]
        if T < 30:
            idx = np.resize(np.arange(T), 30)
        else:
            idx = np.linspace(0, T - 1, 30, dtype=int)
        kp = kp[idx]
        conf = np.ones((30, 17, 1), dtype=np.float32)
        kp3 = np.concatenate([kp, conf], axis=2)  # (30,17,3)
        return center_keypoints(kp3)

    train_rows = [r for r in rows if r["split"] == "train"]
    X = np.stack([prep(r) for r in train_rows[:8000]])  # 子集防 OOM（8k clips 快速 pretext）
    y = np.array([r["label"] for r in train_rows[:8000]])
    print(f"[pretext] training on {len(X)} clips, {epochs} epochs")
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(len(X))
        tot, nb = 0.0, 0
        for s in range(0, len(X), batch):
            idx = perm[s:s + batch]
            x = torch.from_numpy(X[idx]).to(device)
            yt = torch.from_numpy(y[idx]).long().to(device)
            logits, _ = model(x)
            loss = ce(logits, yt)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        if (ep + 1) % 25 == 0:
            print(f"  [pretext] ep{ep+1}/{epochs} loss={tot/max(nb,1):.4f}")
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, CKPT)
    print(f"[pretext] saved {CKPT}")
    model.eval()
    return model


def dump_features(model, rows, device="cuda", max_n=12000):
    import torch
    from psd.training.stgcnbc_feature_extractor import center_keypoints
    feats, labels, splits, keys = [], [], [], []
    sel = rows[:max_n]
    for i, r in enumerate(sel):
        kp = np.asarray(r["kp"], dtype=np.float32)
        T = kp.shape[0]
        idx = np.resize(np.arange(T), 30) if T < 30 else np.linspace(0, T - 1, 30, dtype=int)
        kp = center_keypoints(np.concatenate([kp[idx], np.ones((30, 17, 1), np.float32)], axis=2))
        feats.append(kp.astype(np.float32))
        labels.append(r["label"]); splits.append(r["split"]); keys.append(r["key"])
        if (i + 1) % 2000 == 0:
            print(f"  [feat] {i+1}/{len(sel)}")
    with torch.no_grad():
        out = []
        for s in range(0, len(feats), 64):
            x = torch.from_numpy(np.stack(feats[s:s + 64])).to(device)
            _, f = model(x)
            out.append(f.cpu().numpy())
    return np.vstack(out), np.array(labels), np.array(splits), keys


def main():
    import argparse
    from datetime import datetime
    import torch
    from psd.models.stgcn_bc import build_stgcn_bc
    from psd.training.tcl_selftrain import run_selftrain
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    ap = argparse.ArgumentParser()
    ap.add_argument("--pretext", action="store_true")
    ap.add_argument("--max", type=int, default=12000)
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_ntu120()
    model = build_stgcn_bc(in_channels=3, num_classes=120).to(device)
    if args.pretext or not CKPT.exists():
        model = train_pretext(rows, device=device)
    else:
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        print(f"[ntu120] loaded pretext {CKPT}")

    print("[ntu120] dumping features...")
    feats, labels, splits, keys = dump_features(model, rows, device, args.max)
    tr = splits == "train"; vm = splits == "val"
    print(f"[ntu120] feats {feats.shape} train={tr.sum()} val={vm.sum()} classes={len(set(labels.tolist()))}")

    class ScaledLR:
        def __init__(self):
            from sklearn.linear_model import LogisticRegression
            self.sc = StandardScaler()
            self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)
        def fit(self, X, y):
            self.sc.fit(X, y); self.clf.fit(self.sc.transform(X), y); return self
        def predict(self, X):
            return self.clf.predict(self.sc.transform(X))

    def lin(mask):
        m = ScaledLR().fit(feats[mask], labels[mask])
        return m.predict(feats[vm])

    # (c) 全预算参照
    ref = float(np.mean(lin(tr) == labels[vm]))
    print(f"[c] full ref: {ref:.4f}")

    # 10% 分层子集
    rng = np.random.default_rng(42)
    classes = sorted(set(labels[tr].tolist()))
    mask_a = np.zeros(tr.sum(), dtype=bool)
    tr_idx = np.where(tr)[0]
    tr_labels = labels[tr_idx]
    for c in classes:
        ci = np.where(tr_labels == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    a_pred = lin(tr_idx[mask_a])
    acc_a = float(np.mean(a_pred == labels[vm]))
    print(f"[a] 10% linear: {acc_a:.4f}")

    # (b) 10% + PSD 语义管线（3 seeds）
    accs_b = []
    labels_str = np.array([str(l) for l in labels])
    class_names = [str(c) for c in classes]
    for seed in SEEDS:
        anchor_local = np.zeros(tr.sum(), dtype=bool)
        rng2 = np.random.default_rng(seed)
        for c in classes:
            ci = np.where(tr_labels == c)[0]
            k = max(1, int(round(len(ci) * BUDGET_FRAC)))
            anchor_local[rng2.choice(ci, size=k, replace=False)] = True
        anchor = np.zeros(len(labels), dtype=bool)
        anchor[tr_idx[anchor_local]] = True
        universe = tr & ~anchor
        r = run_selftrain(feats, labels_str, anchor, run_seed=seed,
                          class_names=class_names, head_cfg={"hidden_dim": 64, "epochs": 100,
                                                             "lr": 1e-3, "weight_decay": 1e-4,
                                                             "batch_size": 128, "device": "cuda"},
                          pool_universe_mask=universe, **KW)
        pool_idx = r["final_pool_idx"]
        tm2 = anchor.copy(); tm2[pool_idx] = True
        y2 = np.array([None] * len(labels_str), dtype=object)
        y2[anchor] = labels_str[anchor]
        y2[pool_idx] = r["final_pred_full"][pool_idx]
        clf = ScaledLR().fit(feats[tm2], y2[tm2])
        pred = clf.predict(feats[vm])
        acc_b = float(np.mean(pred == labels[vm]))
        accs_b.append(acc_b)
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(pool_idx)} stop={r['stop_reason']}")

    ret = float(np.mean(accs_b)) / ref
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-NTU120-PREREG-001",
        "full_ref": round(ref, 4), "linear_10pct": round(acc_a, 4),
        "b_arms": [round(a, 4) for a in accs_b],
        "retention": round(ret, 4), "linear_retention": round(acc_a / ref, 4),
        "verdict": verdict,
        "config_echo": {"n_used": int(len(labels)), "budget_frac": BUDGET_FRAC, "seeds": list(SEEDS)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p19-ntu120-retention-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
