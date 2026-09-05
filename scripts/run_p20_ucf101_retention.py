# -*- coding: utf-8 -*-
"""P5-A UCF101 保留率 — 第二独立跨域点（HRNet 2D 骨架，openmmlab CDN 直链）。

UCF101: 101 类动作，13,320 clips，YouTube 野外视频，HRNet 2D 骨架（已预提取）。
与 NTU60/120 真独立：不同视频来源/动作空间/骨架提取器/协议。
判据与 NTU120 同：≥90% CONFIRMS / 85-90 PARTIAL / <85 FAILS。

用法: .venv/Scripts/python.exe scripts/run_p20_ucf101_retention.py [--max 12000]
产出: reports/p20-ucf101-retention-<date>.json
"""
from __future__ import annotations
import json, pickle, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
PKL = REPO / "data" / "pyskl" / "ucf101_hrnet.pkl"
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
BUDGET_FRAC = 0.10
KW = dict(calib_method="softmax_temperature", calib_target=0.10,
          tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
          tau_select={"rule": "quantile", "target_coverage": 0.35},
          alpha=1.0, standing_mode="none", subcluster_k=2, subcluster_min_share=0.05,
          max_iters=6, converge_change_rate=0.01, precision_drop_patience=2,
          precision_stop=False, gate_mode="standing")

def load_ucf101():
    print("[ucf101] loading pkl...")
    d = pickle.load(open(PKL, "rb"))
    anns = d["annotations"]
    # use split1 as canonical
    split = d["split"]
    train_keys = set(split.get("train1", split.get("xsub_train", [])))
    val_keys = set(split.get("test1", split.get("xsub_val", [])))
    rows = []
    for a in anns:
        kp = np.asarray(a["keypoint"], dtype=np.float32)
        if kp.ndim == 4 and kp.shape[0] == 1:
            kp = kp.transpose(1, 0, 2, 3)[0]
        if kp.ndim == 4:
            kp = kp[:, 0]
        kps = np.asarray(a.get("keypoint_score", np.ones(kp.shape[:2], np.float32)), dtype=np.float32)
        if kps.ndim == 2: kps = kps[:, :, np.newaxis]
        kp3 = np.concatenate([kp, kps], axis=-1).astype(np.float32)  # (T,17,3)
        label = int(a["label"])
        key = str(a.get("frame_dir", len(rows)))
        if key in train_keys: sp = "train"
        elif key in val_keys: sp = "val"
        else: sp = "train"  # default to train if not in either
        rows.append({"key": key, "kp": kp3, "label": label, "split": sp})
    print(f"[ucf101] {len(rows)} clips, train={sum(1 for r in rows if r['split']=='train')}")
    return rows

def main():
    import argparse, torch
    from datetime import datetime
    from psd.models.stgcn_bc import build_stgcn_bc
    from psd.training.tcl_selftrain import run_selftrain
    from psd.training.stgcnbc_feature_extractor import center_keypoints
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=10000)
    ap.add_argument("--skip-pretext", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_ucf101()

    CKPT = REPO / "runs" / "p20_ucf101" / "pretext_best.pt"
    if CKPT.exists():
        print("[ucf101] loading existing pretext...")
        model = build_stgcn_bc(in_channels=3, num_classes=101).to(device)
        ck = torch.load(CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
    else:
        print("[ucf101] training from-scratch pretext 100ep...")
        from run_p19_ntu120_retention import train_pretext
        model = train_pretext(rows, epochs=100, batch=16, device=device)  # use train_pretext from p19
        model.eval()

    def prep_and_dump(max_n=10000):
        from psd.training.stgcnbc_feature_extractor import center_keypoints
        feats, labels, splits = [], [], []
        sel = rows[:max_n]
        for i, r in enumerate(sel):
            kp = np.asarray(r["kp"], dtype=np.float32)
            T = kp.shape[0]
            V = kp.shape[-2]
            idx = np.resize(np.arange(T), 30) if T < 30 else np.linspace(0, T - 1, 30, dtype=int)
            kp_sel = center_keypoints(kp[idx])
            feats.append(kp_sel.astype(np.float32))
            labels.append(r["label"]); splits.append(r["split"])
            if (i + 1) % 2000 == 0:
                print(f"  [feat] {i+1}/{len(sel)}")
        return np.stack(feats), np.array(labels), np.array(splits)

    feats, labels, splits = prep_and_dump(args.max)
    tr = splits == "train"; vm = splits == "val"
    print(f"[ucf101] feats {feats.shape} train={tr.sum()} val={vm.sum()} classes={len(set(labels.tolist()))}")

    class ScaledLR:
        def __init__(self):
            from sklearn.linear_model import LogisticRegression
            self.sc = StandardScaler(); self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)
        def fit(self, X, y): self.sc.fit(X, y); self.clf.fit(self.sc.transform(X), y); return self
        def predict(self, X): return self.clf.predict(self.sc.transform(X))

    ref = float(np.mean(ScaledLR().fit(feats[tr], labels[tr]).predict(feats[vm]) == labels[vm]))
    print(f"[c] full ref: {ref:.4f}")

    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]; tr_labels = labels[tr_idx]
    mask_a = np.zeros(tr.sum(), dtype=bool)
    for c in sorted(set(tr_labels.tolist())):
        ci = np.where(tr_labels == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    a_pred = ScaledLR().fit(feats[tr_idx[mask_a]], labels[tr_idx[mask_a]]).predict(feats[vm])
    acc_a = float(np.mean(a_pred == labels[vm]))
    print(f"[a] 10% linear: {acc_a:.4f}")

    # (b) PSD semantic pipeline × 3 seeds
    accs_b = []
    labels_str = np.array([str(l) for l in labels])
    class_names = [str(c) for c in sorted(set(labels.tolist()))]
    for seed in SEEDS:
        anchor = np.zeros(len(labels), dtype=bool)
        rng2 = np.random.default_rng(seed)
        for c in class_names:
            ci = np.where((labels_str == c) & tr)[0]
            k = max(1, int(round(len(ci) * BUDGET_FRAC)))
            anchor[rng2.choice(ci, size=k, replace=False)] = True
        universe = tr & ~anchor
        r = run_selftrain(feats, labels_str, anchor, run_seed=seed,
                          class_names=class_names,
                          head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                    "weight_decay": 1e-4, "batch_size": 128, "device": "cpu"},
                          pool_universe_mask=universe, **KW)
        pool_idx = r["final_pool_idx"]
        tm2 = anchor.copy(); tm2[pool_idx] = True
        y2 = np.array([None] * len(labels_str), dtype=object)
        y2[anchor] = labels_str[anchor]
        y2[pool_idx] = r["final_pred_full"][pool_idx]
        clf = ScaledLR().fit(feats[tm2], y2[tm2])
        acc_b = float(np.mean(clf.predict(feats[vm]) == labels[vm]))
        accs_b.append(acc_b)
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(pool_idx)} stop={r['stop_reason']}")

    ret = float(np.mean(accs_b)) / ref
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-UCF101-PREREG-001 (second independent cross-domain retention point)",
        "full_ref": round(ref, 4), "linear_10pct": round(acc_a, 4),
        "b_arms": [round(a, 4) for a in accs_b],
        "retention": round(ret, 4), "linear_retention": round(acc_a / ref, 4),
        "verdict": verdict,
        "config_echo": {"n_used": int(len(labels)), "budget_frac": BUDGET_FRAC,
                        "classes": len(set(labels.tolist())), "seeds": list(SEEDS)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"p20-ucf101-retention-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
