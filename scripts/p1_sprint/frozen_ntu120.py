# -*- coding: utf-8 -*-
"""P1 冲刺 — NTU120 frozen-backbone 对照臂（真实 tier × 冻结编码器）.

审稿人质疑（PR-D-26-13140）: 对照表中「冻结是否够」目前只有合成层证据
（frozen_backbone_contrast.py，frozen 68.6%）。本脚本把同一问题补到 NTU120
真实 tier：与已报告管线 scripts/run_p5b_generic_retention.py（2026-09-07，
reports/p5b-ntu120-retention-2026-09-07.json：full_ref=0.5434，
linear_10pct=0.4542，retention=0.8886 PARTIAL，10 seeds）完全同协议——
同 pkl（data/pyskl/ntu120_hrnet.pkl）、同加载/预处理（load_dataset/prep_clip，
T=30 重采样+center）、同 encoder（JointMLP flatten(T*V*3)→512→256，seed42/
80ep/AdamW lr1e-3 wd1e-4）、同三臂与判据——唯一差异 = encoder 显式冻结并
持久化:
  1) encoder 权重落盘 runs/p1_sprint/ntu120/encoder.pt（已报告运行未存权重，
     首次全量运行以同协议重训并保存；此后运行直接加载同一 ckpt 冻结复用）；
  2) encoder 全参数 requires_grad=False + eval（trainable=0 断言），仅训分类头；
  3) 10% 分层标注预算抽选与 E9 逐行一致（default_rng(42)，每类
     k=max(1, round(10%)))。
三臂（全部在冻结 256d 特征上、只训头）:
  (c) 100% 线性参照 full_ref / (a) 10% 线性 linear_10pct /
  (b) 10%+PSD 语义管线（run_selftrain，与 E9 同参）→ retention=(b)/full_ref。
判据同 E9: ≥90% CONFIRMS / 85-90 PARTIAL / <85 FAILS。

诚实性声明（blocker 如实记录，不硬造）: 已报告运行
（run_p5b_generic_retention.py:99-121,163-165）不持久化 encoder 权重
（内存训练后即弃，无 torch.save），runs/p19_ntu120/pretext_best.pt 亦不存在。
因此本 frozen 臂的权重 = 同协议重训（同 pkl/同预处理/同 seed/同 80ep
JointMLP），与 2026-09-07 报告运行协议一致而非逐位一致（GPU 非确定性）；
provenance 与 caveat 原样写入 summary.json，供对照表标注口径。

与 p5b 的两处实现性差异（均不改变全量口径）:
  - b 臂 class_names 取 train 内出现类（p5b 取子集全部类；NTU120 12000 子集
    60 类全部在 train 出现，全量下二者等价——已报告 b 臂即 60 类跑通）；
  - smoke 模式 = 全量数据加载 + 2ep pretext + 1 seed（pkl 注解 train 在前，
    按条数截断会得到纯 train 子集、val 为空，故 smoke 不截断数据，只缩训练量
    与 seed；独立 encoder_smoke.pt），数字无意义仅验证管线。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/frozen_ntu120.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/frozen_ntu120.py --seeds 10
产出:
    runs/p1_sprint/ntu120/summary.json（全量；冒烟为 summary_smoke.json）
    runs/p1_sprint/ntu120/encoder.pt（冻结 encoder 权重）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# 复用已报告管线原语（同一函数对象 → encoder/特征/数据路径逐项同源）
from run_p5b_generic_retention import (  # noqa: E402
    BUDGET_FRAC, KW, JointMLP, dump_features, load_dataset, prep_clip,
    train_pretext_mlp,
)

TIER = "ntu120"
PKL = REPO_ROOT / "data" / "pyskl" / "ntu120_hrnet.pkl"
RUN_DIR = REPO_ROOT / "runs" / "p1_sprint" / TIER
PROTOCOL = "PSD-NTU120-PREREG-001"
BASE_PIPELINE = "scripts/run_p5b_generic_retention.py (reported 2026-09-07)"
SEED_MAP = {1: (42,), 3: (42, 43, 44),
            10: tuple(range(42, 52)), 30: tuple(range(42, 72))}


class ScaledLR:
    """与 run_p5b_generic_retention.main 内定义逐字一致（StandardScaler+LR）。"""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.sc = StandardScaler()
        self.clf = LogisticRegression(max_iter=1000, C=1.0, tol=1e-3)

    def fit(self, X_, y_):
        self.sc.fit(X_, y_)
        self.clf.fit(self.sc.transform(X_), y_)
        return self

    def predict(self, X_):
        return self.clf.predict(self.sc.transform(X_))


def seeds_for(n: int) -> tuple:
    return SEED_MAP.get(n, tuple(range(42, 42 + n)))


def get_frozen_encoder(X, labels, tr, n_cls, device, epochs, ckpt: Path):
    """加载已持久化 encoder 并冻结；无则按已报告协议重训并落盘后冻结。"""
    in_dim = int(np.prod(X.shape[1:]))
    model = None
    if ckpt.exists():
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        proto = ck.get("protocol", {})
        if proto.get("n_classes") == int(n_cls) and proto.get("in_dim") == in_dim:
            model = JointMLP(in_dim, n_cls).to(device)
            model.load_state_dict(ck["model_state_dict"])
            provenance = "loaded_persisted_ckpt"
            print(f"[encoder] loaded persisted {ckpt}")
        else:
            print("[encoder] ckpt 口径不符（类别/维度变更），按协议重训")
    if model is None:
        # 与已报告管线同协议: 同 X / 同 train split / 同 seed42 / 同 80ep JointMLP
        model = train_pretext_mlp(X[tr], labels[tr], n_cls, epochs=epochs, device=device)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "protocol": {"arch": "JointMLP", "in_dim": in_dim,
                                 "n_classes": int(n_cls), "epochs": int(epochs),
                                 "seed": 42, "batch": 256, "lr": 1e-3, "wd": 1e-4}},
                   ckpt)
        provenance = "retrained_same_protocol_saved"
        print(f"[encoder] retrained (protocol-identical) + saved {ckpt}")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert n_trainable == 0, "冻结失败: 仍有可训练参数"
    print(f"[encoder] frozen verified: trainable={n_trainable}/{n_total}")
    return model, provenance, n_trainable, n_total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="缩样管线验证（数字无意义）")
    ap.add_argument("--max", type=int, default=12000, help="与已报告口径一致: 12000")
    ap.add_argument("--seeds", type=int, default=10, help="b 臂 seed 数（报告口径 10）")
    ap.add_argument("--epochs", type=int, default=80, help="pretext epoch（报告口径 80）")
    args = ap.parse_args()

    smoke = args.smoke
    if smoke:
        # 全量数据 + 2ep + 1 seed: pkl 注解 train 在前，截断会致 val 为空
        args.epochs = 2
        seeds = SEED_MAP[1]
        ckpt = RUN_DIR / "encoder_smoke.pt"
    else:
        seeds = seeds_for(args.seeds)
        ckpt = RUN_DIR / "encoder.pt"

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from psd.training.tcl_selftrain import run_selftrain

    rows = load_dataset(PKL, args.max)
    X = np.stack([prep_clip(r["kp"]) for r in rows])
    labels = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    tr, vm = splits == "train", splits == "val"
    n_cls = len(set(labels.tolist()))
    print(f"[data] X {X.shape} train={tr.sum()} val={vm.sum()} classes={n_cls}")

    model, provenance, n_trainable, n_total = get_frozen_encoder(
        X, labels, tr, n_cls, device, args.epochs, ckpt)
    feats = dump_features(model, X)
    print(f"[feat] {feats.shape} (frozen encoder)")

    # (c) 100% 线性参照（冻结特征上仅训头）
    ref = float(np.mean(ScaledLR().fit(feats[tr], labels[tr]).predict(feats[vm]) == labels[vm]))
    print(f"[c] full ref: {ref:.4f}")

    # (a) 10% 分层子集线性头（与 E9 逐行同式）
    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    tr_labels = labels[tr_idx]
    mask_a = np.zeros(tr.sum(), dtype=bool)
    for c in sorted(set(tr_labels.tolist())):
        ci = np.where(tr_labels == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    acc_a = float(np.mean(ScaledLR().fit(feats[tr_idx[mask_a]], labels[tr_idx[mask_a]])
                          .predict(feats[vm]) == labels[vm]))
    print(f"[a] 10% linear: {acc_a:.4f}")

    # (b) 10% + PSD 语义管线（与 E9 同参，冻结特征）
    labels_str = np.array([str(l) for l in labels])
    class_names = [str(c) for c in sorted(set(labels[tr].tolist()))]
    accs_b = []
    for seed in seeds:
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
        pred_b = ScaledLR().fit(feats[tm2], y2[tm2]).predict(feats[vm])
        acc_b = float(np.mean(pred_b == labels_str[vm]))
        accs_b.append(acc_b)
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(pool_idx)} stop={r['stop_reason']}")

    ret = float(np.mean(accs_b)) / ref if ref > 0 else 0.0
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "doneMarker": f"P1_SPRINT_FROZEN_NTU120_{'SMOKE_OK' if smoke else 'DONE'}",
        "experiment": "p1_sprint_frozen_backbone_real_tier",
        "tier": TIER,
        "mode": "smoke" if smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": f"{PROTOCOL} (frozen-encoder arm)",
        "base_pipeline": BASE_PIPELINE,
        "encoder": {"arch": "JointMLP flatten(T*V*3)->512->ReLU->256 (+Linear head)",
                    "ckpt": str(ckpt.relative_to(REPO_ROOT)),
                    "epochs": int(args.epochs), "seed": 42, "frozen": True,
                    "trainable_params": int(n_trainable), "total_params": int(n_total)},
        "weights_provenance": provenance,
        "weights_caveat": ("reported run (2026-09-07) did not persist encoder weights; "
                           "frozen arm uses same-protocol retrain (same pkl/prep/seed/80ep "
                           "JointMLP); protocol-identical, not bitwise-identical (GPU "
                           "nondeterminism)"),
        "budget_frac": BUDGET_FRAC,
        "full_ref": round(ref, 4), "linear_10pct": round(acc_a, 4),
        "b_arms": [round(a, 4) for a in accs_b],
        "retention": round(ret, 4), "linear_retention": round(acc_a / ref, 4) if ref > 0 else 0.0,
        "verdict": verdict,
        "config_echo": {"pkl": str(PKL.relative_to(REPO_ROOT)), "max_n": int(args.max),
                        "n_used": int(len(labels)), "classes": int(n_cls),
                        "seeds": list(seeds)},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    suffix = "_smoke" if smoke else ""
    out = RUN_DIR / f"summary{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
