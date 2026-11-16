# -*- coding: utf-8 -*-
"""P1 冲刺 — PanAf500 frozen-backbone 对照臂（真实 tier × 冻结编码器）.

审稿人质疑（PR-D-26-13140）: 对照表中「冻结是否够」目前只有合成层证据
（frozen_backbone_contrast.py，frozen 68.6%）。本脚本把同一问题补到 PanAf500
真实 tier：与已报告管线 scripts/run_p23_panaf_retention.py（2026-09-07，
reports/p23-panaf-retention-2026-09-07.json：full_ref=0.6，
linear_10pct=0.52，retention=0.8578 PARTIAL，30 seeds）完全同协议——
同 pkl（runs/p7_asbar/panaf500_T30.pkl，官方 split 400/25/75，9 类长尾）、
同预处理（prep_clip center）、同 encoder（JointMLP flatten(T*V*3)→512→256，
seed42/80ep/AdamW lr1e-3 wd1e-4，batch64）、同三臂与判据——唯一差异 =
encoder 显式冻结并持久化:
  1) encoder 权重落盘 runs/p1_sprint/panaf500/encoder.pt（已报告运行未存权重，
     首次全量运行以同协议重训并保存；此后运行直接加载同一 ckpt 冻结复用）；
  2) encoder 全参数 requires_grad=False + eval（trainable=0 断言），仅训分类头；
  3) 10% 分层标注预算抽选与 E9 逐行一致（default_rng(42/seed)，每类
     k=max(1, round(10%))；b 臂只遍历 train 内出现类——9 类长尾中有的类仅
     出现在 val/test，与 p23 口径一致）。
三臂（全部在冻结 256d 特征上、只训头）:
  (c) 100% 线性参照 full_ref / (a) 10% 线性 linear_10pct /
  (b) 10%+PSD 语义管线（run_selftrain，与 E9 同参）→ retention=(b)/full_ref。
判据同 E9: ≥90% CONFIRMS / 85-90 PARTIAL / <85 FAILS。

诚实性声明（blocker 如实记录，不硬造）: 已报告运行
（run_p23_panaf_retention.py:73-93 内存训练，无 torch.save）不持久化 encoder
权重。因此本 frozen 臂的权重 = 同协议重训（同 pkl/同预处理/同 seed/同 80ep
JointMLP），与 2026-09-07 报告运行协议一致而非逐位一致（GPU 非确定性）；
provenance 与 caveat 原样写入 summary.json，供对照表标注口径。

smoke 模式缩样（前 120 clips、2ep、1 seed、独立 encoder_smoke.pt），数字无
意义仅验证管线。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/frozen_panaf500.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/frozen_panaf500.py --seeds 30
产出:
    runs/p1_sprint/panaf500/summary.json（全量；冒烟为 summary_smoke.json）
    runs/p1_sprint/panaf500/encoder.pt（冻结 encoder 权重）
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
from run_p23_panaf_retention import (  # noqa: E402
    BUDGET_FRAC, KW, JointMLP, ScaledLR, dump_features, load_rows, prep_clip,
    train_pretext,
)

TIER = "panaf500"
RUN_DIR = REPO_ROOT / "runs" / "p1_sprint" / TIER
PROTOCOL = "PSD-PANAF-PREREG-001"
BASE_PIPELINE = "scripts/run_p23_panaf_retention.py (reported 2026-09-07)"
SEED_MAP = {1: (42,), 3: (42, 43, 44),
            10: tuple(range(42, 52)), 30: tuple(range(42, 72))}


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
        model = train_pretext(X[tr], labels[tr], n_cls, epochs=epochs, device=device)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "protocol": {"arch": "JointMLP", "in_dim": in_dim,
                                 "n_classes": int(n_cls), "epochs": int(epochs),
                                 "seed": 42, "batch": 64, "lr": 1e-3, "wd": 1e-4}},
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
    ap.add_argument("--seeds", type=int, default=30, help="b 臂 seed 数（报告口径 30）")
    ap.add_argument("--epochs", type=int, default=80, help="pretext epoch（报告口径 80）")
    ap.add_argument("--with-test", action="store_true", help="附 test split 次级数字（非判据）")
    args = ap.parse_args()

    smoke = args.smoke
    if smoke:
        args.epochs = 2
        seeds = SEED_MAP[1]
        ckpt = RUN_DIR / "encoder_smoke.pt"
    else:
        seeds = seeds_for(args.seeds)
        ckpt = RUN_DIR / "encoder.pt"

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from psd.training.tcl_selftrain import run_selftrain

    X, labels, splits, classes = load_rows()
    if smoke:  # 缩样: 前 120 clips（管线验证，不代表长尾口径）
        X, labels, splits = X[:120], labels[:120], splits[:120]
    tr, vm = splits == "train", splits == "val"
    n_cls = len(set(labels.tolist()))
    print(f"[data] X {X.shape} train={tr.sum()} val={vm.sum()} classes={n_cls}")

    model, provenance, n_trainable, n_total = get_frozen_encoder(
        X, labels, tr, n_cls, device, args.epochs, ckpt)
    feats = dump_features(model, X)
    print(f"[feat] {feats.shape} (frozen encoder)")

    labels_str = np.array([str(l) for l in labels])

    # (c) 100% 线性参照（冻结特征上仅训头）
    ref = float(np.mean(ScaledLR().fit(feats[tr], labels_str[tr]).predict(feats[vm]) == labels_str[vm]))
    print(f"[c] full ref: {ref:.4f}")

    # (a) 10% 分层子集线性头（与 E9/p23 逐行同式）
    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    mask_a = np.zeros(len(tr_idx), dtype=bool)
    for c in sorted(set(labels[tr_idx].tolist())):
        ci = np.where(labels[tr_idx] == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    acc_a = float(np.mean(ScaledLR().fit(feats[tr_idx[mask_a]], labels_str[tr_idx[mask_a]])
                          .predict(feats[vm]) == labels_str[vm]))
    print(f"[a] 10% linear: {acc_a:.4f}")

    # (b) 10% + PSD 语义管线（与 p23 同参，冻结特征；类集= train 内出现类）
    classes_tr = sorted(set(labels_str[tr].tolist()))
    accs_b, pools = [], []
    for seed in seeds:
        anchor = np.zeros(len(labels_str), dtype=bool)
        rng2 = np.random.default_rng(seed)
        for c in classes_tr:
            ci = np.where((labels_str == c) & tr)[0]
            k = max(1, int(round(len(ci) * BUDGET_FRAC)))
            anchor[rng2.choice(ci, size=k, replace=False)] = True
        universe = tr & ~anchor
        r = run_selftrain(feats, labels_str, anchor, run_seed=seed,
                          class_names=classes_tr,
                          head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                    "weight_decay": 1e-4, "batch_size": 128,
                                    "device": "cuda" if torch.cuda.is_available() else "cpu"},
                          pool_universe_mask=universe, **KW)
        pool_idx = r["final_pool_idx"]
        tm2 = anchor.copy(); tm2[pool_idx] = True
        y2 = np.array([None] * len(labels_str), dtype=object)
        y2[anchor] = labels_str[anchor]
        y2[pool_idx] = r["final_pred_full"][pool_idx]
        pred_b = ScaledLR().fit(feats[tm2], y2[tm2]).predict(feats[vm])
        acc_b = float(np.mean(pred_b == labels_str[vm]))
        accs_b.append(acc_b); pools.append(len(pool_idx))
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(pool_idx)} stop={r['stop_reason']}")

    ret = float(np.mean(accs_b)) / ref if ref > 0 else 0.0
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    # p23 §2: test split 次级数字（非判据）
    test_num = None
    if args.with_test and (splits == "test").any():
        tm = splits == "test"
        test_num = round(float(np.mean(ScaledLR().fit(feats[tr], labels_str[tr])
                                       .predict(feats[tm]) == labels_str[tm])), 4)
        print(f"[test] secondary (non-decision): {test_num}")
    result = {
        "doneMarker": f"P1_SPRINT_FROZEN_PANAF500_{'SMOKE_OK' if smoke else 'DONE'}",
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
        "pools": pools,
        "retention": round(ret, 4), "linear_retention": round(acc_a / ref, 4) if ref > 0 else 0.0,
        "verdict": verdict,
        "test_secondary_nondecision": test_num,
        "config_echo": {"pkl": "runs/p7_asbar/panaf500_T30.pkl",
                        "n_clips": int(len(labels)), "classes": int(n_cls),
                        "class_names_all": classes if not smoke else None,
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
