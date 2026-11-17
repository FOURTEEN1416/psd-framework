# -*- coding: utf-8 -*-
"""真实 tier 数据导出 — PYSKL pkl → AimCLR Feeder 格式（NTU25 槽位映射）.

目的: 协议回归（protocol-return）——真实 tier 的预注册物理层是 from-scratch
ST-GCN pretext（PSD-NTU120-PREREG-001 原文），此前因 17 关节 HRNet 拓扑被
Amendment 2 偏离为 joint-MLP。本脚本把 pkl（load_dataset/prep_clip 同源预处理，
T=30 重采样+center）导出为 AimCLR Feeder 消费的 NTU 格式:
    <out>/train_position.npy (N,3,T=30,V=25,M=1)   <out>/train_label.pkl ([names],[labels])
    <out>/val_position.npy   ...                    <out>/val_label.pkl
17→25 槽位映射（沿 PanAf500 死位映射先例）: COCO-17 → NTU-25，躯干/颈/头由
肩髋均值合成，NTU 16–24 九个槽位为死位（全零）。
V==25 输入直通；V==24（PanAf 24 槽）前 17 槽按同序映射、其余死位（跑 PanAf 前
须先核对其实际槽位语义）。
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p5b_generic_retention import load_dataset, prep_clip  # noqa: E402

TIERS = {
    "ntu120": {"pkl": REPO / "data/pyskl/ntu120_hrnet.pkl", "max_n": 12000},
    "ucf101": {"pkl": REPO / "data/pyskl/ucf101_hrnet.pkl", "max_n": 10000},
    "panaf500": {"pkl": REPO / "runs/p7_asbar/panaf500_T30.pkl", "max_n": 500},
}

# COCO-17 → NTU-25 槽位映射（-1 = 死位）
COCO17_TO_NTU25 = {
    0: 3,                # nose → head
    5: 4, 7: 5, 9: 6,    # L shoulder/elbow/wrist
    6: 7, 8: 8, 10: 9,   # R shoulder/elbow/wrist
    11: 10, 13: 11, 15: 12,  # L hip/knee/ankle
    12: 13, 14: 14, 16: 15,  # R hip/knee/ankle
}


def map_to_ntu25(clip: np.ndarray) -> np.ndarray:
    """(T,V17,3) → (T,25,3)；躯干/颈/头槽位由肩髋均值合成，其余死位全零。"""
    T, V, C = clip.shape
    out = np.zeros((T, 25, C), dtype=np.float32)
    if V >= 25:
        out[:, :, :] = clip[:, :25, :]
        return out
    if V == 24:
        # PanAf 24 槽：前 17 槽为活体关节（与 COCO 同序），8–23 死位——先按前 17 处理
        src = clip[:, :17, :]
    else:
        src = clip
    for coco_idx, ntu_idx in COCO17_TO_NTU25.items():
        out[:, ntu_idx, :] = src[:, coco_idx, :]
    out[:, 0, :] = (src[:, 11, :] + src[:, 12, :]) / 2.0          # spine base ← 双髋均值
    out[:, 1, :] = (src[:, 5, :] + src[:, 6, :] + src[:, 11, :] + src[:, 12, :]) / 4.0  # spine mid
    out[:, 2, :] = (src[:, 5, :] + src[:, 6, :]) / 2.0            # neck ← 双肩均值
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--max", type=int, default=None, help="覆盖 tier 默认样本上限")
    args = ap.parse_args()
    cfg = TIERS[args.tier]
    max_n = args.max if args.max else cfg["max_n"]

    rows = load_dataset(cfg["pkl"], max_n)
    X = np.stack([prep_clip(r["kp"]) for r in rows])          # (N,30,V,3)
    labels = np.array([r["label"] for r in rows])
    splits = np.array([r["split"] for r in rows])
    names = [f"{args.tier}_{i:06d}" for i in range(len(rows))]
    print(f"[export:{args.tier}] X {X.shape} train={int((splits=='train').sum())} "
          f"val={int((splits=='val').sum())} classes={len(set(labels.tolist()))}")

    out_dir = REPO / "runs" / "p1_sprint" / f"labelfree_{args.tier}" / "feeder"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, key in (("train", "train"), ("val", "val")):
        m = splits == key
        if m.sum() == 0:
            raise SystemExit(f"[export:{args.tier}] split {key} 为空——检查 pkl split 键")
        clips = np.stack([map_to_ntu25(X[i]) for i in np.where(m)[0]])  # (n,30,25,3)
        npy = np.ascontiguousarray(clips.transpose(0, 3, 1, 2)[:, :, :, :, np.newaxis])  # (n,3,30,25,1)
        np.save(out_dir / f"{key}_position.npy", npy)
        with open(out_dir / f"{key}_label.pkl", "wb") as f:
            pickle.dump(([names[i] for i in np.where(m)[0]],
                         [int(labels[i]) for i in np.where(m)[0]]), f)
        print(f"[export:{args.tier}] {key}: npy {npy.shape} → {out_dir / (key + '_position.npy')}")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
