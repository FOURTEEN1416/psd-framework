# -*- coding: utf-8 -*-
"""真实 tier 数据导出 — PYSKL pkl → AimCLR Feeder 格式（NTU25 槽位映射）.

目的: 协议回归（protocol-return）——真实 tier 的预注册物理层是 from-scratch
ST-GCN pretext（PSD-NTU120-PREREG-001 原文），此前因 17 关节 HRNet 拓扑被
Amendment 2 偏离为 joint-MLP。本脚本把 pkl（load_dataset/prep_clip 同源预处理，
T=30 重采样+center）导出为 AimCLR Feeder 消费的 NTU 格式:
    <out>/train_position.npy (N,3,T=30,V=25,M=1)   <out>/train_label.pkl ([names],[labels])
    <out>/val_position.npy   ...                    <out>/val_label.pkl
17→25 槽位映射: COCO-17 → 25 槽缓冲前 16 个活跃槽（躯干/颈/头由肩髋均值合成），
槽 16–24 置零。注意: 活跃槽沿用人形紧凑排列（0 髋根/1 脊中/2 颈/3 头/
4-6 左臂/7-9 右臂/10-12 左腿/13-15 右腿），与 NTU-25 标准解剖序不同——
三 tier 统一同一活跃槽约定，图拓扑固定且各臂共享同批特征，臂间比较不受影响。
PanAf500（V==24）槽位语义已核实（run_p21_panaf_pipeline.py AP17_TO_PSD24）:
PSD24 = 0 head 1/2 eye 3 withers 4 throat 5 R_shoulder 6 R_elbow 7 R_wrist
8 L_shoulder 9 L_elbow 10 L_wrist 11 R_hip 12 R_knee 13 R_ankle
14 L_hip 15 L_knee 16 L_ankle 17 tail_base 18-23 dead（AP-10K 四足 17kpt 转换，
非 COCO 序）。四足→25 槽映射 map_psd24_to_ntu25 按功能对应: 前肢→臂槽、
后肢→腿槽，躯干合成规则与 COCO 导出一致（髋根=双髋均值、脊中=四锚均值、
颈=双肩均值、头=head 槽直通）。
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
    "ntu120_full": {"pkl": REPO / "data/pyskl/ntu120_hrnet.pkl", "max_n": 24000},
    "ucf101": {"pkl": REPO / "data/pyskl/ucf101_hrnet.pkl", "max_n": 10000},
    "panaf500": {"pkl": REPO / "runs/p7_asbar/panaf500_T30.pkl", "max_n": 500},
}

def _kp3(a: dict) -> np.ndarray:
    """与 run_p5b.load_dataset 同式：压掉单人维并补 score 维，返回 (T,V,3)。"""
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
    return prep_clip(np.concatenate([kp, kps], axis=-1).astype(np.float32))


def load_balanced(pkl: Path, train_per: int = 100, val_per: int = 20) -> list:
    """NTU120 全 120 类分层取样：每类 train_per/val_per 条，避免按标注序只取到前 60 类。"""
    d = pickle.load(open(pkl, "rb"))
    anns = d["annotations"]
    split = d["split"]
    train_keys = set()
    for k in ("xsub_train", "train1"):
        train_keys.update(split.get(k, []))
    val_keys = set()
    for k in ("xsub_val", "xsub_test", "test1"):
        val_keys.update(split.get(k, []))
    tr_cnt: dict[int, int] = {}
    va_cnt: dict[int, int] = {}
    rows = []
    for a in anns:
        lab = int(a["label"])
        fd = a["frame_dir"]
        if fd in train_keys and tr_cnt.get(lab, 0) < train_per:
            rows.append({"kp": _kp3(a), "label": lab, "split": "train"})
            tr_cnt[lab] = tr_cnt.get(lab, 0) + 1
        elif fd in val_keys and va_cnt.get(lab, 0) < val_per:
            rows.append({"kp": _kp3(a), "label": lab, "split": "val"})
            va_cnt[lab] = va_cnt.get(lab, 0) + 1
    n_cls = len(set(int(a["label"]) for a in anns))
    assert len(tr_cnt) == n_cls and len(va_cnt) == n_cls, (n_cls, len(tr_cnt), len(va_cnt))
    print(f"[balanced] classes={n_cls} train={sum(tr_cnt.values())} val={sum(va_cnt.values())}")
    return rows


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


def map_psd24_to_ntu25(clip: np.ndarray) -> np.ndarray:
    """(T,24,3) PanAf PSD24 → (T,25,3)；四足功能对应映射到与 COCO 导出相同的
    活跃槽（0-15），槽 16-24 死位。PSD24 布局见模块 docstring（已核实）。"""
    T = clip.shape[0]
    out = np.zeros((T, 25, clip.shape[2]), dtype=np.float32)
    out[:, 3, :] = clip[:, 0, :]                                   # head
    out[:, 0, :] = (clip[:, 11, :] + clip[:, 14, :]) / 2.0         # spine base ← 双髋均值
    out[:, 2, :] = (clip[:, 5, :] + clip[:, 8, :]) / 2.0           # neck ← 双肩均值
    out[:, 1, :] = (clip[:, 5, :] + clip[:, 8, :] + clip[:, 11, :] + clip[:, 14, :]) / 4.0  # spine mid
    out[:, 4, :] = clip[:, 8, :]    # L front shoulder  → L arm slot
    out[:, 5, :] = clip[:, 9, :]    # L front elbow
    out[:, 6, :] = clip[:, 10, :]   # L front wrist
    out[:, 7, :] = clip[:, 5, :]    # R front shoulder → R arm slot
    out[:, 8, :] = clip[:, 6, :]    # R front elbow
    out[:, 9, :] = clip[:, 7, :]    # R front wrist
    out[:, 10, :] = clip[:, 14, :]  # L hind hip       → L leg slot
    out[:, 11, :] = clip[:, 15, :]  # L hind knee
    out[:, 12, :] = clip[:, 16, :]  # L hind ankle
    out[:, 13, :] = clip[:, 11, :]  # R hind hip       → R leg slot
    out[:, 14, :] = clip[:, 12, :]  # R hind knee
    out[:, 15, :] = clip[:, 13, :]  # R hind ankle
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--max", type=int, default=None, help="覆盖 tier 默认样本上限")
    args = ap.parse_args()
    cfg = TIERS[args.tier]
    max_n = args.max if args.max else cfg["max_n"]

    if args.tier == "ntu120_full":
        rows = load_balanced(cfg["pkl"], train_per=100, val_per=20)
        X = np.stack([r["kp"] for r in rows])
        labels = np.array([r["label"] for r in rows])
        splits = np.array([r["split"] for r in rows])
        mapper = map_to_ntu25
    elif args.tier == "panaf500":
        # list 格式 pkl（psd full12）——与 interim 读数（run_p23）同一加载器，同源同预处理
        from run_p23_panaf_retention import load_rows  # noqa: E402
        X, labels, splits, _classes = load_rows()      # X 已 prep_clip center，(N,30,24,3)
        mapper = map_psd24_to_ntu25
    else:
        rows = load_dataset(cfg["pkl"], max_n)
        X = np.stack([prep_clip(r["kp"]) for r in rows])          # (N,30,V,3)
        labels = np.array([r["label"] for r in rows])
        splits = np.array([r["split"] for r in rows])
        mapper = map_to_ntu25
    names = [f"{args.tier}_{i:06d}" for i in range(len(X))]
    print(f"[export:{args.tier}] X {X.shape} train={int((splits=='train').sum())} "
          f"val={int((splits=='val').sum())} classes={len(set(labels.tolist()))}")

    out_dir = REPO / "runs" / "p1_sprint" / f"labelfree_{args.tier}" / "feeder"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, key in (("train", "train"), ("val", "val")):
        m = splits == key
        if m.sum() == 0:
            raise SystemExit(f"[export:{args.tier}] split {key} 为空——检查 pkl split 键")
        clips = np.stack([mapper(X[i]) for i in np.where(m)[0]])  # (n,30,25,3)
        npy = np.ascontiguousarray(clips.transpose(0, 3, 1, 2)[:, :, :, :, np.newaxis])  # (n,3,30,25,1)
        np.save(out_dir / f"{key}_position.npy", npy)
        with open(out_dir / f"{key}_label.pkl", "wb") as f:
            pickle.dump(([names[i] for i in np.where(m)[0]],
                         [int(labels[i]) for i in np.where(m)[0]]), f)
        print(f"[export:{args.tier}] {key}: npy {npy.shape} → {out_dir / (key + '_position.npy')}")
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
