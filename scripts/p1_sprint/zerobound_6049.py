"""P1 补强 E1 — 60→49 ten-merge episode 的零重训上界（纯推理，双口径）.

动机（冷审补强清单 E1）: 归因对照表同时给出 76.8%@(60→10 rename-oracle) 与
75.3%@(60→49 decoupled rebuild)，两行 episode 不同不可比。本脚本在 **同一个
60→49 ten-merge episode（TRANS-002 冻结合并表）** 上补测零重训上界，消除口径缺口：

  上界（rename-oracle）: 源 60 类头的 softmax 后验按冻结合并表求和 → 目标 49 类
      argmax。零重训、映射查表零误差假设下的可达天花板。
  下界（argmax-vote）: 同一映射表，源 argmax 硬投票。分数求和 vs 硬投票之差
      = 校准损失。

全部复用既有资产（零新模型）:
  - 冻结 pretext 特征: runs/ntu_lowres/features_joint_ep300.npz（256d，
    run_p14_ntu_featuredump.py 产物，与 E9/TRANS-002 D′ 臂同一特征库）
  - 源 60 类头: run_p14_ntu_lowres.ScaledLR（StandardScaler+LR，p14 协议原样）
  - 60→49 冻结映射: runs/bmin_trans_yprime_map.json（TRANS-001/002 冻结产物），
    加载时与 run_r22_ntu_transition.build_y_to_yp_map_ntu() 交叉校验（不一致即失败）。
  - 同 episode 参照（直接从既有报告读取，不重跑）:
      reports/r23-trans002-full-fixed-2026-09-08.json  → coupled C′ / decoupled D′ / matched M
      runs/p1_sprint/merge_rename_upper/summary.json   → 60→10 coarse-grouping 双口径

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/zerobound_6049.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/zerobound_6049.py --seed 42
产出:
    runs/p1_sprint/zerobound_6049/summary.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import numpy as np  # noqa: E402

from run_p14_ntu_lowres import (  # noqa: E402
    NPZ,
    train_linear_head_scaled,
)

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "zerobound_6049"
MAP_JSON = REPO_ROOT / "runs" / "bmin_trans_yprime_map.json"
N_SRC = 60


def load_frozen_mapping() -> tuple[np.ndarray, list[str]]:
    """读冻结 60→49 映射并与 run_r22 权威实现交叉校验。"""
    spec = importlib.util.spec_from_file_location(
        "r22t", str(REPO_ROOT / "scripts" / "run_r22_ntu_transition.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["r22t"] = mod
    spec.loader.exec_module(mod)
    y2p_authoritative, names49_authoritative = mod.build_y_to_yp_map_ntu()

    blob = json.loads(MAP_JSON.read_text(encoding="utf-8"))
    y2p = np.full(N_SRC, -1, dtype=np.int64)
    for k, v in blob["y2p"].items():
        y2p[int(k)] = int(v)
    names49 = list(blob["names49"])

    assert (y2p >= 0).all(), "冻结映射未覆盖全部 60 源类"
    assert sorted(set(y2p.tolist())) == list(range(49)), "目标类编号应为 0..48 连续"
    assert len(names49) == 49
    # 交叉校验: JSON 冻结产物 == 代码权威实现（防漂移）
    auth = np.full(N_SRC, -1, dtype=np.int64)
    for k, v in y2p_authoritative.items():
        auth[int(k)] = int(v)
    if not np.array_equal(y2p, auth) or names49 != list(names49_authoritative):
        raise SystemExit("冻结映射与 run_r22_ntu_transition.build_y_to_yp_map_ntu 不一致——映射漂移，终止")
    print("[map] 60->49 冻结映射: JSON 与 run_r22 权威实现一致; "
          f"合并源类 {int((np.bincount(y2p) > 1).sum())} 组 -> 49 目标类")
    return y2p, names49


def load_comparators() -> dict:
    """读取同 episode 既有实验数字作参照（缺失则记 null，不编造）。"""
    out = {"trans002_full_fixed": None, "merge_rename_60to10": None}
    p = REPO_ROOT / "reports" / "r23-trans002-full-fixed-2026-09-08.json"
    if p.exists():
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            c = [r["val_acc"] for r in j.get("C_prime", [])]
            d = [r["val_acc"] for r in j.get("D_prime_reused", [])]
            m = [r["val_acc"] for r in j.get("M_matched", [])]
            out["trans002_full_fixed"] = {
                "source": str(p.relative_to(REPO_ROOT)),
                "budget": "full(40091)", "episode": "60->49 ten-merge",
                "coupled_C_prime_acc": round(float(np.mean(c)), 4) if c else None,
                "decoupled_D_prime_acc": round(float(np.mean(d)), 4) if d else None,
                "matched_M_acc": round(float(np.mean(m)), 4) if m else None,
            }
        except Exception as exc:  # noqa: BLE001
            out["trans002_full_fixed"] = {"error": str(exc)}
    p2 = REPO_ROOT / "runs" / "p1_sprint" / "merge_rename_upper" / "summary.json"
    if p2.exists():
        try:
            j2 = json.loads(p2.read_text(encoding="utf-8"))
            out["merge_rename_60to10"] = {
                "source": str(p2.relative_to(REPO_ROOT)),
                "episode": "60->10 coarse grouping",
                "rename_oracle_upper_top1": j2.get("rename_oracle_upper_top1"),
                "argmax_vote_lower_top1": j2.get("argmax_vote_lower_top1"),
                "target_fullsupervision_ref_top1": j2.get("target_fullsupervision_ref_top1"),
            }
        except Exception as exc:  # noqa: BLE001
            out["merge_rename_60to10"] = {"error": str(exc)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="子采样快速验证（数字无意义）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-frac", type=float, default=0.10,
                    help="源 60 类头的标注预算（与 E9/TRANS-002 10%% 档对齐）")
    ap.add_argument("--smoke-n", type=int, default=1200, help="smoke 每 split 子采样数")
    args = ap.parse_args()

    t0 = time.time()
    y2p, target_names = load_frozen_mapping()
    n_target = len(target_names)

    z = np.load(NPZ)
    tr_feat, tr_lab = z["train_feat"], z["train_label"]
    va_feat, va_lab = z["val_feat"], z["val_label"]

    if args.smoke:
        rng = np.random.default_rng(args.seed)
        tri = rng.choice(len(tr_feat), size=min(args.smoke_n, len(tr_feat)), replace=False)
        vai = rng.choice(len(va_feat), size=min(args.smoke_n // 3, len(va_feat)), replace=False)
        tr_feat, tr_lab = tr_feat[tri], tr_lab[tri]
        va_feat, va_lab = va_feat[vai], va_lab[vai]
        print(f"[smoke] 子采样 train={len(tr_feat)} val={len(va_feat)}（数字无意义）")

    # ---------- 1) 源 60 类头（参数化分层预算，p14 同款 ScaledLR） ----------
    rng = np.random.default_rng(args.seed)
    mask = np.zeros(len(tr_lab), dtype=bool)
    for c in np.unique(tr_lab):
        ci = np.where(tr_lab == c)[0]
        k = max(1, int(round(len(ci) * args.budget_frac)))
        mask[rng.choice(ci, size=k, replace=False)] = True
    print(f"[budget] {args.budget_frac:.0%} subset: {int(mask.sum())} clips")

    src_head = train_linear_head_scaled(tr_feat, tr_lab, mask, N_SRC)
    if src_head is None:
        raise SystemExit("源头训练失败（类别覆盖不足）")

    src_proba_val = src_head.clf.predict_proba(src_head.sc.transform(va_feat))
    src_classes = src_head.clf.classes_
    assert len(src_classes) == N_SRC, f"源头只覆盖 {len(src_classes)}/60 类"
    src_proba_val = src_proba_val[:, np.argsort(src_classes)]  # 列重排到 0..59

    src_pred_val = src_head.predict(va_feat)
    src_top1 = float(np.mean(src_pred_val == va_lab))
    print(f"[src-head] 源 60 类 val top1={src_top1:.4f}")

    # ---------- 2) 目标标签 ----------
    tgt_tr_lab = y2p[tr_lab]
    tgt_va_lab = y2p[va_lab]

    # ---------- 3) rename-oracle 上界: 源后验按冻结映射求和 ----------
    tgt_proba = np.zeros((len(va_feat), n_target), dtype=np.float64)
    for si in range(N_SRC):
        tgt_proba[:, y2p[si]] += src_proba_val[:, si]
    upper_pred = tgt_proba.argmax(axis=1)
    upper_top1 = float(np.mean(upper_pred == tgt_va_lab))

    # ---------- 4) argmax-vote 下界: 源 argmax 硬投票 ----------
    vote_mat = np.zeros((len(va_feat), n_target), dtype=np.int64)
    for si in range(N_SRC):
        vote_mat[:, y2p[si]] += (src_pred_val == si).astype(np.int64)
    lower_pred = vote_mat.argmax(axis=1)
    lower_top1 = float(np.mean(lower_pred == tgt_va_lab))

    # ---------- 5) 参照: 目标 49 类全监督线性头（同一批标注） ----------
    tgt_head = train_linear_head_scaled(tr_feat, tgt_tr_lab, mask, n_target)
    tgt_ref_top1 = float(np.mean(tgt_head.predict(va_feat) == tgt_va_lab))

    counts = np.bincount(tgt_va_lab, minlength=n_target).astype(np.float64)
    random_top1 = float((counts / counts.sum() ** 2).sum())

    out = {
        "experiment": "zerobound_6049_ten_merge_episode",
        "mode": "smoke" if args.smoke else "full",
        "seed": args.seed,
        "budget_frac": args.budget_frac,
        "n_train": int(len(tr_feat)), "n_val": int(len(va_feat)),
        "n_train_labeled_used": int(mask.sum()),
        "source_classes": N_SRC, "target_classes": n_target,
        "episode": "TRANS-002 frozen 60->49 ten-merge (runs/bmin_trans_yprime_map.json)",
        "source_60class_val_top1": round(src_top1, 4),
        "rename_oracle_upper_top1": round(upper_top1, 4),
        "argmax_vote_lower_top1": round(lower_top1, 4),
        "target_fullsupervision_ref_top1": round(tgt_ref_top1, 4),
        "retention_upper_vs_ref": round(upper_top1 / max(tgt_ref_top1, 1e-9), 4),
        "random_baseline_top1": round(random_top1, 4),
        "comparators_same_episode": load_comparators(),
        "protocol_notes": [
            "零重训上界=rename-oracle: 源 60 类头 softmax 后验按冻结 60->49 映射求和",
            "零重训下界=argmax-vote: 源 argmax 硬投票（与上界之差=校准损失）",
            "参照=同一批标注在 49 类目标 taxonomy 上的全监督线性头",
            "特征: runs/ntu_lowres/features_joint_ep300.npz (P1.4a 冻结 pretext 256d)",
            "分类器: run_p14 ScaledLR (StandardScaler+LR max_iter=1000) 原样复用",
            "comparators 为既有报告数字（TRANS-002 full-fixed / merge_rename_upper），未重跑",
        ],
        "duration_sec": round(time.time() - t0, 1),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, (list, dict))},
                     ensure_ascii=False, indent=2))
    print(f"[done] {out_path}")
    print("ZEROBOUND_6049_DONE")


if __name__ == "__main__":
    main()
