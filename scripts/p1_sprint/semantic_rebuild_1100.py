"""P1 补强 E2 — 对照预算 1,100 clips 下的语义重建臂（TRANS-002 三臂 × 等预算对齐）.

动机（冷审补强清单 E2）: 论文把「合成层三臂对照（50 clips/class × 22 = 1,100 clips，
frozen_backbone_contrast）」与「E6 n=30/100」跨预算拼接，存在误读风险。本脚本在
**同一 1,100-clip 对照预算**下补测真实域（NTU60 60→49 ten-merge，TRANS-002 冻结协议）
的三臂，闭合跨预算拼接缺口：

  C′ coupled   : net.st_gcn.Model 端到端有监督从零训练（pyskl vanilla 配方，A-fix
                 lr=0.00625，与 full/fixed 报告同款），在 1,100-clip 子集上重训于 Y′；
  D′ decoupled : 冻结 AimCLR joint pretext(epoch300) penultimate + StandardScaler+LR
                 头（语义重建臂），头在 1,100-clip 子集上重建；
  M matched    : 冻结 C′ checkpoint penultimate + 同一 StandardScaler+LR 头。

复用声明（不重复造轮子）: 驱动逻辑全部复用 scripts/run_r23_trans002.py 的
run_supervised / eval_49 / train_lr_head / eval_lr / build_merge_map（import，不改其
内部）；差异仅有预算切片（1,100 clips 分层抽取，seed 固定 42 作预算抽取种子并披露，
臂种子独立）与产出目录。评估统一在全部 16,487 条 val 上进行。

用法:
    .venv/Scripts/python.exe scripts/p1_sprint/semantic_rebuild_1100.py --smoke
    .venv/Scripts/python.exe scripts/p1_sprint/semantic_rebuild_1100.py --seeds 42
    .venv/Scripts/python.exe scripts/p1_sprint/semantic_rebuild_1100.py --seeds 42,43,44
产出:
    runs/p1_sprint/semantic_rebuild_1100/summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_r23_trans002 import (  # noqa: E402  复用 TRANS-002 冻结协议实现
    DATA,
    PRETEXT_CKPT,
    build_merge_map,
    eval_lr,
    eval_49,
    run_supervised,
    train_lr_head,
)
from run_r22_ntu_transition import NTU60_CLASSES  # noqa: E402

OUT_DIR = REPO_ROOT / "runs" / "p1_sprint" / "semantic_rebuild_1100"
WORK = REPO_ROOT / "runs" / "p1_sprint" / "semantic_rebuild_1100" / "work"
VAL_PRETEXT_NPZ = REPO_ROOT / "runs" / "trans002" / "val_feat_penult_pretext.npz"
BUDGET_CLIPS = 1100          # = 合成层三臂对照预算（50/class × 22）
BUDGET_DRAW_SEED = 42        # 预算抽取种子（固定并披露；臂种子独立）


# ---------------------------------------------------------------------------
# 1,100-clip 分层预算切片（一次构造、落盘、复用）
# ---------------------------------------------------------------------------

def build_budget_subset() -> tuple[Path, Path, dict]:
    """从 40,091 条 train 中分层抽取恰好 1,100 条（60 类按类内比例分配、每类≥1）。

    产出 data/ntu60_frame50/xsub/ 下:
      train_position_yp49_1100.npy / train_label_yp49_1100.pkl / budget_1100_meta.json
    已存在且 sha 一致则复用（幂等）。
    """
    meta_path = DATA / "budget_1100_meta.json"
    npy_path = DATA / "train_position_yp49_1100.npy"
    pkl_path = DATA / "train_label_yp49_1100.pkl"

    with open(DATA / "train_label.pkl", "rb") as f:
        _, tr_label60 = pickle.load(f)
    tr_label60 = np.asarray(tr_label60, dtype=np.int64)

    rng = np.random.default_rng(BUDGET_DRAW_SEED)
    counts = np.bincount(tr_label60, minlength=60)
    # 比例分配 + 最大余数法凑齐 1100，每类至少 1
    raw = counts / counts.sum() * BUDGET_CLIPS
    alloc = np.maximum(1, np.floor(raw).astype(np.int64))
    rem = BUDGET_CLIPS - alloc.sum()
    order = np.argsort(-(raw - np.floor(raw)))
    for i in range(rem):
        alloc[order[i % 60]] += 1

    picked = []
    for c in range(60):
        ci = np.where(tr_label60 == c)[0]
        picked.append(rng.choice(ci, size=min(alloc[c], len(ci)), replace=False))
    idx = np.sort(np.concatenate(picked))
    idx_digest = hashlib.sha256(idx.tobytes()).hexdigest()[:16]

    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("indices_sha256_16") == idx_digest and npy_path.exists() and pkl_path.exists():
            print(f"[budget] 复用既有 1,100-clip 切片 (sha={idx_digest})")
            return npy_path, pkl_path, old

    y2p = build_merge_map(list(NTU60_CLASSES))
    full = np.load(DATA / "train_position.npy", mmap_mode="r")
    np.save(npy_path, np.ascontiguousarray(full[idx]))
    del full
    sub_lab60 = tr_label60[idx]
    sub_lab49 = [int(y2p[int(v)]) for v in sub_lab60]
    names = [f"train1100_{i:06d}" for i in range(len(idx))]
    with open(pkl_path, "wb") as f:
        pickle.dump((names, sub_lab49), f)
    meta = {
        "budget_clips": int(len(idx)),
        "draw_seed": BUDGET_DRAW_SEED,
        "indices_sha256_16": idx_digest,
        "per_class_min": int(alloc.min()), "per_class_max": int(alloc.max()),
        "labels": "Y' 49-class (TRANS-002 frozen merge)",
        "note": "预算抽取种子固定 42（披露）；臂种子独立于本切片",
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[budget] 1,100-clip 分层切片落盘: per-class [{alloc.min()},{alloc.max()}] "
          f"sha={idx_digest}")
    return npy_path, pkl_path, meta


# ---------------------------------------------------------------------------
# penultimate 特征导出（参数化路径版；逻辑同 run_r23_trans002.dump_penultimate）
# ---------------------------------------------------------------------------

def dump_penult(ckpt: Path, data_npy: Path, label_pkl: Path, out_npz: Path) -> None:
    import sys as _sys
    aimclr_root = REPO_ROOT / "external" / "AimCLR"
    _sys.path.insert(0, str(aimclr_root))
    _sys.path.insert(0, str(aimclr_root / "torchlight"))
    from feeder.ntu_feeder import Feeder_single  # noqa: E402

    feats_b = {}

    def pre_hook(_m, args):
        feats_b["o"] = args[0].detach()

    is_pretext = "pretext" in ckpt.name or "joint_pretext" in str(ckpt)
    if is_pretext:
        from net.aimclr import AimCLR  # noqa: E402
        model = AimCLR(base_encoder="net.st_gcn.Model", pretrain=False, in_channels=3,
                       hidden_channels=16, hidden_dim=256, num_class=49, dropout=0.5,
                       graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                       edge_importance_weighting=True)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        ck = ck.get("model", ck)
        ck = {k: v for k, v in ck.items()
              if not k.startswith("encoder_k.") and k not in ("queue", "queue_ptr")}
        if any(k.startswith("encoder_q.fc.2.") for k in ck):
            import torch.nn as nn
            model.encoder_q.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(),
                                               nn.Linear(256, 49))
            ck = {k: v for k, v in ck.items() if not k.startswith("encoder_q.fc.")}
        missing, unexpected = model.load_state_dict(ck, strict=False)
        missing = [m for m in missing if not m.startswith("encoder_q.fc.")]
        assert not missing and not unexpected, (missing[:4], unexpected[:4])
        encoder = model.encoder_q
        h = encoder.fc.register_forward_pre_hook(pre_hook)
    else:
        from net.st_gcn import Model as STGCNModel  # noqa: E402
        model = STGCNModel(in_channels=3, hidden_channels=16, hidden_dim=256,
                           num_class=49, dropout=0.5,
                           graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                           edge_importance_weighting=True)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck)
        if any(k.startswith("inner.") for k in sd):
            sd = {k[len("inner."):]: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=True)
        encoder = model
        h = model.fc.register_forward_pre_hook(pre_hook)

    model.eval().cuda()
    feeder = Feeder_single(str(data_npy), str(label_pkl), shear_amplitude=-1,
                           temperal_padding_ratio=-1, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=256, shuffle=False,
                                         num_workers=0)
    feats, labs = [], []
    with torch.no_grad():
        for x, lab in loader:
            _ = encoder(x.float().cuda())
            o = feats_b.get("o")
            assert o is not None, "pre_hook did not fire"
            feats.append(o.cpu().numpy())
            labs.append(np.asarray(lab))
    h.remove()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, feat=np.concatenate(feats),
                        label=np.concatenate(labs))
    print(f"[dump] {out_npz.name}: {feats[0].shape[1]}d × {sum(len(f) for f in feats)}",
          flush=True)


def write_cfg_1100(seed: int, epochs: int, work_dir: Path,
                   train_npy: Path, train_pkl: Path) -> Path:
    """C′ 臂配置——与 run_r23_trans002.write_cfg 同配方，仅数据路径指向 1,100 切片。

    编码注意: FT_Processor 用 open(config) 按本地 locale 解码 cfg；本仓绝对路径含
    中文（PR论文），UTF-8 写出会被 GBK 误读成乱码 work_dir（实证: checkpoint 落进
    D:/Desktop/PR璁烘枃/...）。故 cfg 内一律用**纯 ASCII 相对路径**（run_supervised
    已 os.chdir(REPO)，相对路径正确解析），从根上消除编码隐患。
    """
    lr = round(0.1 * 16 / 128 / 2, 5)  # 官方线性缩放再减半（A-fix，同 full/fixed 报告）
    work_dir_rel = work_dir.relative_to(REPO_ROOT).as_posix()
    train_npy_rel = train_npy.relative_to(REPO_ROOT).as_posix()
    train_pkl_rel = train_pkl.relative_to(REPO_ROOT).as_posix()
    assert all(c.isascii() for c in work_dir_rel + train_npy_rel + train_pkl_rel), \
        "cfg 路径必须为纯 ASCII（FT locale 解码安全）"
    cfg = f"""# TRANS-002 arm C' @ 1100-clip control budget (E2; frozen protocol recipe)
work_dir: {work_dir_rel}

train_feeder: feeder.ntu_feeder.Feeder_single
train_feeder_args:
  data_path: {train_npy_rel}
  label_path: {train_pkl_rel}
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True

test_feeder: feeder.ntu_feeder.Feeder_single
test_feeder_args:
  data_path: data/ntu60_frame50/xsub/val_position.npy
  label_path: data/ntu60_frame50/xsub/val_label_yp49.pkl
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True

model: net.st_gcn.Model
model_args:
  in_channels: 3
  hidden_channels: 16
  hidden_dim: 256
  num_class: 49
  dropout: 0.5
  graph_args:
    layout: 'ntu-rgb+d'
    strategy: 'spatial'
  edge_importance_weighting: True

nesterov: True
weight_decay: 0.0001
base_lr: {lr}
optimizer: SGD
device: [0]
batch_size: 16
test_batch_size: 64
num_epoch: {epochs}
num_worker: 0
save_interval: -1
eval_interval: {epochs}
print_log: False
"""
    work_dir.mkdir(parents=True, exist_ok=True)
    p = work_dir / f"cfg_c_1100_s{seed}.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


def load_comparators() -> dict:
    """预算阶梯参照（既有报告数字，未重跑）。"""
    out = {}
    for key, fname in (("full_40091", "r23-trans002-full-fixed-2026-09-08.json"),
                       ("tenpct_4009", "r23-trans002-full10-2026-09-08.json")):
        p = REPO_ROOT / "reports" / fname
        if not p.exists():
            out[key] = None
            continue
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            row = {}
            if key == "full_40091":
                c = [r["val_acc"] for r in j.get("C_prime", [])]
                d = [r["val_acc"] for r in j.get("D_prime_reused", [])]
                m = [r["val_acc"] for r in j.get("M_matched", [])]
                row = {"C_prime": round(float(np.mean(c)), 4) if c else None,
                       "D_prime": round(float(np.mean(d)), 4) if d else None,
                       "M_matched": round(float(np.mean(m)), 4) if m else None}
            else:
                runs = j.get("runs", [])
                for arm in ("C_prime", "D_prime", "M_matched"):
                    accs = [r["val_acc"] for r in runs if r.get("arm") == arm]
                    row[arm] = round(float(np.mean(accs)), 4) if accs else None
            out[key] = {"source": fname, "val_acc": row}
        except Exception as exc:  # noqa: BLE001
            out[key] = {"error": str(exc)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true", help="2 epochs 冒烟（管线验证，数字无意义）")
    ap.add_argument("--seeds", default="42", help="逗号分隔臂种子，如 42 或 42,43,44")
    ap.add_argument("--epochs", type=int, default=80, help="C′ 训练轮数（协议 80）")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    epochs = 2 if args.smoke else args.epochs
    if args.smoke:
        print("[smoke] 冒烟模式: epochs=2（管线验证，数字无意义）")

    train_npy, train_pkl, budget_meta = build_budget_subset()
    results = {
        "experiment": "semantic_rebuild_1100_trans002_three_arm",
        "mode": "smoke" if args.smoke else "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-NTU-TRANS-002 recipes @ 1,100-clip control budget (E2)",
        "budget_clips": BUDGET_CLIPS,
        "budget_meta": budget_meta,
        "epochs": epochs,
        "seeds": seeds,
        "val_size": 16487,
        "runs": [],
    }

    # D′ 的 val 特征：预算无关，直接复用 TRANS-002 既有产物（缺失才补导出）
    val_npz_d = VAL_PRETEXT_NPZ
    if not val_npz_d.exists():
        val_npz_d = WORK / "val_feat_penult_pretext.npz"
        dump_penult(PRETEXT_CKPT, DATA / "val_position.npy",
                    DATA / "val_label_yp49.pkl", val_npz_d)
    else:
        print(f"[reuse] D′ val 特征复用既有 {val_npz_d.relative_to(REPO_ROOT)}")

    for seed in seeds:
        # ---------- C′ coupled: 1,100-clip 子集上从零重训 ----------
        run_tag = f"c_1100_s{seed}_{datetime.now().strftime('%H%M%S')}"
        cfg = write_cfg_1100(seed, epochs, WORK / run_tag, train_npy, train_pkl)
        t = time.time()
        acc_c = run_supervised(cfg, seed)
        wall_c = round(time.time() - t, 1)
        ck_c = cfg.parent / "best_model.pt"
        print(f"[C'] 1100 s{seed}: acc={acc_c:.4f} wall={wall_c}s", flush=True)

        # ---------- D′ decoupled: 冻结 pretext 特征 + LR 头（1,100 子集重建） ----------
        tr_npz_d = WORK / f"train_feat_1100_d_s{seed}.npz"
        if not tr_npz_d.exists():
            dump_penult(PRETEXT_CKPT, train_npy, train_pkl, tr_npz_d)
        t = time.time()
        clf_d, sc_d = train_lr_head(tr_npz_d)
        acc_d = eval_lr(clf_d, sc_d, val_npz_d)
        wall_d = round(time.time() - t, 1)

        # ---------- M matched: 冻结 C′ 特征 + 同一 LR 头 ----------
        tr_npz_m = WORK / f"train_feat_1100_m_s{seed}.npz"
        val_npz_m = WORK / f"val_feat_1100_m_s{seed}.npz"
        if not (tr_npz_m.exists() and val_npz_m.exists()):
            dump_penult(ck_c, train_npy, train_pkl, tr_npz_m)
            dump_penult(ck_c, DATA / "val_position.npy",
                        DATA / "val_label_yp49.pkl", val_npz_m)
        t = time.time()
        clf_m, sc_m = train_lr_head(tr_npz_m)
        acc_m = eval_lr(clf_m, sc_m, val_npz_m)
        wall_m = round(time.time() - t, 1)
        print(f"[D'] acc={acc_d:.4f} wall={wall_d}s | [M] acc={acc_m:.4f} wall={wall_m}s",
              flush=True)

        results["runs"] += [
            {"arm": "C_prime", "budget": "1100", "seed": seed,
             "wall_clock_sec": wall_c, "val_acc": round(acc_c, 4)},
            {"arm": "D_prime", "budget": "1100", "seed": seed,
             "wall_clock_sec": wall_d, "val_acc": round(acc_d, 4)},
            {"arm": "M_matched", "budget": "1100", "seed": seed,
             "wall_clock_sec": wall_m, "val_acc": round(acc_m, 4)},
        ]

    # ---------- 汇总（与 finalize_trans002 同口径的 verdict 摘要） ----------
    import statistics as st
    summary = {}
    for arm in ("C_prime", "D_prime", "M_matched"):
        rows = [r for r in results["runs"] if r["arm"] == arm]
        if rows:
            summary[arm] = {"seeds": [r["seed"] for r in rows],
                            "acc_mean": round(float(np.mean([r["val_acc"] for r in rows])), 4),
                            "wall_median_s": st.median([r["wall_clock_sec"] for r in rows])}
    if "C_prime" in summary and "D_prime" in summary:
        ratio = summary["C_prime"]["wall_median_s"] / max(summary["D_prime"]["wall_median_s"], 1e-9)
        summary["cost_ratio_C_over_D"] = round(ratio, 2)
        summary["acc_gap_D_minus_C_pp"] = round(
            (summary["D_prime"]["acc_mean"] - summary["C_prime"]["acc_mean"]) * 100, 2)
    results["summary"] = summary
    results["comparators_other_budgets"] = load_comparators()
    results["protocol_notes"] = [
        "预算 1,100 clips = 合成层三臂对照 50/class×22 的等预算对齐点（E2 动机）",
        "C′/D′/M 配方与 TRANS-002 冻结协议一致（A-fix lr=0.00625, batch16, SGD）",
        "预算抽取种子固定 42 并落盘 meta；臂种子独立",
        "评估统一在全部 16,487 val clips；D′ val 特征复用 TRANS-002 既有 pretext 导出",
        "comparators_other_budgets 为既有报告数字（full 40,091 / 10% 4,009），未重跑",
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke.json" if args.smoke else ".json"
    out_path = OUT_DIR / f"summary{suffix}"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "mode": results["mode"]},
                     ensure_ascii=False, indent=2))
    print(f"[done] {out_path}")
    print("SEMANTIC_REBUILD_1100_DONE")


if __name__ == "__main__":
    main()
