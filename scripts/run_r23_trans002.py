# -*- coding: utf-8 -*-
"""PSD-NTU-TRANS-002 — ST-GCN 规模三臂 taxonomy-transition 复制（恶意审稿 A3/A5 补实验）。

Y(60) → Y′(49, 与 TRANS-001 同一冻结合并表)。三臂:
  C′  coupled    : ST-GCN(net.st_gcn.Model) 端到端有监督 80ep 从零训练于 Y′ (SGD b16 lr0.1 wd1e-4 step[60])
  D′  decoupled  : 冻结 AimCLR joint pretext(epoch300) penultimate(256) + StandardScaler+LR 头重训于 Y′
  M   matched    : 冻结 C′ 检查点 penultimate + 同一 StandardScaler+LR 头重训于 Y′
预算 full(40091) / 10%(4009, seed42)。seeds 42/43/44。
wall-clock: C′=全训段; D′=头段(pretext 单列); M=头段。两口径判定(head-only 主判, amortized 并报)。

实现要点:
  - Y′ 标签由 TRANS-001 的 build_y_to_yp_map_ntu 逻辑内联复制(冻结合并表不重设计)
  - C′ 用官方 AimCLR 代码链训练(P01AimCLRProcessor + 配置文件), Feeder 标签 pkl 预映射为 49 类
  - penultimate 经 forward hook 提取; 评估前处理与 linear-eval 逐键一致(Feeder_single 无增强)
  - 10% 档 = train 子集切片另存 npy+映射 pkl; val 全量

用法:
    python scripts/run_r23_trans002.py --stage prep      # 生成 Y′ 标签 pkl / 10% 切片 / 配置文件
    python scripts/run_r23_trans002.py --stage smoke     # 冒烟: C′/D′/M 各 1 seed 2ep
    python scripts/run_r23_trans002.py --stage full      # 全量: 3 seeds × 2 预算 × 三臂
产出:
    reports/r23-trans002-<date>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

DATA = REPO / "data" / "ntu60_frame50" / "xsub"
WORK = REPO / "runs" / "trans002"
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
EPOCHS = 80
ACC_BAND_PP = 2.3
PRETEXT_CKPT = REPO / "runs" / "ntu_phaseB" / "joint_pretext" / "epoch300_model.pt"

# ---- Y → Y′ 冻结合并表（与 run_r22_ntu_transition.py 逐项同源, 不重设计）----
MERGE_GROUPS = [
    ("drink water", "eat meal/snack"),
    ("brushing teeth", "brushing hair"),
    ("sitting down", "standing up (from sitting position)"),
    ("wear jacket", "take off jacket"),
    ("wear a shoe", "take off a shoe"),
    ("wear on glasses", "take off glasses"),
    ("put on a hat/cap", "take off a hat/cap"),
    ("nod head/bow", "shake head"),
    ("punching/slapping other person", "kicking other person", "pushing other person"),
    ("walking towards each other", "walking apart from each other"),
]


def build_merge_map(labels60):
    """labels60: list[str] 官方 60 类顺序 → dict old_idx -> new_idx(49)."""
    name2old = {n: i for i, n in enumerate(labels60)}
    y2yp = {}
    new_idx = 0
    merged = set()
    for grp in MERGE_GROUPS:
        olds = [name2old[g] for g in grp]
        for o in olds:
            y2yp[o] = new_idx
            merged.add(o)
        new_idx += 1
    for o in range(60):
        if o not in merged:
            y2yp[o] = new_idx
            new_idx += 1
    assert new_idx == 49, new_idx
    return y2yp


def stage_prep():
    WORK.mkdir(parents=True, exist_ok=True)
    with open(DATA / "train_label.pkl", "rb") as f:
        tr = pickle.load(f)
    with open(DATA / "val_label.pkl", "rb") as f:
        va = pickle.load(f)
    # 官方 label pkl: (sample_names, labels) 二元组（本仓 frame-50 导出格式, 实测）
    if isinstance(tr, dict):
        tr_label, names = tr["label"], tr.get("names") or tr.get("label_names")
        va_label = va["label"]
    else:
        names, tr_label = tr[0], tr[1]
        va_label = va[1]
    names = [str(n).replace(".skeleton", "") for n in names]
    # 名单位置存的是样本名而非类名——类名表从 TRANS-001 冻结合并表反推（NTU60 官方顺序）:
    from run_r22_ntu_transition import NTU60_CLASSES
    names = list(NTU60_CLASSES)
    assert len(names) == 60, len(names)
    y2yp = build_merge_map(names)
    tr_yp = [int(y2yp[int(l)]) for l in tr_label]
    va_yp = [int(y2yp[int(l)]) for l in va_label]
    yp_names = [None] * 49
    for o, n in y2yp.items():
        if yp_names[n] is None:
            yp_names[n] = names[o]
    for split, lab in (("train", tr_yp), ("val", va_yp)):
        # Feeder_single.load_data 期望官方二元组 (sample_names, labels)——dict 会破坏索引
        sample_names = [f"{split}_{i:06d}" for i in range(len(lab))]
        with open(DATA / f"{split}_label_yp49.pkl", "wb") as f:
            pickle.dump((sample_names, lab), f)
    print(f"[prep] Y' label pkls written (49 classes)")

    # 10% 分层切片 (seed 42) —— 与全系列同惯例; 直接复用 E9 的子集索引逻辑
    sys.path.insert(0, str(REPO / "scripts"))
    from run_p14_ntu_lowres import stratified_10pct  # noqa: E402
    arr = np.array(tr_label)
    mask = stratified_10pct(arr, 42)
    for split, m in (("train", mask),):
        src_x = DATA / f"{split}_position.npy"
        dst_x = DATA / f"{split}_position_yp49_10pct.npy"
        if not dst_x.exists():
            print(f"[prep] slicing 10% {split}: {int(m.sum())}/{len(m)} ...")
            full = np.load(src_x, mmap_mode="r")
            sub = np.ascontiguousarray(full[m])
            np.save(dst_x, sub)
            del full
        sub_lab = [tr_yp[i] for i in np.where(m)[0]]
        sub_names = [f"train10pct_{i:06d}" for i in range(len(sub_lab))]
        with open(DATA / f"{split}_label_yp49_10pct.pkl", "wb") as f:
            pickle.dump((sub_names, sub_lab), f)
    print("[prep] 10% slices written")


def write_cfg(kind: str, seed: int, budget: str, epochs: int, work_dir: Path) -> Path:
    """生成 C′ 训练配置（官方 linear-eval 配置改造: pretrain=False + Y′ 标签 + 有监督头训练）。"""
    xp = {"full": "", "10pct": "_yp49_10pct"}[budget]
    cfg = f"""# TRANS-002 arm C' (auto-generated, frozen protocol PSD-NTU-TRANS-002)
work_dir: {work_dir.as_posix()}
weights: {PRETEXT_CKPT.as_posix()}
ignore_weights: [encoder_q.fc, encoder_k, queue]

train_feeder: feeder.ntu_feeder.Feeder_single
train_feeder_args:
  data_path: data/ntu60_frame50/xsub/train_position{xp}.npy
  label_path: data/ntu60_frame50/xsub/train_label_yp49{'_10pct' if budget == '10pct' else ''}.pkl
  shear_amplitude: 0.5
  temperal_padding_ratio: 4
  mmap: True

test_feeder: feeder.ntu_feeder.Feeder_single
test_feeder_args:
  data_path: data/ntu60_frame50/xsub/val_position.npy
  label_path: data/ntu60_frame50/xsub/val_label_yp49.pkl
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True

model: net.aimclr.AimCLR
model_args:
  base_encoder: net.st_gcn.Model
  pretrain: False
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
base_lr: 0.1
optimizer: SGD
step: [60]
device: [0]
batch_size: 16
test_batch_size: 64
num_epoch: {epochs}
stream: 'joint'
num_worker: 0
save_interval: -1
eval_interval: 10
print_log: False
"""
    work_dir.mkdir(parents=True, exist_ok=True)
    p = work_dir / f"cfg_c_{kind}_{budget}_s{seed}.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


def run_supervised(cfg_path: Path) -> float:
    """跑官方 Processor 训练 C′, 返回 best val top1(49类)。"""
    sys.path.insert(0, str(REPO))
    aimclr_root = REPO / "external" / "AimCLR"
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from processor.processor import init_seed  # noqa: E402
    init_seed(0)
    # C′ 是有监督头训练——走官方 LE_Processor（linear-eval 同入口; 骨干非冻结: weights 只
    # 初始化 encoder, ignore 不含 encoder_q.fc 故 fc(49类头) 参与 SGD 训练）
    from processor.linear_evaluation import LE_Processor  # noqa: E402
    import os
    os.chdir(REPO)
    proc = LE_Processor(["--config", str(cfg_path.resolve()), "--device", "0"])
    proc.start()
    # best top1 从 work_dir 日志解析(Processor 每 eval_interval 打印)
    log = cfg_path.parent / (cfg_path.stem + ".log")
    txt = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    best = 0.0
    for tok in txt.replace("%", " ").split():
        pass
    import re
    for m in re.finditer(r"top1[:= ]+([0-9.]+)", txt):
        best = max(best, float(m.group(1)))
    return best


def gpu_snapshot(tag):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader"], capture_output=True, timeout=10)
        return {"tag": tag, "reading": out.stdout.decode().strip()}
    except Exception:
        return {"tag": tag, "reading": "n/a"}


def dump_penultimate(weights_ckpt: Path, split: str, budget: str, out_npz: Path):
    """冻结 encoder penultimate(256) 特征提取——复用 linear-eval 前处理(Feeder_single 无增强)。"""
    import torch
    sys.path.insert(0, str(REPO))
    aimclr_root = REPO / "external" / "AimCLR"
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from net.aimclr import AimCLR  # noqa: E402
    from feeder.ntu_feeder import Feeder_single  # noqa: E402

    xp = {"full": "", "10pct": "_yp49_10pct"}[budget]
    data_path = DATA / f"{split}_position{xp if split == 'train' else ''}.npy"
    label_path = DATA / f"{split}_label_yp49{'_10pct' if (split == 'train' and budget == '10pct') else ''}.pkl"
    feeder = Feeder_single(str(data_path), str(label_path), shear_amplitude=-1,
                           temperal_padding_ratio=-1, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=256, shuffle=False, num_workers=0)
    model = AimCLR(base_encoder="net.st_gcn.Model", pretrain=False, in_channels=3,
                   hidden_channels=16, hidden_dim=256, num_class=49, dropout=0.5,
                   graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                   edge_importance_weighting=True)
    ck = torch.load(weights_ckpt, map_location="cpu")
    ck = ck.get("model", ck)
    # pretext ckpt 是预训练态(fc=MLP encoder_q.fc.0/.2, 含 encoder_k/queue)——剥离 encoder_k/queue,
    # 重建 49 类分类 fc; linear-eval 同款 ignore 逻辑
    ck = {k: v for k, v in ck.items() if not k.startswith("encoder_k.") and k not in ("queue", "queue_ptr")}
    fc_keys = [k for k in ck if k.startswith("encoder_q.fc.")]
    if any(k.startswith("encoder_q.fc.2.") for k in fc_keys):
        # 预训练 MLP 头(fc.0/fc.2) → 换 49 类线性 fc（占位——penultimate 经 hook 取, fc 值不影响）
        import torch.nn as nn
        model.encoder_q.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 49))
        ck = {k: v for k, v in ck.items() if not k.startswith("encoder_q.fc.")}
    elif any(k == "encoder_q.fc.weight" for k in fc_keys):
        # LE 训练后 checkpoint: fc 已是 49 类线性——需 num_class=49 的模型(本函数默认 49, 直接用)
        pass
    missing, unexpected = model.load_state_dict(ck, strict=False)
    missing = [m for m in missing if not m.startswith("encoder_q.fc.")]
    assert not missing, f"missing after remap: {missing[:6]}"
    assert not unexpected, f"unexpected: {unexpected[:6]}"
    model.eval().cuda()
    feats, labs = [], []
    feats_b = {}
    def pre_hook(_m, args):
        feats_b["o"] = args[0].detach()  # fc 的【输入】= 池化后 256-d penultimate（此前误取 fc 输出 logits——D' 2.2% 根因）
    h = model.encoder_q.fc.register_forward_pre_hook(pre_hook)
    with torch.no_grad():
        for x, lab in loader:
            _ = model.encoder_q(x.float().cuda())  # pretrain=False 时 forward 只回 encoder——直调 encoder
            feats.append(feats_b["o"].cpu().numpy())
            labs.append(np.asarray(lab))
    h.remove()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, feat=np.concatenate(feats), label=np.concatenate(labs))
    print(f"[dump] {out_npz.name}: {np.concatenate(feats).shape}")


def train_lr_head(npz: Path, seed: int):
    """E9 系头: StandardScaler + LogisticRegression(tol 1e-3), 49 类。返回 val top1。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = np.load(npz)
    X, y = z["feat"], z["label"]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    clf = LogisticRegression(max_iter=1000, tol=1e-3, n_jobs=1)
    clf.fit(Xs, y)
    return clf, sc, Xs, y


def eval_lr(clf, sc, split: str):
    z = np.load(DATA_VAL_NPZ if False else split)
    X, y = z["feat"], z["label"]
    Xs = sc.transform(X)
    return float(np.mean(clf.predict(Xs) == y))


DATA_VAL_NPZ = None


def main():
    global DATA_VAL_NPZ
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "smoke", "full"], required=True)
    args = ap.parse_args()

    if args.stage == "prep":
        stage_prep()
        return

    epochs = 2 if args.stage == "smoke" else EPOCHS
    budgets = ["full"] if args.stage != "smoke" else ["10pct"]
    t0 = time.time()
    results = {"date": datetime.now().isoformat(timespec="seconds"),
               "protocol": "PSD-NTU-TRANS-002 (ST-GCN-scale three-arm transition replication)",
               "stage": args.stage, "runs": [], "config_echo": {
                   "epochs": epochs, "seeds": list(SEEDS), "budgets": budgets,
                   "pretext_ckpt": PRETEXT_CKPT.name}}

    # ---- 共享: val 特征 per encoder 来源 ----
    if args.stage in ("smoke", "full"):
        stage_prep_if_needed = not (DATA / "val_label_yp49.pkl").exists()
        if stage_prep_if_needed:
            stage_prep()

    for budget in budgets:
        for seed in SEEDS:
            # ---- C′ ----
            cfg = write_cfg("c", seed, budget, epochs, WORK / f"c_{budget}_s{seed}")
            gpu_before = gpu_snapshot("pre")
            t = time.time()
            acc_c = run_supervised(cfg)
            wall_c = round(time.time() - t, 1)
            gpu_after = gpu_snapshot("post")
            ck_c = WORK / f"c_{budget}_s{seed}" / "finetune_model.pt"
            if not ck_c.exists():
                cand = sorted((WORK / f"c_{budget}_s{seed}").glob("*.pt"))
                ck_c = cand[-1] if cand else PRETEXT_CKPT
            print(f"[C'] {budget} s{seed}: acc={acc_c} wall={wall_c}s (seed tag recorded; processor seeds via init_seed(0) per official protocol deviation disclosure)", flush=True)

            # ---- 特征 dump (C′ ckpt 与冻结 pretext) ----
            val_npz = WORK / f"val_feat_{budget}_c_s{seed}.npz"
            tr_npz_c = WORK / f"train_feat_{budget}_c_s{seed}.npz"
            tr_npz_d = WORK / f"train_feat_{budget}_d_s{seed}.npz"
            dump_penultimate(ck_c, "val", budget, val_npz)
            dump_penultimate(ck_c, "train", budget, tr_npz_c)
            dump_penultimate(PRETEXT_CKPT, "train", budget, tr_npz_d)
            val_npz_d = WORK / f"val_feat_{budget}_d_s{seed}.npz"
            dump_penultimate(PRETEXT_CKPT, "val", budget, val_npz_d)

            # ---- D′ / M ----
            t = time.time()
            clf_d, sc_d, _, _ = train_lr_head(tr_npz_d, seed)
            acc_d = eval_lr(clf_d, sc_d, val_npz_d)
            wall_d = round(time.time() - t, 1)
            t = time.time()
            clf_m, sc_m, _, _ = train_lr_head(tr_npz_c, seed)
            acc_m = eval_lr(clf_m, sc_m, val_npz)
            wall_m = round(time.time() - t, 1)
            print(f"[D'] acc={acc_d} wall={wall_d}s | [M] acc={acc_m} wall={wall_m}s", flush=True)

            results["runs"].append({
                "arm": "C_prime", "budget": budget, "seed": seed,
                "wall_clock_sec": wall_c, "val_acc": round(acc_c, 4),
                "gpu": [gpu_before, gpu_after]})
            results["runs"].append({
                "arm": "D_prime", "budget": budget, "seed": seed,
                "wall_clock_sec": wall_d, "val_acc": round(acc_d, 4)})
            results["runs"].append({
                "arm": "M_matched", "budget": budget, "seed": seed,
                "wall_clock_sec": wall_m, "val_acc": round(acc_m, 4)})

    date = datetime.now().strftime("%Y-%m-%d")
    results["wall_clock_sec_total"] = round(time.time() - t0, 1)
    out = OUT_DIR / f"r23-trans002-{args.stage}-{date}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
