# -*- coding: utf-8 -*-
"""PSD-NTU-TRANS-002 — ST-GCN 规模三臂 taxonomy-transition 复制（重写版 v2）。

Y(60) → Y′(49, TRANS-001 冻结合并表)。三臂:
  C′  coupled   : net.st_gcn.Model 端到端有监督从零训练 80ep 于 Y′
                  （pyskl vanilla NTU60-xsub-HRNet 官方配方: SGD lr0.1 m0.9 wd1e-4
                   CosineAnnealing 等效80ep batch128, 官方 60 类基准 85.7%;
                   本机 8GB → batch16 + 官方线性缩放 lr=0.0125; FT_Processor
                   无 scheduler → 恒定 lr, 如实披露）
  D′  decoupled : 冻结 AimCLR joint pretext(epoch300) penultimate + StandardScaler+LR 头
  M   matched   : 冻结 C′ checkpoint penultimate + 同一 StandardScaler+LR 头
预算 full(40091) / 10%(4009, seed42 分层)。C′ seeds 42/43/44 经 init_seed(seed) 生效;
D′/M 的头为确定性 sklearn LR（pretext 固定, 与 TRANS-001 D 臂同先例）。
top1 自算（模型 forward 返回 logits）, 不依赖日志解析。

用法:
    python scripts/run_r23_trans002.py --stage prep
    python scripts/run_r23_trans002.py --stage smoke   # 2ep×10pct, 验证不 nan 且 C′>40%
    python scripts/run_r23_trans002.py --stage full
产出:
    reports/r23-trans002-{smoke|full}-<date>.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

DATA = REPO / "data" / "ntu60_frame50" / "xsub"
WORK = REPO / "runs" / "trans002"
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
EPOCHS = 80
PRETEXT_CKPT = REPO / "runs" / "ntu_phaseB" / "joint_pretext" / "epoch300_model.pt"

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
    name2old = {n: i for i, n in enumerate(labels60)}
    y2yp, new_idx, merged = {}, 0, set()
    for grp in MERGE_GROUPS:
        for g in grp:
            o = name2old[g]
            y2yp[o] = new_idx
            merged.add(o)
        new_idx += 1
    for o in range(60):
        if o not in merged:
            y2yp[o] = new_idx
            new_idx += 1
    assert new_idx == 49
    return y2yp


def stage_prep():
    WORK.mkdir(parents=True, exist_ok=True)
    with open(DATA / "train_label.pkl", "rb") as f:
        tr_names, tr_label = pickle.load(f)
    with open(DATA / "val_label.pkl", "rb") as f:
        _, va_label = pickle.load(f)
    from run_r22_ntu_transition import NTU60_CLASSES  # noqa: E402
    names = list(NTU60_CLASSES)
    y2yp = build_merge_map(names)
    tr_yp = [int(y2yp[int(l)]) for l in tr_label]
    va_yp = [int(y2yp[int(l)]) for l in va_label]
    for split, lab in (("train", tr_yp), ("val", va_yp)):
        sample_names = [f"{split}_{i:06d}" for i in range(len(lab))]
        with open(DATA / f"{split}_label_yp49.pkl", "wb") as f:
            pickle.dump((sample_names, lab), f)
    sys.path.insert(0, str(REPO / "scripts"))
    from run_p14_ntu_lowres import stratified_10pct  # noqa: E402
    mask = stratified_10pct(np.array(tr_label), 42)
    dst_x = DATA / "train_position_yp49_10pct.npy"
    if not dst_x.exists():
        full = np.load(DATA / "train_position.npy", mmap_mode="r")
        np.save(dst_x, np.ascontiguousarray(full[mask]))
        del full
    sub_lab = [tr_yp[i] for i in np.where(mask)[0]]
    sub_names = [f"train10pct_{i:06d}" for i in range(len(sub_lab))]
    with open(DATA / "train_label_yp49_10pct.pkl", "wb") as f:
        pickle.dump((sub_names, sub_lab), f)
    print("[prep] Y' labels + 10% slices written")


def write_cfg(seed: int, budget: str, epochs: int, work_dir: Path) -> Path:
    lr = round(0.1 * 16 / 128, 4)  # 官方线性缩放: 0.1 × 16/128
    xp = {"full": "", "10pct": "_yp49_10pct"}[budget]
    cfg = f"""# TRANS-002 arm C' (frozen protocol PSD-NTU-TRANS-002; pyskl vanilla recipe, batch-16 linear-scaled LR)
work_dir: {work_dir.as_posix()}

train_feeder: feeder.ntu_feeder.Feeder_single
train_feeder_args:
  data_path: data/ntu60_frame50/xsub/train_position{xp}.npy
  label_path: data/ntu60_frame50/xsub/train_label_yp49{'_10pct' if budget == '10pct' else ''}.pkl
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
    p = work_dir / f"cfg_c_{budget}_s{seed}.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


def run_supervised(cfg_path: Path, seed: int) -> float:
    """FT_Processor 有监督从零训练; 返回 val top1（自算, 不解析日志）。"""
    sys.path.insert(0, str(REPO))
    aimclr_root = REPO / "external" / "AimCLR"
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from processor.processor import init_seed  # noqa: E402
    init_seed(seed)  # 修复: per-seed（此前写死 0, 三 seed 同值）
    from processor.finetune_evaluation import FT_Processor  # noqa: E402
    import net.st_gcn as _stgcn_mod  # noqa: E402

    class _AdapterShim(torch.nn.Module):
        # io.load_model 按 import_class('net.st_gcn.Model') 构建——用模块级替换注入适配层
        _Inner = _stgcn_mod.Model

        def __init__(self, *a, **k):
            super().__init__()
            self.inner = self._Inner(*a, **k)

        def forward(self, _ignored, x):
            return self.inner(x)

    _original_model_cls = _stgcn_mod.Model
    _stgcn_mod.Model = _AdapterShim
    try:
        import os
        os.chdir(REPO)
        proc = FT_Processor(["--config", str(cfg_path.resolve()), "--device", "0"])
        proc.start()
    finally:
        _stgcn_mod.Model = _original_model_cls  # 恢复原类——eval_49 需裸 STGCNModel
    best_ck = cfg_path.parent / "best_model.pt"
    if not best_ck.exists():
        cands = sorted(cfg_path.parent.glob("epoch*.pt"))
        best_ck = cands[-1] if cands else None
    if best_ck is None:
        raise FileNotFoundError(f"no C' checkpoint in {cfg_path.parent}")
    return eval_49(best_ck, DATA / "val_position.npy", DATA / "val_label_yp49.pkl")


def eval_49(ckpt: Path, data_npy: Path, label_pkl: Path) -> float:
    """用 st_gcn.Model 的 49 类 fc 自算 top1（无需解析日志）。"""
    import torch
    sys.path.insert(0, str(REPO))
    aimclr_root = REPO / "external" / "AimCLR"
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from net.st_gcn import Model as STGCNModel  # noqa: E402
    from feeder.ntu_feeder import Feeder_single  # noqa: E402

    model = STGCNModel(in_channels=3, hidden_channels=16, hidden_dim=256, num_class=49,
                       dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                       edge_importance_weighting=True)
    ck = torch.load(ckpt, map_location="cpu")
    sd = ck.get("model", ck)
    if any(k.startswith("inner.") for k in sd):
        sd = { k[len("inner."):]: v for k, v in sd.items() }
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()
    feeder = Feeder_single(str(data_npy), str(label_pkl), shear_amplitude=-1,
                           temperal_padding_ratio=-1, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=256, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for x, lab in loader:
            logits = model(x.float().cuda())
            pred = logits.argmax(1).cpu().numpy()
            correct += int((pred == np.asarray(lab)).sum())
            total += len(lab)
    return correct / total


def dump_penultimate(weights_ckpt: Path, split: str, budget: str, out_npz: Path):
    """penultimate(256) 特征：pretext→AimCLR 包装(fc pre-hook)；st_gcn.Model ckpt→同形直调。"""
    import torch
    sys.path.insert(0, str(REPO))
    aimclr_root = REPO / "external" / "AimCLR"
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from feeder.ntu_feeder import Feeder_single  # noqa: E402

    xp = {"full": "", "10pct": "_yp49_10pct"}[budget]
    data_path = DATA / f"{split}_position{xp if split == 'train' else ''}.npy"
    label_path = DATA / f"{split}_label_yp49{'_10pct' if (split == 'train' and budget == '10pct') else ''}.pkl"
    feeder = Feeder_single(str(data_path), str(label_path), shear_amplitude=-1,
                           temperal_padding_ratio=-1, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=256, shuffle=False, num_workers=0)

    feats_b = {}

    def pre_hook(_m, args):
        feats_b["o"] = args[0].detach()  # fc 输入 = 池化后 256-d penultimate

    is_pretext = "pretext" in weights_ckpt.name or "joint_pretext" in str(weights_ckpt)
    if is_pretext:
        from net.aimclr import AimCLR  # noqa: E402
        model = AimCLR(base_encoder="net.st_gcn.Model", pretrain=False, in_channels=3,
                       hidden_channels=16, hidden_dim=256, num_class=49, dropout=0.5,
                       graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                       edge_importance_weighting=True)
        ck = torch.load(weights_ckpt, map_location="cpu")
        ck = ck.get("model", ck)
        ck = {k: v for k, v in ck.items() if not k.startswith("encoder_k.") and k not in ("queue", "queue_ptr")}
        if any(k.startswith("encoder_q.fc.2.") for k in ck):
            import torch.nn as nn
            model.encoder_q.fc = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 49))
            ck = {k: v for k, v in ck.items() if not k.startswith("encoder_q.fc.")}
        missing, unexpected = model.load_state_dict(ck, strict=False)
        missing = [m for m in missing if not m.startswith("encoder_q.fc.")]
        assert not missing, f"missing: {missing[:6]}"
        assert not unexpected, f"unexpected: {unexpected[:6]}"
        encoder = model.encoder_q
        h = encoder.fc.register_forward_pre_hook(pre_hook)
    else:
        from net.st_gcn import Model as STGCNModel  # noqa: E402
        model = STGCNModel(in_channels=3, hidden_channels=16, hidden_dim=256, num_class=49,
                           dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                           edge_importance_weighting=True)
        ck = torch.load(weights_ckpt, map_location="cpu")
        sd = ck.get("model", ck)
        if any(k.startswith("inner.") for k in sd):
            sd = { k[len("inner."):]: v for k, v in sd.items() }
        model.load_state_dict(sd, strict=True)
        encoder = model
        h = model.fc.register_forward_pre_hook(pre_hook)
    model.eval().cuda()
    feats, labs = [], []

    with torch.no_grad():
        for x, lab in loader:
            xt = x.float().cuda()
            assert xt is not None and xt.dim() == 5, f"bad batch input: {type(xt)}"
            _ = encoder(xt)
            o = feats_b.get("o")
            assert o is not None, "pre_hook did not fire (fc never called)"
            feats.append(o.cpu().numpy())
            labs.append(np.asarray(lab))
    h.remove()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, feat=np.concatenate(feats), label=np.concatenate(labs))
    print(f"[dump] {out_npz.name}: {np.concatenate(feats).shape}", flush=True)


def train_lr_head(npz: Path):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    z = np.load(npz)
    sc = StandardScaler()
    Xs = sc.fit_transform(z["feat"])
    clf = LogisticRegression(max_iter=1000, tol=1e-3)
    clf.fit(Xs, z["label"])
    return clf, sc


def eval_lr(clf, sc, npz: Path) -> float:
    z = np.load(npz)
    return float(np.mean(clf.predict(sc.transform(z["feat"])) == z["label"]))


def gpu_snapshot(tag):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader"], capture_output=True, timeout=10)
        return {"tag": tag, "reading": out.stdout.decode().strip()}
    except Exception:
        return {"tag": tag, "reading": "n/a"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "smoke", "full"], required=True)
    args = ap.parse_args()

    if args.stage == "prep":
        stage_prep()
        return

    epochs = 2 if args.stage == "smoke" else EPOCHS
    budget = "10pct" if args.stage == "smoke" else "full"
    seeds = (42,) if args.stage == "smoke" else SEEDS
    t0 = time.time()
    results = {"date": datetime.now().isoformat(timespec="seconds"),
               "protocol": "PSD-NTU-TRANS-002 (ST-GCN-scale three-arm transition replication)",
               "stage": args.stage, "runs": [],
               "config_echo": {"epochs": epochs, "seeds": list(seeds), "budget": budget,
                               "lr": 0.0125, "batch": 16, "recipe_source": "pyskl vanilla NTU60-xsub-HRNet (85.7% 60-class anchor)",
                               "scheduler": "constant lr (FT_Processor has none; disclosed)",
                               "pretext_ckpt": PRETEXT_CKPT.name}}

    val_npz_d = WORK / "val_feat_penult_pretext.npz"
    if not val_npz_d.exists():
        dump_penultimate(PRETEXT_CKPT, "val", budget, val_npz_d)

    for seed in seeds:
        run_tag = f"c_{budget}_s{seed}_{datetime.now().strftime('%H%M%S')}"
        cfg = write_cfg(seed, budget, epochs, WORK / run_tag)
        gpu_before = gpu_snapshot("pre")
        t = time.time()
        acc_c = run_supervised(cfg, seed)
        wall_c = round(time.time() - t, 1)
        gpu_after = gpu_snapshot("post")
        ck_c = WORK / run_tag / "best_model.pt"
        print(f"[C'] {budget} s{seed}: acc={acc_c:.4f} wall={wall_c}s", flush=True)

        tr_npz_c = WORK / f"train_feat_{budget}_c_s{seed}_{run_tag[-6:]}.npz"
        tr_npz_d = WORK / f"train_feat_{budget}_d_s{seed}.npz"
        val_npz_c = WORK / f"val_feat_{budget}_c_s{seed}_{run_tag[-6:]}.npz"
        dump_penultimate(ck_c, "train", budget, tr_npz_c)
        dump_penultimate(ck_c, "val", budget, val_npz_c)
        dump_penultimate(PRETEXT_CKPT, "train", budget, tr_npz_d)

        t = time.time()
        clf_d, sc_d = train_lr_head(tr_npz_d)
        acc_d = eval_lr(clf_d, sc_d, val_npz_d)
        wall_d = round(time.time() - t, 1)
        t = time.time()
        clf_m, sc_m = train_lr_head(tr_npz_c)
        acc_m = eval_lr(clf_m, sc_m, val_npz_c)
        wall_m = round(time.time() - t, 1)
        print(f"[D'] acc={acc_d:.4f} wall={wall_d}s | [M] acc={acc_m:.4f} wall={wall_m}s", flush=True)

        results["runs"] += [
            {"arm": "C_prime", "budget": budget, "seed": seed, "wall_clock_sec": wall_c,
             "val_acc": round(acc_c, 4), "gpu": [gpu_before, gpu_after]},
            {"arm": "D_prime", "budget": budget, "seed": seed, "wall_clock_sec": wall_d,
             "val_acc": round(acc_d, 4)},
            {"arm": "M_matched", "budget": budget, "seed": seed, "wall_clock_sec": wall_m,
             "val_acc": round(acc_m, 4)},
        ]

    results["wall_clock_sec_total"] = round(time.time() - t0, 1)
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r23-trans002-{args.stage}-{date}.json"
    out.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
