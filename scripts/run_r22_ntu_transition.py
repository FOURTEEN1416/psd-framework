# -*- coding: utf-8 -*-
"""PSD-NTU-TRANS-001 — NTU60 真域 taxonomy-transition cost replication（R22b#5）。

Y(60类官方) → Y′(49类, 10 个冻结合并, ADR 0002 粗粒度日报场景移植)。
两臂同架构（joint-level MLP, E9b/c 口径）、同数据、同 Y′ 标签、同 80ep：
  - Arm D (decoupled): MLP 在 Y 标签上训练一次（自监督位被"Y 标签 pretext"替代
    的披露见协议 §3——MLP 架构无对比目标, 以标签 pretext 训练后冻结 penultimate,
    Ω=线性头在 Y′ 重训。转换时只重训 Ω——wall-clock 只计 Ω 重训）
  - Arm C (coupled): 同架构端到端有监督从零训练于 Y′ 标签（pretext+head 联合）,
    转换 = 全模型重训——wall-clock 计全部 80ep
判定（协议 §4）: CONFIRMS = median wall-clock(D/C) ≥ 3× 且 |acc_D−acc_C| < 2.3pp;
两档(full/10%)独立判定; 非对称预期如实记录。

wall-clock 口径（协议 §6, 与 E6 一致）: 只计 trainer 训练段(perf_counter 包络),
不含特征 dump/评估; GPU 快照前后各一次作干扰证据。GPU 独占要求由调度保证
(扩容链收官后启动)。

用法:
    python scripts/run_r22_ntu_transition.py
产出:
    reports/r22-ntu-transition-<date>.json
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from run_p5b_generic_retention import (  # noqa: E402
    JointMLP,
    dump_features,
    load_dataset,
    prep_clip,
    train_pretext_mlp,
)

PKL = REPO / "data" / "pyskl" / "ntu60_hrnet.pkl"
OUT_DIR = REPO / "reports"
SEEDS = (42, 43, 44)
EPOCHS = 80
BUDGET_FRAC = 0.10
ACC_BAND_PP = 2.3

# ---- Y → Y′ 冻结合并表（协议 §2; pyskl label 为官方顺序 0-59）----
# 官方类名表(NTU60 xsub 60 类顺序):
NTU60_CLASSES = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pick up", "throw", "sitting down", "standing up (from sitting position)",
    "clapping", "reading", "writing", "tear up paper", "wear jacket",
    "take off jacket", "wear a shoe", "take off a shoe", "wear on glasses",
    "take off glasses", "put on a hat/cap", "take off a hat/cap", "cheer up",
    "hand waving", "kicking something", "reach into pocket", "hopping",
    "jump up", "phone call", "play with phone/tablet", "type on a keyboard",
    "point to something", "taking a selfie", "check time (from watch)",
    "rub two hands together", "nod head/bow", "shake head", "wipe face",
    "salute", "put the palms together", "cross hands in front", "sneeze/cough",
    "staggering", "falling", "touch head", "touch chest", "touch back",
    "touch neck", "nausea or vomiting", "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person", "kicking other person",
    "pushing other person", "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking", "walking towards each other",
    "walking apart from each other",
]
NAME_TO_IDX = {n: i for i, n in enumerate(NTU60_CLASSES)}

# 协议 §2 十组合并组
MERGE_GROUPS = [
    (["drink water", "eat meal/snack"], "consume"),
    (["brushing teeth", "brushing hair"], "groom-brush"),
    (["sitting down", "standing up (from sitting position)"], "sit-stand transition"),
    (["wear jacket", "take off jacket"], "jacket on/off"),
    (["wear a shoe", "take off a shoe"], "shoe on/off"),
    (["wear on glasses", "take off glasses"], "glasses on/off"),
    (["put on a hat/cap", "take off a hat/cap"], "hat on/off"),
    (["nod head/bow", "shake head"], "head gesture"),
    (["punching/slapping other person", "kicking other person", "pushing other person"], "strike other person"),
    (["walking towards each other", "walking apart from each other"], "bidirectional walk"),
]


def build_y_to_yp_map_ntu() -> tuple[dict, list]:
    """返回 (y_label -> yp_label 映射 dict) 与 Y′ 类名列表。合并组外原样保留。"""
    yp_names: list = []
    y2yp: dict = {}
    for members, new_name in MERGE_GROUPS:
        yp_names.append(new_name)
        new_id = len(yp_names) - 1
        for m in members:
            y2yp[NAME_TO_IDX[m]] = new_id
    kept = 0
    for i, name in enumerate(NTU60_CLASSES):
        if i not in y2yp:
            yp_names.append(name)
            y2yp[i] = len(yp_names) - 1
            kept += 1
    assert len(yp_names) == 49, f"Y′ class count {len(yp_names)} != 49"
    assert set(y2yp) == set(range(60))
    return y2yp, yp_names


class ScaledLR:
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


def train_endtoend_mlp(X, y, n_classes, epochs, device, seed):
    """耦合臂: 与 pretext 同架构同超参, 端到端有监督训练, 返回 (model, wall_clock, epochs_run)。"""
    import torch
    torch.manual_seed(seed)
    model = JointMLP(int(np.prod(X.shape[1:])), n_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = torch.nn.CrossEntropyLoss()
    xt = torch.from_numpy(X.reshape(len(X), -1)).to(device)
    yt = torch.from_numpy(y).long().to(device)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for s in range(0, len(xt), 256):
            idx = perm[s:s + 256]
            loss = ce(model(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    wall = time.perf_counter() - t0
    model.eval()
    return model, wall, epochs


def one_shot(results_list, arm, budget, seed, wall, acc, gpu_before, gpu_after):
    results_list.append({
        "arm": arm, "budget": budget, "seed": seed,
        "wall_clock_sec": round(wall, 2), "val_acc": round(acc, 4),
        "gpu_before": gpu_before, "gpu_after": gpu_after,
    })


def main():
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device == "cuda", "wall-clock 计时要求 GPU 独占（协议 §6）; 当前无 CUDA"
    t0 = time.time()
    y2yp, yp_names = build_y_to_yp_map_ntu()

    rows = load_dataset(PKL, 60000)   # NTU60 全量
    print("[prep] resampling clips...", flush=True)
    X = np.stack([prep_clip(r["kp"]) for r in rows])
    y60 = np.array([r["label"] for r in rows])
    y49 = np.array([y2yp[int(v)] for v in y60])
    splits = np.array([r["split"] for r in rows])
    tr, vm = splits == "train", splits == "val"
    print(f"[data] X {X.shape} train={tr.sum()} val={vm.sum()} Y'=49 classes", flush=True)

    # 10% 分层子集（seed 42, 与 E9 同约定, 两臂共用）
    rng = np.random.default_rng(42)
    tr_idx = np.where(tr)[0]
    mask_10 = np.zeros(len(tr_idx), dtype=bool)
    for c in sorted(set(y49[tr_idx].tolist())):
        ci = np.where(y49[tr_idx] == c)[0]
        k = max(1, int(round(len(ci) * BUDGET_FRAC)))
        mask_10[rng.choice(ci, size=k, replace=False)] = True

    def gpu_snap():
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader"], text=True).strip()
            return out
        except Exception:
            return "n/a"

    runs = []
    # ---- Arm D: pretext 一次（Y 标签, 80ep）→ 冻结 → Ω 线性头 Y′ 重训（两档共用 pretext）----
    print("[D] pretext (Y-label MLP 80ep)...", flush=True)
    gb = gpu_snap()
    tD = time.perf_counter()
    modelD = train_pretext_mlp(X[tr], y60[tr], 60, epochs=EPOCHS, device=device)
    wall_pretext = time.perf_counter() - tD
    ga = gpu_snap()
    print(f"[D] pretext wall={wall_pretext:.1f}s", flush=True)
    feats = dump_features(modelD, X)
    for budget, mask in (("full", np.ones(len(tr_idx), bool)), ("10pct", mask_10)):
        for seed in SEEDS:
            idx = tr_idx[mask]
            tH = time.perf_counter()
            clf = ScaledLR().fit(feats[idx], y49[idx])
            wall_head = time.perf_counter() - tH
            acc = float(np.mean(clf.predict(feats[vm]) == y49[vm]))
            # 转换成本 = Ω 重训; pretext 摊销披露进 JSON（ amortized_pretext_sec ）
            one_shot(runs, "D", budget, seed, wall_head, acc, gb, ga)
            print(f"[D {budget} s{seed}] head={wall_head:.2f}s acc={acc:.4f}", flush=True)
    # 每臂 seed 间 head 训练含微小的 solver 随机性; 线性头确定性 → wall 差异来自环境噪声
    # （如实记录; 比值判定用 median 抗噪声）

    # ---- Arm C: 端到端从零（Y′ 标签, 80ep）× 两档 × 3 seeds ----
    for budget, mask in (("full", np.ones(len(tr_idx), bool)), ("10pct", mask_10)):
        idx = tr_idx[mask]
        for seed in SEEDS:
            gb = gpu_snap()
            modelC, wallC, epC = train_endtoend_mlp(
                X[idx], y49[idx], 49, EPOCHS, device, seed)
            ga = gpu_snap()
            with torch.no_grad():
                xt = torch.from_numpy(X[vm].reshape(len(X[vm]), -1)).to(device)
                pred = modelC(xt).argmax(1).cpu().numpy()
            acc = float(np.mean(pred == y49[vm]))
            one_shot(runs, "C", budget, seed, wallC, acc, gb, ga)
            print(f"[C {budget} s{seed}] wall={wallC:.1f}s acc={acc:.4f}", flush=True)
            del modelC
            torch.cuda.empty_cache()

    # ---- 判定（协议 §4）----
    verdicts = {}
    for budget in ("full", "10pct"):
        d_w = [r["wall_clock_sec"] for r in runs if r["arm"] == "D" and r["budget"] == budget]
        c_w = [r["wall_clock_sec"] for r in runs if r["arm"] == "C" and r["budget"] == budget]
        d_a = [r["val_acc"] for r in runs if r["arm"] == "D" and r["budget"] == budget]
        c_a = [r["val_acc"] for r in runs if r["arm"] == "C" and r["budget"] == budget]
        ratio = float(np.median(c_w) / max(np.median(d_w), 1e-9))
        acc_gap_pp = 100 * (float(np.mean(d_a)) - float(np.mean(c_a)))
        if ratio >= 3.0 and abs(acc_gap_pp) < ACC_BAND_PP:
            v = "CONFIRMS"
        elif ratio >= 3.0:
            v = "PARTIAL"
        else:
            v = "FAILS"
        verdicts[budget] = {
            "median_ratio_D_over_C": round(ratio, 2),
            "acc_D_mean": round(float(np.mean(d_a)), 4),
            "acc_C_mean": round(float(np.mean(c_a)), 4),
            "acc_gap_pp_D_minus_C": round(acc_gap_pp, 2),
            "band_pp": ACC_BAND_PP, "verdict": v,
        }
        print(f"[verdict {budget}] ratio={ratio:.2f}x gap={acc_gap_pp:+.2f}pp -> {v}", flush=True)

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-NTU-TRANS-001 (real-domain taxonomy-transition cost replication, R22b#5)",
        "layer": "public_human_benchmark",
        "y_prime": {"n_classes": 49, "merge_groups": [m for m, _ in MERGE_GROUPS],
                    "yp_names": yp_names},
        "arms_note": "D = frozen Y-pretext MLP penultimate + linear head retrain on Y' (transition cost = head only); "
                     "C = same architecture end-to-end supervised from scratch on Y' (transition cost = full 80ep). "
                     "D-pretext trained once on Y labels (80ep) and amortized across transitions: "
                     f"pretext wall {wall_pretext:.1f}s disclosed, excluded from D per-transition cost per protocol §3.",
        "runs": runs,
        "verdicts": verdicts,
        "config_echo": {"epochs": EPOCHS, "seeds": list(SEEDS), "budget_frac": BUDGET_FRAC,
                        "device": device, "n_train": int(tr.sum()), "n_val": int(vm.sum())},
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out = OUT_DIR / f"r22-ntu-transition-{date}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out}", flush=True)
    print(json.dumps(verdicts, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
