# -*- coding: utf-8 -*-
"""真实 tier 协议回归 — 无标签 AimCLR 物理层（预注册 ST-GCN pretext）+ 同三臂.

背景: PSD-NTU120/UCF101/PANAF-PREREG-001 预注册的物理层是 from-scratch
ST-GCN pretext（AimCLR 式、无标签）；执行时经 Amendment 2 偏离为
joint-MLP（80ep，且以全量标注 CE 训练——监督状态未在正文披露）。本脚本
执行预注册配置本身: 导出的 NTU25 槽位数据（export_tier_feeder.py）上训练
AimCLR 对比 pretext（150ep，单流 joint，无标签），冻结后重跑与已报告运行
逐参一致的三臂（(c) 全量线性参照 / (a) 10% 线性 / (b) 10%+PSD 语义管线
×10 seeds），判据同 E9（≥90 CONFIRMS / 85-90 PARTIAL / <85 FAILS）。

诚实条款: 旧 MLP 读数原样保留在 summary.prior_readout（不删不改判据）；
新读数为协议回归结果，两者并排发布。数字全部来自实跑。

用法:
    .venv/Scripts/python.exe -u scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120
产出:
    runs/p1_sprint/labelfree_<tier>/summary.json
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

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "external" / "AimCLR"))
sys.path.insert(0, str(REPO / "external" / "AimCLR" / "torchlight"))

TIERS = {"ntu120": 60, "ntu120_full": 120, "ucf101": 101, "panaf500": 9}
BUDGET_FRAC = 0.10
SEED_MAP = {10: tuple(range(42, 52))}


class ScaledLR:
    """与 run_p5b_generic_retention / frozen_ntu120 逐字一致。"""

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


def latest_ckpt(pretext_dir: Path) -> Path:
    cks = sorted(pretext_dir.glob("epoch*_model.pt"),
                 key=lambda p: int(p.name.split("epoch")[1].split("_")[0]))
    if not cks:
        raise SystemExit(f"[replicate] pretext ckpt 缺失: {pretext_dir}")
    return cks[-1]


def build_model(n_cls: int, device: str):
    from torchlight import import_class  # noqa: E402  (external/AimCLR torchlight)
    model_cls = import_class("net.aimclr.AimCLR")
    model = model_cls(
        base_encoder="net.st_gcn.Model", pretrain=True, feature_dim=128,
        queue_size=32768, momentum=0.999, Temperature=0.07, mlp=True,
        in_channels=3, hidden_channels=16, hidden_dim=256, num_class=n_cls,
        dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    )
    return model


def extract_backbone(enc, x):
    """st_gcn.Model fc 前全局池化特征 (N,256) — 与 run_p14_ntu_featuredump 逐字一致。"""
    n, c, t, v, m = x.size()
    h = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
    h = enc.data_bn(h)
    h = h.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
    for gcn, imp in zip(enc.st_gcn_networks, enc.edge_importance):
        h, _ = gcn(h, enc.A * imp)
    h = torch.nn.functional.avg_pool2d(h, h.size()[2:])
    h = h.view(n, m, -1).mean(dim=1)
    return h


def dump_features(model, tier: str, device: str) -> dict:
    from feeder.ntu_feeder import Feeder_single  # noqa: E402  (external/AimCLR)
    import torch.utils.data as tud  # noqa: E402

    enc = model.encoder_q
    feeder_dir = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
    store = {}
    for split in ("train", "val"):
        feeder = Feeder_single(str(feeder_dir / f"{split}_position.npy"),
                               str(feeder_dir / f"{split}_label.pkl"),
                               shear_amplitude=-1, temperal_padding_ratio=-1, mmap=True)
        loader = tud.DataLoader(feeder, batch_size=128, shuffle=False, num_workers=0)
        feats, labs = [], []
        for data, label in loader:
            x = data.float().to(device)
            with torch.no_grad():
                feats.append(extract_backbone(enc, x).cpu().numpy())
            labs.append(np.asarray(label).reshape(-1))
        store[f"{split}_feat"] = np.vstack(feats).astype(np.float32)
        store[f"{split}_label"] = np.concatenate(labs).astype(np.int64)
        print(f"[feat:{split}] {store[f'{split}_feat'].shape}", flush=True)
    return store


def run_arms(model, tier: str, seeds, device: str, budget_frac: float = BUDGET_FRAC) -> dict:
    """E9 三臂统一测量（唯一实现；adapt 系脚本复用同一把尺，防基线漂移）:
      (c) 全量线性参照 / (a) 10% 分层线性头 / (b) 10%+PSD 语义管线 ×seeds。
    判据同 E9: ≥90 CONFIRMS / 85-90 PARTIAL / <85 FAILS。
    返回 dict 含 store(train/val 特征与标签) 与 last_b(末种子伪标签工件，
    供伪标签回灌线 supcon_feedback_tier 复用)。"""
    from psd.training.tcl_selftrain import run_selftrain  # noqa: E402
    from run_p5b_generic_retention import KW  # noqa: E402  （与已报告运行同参）

    store = dump_features(model, tier, device)
    tr_feat, tr_lab = store["train_feat"], store["train_label"]
    va_feat, va_lab = store["val_feat"], store["val_label"]

    # (c) 100% 线性参照
    ref = float(np.mean(ScaledLR().fit(tr_feat, tr_lab).predict(va_feat) == va_lab))
    print(f"[c] full ref: {ref:.4f}", flush=True)

    # (a) 10% 分层子集线性头（E9 逐行同式, seed 42）
    rng = np.random.default_rng(42)
    mask_a = np.zeros(len(tr_lab), dtype=bool)
    for c in sorted(set(tr_lab.tolist())):
        ci = np.where(tr_lab == c)[0]
        k = max(1, int(round(len(ci) * budget_frac)))
        mask_a[rng.choice(ci, size=k, replace=False)] = True
    acc_a = float(np.mean(ScaledLR().fit(tr_feat[mask_a], tr_lab[mask_a])
                          .predict(va_feat) == va_lab))
    print(f"[a] 10% linear: {acc_a:.4f}", flush=True)

    # (b) 10% + PSD 语义管线 × seeds（与已报告运行同参同判据）
    labels_str = np.array([str(l) for l in tr_lab])
    full_labels_str = np.array([str(l) for l in np.concatenate([tr_lab, va_lab])])
    class_names = [str(c) for c in sorted(set(tr_lab.tolist()))]
    accs_b = []
    last_b: dict = {}
    for seed in seeds:
        anchor = np.zeros(len(tr_lab), dtype=bool)
        rng2 = np.random.default_rng(seed)
        for c in class_names:
            ci = np.where(labels_str == c)[0]
            k = max(1, int(round(len(ci) * budget_frac)))
            anchor[rng2.choice(ci, size=k, replace=False)] = True
        universe = np.ones(len(tr_lab), dtype=bool)
        universe[anchor] = False
        r = run_selftrain(tr_feat, labels_str, anchor, run_seed=seed,
                          class_names=class_names,
                          head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                    "weight_decay": 1e-4, "batch_size": 128, "device": "cpu"},
                          pool_universe_mask=universe, **KW)
        tm2 = anchor.copy()
        tm2[r["final_pool_idx"]] = True
        y2 = np.array([None] * len(labels_str), dtype=object)
        y2[anchor] = labels_str[anchor]
        y2[r["final_pool_idx"]] = r["final_pred_full"][r["final_pool_idx"]]
        va_lab_str = full_labels_str[len(tr_lab):]  # 字符串域评分（r16 首跑缺陷教训：str/int 比较恒 False）
        acc_b = float(np.mean(ScaledLR().fit(tr_feat[tm2], y2[tm2]).predict(va_feat)
                              == va_lab_str))
        accs_b.append(acc_b)
        print(f"[b] seed{seed}: {acc_b:.4f} pool={len(r['final_pool_idx'])} stop={r['stop_reason']}", flush=True)
        last_b = {"seed": int(seed), "anchor": anchor, "train_mask": tm2, "y2": y2,
                  "final_pool_idx": r["final_pool_idx"], "final_pred_full": r["final_pred_full"],
                  "stop_reason": r["stop_reason"]}

    ret = float(np.mean(accs_b)) / ref if ref > 0 else 0.0
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    return {"full_ref": ref, "linear_10pct": acc_a, "b_arms": accs_b,
            "retention": ret, "verdict": verdict, "store": store, "last_b": last_b}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True, choices=sorted(TIERS))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="显式种子列表（覆盖 --seeds；用于断点补跑个别种子）")
    args = ap.parse_args()
    tier = args.tier
    n_cls = TIERS[tier]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = tuple(args.seed_list) if args.seed_list else SEED_MAP.get(args.seeds, tuple(range(42, 42 + args.seeds)))

    base = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}"
    ckpt = latest_ckpt(base / "pretext")
    print(f"[replicate:{tier}] ckpt = {ckpt}")

    model = build_model(n_cls, "cpu")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_trainable == 0, "冻结失败"
    print(f"[encoder] frozen verified: trainable=0/{sum(p.numel() for p in model.parameters())}")

    t0 = time.time()
    arms = run_arms(model, tier, seeds, device)
    ref, acc_a = arms["full_ref"], arms["linear_10pct"]
    accs_b, ret, verdict = arms["b_arms"], arms["retention"], arms["verdict"]

    prior_path = REPO / "runs" / "p1_sprint" / tier / "summary.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8")) if prior_path.exists() else {}
    prior_readout = {k: prior.get(k) for k in ("full_ref", "linear_10pct", "b_arms",
                                               "retention", "verdict", "encoder")} if prior else {}

    result = {
        "doneMarker": f"P1_LABELFREE_TIER_{tier.upper()}_DONE",
        "experiment": "p1_labelfree_tier_protocol_return",
        "tier": tier,
        "mode": "full",
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": f"protocol-return of the pre-registered from-scratch ST-GCN pretext "
                    f"(150 ep, label-free AimCLR objective, single stream joint); "
                    f"the interim joint-MLP deviation (Amendment 2) is retained as prior readout",
        "encoder": {"arch": "AimCLR (net.st_gcn base, 16 hidden ch.), from-scratch, label-free",
                    "ckpt": str(ckpt.relative_to(REPO)), "epochs": 150,
                    "frozen": True, "trainable_params": 0,
                    "total_params": int(sum(p.numel() for p in model.parameters()))},
        "budget_frac": BUDGET_FRAC,
        "full_ref": round(ref, 4), "linear_10pct": round(acc_a, 4),
        "b_arms": [round(a, 4) for a in accs_b],
        "retention": round(ret, 4),
        "linear_retention": round(acc_a / ref, 4) if ref > 0 else 0.0,
        "verdict": verdict,
        "seeds": list(seeds),
        "prior_readout": prior_readout,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = base / "summary.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps({k: result[k] for k in ("full_ref", "linear_10pct", "retention", "verdict")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
