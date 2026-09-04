"""P0.8 同协议方法对比消融臂 — AimCLR(通用 SSL 预训练) 替代 PSD 语义 warm-start。

任务背景: PR 审稿风险"无同数据集对比基线" + 核心主张"语义层 warm-start 是低标注
预算关键"需要 −语义层 消融证据。本臂在 E7/E8 完全相同的端到端协议下
(seeds spc -> run_selftrain anchor-cluster-pseudo-label -> seeds+pool linear head -> val),
仅替换 backbone 初始化: PSD warm(Y_CKPT, AK 22 类监督) -> AimCLR(encoder_q,
InterPet4D mocap 通用 SSL)。scratch 随机臂数字直接引 p07 报告, 不重跑。

口径声明: 公开真实层 (AK full12, 197 clips, 9 类有样本)。各臂使用其原生预处理
(AimCLR: NTU 兼容视图+序列归一; PSD: 原始 (30,24,3))——预处理随方法绑定是
公平对比惯例, 已在报告披露。数据混淆披露: AimCLR 预训练源为 mocap (InterPet4D),
与 in-the-wild AK 存在采集域差, 本臂为"通用 SSL 表征迁移"下界参照, 非 AimCLR 最优调参。

用法:
    .venv/Scripts/python.exe scripts/run_p08_aimclr_arm.py [--smoke]
产出:
    reports/p08-aimclr-arm-<date>.json / .md
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

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from psd.data.interpet4d import resample_to_fixed_t, to_ntu_view  # noqa: E402
from run_p07_endtoend_ak import (  # noqa: E402
    PKL,
    SEEDS,
    Y_CKPT,
    load_dataset,
    run_one,
)

AIMCLR_ROOT = REPO / "external" / "AimCLR"
AIMCLR_CKPT = REPO / "runs" / "p01_aimclr_pretext" / "epoch120_model.pt"
P07_REPORT = REPO / "reports" / "p07-endtoend-ak-full12-2026-09-04.json"
OUT_DIR = REPO / "reports"


def build_aimclr_views(kp: np.ndarray) -> np.ndarray:
    """AK (N,30,24,3)[x,y,conf] -> AimCLR NTU 视图 (N,3,64,25,1)。

    与 p01 导出管线逐字同构: 线性重采样 T=64 -> 置信度通道拆出 ->
    to_ntu_view(恒等映射 0-23 + 死槽 24, conf<0.5 置零, 序列归一)。
    """
    views = []
    for i in range(len(kp)):
        kp64 = resample_to_fixed_t(kp[i], target_t=64)          # (64,24,3)
        w64 = kp64[..., 2].copy()                                # (64,24)
        views.append(to_ntu_view(kp64, weight=w64))              # (3,64,25,1)
    return np.stack(views).astype(np.float32)


def extract_aimclr_features(views: np.ndarray, device: str = "cuda") -> np.ndarray:
    """AimCLR encoder_q backbone 特征 (N,256) — 与 eval_aimclr.extract_backbone 逐字一致。"""
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))
    from torchlight import import_class  # noqa: E402

    model_cls = import_class("net.aimclr.AimCLR")
    model = model_cls(
        base_encoder="net.st_gcn.Model", pretrain=True, feature_dim=128,
        queue_size=1024, momentum=0.999, Temperature=0.07, mlp=True,
        in_channels=3, hidden_channels=16, hidden_dim=256, num_class=12,
        dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    )
    sd = torch.load(AIMCLR_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    def extract_backbone(x):
        enc = model.encoder_q
        n, c, t, v, m = x.size()
        h = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
        h = enc.data_bn(h)
        h = h.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        for gcn, imp in zip(enc.st_gcn_networks, enc.edge_importance):
            h, _ = gcn(h, enc.A * imp)
        h = torch.nn.functional.avg_pool2d(h, h.size()[2:])
        h = h.view(n, m, -1).mean(dim=1)
        return h

    feats = []
    with torch.no_grad():
        for i in range(0, len(views), 64):
            batch = torch.from_numpy(views[i:i + 64]).float().to(device)
            feats.append(extract_backbone(batch).cpu().numpy())
    return np.vstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    data, kp, labels, splits = load_dataset()
    labels_str = np.array([str(d["psd_class"]) for d in data])
    class_names = sorted({str(d["psd_class"]) for d in data})
    print(f"[data] {len(data)} clips | train={int((splits=='train').sum())} val={int((splits=='val').sum())}")

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[feat] building AimCLR views + extracting ({device})...")
    views = build_aimclr_views(kp)
    emb = extract_aimclr_features(views, device=device)
    print(f"  aimclr: {emb.shape}")

    runs = []
    for spc in (2, 4, -1):
        for seed in SEEDS:
            r = run_one(emb, labels, labels_str, splits, class_names, "aimclr", spc, seed, args.smoke)
            r["budget"] = "full" if spc < 0 else f"spc{spc}"
            runs.append(r)
            print(f"[P08] aimclr {r['budget']} seed{seed}: top1={r.get('top1')} macroF1={r.get('macro_f1')}")

    def agg(budget):
        sel = [r for r in runs if r.get("spc") == budget and "top1" in r]
        if not sel:
            return None
        t = [r["top1"] for r in sel]; m = [r["macro_f1"] for r in sel]
        return {"n": len(sel), "top1_mean": round(float(np.mean(t)), 4), "top1_std": round(float(np.std(t)), 4),
                "macro_f1_mean": round(float(np.mean(m)), 4), "macro_f1_std": round(float(np.std(m)), 4)}

    # 对照: p07 的 warm/scratch 同协议数字
    p07_ref = {}
    if P07_REPORT.exists():
        p07 = json.loads(P07_REPORT.read_text(encoding="utf-8"))
        p07_ref = p07.get("agg", {})

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "public_real",
        "protocol": "identical to p07 E7/E8 end-to-end pipeline; ONLY backbone init replaced: AimCLR encoder_q (InterPet4D SSL) instead of PSD warm (Y_CKPT AK-22 supervised)",
        "fairness_notes": [
            "each arm uses its native preprocessing (AimCLR: NTU-view + sequence norm; PSD: raw (30,24,3))",
            "AimCLR pretrain source is mocap (domain gap to in-the-wild AK disclosed); this arm is a generic-SSL transfer lower-bound reference, not AimCLR best-tuned",
            "scratch arm numbers quoted from p07 report, not re-run",
        ],
        "config_echo": {"pkl": str(PKL), "aimclr_ckpt": str(AIMCLR_CKPT), "y_ckpt_for_reference": str(Y_CKPT),
                        "seeds": list(SEEDS), "n_clips": len(data), "classes": class_names},
        "runs": runs,
        "agg": {f"aimclr_spc{b}": agg(b) for b in (2, 4, -1)},
        "p07_reference_agg": p07_ref,
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    date = datetime.now().strftime("%Y-%m-%d")
    out_json = OUT_DIR / f"p08-aimclr-arm-{date}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out_json} ({result['wall_clock_sec']}s)")
    for k, v in result["agg"].items():
        if v:
            print(f"  {k}: top1={v['top1_mean']}±{v['top1_std']} macroF1={v['macro_f1_mean']}±{v['macro_f1_std']}")
    print("  p07 ref:", json.dumps(p07_ref, ensure_ascii=False))


if __name__ == "__main__":
    main()
