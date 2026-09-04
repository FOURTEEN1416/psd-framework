"""P1.4a NTU 低资源臂 — 冻结 pretext 骨干特征导出（PSD-NTU-PREREG-001 §2）。

官方 Feeder_single（无增强评估口径 shear/padding=-1，与 lineareval 配置逐字一致）
+ epoch300 joint pretext encoder_q 的 fc 前 256d backbone 特征（extract_backbone
与 eval_aimclr/p08 同一实现，M=2 双主体均值）。一次导出 train 40091 + val 16487，
协议层（p14b）只读 npz，不再触碰原始 npy。

用法:
    .venv/Scripts/python.exe scripts/run_p14_ntu_featuredump.py
产出:
    runs/ntu_lowres/features_joint_ep300.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from psd.data.ntu_aimclr_env import resolve_aimclr_root  # noqa: E402

WEIGHTS = REPO / "runs" / "ntu_phaseB" / "joint_pretext" / "epoch300_model.pt"
OUT_NPZ = REPO / "runs" / "ntu_lowres" / "features_joint_ep300.npz"
SPLITS = {
    "train": (REPO / "data/ntu60_frame50/xsub/train_position.npy",
              REPO / "data/ntu60_frame50/xsub/train_label.pkl"),
    "val": (REPO / "data/ntu60_frame50/xsub/val_position.npy",
            REPO / "data/ntu60_frame50/xsub/val_label.pkl"),
}


def build_model(device):
    aimclr_root = resolve_aimclr_root(REPO)
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))
    from torchlight import import_class  # noqa: E402

    model_cls = import_class("net.aimclr.AimCLR")
    model = model_cls(
        base_encoder="net.st_gcn.Model", pretrain=True, feature_dim=128,
        queue_size=32768, momentum=0.999, Temperature=0.07, mlp=True,
        in_channels=3, hidden_channels=16, hidden_dim=256, num_class=60,
        dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    )
    sd = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    return model.to(device).eval()


def extract_backbone(enc, x):
    """st_gcn.Model fc 前全局池化特征 (N,256) — 与 eval_aimclr/p08 逐字一致。"""
    n, c, t, v, m = x.size()
    h = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
    h = enc.data_bn(h)
    h = h.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
    for gcn, imp in zip(enc.st_gcn_networks, enc.edge_importance):
        h, _ = gcn(h, enc.A * imp)
    h = torch.nn.functional.avg_pool2d(h, h.size()[2:])
    h = h.view(n, m, -1).mean(dim=1)
    return h


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device)
    enc = model.encoder_q

    from feeder.ntu_feeder import Feeder_single  # noqa: E402
    import torch.utils.data as tud  # noqa: E402

    store = {}
    t0 = time.time()
    for split, (dp, lp) in SPLITS.items():
        feeder = Feeder_single(str(dp), str(lp), shear_amplitude=-1,
                               temperal_padding_ratio=-1, mmap=True)
        loader = tud.DataLoader(feeder, batch_size=128, shuffle=False, num_workers=0)
        feats, labs = [], []
        for bi, (data, label) in enumerate(loader):
            x = data.float().to(device)
            with torch.no_grad():
                feats.append(extract_backbone(enc, x).cpu().numpy())
            labs.append(np.asarray(label).reshape(-1))
            if (bi + 1) % 50 == 0:
                print(f"  [{split}] batch {bi+1} ({time.time()-t0:.0f}s)", flush=True)
        store[f"{split}_feat"] = np.vstack(feats).astype(np.float32)
        store[f"{split}_label"] = np.concatenate(labs).astype(np.int64)
        print(f"[{split}] feat {store[f'{split}_feat'].shape} labels {store[f'{split}_label'].shape}")

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **store)
    print(f"[done] {OUT_NPZ} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
