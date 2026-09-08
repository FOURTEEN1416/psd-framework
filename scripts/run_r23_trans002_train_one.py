# -*- coding: utf-8 -*-
"""单 seed C′ 训练入口（fixfull 管线子进程用; NaN 看门狗由父管线 log 校验 + 此处自检双保险）。"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "AimCLR"))
sys.path.insert(0, str(REPO / "external" / "AimCLR" / "torchlight"))
sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    from processor.processor import init_seed
    init_seed(args.seed)

    import net.st_gcn as _stgcn_mod
    import torch

    _OriginalSTGCN = _stgcn_mod.Model  # 在替换前捕获原类（防止 shim 自引用递归）

    class _AdapterShim(torch.nn.Module):
        """FT 训练约定 model(None, x)——包 st_gcn.Model; forward 内强制 train 态
        （FT.train() 第一行写死 eval(), 含 BN 模型必须回 train——首跑实证 bug）。"""

        def __init__(self, *a, **k):
            super().__init__()
            self.inner = _OriginalSTGCN(*a, **k)

        def forward(self, _ignored, x):
            self.inner.train()
            return self.inner(x)

    _stgcn_mod.Model = _AdapterShim

    from processor.finetune_evaluation import FT_Processor
    import os

    os.chdir(REPO)
    proc = FT_Processor(["--config", str(Path(args.config).resolve()), "--device", "0"])
    proc.start()

    # 训练后: 加载 best, 自算 top1 输出（最后一行）
    from net.st_gcn import Model as STGCNModel
    from feeder.ntu_feeder import Feeder_single
    import numpy as np

    best = Path(args.config).parent / "best_model.pt"
    if not best.exists():
        cands = sorted(Path(args.config).parent.glob("epoch*.pt"))
        best = cands[-1] if cands else None
    if best is None:
        raise FileNotFoundError("no checkpoint")

    # NaN 看门狗: 日志尾段含 nan 即失败
    log_f = Path(args.config).parent / "log.txt"
    if log_f.exists():
        txt = log_f.read_text(encoding="utf-8", errors="replace")
        tail = txt[txt.rfind("Networks initialized"):] if "Networks initialized" in txt else txt
        if "train_mean_loss: nan" in tail or "eval_mean_loss: nan" in tail:
            raise RuntimeError("NaN collapse detected in this seed")

    ck = torch.load(best, map_location="cpu")
    sd = ck.get("model", ck)
    if any(k.startswith("inner.") for k in sd):
        sd = {k[len("inner."):]: v for k, v in sd.items()}
    model = STGCNModel(in_channels=3, hidden_channels=16, hidden_dim=256, num_class=49,
                       dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
                       edge_importance_weighting=True)
    model.load_state_dict(sd, strict=True)
    model.eval().cuda()

    dpath = REPO / "data" / "ntu60_frame50" / "xsub" / "val_position.npy"
    lpath = REPO / "data" / "ntu60_frame50" / "xsub" / "val_label_yp49.pkl"
    feeder = Feeder_single(str(dpath), str(lpath), shear_amplitude=-1,
                           temperal_padding_ratio=-1, mmap=True)
    loader = torch.utils.data.DataLoader(feeder, batch_size=256, shuffle=False, num_workers=0)
    correct = total = 0
    with torch.no_grad():
        for x, lab in loader:
            pred = model(x.float().cuda()).argmax(1).cpu().numpy()
            correct += int((pred == np.asarray(lab)).sum())
            total += len(lab)
    print(f"{correct / total:.6f}")


if __name__ == "__main__":
    main()
