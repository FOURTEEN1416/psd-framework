# -*- coding: utf-8 -*-
"""单 seed C′ 训练入口（fixfull 管线子进程用; NaN 看门狗由父管线 log 校验 + 此处自检双保险）。

v2 断点续训: --resume-dir <work_dir> 时扫描该目录最新检查点, 经 FT_Processor 继承参数
--weights/--start_epoch 续跑（NTU 链同款机制, 恒定 lr 无需恢复调度器状态）。
最终评估点选择改为崩溃安全: 从 epochNNN_accMM.MM 文件名解析取最高 val acc
（best_model.pt 在续跑后可能被 best_result=0 重置覆盖, 不可信）。
"""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "external" / "AimCLR"))
sys.path.insert(0, str(REPO / "external" / "AimCLR" / "torchlight"))
sys.path.insert(0, str(REPO / "scripts"))


def pick_best_ckpt(work_dir: Path):
    """崩溃安全的最终评估点: 文件名 acc 最大者; 回退 best_model.pt / 最后周期检查点。
    返回 (acc_or_None, path); 无任何检查点抛 FileNotFoundError。"""
    cands = []
    for p in work_dir.glob("epoch*_acc*_model.pt"):
        m = re.match(r"epoch(\d+)_acc([\d.]+)_model\.pt", p.name)
        if m:
            cands.append((float(m.group(2)), int(m.group(1)), p))
    if cands:
        acc, _ep, p = max(cands, key=lambda t: (t[0], t[1]))
        return acc, p
    b = work_dir / "best_model.pt"
    if b.exists():
        return None, b
    periodic = sorted(work_dir.glob("epoch[0-9]*_model.pt"))
    if periodic:
        return None, periodic[-1]
    raise FileNotFoundError(f"no checkpoint in {work_dir}")


def pick_resume_ckpt(work_dir: Path):
    """续训起点: 周期检查点(epochNNN_model)与评测检查点(epochNNN_accMM_model)中 epoch 最大者。
    返回 (epoch_int, path) 或 None（无可判 epoch 的检查点, 如仅 best_model.pt）。"""
    pts = []
    for p in work_dir.glob("epoch*_model.pt"):
        m = re.match(r"epoch(\d+)_model\.pt", p.name)
        if m:
            pts.append((int(m.group(1)), p))
            continue
        m = re.match(r"epoch(\d+)_acc[\d.]+_model\.pt", p.name)
        if m:
            pts.append((int(m.group(1)), p))
    if not pts:
        return None
    return max(pts, key=lambda t: t[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--resume-dir", default=None)
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
    ft_args = ["--config", str(Path(args.config).resolve()), "--device", "0"]
    resume_dir = Path(args.resume_dir) if args.resume_dir else None
    if resume_dir is not None:
        rp = pick_resume_ckpt(resume_dir)
        if rp is not None:
            ep, ckpt = rp
            ft_args += ["--start_epoch", str(ep), "--weights", str(ckpt)]
            print(f"[train_one] resuming from epoch {ep}: {ckpt.name}", flush=True)
    proc = FT_Processor(ft_args)
    proc.start()

    # 训练后: 选最高 acc 检查点, 自算 top1 输出（最后一行）
    # 注意: 此处必须用补丁前捕获的 _OriginalSTGCN——from net.st_gcn import Model 拿到的是被替换后的 shim
    from feeder.ntu_feeder import Feeder_single
    import numpy as np

    work_dir = Path(args.config).parent
    _name_acc, best = pick_best_ckpt(work_dir)

    # NaN 看门狗: 日志尾段含 nan 即失败
    log_f = work_dir / "log.txt"
    if log_f.exists():
        txt = log_f.read_text(encoding="utf-8", errors="replace")
        tail = txt[txt.rfind("Networks initialized"):] if "Networks initialized" in txt else txt
        if "train_mean_loss: nan" in tail or "eval_mean_loss: nan" in tail:
            raise RuntimeError("NaN collapse detected in this seed")

    ck = torch.load(best, map_location="cpu")
    sd = ck.get("model", ck)
    # 键位归一: shim 存盘可能带 inner./module. 前缀, 裸 STGCN 都剥掉
    sd = {k.split("module.")[-1]: v for k, v in sd.items()}
    if any(k.startswith("inner.") for k in sd):
        sd = {k[len("inner."):]: v for k, v in sd.items()}
    model = _OriginalSTGCN(in_channels=3, hidden_channels=16, hidden_dim=256, num_class=49,
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
