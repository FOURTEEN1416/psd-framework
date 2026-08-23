"""P0.1 AimCLR 预训练入口 — external/AimCLR 官方链路的薄适配器。

不复制官方训练逻辑：实例化本仓 psd/training/p01_processor.P01AimCLRProcessor
（继承官方 AimCLR_Processor，仅跳过坍缩根因 weights_init）。与官方 main.py 的差异：
1. 不执行 save_src 源码打包（可复现性由本仓 config + 报告命令序列承担）；
2. 默认 config 指向本仓 configs/p01_aimclr.yaml。

用法：
    python scripts/train_aimclr.py                       # 全量预训练
    python scripts/train_aimclr.py --num_epoch 2         # 冒烟运行（参数透传）
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AIMCLR_ROOT = REPO_ROOT / "external" / "AimCLR"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "p01_aimclr.yaml"),
    )
    args, forward = ap.parse_known_args()

    if not AIMCLR_ROOT.exists():
        raise SystemExit(f"[train] external/AimCLR 缺失：{AIMCLR_ROOT}")

    # AimCLR 的 torchlight 是内嵌 setup.py 布局（真包位于 torchlight/torchlight/），
    # 注入其父目录即可导入，不改动 external/ 内部任何文件
    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))
    from processor.processor import init_seed  # noqa: E402

    init_seed(0)
    # 本仓适配处理器：跳过官方 weights_init（坍缩根因，证据见 reports/p01 报告）
    sys.path.insert(0, str(REPO_ROOT))
    from psd.training.p01_processor import P01AimCLRProcessor

    argv = ["--config", str(Path(args.config).resolve())] + forward
    proc = P01AimCLRProcessor(argv)

    # 官方 Processor 自带 work_dir 创建与日志；cwd 需为 repo root 以解析相对路径
    import os

    os.chdir(REPO_ROOT)
    proc.start()


if __name__ == "__main__":
    main()
