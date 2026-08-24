"""W9 Phase B — AimCLR NTU60 xsub 复现训练入口（排程任务书引用的执行件）。

薄适配器：复用 psd/training/p01_processor.P01AimCLRProcessor（继承官方
AimCLR_Processor，唯一差异=跳过坍缩根因 weights_init，证据见 reports/p01 报告）。
默认配置 configs/ntu60_phaseb_xsub_joint.yaml 改编自官方 pretext yaml，
仅路径/device 适配本机，batch/lr 保持官方保真口径，差异清单见配置头注释。

用法：
    python scripts/run_ntu_phaseb.py                     # 全量预训练（300ep, joint 流）
    python scripts/run_ntu_phaseb.py --num_epoch 1      # 冒烟运行（参数透传）
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
        default=str(REPO_ROOT / "configs" / "ntu60_phaseb_xsub_joint.yaml"),
    )
    args, forward = ap.parse_known_args()

    if not AIMCLR_ROOT.exists():
        raise SystemExit(f"[ntu-phaseb] external/AimCLR 缺失：{AIMCLR_ROOT}")
    _data_dir = REPO_ROOT / "data" / "ntu60_frame50" / "xsub"
    if not _data_dir.exists():
        raise SystemExit(f"[ntu-phaseb] NTU 数据缺失：{_data_dir}（先完成 Phase A 数据获取）")

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
