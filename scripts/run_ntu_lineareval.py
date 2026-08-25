"""W33 — NTU60 xsub 线性评估入口（官方 LE_Processor 直连，零适配保真）。

与 run_ntu_phaseb.py 的关键差异: 线性评估必须保留官方 weights_init——
LE_Processor.load_model 先对全模型 N(0,0.02) 初始化、随后 load_weights 用
pretext checkpoint 覆盖（ignore_weights=[encoder_q.fc, encoder_k, queue] 前缀
过滤），fc 分类头保留随机初始化属官方 released-model 复测配方（79.18% 出自该
链路）。P01 预训练期跳过 weights_init 的坍缩修复适配在此不适用、不可复用。

用法:
    python scripts/run_ntu_lineareval.py                       # joint 流（默认）
    python scripts/run_ntu_lineareval.py --config configs/ntu60_phaseb_lineareval_xsub_bone.yaml
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.ntu_aimclr_env import resolve_aimclr_root  # noqa: E402


def preflight(config_path: Path, repo_root: Path | None = None) -> list[str]:
    """重依赖导入前的 fail-fast 校验，返回问题描述列表（空=通过）。"""
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    if not config_path.exists():
        return [f"配置不存在: {config_path}"]

    problems: list[str] = []
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    weights = cfg.get("weights")
    if not weights:
        problems.append("配置缺 weights 字段（线性评估必须挂 pretext checkpoint）")
    elif not (repo_root / weights).exists():
        problems.append(f"pretext checkpoint 不存在: {weights}")

    for key in ("train_feeder_args", "test_feeder_args"):
        fa = cfg.get(key) or {}
        for pk in ("data_path", "label_path"):
            rel = fa.get(pk)
            if not rel:
                problems.append(f"{key}.{pk} 缺失")
            elif not (repo_root / rel).exists():
                problems.append(f"{key}.{pk} 不存在: {rel}")

    try:
        resolve_aimclr_root(repo_root)
    except FileNotFoundError as exc:
        problems.append(str(exc).splitlines()[0])

    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs" / "ntu60_phaseb_lineareval_xsub_joint.yaml"),
    )
    args, forward = ap.parse_known_args()

    config_path = Path(args.config).resolve()
    problems = preflight(config_path)
    if problems:
        print("[ntu-lineareval] 预检未通过:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        raise SystemExit(2)

    # AimCLR 内嵌 torchlight 布局注入；external/ 只读消费（gitignore，回退主检出）
    aimclr_root = resolve_aimclr_root(REPO_ROOT)
    sys.path.insert(0, str(aimclr_root))
    sys.path.insert(0, str(aimclr_root / "torchlight"))

    from processor.processor import init_seed  # noqa: E402

    init_seed(0)  # 本仓锚定种子；官方未全局定种（已声明的本地差异，数值同分布）

    from processor.linear_evaluation import LE_Processor  # noqa: E402

    argv = ["--config", str(config_path)] + forward
    proc = LE_Processor(argv)

    # 官方 Processor 自带 work_dir 创建与日志；cwd 需为 repo root 以解析相对路径
    os.chdir(REPO_ROOT)
    proc.start()


if __name__ == "__main__":
    main()
