"""C1 解耦切换成本实验（W19 窗口重写版，TDD 先行）.

场景: 评估标准演化 Y(22类) → Y′(21类, stand+track→locomotion)。
对比两臂的切换成本（标注单元数 + 墙钟时间双维度）:
  - decouple 臂: 加载 Y checkpoint → 剥离 head → 冻结 backbone → 仅重训语义层(head)
  - baseline 臂: 同一份数据从头随机初始化训练完整模型(backbone+head)
两臂使用完全相同的数据集与切分（同 seed 配对），报 mean±std。

用法:
    # 冒烟档（CPU 或抢卡间隙）
    python scripts/run_c1_decouple.py --tier smoke --device cpu
    # 小档 / full 档
    python scripts/run_c1_decouple.py --tier small --device cpu
    python scripts/run_c1_decouple.py --tier full --device auto

成本口径声明:
  - 标注单元数 = 训练样本量（labeled_units_train）；验证样本单列 val_size。
  - 墙钟时间为当次实测，含 GPU 快照（前后各一次）；若与其他任务共享 GPU，
    以快照为干扰证据在报告中注明。std 为总体标准差（ddof=0）。

W15 草稿遗留缺陷已修复: 单臂→双臂、seed 硬编码→参数化、KeyError 打印、
history 字典式访问（EpochMetrics 是 dataclass）、未知标签静默跳过→显式报错、
Y′ 标签表重复定义→统一引用 psd.models.stgcn_bc_constants。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psd.data.synth_stgcn import make_synthetic_dataset  # noqa: E402
from psd.models.stgcn_bc import build_stgcn_bc  # noqa: E402
from psd.models.stgcn_bc_constants import (  # noqa: E402
    ALL_BEHAVIORS_22,
    Y_PRIME_LABEL_NAMES,
)
from psd.training.train_stgcn_bc import STGCNBCTrainer, TrainConfig  # noqa: E402

# ---------------------------------------------------------------------------
# 分类体系常量（truth 单一性: 全部引自 constants 模块）
# ---------------------------------------------------------------------------
Y_LABEL_NAMES: tuple = tuple(ALL_BEHAVIORS_22)
Y_NUM_CLASSES = len(Y_LABEL_NAMES)          # 22
Y_PRIME_NUM_CLASSES = len(Y_PRIME_LABEL_NAMES)  # 21
LOCOMOTION_IDX = list(Y_PRIME_LABEL_NAMES).index("locomotion")  # 2

VALID_ARMS = ("decouple", "baseline")


def build_y_to_yp_map() -> Dict[str, int]:
    """构建 Y(22类) 类名 → Y′(21类) 类下标映射.

    stand/track 合并进 locomotion；其余类一一对应到同名类的下标。
    """
    mapping: Dict[str, int] = {}
    for name in Y_LABEL_NAMES:
        if name in ("stand", "track"):
            mapping[name] = LOCOMOTION_IDX
        elif name in Y_PRIME_LABEL_NAMES:
            mapping[name] = list(Y_PRIME_LABEL_NAMES).index(name)
        else:  # 防御: constants 两表若失配立即暴露
            raise KeyError(f"Y 类 {name!r} 不存在于 Y′ 表中，constants 失配")
    assert len(mapping) == Y_NUM_CLASSES
    return mapping


def map_samples_to_yprime(samples: List[dict]) -> List[dict]:
    """将 Y(22类) 样本映射为 Y′(21类) 标签；keypoints/boundary 原样保留.

    遇到映射表中不存在的标签名直接抛 ValueError（禁止静默跳过——
    静默跳过会让 Y 标签混入 Y′ 训练集造成标签腐蚀）。
    """
    mapping = build_y_to_yp_map()
    mapped: List[dict] = []
    for s in samples:
        new_s = dict(s)
        orig_name = s.get("label_name")
        if orig_name not in mapping:
            raise ValueError(f"未知标签名 {orig_name!r}，无法映射到 Y′")
        new_s["label"] = mapping[orig_name]
        new_s["label_name"] = Y_PRIME_LABEL_NAMES[new_s["label"]]
        mapped.append(new_s)
    return mapped


# ---------------------------------------------------------------------------
# checkpoint 加载与冻结
# ---------------------------------------------------------------------------

@dataclass
class BackboneLoadInfo:
    """加载结果审计信息."""
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)
    loaded_backbone_tensors: int = 0


def load_y_backbone(model: torch.nn.Module, checkpoint_path: str | Path) -> BackboneLoadInfo:
    """从 Y(22类) checkpoint 加载 backbone 权重到 Y′(21类) 模型.

    做法: 从 state_dict 中剥离所有 head.* 键后 strict=False 加载——
    head 保持随机初始化（这正是解耦臂要重训的部分）。
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"]
    stripped = {k: v for k, v in sd.items() if not k.startswith("head.")}
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    info = BackboneLoadInfo(
        missing=list(missing),
        unexpected=list(unexpected),
        loaded_backbone_tensors=sum(1 for k in stripped if k.startswith("backbone.")),
    )
    if info.unexpected:
        raise RuntimeError(f"checkpoint 存在无法对齐的多余键: {info.unexpected[:5]}")
    return info


def freeze_backbone(model: torch.nn.Module) -> None:
    """冻结 backbone 参数并钉死其 BN 统计量.

    两层动作缺一不可:
      1. requires_grad=False —— 阻断梯度;
      2. 补丁实例 train() 强制 backbone 恒为 eval —— BatchNorm 的
         running_mean/running_var 是 buffer 不受 requires_grad 约束，
         若留在 train 模式会在每次前向时漂移，"冻结"就是假的。
    """
    for name, param in model.named_parameters():
        param.requires_grad = not name.startswith("backbone.")

    import types

    orig_train = model.train

    def patched_train(self, mode: bool = True):  # noqa: ARG001 — MethodType 注入 self
        result = orig_train(mode)
        self.backbone.eval()  # backbone BN 统计恒冻结
        return self

    model.train = types.MethodType(patched_train, model)
    model.backbone.eval()  # 构造后立即生效，不等下一次 train() 调用


# ---------------------------------------------------------------------------
# GPU 状态快照（墙钟干扰取证用）
# ---------------------------------------------------------------------------

def snapshot_gpu_state() -> dict:
    """通过 nvidia-smi 抓取 GPU 占用快照；不可用时返回 available=False."""
    try:
        proc = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if proc.returncode != 0:
            return {"available": False, "reason": f"nvidia-smi rc={proc.returncode}"}
        line = proc.stdout.strip().splitlines()[0]
        name, mem_used, mem_total, util = [c.strip() for c in line.split(",")]
        return {
            "available": True,
            "name": name,
            "memory_used_mib": int(mem_used),
            "memory_total_mib": int(mem_total),
            "utilization_pct": int(util),
            "captured_at": datetime.now().isoformat(timespec="seconds"),
        }
    except Exception as exc:  # noqa: BLE001 — 快照失败不应中断实验
        return {"available": False, "reason": repr(exc)}


# ---------------------------------------------------------------------------
# 单臂运行
# ---------------------------------------------------------------------------

def _split_train_val(samples: List[dict], seed: int, val_ratio: float = 0.2):
    """W12 口径切分: rng(seed).permutation 后前 val_ratio 为验证集."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(samples))
    val_n = int(len(samples) * val_ratio)
    val_samples = [samples[i] for i in indices[:val_n]]
    train_samples = [samples[i] for i in indices[val_n:]]
    return train_samples, val_samples


def run_arm(
    arm: str,
    seed: int,
    n_per_class: int,
    epochs: int,
    patience: int = 15,
    device: str = "auto",
    T: int = 30,
    batch_size: int = 32,
    warmup_epochs: int = 5,
    base_channels: int = 64,
    num_stages: int = 10,
    output_dir: str = "",
    checkpoint_path: Optional[str | Path] = None,
) -> dict:
    """跑一臂一次并返回完整成本记录.

    arm="decouple": 需要 checkpoint_path（None 时退化为随机 backbone，
                    仅用于封闭测试；生产 CLI 会强制校验 checkpoint 存在）。
    arm="baseline": 忽略 checkpoint，始终随机初始化全模型。
    """
    if arm not in VALID_ARMS:
        raise ValueError(f"arm 必须是 {VALID_ARMS}，收到 {arm!r}")

    # 可复现性: 数据生成/切分/初始化/shuffle 全部锚定 seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. 数据: 合成 Y 数据 → 映射 Y′ → W12 口径 8:2 切分（两臂同数据同切分）
    raw = make_synthetic_dataset(samples_per_class=n_per_class, T=T, seed=seed)
    samples = map_samples_to_yprime(raw)
    train_samples, val_samples = _split_train_val(samples, seed)

    # 2. 模型
    model = build_stgcn_bc(
        in_channels=3, num_classes=Y_PRIME_NUM_CLASSES,
        base_channels=base_channels, num_stages=num_stages,
    )
    frozen_params = 0
    if arm == "decouple":
        if checkpoint_path is not None:
            info = load_y_backbone(model, checkpoint_path)
            print(f"[c1] ckpt 加载: backbone={info.loaded_backbone_tensors} 张量, "
                  f"missing(head)={len(info.missing)}, unexpected={len(info.unexpected)}")
        else:
            print("[c1] 警告: decouple 臂无 checkpoint，backbone 随机初始化（仅限测试）")
        freeze_backbone(model)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 3. 训练配置（warmup 不得 ≥ epochs，防 cosine T_max ≤ 0）
    safe_warmup = min(warmup_epochs, max(epochs - 1, 0))
    tc = TrainConfig(
        lr=1e-3, weight_decay=1e-4,
        epochs=epochs, batch_size=batch_size, num_workers=0,
        val_interval=1, save_interval=max(epochs, 1),
        lr_scheduler="cosine", warmup_epochs=safe_warmup,
        early_stopping=True, patience=patience,
        use_amp=True, device=device, grad_clip=1.0,
        output_dir=output_dir or f"runs/c1_{arm}_seed{seed}_n{n_per_class}",
    )
    trainer = STGCNBCTrainer(model, train_samples, val_samples, config=tc)

    # 4. 运行 + 成本计时（GPU 快照前后各一次作干扰证据）
    gpu_before = snapshot_gpu_state()
    t0 = time.perf_counter()
    summary = trainer.fit()
    wall_clock_sec = round(time.perf_counter() - t0, 2)
    gpu_after = snapshot_gpu_state()

    record = {
        "arm": arm,
        "seed": seed,
        "n_per_class": n_per_class,
        "taxonomy": "Y_prime",
        "num_classes": Y_PRIME_NUM_CLASSES,
        # ---- 成本维度 1: 标注单元数 ----
        "labeled_units_train": len(train_samples),
        "val_size": len(val_samples),
        "total_labeled_units": len(samples),
        # ---- 成本维度 2: 墙钟时间 ----
        "wall_clock_sec": wall_clock_sec,
        "gpu_state": gpu_before,
        "gpu_state_after": gpu_after,
        # ---- 收敛画像 ----
        "epochs_configured": epochs,
        "epochs_run": summary["total_epochs_trained"],
        "best_epoch": summary["best_epoch"],
        "best_val_acc": round(float(summary["best_val_acc"]), 6),
        "final_val_acc": round(float(summary["final_val_acc"]), 6),
        "final_train_acc": round(float(summary["final_train_acc"]), 6),
        # ---- 参数量审计 ----
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
        "device": str(summary["device"]),
        "use_amp": bool(summary["use_amp"]),
        "output_dir": tc.output_dir,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }

    print(f"[c1] {arm}/seed{seed}: best_val_acc={record['best_val_acc']:.4f} "
          f"@epoch{record['best_epoch']} epochs_run={record['epochs_run']} "
          f"wall={wall_clock_sec}s labeled={record['labeled_units_train']}")
    return record


# ---------------------------------------------------------------------------
# 聚合统计
# ---------------------------------------------------------------------------

_AGG_METRICS = ("wall_clock_sec", "best_val_acc", "epochs_run",
                "labeled_units_train")


def aggregate_runs(runs: List[dict]) -> dict:
    """按臂聚合 mean/std（总体标准差 ddof=0），附两臂对比比率."""
    agg: Dict[str, dict] = {}
    for arm in VALID_ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        if not rs:
            continue
        agg[arm] = {}
        for metric in _AGG_METRICS:
            vals = np.array([float(r[metric]) for r in rs], dtype=float)
            agg[arm][metric] = {
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "values": vals.tolist(),
            }

    if "decouple" in agg and "baseline" in agg:
        d_wc = agg["decouple"]["wall_clock_sec"]["mean"]
        b_wc = agg["baseline"]["wall_clock_sec"]["mean"]
        comparison = {
            "wall_clock_ratio_baseline_over_decouple":
                round(b_wc / d_wc, 4) if d_wc > 0 else None,
            "acc_delta_decouple_minus_baseline":
                round(agg["decouple"]["best_val_acc"]["mean"]
                      - agg["baseline"]["best_val_acc"]["mean"], 6),
            "epochs_ratio_baseline_over_decouple":
                round(agg["baseline"]["epochs_run"]["mean"]
                      / agg["decouple"]["epochs_run"]["mean"], 4)
                if agg["decouple"]["epochs_run"]["mean"] > 0 else None,
        }
        agg["_comparison"] = comparison
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="C1 解耦切换成本实验")
    ap.add_argument("--tier", choices=["smoke", "small", "full"], required=True,
                    help="分档标签: smoke(n10/e3)/small(n30)/full(n100)，仅入档记录")
    ap.add_argument("--n-per-class", type=int, default=None,
                    help="每类样本数（默认按档位: smoke=10 small=30 full=100）")
    ap.add_argument("--epochs", type=int, default=None,
                    help="训练轮数（默认按档位: smoke=3 其余=50）")
    ap.add_argument("--patience", type=int, default=15, help="早停耐心值")
    ap.add_argument("--seeds", type=str, default="42,43,44",
                    help="逗号分隔 seed 列表（默认 42,43,44）")
    ap.add_argument("--arms", type=str, default="decouple,baseline",
                    help="逗号分隔臂列表（默认双臂对照）")
    ap.add_argument("--device", type=str, default="cpu",
                    help="cpu / cuda / auto（默认 cpu，GPU 空闲时可用 auto）")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="Y checkpoint 路径（默认 runs/p05_stgcn_bc_full/best.pt）")
    ap.add_argument("--output-json", type=str, default=None,
                    help="输出 JSON 路径（默认 reports/c1-decouple-cost-<日期>.json）")
    args = ap.parse_args(argv)

    tier_defaults = {"smoke": (10, 3), "small": (30, 50), "full": (100, 50)}
    n_per_class = args.n_per_class or tier_defaults[args.tier][0]
    epochs = args.epochs or tier_defaults[args.tier][1]
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in VALID_ARMS:
            ap.error(f"--arms 含非法臂 {a!r}，合法值 {VALID_ARMS}")

    checkpoint_path = args.checkpoint or str(
        REPO_ROOT / "runs" / "p05_stgcn_bc_full" / "best.pt")
    if "decouple" in arms and not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"decouple 臂需要 Y checkpoint，但文件不存在: {checkpoint_path}"
            "（生产运行禁止静默退化为随机 backbone）")

    out_json = Path(args.output_json or
                    REPO_ROOT / "reports" /
                    f"c1-decouple-cost-{datetime.now():%Y-%m-%d}.json")

    print("=" * 68)
    print(f"[c1] C1 解耦切换成本实验 | tier={args.tier} n={n_per_class} "
          f"epochs={epochs} seeds={seeds} arms={arms} device={args.device}")
    print(f"[c1] checkpoint={checkpoint_path}")
    print("=" * 68)

    runs: List[dict] = []
    for arm in arms:
        for seed in seeds:
            outdir = REPO_ROOT / "runs" / f"c1_{args.tier}" / f"{arm}_seed{seed}"
            rec = run_arm(
                arm=arm, seed=seed, n_per_class=n_per_class, epochs=epochs,
                patience=args.patience, device=args.device,
                output_dir=str(outdir), checkpoint_path=checkpoint_path,
            )
            rec["tier"] = args.tier
            runs.append(rec)

    result = {
        "experiment": "C1 解耦切换成本（Y→Y′ 评估标准演化场景）",
        "date": datetime.now().isoformat(timespec="seconds"),
        "layer": "synthetic",  # 三层指标口径: 本实验属合成层
        "protocol": {
            "tier": args.tier,
            "n_per_class": n_per_class,
            "epochs": epochs,
            "patience": args.patience,
            "seeds": seeds,
            "arms": arms,
            "device_requested": args.device,
            "split": "rng(seed).permutation, val_ratio=0.2（W12 口径）",
            "cost_definition": {
                "labeled_units_train": "训练样本量（标注单元数主口径）",
                "wall_clock_sec": "trainer.fit() 当次实测秒数，含前后 GPU 快照",
                "std_note": "mean±std 的 std 为总体标准差(ddof=0)",
            },
            "checkpoint": checkpoint_path,
        },
        "runs": runs,
        "aggregated": aggregate_runs(runs),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n[c1] 结果 JSON -> {out_json}")

    comp = result["aggregated"].get("_comparison", {})
    if comp:
        print("[c1] ===== 两臂对比 =====")
        print(f"  墙钟比 baseline/decouple : {comp.get('wall_clock_ratio_baseline_over_decouple')}×")
        print(f"  精度差 decouple-baseline : {comp.get('acc_delta_decouple_minus_baseline')}")
        print(f"  epoch 比 baseline/decouple: {comp.get('epochs_ratio_baseline_over_decouple')}")


if __name__ == "__main__":
    main()
