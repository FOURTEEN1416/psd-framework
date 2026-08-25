"""W31 tab3 补残「−自监督预训练」消融入口 — 两臂 × 3 seeds × 50ep 对照.

任务书: dev-docs/handovers/NEXT-BATCH-plan.md W31 节
产出:   reports/w31-ablation-pretrain-<日期>.json / .md（full 档）
        reports/w31-ablation-pretrain-smoke-<日期>.json / .md（--smoke CPU 冒烟）

用法:
    # CPU 冒烟（本窗交付物——tiny 档全链路证据，不占 GPU）
    python scripts/run_ablation_pretrain.py --smoke

    # full 档（GPU，排队位于 relay ALL_DONE 之后，由协调者/用户触发）
    python scripts/run_ablation_pretrain.py

GPU 纪律: relay 队列未清空（runs/relay_exec/state.json 非 ALL_DONE）时禁止 full 档点火。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.training.ablation_pretrain import run_ablation  # noqa: E402

SMOKE_OVERRIDES = {
    "data": {"samples_per_class": 2, "T": 30, "seed": 42, "val_split": 0.2},
    "train": {
        "lr": 0.001,
        "weight_decay": 0.0001,
        "epochs": 1,
        "batch_size": 4,
        "warmup_epochs": 0,
        "device": "cpu",
        "early_stopping": False,
        "patience": 5,
        "output_dir": "runs/w31_ablation_pretrain_smoke",
    },
    "ablation": {"arms": ["scratch", "warm"], "seeds": [0]},
}


def _deep_update(base: dict, patch: dict) -> dict:
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def write_reports(result: dict, smoke: bool, cfg_path: str) -> tuple[Path, Path]:
    """结果 JSON + MD 双格式归档 reports/（三层口径披露）。"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    tag = "w31-ablation-pretrain-smoke" if smoke else "w31-ablation-pretrain"
    json_path = REPO_ROOT / "reports" / f"{tag}-{date_str}.json"
    md_path = REPO_ROOT / "reports" / f"{tag}-{date_str}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "window": "W31",
        "task": "tab3 补残之一：−自监督预训练消融（scratch vs warm-init）",
        "metric_layer": "synthetic",       # 合成层口径
        "init_provenance_layer": "public_real",  # warm 臂权重来源=公开真实层 P0.1 InterPet4D
        "config_path": cfg_path,
        **result,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    agg = result.get("aggregate", {})
    data_info = result.get("data", {})
    rows = "\n".join(
        f"| {arm} | {stat['best_val_acc_mean']:.4f} | {stat['best_val_acc_std']:.4f} "
        f"| {stat['n_seeds']} | {stat['per_seed']} |"
        for arm, stat in agg.items()
    )
    mode = "CPU 冒烟（tiny 档，仅验证管线，非科学结论）" if smoke else "full（GPU）"
    md = f"""# W31 tab3「−自监督预训练」消融{'（冒烟）' if smoke else ''}

> 日期: {date_str} | 模式: {mode} | 指标口径: **合成层**（val_acc, 22 类）
> warm 臂初始权重来源: 公开真实层（P0.1 AimCLR @ InterPet4D, epoch120_model.pt）——两层口径分别披露

## 设计

- 数据: 合成 {data_info.get('samples_per_class', '?')}×22={data_info.get('samples_per_class', 0) * 22 if data_info.get('samples_per_class') else '?'} 样本（W12 口径, T=30, seed=42, 切分 8:2）
- 两臂: scratch（随机初始化）vs warm（加载 P0.1 encoder_q 权重）；同 seed 下头初始化逐位相等、切分与洗牌一致，唯一差异 = encoder 初始权重
- 训练: STGCNBCTrainer 复用, 等预算（早停关闭）

## 结果（best_val_acc, mean±std over seeds）

| 臂 | mean | std | n_seeds | per_seed |
|----|------|-----|---------|----------|
{rows}

## 判读

- 方向判定: Δ = mean(warm) − mean(scratch)
{"- ⚠️ 冒烟档数字无科学意义——只证明管线端到端可跑通；full 档待 relay ALL_DONE 后点火。" if smoke else "- 结论判读归 W31 报告正式版。"}

## 复现命令

```bash
# 冒烟（CPU）
python scripts/run_ablation_pretrain.py --smoke
# full（GPU, 排 relay 之后）
python scripts/run_ablation_pretrain.py --config configs/ablation_pretrain.yaml
```
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(REPO_ROOT / "configs" / "ablation_pretrain.yaml"))
    ap.add_argument("--smoke", action="store_true",
                    help="CPU 冒烟档：tiny 数据/1ep/单 seed/强制 cpu")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.smoke:
        _deep_update(cfg, SMOKE_OVERRIDES)

    # ckpt 前置存在性校验（fail-fast, 防 full 档空跑到 warm 臂才炸）
    from psd.training.ablation_pretrain import resolve_ckpt_path

    ckpt = resolve_ckpt_path(cfg["pretrained"]["ckpt"])
    if not ckpt.exists():
        raise SystemExit(f"[w31] 预训练 ckpt 缺失: {ckpt}（worktree 需从主检出复制）")

    print(f"[w31] config={args.config} smoke={args.smoke}")
    print(f"[w31] pretrained={ckpt}")
    t0 = datetime.now()
    result = run_ablation(cfg)
    elapsed = (datetime.now() - t0).total_seconds()

    json_path, md_path = write_reports(result, args.smoke, args.config)
    print(f"\n[w31] 耗时 {elapsed:.1f}s")
    print(f"[w31] JSON -> {json_path}")
    print(f"[w31] MD   -> {md_path}")

    for arm, stat in result["aggregate"].items():
        print(f"[w31] {arm}: best_val_acc = "
              f"{stat['best_val_acc_mean']:.4f} ± {stat['best_val_acc_std']:.4f}")


if __name__ == "__main__":
    main()
