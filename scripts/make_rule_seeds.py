"""W6 入口：规则引擎粗标 — 物理层先验种子生成。

用法:
    python scripts/make_rule_seeds.py --config configs/rule_seeds.yaml

流程: 遍历 smal_npy/*.npz → 有效性过滤(全 NaN clip 剔除, 同 P0.1 口径)
      → generate_seeds() → 落盘 NPZ + 汇总 JSON + QC 抽样 CSV。
产出均为 data/seeds/ 下生成物（gitignore）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from psd.data.rule_seeds import generate_seeds  # noqa: E402


def _load_clip(path: Path) -> dict | None:
    """加载单 clip；与 W3 加载器同口径。返回 None 表示无效（缺键/全 NaN）。"""
    try:
        with np.load(path) as npz:
            kp = np.ascontiguousarray(npz["kp_world"], dtype=np.float64)
            weight = np.ascontiguousarray(npz["kp_weight"], dtype=np.float64)
            frame_idx = np.asarray(npz["frame_idx"])
    except (KeyError, OSError) as exc:
        print(f"  [跳过] {path.name}: {exc}")
        return None
    if not np.isfinite(kp).any():
        return None
    return {"kp": kp, "weight": weight, "frame_idx": frame_idx}


def _frame_class_stats(labels: np.ndarray, classes: list[str]) -> dict[str, float]:
    n = len(labels)
    out: dict[str, float] = {}
    for c in classes + ["unknown"]:
        cnt = int((labels == c).sum())
        out[f"frames_{c}"] = cnt / n if n else 0.0
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="YAML 配置路径")
    args = ap.parse_args(argv)

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    src_dir = Path(cfg["data"]["smal_npy_dir"])
    out_dir = ROOT / cfg["data"]["output_dir"]
    seeds_dir = out_dir / "rule_seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)
    classes = list(cfg["classes"])

    files = sorted(src_dir.glob("*.npz"))
    print(f"[1/3] 发现 {len(files)} 个 smal_npy clip")

    # NPZ 字段宽度由内容派生，杜绝静默截断（U16/U64 教训：类名/规则联合串超宽即截）
    label_w = max(max(len(c) for c in classes), len("unknown")) + 1
    rule_w = 128

    summary_rows: list[dict] = []
    skipped: list[str] = []
    for i, path in enumerate(files, 1):
        clip = _load_clip(path)
        if clip is None:
            skipped.append(path.stem)
            continue
        # frame_idx 单调性守卫：非单调差分会产生负 dt → 负速度 → 步态漏判
        if not np.all(np.diff(clip["frame_idx"].astype(np.float64)) >= 0):
            print(f"  [跳过] {path.name}: frame_idx 非单调")
            skipped.append(path.stem)
            continue
        res = generate_seeds(clip["kp"], clip["weight"], clip["frame_idx"], cfg)
        labels = res["frame_labels"].astype(str)

        # 单 clip NPZ：帧级标签/置信度 + 种子段表
        seg_arr = np.array(
            [(s["start_frame"], s["end_frame"], s["label"], s["confidence"],
              "|".join(s["rule_ids"])) for s in res["segments"]],
            dtype=[("start", "i8"), ("end", "i8"), ("label", f"U{label_w}"),
                   ("conf", "f4"), ("rules", f"U{rule_w}")],
        )
        np.savez_compressed(
            seeds_dir / f"{path.stem}.npz",
            frame_labels=labels.astype(f"U{label_w}"),
            frame_confidence=res["frame_confidence"],
            segments=seg_arr,
            body_scale=np.float64(res["body_scale"]),
            ground_height=np.float64(res["ground_height"]),
        )

        conf = res["frame_confidence"].astype(np.float64)
        row = {
            "clip": path.stem,
            "n_frames": len(labels),
            "n_segments": len(res["segments"]),
            "mean_conf": float(conf.mean()) if len(conf) else 0.0,
            "body_scale_m": round(float(res["body_scale"]), 4),
        }
        row.update(_frame_class_stats(labels, classes))
        summary_rows.append(row)
        if i % 50 == 0 or i == len(files):
            print(f"  进度 {i}/{len(files)}")

    print(f"[2/3] 有效 clip {len(summary_rows)}，跳过 {len(skipped)}")

    # 汇总 JSON（config_echo 回显完整配置快照，保障学术复现溯源）
    agg = {
        "config_echo": cfg,
        "n_clips_total": len(files),
        "n_clips_valid": len(summary_rows),
        "skipped_clips": skipped,
        "per_clip": summary_rows,
    }
    cls_totals = {c: float(np.mean([r.get(f"frames_{c}", 0.0) for r in summary_rows]))
                  for c in classes + ["unknown"]} if summary_rows else {}
    agg["dataset_frame_share"] = {k: round(v, 4) for k, v in cls_totals.items()}
    summary_path = out_dir / "seed_summary.json"
    summary_path.write_text(json.dumps(agg, ensure_ascii=False, indent=2), encoding="utf-8")

    # QC 抽样 CSV（≥30）
    rng = np.random.default_rng(int(cfg["data"].get("qc_seed", 42)))
    n_sample = min(int(cfg["data"].get("qc_sample_n", 40)), len(summary_rows))
    idx = rng.choice(len(summary_rows), size=n_sample, replace=False) if n_sample else []
    sample_rows = [summary_rows[j] for j in sorted(idx)]
    csv_path = out_dir / "qc_sample.csv"
    header = list(sample_rows[0].keys()) if sample_rows else ["clip"]
    lines = [",".join(header)]
    for r in sample_rows:
        lines.append(",".join(str(r[h]) for h in header))
    csv_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[3/3] 完成: {seeds_dir} | {summary_path} | {csv_path}")
    if cls_totals:
        share_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(cls_totals.items(),
                                                              key=lambda kv: -kv[1]))
        print(f"数据集帧占比: {share_str}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
