"""生成 22 类合成骨架集并落盘（支持 samples_per_class 与 output 参数）."""
import argparse
import json
from pathlib import Path
import numpy as np

from psd.data.stgcn_bc_dataset import make_synthetic_dataset, save_synthetic_dataset
from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 22 类合成骨架集")
    ap.add_argument("--samples-per-class", type=int, default=100, help="每类样本数（默认 100）")
    ap.add_argument("--output", type=str, default=None, help="输出路径（默认 data/synthetic/syn_22class_{n}per_class_seed42.pkl）")
    ap.add_argument("--T", type=int, default=30, help="时序长度（默认 30）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = ap.parse_args()

    n = args.samples_per_class
    out_path_str = args.output or f"data/synthetic/syn_22class_{n}per_class_seed{args.seed}.pkl"
    out_path = Path(out_path_str)

    print(f"[gen] samples_per_class={n}, T={args.T}, seed={args.seed}")
    print(f"[gen] output={out_path}")

    samples = make_synthetic_dataset(samples_per_class=n, T=args.T, seed=args.seed)
    save_synthetic_dataset(samples, str(out_path))

    dist = {}
    for s in samples:
        dist[s["label_name"]] = dist.get(s["label_name"], 0) + 1

    total_bytes = sum(s["keypoints"].nbytes for s in samples)
    manifest_entry = {
        "path": str(out_path),
        "total_samples": len(samples),
        "samples_per_class": n,
        "num_classes": len(ALL_BEHAVIORS_22),
        "T": args.T,
        "seed": args.seed,
        "class_distribution": dist,
        "total_bytes": total_bytes,
        "generation_command": f".venv/Scripts/python.exe scripts/gen_synth_22class.py --samples-per-class {n} --output {out_path}",
    }

    manifest_path = Path("data/synthetic/_manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if isinstance(manifest, dict):
            # Convert to list of entries
            manifest = [manifest]
        # Append new entry (avoid duplicates)
        existing_paths = {e.get("path") for e in manifest if isinstance(e, dict)}
        if out_path_str not in existing_paths:
            manifest.append(manifest_entry)
    else:
        manifest = [manifest_entry]

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Total samples: {len(samples)}")
    print(f"Total bytes: {total_bytes / 1e6:.2f} MB")
    print(f"Class dist: {dist}")


if __name__ == "__main__":
    main()
