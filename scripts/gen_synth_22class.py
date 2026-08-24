"""生成 22 类合成骨架集并落盘."""
import json
from pathlib import Path
import numpy as np

from psd.data.stgcn_bc_dataset import make_synthetic_dataset, save_synthetic_dataset
from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22

samples = make_synthetic_dataset(samples_per_class=20, T=30, seed=42)
out_path = Path("data/synthetic/syn_22class_20per_class_seed42.pkl")
save_synthetic_dataset(samples, str(out_path))

dist = {}
for s in samples:
    dist[s["label_name"]] = dist.get(s["label_name"], 0) + 1

total_bytes = sum(s["keypoints"].nbytes for s in samples)
manifest = {
    "path": str(out_path),
    "total_samples": len(samples),
    "samples_per_class": 20,
    "num_classes": len(ALL_BEHAVIORS_22),
    "T": 30,
    "seed": 42,
    "class_distribution": dist,
    "total_bytes": total_bytes,
    "generation_command": (
        '.venv/Scripts/python.exe -c '
        '"from psd.data.stgcn_bc_dataset import make_synthetic_dataset, save_synthetic_dataset; '
        'save_synthetic_dataset(make_synthetic_dataset(samples_per_class=20, T=30, seed=42), '
        '"data/synthetic/syn_22class_20per_class_seed42.pkl")"'
    ),
}
with open("data/synthetic/_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"Total samples: {len(samples)}")
print(f"Total bytes: {total_bytes / 1e6:.2f} MB")
print(f"Class dist: {dist}")
