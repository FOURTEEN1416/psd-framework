"""P0.1 导出脚本：InterPet4D smal_npy → AimCLR NTU 兼容 mmap 格式。

产出（gitignore 内，不入库）：
- data/processed/p01/train_data.npy : (N, 3, T=64, V=25, M=1) float32
- data/processed/p01/train_label.pkl: ([sample_name...], [dog_label...])，13 类

用法：
    python scripts/export_interpet4d.py \
        --data-root "D:/Desktop/k9-training-system/data/interpet4d/smal_npy" \
        --out-dir data/processed/p01 --target-t 64
"""
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.interpet4d import (  # noqa: E402
    build_label_index,
    is_valid_clip,
    load_clip,
    resample_to_fixed_t,
    to_ntu_view,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data-root",
        default=r"D:\Desktop\k9-training-system\data\interpet4d\smal_npy",
    )
    ap.add_argument("--out-dir", default="data/processed/p01")
    ap.add_argument("--target-t", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.data_root)
    clips = sorted(src.glob("*.npz"))
    if not clips:
        raise SystemExit(f"[export] 未找到 .npz：{src}")
    print(f"[export] 发现 {len(clips)} 个 clip，target_t={args.target_t}")

    views, names = [], []
    skipped = []
    for i, path in enumerate(clips, 1):
        clip = load_clip(path)
        if not is_valid_clip(clip["kp_world"]):
            skipped.append(path.stem)
            continue
        kp64 = resample_to_fixed_t(clip["kp_world"], target_t=args.target_t)
        w64 = resample_to_fixed_t(clip["kp_weight"], target_t=args.target_t)
        view = to_ntu_view(kp64, weight=w64)
        views.append(view)
        names.append(path.stem)
        if i % 50 == 0 or i == len(clips):
            print(f"[export] {i}/{len(clips)}")
    if skipped:
        print(f"[export] 剔除无效 clip（非有限值）: {skipped}")

    data = np.stack(views).astype(np.float32)  # (N,3,T,25,1)
    sample_names, labels, num_class = build_label_index(names)

    out_dir = REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "train_data.npy", data)
    with open(out_dir / "train_label.pkl", "wb") as f:
        pickle.dump((sample_names, labels), f)

    print(f"[export] 数据形状: {data.shape} dtype={data.dtype}")
    print(f"[export] 类别数(dog ID): {num_class}，随机基线 = 100/{num_class} ≈ {100/num_class:.2f}%")
    print(f"[export] 有限性检查: {'OK' if np.isfinite(data).all() else 'FAIL'}")
    print(f"[export] 已写出: {out_dir/'train_data.npy'} + train_label.pkl")


if __name__ == "__main__":
    main()
