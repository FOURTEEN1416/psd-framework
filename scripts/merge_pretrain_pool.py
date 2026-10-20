"""合并 InterPet4D + MammalNet 为 AimCLR 预训练池（P1 数据扩容）.

裁决口径（2026-10-20 对抗验收后）:
  - **自监督预训练不使用 label**（AimCLR 对比学习）+ 所以合并后训练本身安全；
  - 但 kNN probe 的评价标签必须**语义一致**，否则指标会被污染。两池 label 语义不同:
        InterPet4D p01 : dog ID（12 类个体）→ 可作为 probe 标签
        MammalNet      : 品种（8 类）      → **不同语义，不可与 dog ID 同空间**
    处理: 合并时给 MammalNet 打 label = -1（probe 时排除），并单独记录其品种索引
    供「跨数据集表征泛化」实验单独使用。绝不把两者塞进同一个 label 空间。
  - 产物落 data/processed/p01_merged/（gitignore）。

用法:
    & "D:\\Desktop\\PR论文\\psd-framework\\.venv\\Scripts\\python.exe" `
        scripts/merge_pretrain_pool.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(pool_dir: Path) -> tuple[np.ndarray, list[str], list[int]]:
    data = np.load(pool_dir / "train_data.npy")
    names, labels = pickle.load(open(pool_dir / "train_label.pkl", "rb"))
    return data, list(names), list(labels)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interpet-dir", default="data/processed/p01")
    ap.add_argument("--mammalnet-dir", default="data/processed/p01_mammalnet")
    ap.add_argument("--out-dir", default="data/processed/p01_merged")
    args = ap.parse_args()

    d1, n1, l1 = _load(REPO_ROOT / args.interpet_dir)
    d2, n2, l2 = _load(REPO_ROOT / args.mammalnet_dir)

    if d1.shape[1:] != d2.shape[1:]:
        raise SystemExit(f"[merge] 张量形状不兼容: {d1.shape} vs {d2.shape}")

    # 标签空间裁决：InterPet4D 的 dog ID 保留；MammalNet 全部标 -1（不参与 probe）
    merged_data = np.concatenate([d1, d2], axis=0).astype(np.float32)
    merged_names = [f"interpet::{n}" for n in n1] + [f"mammalnet::{n}" for n in n2]
    merged_labels = list(l1) + [-1] * len(n2)

    out = REPO_ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_data.npy", merged_data)
    with open(out / "train_label.pkl", "wb") as f:
        pickle.dump((merged_names, merged_labels), f)
    # MammalNet 品种索引单独存（供跨数据集泛化实验，不混入 probe 空间）
    with open(out / "mammalnet_breed.pkl", "wb") as f:
        pickle.dump((list(n2), list(l2)), f)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "interpet4d": {"dir": args.interpet_dir, "n": len(n1), "classes": len(set(l1))},
        "mammalnet": {"dir": args.mammalnet_dir, "n": len(n2), "breeds": len(set(l2))},
        "merged_shape": list(merged_data.shape),
        "merged_n": len(merged_names),
        "label_policy": (
            "InterPet4D dog-ID 保留原索引；MammalNet 一律 -1（不进 kNN probe），"
            "其品种索引单存 mammalnet_breed.pkl。原因：dog ID 与品种是不同语义，"
            "同空间会污染 probe 指标（对抗验收发现#3）"
        ),
        "pretrain_note": "AimCLR 自监督不使用 label，合并池可直接训练；仅 probe 侧需按 label_policy 过滤",
        "ch2_semantics": "InterPet4D 为真实 3D z；MammalNet 为 2D 源、ch2 恒 0（非伪造）",
    }
    rpath = REPO_ROOT / "reports" / f"p1-merged-pretrain-pool-{datetime.now():%Y-%m-%d}.json"
    (REPO_ROOT / "reports").mkdir(exist_ok=True)
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[merge] {len(n1)} + {len(n2)} = {len(merged_names)} 样本")
    print(f"[merge] train_data.npy {tuple(merged_data.shape)} -> {out}")
    print(f"[merge] probe 可用样本(InterPet4D): {sum(1 for x in merged_labels if x >= 0)}")
    print(f"[merge] 报告 -> {rpath}")


if __name__ == "__main__":
    main()
