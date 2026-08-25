# -*- coding: utf-8 -*-
"""W29/C5 dog-pose 静态池入库脚本（收敛契约格式 B）.

任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §0-格式B / §2-C5
C5 结论(2026-08-25): 三选一取 **(b) 仅作预训练/增广池**——
  - 全库无序列分组元数据(inventory-evidence 佐证), 拒绝 (a) 时序构造;
  - 数据完整可用(配对/标注零缺陷), 排除 (c);
  - 以 T=1 静态骨架条目入库, 显式声明 static, 不做任何时序合成。

格式 B 契约字段(每条目):
    {keypoints: (T,V,C), topology_name, V, fps_or_sampling, source, split}
本池特化(T=1):
    keypoints = (1, 24, 3) float32 —— [x_px, y_px, visibility01], 图像像素坐标
    fps_or_sampling = None(静态); 另附 sample_id / n_visible / coords_semantic 注记
打包: 每 split 一个合并式 .pkl(仓库既有合成集同为合并式惯例),
      目录仍满足契约 runs/data_campaign/dogpose/sequences/*.pkl。

诚实声明(manifest 内固化):
    - GT 有效关节 20/24: left_eye/right_eye/withers/throat 全库可见率 0%;
    - 本脚本不做插值/升采样, synthetic_dynamic=false。

用法:
    & D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe \\
        scripts/assess_dogpose_ingest.py \\
        --root "D:\\Desktop\\datasets\\dog-pose" \\
        --out runs/data_campaign/dogpose [--min-visible N]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_FIELDS = 77
NUM_KPTS = 24
SPLITS = ("train", "val")
TOPOLOGY_NAME = "K9Graph"
#: 与 psd/models/stgcn_k9_graph.py 同源; 此处仅存名字符串, 不跨包强依赖
COORDS_SEMANTIC = "image_pixel_xy"
SCHEMA_ID = "psd.data_campaign.format_b.static_v1"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_label(path: Path) -> np.ndarray:
    """解析 YOLO-pose 标注 → (24,3) 归一化 [x,y,v]. 字段数不符即 fail-fast."""
    toks = path.read_text(encoding="utf-8").split()
    if len(toks) != EXPECTED_FIELDS:
        raise ValueError(f"{path.name}: 字段数 {len(toks)} != {EXPECTED_FIELDS}")
    vals = np.asarray(toks, dtype=np.float64)
    if int(vals[0]) != 0:
        raise ValueError(f"{path.name}: class={int(vals[0])} != 0")
    return vals[5:].reshape(NUM_KPTS, 3)


def image_size(path: Path) -> tuple:
    """PIL 只读图片头取尺寸(不解码像素体)."""
    with Image.open(path) as im:
        return im.size


def build_split(root: Path, split: str, min_visible: int):
    """单 split 转换. 返回 (entries 列表, 跳过统计 dict)."""
    img_dir, lbl_dir = root / "images" / split, root / "labels" / split
    lbl_paths = sorted(lbl_dir.glob("*.txt"))
    entries, skipped, vis_joint = [], [], np.zeros(NUM_KPTS, dtype=np.int64)
    for lp in lbl_paths:
        stem = lp.stem
        ip = next(iter([img_dir / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png")
                        if (img_dir / f"{stem}{ext}").exists()]), None)
        if ip is None:
            skipped.append(f"{split}/{stem}: 无对应图片")
            continue
        w_img, h_img = image_size(ip)
        kpts_norm = parse_label(lp)
        kpts = np.empty((NUM_KPTS, 3), dtype=np.float32)
        kpts[:, 0] = kpts_norm[:, 0] * w_img          # 归一化中心坐标 → 像素坐标
        kpts[:, 1] = kpts_norm[:, 1] * h_img
        kpts[:, 2] = (kpts_norm[:, 2] > 0).astype(np.float32)  # {2.0,0.0} → {1,0}
        n_vis = int(kpts[:, 2].sum())
        vis_joint += kpts[:, 2].astype(np.int64)
        if n_vis < min_visible:
            skipped.append(f"{split}/{stem}: 可见点 {n_vis} < {min_visible}")
            continue
        entries.append({
            # ── 格式 B 契约字段 ──
            "keypoints": kpts.reshape(1, NUM_KPTS, 3),   # (T=1, V=24, C=3)
            "topology_name": TOPOLOGY_NAME,
            "V": NUM_KPTS,
            "fps_or_sampling": None,                     # 静态单帧, 无采样率
            "source": f"dog-pose@{root}",
            "split": split,
            # ── 注记字段(manifest schema 已登记) ──
            "sample_id": stem,
            "n_visible": n_vis,
            "coords_semantic": COORDS_SEMANTIC,
            "static": True,
        })
    stats = {"labels": len(lbl_paths), "ingested": len(entries),
             "skipped": skipped[:30], "n_skipped": len(skipped),
             "joint_visibility": vis_joint.tolist()}
    return entries, stats


def main() -> int:
    ap = argparse.ArgumentParser(description="dog-pose 静态池入库(W29/C5, 格式 B)")
    ap.add_argument("--root", default=r"D:\Desktop\datasets\dog-pose")
    ap.add_argument("--out", default=r"runs\data_campaign\dogpose")
    ap.add_argument("--min-visible", type=int, default=0,
                    help="入池最小可见关键点数(默认 0=全量入池, 过滤交给下游)")
    args = ap.parse_args()

    root, out_dir = Path(args.root), Path(args.out)
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    import pickle

    manifest = {
        "schema": SCHEMA_ID,
        "channel": "dogpose",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "generator": "scripts/assess_dogpose_ingest.py",
        "command": f"scripts/assess_dogpose_ingest.py --root \"{root}\" --out \"{out_dir}\""
                   + (f" --min-visible {args.min_visible}" if args.min_visible else ""),
        "contract": "dev-docs/handovers/DATA-CAMPAIGN-plan.md §0-格式B",
        "verdict": "(b) 仅作预训练/增广池",
        "honesty": {
            "has_sequence_grouping_metadata": False,
            "evidence": "inventory-evidence-2026-08-25.json: 全库唯一元数据文件为 dog-pose.yaml;"
                        " 文件名=synset_照片ID 独立照片; 无帧号/视频ID/时序字段",
            "synthetic_dynamic": False,
            "note": "T=1 静态条目, 未做任何插值/时序升采样",
            "effective_joints": "20/24 —— left_eye/right_eye/withers/throat 全库可见率 0%"
                                "(K9Graph 根关节 withers 在内)",
        },
        "license_note": "ultralytics 打包 AGPL-3.0(dog-pose.yaml 头部); "
                        "图像源自 Stanford Dogs 数据集(学术研究用途, 论文须引用其来源)",
        "splits": {},
        "files": {},
    }

    for split in SPLITS:
        entries, stats = build_split(root, split, args.min_visible)
        out_pkl = seq_dir / f"dogpose_{split}.pkl"
        blob = {"schema": SCHEMA_ID, "channel": "dogpose", "split": split,
                "topology_name": TOPOLOGY_NAME, "entries": entries}
        with open(out_pkl, "wb") as fh:
            pickle.dump(blob, fh, protocol=pickle.HIGHEST_PROTOCOL)

        # ── 写后即验: 重载 + 结构断言(fail-fast) ──
        with open(out_pkl, "rb") as fh:
            back = pickle.load(fh)
        assert len(back["entries"]) == len(entries), f"{split}: 条目数回读不一致"
        e0 = back["entries"][0]
        assert e0["keypoints"].shape == (1, NUM_KPTS, 3), "keypoints 形状契约违反"
        assert set(("keypoints", "topology_name", "V", "fps_or_sampling",
                    "source", "split")).issubset(e0.keys()), "契约字段缺失"

        manifest["splits"][split] = {
            "labels": stats["labels"], "ingested": stats["ingested"],
            "n_skipped": stats["n_skipped"],
            "joint_visibility": stats["joint_visibility"],
            "min_visible_gate": args.min_visible,
        }
        manifest["files"][out_pkl.name] = {"sha256": sha256_file(out_pkl),
                                           "bytes": out_pkl.stat().st_size,
                                           "entries": len(entries)}
        print(f"[{split}] ingested={stats['ingested']} skipped={stats['n_skipped']}")

    stamp = _dt.date.today().isoformat()
    man_path = out_dir / f"manifest-{stamp}.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(man_path),
                      "files": list(manifest["files"].keys())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
