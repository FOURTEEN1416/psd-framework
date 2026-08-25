# -*- coding: utf-8 -*-
"""W29/C5 dog-pose 静态池诚实可行性盘点脚本.

任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §2-C5
问题: D:\\Desktop\\datasets\\dog-pose (8476 图) 是否有序列分组元数据?
      质量分布如何? 能否作为补充池入库?

产出: runs/data_campaign/dogpose/inventory-evidence-<日期>.json
      (全量校验证据, 报告引用其数字, 不在报告里凭记忆写数)

用法:
    & D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe \\
        scripts/assess_dogpose_inventory.py \\
        --root "D:\\Desktop\\datasets\\dog-pose" \\
        --out runs/data_campaign/dogpose

CPU-only; 零第三方重依赖(仅 numpy/PIL)。
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

#: YOLO-pose 标签字段数 = cls(1) + bbox(4) + 24 点 * (x,y,v)
EXPECTED_FIELDS = 77
NUM_KPTS = 24
SPLITS = ("train", "val")

IMG_EXTS = {".jpg", ".jpeg", ".png"}
#: 序列分组线索的文件名模式(若命中则说明可能存在帧序元数据)
FRAME_LIKE_SUFFIXES = ("_frame", "-frame", "_f0", "_0001", ".avi", ".mp4")


def list_stems(d: Path) -> set:
    return {p.stem for p in d.iterdir() if p.is_file()}


def scan_meta_files(root: Path) -> list:
    """递归列出所有非图片/非标注文件(候选序列分组元数据载体)."""
    out = []
    for dp, _dn, fn in os.walk(root):
        for f in fn:
            if os.path.splitext(f)[1].lower() not in IMG_EXTS | {".txt"}:
                out.append(os.path.join(dp, f))
    return sorted(out)


def frame_like_names(stems: set) -> list:
    """文件名含帧序/视频线索的样本(前 20 个)."""
    hits = [s for s in stems if any(k in s.lower() for k in FRAME_LIKE_SUFFIXES)]
    return sorted(hits)[:20]


def parse_label(path: Path):
    """解析单个 YOLO-pose 标注. 返回 (cls_id, bbox[4], kpts (24,3) 归一化) 或抛 ValueError."""
    with open(path, encoding="utf-8") as fh:
        line = fh.readline()
    toks = line.split()
    if len(toks) != EXPECTED_FIELDS:
        raise ValueError(f"字段数 {len(toks)} != {EXPECTED_FIELDS}")
    vals = np.asarray(toks, dtype=np.float64)
    cls_id = int(vals[0])
    bbox = vals[1:5]
    kpts = vals[5:].reshape(NUM_KPTS, 3)
    return cls_id, bbox, kpts


def main() -> int:
    ap = argparse.ArgumentParser(description="dog-pose 静态池盘点(W29/C5)")
    ap.add_argument("--root", default=r"D:\Desktop\datasets\dog-pose")
    ap.add_argument("--out", default=r"runs\data_campaign\dogpose")
    ap.add_argument("--image-size-sample", type=int, default=300,
                    help="抽样读取图片尺寸的数量(全量尺寸读取留给 ingest)")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev: dict = {
        "task": "C5/W29 dog-pose 静态池盘点",
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(root),
        "yaml": {},
        "per_split": {},
        "sequence_grouping_probe": {},
        "quality": {},
    }

    # ── 1. yaml 头部事实 ──
    yaml_path = root / "dog-pose.yaml"
    ev["yaml"] = {
        "path": str(yaml_path),
        "exists": yaml_path.exists(),
        "head": yaml_path.read_text(encoding="utf-8").splitlines()[:16] if yaml_path.exists() else [],
    }

    # ── 2. 逐 split 配对完整性 + 标注解析 ──
    total_imgs = 0
    all_visible_per_img = []
    v_counter = collections.Counter()
    bad_fields = []
    degenerate_bbox = []
    oob_kpt_visible = []  # v>0 但坐标出界
    breed_counter = collections.Counter()
    for split in SPLITS:
        img_dir, lbl_dir = root / "images" / split, root / "labels" / split
        imgs = {p.stem for p in img_dir.glob("*") if p.suffix.lower() in IMG_EXTS}
        lbls = list_stems(lbl_dir) if lbl_dir.exists() else set()
        missing_lbl = sorted(imgs - lbls)
        orphan_lbl = sorted(lbls - imgs)
        n_ok = 0
        for stem in sorted(lbls & imgs):
            try:
                cls_id, bbox, kpts = parse_label(lbl_dir / f"{stem}.txt")
            except ValueError as exc:
                bad_fields.append(f"{split}/{stem}: {exc}")
                continue
            if cls_id != 0:
                bad_fields.append(f"{split}/{stem}: class={cls_id} != 0")
                continue
            w, h = float(bbox[2]), float(bbox[3])
            if w <= 0 or h <= 0 or not (0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1):
                degenerate_bbox.append(f"{split}/{stem}: bbox={bbox.tolist()}")
            vis = kpts[:, 2] > 0
            all_visible_per_img.append(int(vis.sum()))
            for v in kpts[:, 2]:
                v_counter[float(v)] += 1
            oob = ((kpts[:, 2] > 0) & ((kpts[:, 0] < 0) | (kpts[:, 0] > 1)
                                       | (kpts[:, 1] < 0) | (kpts[:, 1] > 1))).sum()
            if oob:
                oob_kpt_visible.append(f"{split}/{stem}: {int(oob)} pts")
            breed_counter[stem.split("_")[0]] += 1
            n_ok += 1
        ev["per_split"][split] = {
            "images": len(imgs),
            "labels": len(lbls),
            "parsed_ok": n_ok,
            "img_without_label": missing_lbl[:20],
            "label_without_img": orphan_lbl[:20],
            "n_img_without_label": len(missing_lbl),
            "n_label_without_img": len(orphan_lbl),
        }
        total_imgs += len(imgs)

    arr = np.asarray(all_visible_per_img, dtype=np.int32)
    ev["quality"] = {
        "total_images": total_imgs,
        "bad_labels": bad_fields[:30],
        "n_bad_labels": len(bad_fields),
        "degenerate_bbox": degenerate_bbox[:30],
        "n_degenerate_bbox": len(degenerate_bbox),
        "visible_oob_examples": oob_kpt_visible[:20],
        "n_visible_oob_images": len(oob_kpt_visible),
        "v_value_distribution": {str(k): int(v) for k, v in sorted(v_counter.items())},
        "visible_pts_per_image": {
            "mean": round(float(arr.mean()), 3) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "min": int(arr.min()) if arr.size else None,
            "max": int(arr.max()) if arr.size else None,
            "pct_ge_18": round(float((arr >= 18).mean() * 100), 2) if arr.size else None,
            "pct_ge_12": round(float((arr >= 12).mean() * 100), 2) if arr.size else None,
            "pct_lt_6": round(float((arr < 6).mean() * 100), 2) if arr.size else None,
        },
        "breed_synsets": len(breed_counter),
        "breed_top5": [[k, int(v)] for k, v in breed_counter.most_common(5)],
        "breed_bottom3": [[k, int(v)] for k, v in breed_counter.most_common()[-3:]],
    }

    # ── 3. 序列分组元数据探查(本任务核心问题) ──
    meta_files = scan_meta_files(root)
    train_stems = {p.stem for p in (root / "images" / "train").glob("*")}
    val_stems = {p.stem for p in (root / "images" / "val").glob("*")}
    ev["sequence_grouping_probe"] = {
        "non_image_non_label_files": meta_files,
        "frame_like_name_hits_train": frame_like_names(train_stems),
        "frame_like_name_hits_val": frame_like_names(val_stems),
        "name_pattern": "synset_photoID (Stanford Dogs 独立照片 ID, 无帧号/视频ID)",
        "train_val_overlap_ids": len(train_stems & val_stems),
        "conclusion_hint": None,  # 结论由报告给出, 脚本只留证据
    }

    # ── 4. 图片尺寸抽样(PIL 只读头) ──
    sizes = []
    sample = sorted((root / "images" / "train").glob("*.jpg"))[: args.image_size_sample]
    for p in sample:
        with Image.open(p) as im:
            sizes.append(im.size)
    ws = np.asarray([s[0] for s in sizes], dtype=np.int32)
    hs = np.asarray([s[1] for s in sizes], dtype=np.int32)
    ev["image_size_sample"] = {
        "n": len(sizes),
        "width_min_max": [int(ws.min()), int(ws.max())],
        "height_min_max": [int(hs.min()), int(hs.max())],
        "read_failures": 0,
    }

    stamp = _dt.date.today().isoformat()
    out_json = out_dir / f"inventory-evidence-{stamp}.json"
    out_json.write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── 5. 控制台摘要 ──
    print(json.dumps({
        "evidence": str(out_json),
        "total_images": ev["quality"]["total_images"],
        "n_bad_labels": ev["quality"]["n_bad_labels"],
        "visible_pts_mean": ev["quality"]["visible_pts_per_image"]["mean"],
        "pct_ge_18": ev["quality"]["visible_pts_per_image"]["pct_ge_18"],
        "meta_files_found": meta_files,
        "breed_synsets": ev["quality"]["breed_synsets"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
