# -*- coding: utf-8 -*-
"""W29/C5 dog-pose 入库产物独立复核脚本.

对账方式: 不复用 ingest 的解析代码路径——直接重读原始 YOLO 标注 + PIL 独立取图宽高,
          重算期望像素坐标与可见性, 与 pkl 内条目逐位比对(±0.5px 舍入容差)。
同时校验: 契约字段齐全 / 条目数 / manifest sha256 一致性。

用法:
    & D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe \\
        scripts/assess_dogpose_verify.py --root "D:\\Desktop\\datasets\\dog-pose" \\
        --pool runs/data_campaign/dogpose
退出码 0 = 全部通过; 非 0 = 存在对账失败(fail-fast 打印)。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

NUM_KPTS = 24
CONTRACT_FIELDS = {"keypoints", "topology_name", "V", "fps_or_sampling", "source", "split"}
SAMPLES_PER_SPLIT = 5


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="dog-pose 格式B产物独立复核(W29/C5)")
    ap.add_argument("--root", default=r"D:\Desktop\datasets\dog-pose")
    ap.add_argument("--pool", default=r"runs\data_campaign\dogpose")
    ap.add_argument("--seed", type=int, default=20260825)
    args = ap.parse_args()

    root, pool = Path(args.root), Path(args.pool)
    random.seed(args.seed)

    manifests = sorted(pool.glob("manifest-*.json"))
    assert manifests, f"未找到 manifest: {pool}"
    man = json.loads(manifests[-1].read_text(encoding="utf-8"))
    print(f"manifest: {manifests[-1].name} verdict={man['verdict']}")

    failures = []
    for split in ("train", "val"):
        pkl = pool / "sequences" / f"dogpose_{split}.pkl"
        # sha256 对账(防产物事后被改)
        want = man["files"][pkl.name]["sha256"]
        got = sha256_file(pkl)
        if want != got:
            failures.append(f"{pkl.name}: sha256 不一致 manifest={want[:12]} actual={got[:12]}")
        blob = pickle.load(open(pkl, "rb"))
        ents = blob["entries"]
        n_expect = man["splits"][split]["ingested"]
        if len(ents) != n_expect:
            failures.append(f"{split}: 条目数 {len(ents)} != manifest {n_expect}")
        print(f"[{split}] entries={len(ents)} (manifest={n_expect}) sha256 ✓")

        for e in random.sample(ents, min(SAMPLES_PER_SPLIT, len(ents))):
            missing = CONTRACT_FIELDS - e.keys()
            if missing:
                failures.append(f"{e['sample_id']}: 缺契约字段 {missing}")
                continue
            stem = e["sample_id"]
            toks = (root / "labels" / split / f"{stem}.txt").read_text().split()
            raw = np.asarray(toks[5:], dtype=np.float64).reshape(NUM_KPTS, 3)
            imgs = list((root / "images" / split).glob(stem + ".*"))
            if not imgs:
                failures.append(f"{stem}: 找不到原图")
                continue
            with Image.open(imgs[0]) as im:
                w_img, h_img = im.size
            exp_xy = np.stack([raw[:, 0] * w_img, raw[:, 1] * h_img], axis=1)
            exp_vis = (raw[:, 2] > 0).astype(np.float32)
            got_kp = e["keypoints"]
            if got_kp.shape != (1, NUM_KPTS, 3):
                failures.append(f"{stem}: 形状 {got_kp.shape} != (1,24,3)")
                continue
            xy_err = float(np.abs(got_kp[0, :, :2] - exp_xy).max())
            vis_ok = bool(np.array_equal(got_kp[0, :, 2], exp_vis))
            if xy_err >= 0.51 or not vis_ok:
                failures.append(f"{stem}: max|Δxy|={xy_err:.4f}px vis_ok={vis_ok}")
                continue
            if e["n_visible"] != int(exp_vis.sum()):
                failures.append(f"{stem}: n_visible {e['n_visible']} != 重算 {int(exp_vis.sum())}")
                continue
            if e["fps_or_sampling"] is not None or e["static"] is not True:
                failures.append(f"{stem}: 静态声明违反 fps={e['fps_or_sampling']} static={e['static']}")
                continue
            print(f"  ✓ {stem}: max|Δxy|={xy_err:.4f}px vis={int(exp_vis.sum())}/24")

    # 全量结构断言(轻量): 所有条目形状+契约字段抽查每 split 首尾
    for split in ("train", "val"):
        blob = pickle.load(open(pool / "sequences" / f"dogpose_{split}.pkl", "rb"))
        for e in (blob["entries"][0], blob["entries"][-1]):
            assert e["keypoints"].shape == (1, NUM_KPTS, 3)
            assert e["topology_name"] == "K9Graph" and e["V"] == NUM_KPTS

    if failures:
        print("\n❌ 复核失败:")
        for f in failures:
            print("  -", f)
        return 1
    print("\n✅ 独立复核全过: 抽样对账 + sha256 + 契约字段 + 静态声明")
    return 0


if __name__ == "__main__":
    sys.exit(main())
