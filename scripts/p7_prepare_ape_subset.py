# -*- coding: utf-8 -*-
"""P7 前置：从 AP-10K(LibreYOLO 17kpt) 筛灵长类子集构建 ape 微调集。

背景: DLC snapshot-60000 是 TF checkpoint, DLC 3.0.1(pytorch) 无法加载且无权重
转换器; 改走与 P2' 同栈的 YOLO11x-pose 微调路线。AP-10K 54 物种共享 17kpt 四足
骨架, 筛 Hominidae(黑猩猩/大猩猩/红毛猩猩) + Cercopithecidae(猕猴) 作 PanAf
(黑猩猩/大猩猩野外视频) 的域内提取器; class 重映射为 0..n-1。

用法: .venv/Scripts/python.exe scripts/p7_prepare_ape_subset.py
产出: data/ap10k/ape-pose/{images,labels}/{train,val} + ape-pose.yaml
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ZIP = REPO / "data" / "ap10k" / "ap10k-pose.zip"
SRC = REPO / "data" / "ap10k" / "ap10k-pose"
OUT = REPO / "data" / "ap10k" / "ape-pose"

# 目标物种(小写子串匹配): 类人猿 + 旧世界猴(形态最接近 PanAf 的黑猩猩/大猩猩)
TARGET_SPECIES = ["chimpanzee", "gorilla", "orangutan", "macaque"]


def main():
    if not (OUT / "ape-pose.yaml").exists():
        if SRC.exists() and (SRC / "ap10k-pose.yaml").exists():
            print("[p7] ap10k-pose already extracted")
        else:
            print(f"[p7] extracting {ZIP.name} ({ZIP.stat().st_size/1e9:.2f} GB)...")
            with zipfile.ZipFile(ZIP) as z:
                z.extractall(SRC.parent)
        base = yaml.safe_load((SRC / "ap10k-pose.yaml").read_text(encoding="utf-8"))
        names: dict[int, str] = base["names"]
        tgt = {i: n for i, n in names.items()
               if any(t in n.lower() for t in TARGET_SPECIES)}
        remap = {old: new for new, old in enumerate(sorted(tgt))}
        print(f"[p7] target species: { {tgt[o]: remap[o] for o in sorted(remap)} }")

        for split in ("train", "val"):
            (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
            (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
            n_img = n_inst = 0
            for lf in sorted((SRC / "labels" / split).glob("*.txt")):
                lines = lf.read_text().strip().splitlines()
                keep = []
                for ln in lines:
                    p = ln.split()
                    if int(p[0]) in remap:
                        p[0] = str(remap[int(p[0])])
                        keep.append(" ".join(p))
                if not keep:
                    continue
                img = SRC / "images" / split / (lf.stem + ".jpg")
                if not img.exists():
                    continue
                shutil.copy2(img, OUT / "images" / split / img.name)
                (OUT / "labels" / split / lf.name).write_text(
                    "\n".join(keep) + "\n", encoding="utf-8")
                n_img += 1
                n_inst += len(keep)
            print(f"[p7] {split}: {n_img} images / {n_inst} ape instances")

        out_yaml = {
            "path": str(OUT),
            "train": "images/train",
            "val": "images/val",
            "kpt_shape": base["kpt_shape"],
            "flip_idx": base["flip_idx"],
            "skeleton": base.get("skeleton", []),
            "oks_sigmas": base.get("oks_sigmas"),
            "names": {new: tgt[old] for old, new in remap.items()},
        }
        (OUT / "ape-pose.yaml").write_text(
            yaml.safe_dump(out_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"[p7] wrote {OUT / 'ape-pose.yaml'} (kpt_shape={base['kpt_shape']})")
    print("[p7] ape subset ready")


if __name__ == "__main__":
    main()
