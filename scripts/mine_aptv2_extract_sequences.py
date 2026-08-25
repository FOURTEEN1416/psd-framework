#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""W26 / C2 后续①：APTv2 canidae 关键点序列抽取器（Format B 装弹）。

任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §0 格式 B + §2-C2（领地内顺延）
上游  : scripts/mine_aptv2_inventory.py 的盘点结论（15 帧微序列池，canidae 326 组/646 轨迹）
产出  : runs/data_campaign/aptv2/sequences/<scope>/*.pkl —— 每条一个序列:
          {keypoints: (T,V,C), topology_name, V, fps_or_sampling, source, split, ...}
        runs/data_campaign/aptv2/sequences/_manifest.json —— 全量索引与质量统计

设计要点（诚实口径）:
  - 按 (video_id, track_id) 分组 = 单主体时序；帧序取自文件名数字主干
  - 连续段在"已标注帧"上定义；跨 split 同轨迹自动合并并标记 mixed
  - 默认只出整长窗（--window 15，滑步=窗长不重叠），短段数量如实登记不静默丢弃
  - 置信通道: APTv2 v-flag ∈{0,2} → 线性映射 /2.0 ∈{0.0,1.0}（manifest 注明，原始 flag 可由 source 回溯）
  - num_keypoints 字段已证不可信（inventory §5），可见率一律按 c3>0 现算
  - CPU-only；K9 仓零写入；输出确定性排序可复跑

用法:
  & "D:\Desktop\psd-framework\.venv\Scripts\python.exe" scripts/mine_aptv2_extract_sequences.py
可选: --species all | --window 15 | --stride 15 | --quick N（只处理前 N 个轨迹做冒烟）
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K9_DATA = Path(r"D:\Desktop\k9-training-system\data")
MAIN_SPLITS = ["train_annotations.json", "val_annotations.json", "test_annotations.json"]
CANIDAE = ("dog", "fox", "wolf")
TOPOLOGY_NAME = "aptv2_quadruped_17kp"


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def contiguous_runs(frames: list[int]) -> list[list[int]]:
    """把升序帧号列表切成连续段。"""
    runs, cur = [], []
    for f in frames:
        if cur and f == cur[-1] + 1:
            cur.append(f)
        else:
            if cur:
                runs.append(cur)
            cur = [f]
    if cur:
        runs.append(cur)
    return runs


def main() -> int:
    ap = argparse.ArgumentParser(description="APTv2 canidae 序列抽取（Format B）")
    ap.add_argument("--k9-data-root", type=Path, default=DEFAULT_K9_DATA)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "data_campaign" / "aptv2" / "sequences")
    ap.add_argument("--species", choices=["canidae", "all"], default="canidae")
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--stride", type=int, default=None, help="默认=window（不重叠）")
    ap.add_argument("--quick", type=int, default=0, help="仅处理前 N 条轨迹（冒烟用）")
    args = ap.parse_args()
    stride = args.stride or args.window

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ann_dir = args.k9_data_root / "APTv2" / "APTv2" / "annotations"
    keep_species = set(CANIDAE) if args.species == "canidae" else None

    # ── 1. 三 split 合并分组：track_key -> {frame_idx: (kp51, category)} ──
    tracks: dict[tuple, dict] = collections.defaultdict(
        lambda: {"frames": {}, "species": None, "splits": set(), "size": None})
    cat_name_global: dict[int, str] = {}
    for split in MAIN_SPLITS:
        d = load_json(ann_dir / split)
        for c in d["categories"]:
            cat_name_global[c["id"]] = c["name"]
            assert len(c["keypoints"]) == 17 and len(c["skeleton"]) == 17
        img = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            sp = cat_name_global[a["category_id"]]
            if keep_species is not None and sp not in keep_species:
                continue
            meta = img[a["image_id"]]
            stem = Path(meta["file_name"]).stem
            if not stem.isdigit():
                continue
            key = (meta.get("video_id"), a.get("track_id"))
            t = tracks[key]
            t["frames"][int(stem)] = a["keypoints"]
            t["species"] = sp
            t["splits"].add(split.split("_")[0])
            t["size"] = (meta.get("width"), meta.get("height"))
    print(f"[1] 轨迹分组: {len(tracks)} 条 (species={args.species})")

    # ── 2. 连续段 → 整长窗落盘 ──
    out_dir = args.out_dir / args.species
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    stat = collections.Counter()
    run_len_hist = collections.Counter()

    keys = sorted(tracks.keys(), key=lambda k: (str(k[0]), str(k[1])))
    if args.quick:
        keys = keys[: args.quick]

    for vid, tid in keys:
        t = tracks[(vid, tid)]
        frames_sorted = sorted(t["frames"])
        for run in contiguous_runs(frames_sorted):
            run_len_hist[len(run)] += 1
            n_win = 1 + max(0, (len(run) - args.window)) // stride if len(run) >= args.window else 0
            stat["runs_seen"] += 1
            if n_win == 0:
                stat["runs_below_window"] += 1
                continue
            split_label = "+".join(sorted(t["splits"]))
            for w in range(n_win):
                idxs = run[w * stride: w * stride + args.window]
                if len(idxs) < args.window:
                    break
                kp = np.asarray([t["frames"][i] for i in idxs], dtype=np.float32).reshape(len(idxs), 17, 3)
                vis01 = kp[:, :, 2] / 2.0  # {0,2}→{0.0,1.0}，见模块 docstring
                arr = np.stack([kp[:, :, 0], kp[:, :, 1], vis01], axis=-1).astype(np.float32)
                seq_id = f"aptv2_{t['species']}_v{vid}_t{tid}_w{w:03d}"
                payload = {
                    "keypoints": arr,
                    "topology_name": TOPOLOGY_NAME,
                    "V": 17,
                    "fps_or_sampling": "unknown_fps_consecutive_annotated_frames",
                    "source": {
                        "dataset": "APTv2 (ViTAE-Transformer/APTv2)",
                        "annotation_jsons": MAIN_SPLITS,
                        "root": str(args.k9_data_root / "APTv2" / "APTv2"),
                        "video_id": vid,
                        "track_id": tid,
                        "frame_indices": idxs,
                        "image_size_wh": list(t["size"]) if t["size"] else None,
                        "visibility_mapping": "raw_v_flag/2.0 ({0,2}->{0.0,1.0})",
                    },
                    "split": split_label,
                    "species": t["species"],
                    "sequence_id": seq_id,
                }
                fp = out_dir / f"{seq_id}.pkl"
                with open(fp, "wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
                vis_frac = float((arr[:, :, 2] > 0).mean())
                entries.append({
                    "file": str(fp.relative_to(args.out_dir)),
                    "sequence_id": seq_id,
                    "species": t["species"],
                    "split": split_label,
                    "T": len(idxs), "V": 17, "C": 3,
                    "visible_frac_mean": round(vis_frac, 4),
                    "sha256": hashlib.sha256(fp.read_bytes()).hexdigest(),
                })
                stat["windows_written"] += 1
                stat[f"species_{t['species']}"] += 1
                stat[f"split_{split_label}"] += 1

    # ── 3. manifest 落盘 ──
    manifest = {
        "campaign": "DATA-CAMPAIGN C2/W26 后续①",
        "format_b_contract": "{keypoints:(T,V,C), topology_name, V, fps_or_sampling, source, split}",
        "generator": "scripts/mine_aptv2_extract_sequences.py",
        "params": {"species_scope": args.species, "window": args.window,
                   "stride": stride, "confidence": "raw_v_flag/2.0"},
        "topology_name": TOPOLOGY_NAME,
        "aggregate": {
            "runs_seen": stat["runs_seen"],
            "runs_below_window": stat["runs_below_window"],
            "run_len_hist": dict(sorted(run_len_hist.items())),
            "windows_written": stat["windows_written"],
            "by_species": {k.replace("species_", ""): v for k, v in stat.items() if k.startswith("species_")},
            "by_split": {k.replace("split_", ""): v for k, v in stat.items() if k.startswith("split_")},
            "visible_frac_mean_overall": round(float(np.mean([e["visible_frac_mean"] for e in entries])), 4) if entries else None,
        },
        "entries": sorted(entries, key=lambda e: e["sequence_id"]),
    }
    mf = args.out_dir / "_manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    agg = manifest["aggregate"]
    print(f"[2] 写入 {agg['windows_written']} 条序列 → {out_dir}")
    print(f"    按种: {agg['by_species']} | 按 split: {agg['by_split']}")
    print(f"    低于窗长的段: {agg['runs_below_window']} | 段长直方图 top: "
          f"{dict(sorted(agg['run_len_hist'].items(), key=lambda kv: -kv[1])[:5])}")
    print(f"    平均可见关节占比: {agg['visible_frac_mean_overall']}")
    print(f"[3] manifest → {mf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
