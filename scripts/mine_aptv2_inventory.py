#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W26 / C2：APTv2 83K 本地文件池盘点挖掘器。

任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §2-C2
领地  : scripts/mine_aptv2_* + runs/data_campaign/aptv2/（本脚本零写入 K9 仓，纯 CPU）

盘点目标:
  1. 物种分布（30 类 COCO category，标注计数三层拆分 train/val/test）
  2. 图像 vs 序列构成（41,179 图 ↔ video_id/clip 目录；帧连续性判定静态池 or 序列池）
  3. canidae 子集规模（主池 dog/fox/wolf + aptv2_canidae 24 视频 MOT GT）
  4. 关键点拓扑与置信度结构（17 kp 统一拓扑 / skeleton / v-flag 分布 / num_keypoints 异常）

输出（全部落在 psd 本仓 worktree）:
  runs/data_campaign/aptv2/aptv2_inventory_summary.json   全量机读结果
  runs/data_campaign/aptv2/canidae_sequence_manifest.json 可入库序列清单（Format B 导向）

用法:
  .venv/Scripts/python.exe scripts/mine_aptv2_inventory.py            # 全量
  .venv/Scripts/python.exe scripts/mine_aptv2_inventory.py --quick    # 跳过文件池全盘走查
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_K9_DATA = Path(r"D:\Desktop\k9-training-system\data")

# APTv2 主池顶层标注（train/val/test 为标注级切分，共享同一 images 清单；
# easy/hard 变体为难度视图，与主文件存在包含关系——用 ann 身份集合量化重叠）
MAIN_SPLITS = [
    "train_annotations.json",
    "val_annotations.json",
    "test_annotations.json",
    "test_annotations_easy.json",
    "test_annotations_hard.json",
    "val_annotations_easy.json",
    "val_annotations_hard.json",
]

CANIDAE_SPECIES = {"dog", "fox", "wolf"}  # APTv2 Canidae 科（fewshot/leaveoneout 协议同口径）
SEQ_WINDOWS = (8, 16, 30)  # ST-GCN 常用 T 档位：可连续采样窗口长度门槛


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def analyze_split(data: dict) -> dict:
    """对单个 COCO 式标注文件做结构化统计。"""
    cats = {c["id"]: c["name"] for c in data["categories"]}
    per_cat = collections.Counter(cats[a["category_id"]] for a in data["annotations"])
    video_frames = collections.Counter(im.get("video_id") for im in data["images"])

    vis_total = collections.Counter()          # v-flag 总分布 {0,1,2}
    vis_per_joint = [collections.Counter() for _ in range(17)]  # 逐关节 v-flag
    nk_hist = collections.Counter()
    kp_len_hist = collections.Counter()
    has_bbox = has_score = has_track = has_video = 0
    ann_ids = set()
    file_names = set()

    for a in data["annotations"]:
        ann_ids.add((a.get("image_id"), a.get("id")))
        if a.get("bbox") is not None:
            has_bbox += 1
        if "score" in a:
            has_score += 1
        if "track_id" in a:
            has_track += 1
        if "video_id" in a:
            has_video += 1
        kp = a.get("keypoints", [])
        kp_len_hist[len(kp)] += 1
        nk_hist[a.get("num_keypoints", -1)] += 1
        n_joints = len(kp) // 3
        for j in range(min(17, n_joints)):
            v = kp[3 * j + 2]
            vis_total[v] += 1
            vis_per_joint[j][v] += 1

    for im in data["images"]:
        file_names.add(im["file_name"])
        if "video_id" not in im:
            has_video = -1  # 标记图像侧缺 video_id 的异常情况

    return {
        "n_images": len(data["images"]),
        "n_annotations": len(data["annotations"]),
        "n_unique_ann_identity": len(ann_ids),
        "n_videos": len(video_frames),
        "anns_per_category": dict(sorted(per_cat.items())),
        "video_frame_hist_top": dict(video_frames.most_common(5)),
        "visibility_flags_total": dict(sorted(vis_total.items())),
        "visibility_per_joint": {f"kp{j+1}": dict(sorted(c.items())) for j, c in enumerate(vis_per_joint)},
        "num_keypoints_hist": dict(sorted(nk_hist.items())),
        "keypoint_array_len_hist": dict(sorted(kp_len_hist.items())),
        "has_bbox": has_bbox,
        "has_score_field": has_score,
        "has_track_id": has_track,
        "file_names": file_names,
        "_ann_ids": ann_ids,
    }


def topology_fingerprint(categories: list) -> dict:
    """校验 30 个 category 是否共用同一 17-kp 拓扑；返回指纹。"""
    sigs = collections.Counter()
    names = skeleton = kps = None
    for c in categories:
        sig = json.dumps([c.get("keypoints"), c.get("skeleton")], ensure_ascii=False)
        sigs[sig] += 1
        if names is None:
            names, skeleton, kps = c["name"], c.get("skeleton"), c.get("keypoints")
    return {
        "n_categories": len(categories),
        "unique_topology_count": len(sigs),
        "example_species": names,
        "keypoint_names": kps,
        "skeleton_edges_1based": skeleton,
    }


def seq_stats(tracks: dict) -> dict:
    """tracks: {(video_id, track_id): sorted unique frame_idx list} → 连续性统计。"""
    spans = [len(v) for v in tracks.values()]
    window_counts = {t: 0 for t in SEQ_WINDOWS}
    max_runs = []
    consec_ratio_num = consec_ratio_den = 0
    for frames in tracks.values():
        fs = sorted(set(frames))
        run = best = 1
        for i in range(1, len(fs)):
            if fs[i] == fs[i - 1] + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        max_runs.append(best)
        for t in SEQ_WINDOWS:
            if best >= t:
                window_counts[t] += 1
        consec_ratio_num += sum(1 for i in range(1, len(fs)) if fs[i] == fs[i - 1] + 1)
        consec_ratio_den += max(0, len(fs) - 1)

    def q(p):
        s = sorted(max_runs)
        if not s:
            return None
        i = min(len(s) - 1, int(p * (len(s) - 1)))
        return s[i]

    return {
        "n_tracks": len(spans),
        "span_median": median(spans) if spans else 0,
        "span_p90": q(0.90),
        "max_run_median": median(max_runs) if max_runs else 0,
        "consecutive_pair_ratio": round(consec_ratio_num / consec_ratio_den, 4) if consec_ratio_den else 0.0,
        f"tracks_with_run>=8": window_counts[8],
        f"tracks_with_run>=16": window_counts[16],
        f"tracks_with_run>=30": window_counts[30],
    }


def scan_file_pool(data_dir: Path) -> dict:
    """递归扫描 data/{easy,hard} 文件池，按扩展名与物种目录聚合。"""
    ext_counter = collections.Counter()
    species_files = collections.Counter()
    total = 0
    jpg_relpaths = set()
    for p in data_dir.rglob("*"):
        if p.is_file():
            total += 1
            ext = p.suffix.lower() or "(无扩展名)"
            ext_counter[ext] += 1
            rel = p.relative_to(data_dir)
            parts = rel.parts
            if len(parts) >= 2:
                species_files[f"{parts[0]}/{parts[1]}"] += 1
            if ext in (".jpg", ".jpeg", ".png"):
                jpg_relpaths.add(rel.as_posix())
    return {
        "total_files": total,
        "ext_distribution": dict(ext_counter.most_common()),
        "species_file_counts": dict(sorted(species_files.items())),
        "_jpg_relpaths": jpg_relpaths,
    }


def parse_mot_gt(gt_path: Path) -> dict:
    """解析 MOTChallenge 格式逐行: frame,id,x,y,w,h,conf,class,vis。"""
    rows = 0
    frames = set()
    ids = set()
    classes = collections.Counter()
    with open(gt_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 9:
                continue
            rows += 1
            frames.add(int(float(parts[0])))
            ids.add(parts[1])
            classes[parts[7]] += 1
    return {"rows": rows, "frames": len(frames), "track_ids": len(ids), "classes": dict(classes)}


def main() -> int:
    ap = argparse.ArgumentParser(description="APTv2 本地池盘点（CPU-only，K9 只读）")
    ap.add_argument("--k9-data-root", type=Path, default=DEFAULT_K9_DATA)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "runs" / "data_campaign" / "aptv2")
    ap.add_argument("--quick", action="store_true", help="跳过 data/{easy,hard} 文件池全盘走查")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    main_root = args.k9_data_root / "APTv2" / "APTv2"
    ann_dir = main_root / "annotations"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"generated_by": "scripts/mine_aptv2_inventory.py", "k9_data_root": str(args.k9_data_root)}

    # ── A. 主池标注解析 ──────────────────────────────────────────
    split_results: dict[str, dict] = {}
    union_cat = collections.Counter()
    all_file_names: set[str] = set()
    canidae_union = collections.Counter()

    for name in MAIN_SPLITS:
        path = ann_dir / name
        if not path.exists():
            print(f"[warn] 缺失标注文件: {path}")
            continue
        d = load_json(path)
        st = analyze_split(d)
        topo = topology_fingerprint(d["categories"])
        st["topology"] = topo
        st.pop("_ann_ids")
        split_results[name] = st
        all_file_names |= st.pop("file_names")
        union_cat.update({k: 0 for k in st["anns_per_category"]})
        for k, v in st["anns_per_category"].items():
            union_cat[k] += v
            if k in CANIDAE_SPECIES:
                canidae_union[k] += v
        print(f"[A] {name}: images={st['n_images']} anns={st['n_annotations']} videos={st['n_videos']} "
              f"canidae={sum(v for k, v in st['anns_per_category'].items() if k in CANIDAE_SPECIES)}")

    # train/val/test 三文件的重叠量化（身份 = (image_id, id)）
    ids_main = {}
    for name in ["train_annotations.json", "val_annotations.json", "test_annotations.json"]:
        p = ann_dir / name
        if p.exists():
            d = load_json(p)
            ids_main[name] = {(a.get("image_id"), a.get("id")) for a in d["annotations"]}
    tr, va, te = (ids_main.get(k, set()) for k in
                  ["train_annotations.json", "val_annotations.json", "test_annotations.json"])
    summary["split_overlap"] = {
        "train∩val": len(tr & va), "train∩test": len(tr & te), "val∩test": len(va & te),
        "union_unique": len(tr | va | te),
    }

    summary["main_annotations"] = {k: v for k, v in split_results.items()}
    summary["species_union_all_splits"] = dict(sorted(union_cat.items()))
    summary["canidae_species_union"] = dict(sorted(canidae_union.items()))

    # ── B. 序列结构与静态池判定（基于三主 split 联合、按 track 分组）──────────
    tracks_all: dict[tuple, list] = collections.defaultdict(list)
    tracks_canidae: dict[tuple, list] = collections.defaultdict(list)
    cat_name = {}
    video_dirs = collections.defaultdict(set)
    for name in ["train_annotations.json", "val_annotations.json", "test_annotations.json"]:
        p = ann_dir / name
        d = load_json(p)
        for c in d["categories"]:
            cat_name[c["id"]] = c["name"]
        img_video = {im["id"]: im.get("video_id") for im in d["images"]}
        img_file = {im["id"]: im["file_name"] for im in d["images"]}
        for a in d["annotations"]:
            vid = img_video.get(a.get("image_id"))
            key = (vid, a.get("track_id"))
            # 帧号取自 file_name 数字主干（0000.jpg → 0），跨 split 同图同帧不重复计入
            fname = img_file[a["image_id"]]
            stem = Path(fname).stem
            fidx = int(stem) if stem.isdigit() else -1
            tracks_all[key].append(fidx)
            video_dirs[vid].add(str(Path(fname).parent))
            if cat_name.get(a["category_id"]) in CANIDAE_SPECIES:
                tracks_canidae[key].append(fidx)

    # 去重帧索引后统计
    tracks_all_u = {k: sorted(set(v)) for k, v in tracks_all.items()}
    tracks_cani_u = {k: sorted(set(v)) for k, v in tracks_canidae.items()}
    summary["sequence_structure"] = {
        "n_videos_with_annotations": len(video_dirs),
        "overall_tracks": seq_stats(tracks_all_u),
        "canidae_tracks": seq_stats(tracks_cani_u),
        "annotated_images_union": len(all_file_names),
    }
    print(f"[B] videos={len(video_dirs)} tracks_all={len(tracks_all_u)} tracks_canidae={len(tracks_cani_u)}")

    # canidae 逐视频序列清单（可入库候选）
    # canidae 逐视频序列清单（可入库候选）——以视频为粒度重组
    manifest_entries = []
    cani_by_video: dict[int, dict] = {}
    for (vid, tid), frames in tracks_cani_u.items():
        e = cani_by_video.setdefault(vid, {"video_id": vid, "species": {}, "tracks": 0,
                                           "annotated_frames": set(), "best_run": 0})
        e["tracks"] += 1
        e["annotated_frames"] |= set(frames)
    # 回填物种与目录（再扫一遍轻量化：只查 canidae anns）
    for name in ["train_annotations.json", "val_annotations.json", "test_annotations.json"]:
        d = load_json(ann_dir / name)
        img_meta = {im["id"]: im for im in d["images"]}
        for a in d["annotations"]:
            sp = cat_name.get(a["category_id"])
            if sp in CANIDAE_SPECIES:
                vid = img_meta[a["image_id"]].get("video_id")
                if vid in cani_by_video:
                    e = cani_by_video[vid]
                    e["species"][sp] = e["species"].get(sp, 0) + 1
                    e.setdefault("dir", str(Path(img_meta[a["image_id"]]["file_name"]).parent))
                    e.setdefault("split_dirs", set()).add(str(Path(img_meta[a["image_id"]]["file_name"]).parent))

    def best_run(frames_sorted):
        run = best = 1
        for i in range(1, len(frames_sorted)):
            if frames_sorted[i] == frames_sorted[i - 1] + 1:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return best

    for vid, e in cani_by_video.items():
        frames = sorted(e["annotated_frames"])
        br = best_run(frames) if frames else 0
        windows = {}
        for t in SEQ_WINDOWS:
            cnt = sum(1 for s in range(len(frames)) if s + t <= len(frames)
                      and frames[s + t - 1] - frames[s] == t - 1)
            windows[f"windows_T{t}"] = cnt
        manifest_entries.append({
            "sequence_id": f"aptv2_v{vid}",
            "source_channel": "C2_aptv2_local",
            "origin_path_pattern": f"{main_root}\\data\\{{easy|hard}}\\<species_dir>\\<clip_dir>\\NNNN.jpg",
            "annotation_jsons": ["train_annotations.json", "val_annotations.json", "test_annotations.json"],
            "species_ann_counts": e["species"],
            "n_tracks": e["tracks"],
            "n_annotated_frames": len(frames),
            "longest_contiguous_run": br,
            **windows,
            "format_b_ready": bool(br >= 16),
            "note": "关键点需从主池 COCO JSON 按 video_id+track_id 抽取转 (T,V,C)；本清单只登记不搬数据",
        })
    manifest_entries.sort(key=lambda x: (-x["n_annotated_frames"], x["sequence_id"]))
    manifest = {
        "campaign": "DATA-CAMPAIGN C2/W26",
        "format": "B(keypoint sequences)导向的登记清单——数据不搬移，路径引用 K9 只读池",
        "canidae_family_species": sorted(CANIDAE_SPECIES),
        "n_sequence_groups": len(manifest_entries),
        "entries": manifest_entries,
    }
    (out_dir / "canidae_sequence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[B] canidae_sequence_manifest.json: {len(manifest_entries)} 序列组 → {out_dir}")

    # ── C. 文件池扫描 + 标注覆盖率 ───────────────────────────────
    pool = {"skipped": True}
    if not args.quick:
        pool = scan_file_pool(main_root / "data")
        on_disk = pool.pop("_jpg_relpaths")
        annotated = {fn.replace("\\", "/") for fn in all_file_names}
        pool["jpg_annotated_in_splits"] = len(on_disk & annotated)
        pool["jpg_on_disk_not_annotated"] = len(on_disk - annotated)
        pool["annotated_missing_on_disk"] = len(annotated - on_disk)
        w2_reference = 83304
        pool["w2_20260823_reference"] = w2_reference
        pool["delta_vs_w2"] = pool["total_files"] - w2_reference
        print(f"[C] 文件池 total={pool['total_files']} (W2 参考 {w2_reference}, Δ={pool['delta_vs_w2']}) "
              f"已标注={pool['jpg_annotated_in_splits']} 磁盘未标注={pool['jpg_on_disk_not_annotated']} "
              f"标注缺失于磁盘={pool['annotated_missing_on_disk']}")
    summary["file_pool"] = pool

    # ── D. 伴生目录（K9 产品线产物，只读登记）───────────────────
    companion: dict = {}
    cani_dir = args.k9_data_root / "aptv2_canidae"
    if cani_dir.exists():
        vids = sorted((cani_dir / "videos").glob("*.mp4"))
        gts = sorted((cani_dir / "gt_per_video").glob("*.txt"))
        gt_stats = []
        total_rows = 0
        for g in gts:
            st = parse_mot_gt(g)
            total_rows += st["rows"]
            gt_stats.append({"file": g.name, **st})
        frames_hard = list((cani_dir / "frames" / "hard").rglob("*"))
        frames_files = [p for p in frames_hard if p.is_file()]
        sample_json = next((p for p in frames_files if p.suffix == ".json"), None)
        sample_keys = None
        if sample_json:
            try:
                dj = load_json(sample_json)
                sample_keys = list(dj.keys()) if isinstance(dj, dict) else type(dj).__name__
            except Exception as e:
                sample_keys = f"(解析失败: {e})"
        companion["aptv2_canidae"] = {
            "root": str(cani_dir),
            "n_videos_mp4": len(vids),
            "videos_total_mb": round(sum(v.stat().st_size for v in vids) / 1e6, 2),
            "gt_format": "MOTChallenge(frame,id,x,y,w,h,conf,class,vis)",
            "gt_total_rows": total_rows,
            "gt_per_video": gt_stats,
            "frames_hard_tree_files": len(frames_files),
            "frames_hard_sample_json_keys": sample_keys,
            "gt_txt_combined_exists": (cani_dir / "gt.txt").exists(),
        }
        print(f"[D] aptv2_canidae: {len(vids)} 视频 / MOT GT {total_rows} 行")
    for sub in ("aptv2_yolo", "aptv2_yolo_pose"):
        d_ = args.k9_data_root / sub
        if d_.exists():
            imgs = len(list((d_ / "images").rglob("*"))) if (d_ / "images").exists() else 0
            lbls = len(list((d_ / "labels").rglob("*"))) if (d_ / "labels").exists() else 0
            yaml_files = {p.name: p.read_text(encoding="utf-8", errors="replace") for p in d_.glob("*.yaml")}
            companion[sub] = {"images": imgs, "labels": lbls, "yaml": yaml_files}
            print(f"[D] {sub}: images={imgs} labels={lbls}")
    # aptv2_annotations 与主池重复性（字节级尺寸对照即可定性）
    dup_dir = args.k9_data_root / "aptv2_annotations" / "APTv2" / "annotations"
    if dup_dir.exists():
        pairs = []
        for name in MAIN_SPLITS[:3]:
            p1, p2 = ann_dir / name, dup_dir / name
            if p1.exists() and p2.exists():
                pairs.append({"file": name, "size_main": p1.stat().st_size,
                              "size_dup": p2.stat().st_size,
                              "same_size": p1.stat().st_size == p2.stat().st_size})
        companion["aptv2_annotations_duplicate_check"] = pairs
    summary["companion_dirs"] = companion

    # ── 汇总落盘 ───────────────────────────────────────────────
    (out_dir / "aptv2_inventory_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] aptv2_inventory_summary.json → {out_dir}")

    # 关键结论速览
    ss = summary["sequence_structure"]
    print("\n===== 速览 =====")
    print("物种联合分布 top10:", dict(sorted(summary['species_union_all_splits'].items(),
                                           key=lambda kv: -kv[1])[:10]))
    print("canidae 联合:", summary["canidae_species_union"])
    print("整体轨迹:", ss["overall_tracks"])
    print("canidae 轨迹:", ss["canidae_tracks"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
