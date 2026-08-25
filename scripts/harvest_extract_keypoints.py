#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W35: C1 视频片段池提点执行器（数据飞轮第二圈第一段）.

任务书: dev-docs/handovers/NEXT-BATCH-plan.md §W35 第①步
领地:   scripts/harvest_*、runs/data_campaign/video/keypoints_*

对主检出 runs/data_campaign/video/ 的 manifest 准入片段（实测 642 条,
与 manifest.jsonl 零差额）用 Q3a 犬类 pose 权重(dog-pose 微调, 24 点)提点。

双产物单次过帧设计:
  A. rule_pkls/<fid>.pkl —— 规则种子轨: 按 --rule-target-fps(默认10fps) 步进采样,
     kp_world=(F,24,3) 图像像素坐标 (x, y, -y)；第三通道取负 y 作为"高度代理",
     使 W6 引擎的竖直轴语义(z 向上)在单目 2D 视频上成立。kp_weight=(F,24)
     为逐关节置信度。帧缺检 → 该行 NaN + 权重 0(引擎 valid-mask 原生消费)。
     ⚠️ 本轨保留模型原始输出(死关节不清零——NaN 化由 harvest_rule_seeds
     载入端按契约执行), 像素各向同性保真供体尺度归一。
  B. seq30/<fid>.pkl —— ST-GCN 轨(收敛契约格式 B): uniform_frame_indices
     取 T=30 均匀帧, 复用 psd/data/ak_pose_extract.assemble_clip(插值+
     DEAD_JOINTS 硬掩码自动生效), 坐标 xyn 归一化+conf, 与 AK q3b 血统一致。

工程纪律:
  - 断点续跑: extract_index.jsonl 每 --batch-size 片段追加落盘, 重启跳过已完成;
  - GPU 错峰让行: 每批开始前查询 nvidia-smi 计算进程, 发现非本进程 python
    占卡即挂起轮询(默认最长等 4h), 保证 W33 线性评估优先;
  - 拓扑契约: 非 24 点权重 fail-fast(COCO 17 点事故防线, 承 W28 P3 提案)。

用法:
  & "D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe" `
      scripts/harvest_extract_keypoints.py `
      --manifest  D:/Desktop/psd-framework/runs/data_campaign/video/manifest.jsonl `
      --fragments-dir D:/Desktop/psd-framework/runs/data_campaign/video/fragments `
      --weights   D:/Desktop/psd-framework/runs/public_real_yolo_dogpose/train/weights/best.pt `
      --out-root  runs/data_campaign/video/keypoints_w35 `
      --batch-size 50 [--limit 2 冒烟]
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

from psd.data.ak_pose_extract import (  # noqa: E402
    CLIP_LEN_T,
    DEAD_JOINTS,
    assemble_clip,
    pick_best_instance,
    uniform_frame_indices,
)

#: K9Graph/dog-pose 拓扑契约(与 run_p05_public_real_pipeline.EXPECTED_KPTS 同源)
EXPECTED_KPTS = 24
#: 单帧实例置信度门槛(低于视为未检出→插值)——承 AK 血统 CONF_THRESHOLD
MEAN_VIS_THRESHOLD = 0.20

SEQ_SOURCE_TAG = "video_c1_w25_fragments"
SPLIT_TAG = "unlabeled_draft"
UNLABELED_SENTINEL = -1


# ---------------------------------------------------------------------------
# 纯逻辑层(TDD 覆盖)
# ---------------------------------------------------------------------------

def compute_rule_indices(n_frames: int, fps_src: float, target_fps: float) -> List[int]:
    """规则种子轨采样下标(步进取整≥1): 近似匀速降到 target_fps."""
    if n_frames <= 0:
        return []
    fps_src = float(fps_src) if fps_src and fps_src > 0 else 30.0
    target_fps = float(target_fps) if target_fps and target_fps > 0 else 10.0
    stride = max(1, int(round(fps_src / target_fps)))
    return list(range(0, n_frames, stride))


def union_sample_plan(n_frames: int, fps_src: float, target_fps: float,
                      t_seq: int = CLIP_LEN_T) -> Tuple[List[int], List[int]]:
    """单次过帧的采样计划: 返回 (want_sorted, rule_idx).

    want = rule ∪ seq30 去重并序; seq30 由 uniform_frame_indices 给出。
    """
    rule_idx = compute_rule_indices(n_frames, fps_src, target_fps)
    seq_idx = uniform_frame_indices(n_frames, t_seq) if n_frames > 0 else []
    want = sorted(set(rule_idx) | set(seq_idx))
    return want, rule_idx


def to_kp_world_pixel(xyn_conf: np.ndarray, width: int, height: int) -> np.ndarray:
    """(24,3)[xyn归一+conf] → (24,3)[像素 x, 像素 y, -y 高度代理].

    高度代理取 -y: 图像 y 轴向下, 引擎假设 z 轴向上(站立四爪低、躯干高)。
    使用原始像素而非 xyn: 保各向同性, 体尺度归一后全局比例因子相消。
    """
    kp = np.asarray(xyn_conf, dtype=np.float32)
    out = np.empty_like(kp)
    out[:, 0] = kp[:, 0] * float(width)
    out[:, 1] = kp[:, 1] * float(height)
    out[:, 2] = -kp[:, 1] * float(height)
    return out


def build_rule_pkl(dets: Dict[int, Optional[np.ndarray]], rule_idx: List[int],
                   width: int, height: int, fps_src: float,
                   actual_read: int) -> dict:
    """组装规则种子轨 pkl(缺检行 NaN + 权重 0, 交引擎 valid-mask)."""
    f = len(rule_idx)
    kp_world = np.full((f, EXPECTED_KPTS, 3), np.nan, dtype=np.float32)
    kp_weight = np.zeros((f, EXPECTED_KPTS), dtype=np.float32)
    hit = 0
    for i, fi in enumerate(rule_idx):
        d = dets.get(fi)
        if d is None:
            continue
        kp_world[i] = to_kp_world_pixel(d, width, height)
        kp_weight[i] = d[:, 2].astype(np.float32)
        hit += 1
    return {
        "kp_world": kp_world,
        "kp_weight": kp_weight,
        # 缺检/未读到帧保留在序列中(对应行 NaN+权重0), 引擎 valid-mask 原生消费
        "frame_idx": np.asarray(rule_idx, dtype=np.int64),
        "coords_semantic": "image_pixel_xy_with_negy_height_proxy",
        "height_proxy_note": "channel2 = -pixel_y (image up), 单目2D高度代理非度量3D",
        "src_fps": float(fps_src),
        "n_detect_hit": int(hit),
        "frame_width": int(width),
        "frame_height": int(height),
        "dead_joints_note": f"模型原始输出未清零; 死关节{list(DEAD_JOINTS)}由 harvest_rule_seeds 载入端 NaN 化",
    }


def build_seq_entry(dets: Dict[int, Optional[np.ndarray]], seq_idx: List[int]) -> Tuple[Optional[dict], dict]:
    """组装格式 B 序列条目(assemble_clip 全缺检返回 None)."""
    frames: List[Optional[np.ndarray]] = []
    no_detect = 0
    for fi in seq_idx:
        d = dets.get(fi)
        if d is None:
            frames.append(None)
            no_detect += 1
        else:
            frames.append(d.astype(np.float32))
    clip = assemble_clip(frames, UNLABELED_SENTINEL)
    quality = {"n_interpolated": None, "no_detect": no_detect}
    if clip is None:
        return None, {**quality, "status": "all_missing"}
    entry = {
        "sample_id": None,  # 由调用方填 fragment_id
        "keypoints": clip["keypoints"],
        "label": UNLABELED_SENTINEL,
        "boundary": clip["boundary"],
        "topology_name": "K9Graph",
        "V": EXPECTED_KPTS,
        "T": int(clip["keypoints"].shape[0]),
        "coords_semantic": "image_norm_xy_conf01_deadmasked",
        "split": SPLIT_TAG,
        "source": SEQ_SOURCE_TAG,
        "n_interpolated": clip["n_interpolated"],
        "dead_joints_masked": clip["dead_joints_masked"],
    }
    return entry, {**quality, "status": "ok", "n_interpolated": clip["n_interpolated"]}


# ---------------------------------------------------------------------------
# GPU 让行门禁
# ---------------------------------------------------------------------------

def gpu_foreign_python_pids(mypid: int) -> List[int]:
    """返回占用 GPU 计算的 python 进程 pid(排除自身). 查询失败视为空闲."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=30,
        ).stdout
    except Exception:
        return []
    pids: List[int] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if "python" in parts[1].lower() and pid != mypid:
            pids.append(pid)
    return pids


def wait_gpu_free(mypid: int, poll_s: int = 120, max_wait_s: int = 4 * 3600) -> bool:
    """GPU 被其他 python 进程占卡则挂起等待(W33 优先). 超时返回 False."""
    waited = 0
    while True:
        pids = gpu_foreign_python_pids(mypid)
        if not pids:
            return True
        if waited >= max_wait_s:
            print(f"[gpu-yield] 等待超时({max_wait_s}s), 外部占卡 pid={pids}, 优雅退出")
            return False
        if waited % (poll_s * 10) == 0:
            print(f"[gpu-yield] 外部 python 占卡 pid={pids}, 挂起等待中(已候 {waited}s) ...", flush=True)
        time.sleep(poll_s)
        waited += poll_s


# ---------------------------------------------------------------------------
# IO/模型层(集成冒烟覆盖)
# ---------------------------------------------------------------------------

def probe_video(video_path: str) -> Tuple[int, float]:
    """cv2 现场探测 (n_frames, fps)——manifest 无帧数字段, 不能信元数据兜底."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"open_failed: {video_path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if n <= 0:
        raise RuntimeError(f"zero_frames_meta: {video_path}")
    return n, fps


def detect_frames(model, video_path: str, want: List[int],
                  n_meta: int) -> Tuple[Dict[int, Optional[np.ndarray]], int, int, int, dict]:
    """顺序读帧并对 want 下标推理. 返回 (dets, width, height, actual_read, stats).

    dets: fi -> (24,3)[xyn+conf] 或 None(未检出/低可见).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"open_failed: {video_path}")
    dets: Dict[int, Optional[np.ndarray]] = {}
    want_set = set(want)
    stats = {"no_detect": 0, "low_conf": 0, "ok": 0}
    width = height = 0
    fi = 0
    top = max(want) + 1 if want else 0
    actual_read = 0
    while fi < top:
        ok, frame = cap.read()
        if not ok:
            break
        actual_read += 1
        height, width = frame.shape[:2]
        if fi in want_set:
            res = model(frame, verbose=False)[0]
            inst_list = []
            if res.keypoints is not None and res.keypoints.xyn is not None:
                if res.keypoints.xyn.shape[1] != EXPECTED_KPTS:
                    cap.release()
                    raise RuntimeError(
                        f"权重拓扑不匹配: 输出 {res.keypoints.xyn.shape[1]} 点, "
                        f"需要 {EXPECTED_KPTS} 点(K9Graph/dog-pose)")
                if len(res.boxes) > 0:
                    xyn = res.keypoints.xyn.cpu().numpy()
                    kconf = res.keypoints.conf.cpu().numpy()
                    bconf = res.boxes.conf.cpu().numpy()
                    for i in range(len(bconf)):
                        vis = np.clip(np.nan_to_num(kconf[i], nan=0.0), 0, 1)
                        kp = np.concatenate([xyn[i], vis[:, None]], axis=1).astype(np.float32)
                        inst_list.append({"kp": kp, "score": float(bconf[i])})
            best = pick_best_instance(inst_list)
            if best is None:
                dets[fi] = None
                stats["no_detect"] += 1
            elif best[0, 0] == -1 or float(best[:, 2].mean()) < MEAN_VIS_THRESHOLD:
                dets[fi] = None
                stats["low_conf"] += 1
            else:
                dets[fi] = best
                stats["ok"] += 1
        fi += 1
    cap.release()
    # 视频实际帧数少于元数据时, 未读到的 want 位标记缺检
    for fi_miss in want:
        if fi_miss >= actual_read:
            dets.setdefault(fi_miss, None)
    return dets, width, height, actual_read, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--fragments-dir", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out-root", default=str(REPO / "runs/data_campaign/video/keypoints_w35"))
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--rule-target-fps", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=None, help="冒烟: 只处理前 N 个未完成片段")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    rule_dir = out_root / "rule_pkls"
    seq_dir = out_root / "seq30"
    idx_path = out_root / "extract_index.jsonl"
    for d in (rule_dir, seq_dir):
        d.mkdir(parents=True, exist_ok=True)

    man_rows = [json.loads(l) for l in
                Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    frag_dir = Path(args.fragments_dir)

    # 断点续跑: 已 ok 的片段跳过
    done: set = set()
    if idx_path.exists():
        for l in idx_path.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("status") == "ok":
                done.add(r["fragment_id"])
    todo = [r for r in man_rows
            if r["fragment_id"] not in done and (frag_dir / f"{r['fragment_id']}.mp4").exists()]
    if args.limit:
        todo = todo[:args.limit]

    print(f"[extract] manifest={len(man_rows)} 已完成={len(done)} 待处理={len(todo)}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    mypid = __import__("os").getpid()

    idx_f = idx_path.open("a", encoding="utf-8")
    n_in_batch = 0
    t0 = time.time()
    try:
        for k, row in enumerate(todo, 1):
            # 每批(含首批)开始前让行检查——W33 线性评估优先
            if n_in_batch == 0:
                if not wait_gpu_free(mypid):
                    sys.exit(3)

            fid = row["fragment_id"]
            vpath = frag_dir / f"{fid}.mp4"
            fps_src = float(row.get("fps_src") or 0)
            try:
                n_probe, fps_probe = probe_video(str(vpath))
                # 帧数以现场探测为准; fps 优先 manifest(与抓取期口径一致), 缺失用探测值
                fps_use = fps_src if fps_src > 0 else fps_probe
                want, rule_idx = union_sample_plan(n_probe, fps_use,
                                                   args.rule_target_fps)
                dets, w, h, actual_read, st = detect_frames(model, str(vpath), want, n_probe)
                if w == 0 or h == 0:
                    raise RuntimeError("zero_frames_or_unreadable")
                rule_pkl = build_rule_pkl(dets, rule_idx, w, h, fps_use, actual_read)
                with (rule_dir / f"{fid}.pkl").open("wb") as f:
                    pickle.dump(rule_pkl, f)
                seq_entry, seq_q = build_seq_entry(
                    dets, uniform_frame_indices(actual_read))
                if seq_entry is not None:
                    seq_entry["sample_id"] = f"w25::{fid}"
                    seq_entry["fps_or_sampling"] = {
                        "src_fps": fps_use, "strategy": f"uniform_T{CLIP_LEN_T}"}
                    with (seq_dir / f"{fid}.pkl").open("wb") as f:
                        pickle.dump(seq_entry, f)
                rec = {"fragment_id": fid, "status": seq_q["status"],
                       "src_fps": fps_use, "frames_read": actual_read,
                       "rule_frames": len(rule_idx), "rule_hits": rule_pkl["n_detect_hit"],
                       "no_detect": st["no_detect"], "low_conf": st["low_conf"],
                       "detect_ok": st["ok"], **seq_q}
            except RuntimeError as e:
                rec = {"fragment_id": fid, "status": f"error:{str(e)[:120]}"}

            idx_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_in_batch += 1
            if n_in_batch >= args.batch_size or k == len(todo):
                idx_f.flush()
                print(f"[extract] {k}/{len(todo)} 已落盘索引 "
                      f"(累计耗时 {time.time()-t0:.0f}s)", flush=True)
                n_in_batch = 0
    finally:
        idx_f.close()

    print(f"[done] 提点完成: {idx_path}")


if __name__ == "__main__":
    main()
