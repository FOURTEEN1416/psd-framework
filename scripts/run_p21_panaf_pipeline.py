# -*- coding: utf-8 -*-
"""P7 PanAf 骨架提取 — ASBAR 预训练 DLC 模型推理管线。

前置（自动检查）:
  1. runs/p7_asbar/panaf_dataset.zip 下载完成并解压出 PanAf500/videos/*.mp4
  2. ASBAR pretrained DLC: runs/p7_asbar/asbar/pretrained_models/deeplabcut/snapshot-60000
  3. deeplabcut 已安装 (3.0.1)

流程:
  Step A: 从 ASBAR 仓读 conf/specs，重建 DLC 项目 config（指向预训练 snapshot）
  Step B: 对每条 PanAf500 视频跑 dlc.analyze_videos → (frames, n_kpt, 3) 概率图/坐标
  Step C: 把 DLC 灵长类关键点映射到 PSD 24-slot 拓扑（四肢+躯干对应槽位）
  Step D: 落盘 panaf500_T30.pkl（同 full12_T30.pkl 格式：keypoints/label/split/boundary/video_id）
          label = PanAf500 JSON frame-level 行为（9 类）多数投票到 clip
产出: runs/p7_asbar/panaf500_T30.pkl + reports/p7-extract-log.md
"""
from __future__ import annotations
import json, pickle, sys, time
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parents[1]
ASBAR = REPO / "runs" / "p7_asbar" / "asbar"
DLC_SNAP_DIR = ASBAR / "pretrained_models" / "deeplabcut"
ZIP_PATH = REPO / "runs" / "p7_asbar" / "panaf_dataset.zip"
EXTRACT_DIR = REPO / "runs" / "p7_asbar" / "panaf_extract"
OUT_PKL = REPO / "runs" / "p7_asbar" / "panaf500_T30.pkl"

# ASBAR/OpenMonkeyChallenge 灵长类关键点 → PSD 24-slot 映射
# OMC 17+ 关键点（鼻/眼/肩/肘/腕/髋/膝/踝…），PSD 24-slot 犬科拓扑
# 映射原则：四肢关节功能对应（shoulder→shoulder 等），躯干槽位用骨盆/脊柱近似，
# 犬科特有的 withers/throat/eyes 死槽保持 0（与 L8 20/24 有效监督口径一致）
OMC_TO_PSD24 = {
    # PSD slot: OMC landmark name (大小写不敏感, 匹配失败则置 0)
    0: "nose",          # head
    1: None,            # (dead slot - eye, primate 有但犬拓扑死槽)
    2: None,            # (dead slot - eye)
    3: None,            # (dead slot - withers)
    4: None,            # (dead slot - throat)
    5: "right_shoulder",
    6: "right_elbow",
    7: "right_wrist",
    8: "left_shoulder",
    9: "left_elbow",
    10: "left_wrist",
    11: "right_hip",
    12: "right_knee",
    13: "right_ankle",
    14: "left_hip",
    15: "left_knee",
    16: "left_ankle",
    17: "tail_base",    # 或 spine
    18: None,
    19: None,
    20: None,           # dead
    21: None,           # dead
    22: None,           # dead
    23: None,           # dead
}
PANAF_BEHAVIORS = ["sitting","standing","walking","running","climbing_up",
                   "climbing_down","hanging","sitting_on_back","camera_interaction"]


def step0_check():
    """检查前置条件"""
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"PanAf zip 未找到: {ZIP_PATH}")
    size = ZIP_PATH.stat().st_size
    print(f"[p7] PanAf zip: {size/1e9:.1f} GB")
    if not DLC_SNAP_DIR.exists():
        raise FileNotFoundError(f"DLC snapshot 未找到: {DLC_SNAP_DIR}")
    snaps = list(DLC_SNAP_DIR.glob("snapshot-*"))
    print(f"[p7] DLC snapshots: {[s.name for s in snaps]}")
    # 检查 ffmpeg 可用于视频解帧
    import shutil
    assert shutil.which("ffmpeg") or shutil.which("ffprobe"), "ffmpeg 未找到（视频解帧需要）"


def step1_extract_zip():
    """只解压 PanAf500 子目录（跳过 PanAf20K 的 40GB 视频以省时间）"""
    import zipfile
    out = EXTRACT_DIR
    out.mkdir(parents=True, exist_ok=True)
    if (out / "PanAf500").exists():
        print("[p7] PanAf500 already extracted")
        return
    print("[p7] extracting PanAf500 subset (skip PanAf20K)...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        names = z.namelist()
        panaf500 = [n for n in names if "PanAf500" in n]
        print(f"  PanAf500 entries: {len(panaf500)}/{len(names)}")
        for n in panaf500:
            z.extract(n, out)
    print(f"[p7] extracted to {out}")


def step2_build_dlc_config():
    """用 ASBAR 预训练 snapshot 重建最小 DLC config（推理用）"""
    import deeplabcut as dlc
    # OMC 关键点名表（从 ASBAR 源码/标注格式读取，找不到就用 OMC 标准 17 点）
    omc_keypoints = ["nose","left_eye","right_eye","left_ear","right_ear",
                     "left_shoulder","right_shoulder","left_elbow","right_elbow",
                     "left_wrist","right_wrist","left_hip","right_hip",
                     "left_knee","right_knee","left_ankle","right_ankle"]
    # 读 ASBAR 的 keypoints 定义文件（如果有更完整的）
    kpt_file = ASBAR / "pretrained_models" / "openmonkeychallenge_keypoints.txt"
    if kpt_file.exists():
        omc_keypoints = [l.strip() for l in kpt_file.read_text().splitlines() if l.strip()]
    print(f"[p7] OMC keypoints ({len(omc_keypoints)}): {omc_keypoints[:6]}...")
    return omc_keypoints


def step3_analyze_videos(omc_keypoints, max_videos=100):
    """DLC analyze_videos 批量提取 PanAf500 骨架"""
    import deeplabcut as dlc
    video_dir = EXTRACT_DIR / "PanAf500" / "videos"
    videos = sorted(video_dir.glob("*.mp4"))[:max_videos]
    print(f"[p7] analyzing {len(videos)} videos with pretrained DLC...")
    # DLC analyze 需要项目 config——ASBAR 仓没有完整 config.yaml，
    # 用 dlc.create_functional_videos 或直接从 snapshot 目录组装。
    # 此处记录 TODO：如果 DLC 3.x API 需要 config，用临时 project 包 snapshot 权重。
    results = {}
    for vi, vp in enumerate(videos):
        try:
            h5s = dlc.analyze_videos(str(DLC_SNAP_DIR), videos=[str(vp)],
                                     videotype="mp4", shuffle=1, save_as_csv=True)
            results[vp.stem] = h5s
            if vi % 10 == 0:
                print(f"  [{vi}/{len(videos)}] {vp.stem}")
        except Exception as e:
            print(f"  [warn] {vp.stem}: {e}")
    return results


def step4_map_to_psd24(dlc_results):
    """DLC 灵长类 kpt → PSD 24-slot。输入 (frames, V, 3)；输出 (frames, 24, 3)。"""
    import numpy as np
    name_to_idx = {}
    # 从 DLC 输出读关键点名（h5 columns 第三层）
    # 简化：按 OMC 标准顺序
    omc_keypoints = ["nose","left_eye","right_eye","left_ear","right_ear",
                     "left_shoulder","right_shoulder","left_elbow","right_elbow",
                     "left_wrist","right_wrist","left_hip","right_hip",
                     "left_knee","right_knee","left_ankle","right_ankle"]
    for i, name in enumerate(omc_keypoints):
        name_to_idx[name.lower()] = i
    out = {}
    for vid, arr in dlc_results.items():
        F = arr.shape[0]
        psd = np.zeros((F, 24, 3), dtype=np.float32)
        for psd_slot, omc_name in OMC_TO_PSD24.items():
            if omc_name is None:
                continue
            oi = name_to_idx.get(omc_name.lower())
            if oi is None or oi >= arr.shape[1]:
                continue
            psd[:, psd_slot, :] = arr[:, oi, :]
        out[vid] = psd
    return out


def step5_build_pkl(psd_by_vid):
    """PanAf500 JSON frame 标注 → clip label（多数投票）→ pkl"""
    ann_dir = EXTRACT_DIR / "PanAf500" / "annotations"
    clips = []
    for vid, psd in sorted(psd_by_vid.items()):
        # 找对应 JSON 标注
        jfiles = list(ann_dir.rglob(f"{vid}*.json"))
        if not jfiles:
            continue
        j = json.load(open(jfiles[0], encoding="utf-8"))
        # frame-level behaviors → clip 多数投票
        votes = Counter()
        for frame in j.get("frames", j if isinstance(j, list) else []):
            for bbox in (frame.get("bboxes", []) if isinstance(frame, dict) else []):
                b = bbox.get("behavior", "").lower().replace(" ", "_")
                if b in PANAF_BEHAVIORS:
                    votes[b] += 1
        if not votes:
            continue
        label = votes.most_common(1)[0][0]
        clips.append({"keypoints": psd, "label": PANAF_BEHAVIORS.index(label),
                      "psd_class": label, "video_id": vid,
                      "split": "train" if hash(vid) % 5 else "val",  # 80/20 split
                      "boundary": [0.0] * psd.shape[0]})
    with open(OUT_PKL, "wb") as f:
        pickle.dump(clips, f)
    print(f"[p7] saved {len(clips)} clips → {OUT_PKL}")
    return len(clips)


if __name__ == "__main__":
    t0 = time.time()
    step0_check()
    step1_extract_zip()
    kp = step2_build_dlc_config()
    print("[p7] NOTE: step3 DLC inference needs TF backend; run when GPU free.")
    print(f"[p7] prep done in {time.time()-t0:.0f}s")
