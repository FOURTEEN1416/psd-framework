# -*- coding: utf-8 -*-
"""P7 PanAf 骨架提取 — YOLO11x-pose ape 微调模型推理管线（v2, 2026-09-06）。

路线变更记录: v1 用 ASBAR 预训练 DLC snapshot-60000, 但其为 TF checkpoint,
DLC 3.0.1(pytorch) 无法加载且无官方权重转换器, DLC 3.x analyze_videos 亦需完整
项目 config(ASBAR 仓不含) → DLC 路线不可行。v2 改与 P2' 同栈: AP-10K
(chimpanzee/gorilla/orangutan, 17kpt, LibreYOLO 转换版, CC-BY-4.0) 微调
yolo11x-pose(COCO 17kpt, 零拓扑重塑) → PanAf500 bbox 裁剪推理 → 17→PSD24
映射(四肢/躯干功能对应, 7 死槽置 0, 与 L8 20/24 口径同构) → pkl 装配。

PanAf500 结构(实查):
  panaf_extract/1h73erszj3ckn2qjwm4sqmr2wt/PanAf500/videos/*.mp4 (500)
  .../annotations/{train:400,validation:25,test:75}/*.json
  JSON: {"video": id, "annotations": [{"frame_id": int,
         "detections": [{"bbox": [x1,y1,x2,y2], "ape_id", "species", "behaviour"}]}]}
官方 split 直接采用(train/val/test = 400/25/75), 不再 hash。

用法:
    .venv/Scripts/python.exe -u scripts/run_p21_panaf_pipeline.py --max-videos 3   # 烟测
    .venv/Scripts/python.exe -u scripts/run_p21_panaf_pipeline.py                  # 全量 500
产出:
    runs/p7_asbar/panaf500_T30.pkl   (full12 格式: keypoints T30×24×3)
    reports/p7-extract-quality.json  (检测率/conf 分布/行为标签统计)
    runs/p7_asbar/vis_check/*.jpg    (骨架叠加抽查帧)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

PANAF = REPO / "runs" / "p7_asbar" / "panaf_extract" / "1h73erszj3ckn2qjwm4sqmr2wt" / "PanAf500"
VIDEO_DIR = PANAF / "videos"
ANN_DIR = PANAF / "annotations"
APE_MODEL = Path(r"D:\Desktop\k9-training-system\runs\pose\runs\p7_ape_pose\weights\best.pt")
OUT_PKL = REPO / "runs" / "p7_asbar" / "panaf500_T30.pkl"
VIS_DIR = REPO / "runs" / "p7_asbar" / "vis_check"
QUALITY_JSON = REPO / "reports" / "p7-extract-quality.json"

T_CLIP = 30
CONF_MIN = 0.20       # 关键点置信门(低于置 0); 与 psd 犬科管线预处理口径一致
BBOX_PAD = 0.10       # bbox 外扩比例
VIS_CHECK_N = 6       # 随机抽查视频数

PANAF_BEHAVIORS = ["sitting", "standing", "walking", "running", "climbing_up",
                   "climbing_down", "hanging", "sitting_on_back", "camera_interaction"]

# AP-10K 17kpt (LibreYOLO 版, README 顺序) → PSD 24-slot
# AP17: 0 left_eye 1 right_eye 2 nose 3 neck 4 root_of_tail
#       5 left_shoulder 6 right_shoulder 7 left_elbow 8 right_elbow
#       9 left_front_paw 10 right_front_paw 11 left_hip 12 right_hip
#       13 left_knee 14 right_knee 15 left_back_paw 16 right_back_paw
# PSD24: 0 head 1/2 eye 3 withers 4 throat 5 R_shoulder 6 R_elbow 7 R_wrist
#        8 L_shoulder 9 L_elbow 10 L_wrist 11 R_hip 12 R_knee 13 R_ankle
#        14 L_hip 15 L_knee 16 L_ankle 17 tail_base 18-23 dead
AP17_TO_PSD24 = {
    2: 0,        # nose → head
    0: 1, 1: 2,  # eyes
    3: 3,        # neck → withers
    6: 5, 8: 6, 10: 7,       # right shoulder/elbow/front_paw(wrist)
    5: 8, 7: 9, 9: 10,       # left shoulder/elbow/front_paw(wrist)
    12: 11, 14: 12, 16: 13,  # right hip/knee/back_paw(ankle)
    11: 14, 13: 15, 15: 16,  # left hip/knee/back_paw(ankle)
    4: 17,       # root_of_tail → tail_base
}


def load_annotations() -> dict:
    """→ {vid: {"split", "frames": {fid: {"beh": Counter, "bbox": [x1,y1,x2,y2]|None}}, "species"}}"""
    out = {}
    for split in ("train", "validation", "test"):
        for jf in sorted((ANN_DIR / split).glob("*.json")):
            j = json.loads(jf.read_text(encoding="utf-8"))
            vid = j["video"]
            frames = {}
            species = set()
            for ann in j.get("annotations", []):
                fid = int(ann["frame_id"])
                slot = frames.setdefault(fid, {"beh": Counter(), "bbox": None})
                best_area = -1.0
                for det in ann.get("detections", []):
                    b = str(det.get("behaviour", "")).strip().lower().replace(" ", "_")
                    species.add(str(det.get("species", "")).lower())
                    if b in PANAF_BEHAVIORS:
                        slot["beh"][b] += 1
                    bbox = det.get("bbox")
                    if bbox and len(bbox) == 4:
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        if area > best_area:
                            best_area = area
                            slot["bbox"] = [float(v) for v in bbox]
            out[vid] = {"split": {"validation": "val"}.get(split, split),
                        "frames": frames, "species": species}
    n = {s: sum(1 for v in out.values() if v["split"] == s) for s in ("train", "val", "test")}
    print(f"[p7] annotations: {len(out)} videos (train {n['train']} / val {n['val']} / test {n['test']})")
    return out


def infer_video(model, vid: str, frames: dict) -> tuple[np.ndarray, dict]:
    """顺序解码 + 命中标注帧才推理 → kpts (F,17,3) 原图坐标 + 质量统计。
    (cap.set 逐帧 seek 随机访问慢一个量级, 500 视频全量按顺序读省数小时)"""
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{vid}.mp4"))
    fids = sorted(frames.keys())
    kpts = np.zeros((len(fids), 17, 3), dtype=np.float32)
    n_det = 0
    fid_i, fid_no = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or fid_i >= len(fids):
            break
        if fid_no == fids[fid_i]:
            slot = frames[fids[fid_i]]
            kpts[fid_i] = model_frame(model, frame, slot["bbox"])
            if slot["bbox"] is not None:
                n_det += 1
            fid_i += 1
        fid_no += 1
    cap.release()
    mean_conf = float(kpts[..., 2][kpts[..., 2] > 0].mean()) if (kpts[..., 2] > 0).any() else 0.0
    return kpts, {"n_frames": len(fids), "n_detected": n_det,
                  "mean_conf": round(mean_conf, 3)}


def model_frame(model, frame: np.ndarray, bbox) -> np.ndarray:
    """单帧 bbox 裁剪推理 → (17,3) 原图坐标; 低 conf 关键点置 0。"""
    H, W = frame.shape[:2]
    if bbox is None:
        x1, y1, x2, y2, ox, oy = 0, 0, W, H, 0, 0
    else:
        bx1, by1, bx2, by2 = bbox
        pw, ph = (bx2 - bx1) * BBOX_PAD, (by2 - by1) * BBOX_PAD
        x1, y1 = max(0, int(bx1 - pw)), max(0, int(by1 - ph))
        x2, y2 = min(W, int(bx2 + pw)), min(H, int(by2 + ph))
        ox, oy = x1, y1
    crop = frame[y1:y2, x1:x2]
    res = model.predict(crop, verbose=False, conf=CONF_MIN, device=0)[0]
    out = np.zeros((17, 3), dtype=np.float32)
    if res.keypoints is not None and len(res.keypoints.data):
        k = res.keypoints.data[0].cpu().numpy()  # 置信最高实例
        out[:, :2] = k[:, :2] + np.array([ox, oy], dtype=np.float32)
        out[:, 2] = k[:, 2]
        low = k[:, 2] < CONF_MIN
        out[low, :2] = 0
        out[low, 2] = 0
    return out


def to_psd24(kpts17: np.ndarray) -> np.ndarray:
    """(F,17,3) → (F,24,3); 未映射槽位保持 0(与 L8 有效监督口径同构)。"""
    F = kpts17.shape[0]
    psd = np.zeros((F, 24, 3), dtype=np.float32)
    for src, dst in AP17_TO_PSD24.items():
        psd[:, dst, :] = kpts17[:, src, :]
    return psd


def resample_t(kps: np.ndarray, T: int = T_CLIP) -> np.ndarray:
    F = kps.shape[0]
    if F == 0:
        return np.zeros((T, kps.shape[1], kps.shape[2]), dtype=np.float32)
    idx = np.resize(np.arange(F), T) if F < T else np.linspace(0, F - 1, T, dtype=int)
    return kps[idx]


def assemble_pkl(clips: list, per_video: dict, t0: float):
    """→ panaf500_T30.pkl (full12 格式) + quality json。"""
    label_to_idx = {b: i for i, b in enumerate(PANAF_BEHAVIORS)}
    rows = []
    for c in clips:
        psd = to_psd24(c["kpts"])
        T = resample_t(psd)
        T = T - T.mean(axis=(0, 1), keepdims=True)  # center 与 psd 管线一致
        rows.append({"keypoints": T.astype(np.float32),
                     "label": label_to_idx[c["label"]],
                     "psd_class": c["label"], "video_id": c["vid"],
                     "split": c["split"],
                     "boundary": [0.0] * T.shape[0]})
    with open(OUT_PKL, "wb") as f:
        pickle.dump(rows, f)
    n = Counter(r["split"] for r in rows)
    print(f"[p7] saved {len(rows)} clips (train {n['train']} / val {n['val']} / test {n['test']}) → {OUT_PKL}")

    det_rates = [v["det_rate"] for v in per_video.values()]
    confs = [v["mean_conf"] for v in per_video.values() if v["mean_conf"] > 0]
    labels = Counter(v["label"] for v in per_video.values())
    q = {"date": datetime.now().isoformat(timespec="seconds"),
         "model": str(APE_MODEL),
         "n_videos": len(per_video),
         "det_rate_mean": round(float(np.mean(det_rates)), 3) if det_rates else 0,
         "det_rate_min": round(float(np.min(det_rates)), 3) if det_rates else 0,
         "mean_conf_over_detected": round(float(np.mean(confs)), 3) if confs else 0,
         "label_distribution": dict(labels),
         "behavior_taxonomy": PANAF_BEHAVIORS,
         "mapping": "AP17_TO_PSD24 (17 mapped slots / 7 dead slots 18-23 + throat)",
         "pkl": str(OUT_PKL), "wall_clock_sec": round(time.time() - t0, 1)}
    QUALITY_JSON.parent.mkdir(parents=True, exist_ok=True)
    QUALITY_JSON.write_text(json.dumps(q, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[p7] quality → {QUALITY_JSON}\n{json.dumps(q, ensure_ascii=False, indent=1)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-videos", type=int, default=0, help="0=all")
    ap.add_argument("--vis", type=int, default=VIS_CHECK_N)
    args = ap.parse_args()
    from ultralytics import YOLO
    assert APE_MODEL.exists(), f"ape 模型未找到: {APE_MODEL} (先跑 p7_finetune_ape_pose.py)"
    model = YOLO(str(APE_MODEL))

    t0 = time.time()
    anns = load_annotations()
    vids = sorted(anns.keys())
    if args.max_videos:
        vids = vids[:args.max_videos]

    per_video = {}
    clips = []
    vis_pick = set(np.random.default_rng(42).choice(len(vids),
                    size=min(args.vis, len(vids)), replace=False))
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    for vi, vid in enumerate(vids):
        a = anns[vid]
        votes = Counter()
        for slot in a["frames"].values():
            votes.update(slot["beh"])
        if not votes:
            continue
        kpts, stat = infer_video(model, vid, a["frames"])
        label = votes.most_common(1)[0][0]
        per_video[vid] = {**stat, "label": label, "species": sorted(a["species"]),
                          "det_rate": round(stat["n_detected"] / max(1, stat["n_frames"]), 3)}
        clips.append({"vid": vid, "kpts": kpts, "label": label, "split": a["split"]})
        if vi in vis_pick:
            _dump_vis(model, vid, a["frames"])
        if vi % 25 == 0:
            print(f"[p7] [{vi}/{len(vids)}] {vid} det={per_video[vid]['det_rate']} label={label} ({time.time()-t0:.0f}s)")
    assemble_pkl(clips, per_video, t0)


def _dump_vis(model, vid: str, frames: dict):
    cap = cv2.VideoCapture(str(VIDEO_DIR / f"{vid}.mp4"))
    fid = sorted(frames.keys())[max(0, len(frames) // 2)]
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return
    res = model.predict(frame, verbose=False, conf=CONF_MIN, device=0)[0]
    cv2.imwrite(str(VIS_DIR / f"{vid}_f{fid}.jpg"), res.plot())


if __name__ == "__main__":
    main()
