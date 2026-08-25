#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BVH → (T,V,3) 关键点序列转换器 —— C3/W27 动捕犬数据链 (MANN DogSet)。

用途
----
把 AI4Animation SIGGRAPH2018 "Mode-Adaptive Neural Networks for Quadruped Motion Control"
发布的真实犬类动捕 BVH（51 条，60fps，厘米制）转换为格式 B 关键点序列：
    {keypoints:(T,V,C=3), topology_name, V, fps_or_sampling, source, split}

⚠️ 防语义事故硬要求：本脚本内嵌 JOINT_MAPPING 映射表。MANN 骨架沿用人形命名，
   但 LeftArm/RightArm 是【前腿】而非手臂，UpLeg/Leg 是【后腿】。
   一律按"关节名→规范语义名"显式映射并校验，禁止任何按位置/索引的隐式假设。

用法
----
    # 全量转换（默认输出 runs/data_campaign/mocap/sequences/）
    python scripts/bvh_dogset_to_sequence.py \
        --src external/dogset-mann-siggraph2018/raw \
        --out-dir runs/data_campaign/mocap/sequences \
        --manifest runs/data_campaign/mocap/manifest.jsonl

    # 自测（内置合成 BVH 解析验证，无外部依赖）
    python scripts/bvh_dogset_to_sequence.py --self-test

许可注记
--------
源数据仅限研究/教育用途，禁止商用与再分发（University of Edinburgh IP）。
见 external/dogset-mann-siggraph2018/raw 内 LICENSE 与 dev-docs/research/MOCAP_DATASETS.md。
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 1. 关节映射表（防语义事故核心，勿动）
# ─────────────────────────────────────────────────────────────────────────────
# 规范式样: (规范语义名, BVH原始名)。canonical index = 本列表顺序。
# 语义审查要点（2026-08-25 W27 调研结论，证据 dev-docs/research/MOCAP_DATASETS.md §4）:
#   * MANN 犬骨架复用人形命名模板：
#     - LeftArm / RightArm      → 前腿上段(肱骨)，【不是】手臂
#     - LeftHand / RightHand    → 前爪，【不是】手
#     - LeftUpLeg / LeftLeg     → 后腿股骨/胫骨（每侧仅 3 关节）
#     - Tail / Tail1            → 尾椎两节，犬类特有，人形骨架不存在
#   * 若上游换数据集导致名称集合变化，必须先人工修订本表再跑，脚本会拒绝静默跳过。
JOINT_MAPPING: list[tuple[str, str]] = [
    ("hips",       "Hips"),
    ("spine_a",    "Spine"),
    ("spine_b",    "Spine1"),
    ("neck",       "Neck"),
    ("head",       "Head"),
    ("tail_a",     "Tail"),
    ("tail_b",     "Tail1"),
    ("fl_scapula", "LeftShoulder"),
    ("fl_upper",   "LeftArm"),        # 前腿！非手臂
    ("fl_lower",   "LeftForeArm"),
    ("fl_paw",     "LeftHand"),
    ("fr_scapula", "RightShoulder"),
    ("fr_upper",   "RightArm"),       # 前腿！非手臂
    ("fr_lower",   "RightForeArm"),
    ("fr_paw",     "RightHand"),
    ("hl_femur",   "LeftUpLeg"),      # 后腿
    ("hl_tibia",   "LeftLeg"),
    ("hl_paw",     "LeftFoot"),
    ("hr_femur",   "RightUpLeg"),
    ("hr_tibia",   "RightLeg"),
    ("hr_paw",     "RightFoot"),
]

TOPOLOGY_NAME = "mann_dogset_21j"
EXPECTED_BVH_NAMES = frozenset(bvh for _, bvh in JOINT_MAPPING)

SOURCE_URL = "https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip"
SOURCE_PAPER = "Zhang & Starke et al., Mode-Adaptive Neural Networks for Quadruped Motion Control, SIGGRAPH 2018"
LICENSE_NOTE = (
    "research/education only; no commercial use or redistribution; "
    "(c) University of Edinburgh; cite SIGGRAPH 2018 paper"
)


# ─────────────────────────────────────────────────────────────────────────────
# 2. BVH 解析
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Joint:
    name: str
    offset: np.ndarray                 # (3,) 厘米
    channels: list[str] = field(default_factory=list)  # 如 ['Zrotation','Xrotation','Yrotation']
    parent: int = -1                   # 父关节索引；根为 -1


@dataclass
class BvhMotion:
    joints: list[Joint]
    n_frames: int
    frame_time: float                  # 秒
    values: np.ndarray                 # (n_frames, total_channels) float32


def _tokens(text: str):
    return re.findall(r"[^\s]+", text)


def parse_bvh(path: Path) -> BvhMotion:
    """通用 BVH 解析：逐关节读取 OFFSET/CHANNELS，尊重各自欧拉轴顺序。"""
    tokens = _tokens(Path(path).read_text(encoding="utf-8", errors="replace"))
    i, n = 0, len(tokens)

    def expect(tok: str) -> None:
        nonlocal i
        if tokens[i] != tok:
            raise ValueError(f"{path.name}: 期望 '{tok}' 实得 '{tokens[i]}' @token{i}")
        i += 1

    expect("HIERARCHY")
    expect("ROOT")
    joints: list[Joint] = []
    stack: list[int] = []

    def read_joint(is_root: bool) -> None:
        nonlocal i
        name = tokens[i]; i += 1
        expect("{")
        expect("OFFSET")
        off = np.array([float(tokens[i]), float(tokens[i + 1]), float(tokens[i + 2])]); i += 3
        parent = stack[-1] if stack else -1
        idx = len(joints)
        joints.append(Joint(name=name, offset=off, parent=parent))
        stack.append(idx)
        if tokens[i] == "CHANNELS":
            i += 1
            cnt = int(tokens[i]); i += 1
            joints[idx].channels = tokens[i:i + cnt]
            i += cnt
        while i < n and tokens[i] == "JOINT":
            i += 1
            read_joint(False)
        if i < n and tokens[i] == "End":
            i += 2  # 'End' 'Site'
            expect("{")
            while tokens[i] != "}":
                i += 1
            expect("}")
        expect("}")
        stack.pop()

    read_joint(True)

    while i < n and tokens[i] != "MOTION":
        i += 1
    expect("MOTION")
    expect("Frames:")
    n_frames = int(tokens[i]); i += 1
    expect("Frame")
    expect("Time:")
    frame_time = float(tokens[i]); i += 1

    total_ch = sum(len(j.channels) for j in joints)
    vals = np.asarray(tokens[i:i + n_frames * total_ch], dtype=np.float64).astype(np.float32)
    if vals.size != n_frames * total_ch:
        raise ValueError(
            f"{path.name}: 运动数据数量不符 frames={n_frames}×ch={total_ch} vs got={vals.size}"
        )
    return BvhMotion(joints=joints, n_frames=n_frames, frame_time=frame_time,
                     values=vals.reshape(n_frames, total_ch))


# ─────────────────────────────────────────────────────────────────────────────
# 3. 正向运动学 → 全局关节位置 (T,V,3)
# ─────────────────────────────────────────────────────────────────────────────
def _axis_rot(axis: str, deg: np.ndarray) -> np.ndarray:
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    if axis == "X":
        m = np.zeros((len(r), 3, 3), dtype=np.float64)
        m[:, 0, 0] = 1; m[:, 1, 1] = c; m[:, 1, 2] = -s
        m[:, 2, 1] = s; m[:, 2, 2] = c
        return m
    if axis == "Y":
        m = np.zeros((len(r), 3, 3), dtype=np.float64)
        m[:, 1, 1] = 1; m[:, 0, 0] = c; m[:, 0, 2] = s
        m[:, 2, 0] = -s; m[:, 2, 2] = c
        return m
    if axis == "Z":
        m = np.zeros((len(r), 3, 3), dtype=np.float64)
        m[:, 2, 2] = 1; m[:, 0, 0] = c; m[:, 0, 1] = -s
        m[:, 1, 0] = s; m[:, 1, 1] = c
        return m
    raise ValueError(f"未知旋转轴 {axis}")


def fk_positions(motion: BvhMotion) -> np.ndarray:
    """返回 (T, V_all, 3)：所有可动关节的全局位置，厘米制、BVH 原坐标系。"""
    T, J = motion.n_frames, len(motion.joints)
    ch_index = []
    for j in motion.joints:
        cols = [motion.joints[:J] ]  # placeholder, replaced below
        ch_index.append(None)
    # 通道列偏移
    offsets, cur = [], 0
    for j in motion.joints:
        offsets.append(cur)
        cur += len(j.channels)

    world_pos = np.zeros((T, J, 3), dtype=np.float64)
    # 逐关节累积世界矩阵（迭代实现，避免深递归）
    mats = np.tile(np.eye(4), (T, J, 1, 1))

    order = sorted(range(J), key=lambda k: 0 if motion.joints[k].parent == -1 else 1)
    # 保证父在子前：拓扑序（BVH 天然父在前，直接按定义序即可）
    for ji in range(J):
        jt = motion.joints[ji]
        base = offsets[ji]
        loc = np.tile(np.eye(4), (T, 1, 1))
        rot_axes = [(c[0], base + k) for k, c in enumerate(jt.channels) if c.endswith("rotation")]
        if not rot_axes:
            R = np.tile(np.eye(3), (T, 1, 1))
        else:
            R = np.tile(np.eye(3), (T, 1, 1))
            for axis, col in rot_axes:  # 列向量约定：R = M_first @ ... @ M_last
                R = _axis_rot(axis, motion.values[:, col]) @ R
        loc[:, :3, :3] = R
        loc[:, :3, 3] = jt.offset[None, :]
        if jt.parent == -1:
            pos_axes = [(c[0], base + k) for k, c in enumerate(jt.channels) if c.endswith("position")]
            p = np.zeros((T, 3), dtype=np.float64)
            for axis, col in pos_axes:
                ax = {"X": 0, "Y": 1, "Z": 2}[axis]
                p[:, ax] = motion.values[:, col]
            loc[:, :3, 3] += p
            mats[:, ji] = loc
        else:
            mats[:, ji] = mats[:, jt.parent] @ loc
        world_pos[:, ji, :] = mats[:, ji][:, :3, 3]
    return world_pos


def map_to_canonical(all_pos: np.ndarray, joints: list[Joint]) -> tuple[np.ndarray, dict]:
    """按名字映射到规范序 (T,V=21,3)；名字集合不匹配立即报错（防语义事故）。"""
    name2idx = {j.name: k for k, j in enumerate(joints)}
    got = set(name2idx)
    if got != EXPECTED_BVH_NAMES:
        missing = sorted(EXPECTED_BVH_NAMES - got)
        extra = sorted(got - EXPECTED_BVH_NAMES)
        raise ValueError(
            "关节名集合与映射表不符——拒绝静默转换。\n"
            f"  缺失: {missing}\n  多余: {extra}\n"
            "如确为合法新拓扑，请人工修订 scripts/bvh_dogset_to_sequence.py 的 JOINT_MAPPING。"
        )
    sel = [name2idx[bvh_name] for _, bvh_name in JOINT_MAPPING]
    meta = {"topology_name": TOPOLOGY_NAME,
            "joint_order_canonical": [canon for canon, _ in JOINT_MAPPING],
            "joint_order_bvh": [bvh for _, bvh in JOINT_MAPPING]}
    return all_pos[:, sel, :].astype(np.float32), meta


# ─────────────────────────────────────────────────────────────────────────────
# 4. 单文件转换 + 质量体检
# ─────────────────────────────────────────────────────────────────────────────
def convert_one(bvh_path: Path) -> tuple[np.ndarray, dict]:
    motion = parse_bvh(bvh_path)
    all_pos = fk_positions(motion)
    kp, meta = map_to_canonical(all_pos, motion.joints)
    meta.update({
        "fps_or_sampling": round(1.0 / motion.frame_time, 3),
        "T": int(kp.shape[0]),
        "duration_sec": round(kp.shape[0] / (1.0 / motion.frame_time), 2),
        "units": "centimeter",
        "source": SOURCE_URL,
        "source_paper": SOURCE_PAPER,
        "license_note": LICENSE_NOTE,
    })
    return kp, meta


def quality_check(kp: np.ndarray) -> dict:
    """NaN/骨长稳定性硬门禁 + 速度尖峰软标记。

    硬失败（数据不可用/FK错误）: NaN、骨长变异系数≥5%。
    软标记（源数据毛刺，如实记录不丢弃）: 单关节帧间峰值速度>1500cm/s(15m/s)
    ——2026-08-25 W27 诊断证实为源动捕标记遮挡/交换毛刺（单关节孤立跳变，
    根速度平滑，骨长恒定），下游规则种子/伪标签环节按 manifest 的 qc_flag 过滤。
    """
    T, V, C = kp.shape
    assert C == 3, f"C 必须为 3, got {C}"
    nan_cnt = int(np.isnan(kp).sum())
    if nan_cnt:
        raise ValueError(f"含 NaN {nan_cnt} 处")
    # 相邻帧速度（cm/s）
    vel = np.linalg.norm(np.diff(kp, axis=0), axis=-1) * 60.0  # 60fps
    vmax = float(vel.max())
    # 骨长稳定性（相邻规范关节对）
    bone_pairs = [(0, 1), (1, 2), (2, 3), (5, 6),
                  (7, 8), (8, 9), (9, 10), (11, 12), (12, 13), (13, 14),
                  (15, 16), (16, 17), (18, 19), (19, 20)]
    bl = np.linalg.norm(kp[:, [a for a, _ in bone_pairs]] - kp[:, [b for _, b in bone_pairs]],
                        axis=-1)
    cv = float(bl.std(axis=0).mean() / max(bl.mean(axis=0).mean(), 1e-6))
    if cv > 0.05:  # 刚体骨长帧间变异系数应 <5%
        raise ValueError(f"骨长变异系数 {cv:.3f} ≥0.05，FK 或映射有误")
    return {"v_max_cm_s": round(vmax, 1),
            "bone_len_cv": round(cv, 4),
            "bbox_diag_cm": round(float(np.linalg.norm(kp.max(axis=(0, 1)) - kp.min(axis=(0, 1)))), 1),
            "qc_flag": "suspect_glitch" if vmax > 1500 else "ok"}


# ─────────────────────────────────────────────────────────────────────────────
# 5. 批量入口
# ─────────────────────────────────────────────────────────────────────────────
def run_batch(src: Path, out_dir: Path, manifest_path: Path, limit: int | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(src.rglob("*.bvh"))
    if limit:
        files = files[:limit]
    if not files:
        raise SystemExit(f"未找到 BVH: {src}")

    lines = []
    summary = []
    for f in files:
        kp, meta = convert_one(f)
        qc = quality_check(kp)
        seq_id = f.stem  # D1_xxx_KANxx_xxx
        rec = {
            "sequence_id": seq_id,
            "source_channel": "mocap_c3_w27",
            "origin_url_or_path": SOURCE_URL,
            "capture_context": "optical dog mocap, indoor studio (SIGGRAPH2018 DogSet)",
            "species_note": "dog (real motion capture)",
            "license_note": LICENSE_NOTE,
            "collected_at": str(date.today()),
            "keypoints_file": f"{seq_id}.pkl",
            "topology_name": TOPOLOGY_NAME,
            "V": kp.shape[1],
            "T": kp.shape[0],
            "fps": meta["fps_or_sampling"],
            "split": "unsplit",
            "qc_flag": qc["qc_flag"],
            "v_max_cm_s": qc["v_max_cm_s"],
            "bone_len_cv": qc["bone_len_cv"],
        }
        payload = {
            "keypoints": kp,
            "topology_name": TOPOLOGY_NAME,
            "V": kp.shape[1],
            "fps_or_sampling": meta["fps_or_sampling"],
            "source": SOURCE_URL,
            "source_paper": SOURCE_PAPER,
            "license_note": LICENSE_NOTE,
            "split": "unsplit",
            "joint_order_canonical": meta["joint_order_canonical"],
        }
        with open(out_dir / f"{seq_id}.pkl", "wb") as fh:
            pickle.dump(payload, fh)
        lines.append(json.dumps(rec, ensure_ascii=False))
        summary.append({"sequence_id": seq_id, **{k: meta[k] for k in
                        ("T", "duration_sec", "fps_or_sampling")}, **qc})
        flag = " ⚠glitch" if qc["qc_flag"] != "ok" else ""
        print(f"[ok] {seq_id}: T={meta['T']} dur={meta['duration_sec']}s "
              f"vmax={qc['v_max_cm_s']}cm/s boneCV={qc['bone_len_cv']}{flag}")

    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tot_T = sum(s["T"] for s in summary)
    print(f"\n=== 完成 {len(summary)} 条 | 总帧数 {tot_T} ({tot_T/60:.1f}s@60fps) ===")
    print(f"manifest → {manifest_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. 自测：合成 BVH（已知解析答案）验证解析器+FK+映射全链路
# ─────────────────────────────────────────────────────────────────────────────
SELF_TEST_BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 100 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 10 0 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Spine1
    {
      OFFSET 10 0 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT Neck
      {
        OFFSET 10 0 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT Head
        {
          OFFSET 10 0 0
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET 5 0 0
          }
        }
      }
      JOINT LeftShoulder
      {
        OFFSET 0 0 5
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT LeftArm
        {
          OFFSET 5 0 0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT LeftForeArm
          {
            OFFSET 5 0 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT LeftHand
            {
              OFFSET 5 0 0
              CHANNELS 3 Zrotation Xrotation Yrotation
              End Site
              {
                OFFSET 3 0 0
              }
            }
          }
        }
      }
      JOINT RightShoulder
      {
        OFFSET 0 0 -5
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT RightArm
        {
          OFFSET 5 0 0
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT RightForeArm
          {
            OFFSET 5 0 0
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT RightHand
            {
              OFFSET 5 0 0
              CHANNELS 3 Zrotation Xrotation Yrotation
              End Site
              {
                OFFSET 3 0 0
              }
            }
          }
        }
      }
    }
  }
  JOINT Tail
  {
    OFFSET -5 5 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT Tail1
    {
      OFFSET -5 5 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET -3 0 0
      }
    }
  }
  JOINT LeftUpLeg
  {
    OFFSET 0 -5 5
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT LeftLeg
    {
      OFFSET 0 -5 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT LeftFoot
      {
        OFFSET 0 -5 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0 -1 0
        }
      }
    }
  }
  JOINT RightUpLeg
  {
    OFFSET 0 -5 -5
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT RightLeg
    {
      OFFSET 0 -5 0
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT RightFoot
      {
        OFFSET 0 -5 0
        CHANNELS 3 Zrotation Xrotation Yrotation
        End Site
        {
          OFFSET 0 -1 0
        }
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0166667
"""


def self_test() -> None:
    import tempfile
    # 两帧全零旋转：全局位置必须严格等于层级 OFFSET 沿链累加（验证解析/FK 无旋转耦合）
    # 每帧通道数 = 根6 + 其余20关节×3 = 66
    zero_frame = " ".join(["0"] * 66)
    bvh_text = SELF_TEST_BVH + f"{zero_frame}\n{zero_frame}\n"
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "synthetic.bvh"
        p.write_text(bvh_text, encoding="utf-8")
        kp, meta = convert_one(p)
    canon = {n: i for i, n in enumerate(meta["joint_order_canonical"])}
    assert kp.shape == (2, 21, 3), f"形状错误 {kp.shape}"
    # 关键地标位（零旋转 ⇒ 型位=OFFSET 累加）：
    assert np.allclose(kp[0, canon["hips"]], [0, 100, 0]), f"hips {kp[0, canon['hips']]}"
    assert np.allclose(kp[0, canon["head"]], [40, 100, 0]), f"head {kp[0, canon['head']]}"
    # 前爪：Spine1(x=20)+肩(z+5)+三段前腿(各+5x) ⇒ (35,100,5)
    assert np.allclose(kp[0, canon["fl_paw"]], [35, 100, 5]), f"fl_paw {kp[0, canon['fl_paw']]}"
    assert np.allclose(kp[0, canon["fr_paw"]], [35, 100, -5])
    # 尾巴挂在髋部而非头部：(-5,+5)+(−5,+5) ⇒ (-10,110,0)
    assert np.allclose(kp[0, canon["tail_b"]], [-10, 110, 0]), f"tail_b {kp[0, canon['tail_b']]}"
    # 后爪：髋下三段各 -5y ⇒ (0,85,±5)
    assert np.allclose(kp[0, canon["hl_paw"]], [0, 85, 5]), f"hl_paw {kp[0, canon['hl_paw']]}"
    assert np.allclose(kp[0, canon["hr_paw"]], [0, 85, -5])
    qc = quality_check(kp)
    assert qc["bone_len_cv"] < 0.05
    print("[self-test] PASS — 解析/FK/映射/质检全链路 OK,", f"kp{kp.shape}, qc={qc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=Path("external/dogset-mann-siggraph2018/raw"))
    ap.add_argument("--out-dir", type=Path, default=Path("runs/data_campaign/mocap/sequences"))
    ap.add_argument("--manifest", type=Path, default=Path("runs/data_campaign/mocap/manifest.jsonl"))
    ap.add_argument("--limit", type=int, default=None, help="只转换前 N 条（调试用）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    run_batch(args.src, args.out_dir, args.manifest, args.limit)


if __name__ == "__main__":
    main()
