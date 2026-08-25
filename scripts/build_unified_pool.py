#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""W30: 统一真实扩展池组装器（unified real-expansion pool builder）.

任务书: dev-docs/handovers/NEXT-BATCH-plan.md §W30（领地内文件）
会师蓝图: dev-docs/handovers/DATA-CAMPAIGN-plan.md §3

四源汇聚:
  ak_public_q3b   AK partialclass4 有标 4 类 (T=30, V=24) —— 内容级断言，冒烟残留自动隔离
  aptv2_c2_w26    APTv2 canidae 轨迹 503 条 17kp → K9Graph 24kp 拓扑映射（本模块核心）
  mocap_c3_w27    DogSet 动捕 (T,21,3) 原样收编（运动学先验池，不映射——超本窗范围）
  dogpose_c5_w29  dog-pose 静态池 8476 条 K9Graph 24kp 原样收编（增广池）

关键裁决（详见 reports/unified-pool-w30-*.md）:
  ① 映射表: 17 源点全消费（16 直接 + neck→withers 近似），7 目标槽位诚实置 NaN:
     tail_end(13)/双耳base(14,15)/chin(17)/双耳tip(18,19)/throat(23)
  ② 时序策略三选一 = 仅预训练池: APTv2 保留原生 T=15 不插值不拼接，
     usage_scope="pretrain_geometric"；滑窗拼接/短序列训练支持的否决理由见报告
  ③ 规则种子打标降级 label_status="deferred_pixel_domain":
     psd/data/rule_seeds.py 特征层硬依赖度量 3D z 高度轴 + SMAL 索引组，
     APTv2 为 2D 图像像素 (x,y,vis)，阈值不可跨域迁移——标签适配归后续窗口

用法:
  & "D:\Desktop\psd-framework\.venv\Scripts\python.exe" scripts/build_unified_pool.py `
      --config configs/unified_pool.yaml
CPU-only；源目录只读；产物落 runs/data_campaign/unified/（gitignore，报告快照入 git）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # 直接执行本脚本时保证可 import psd

# ---------------------------------------------------------------------------
# 源通道常量（溯源 manifest 的 source_channel 枚举）
# ---------------------------------------------------------------------------

CH_AKV2 = "aptv2_c2_w26"
CH_MOCAP = "mocap_c3_w27"
CH_DOGPOSE = "dogpose_c5_w29"
CH_AK = "ak_public_q3b"

#: 用途域闭集（防下游自由文本漂移）
USAGE_SCOPES = {
    "pretrain_geometric",        # APTv2 微序列：自监督预训练 / 几何监督
    "kinematic_prior",           # DogSet 动捕：真实运动学先验（非行为分类主粮）
    "augment_static",            # dog-pose 静态 GT：单帧增广 / 预训练
    "supervised_partialclass4",  # AK partialclass4：有标监督样本
}

# ---------------------------------------------------------------------------
# ① APTv2 17kp → K9Graph 24kp 拓扑映射表
# ---------------------------------------------------------------------------

#: APTv2 官方 17 点名与顺序（权威出处: train_annotations.json categories.keypoints，
#: 2026-08-25 W30 当次核对；与 W26 报告 §5 一致）
APTV2_KPT_NAMES: List[str] = [
    "left_eye", "right_eye", "nose", "neck", "root_of_tail",
    "left_shoulder", "left_elbow", "left_front_paw",
    "right_shoulder", "right_elbow", "right_front_paw",
    "left_hip", "left_knee", "left_back_paw",
    "right_hip", "right_knee", "right_back_paw",
]

#: 映射状态语义:
#:   exact      同名同义或解剖学一一对应
#:   positional 肢体链位对齐（近端/中段/末端），解剖命名粗化不逐字一致
#:   approx     位置近似（保留不确定性注记）
APT2_TO_K9: Dict[int, Tuple[int, str]] = {
    # 头部
    0: (20, "exact"),       # left_eye  → left_eye（dog-pose 死关节事件中该点 APTv2 原生覆盖）
    1: (21, "exact"),       # right_eye → right_eye（同上）
    2: (16, "exact"),       # nose      → nose
    3: (22, "approx"),      # neck      → withers（颈背基部≈鬐甲，可能偏上，唯一近似点）
    # 尾部
    4: (12, "exact"),       # root_of_tail → tail_start（尾根同名同义）
    # 前左肢（链位: 近端/中段/末端）
    5: (2, "positional"),   # left_shoulder → front_left_elbow（K9Graph 近端简化命名）
    6: (1, "positional"),   # left_elbow    → front_left_knee （中段: 肘↔腕部膝位）
    7: (0, "exact"),        # left_front_paw→ front_left_paw
    # 前右肢
    8: (8, "positional"),   # right_shoulder→ front_right_elbow
    9: (7, "positional"),   # right_elbow   → front_right_knee
    10: (6, "exact"),       # right_front_paw→ front_right_paw
    # 后左肢（髋=近端 / 膝=中段 / 爪=末端）
    11: (5, "positional"),  # left_hip      → rear_left_elbow（K9Graph 后肢近端简化命名）
    12: (4, "exact"),       # left_knee     → rear_left_knee
    13: (3, "exact"),       # left_back_paw → rear_left_paw
    # 后右肢
    14: (11, "positional"), # right_hip     → rear_right_elbow
    15: (10, "exact"),      # right_knee    → rear_right_knee
    16: (9, "exact"),       # right_back_paw→ rear_right_paw
}

#: K9Graph 中无 APTv2 对应点的诚实槽位（NaN 坐标 + vis=0）:
#:   tail_end(13) 尾尖 | 双耳 base(14,15)/tip(18,19) | chin(17) 下巴 | throat(23) 喉部
K9_UNMAPPED_FROM_APTV2 = frozenset({13, 14, 15, 17, 18, 19, 23})

#: 24 槽位逐点语义理由（manifest/报告导出用；truth 与 K9Graph.NODE_NAMES 索引一致）
MAPPING_SEMANTICS: Dict[int, str] = {
    0: "front_left_paw ← left_front_paw: 前左肢末端着地点同名同义",
    1: "front_left_knee ← left_elbow: 前左肢中段链位对齐（APTv2 称肘/K9Graph 简化称腕部膝位）",
    2: "front_left_elbow ← left_shoulder: 前左肢近端链位对齐（肩胛点接躯干）",
    3: "rear_left_paw ← left_back_paw: 后左肢末端同名同义",
    4: "rear_left_knee ← left_knee: 后左肢膝关节同名同义",
    5: "rear_left_elbow ← left_hip: 后左肢近端链位对齐（髋点接骨盆侧）",
    6: "front_right_paw ← right_front_paw: 前右肢末端同名同义",
    7: "front_right_knee ← right_elbow: 前右肢中段链位对齐",
    8: "front_right_elbow ← right_shoulder: 前右肢近端链位对齐",
    9: "rear_right_paw ← right_back_paw: 后右肢末端同名同义",
    10: "rear_right_knee ← right_knee: 后右肢膝关节同名同义",
    11: "rear_right_elbow ← right_hip: 后右肢近端链位对齐",
    12: "tail_start ← root_of_tail: 尾根同名同义",
    13: "tail_end 无对应: APTv2 仅尾根单点，尾尖缺失——诚实置 NaN",
    14: "left_ear_base 无对应: APTv2 拓扑无耳部点——诚实置 NaN",
    15: "right_ear_base 无对应: APTv2 拓扑无耳部点——诚实置 NaN",
    16: "nose ← nose: 鼻尖同名同义",
    17: "chin 无对应: APTv2 拓扑无下巴点——诚实置 NaN",
    18: "left_ear_tip 无对应: APTv2 拓扑无耳部点——诚实置 NaN",
    19: "right_ear_tip 无对应: APTv2 拓扑无耳部点——诚实置 NaN",
    20: "left_eye ← left_eye: 左眼同名同义（dog-pose 死关节 4 点中 APTv2 原生覆盖的 2 点之一）",
    21: "right_eye ← right_eye: 右眼同名同义（同上）",
    22: "withers ← neck(approx): APTv2 无独立鬐甲定义，neck 标注于颈背基部与鬐甲相近但可能偏上——全表唯一近似点",
    23: "throat 无对应: APTv2 无喉部点（呼应死关节事件: dog-pose GT 该点亦零标注）——诚实置 NaN",
}


def map_aptv2_keypoints(kp17: np.ndarray) -> np.ndarray:
    """APTv2 (T,17,3) 图像像素 [x,y,vis01] → K9Graph (T,24,3).

    映射槽位复制 x,y 并透传 vis；不可映射槽位 x,y=NaN、vis=0（下游按缺失消费，
    与 ak_pose_extract.DEAD_JOINTS 硬掩码同一处置哲学）。
    """
    kp17 = np.asarray(kp17, dtype=np.float32)
    if kp17.ndim != 3 or kp17.shape[1] != 17 or kp17.shape[2] != 3:
        raise ValueError(f"期望 (T,17,3)，实际 {kp17.shape}")
    out = np.full((kp17.shape[0], 24, 3), np.nan, dtype=np.float32)
    out[:, :, 2] = 0.0
    for src, (dst, _status) in APT2_TO_K9.items():
        out[:, dst, :] = kp17[:, src, :]
    return out


# ---------------------------------------------------------------------------
# AK partialclass4 内容级断言（防冒烟残留静默混入——记忆库 2026-08-25 警报复现防御）
# ---------------------------------------------------------------------------

def check_ak_source(pkl_path: Path, manifest_path: Path,
                    expected_v: int = 24, expected_t: int = 30) -> Dict[str, Any]:
    """内容级校验 AK 池 pkl 与 manifest 的一致性.

    Returns:
        {"status": "ok"|"quarantined"|"missing", "reasons": [...],
         "n_clips": int, "n_manifest": int}
    """
    pkl_path, manifest_path = Path(pkl_path), Path(manifest_path)
    if not pkl_path.exists() or not manifest_path.exists():
        return {"status": "missing",
                "reasons": [f"pkl_exists={pkl_path.exists()} manifest_exists={manifest_path.exists()}"],
                "n_clips": 0, "n_manifest": 0}
    reasons: List[str] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.loads(f.read())
    n_manifest = len(manifest.get("samples", []))
    with open(pkl_path, "rb") as f:
        clips = pickle.load(f)
    if not isinstance(clips, list):
        reasons.append("pkl_not_list")
        clips = []
    n_clips = len(clips)
    if n_clips != n_manifest:
        reasons.append(f"count_mismatch: pkl {n_clips} != manifest {n_manifest}")
    gate = list(manifest.get("gate_classes", []))
    for i, clip in enumerate(clips):
        if not isinstance(clip, dict):
            reasons.append(f"clip[{i}] not_dict")
            continue
        kp = clip.get("keypoints")
        if not hasattr(kp, "shape") or tuple(kp.shape) != (expected_t, expected_v, 3):
            reasons.append(f"clip[{i}] keypoints shape {getattr(kp, 'shape', None)} != {(expected_t, expected_v, 3)}")
        need = {"keypoints", "label", "psd_class", "video_id", "split"}
        missing = need - set(clip.keys())
        if missing:
            reasons.append(f"clip[{i}] missing keys {sorted(missing)}")
            continue
        if gate and clip["psd_class"] not in gate:
            reasons.append(f"clip[{i}] psd_class {clip['psd_class']} 不在 gate_classes")
    status = "ok" if not reasons else "quarantined"
    return {"status": status, "reasons": reasons, "n_clips": n_clips, "n_manifest": n_manifest}


# ---------------------------------------------------------------------------
# 条目构造（统一 schema）
# ---------------------------------------------------------------------------

def build_aptv2_entry(seq: Dict[str, Any], sha256: Optional[str] = None) -> Dict[str, Any]:
    """APTv2 单序列 → 统一池条目（拓扑已映射至 K9Graph 24kp）。"""
    kp24 = map_aptv2_keypoints(np.asarray(seq["keypoints"]))
    sid = str(seq.get("sequence_id", "unknown"))
    provenance: Dict[str, Any] = {
        "dataset": "APTv2 (ViTAE-Transformer/APTv2)",
        "source_topology": "aptv2_quadruped_17kp",
        "species": seq.get("species"),
        "sequence_id": sid,
        "topology_mapping": {
            "method": "name_semantic_chain_position_v1",
            "mapped_points": len(APT2_TO_K9),
            "approx_points": ["withers(22)←neck(3)"],
            "unmapped_slots": sorted(K9_UNMAPPED_FROM_APTV2),
        },
        "visible_frac_mean_source": seq.get("visible_frac_mean"),
    }
    if sha256 is not None:
        provenance["source_file_sha256"] = sha256
    entry = {
        "sample_id": f"aptv2::{sid}",
        "source_channel": CH_AKV2,
        "split": seq.get("split"),
        "topology_name": "K9Graph",
        "V": 24,
        "T": int(kp24.shape[0]),
        "keypoints": kp24,
        "coords_semantic": "image_pixel_xy_vis01",
        "fps_or_sampling": seq.get("fps_or_sampling"),
        "usage_scope": "pretrain_geometric",
        "label_status": "deferred_pixel_domain",
        "static": False,
        "provenance": provenance,
    }
    return entry


def _build_mocap_entry(seq: Dict[str, Any], sid_hint: Optional[str] = None) -> Dict[str, Any]:
    """DogSet 动捕条目：原样收编 21 关节不映射（运动学先验用途）。

    sid_hint: manifest.jsonl 行内的 sequence_id（真实 pkl 字典无此键，必须外送）。
    """
    sid = str(sid_hint or seq.get("sequence_id") or Path(str(seq.get("source", ""))).stem)
    kp = np.asarray(seq["keypoints"], dtype=np.float32)
    return {
        "sample_id": f"mocap::{sid}",
        "source_channel": CH_MOCAP,
        "split": seq.get("split", "unsplit"),
        "topology_name": seq.get("topology_name", "mann_dogset_21j"),
        "V": int(kp.shape[1]),
        "T": int(kp.shape[0]),
        "keypoints": kp,
        "coords_semantic": "metric_cm_xyz",
        "fps_or_sampling": seq.get("fps_or_sampling"),
        "usage_scope": "kinematic_prior",
        "label_status": "not_applicable_prior",
        "static": False,
        "provenance": {
            "source_url": seq.get("source"),
            "source_paper": seq.get("source_paper"),
            "license_note": seq.get("license_note"),
            "joint_order_canonical": seq.get("joint_order_canonical"),
        },
    }


def _build_dogpose_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """dog-pose 静态条目：已是 K9Graph 24kp 合并式，补齐统一字段。"""
    out = dict(e)
    kp = np.asarray(out["keypoints"], dtype=np.float32)
    out["T"] = out.get("T") or int(kp.shape[0])
    out.setdefault("V", int(kp.shape[1]))
    out.setdefault("topology_name", "K9Graph")
    out["sample_id"] = f"dogpose::{e.get('sample_id', e.get('sequence_id', 'unknown'))}"
    out["source_channel"] = CH_DOGPOSE
    out.setdefault("usage_scope", "augment_static")
    out.setdefault("label_status", "none_static_gt")
    out["usage_scope"] = "augment_static"
    out["label_status"] = "none_static_gt"
    out["static"] = bool(e.get("static", True))
    return out


def _build_ak_entry(clip: Dict[str, Any]) -> Dict[str, Any]:
    """AK partialclass4 有标条目。"""
    kp = np.asarray(clip["keypoints"], dtype=np.float32)
    vid = str(clip["video_id"])
    sp = str(clip.get("split", "train"))
    return {
        "sample_id": f"akq3b::{vid}::{sp}",
        "source_channel": CH_AK,
        "split": sp,
        "topology_name": "K9Graph",
        "V": int(kp.shape[1]),
        "T": int(kp.shape[0]),
        "keypoints": kp,
        "coords_semantic": "image_pixel_xy_score",
        "fps_or_sampling": 30.0,
        "usage_scope": "supervised_partialclass4",
        "label_status": "gate4_labeled",
        "label": int(clip["label"]),
        "psd_class": str(clip["psd_class"]),
        "static": False,
        "provenance": {"video_id": vid},
    }


# ---------------------------------------------------------------------------
# 四源汇聚
# ---------------------------------------------------------------------------

def _load_aptv2_entries(sources: Dict[str, str]) -> List[Dict[str, Any]]:
    seq_dir = Path(sources["aptv2_sequences_dir"])
    mf_path = Path(sources["aptv2_manifest"])
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    entries: List[Dict[str, Any]] = []
    for meta in mf.get("entries", []):
        fp = seq_dir / str(meta["file"])
        with open(fp, "rb") as f:
            seq = pickle.load(f)
        entries.append(build_aptv2_entry(seq, sha256=meta.get("sha256")))
    return entries


def _load_mocap_entries(sources: Dict[str, str]) -> List[Dict[str, Any]]:
    seq_dir = Path(sources["mocap_sequences_dir"])
    lines = Path(sources["mocap_manifest"]).read_text(encoding="utf-8").splitlines()
    entries: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        meta = json.loads(line)
        sid = meta["sequence_id"]
        fp = seq_dir / f"{sid}.pkl"
        with open(fp, "rb") as f:
            seq = pickle.load(f)
        entries.append(_build_mocap_entry(seq, sid_hint=sid))
    return entries


def _load_dogpose_entries(sources: Dict[str, str]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for p in sources["dogpose_sequence_pkls"]:
        with open(p, "rb") as f:
            bundle = pickle.load(f)
        for e in bundle["entries"]:
            entries.append(_build_dogpose_entry(e))
    return entries


def assemble_pool(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """按配置汇聚四源 → (pool_dict, manifest_dict)。不落盘（IO 由 main 负责）."""
    sources = cfg["sources"]
    options = cfg.get("options", {})
    entries: List[Dict[str, Any]] = []

    # -- AK partialclass4: 内容断言先行 ------------------------------------
    ak_verdict = check_ak_source(sources["ak_pkl"], sources["ak_manifest"])
    require_full = bool(options.get("require_ak_full", False))
    if ak_verdict["status"] == "ok":
        with open(sources["ak_pkl"], "rb") as f:
            clips = pickle.load(f)
        entries.extend(_build_ak_entry(c) for c in clips)
    elif require_full:
        raise RuntimeError(
            f"require_ak_full=true 且 AK 源未通过内容断言: {ak_verdict['reasons'][:5]}")

    # -- 其余三源 -----------------------------------------------------------
    entries.extend(_load_aptv2_entries(sources))
    entries.extend(_load_mocap_entries(sources))
    entries.extend(_load_dogpose_entries(sources))

    honesty = {
        "ak_source": {
            "status": ak_verdict["status"],
            "n_clips": ak_verdict["n_clips"],
            "n_manifest": ak_verdict["n_manifest"],
            "reasons": ak_verdict["reasons"][:20],
            "note": ("Q3b 全量产物未落盘前，AK 分支按 quarantine 处理不入池"
                     if ak_verdict["status"] != "ok" else "内容断言通过"),
        },
        "aptv2_label_deferral": (
            "label_status=deferred_pixel_domain: rule_seeds.py 特征层依赖度量 3D z 高度轴"
            "+SMAL 索引组，APTv2 为 2D 图像像素(x,y,vis)，物理规则阈值不可跨域迁移；"
            "像素域规则适配归后续窗口，本池仅承载几何与预训练用途"),
        "topology_heterogeneity": (
            "池内并存 K9Graph-24(aptv2 映射/dogpose/ak) 与 mann_dogset_21j(mocap 原生)；"
            "下游必须按 topology_name 过滤，禁止跨拓扑混批"),
    }

    pool = {
        "schema": "psd.data_campaign.unified.real_expansion_v1",
        "channel": "unified_real_expansion",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": "scripts/build_unified_pool.py",
        "honesty": honesty,
        "entries": entries,
    }
    manifest = {
        "schema": pool["schema"],
        "generated_at": pool["generated_at"],
        "generator": pool["generator"],
        "aggregate": aggregate_stats(entries),
        "honesty": honesty,
        "mapping_table": [
            {"aptv2_idx": s, "aptv2_name": APTV2_KPT_NAMES[s],
             "k9_idx": d, "k9_name": _k9_name(d), "status": st,
             "reason": MAPPING_SEMANTICS[d]}
            for s, (d, st) in sorted(APT2_TO_K9.items())
        ] + [
            {"k9_idx": i, "k9_name": _k9_name(i),
             "status": "unmapped", "reason": MAPPING_SEMANTICS[i]}
            for i in sorted(K9_UNMAPPED_FROM_APTV2)
        ],
        "config_echo": cfg,
    }
    return pool, manifest


def _k9_name(idx: int) -> str:
    """K9Graph 权威点名（truth 单一性: psd/models/stgcn_k9_graph.py）。"""
    from psd.models.stgcn_k9_graph import K9Graph
    return K9Graph.NODE_NAMES[idx]


def aggregate_stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """池分布统计（任务书 TDD 第 4 点：池分布统计）。"""
    by_ch: Dict[str, int] = {}
    by_topo: Dict[str, int] = {}
    by_scope: Dict[str, int] = {}
    by_split: Dict[str, int] = {}
    ts: List[int] = []
    for e in entries:
        by_ch[e["source_channel"]] = by_ch.get(e["source_channel"], 0) + 1
        by_topo[e["topology_name"]] = by_topo.get(e["topology_name"], 0) + 1
        by_scope[e["usage_scope"]] = by_scope.get(e["usage_scope"], 0) + 1
        by_split[e.get("split") or "unsplit"] = by_split.get(e.get("split") or "unsplit", 0) + 1
        ts.append(int(e["T"]))
    t_summary = {
        "min": min(ts), "max": max(ts), "mean": round(float(np.mean(ts)), 2),
    } if ts else {}
    return {
        "total_samples": len(entries),
        "by_source_channel": dict(sorted(by_ch.items())),
        "by_topology": dict(sorted(by_topo.items())),
        "by_usage_scope": dict(sorted(by_scope.items())),
        "by_split": dict(sorted(by_split.items())),
        "T_summary": t_summary,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = REPO_ROOT / "configs" / "unified_pool.yaml"


def main(argv: Optional[List[str]] = None) -> int:
    import yaml

    parser = argparse.ArgumentParser(description="W30 统一真实扩展池组装器")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YAML 配置路径")
    parser.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    args = parser.parse_args(argv)

    with open(args.config, "rb") as f:
        cfg = yaml.safe_load(f)

    # 相对路径锚定到仓库根（worktree 内运行时产物落本窗 runs/，源可指主检出绝对路径）
    def _anchor(p: str) -> str:
        pp = Path(p)
        return str(pp if pp.is_absolute() else REPO_ROOT / pp)

    for key, val in cfg["sources"].items():
        if isinstance(val, list):
            cfg["sources"][key] = [_anchor(v) for v in val]
        else:
            cfg["sources"][key] = _anchor(val)
    for key, val in cfg.get("output", {}).items():
        cfg["output"][key] = _anchor(val)

    pool, manifest = assemble_pool(cfg)
    print(json.dumps(manifest["aggregate"], ensure_ascii=False, indent=2))

    if getattr(args, "dry_run", False):
        print("[dry-run] 未落盘")
        return 0

    out_cfg = cfg["output"]
    pool_path = Path(out_cfg["pool_pkl"])
    man_path = Path(out_cfg["manifest_json"])
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pool_path, "wb") as f:
        pickle.dump(pool, f, protocol=pickle.HIGHEST_PROTOCOL)
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] pool={pool_path} ({pool_path.stat().st_size} bytes)")
    print(f"[done] manifest={man_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
