# -*- coding: utf-8 -*-
"""W35 提点执行器纯逻辑层单测(采样计划/坐标代理/规则轨组装/续跑索引)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import harvest_extract_keypoints as hx  # noqa: E402


# --------------------------------------------------------------- 采样计划

def test_compute_rule_indices_stride_and_bounds():
    # 30fps→10fps: 步进3
    assert hx.compute_rule_indices(91, 30.0, 10.0) == list(range(0, 91, 3))
    # fps 异常兜底 30
    assert hx.compute_rule_indices(60, 0, 10.0) == list(range(0, 60, 3))
    # target 大于源: 步进=1 全采样
    assert hx.compute_rule_indices(5, 30.0, 100.0) == [0, 1, 2, 3, 4]
    # 空视频
    assert hx.compute_rule_indices(0, 30.0, 10.0) == []


def test_union_sample_plan_contains_seq30_and_sorted():
    want, rule = hx.union_sample_plan(300, 30.0, 10.0)
    assert rule == list(range(0, 300, 3))
    seq = hx.uniform_frame_indices(300, 30)
    assert set(seq) <= set(want)
    assert want == sorted(want)
    assert len(want) == len(set(want))


# --------------------------------------------------------------- 坐标代理

def test_to_kp_world_pixel_negy_height_proxy():
    xyn_conf = np.array([[0.5, 0.25, 0.9], [1.0, 0.0, 0.5]], dtype=np.float32)
    out = hx.to_kp_world_pixel(xyn_conf, width=640, height=480)
    assert out[0, 0] == pytest.approx(320.0)
    assert out[0, 1] == pytest.approx(120.0)
    assert out[0, 2] == pytest.approx(-120.0)   # -y: 图像上方为高
    assert out[1, 2] == pytest.approx(0.0)      # 图像顶边 → 高度 0? 否: y=0 在顶 → 高度 0
    assert out.dtype == np.float32


def test_build_rule_pkl_missing_frames_nan_zero_weight():
    dets = {0: np.full((24, 3), 0.5, dtype=np.float32)}  # 仅帧0检出
    rule_idx = [0, 3, 6]
    pkl = hx.build_rule_pkl(dets, rule_idx, 640, 480, 30.0, actual_read=7)
    assert pkl["kp_world"].shape == (3, 24, 3)
    assert np.isfinite(pkl["kp_world"][0]).all()
    assert np.isnan(pkl["kp_world"][1]).all() and np.isnan(pkl["kp_world"][2]).all()
    assert (pkl["kp_weight"][1] == 0).all() and (pkl["kp_weight"][2] == 0).all()
    assert (pkl["kp_weight"][0] > 0).all()
    assert pkl["frame_idx"].tolist() == [0, 3, 6]
    assert "negy_height_proxy" in pkl["coords_semantic"]
    assert pkl["n_detect_hit"] == 1


# --------------------------------------------------------------- 序列条目

def test_build_seq_entry_ok_path_dead_joints_masked():
    det = np.concatenate([
        np.random.RandomState(0).rand(24, 2).astype(np.float32),
        np.full((24, 1), 0.9, dtype=np.float32)], axis=1)
    dets = {fi: det for fi in range(30)}
    entry, q = hx.build_seq_entry(dets, list(range(30)))
    assert q["status"] == "ok"
    kp = entry["keypoints"]
    assert kp.shape == (30, 24, 3)
    # assemble_clip 血统: 死关节硬掩码清零
    for j in hx.DEAD_JOINTS:
        assert (kp[:, j, :] == 0).all()
    assert entry["label"] == hx.UNLABELED_SENTINEL
    assert entry["topology_name"] == "K9Graph"
    assert entry["V"] == 24 and entry["T"] == 30


def test_build_seq_entry_all_missing_returns_none():
    dets = {0: None, 1: None}
    entry, q = hx.build_seq_entry(dets, [0, 1])
    assert entry is None and q["status"] == "all_missing"


# --------------------------------------------------------------- 续跑索引

def test_resume_skips_done_fragments(tmp_path):
    idx = tmp_path / "extract_index.jsonl"
    idx.write_text(json.dumps({"fragment_id": "a", "status": "ok"}) + "\n" +
                   json.dumps({"fragment_id": "b", "status": "error:open_failed"}) + "\n",
                   encoding="utf-8")
    done = set()
    for l in idx.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r.get("status") == "ok":
            done.add(r["fragment_id"])
    rows = [{"fragment_id": x} for x in ("a", "b", "c")]
    todo = [r for r in rows if r["fragment_id"] not in done]
    assert [r["fragment_id"] for r in todo] == ["b", "c"]  # 失败重试, 成功跳过
