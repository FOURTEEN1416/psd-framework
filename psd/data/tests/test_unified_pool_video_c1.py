# -*- coding: utf-8 -*-
"""W35 统一池第五源(video_c1_w35)TDD——可选性/条目契约/对账诚实块.

任务书: dev-docs/handovers/NEXT-BATCH-plan.md §W35 第④步(Permit 特批修改
scripts/build_unified_pool.py 的配套测试, 新文件不触碰 W30 既有测试).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unified_pool import (  # noqa: E402
    CH_VIDEO,
    _load_video_entries,
    aggregate_stats,
    assemble_pool,
)

REQUIRED_ENTRY_KEYS = {
    "sample_id", "source_channel", "split", "topology_name", "V", "T",
    "keypoints", "coords_semantic", "fps_or_sampling",
    "usage_scope", "label_status", "static",
}


# ---------------------------------------------------------------------------
# 迷你五源夹具(其余四源各一条, 视频源参数化)
# ---------------------------------------------------------------------------

def _mk_four_sources(tmp_path: Path) -> dict:
    # aptv2 ×1
    can = tmp_path / "aptv2" / "sequences" / "canidae"
    can.mkdir(parents=True)
    sid = "aptv2_dog_v0_t0_w000"
    with open(can / f"{sid}.pkl", "wb") as f:
        pickle.dump({"keypoints": np.zeros((15, 17, 3), dtype=np.float32),
                     "topology_name": "aptv2_quadruped_17kp", "V": 17,
                     "fps_or_sampling": None, "source": {"dataset": "APTv2"},
                     "split": "train", "species": "dog", "sequence_id": sid}, f)
    mf = tmp_path / "aptv2" / "sequences" / "_manifest.json"
    mf.write_text(json.dumps({"entries": [
        {"file": f"canidae\\{sid}.pkl", "sequence_id": sid, "species": "dog",
         "split": "train", "T": 15, "V": 17, "C": 3,
         "visible_frac_mean": 0.5, "sha256": "h0"}]}), encoding="utf-8")

    # mocap ×1
    mdir = tmp_path / "mocap" / "sequences"
    mdir.mkdir(parents=True)
    with open(mdir / "D1_001.pkl", "wb") as f:
        pickle.dump({"keypoints": np.zeros((10, 21, 3), dtype=np.float32),
                     "topology_name": "mann_dogset_21j", "V": 21,
                     "fps_or_sampling": 60.0, "source": "u", "split": "unsplit"}, f)
    (tmp_path / "mocap" / "manifest.jsonl").write_text(
        json.dumps({"sequence_id": "D1_001"}) + "\n", encoding="utf-8")

    # dogpose ×1(合并式 bundle)
    ddir = tmp_path / "dogpose" / "sequences"
    ddir.mkdir(parents=True)
    with open(ddir / "dogpose_train.pkl", "wb") as f:
        pickle.dump({"entries": [{
            "keypoints": np.zeros((1, 24, 3), dtype=np.float32),
            "topology_name": "K9Graph", "V": 24, "source": "dog-pose",
            "split": "train", "sample_id": "dp_0", "static": True}]}, f)

    # ak ×1(一致全量)
    ak_dir = tmp_path / "ak"
    ak_dir.mkdir(parents=True)
    man = {"gate_classes": ["stay"], "clip_t": 30,
           "samples": [{"video_id": "VA", "split": "train", "psd_class": "stay",
                        "class_idx": 0, "source": "tar", "video_path": "x.mp4"}]}
    (ak_dir / "ak_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    with open(ak_dir / "ak.pkl", "wb") as f:
        pickle.dump([{"keypoints": np.zeros((30, 24, 3), dtype=np.float32),
                      "label": 0, "boundary": np.zeros(30, dtype=np.float32),
                      "video_id": "VA", "split": "train", "psd_class": "stay"}], f)

    return {
        "aptv2_sequences_dir": str(can.parent),
        "aptv2_manifest": str(mf),
        "mocap_sequences_dir": str(mdir),
        "mocap_manifest": str(tmp_path / "mocap" / "manifest.jsonl"),
        "dogpose_sequence_pkls": [str(ddir / "dogpose_train.pkl")],
        "ak_pkl": str(ak_dir / "ak.pkl"),
        "ak_manifest": str(ak_dir / "ak_manifest.json"),
    }


def _mk_video_source(tmp_path: Path, n_clips: int = 2,
                     with_error_row: bool = True, drop_one_pkl: bool = False):
    """seq30 pkls + extract_index.jsonl; 可注入 error 行与缺失 pkl 场景."""
    seq_dir = tmp_path / "video" / "seq30"
    seq_dir.mkdir(parents=True)
    lines = []
    for i in range(n_clips):
        fid = f"w25v-{i:08x}"
        kp = np.random.RandomState(i).rand(30, 24, 3).astype(np.float32)
        kp[:, 20:24, :] = 0.0  # 模拟死关节硬掩码后的形态(builder 原样透传不再加工)
        with open(seq_dir / f"{fid}.pkl", "wb") as f:
            pickle.dump({"sample_id": f"w25::{fid}", "keypoints": kp,
                         "label": -1, "boundary": np.zeros(30, dtype=np.float32),
                         "topology_name": "K9Graph", "V": 24, "T": 30,
                         "coords_semantic": "image_norm_xy_conf01_deadmasked",
                         "split": "unlabeled_draft", "source": "video_c1_w25_fragments",
                         "n_interpolated": i, "dead_joints_masked": [20, 21, 22, 23],
                         "fps_or_sampling": {"src_fps": 30.0, "strategy": "uniform_T30"}}, f)
        lines.append(json.dumps({"fragment_id": fid, "status": "ok",
                                 "src_fps": 30.0, "frames_read": 900,
                                 "rule_frames": 300, "detect_ok": 296,
                                 "n_interpolated": i}))
    if with_error_row:
        lines.append(json.dumps({"fragment_id": "w25v-deadbeef",
                                 "status": "error:open_failed"}))
    idx = tmp_path / "video" / "extract_index.jsonl"
    idx.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if drop_one_pkl and n_clips > 0:
        victim = json.loads(lines[0])["fragment_id"]
        (seq_dir / f"{victim}.pkl").unlink()
    return str(seq_dir), str(idx)


# ---------------------------------------------------------------------------
# 单元: _load_video_entries
# ---------------------------------------------------------------------------

class TestLoadVideoEntries:

    def test_entry_contract_and_passthrough(self, tmp_path):
        sdir, idx = _mk_video_source(tmp_path, n_clips=2, with_error_row=True)
        sources = {"video_sequences_dir": sdir, "video_extract_index": idx}
        entries, verdict = _load_video_entries(sources)
        assert verdict == {"n_index_ok": 2, "n_loaded": 2, "note": "ok"}
        assert len(entries) == 2
        e = entries[0]
        assert REQUIRED_ENTRY_KEYS <= set(e.keys())
        assert e["source_channel"] == CH_VIDEO == "video_c1_w35"
        assert e["sample_id"].startswith("videoc1::w25v-")
        assert e["usage_scope"] == "pretrain_geometric"
        assert e["label_status"] == "deferred_pixel_domain"
        assert e["split"] == "unsplit" and e["static"] is False
        assert e["coords_semantic"] == "image_norm_xy_conf01_deadmasked"
        # 原样透传: 死关节清零形态不被二次加工
        assert (e["keypoints"][:, 20:24, :] == 0).all()
        prov = e["provenance"]
        assert prov["dead_joints_masked"] == [20, 21, 22, 23]
        assert "withers(idx22)" in prov["seed_draft_note"]
        assert "Q3a" in prov["extract_weights"]

    def test_missing_pkl_reported_not_silent(self, tmp_path):
        sdir, idx = _mk_video_source(tmp_path, n_clips=2, drop_one_pkl=True)
        sources = {"video_sequences_dir": sdir, "video_extract_index": idx}
        _, verdict = _load_video_entries(sources)
        assert verdict["n_index_ok"] == 2 and verdict["n_loaded"] == 1
        assert "缺 seq30 pkl" in verdict["note"]

    def test_sample_ids_unique(self, tmp_path):
        sdir, idx = _mk_video_source(tmp_path, n_clips=3)
        entries, _ = _load_video_entries(
            {"video_sequences_dir": sdir, "video_extract_index": idx})
        ids = [e["sample_id"] for e in entries]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 集成: assemble_pool 向后兼容 + 第五源汇入
# ---------------------------------------------------------------------------

class TestAssemblePoolVideoSource:

    def test_absent_video_keys_zero_behavior_change(self, tmp_path):
        """config 缺省(v1 形态)→ 零视频条目 + 无 honesty 块(W30 行为不变)."""
        cfg = {"sources": _mk_four_sources(tmp_path), "options": {}}
        pool, manifest = assemble_pool(cfg)
        chans = {e["source_channel"] for e in pool["entries"]}
        assert CH_VIDEO not in chans
        assert "video_c1_w35" not in manifest["aggregate"]["by_source_channel"]
        assert "video_c1_w35" not in pool["honesty"]

    def test_video_channel_merged_and_counted(self, tmp_path):
        sources = _mk_four_sources(tmp_path)
        sdir, idx = _mk_video_source(tmp_path / "vs", n_clips=2)
        sources["video_sequences_dir"] = sdir
        sources["video_extract_index"] = idx
        pool, manifest = assemble_pool({"sources": sources, "options": {}})
        vids = [e for e in pool["entries"] if e["source_channel"] == CH_VIDEO]
        assert len(vids) == 2
        agg = manifest["aggregate"]
        assert agg["by_source_channel"][CH_VIDEO] == 2
        assert agg["by_usage_scope"]["pretrain_geometric"] == 3  # aptv2 1 + video 2
        stats = aggregate_stats(pool["entries"])
        assert stats["total_samples"] == 6  # aptv2 1 + mocap 1 + dogpose 1 + ak 1 + video 2
        h = pool["honesty"]["video_c1_w35"]
        assert "deferred_pixel_domain" in h["label_policy"]

    def test_all_missing_pkls_still_honest(self, tmp_path):
        sources = _mk_four_sources(tmp_path)
        sdir, idx = _mk_video_source(tmp_path / "vs", n_clips=1, drop_one_pkl=True)
        sources["video_sequences_dir"] = sdir
        sources["video_extract_index"] = idx
        pool, manifest = assemble_pool({"sources": sources, "options": {}})
        assert manifest["aggregate"]["by_source_channel"].get(CH_VIDEO, 0) == 0
        assert "仅加载 0 条" in pool["honesty"]["video_c1_w35"]["note"]
