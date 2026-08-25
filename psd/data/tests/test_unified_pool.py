# -*- coding: utf-8 -*-
"""W30 统一真实扩展池 — TDD 测试（任务书: dev-docs/handovers/NEXT-BATCH-plan.md §W30）.

覆盖四轴:
  1. APTv2 17kp→K9Graph 24kp 拓扑映射（映射表完整性 + spot-check 已知答案）
  2. AK partialclass4 内容级断言（冒烟残留隔离，防 1≠172 静默混入）
  3. 池条目 schema 完整性 + usage_scope/label_status 契约
  4. 组装 split 完整性（sample_id 全局唯一 / 分源 split 计数对账）+ 分布统计
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unified_pool import (  # noqa: E402
    APT2_TO_K9,
    APTV2_KPT_NAMES,
    K9_UNMAPPED_FROM_APTV2,
    MAPPING_SEMANTICS,
    USAGE_SCOPES,
    aggregate_stats,
    assemble_pool,
    build_aptv2_entry,
    check_ak_source,
    map_aptv2_keypoints,
)
from psd.models.stgcn_k9_graph import K9Graph  # noqa: E402


# ---------------------------------------------------------------------------
# 1) 映射表完整性
# ---------------------------------------------------------------------------

class TestMappingTableIntegrity:

    def test_official_names_match_source_annotations(self):
        """17 点名与顺序 = APTv2 train_annotations.json categories 权威序."""
        assert len(APTV2_KPT_NAMES) == 17
        assert APTV2_KPT_NAMES[:5] == ["left_eye", "right_eye", "nose", "neck", "root_of_tail"]
        assert APTV2_KPT_NAMES[-3:] == ["right_hip", "right_knee", "right_back_paw"]

    def test_every_aptv2_point_mapped_exactly_once(self):
        srcs = list(APT2_TO_K9.keys())
        assert sorted(srcs) == list(range(17)), "17 个源点必须各映射一次"

    def test_target_slots_unique_and_disjoint_from_unmapped(self):
        dsts = [d for (d, _) in APT2_TO_K9.values()]
        assert len(dsts) == len(set(dsts)), "目标槽位不得重复"
        assert not (set(dsts) & K9_UNMAPPED_FROM_APTV2), "映射目标不得落入不可映射槽位"
        assert set(dsts) | K9_UNMAPPED_FROM_APTV2 == set(range(24)), "24 槽位必须全覆盖"

    def test_unmapped_set_matches_honest_audit(self):
        """7 个诚实不可映射点: 尾尖/双耳base/下巴/双耳tip/喉部."""
        assert K9_UNMAPPED_FROM_APTV2 == {13, 14, 15, 17, 18, 19, 23}

    def test_names_valid_against_k9graph(self):
        names = K9Graph.NODE_NAMES
        for (d, _) in APT2_TO_K9.values():
            assert 0 <= d < 24
            assert names[d].isidentifier()
        for idx in K9_UNMAPPED_FROM_APTV2:
            assert 0 <= idx < 24

    def test_neck_to_withers_is_approx(self):
        assert APT2_TO_K9[3] == (22, "approx"), "neck→withers 必须标 approx"

    def test_semantic_reason_documented_for_all_24(self):
        assert set(MAPPING_SEMANTICS.keys()) == set(range(24))
        for idx, reason in MAPPING_SEMANTICS.items():
            assert isinstance(reason, str) and len(reason) >= 4


# ---------------------------------------------------------------------------
# 2) 映射 spot-check（已知答案）
# ---------------------------------------------------------------------------

class TestMapAptv2KnownAnswer:

    def _mk(self, t=4):
        kp = np.zeros((t, 17, 3), dtype=np.float32)
        kp[..., 0] = np.arange(t * 17).reshape(t, 17) * 1.0   # x 唯一可溯源
        kp[..., 1] = 100.0                                     # y 固定
        kp[..., 2] = (np.arange(t * 17).reshape(t, 17) % 2)    # vis 交替 0/1
        return kp

    def test_shape_and_dtype(self):
        out = map_aptv2_keypoints(self._mk())
        assert out.shape == (4, 24, 3)
        assert out.dtype == np.float32

    def test_known_answer_xy_vis_passthrough(self):
        kp = self._mk()
        out = map_aptv2_keypoints(kp)
        for s, (d, _status) in APT2_TO_K9.items():
            assert np.allclose(out[:, d, 0], kp[:, s, 0]), f"k9[{d}] xy.x 应= aptv2[{s}]"
            assert np.allclose(out[:, d, 1], kp[:, s, 1])
            assert np.allclose(out[:, d, 2], kp[:, s, 2]), "vis 通道透传"

    def test_unmapped_slots_nan_xy_zero_vis(self):
        out = map_aptv2_keypoints(self._mk())
        for idx in K9_UNMAPPED_FROM_APTV2:
            assert np.isnan(out[:, idx, 0]).all() and np.isnan(out[:, idx, 1]).all()
            assert (out[:, idx, 2] == 0.0).all()

    def test_input_not_mutated(self):
        kp = self._mk()
        snap = kp.copy()
        map_aptv2_keypoints(kp)
        assert np.array_equal(kp, snap)


# ---------------------------------------------------------------------------
# 3) AK 内容级断言（冒烟残留隔离）
# ---------------------------------------------------------------------------

def _write_ak_fixture(tmp_path: Path, n_manifest: int, clips: list[dict]):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    man = {
        "gate_classes": ["stay", "track", "watch", "jump"],
        "clip_t": 30,
        "samples": [
            {"video_id": f"VID{i:03d}", "split": "train", "psd_class": "watch",
             "class_idx": 2, "source": "tar", "video_path": f"x{i}.mp4"}
            for i in range(n_manifest)
        ],
    }
    mp = tmp_path / "ak_manifest.json"
    mp.write_text(json.dumps(man), encoding="utf-8")
    pp = tmp_path / "ak.pkl"
    with open(pp, "wb") as f:
        pickle.dump(clips, f)
    return pp, mp


def _mk_clip(v=24, label=2, psd_class="watch", vid="X"):
    return {
        "keypoints": np.zeros((30, v, 3), dtype=np.float32),
        "label": label,
        "boundary": np.zeros((30,), dtype=np.float32),
        "video_id": vid,
        "split": "train",
        "psd_class": psd_class,
    }


class TestCheckAkSource:

    def test_smoke_residue_count_mismatch_detected(self):
        """真实事故复现: manifest 声称 172 样本而 pkl 仅 1 clip → 必须判 quarantine."""
        pp, mp = _write_ak_fixture(_tmp(), 172, [_mk_clip(v=17)])
        verdict = check_ak_source(pp, mp)
        assert verdict["status"] == "quarantined"
        assert verdict["reasons"], "必须给出具体断言失败原因"

    def test_v_dimension_mismatch_detected(self):
        pp, mp = _write_ak_fixture(_tmp(), 1, [_mk_clip(v=17)])
        verdict = check_ak_source(pp, mp)
        assert verdict["status"] == "quarantined"

    def test_ok_when_consistent_full_product(self):
        clips = [_mk_clip(v=24), _mk_clip(v=24)]
        pp, mp = _write_ak_fixture(_tmp(), 2, clips)
        verdict = check_ak_source(pp, mp)
        assert verdict["status"] == "ok"

    def test_missing_files_reported(self, tmp_path: Path):
        verdict = check_ak_source(tmp_path / "nope.pkl", tmp_path / "nope.json")
        assert verdict["status"] == "missing"


def _tmp() -> Path:
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="w30_ak_"))
    return d


# ---------------------------------------------------------------------------
# 4) 条目 schema + usage_scope 契约
# ---------------------------------------------------------------------------

REQUIRED_ENTRY_KEYS = {
    "sample_id", "source_channel", "split", "topology_name", "V", "T",
    "keypoints", "coords_semantic", "fps_or_sampling",
    "usage_scope", "label_status", "static",
}


class TestEntrySchema:

    def _mk_seq(self):
        return {
            "keypoints": np.zeros((15, 17, 3), dtype=np.float32),
            "topology_name": "aptv2_quadruped_17kp",
            "V": 17,
            "fps_or_sampling": "unknown_fps_consecutive_annotated_frames",
            "source": {"dataset": "APTv2"},
            "split": "train",
            "species": "dog",
            "sequence_id": "aptv2_dog_x_w000",
        }

    def test_aptv2_entry_contract(self):
        e = build_aptv2_entry(self._mk_seq())
        assert REQUIRED_ENTRY_KEYS <= set(e.keys())
        assert e["source_channel"] == "aptv2_c2_w26"
        assert e["usage_scope"] == "pretrain_geometric"
        assert e["label_status"] == "deferred_pixel_domain"
        assert e["topology_name"] == "K9Graph"
        assert e["keypoints"].shape == (15, 24, 3)
        assert e["coords_semantic"] == "image_pixel_xy_vis01"
        assert e["static"] is False

    def test_usage_scopes_closed_set(self):
        assert USAGE_SCOPES == {
            "pretrain_geometric", "kinematic_prior", "augment_static",
            "supervised_partialclass4",
        }


# ---------------------------------------------------------------------------
# 5) 组装 split 完整性 + 分布统计
# ---------------------------------------------------------------------------

def _mk_fake_sources(tmp_path: Path):
    """构造迷你四源 fixture：aptv2×3(train2/val1) + mocap×1 + dogpose train×2 + ak ok×2."""
    # aptv2: 3 个单序列 pkl + _manifest.json
    can = tmp_path / "aptv2" / "sequences" / "canidae"
    can.mkdir(parents=True)
    entries = []
    specs = [("dog", "train"), ("dog", "train"), ("fox", "val")]
    for i, (sp, sp_split) in enumerate(specs):
        sid = f"aptv2_{sp}_v{i}_t0_w000"
        kp = np.full((15, 17, 3), float(i), dtype=np.float32)
        seq = {
            "keypoints": kp, "topology_name": "aptv2_quadruped_17kp", "V": 17,
            "fps_or_sampling": None, "source": {"dataset": "APTv2"},
            "split": sp_split, "species": sp, "sequence_id": sid,
        }
        fp = can / f"{sid}.pkl"
        with open(fp, "wb") as f:
            pickle.dump(seq, f)
        entries.append({
            "file": f"canidae\\{sid}.pkl", "sequence_id": sid, "species": sp,
            "split": sp_split, "T": 15, "V": 17, "C": 3,
            "visible_frac_mean": 0.5, "sha256": f"hash{i}",
        })
    mf = tmp_path / "aptv2" / "sequences" / "_manifest.json"
    mf.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    # mocap: 1 pkl + manifest.jsonl
    mdir = tmp_path / "mocap" / "sequences"
    mdir.mkdir(parents=True)
    with open(mdir / "D1_001.pkl", "wb") as f:
        pickle.dump({
            "keypoints": np.zeros((10, 21, 3), dtype=np.float32),
            "topology_name": "mann_dogset_21j", "V": 21, "fps_or_sampling": 60.0,
            "source": "mocap-url", "split": "unsplit",
            "joint_order_canonical": [f"j{k}" for k in range(21)],
            "license_note": "research only",
            # 注: 真实 W27 pkl 不含 sequence_id（id 只在 manifest.jsonl 行内）
        }, f)
    (tmp_path / "mocap" / "manifest.jsonl").write_text(
        json.dumps({"sequence_id": "D1_001"}) + "\n", encoding="utf-8")

    # dogpose: 合并式 train pkl
    ddir = tmp_path / "dogpose" / "sequences"
    ddir.mkdir(parents=True)
    dp_entries = []
    for i in range(2):
        dp_entries.append({
            "keypoints": np.zeros((1, 24, 3), dtype=np.float32),
            "topology_name": "K9Graph", "V": 24, "fps_or_sampling": None,
            "source": "dog-pose", "split": "train", "sample_id": f"dp_{i}",
            "n_visible": 12, "coords_semantic": "image_pixel_xy", "static": True,
        })
    with open(ddir / "dogpose_train.pkl", "wb") as f:
        pickle.dump({"schema": "psd.data_campaign.format_b.static_v1", "channel": "dogpose",
                     "split": "train", "topology_name": "K9Graph", "entries": dp_entries}, f)

    # ak: 一致全量 2 样本
    app, amp = _write_ak_fixture(tmp_path / "ak", 2,
                                 [_mk_clip(v=24, vid="VA"), _mk_clip(v=24, vid="VB")])

    cfg = {
        "sources": {
            "aptv2_sequences_dir": str(can.parent),
            "aptv2_manifest": str(mf),
            "mocap_sequences_dir": str(mdir),
            "mocap_manifest": str(tmp_path / "mocap" / "manifest.jsonl"),
            "dogpose_sequence_pkls": [str(ddir / "dogpose_train.pkl")],
            "ak_pkl": str(app),
            "ak_manifest": str(amp),
        },
        "options": {"require_ak_full": False},
    }
    return cfg


class TestAssemblePool:

    def test_counts_and_split_integrity(self, tmp_path: Path):
        cfg = _mk_fake_sources(tmp_path)
        pool, manifest = assemble_pool(cfg)
        by_ch: dict[str, int] = {}
        ids = set()
        for e in pool["entries"]:
            by_ch[e["source_channel"]] = by_ch.get(e["source_channel"], 0) + 1
            assert e["sample_id"] not in ids, "sample_id 必须全局唯一"
            ids.add(e["sample_id"])
            assert REQUIRED_ENTRY_KEYS <= set(e.keys())
        assert by_ch == {
            "aptv2_c2_w26": 3, "mocap_c3_w27": 1, "dogpose_c5_w29": 2,
            "ak_public_q3b": 2,
        }
        # 分源 split 对账: aptv2 train=2/val=1
        apt_splits = [e["split"] for e in pool["entries"] if e["source_channel"] == "aptv2_c2_w26"]
        assert apt_splits.count("train") == 2 and apt_splits.count("val") == 1
        # mocap id 必须取自 manifest.jsonl 行（真实 pkl 无 sequence_id 键）
        mc_ids = [e["sample_id"] for e in pool["entries"] if e["source_channel"] == "mocap_c3_w27"]
        assert mc_ids == ["mocap::D1_001"]

    def test_manifest_aggregates(self, tmp_path: Path):
        cfg = _mk_fake_sources(tmp_path)
        _, manifest = assemble_pool(cfg)
        agg = manifest["aggregate"]
        assert agg["total_samples"] == 8
        assert agg["by_source_channel"]["aptv2_c2_w26"] == 3
        assert agg["by_usage_scope"]["pretrain_geometric"] == 3
        assert agg["by_usage_scope"]["kinematic_prior"] == 1
        assert agg["by_usage_scope"]["augment_static"] == 2
        assert agg["by_usage_scope"]["supervised_partialclass4"] == 2

    def test_honesty_block_records_ak_verdict_and_label_deferral(self, tmp_path: Path):
        cfg = _mk_fake_sources(tmp_path)
        pool, manifest = assemble_pool(cfg)
        assert pool["honesty"]["ak_source"]["status"] == "ok"
        assert "deferred_pixel_domain" in json.dumps(pool["honesty"])

    def test_require_ak_full_hard_fails_on_quarantine(self, tmp_path: Path):
        cfg = _mk_fake_sources(tmp_path)
        # 篡改 AK pkl → 与 manifest 不一致
        with open(cfg["sources"]["ak_pkl"], "wb") as f:
            pickle.dump([_mk_clip(v=17)], f)
        cfg["options"]["require_ak_full"] = True
        with pytest.raises(RuntimeError):
            assemble_pool(cfg)

    def test_stats_aggregate_shape(self, tmp_path: Path):
        cfg = _mk_fake_sources(tmp_path)
        pool, _ = assemble_pool(cfg)
        stats = aggregate_stats(pool["entries"])
        assert stats["total_samples"] == 8
        assert set(stats.keys()) >= {"total_samples", "by_source_channel", "by_topology",
                                     "by_usage_scope", "T_summary"}
