"""P0.3 种子消费适配器测试 — 先测后码（W8 交接 Step 1）。

覆盖：NPZ 段解析 / 置信度+时长过滤 / clip 级不相交切分 / 分层采样。
种子 NPZ 格式以 data/seeds/rule_seeds/*.npz 实测结构为准：
segments 为结构化数组 (start<i8, end<i8, label<U16, conf<f4, rules<U128)。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.data.p03_seed_consumer import (
    FPS_DEFAULT,
    filter_segments,
    label_stats,
    load_seed_segments,
    sample_anchor_segments,
    segment_duration_s,
    split_clips,
)


# ---------------------------------------------------------------- fixtures

def _write_seed_npz(path, segments):
    """按 W6 实测字段写一个种子 NPZ（rules 用 '|' 连接，空为 ''）。"""
    arr = np.array(
        [
            (
                int(s),
                int(e),
                str(lab),
                np.float32(cf),
                "|".join(rules),
            )
            for (s, e, lab, cf, rules) in segments
        ],
        dtype=[("start", "<i8"), ("end", "<i8"), ("label", "<U16"),
               ("conf", "<f4"), ("rules", "<U128")],
    )
    np.savez(
        path,
        segments=arr,
        frame_labels=np.zeros(10, dtype="<U16"),
        frame_confidence=np.zeros(10, dtype=np.float32),
        body_scale=np.float64(0.25),
        ground_height=np.float64(0.0),
    )


@pytest.fixture
def seeds_dir(tmp_path):
    _write_seed_npz(
        tmp_path / "clipA.npz",
        [
            (0, 29, "sitting", 0.91, ["sitting_posture"]),
            (30, 59, "unknown", 0.95, []),          # 高置信 unknown（防御性剔除对象）
            (60, 74, "walking", 0.60, ["gait_walk"]),  # 置信不足 → 过滤掉
            (75, 89, "running|gait_run", 0.88, ["gait_run"]),  # 复合规则 ID
        ],
    )
    # 时长过滤对象: 0.5s@30fps = 15 帧；14 帧 = 0.4667s < 0.5s → 剔除
    _write_seed_npz(
        tmp_path / "clipB.npz",
        [
            (0, 13, "jump", 0.90, ["jump_airborne"]),   # 14 帧过短
            (14, 43, "lying", 0.80, ["lying_posture"]), # 30 帧、conf=边界值 0.80
        ],
    )
    return tmp_path


# ---------------------------------------------------------------- 加载解析

class TestLoadSeedSegments:
    def test_parses_all_segments_with_fields(self, seeds_dir):
        segs = load_seed_segments(seeds_dir)
        assert len(segs) == 6
        first = segs[0]
        assert set(first.keys()) >= {
            "clip_id", "start_frame", "end_frame", "label", "confidence", "rule_ids",
        }
        assert first["clip_id"] == "clipA"
        assert first["start_frame"] == 0 and first["end_frame"] == 29
        assert first["label"] == "sitting"
        assert abs(first["confidence"] - 0.91) < 1e-6
        assert first["rule_ids"] == ["sitting_posture"]

    def test_compound_rule_ids_split_on_pipe(self, seeds_dir):
        segs = load_seed_segments(seeds_dir)
        compound = [s for s in segs if s["clip_id"] == "clipA" and s["label"].startswith("running")]
        assert len(compound) == 1
        assert compound[0]["rule_ids"] == ["gait_run"]

    def test_empty_rules_yield_empty_list(self, seeds_dir):
        segs = load_seed_segments(seeds_dir)
        unk = [s for s in segs if s["label"] == "unknown"]
        assert unk[0]["rule_ids"] == []

    def test_sorted_by_clip_then_frame(self, seeds_dir):
        segs = load_seed_segments(seeds_dir)
        keys = [(s["clip_id"], s["start_frame"]) for s in segs]
        assert keys == sorted(keys)

    def test_missing_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_seed_segments(tmp_path / "nope")


# ---------------------------------------------------------------- 过滤

class TestFilterSegments:
    def test_confidence_and_duration_gate(self, seeds_dir):
        segs = load_seed_segments(seeds_dir)
        kept = filter_segments(segs, conf_min=0.8, min_duration_s=0.5)
        got = [(s["clip_id"], s["label"], round(s["confidence"], 2)) for s in kept]
        # clipA sitting(0.91,30帧) + running 复合段(15帧? 75..89=15帧, conf .88)
        #   75..89 共 15 帧 = 0.5s 边界 → 保留
        # clipB lying conf=0.80 边界保留; jump 14 帧剔除; walking conf 低剔除;
        # unknown 即便高置信也剔除
        assert ("clipA", "sitting", 0.91) in got
        assert ("clipB", "lying", 0.8) in got
        labels = [s["label"] for s in kept]
        assert "unknown" not in labels
        assert all(s["confidence"] >= 0.8 - 1e-9 for s in kept)
        assert all(segment_duration_s(s) >= 0.5 - 1e-9 for s in kept)

    def test_boundary_duration_exactly_half_second_kept(self, seeds_dir):
        segs = [s for s in load_seed_segments(seeds_dir) if s["label"].startswith("running")]
        assert segment_duration_s(segs[0]) == pytest.approx(0.5)
        kept = filter_segments(segs, conf_min=0.8, min_duration_s=0.5)
        assert len(kept) == 1

    def test_relaxed_thresholds_keep_more(self, seeds_dir):
        strict = filter_segments(load_seed_segments(seeds_dir), 0.8, 0.5)
        loose = filter_segments(load_seed_segments(seeds_dir), 0.55, 0.3)
        assert len(loose) > len(strict)


class TestHelpers:
    def test_segment_duration_uses_fps(self):
        seg = {"start_frame": 0, "end_frame": 59}
        assert segment_duration_s(seg, fps=FPS_DEFAULT) == pytest.approx(2.0)
        assert segment_duration_s(seg, fps=60.0) == pytest.approx(1.0)

    def test_label_stats_counts(self, seeds_dir):
        stats = label_stats(filter_segments(load_seed_segments(seeds_dir), 0.8, 0.5))
        assert stats["sitting"] == 1
        assert stats["lying"] == 1
        assert "unknown" not in stats


# ---------------------------------------------------------------- clip 切分

class TestSplitClips:
    def _make_clip_ids(self, n=40):
        return [f"clip{i:03d}" for i in range(n)]

    def test_disjoint_and_complete(self):
        ids = self._make_clip_ids()
        anchor, ev = split_clips(ids, eval_ratio=0.3, seed=42)
        assert not (set(anchor) & set(ev))
        assert set(anchor) | set(ev) == set(ids)
        assert len(ev) == 12  # 40 * 0.3

    def test_deterministic_same_seed(self):
        ids = self._make_clip_ids()
        a1, e1 = split_clips(ids, eval_ratio=0.3, seed=42)
        a2, e2 = split_clips(ids, eval_ratio=0.3, seed=42)
        assert a1 == a2 and e1 == e2

    def test_different_seed_changes_split(self):
        ids = self._make_clip_ids()
        _, e1 = split_clips(ids, eval_ratio=0.3, seed=42)
        _, e2 = split_clips(ids, eval_ratio=0.3, seed=43)
        assert e1 != e2

    def test_tiny_input_keeps_at_least_one_eval_and_one_anchor(self):
        anchor, ev = split_clips(["c1", "c2", "c3"], eval_ratio=0.3, seed=0)
        assert len(anchor) >= 1 and len(ev) >= 1
        assert len(anchor) + len(ev) == 3


# ---------------------------------------------------------------- 分层采样

def _mk(label, i, conf=0.9, dur_frames=30):
    return {
        "clip_id": f"c{i}",
        "start_frame": 0,
        "end_frame": dur_frames - 1,
        "label": label,
        "confidence": conf,
        "rule_ids": [],
    }


class TestSampleAnchorSegments:
    def test_ratio_one_returns_all(self):
        segs = [_mk("a", i) for i in range(5)] + [_mk("b", i) for i in range(5)]
        out = sample_anchor_segments(segs, ratio=1.0, seed=42)
        assert len(out) == 10

    def test_zero_ratio_returns_empty(self):
        segs = [_mk("a", i) for i in range(4)]
        assert sample_anchor_segments(segs, ratio=0.0, seed=42) == []

    def test_stratified_each_class_sampled(self):
        segs = ([_mk("a", i) for i in range(20)] + [_mk("b", i) for i in range(4)]
                + [_mk("c", i) for i in range(1)])
        out = sample_anchor_segments(segs, ratio=0.5, seed=42)
        stats = label_stats(out)
        assert stats["a"] == 10
        assert stats["b"] == 2
        assert stats["c"] == 1  # ceil(0.5*1)=1：稀有类至少保底 1 条

    def test_deterministic(self):
        segs = [_mk("a", i) for i in range(20)]
        s1 = sample_anchor_segments(segs, 0.5, 42)
        s2 = sample_anchor_segments(segs, 0.5, 42)
        assert s1 == s2

    def test_invalid_ratio_raises(self):
        with pytest.raises(ValueError):
            sample_anchor_segments([_mk("a", 0)], ratio=-0.1, seed=0)
        with pytest.raises(ValueError):
            sample_anchor_segments([_mk("a", 0)], ratio=1.5, seed=0)
