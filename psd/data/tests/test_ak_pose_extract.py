# -*- coding: utf-8 -*-
"""W20-C 提点管线纯逻辑测试（mock 模型，不依赖真实权重/GPU/大文件）.

口径依据（用户裁决 2026-08-24）:
  - 骨架路线 C: 犬科 mp4 自提取（YOLO11-pose + dog-pose 微调权重）
  - 宽松门禁 4 类: jump/stay/track/watch（train+val 合计 ≥10）
  - 样本判定 R2(first-mapped-hit): 标签序列中首个属于部分类协议的动作
"""
import numpy as np
import pytest
import torch

from psd.data.ak_pose_extract import (
    GATE4_CLASS_TO_IDX,
    GATE4_CLASSES,
    assemble_clip,
    first_mapped_label,
    pick_best_instance,
    select_samples,
    uniform_frame_indices,
)


class TestGate4Constants:
    """宽松门禁 4 类常量与编号。"""

    def test_gate4_members(self):
        assert sorted(GATE4_CLASSES) == ["jump", "stay", "track", "watch"]

    def test_gate4_order_follows_assets_map(self):
        # 保持 assets-map §1 的 22 类相对顺序: stay(5)<track(8)<watch(15)<jump(19)
        assert GATE4_CLASSES == ["stay", "track", "watch", "jump"]
        assert GATE4_CLASS_TO_IDX == {"stay": 0, "track": 1, "watch": 2, "jump": 3}


class TestFirstMappedLabel:
    """R2 样本判定规则。"""

    def test_first_item_in_protocol_wins(self):
        assert first_mapped_label(["68", "45"]) == "stay"  # 68 Keeping still

    def test_skips_unmapped_prefix(self):
        # 首项 5=Being Carried In Mouth 不在协议; 次项 67=Jumping 命中
        assert first_mapped_label(["5", "67", "3"]) == "jump"

    def test_no_hit_returns_none(self):
        assert first_mapped_label(["73", "15"]) is None  # Licking/Chirping 均不在协议

    def test_empty_and_garbage(self):
        assert first_mapped_label([]) is None
        assert first_mapped_label(["", "abc"]) is None


class TestSelectSamples:
    """清单构建纯逻辑: 犬科∩R2∩4类 + 来源标记。"""

    def test_basic_selection(self):
        video_labels = {
            "AAAA": ["68"],              # stay ✓
            "BBBB": ["1", "14"],         # attacking/chasing → R2 命中 track ✓
            "CCCC": ["73"],              # licking ✗ 排除
            "DDDD": ["2"],               # watch ✓
            "EEEE": ["108"],             # sit 在 12 类池但不在 4 类门禁 ✗
        }
        canine = {"AAAA", "BBBB", "CCCC", "DDDD", "EEEE"}
        local_mp4 = {"AAAA", "BBBB"}  # DDDD 缺 mp4 → source=tar
        samples = select_samples(
            video_labels_by_split={"train": video_labels},
            canine_ids=canine,
            local_mp4_ids=local_mp4,
        )
        by_vid = {s["video_id"]: s for s in samples}
        assert set(by_vid) == {"AAAA", "BBBB", "DDDD"}
        assert by_vid["AAAA"]["psd_class"] == "stay" and by_vid["AAAA"]["class_idx"] == 0
        assert by_vid["BBBB"]["psd_class"] == "track" and by_vid["BBBB"]["class_idx"] == 1
        assert by_vid["DDDD"]["source"] == "tar"
        assert by_vid["AAAA"]["split"] == "train"

    def test_val_split_respected(self):
        samples = select_samples(
            video_labels_by_split={"train": {}, "val": {"VVVV": ["67"]}},  # 67=Jumping（15=Chirping 陷阱勿用）
            canine_ids={"VVVV"},
            local_mp4_ids=set(),
        )
        assert len(samples) == 1
        assert samples[0]["psd_class"] == "jump" and samples[0]["class_idx"] == 3
        assert samples[0]["split"] == "val"


class TestUniformFrames:
    """均匀抽帧索引。"""

    def test_even_sampling(self):
        idx = uniform_frame_indices(300, 30)
        assert len(idx) == 30 and idx[0] == 0 and idx[-1] < 300
        assert idx == sorted(idx) and len(set(idx)) == 30

    def test_short_video_cycles(self):
        idx = uniform_frame_indices(10, 30)
        assert len(idx) == 30
        assert all(0 <= i < 10 for i in idx)

    def test_exact_length(self):
        idx = uniform_frame_indices(30, 30)
        assert idx == list(range(30))

    def test_zero_frames_raises(self):
        with pytest.raises(ValueError):
            uniform_frame_indices(0, 30)


class TestPickBestInstance:
    """多实例取最高置信度。"""

    def _inst(self, score):
        return torch.tensor([[float(score)] * 24 + [1.0] * 24]).reshape(1, 24, 3), score

    def test_picks_highest_confidence(self):
        # ultralytics kpts 布局 (N,24,2)+conf(N,24) —— 抽象为传入 (N,V,C+conf) 由实现定义;
        # 此处用简化接口: list[dict(kp=(24,3), score=float)]
        best = pick_best_instance(
            [{"kp": np.full((24, 3), 1.0), "score": 0.3},
             {"kp": np.full((24, 3), 2.0), "score": 0.9},
             {"kp": np.full((24, 3), 3.0), "score": 0.6}]
        )
        assert best is not None and best[0, 0] == 2.0 and best.shape == (24, 3)

    def test_empty_returns_none(self):
        assert pick_best_instance([]) is None


class TestAssembleClip:
    """时序组装 + 缺帧插值。"""

    def _kp(self, v):
        return np.full((24, 3), v, dtype=np.float32)

    def test_normal_assembly(self):
        clip = assemble_clip([self._kp(float(i)) for i in range(30)], label_idx=2)
        assert clip["keypoints"].shape == (30, 24, 3)
        assert clip["keypoints"].dtype == np.float32
        assert clip["label"] == 2
        assert clip["boundary"].shape == (30,) and clip["boundary"].sum() == 0
        assert clip["n_interpolated"] == 0

    def test_middle_gap_linear_interpolation(self):
        frames = [self._kp(0.0), None, self._kp(2.0)]
        clip = assemble_clip(frames, label_idx=0)
        assert clip["n_interpolated"] == 1
        # 中间插值帧 = 两端均值 1.0
        assert abs(float(clip["keypoints"][1, 0, 0]) - 1.0) < 1e-6

    def test_leading_gap_nearest_fill(self):
        frames = [None, self._kp(5.0)]
        clip = assemble_clip(frames, label_idx=1)
        assert clip["n_interpolated"] == 1
        assert float(clip["keypoints"][0, 0, 0]) == 5.0

    def test_all_missing_returns_none(self):
        assert assemble_clip([None, None], label_idx=0) is None

    def test_output_uses_gate_index(self):
        clip = assemble_clip([self._kp(1.0)], label_idx=GATE4_CLASS_TO_IDX["jump"])
        assert clip["label"] == 3
