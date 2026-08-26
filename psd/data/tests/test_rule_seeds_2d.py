"""2D 规则引擎（像素域规则种子适配）— W47 补救窗 TDD。

被测对象: psd/data/rule_seeds_2d.py
设计契约要点:
1. y 轴约定归一化层: normalize_y_orientation 把任意约定(down/up)输入归一到
   内部 up=+y 约定 —— 恢复 rise_up/lie_down 方向语义(W41 上版 abs 后求导 bug),
   同时保证同一物理姿态在不同 y 约定输入下分类结果一致。
2. bounding-box 体尺度归一替代 z 高度。
3. 七类规则标签 → gate4 {watch, track, jump, stay} 显式映射协议(AK partialclass4
   校准/门禁用); watch 为骨架几何不可判别类, 如实映射 None。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.data.rule_seeds_2d import (
    SMAL_GROUPS,
    DEFAULT_CONFIG_2D,
    normalize_y_orientation,
    estimate_ground_2d,
    _estimate_body_scale_2d,
    compute_frame_features_2d,
    classify_frames_2d,
    generate_seeds_2d,
    RULE_TO_GATE4,
    vote_clip_gate4,
)

FPS = 30.0


# ---------------------------------------------------------------- 合成数据工厂
# 内部 up 约定构造(y 大=高)。坐标域 [0,1], 语义组位置对齐 SMAL_GROUPS。

def _mk_pose(T: int, kind: str, vx: float = 0.0, lift: float = 0.0,
             lift_frames: tuple[int, int] | None = None) -> np.ndarray:
    """生成 (T,24,2) 合成骨架序列。kind in {standing, lying}."""
    kp = np.zeros((T, 24, 2), dtype=np.float64)
    if kind == "standing":
        ys = {"paw": 0.10, "mid": 0.25, "top": 0.50, "tail": 0.45,
              "head": 0.65, "withers": 0.55, "throat": 0.58}
    elif kind == "lying":
        ys = {"paw": 0.08, "mid": 0.10, "top": 0.15, "tail": 0.13,
              "head": 0.22, "withers": 0.14, "throat": 0.16}
    else:
        raise ValueError(kind)
    for t in range(T):
        dx = vx * t
        dy = 0.0
        if lift and lift_frames and lift_frames[0] <= t < lift_frames[1]:
            # 5 帧线性渐变抬升, 模拟真实起跳
            t0 = lift_frames[0]
            frac = min(1.0, (t - t0 + 1) / 5.0)
            dy = lift * frac
        kp[t, [0, 3, 6, 9], 0] = np.array([0.30, 0.42, 0.58, 0.70]) + dx
        kp[t, [0, 3, 6, 9], 1] = ys["paw"] + dy
        kp[t, [1, 4, 7, 10], 0] = np.array([0.33, 0.44, 0.58, 0.69]) + dx
        kp[t, [1, 4, 7, 10], 1] = ys["mid"] + dy
        kp[t, [2, 5, 8, 11], 0] = np.array([0.36, 0.46, 0.60, 0.68]) + dx
        kp[t, [2, 5, 8, 11], 1] = ys["top"] + dy
        kp[t, [12, 13], 0] = np.array([0.80, 0.88]) + dx
        kp[t, [12, 13], 1] = ys["tail"] + dy
        kp[t, list(range(14, 22)), 0] = np.linspace(0.15, 0.26, 8) + dx
        kp[t, list(range(14, 22)), 1] = ys["head"] + dy
        kp[t, 22, 0] = 0.48 + dx
        kp[t, 22, 1] = ys["withers"] + dy
        kp[t, 23, 0] = 0.30 + dx
        kp[t, 23, 1] = ys["throat"] + dy
    return kp


def _weights(T: int, conf: float = 0.9) -> np.ndarray:
    return np.full((T, 24), conf, dtype=np.float64)


# ---------------------------------------------------------------- 1. y 约定归一层

class TestNormalizeYOrientation:
    def test_down_flips_to_internal_up(self):
        kp = np.zeros((4, 24, 2))
        kp[:, :, 1] = 0.7  # down 约定: 图像下部
        out = normalize_y_orientation(kp, y_axis="down")
        assert np.allclose(out[:, :, 1], -0.7)

    def test_up_is_passthrough(self):
        kp = np.zeros((4, 24, 2))
        kp[:, :, 1] = 0.3
        out = normalize_y_orientation(kp, y_axis="up")
        assert np.allclose(out, kp)

    def test_invalid_axis_raises(self):
        with pytest.raises(ValueError):
            normalize_y_orientation(np.zeros((2, 24, 2)), y_axis="diagonal")

    def test_x_channel_untouched(self):
        rng = np.random.default_rng(0)
        kp = rng.random((5, 24, 2))
        out = normalize_y_orientation(kp.copy(), y_axis="down")
        assert np.allclose(out[:, :, 0], kp[:, :, 0])


# ---------------------------------------------------------------- 2. 体尺度/地面

class TestScaleAndGround:
    def test_body_scale_positive(self):
        kp = _mk_pose(10, "standing")
        assert _estimate_body_scale_2d(kp) > 0

    def test_body_scale_translation_invariant(self):
        kp = _mk_pose(10, "standing")
        shifted = kp.copy()
        shifted[:, :, 0] += 5.0
        shifted[:, :, 1] -= 3.0
        assert abs(_estimate_body_scale_2d(kp) - _estimate_body_scale_2d(shifted)) < 1e-9

    def test_ground_below_withers_for_standing(self):
        kp = _mk_pose(10, "standing")
        w = _weights(10)
        ground = estimate_ground_2d(normalize_y_orientation(kp, "up"), w)
        withers_y = float(np.mean(kp[:, 22, 1]))
        assert ground < withers_y  # 内部 up 约定: 地面低于 withers


# ---------------------------------------------------------------- 3. 方向语义回归(W41 bug 修复)

class TestTransitionDirection:
    def test_rise_up_detected_on_lie_to_stand(self):
        lie = _mk_pose(12, "lying")
        stand = _mk_pose(12, "standing")
        kp = np.concatenate([lie, stand], axis=0)
        res = classify_frames_2d(kp, _weights(24), np.arange(24), DEFAULT_CONFIG_2D, y_axis="up")
        rule_hits = [r for fr in res["frame_rule_ids"] for r in fr]
        assert "rise_up" in rule_hits

    def test_lie_down_detected_on_stand_to_lie(self):
        stand = _mk_pose(12, "standing")
        lie = _mk_pose(12, "lying")
        kp = np.concatenate([stand, lie], axis=0)
        res = classify_frames_2d(kp, _weights(24), np.arange(24), DEFAULT_CONFIG_2D, y_axis="up")
        rule_hits = [r for fr in res["frame_rule_ids"] for r in fr]
        assert "lie_down" in rule_hits


# ---------------------------------------------------------------- 4. y 翻转输入不变性(核心契约)

class TestYFlipInputInvariance:
    @pytest.mark.parametrize("kind,vx,lift,lf", [
        ("standing", 0.0, 0.0, None),
        ("lying", 0.0, 0.0, None),
        ("standing", 0.03, 0.0, None),          # walking 速度
        ("standing", 0.0, 0.35, (10, 20)),       # 跳跃
    ])
    def test_same_labels_under_both_conventions(self, kind, vx, lift, lf):
        """同一物理姿态的两种数据表示(up 表示 / 图像域 flip 表示)，
        各以匹配的 y_axis 声明输入 → 分类必须一致。"""
        T = 30
        kp_up = _mk_pose(T, kind, vx=vx, lift=lift, lift_frames=lf)
        kp_img = kp_up.copy()
        kp_img[:, :, 1] = 1.0 - kp_img[:, :, 1]  # 同一姿态的图像域(y-down)表示
        fi = np.arange(T)
        res_up = classify_frames_2d(kp_up, _weights(T), fi, DEFAULT_CONFIG_2D, y_axis="up")
        res_down = classify_frames_2d(kp_img, _weights(T), fi, DEFAULT_CONFIG_2D, y_axis="down")
        assert res_up["labels"] == res_down["labels"]

    def test_flip_of_pixel_data_matches_up_original(self):
        """图像域翻转数据(y'=H-y) 以 down 声明输入 ≡ 原始 up 序列。"""
        T = 20
        kp_up = _mk_pose(T, "standing")
        kp_img = kp_up.copy()
        kp_img[:, :, 1] = 1.0 - kp_img[:, :, 1]  # H=1 翻转
        r1 = classify_frames_2d(kp_up, _weights(T), np.arange(T), DEFAULT_CONFIG_2D, y_axis="up")
        r2 = classify_frames_2d(kp_img, _weights(T), np.arange(T), DEFAULT_CONFIG_2D, y_axis="down")
        assert r1["labels"] == r2["labels"]


# ---------------------------------------------------------------- 5. 合成姿态端到端

class TestSyntheticPostures:
    def test_standing_dominates_static_upright(self):
        kp = _mk_pose(30, "standing")
        res = classify_frames_2d(kp, _weights(30), np.arange(30), DEFAULT_CONFIG_2D, y_axis="up")
        assert res["labels"].count("standing") >= 24

    def test_lying_dominates_low_profile(self):
        kp = _mk_pose(30, "lying")
        res = classify_frames_2d(kp, _weights(30), np.arange(30), DEFAULT_CONFIG_2D, y_axis="up")
        assert res["labels"].count("lying") >= 24

    def test_walking_at_moderate_speed(self):
        kp = _mk_pose(30, "standing", vx=0.02)
        res = classify_frames_2d(kp, _weights(30), np.arange(30), DEFAULT_CONFIG_2D, y_axis="up")
        assert res["labels"].count("walking") >= 20

    def test_running_at_high_speed(self):
        kp = _mk_pose(30, "standing", vx=0.06)
        res = classify_frames_2d(kp, _weights(30), np.arange(30), DEFAULT_CONFIG_2D, y_axis="up")
        assert res["labels"].count("running") >= 20

    def test_jump_has_airborne_frames(self):
        kp = _mk_pose(30, "standing", lift=0.35, lift_frames=(10, 20))
        res = classify_frames_2d(kp, _weights(30), np.arange(30), DEFAULT_CONFIG_2D, y_axis="up")
        assert "jump" in res["labels"]
        rule_hits = [r for fr in res["frame_rule_ids"] for r in fr]
        assert "jump_airborne" in rule_hits


# ---------------------------------------------------------------- 6. 置信度掩码与死关节数据防御

class TestMasksAndDegenerateData:
    def test_zero_conf_joints_do_not_crash(self):
        kp = _mk_pose(10, "standing")
        w = _weights(10).copy()
        w[:, :12] = 0.0  # 下半身不可见
        feat = compute_frame_features_2d(
            normalize_y_orientation(kp, "up"), w, np.arange(10), DEFAULT_CONFIG_2D)
        for key in ("clearance", "hip_ratio", "centroid_speed"):
            assert np.all(np.isfinite(feat[key]))

    def test_dead_joint_allzero_channels_defended(self):
        """AK 死关节 idx20-23 全零通道(conf=0)不得产生伪地面/伪爪高。"""
        kp = _mk_pose(10, "standing")
        w = _weights(10).copy()
        kp[:, 20:24, :] = 0.0
        w[:, 20:24] = 0.0
        feat = compute_frame_features_2d(
            normalize_y_orientation(kp, "up"), w, np.arange(10), DEFAULT_CONFIG_2D)
        # 站立犬 paw_air 应接近 0(爪在地上), 不能被零点拉爆
        assert float(np.max(feat["paw_air"])) < 0.2


# ---------------------------------------------------------------- 7. gate4 映射协议

class TestGate4Mapping:
    def test_mapping_covers_full_rule_vocabulary(self):
        expected = {"lying", "sitting", "standing", "walking",
                    "running", "rise_transition", "jump", "unknown"}
        assert set(RULE_TO_GATE4.keys()) == expected

    def test_posture_maps_to_stay_and_gait_to_track(self):
        assert RULE_TO_GATE4["standing"] == "stay"
        assert RULE_TO_GATE4["sitting"] == "stay"
        assert RULE_TO_GATE4["lying"] == "stay"
        assert RULE_TO_GATE4["walking"] == "track"
        assert RULE_TO_GATE4["running"] == "track"
        assert RULE_TO_GATE4["jump"] == "jump"

    def test_watch_not_derivable_from_geometry(self):
        assert RULE_TO_GATE4["watch"] if False else RULE_TO_GATE4.get("watch", None) is None

    def test_transition_and_unknown_map_to_none(self):
        assert RULE_TO_GATE4["rise_transition"] is None
        assert RULE_TO_GATE4["unknown"] is None

    def test_vote_majority_stay(self):
        labels = ["standing"] * 30
        assert vote_clip_gate4(labels) == "stay"

    def test_vote_track_with_minor_unknown(self):
        labels = ["walking"] * 25 + ["unknown"] * 5
        assert vote_clip_gate4(labels) == "track"

    def test_vote_abstain_when_coverage_below_half(self):
        labels = ["unknown"] * 16 + ["lying"] * 14
        assert vote_clip_gate4(labels) == "abstain"

    def test_vote_empty_clip_abstains(self):
        assert vote_clip_gate4([]) == "abstain"


# ---------------------------------------------------------------- 8. 主入口契约

class TestGenerateSeeds2DContract:
    def test_output_structure(self):
        kp = _mk_pose(30, "standing", vx=0.02)
        out = generate_seeds_2d(kp, _weights(30), np.arange(30), DEFAULT_CONFIG_2D, y_axis="up")
        for key in ("frame_labels", "frame_confidence", "frame_rule_ids",
                    "segments", "body_scale", "ground_height"):
            assert key in out
        assert len(out["frame_labels"]) == 30
        assert len(out["frame_confidence"]) == 30
        for seg in out["segments"]:
            for k in ("start_frame", "end_frame", "label", "confidence", "rule_ids"):
                assert k in seg

    def test_config_overrides_thresholds(self):
        kp = _mk_pose(30, "standing", vx=0.02)
        cfg = dict(DEFAULT_CONFIG_2D)
        cfg["speed"] = dict(cfg["speed"], walk_min=5.0)  # 提到不可能的速度
        res = classify_frames_2d(kp, _weights(30), np.arange(30), cfg, y_axis="up")
        assert "walking" not in res["labels"]
