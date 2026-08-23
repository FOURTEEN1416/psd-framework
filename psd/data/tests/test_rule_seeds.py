"""规则引擎粗标 TDD 测试 — W6 物理层先验种子。

测试对象：psd/data/rule_seeds.py 纯函数规则引擎。
输入约定与 W3 加载器一致：kp_world (T,24,3) + kp_weight (T,24) + frame_idx (T,)。
合成骨架按功能关节组构造（关节语义经 2026-08-24 几何实测验证，见 reports/rule-seeds-*.md）：
  前肢链 {0,1,2}/{6,7,8}，后肢链 {3,4,5}/{9,10,11}（paw→mid→top），
  尾 {12,13}，头簇 {14..21}，背参考点 22（荐部），颈参考点 23。
"""
import numpy as np
import pytest

from psd.data.rule_seeds import (
    SMAL_GROUPS,
    compute_joint_speeds,
    estimate_ground,
    torso_clearance_profile,
    classify_frames,
    merge_segments,
    generate_seeds,
    _estimate_body_scale,
)

FPS = 30.0


# ---------- 合成骨架构造器 ----------

def make_dog_pose(
    n_frames: int,
    torso_h: float = 0.50,      # 荐部(22)离地高（米）
    shoulder_h: float = 0.45,   # 前肩(2,8)高
    hip_h: float = 0.38,        # 后髋(5,11)高
    paw_h: float = 0.0,         # 四爪高
    body_len: float = 0.60,     # 躯干前后跨度（肩 x=+0.25，髋 x=-0.25 附近）
    scale_ref: float = 1.0,     # 整体缩放
    wobble: float = 0.0,        # 每帧随机扰动幅度（固定种子）
    seed: int = 0,
):
    """构造 (n_frames,24,3) 合成犬骨架。

    布局：x 轴向前（鼻 +，尾 -），y 向左，z 向上。
    站立参考：paw z≈0，mid 在 paw/top 中点偏上，top 即肩/髋。
    """
    rng = np.random.default_rng(seed)
    s = scale_ref

    def joint(x, y, z):
        return np.array([x * s, y * s, z * s], dtype=np.float64)

    base = {}
    # 四肢链：前左{0,1,2} 后左{3,4,5} 前右{6,7,8} 后右{9,10,11}
    legs = {
        0: (+0.22, +0.10, shoulder_h), 2: (+0.22, +0.10, shoulder_h),
        6: (+0.22, -0.10, shoulder_h), 8: (+0.22, -0.10, shoulder_h),
        3: (-0.22, +0.10, hip_h), 5: (-0.22, +0.10, hip_h),
        9: (-0.22, -0.10, hip_h), 11: (-0.22, -0.10, hip_h),
    }
    for paw, mid, top in ((0, 1, 2), (6, 7, 8), (3, 4, 5), (9, 10, 11)):
        x, y, topz = legs[top]
        base[paw] = joint(x, y, paw_h)
        base[mid] = joint(x, y, (paw_h + topz) / 2 * 0.95)
        base[top] = joint(x, y, topz)
    # 尾：12 尾根在臀部后方高位，13 尾尖更后更高
    base[12] = joint(-0.28, 0.0, hip_h + 0.10)
    base[13] = joint(-0.40, 0.0, hip_h + 0.20)
    # 头簇：14/15 耳尖对，16 鼻（最前），17 下巴（鼻下），
    #       18/19 耳基对，20/21 眼对 —— 全部挂在肩前方上方
    hx, hy, hz = 0.34, 0.0, shoulder_h + 0.22
    base[14] = joint(hx, +0.05, hz + 0.03)
    base[15] = joint(hx, -0.05, hz + 0.03)
    base[16] = joint(hx + 0.08, hy, hz - 0.02)
    base[17] = joint(hx + 0.06, hy, hz - 0.09)
    base[18] = joint(hx - 0.01, +0.04, hz + 0.05)
    base[19] = joint(hx - 0.01, -0.04, hz + 0.05)
    base[20] = joint(hx + 0.04, +0.03, hz)
    base[21] = joint(hx + 0.04, -0.03, hz)
    # 22 荐部（躯干最高背点）、23 颈前
    base[22] = joint(0.0, 0.0, torso_h)
    base[23] = joint(+0.26, 0.0, shoulder_h + 0.06)

    kp = np.stack([base[j] for j in range(24)], axis=0)  # (24,3)
    kp = np.broadcast_to(kp[None], (n_frames, 24, 3)).copy()
    if wobble > 0:
        kp = kp + rng.normal(0.0, wobble, kp.shape)
    weight = np.full((n_frames, 24), 0.9, dtype=np.float64)
    frame_idx = np.arange(n_frames, dtype=np.int32)  # 稠密采样：相邻帧号差 1
    return kp, weight, frame_idx


DEFAULT_CONFIG = {
    "nominal_fps": FPS,
    "classes": ["lying", "sitting", "standing", "walking", "running",
                "rise_transition", "jump"],
    "speed": {"walk_min": 0.30, "run_min": 1.20},
    "posture": {"standing_min_clearance": 0.35, "lying_max_clearance": 0.18,
                "sitting_max_hip_ratio": 0.55},
    "transition": {"rate_min": 1.5, "window": 5},
    "jump": {"min_air_clearance": 0.25, "spike_over_standing": 0.15},
    "segment": {"min_duration_s": 0.3, "unknown_gap_fill": False},
}


# ---------- 用例 1：关节速度计算与阈值边界 ----------

def test_velocity_computation_exact_for_constant_motion():
    """恒速移动的爪：有限差分速度应精确还原 |v|；静止段速度≈0。"""
    kp, weight, fidx = make_dog_pose(60)
    speed_true = 1.0                      # 米/秒
    dt = 1.0 / FPS
    # 整只狗沿 x 平移（质心运动不改变姿态分类）
    shift = np.zeros((60, 1, 3))
    shift[30:, :, 0] = speed_true * dt
    kp = kp + np.cumsum(shift, axis=0)
    speeds = compute_joint_speeds(kp, fidx, nominal_fps=FPS)  # (T,24) 归一化单位
    assert speeds.shape == (60, 24)
    assert np.isfinite(speeds).all()
    # 前 30 帧静止：所有关节速度≈0（平移从第 30 帧开始）
    assert np.allclose(speeds[:29], 0.0, atol=1e-9)
    # 平移开始后：非零，且量级 ≈ speed_true / body_scale（体尺度由躯干对角估计 ~0.49m）
    body_scale = _estimate_body_scale(kp)
    tail_mean = float(np.mean(speeds[32:, :]))
    expected = speed_true / body_scale
    assert abs(tail_mean - expected) < 0.2 * expected, \
        f"稳态速度 {tail_mean:.3f} 应接近 {expected:.3f}"


def test_velocity_threshold_boundary_static_vs_walk():
    """速度阈值边界：低于 walk_min 判静止姿态类，高于则进入步态类。"""
    slow_kp, w, fidx = make_dog_pose(90, wobble=0.002, seed=1)
    # 缓慢整体漂移：0.001 m/帧 @30fps = 0.03 m/s ≈ 0.06 体长单位/s < walk_min
    drift_slow = np.zeros((90, 1, 3)); drift_slow[:, :, 0] = 0.001
    kp_slow = slow_kp + np.cumsum(drift_slow, axis=0)
    res_slow = classify_frames(kp_slow, w, fidx, DEFAULT_CONFIG)
    fast_kp = slow_kp.copy()
    # 快速平移：0.03 m/帧 @30fps = 0.9 m/s ≈ 1.8 体长单位/s > walk_min
    drift_fast = np.zeros((90, 1, 3)); drift_fast[:, :, 0] = 0.03
    kp_fast = fast_kp + np.cumsum(drift_fast, axis=0)
    res_fast = classify_frames(kp_fast, w, fidx, DEFAULT_CONFIG)
    assert set(res_slow["labels"]) <= set(DEFAULT_CONFIG["classes"]) | {"unknown"}
    assert "walking" not in res_slow["labels"], "慢速漂移不应触发 walking"
    # 0.9 m/s ≈ 1.8 体长单位/s，超过 run_min(0.60) —— 正确触发 running 步态规则
    assert "running" in res_fast["labels"], "快速平移帧应触发跑步步态规则"
    assert not ({"walking", "running"} & set(res_slow["labels"])), \
        "慢速漂移不应进入任何步态类"


# ---------- 用例 2：尺度不变性 ----------

def test_posture_classification_is_scale_invariant():
    """同一姿态放大 2 倍后类别不变（规则按体尺度归一）。"""
    kp_a, w_a, f_a = make_dog_pose(45, torso_h=0.50, shoulder_h=0.45, hip_h=0.38)
    kp_b, w_b, f_b = make_dog_pose(45, torso_h=1.00, shoulder_h=0.90, hip_h=0.76,
                                   body_len=1.20, scale_ref=2.0)
    ra = classify_frames(kp_a, w_a, f_a, DEFAULT_CONFIG)
    rb = classify_frames(kp_b, w_b, f_b, DEFAULT_CONFIG)
    assert ra["labels"] == rb["labels"]
    assert ra["labels"][0] == "standing"


# ---------- 用例 3：卧姿 vs 站姿 ----------

def test_lying_vs_standing_postures():
    """荐部贴地 → lying；荐部高位 → standing。"""
    stand_kp, sw, sf = make_dog_pose(40, torso_h=0.50, shoulder_h=0.45, hip_h=0.38)
    r_stand = classify_frames(stand_kp, sw, sf, DEFAULT_CONFIG)
    assert r_stand["labels"][0] == "standing"
    # 卧姿：整个身体压到近地，腿折叠（mid/top 都贴近地面）
    lie_kp, lw, lf = make_dog_pose(40, torso_h=0.14, shoulder_h=0.12, hip_h=0.11,
                                   paw_h=0.0, body_len=0.62)
    # 折叠腿：mid 不再位于 paw/top 中垂线上方而是接近水平——直接压低 top 即可表达
    r_lie = classify_frames(lie_kp, lw, lf, DEFAULT_CONFIG)
    assert r_lie["labels"][0] == "lying"


# ---------- 用例 4：坐姿（后躯下沉、前躯保持） ----------

def test_sitting_posture_rear_down_front_up():
    """髋部高度比低于阈值而荐部离地高于卧线 → sitting。"""
    sit_kp, cw, cf = make_dog_pose(40, torso_h=0.34, shoulder_h=0.44, hip_h=0.10)
    r = classify_frames(sit_kp, cw, cf, DEFAULT_CONFIG)
    assert r["labels"][0] == "sitting"


# ---------- 用例 5：起卧过渡事件检测 ----------

def test_rise_transition_event_detected():
    """前半段卧、后半段站：翻转窗口应产生 rise_transition 种子段。"""
    lie_kp, _, _ = make_dog_pose(30, torso_h=0.14, shoulder_h=0.12, hip_h=0.11)
    stand_kp, _, _ = make_dog_pose(30, torso_h=0.50, shoulder_h=0.45, hip_h=0.38)
    kp = np.concatenate([lie_kp, stand_kp], axis=0)
    w = np.full((60, 24), 0.9)
    fidx = np.arange(60, dtype=np.int32)  # 稠密采样，与 make_dog_pose 约定一致
    out = generate_seeds(kp, w, fidx, DEFAULT_CONFIG)
    labels = [seg["label"] for seg in out["segments"]]
    assert "rise_transition" in labels, f"segments={out['segments']}"
    trans_seg = next(s for s in out["segments"] if s["label"] == "rise_transition")
    assert trans_seg["confidence"] >= 0.0
    assert trans_seg["rule_ids"]


# ---------- 用例 6：分段合并与字段完整性 ----------

def test_segment_merge_respects_min_duration_and_fields():
    """短于 min_duration 的抖动标签被并入邻段；段记录字段完整、置信度有界。"""
    kp, w, fidx = make_dog_pose(120, wobble=0.001, seed=7)
    out = generate_seeds(kp, w, fidx, DEFAULT_CONFIG)
    segs = out["segments"]
    assert len(segs) >= 1
    need = {"start_frame", "end_frame", "label", "confidence", "rule_ids"}
    for s in segs:
        assert need <= set(s.keys())
        assert 0.0 <= s["confidence"] <= 1.0
        assert s["start_frame"] <= s["end_frame"]
        dur = (s["end_frame"] - s["start_frame"] + 1) / FPS
        assert dur >= DEFAULT_CONFIG["segment"]["min_duration_s"] - 1e-6, \
            f"段过短: {s}"
    # 无缝覆盖全时间轴
    assert segs[0]["start_frame"] == 0
    assert segs[-1]["end_frame"] == 119
    for a, b in zip(segs[:-1], segs[1:]):
        assert b["start_frame"] == a["end_frame"] + 1


# ---------- 用例 7：NaN 与零置信度鲁棒性 ----------

def test_nan_and_zero_weight_frames_degrade_to_unknown():
    """含 NaN 关节/零权重帧：不崩溃、不传播 NaN、退化 unknown 或跳过该证据。"""
    kp, w, fidx = make_dog_pose(40)
    kp_nan = kp.copy()
    kp_nan[10:15, :, :] = np.nan           # 连续 5 帧全 NaN
    w_zero = w.copy()
    w_zero[:, 16] = 0.0                     # 鼻部全程零置信
    out = generate_seeds(kp_nan, w_zero, fidx, DEFAULT_CONFIG)
    arr = np.asarray(out["frame_confidence"], dtype=np.float64)
    assert np.isfinite(arr).all()
    assert len(out["segments"]) >= 1


# ---------- 用例 8：步态摆动腿不得误判为跳跃（真实数据回归缺陷） ----------

def test_partial_paw_lift_never_trigger_jump():
    """部分爪抬离地面 ≠ 跳跃：必须四爪同时离地才可判 jump。

    回归背景：全量跑真实数据时 jump 占 55.4%，根因是四爪"均值"离地
    被摆动腿抬高。物理判据应为四爪最小离地高度（min-air）。
    构造：双前爪踩上 0.20m 高台静止 —— 均值语义下 air=0.20*2/4/scale≈0.2
    会越过 0.08 阈值误触发；min 语义下 air=0 正确放行。
    """
    n = 60
    kp, w, fidx = make_dog_pose(n)
    kp[:, 0, 2] += 0.20   # 左前爪上台
    kp[:, 6, 2] += 0.20   # 右前爪上台
    res = classify_frames(kp, w, fidx, DEFAULT_CONFIG)
    assert "jump" not in set(res["labels"]), \
        f"双爪踩台静止被判为跳跃: {sum(1 for x in res['labels'] if x == 'jump')} 帧"


# ---------- 辅助函数单元行为 ----------

def test_estimate_ground_and_groups_contract():
    """地面估计：四爪所在平面高度；SMAL_GROUPS 必须暴露规则所需功能组键。"""
    kp, _, _ = make_dog_pose(10)
    g = estimate_ground(kp)
    assert abs(g - 0.0) < 0.05
    for key in ("paws", "front_tops", "rear_tops", "withers", "head", "tail"):
        assert key in SMAL_GROUPS
        idx = SMAL_GROUPS[key]
        assert all(0 <= i < 24 for i in idx)


def test_torso_clearance_profile_normalized():
    """躯干离地廓线：站立>卧姿；输出无量纲（除以体尺度）且有限。"""
    st, _, _ = make_dog_pose(20, torso_h=0.50, shoulder_h=0.45, hip_h=0.38)
    ly, _, _ = make_dog_pose(20, torso_h=0.14, shoulder_h=0.12, hip_h=0.11)
    cst = torso_clearance_profile(st)
    cly = torso_clearance_profile(ly)
    assert cst.shape == (20,) and cly.shape == (20,)
    assert np.isfinite(cst).all() and np.isfinite(cly).all()
    assert cst.mean() > cly.mean() * 2, f"站{cst.mean():.2f} 应显著高于卧 {cly.mean():.2f}"
