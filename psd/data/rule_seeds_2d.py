"""2D 规则引擎粗标（像素域规则种子）— W47 补救窗重建（W30 移交项）。

替代 3D 引擎 psd/data/rule_seeds.py 消费 2D 像素域骨架
(AK partialclass4 / APTv2 / w35 提点等 (T,24,[x,y,conf]) 数据)。

与 3D 版的三点结构对应（W40 round2 报告 §5 燃料第 2 条的设计回应）：

1. **y 轴约定归一化层**（本版核心修正）：像素域 y 向下为正（图像约定），
   也有数据源向上为正——引擎入口 normalize_y_orientation 把任意约定归一
   到内部 up=+y，其后全部特征**带符号**计算。由此同时获得：
   - rise_up/lie_down 方向语义正确（W41 上版"先 abs 后求导"bug 的根治；
     方向判定与 y 翻转不变性在特征层不可兼得，故把不变性上移到入口层）；
   - 输入约定无关性：同一物理姿态无论以何种 y 约定输入，输出一致。
2. **bounding-box 体尺度归一**替代 3D 躯干四顶点距离：逐帧可见关键点
   包围盒对角线的中位数，纯 2D 可得且平移不变。
3. **七类规则 → gate4 显式映射协议**：AK partialclass4 有标体系为行为类
   {watch, track, jump, stay}，与姿态七类不同构——映射表 RULE_TO_GATE4
   是唯一的语义桥，watch 为骨架几何不可判别类如实映射 None（禁止臆造）。
   阈值初值承 3D 物理先验，校准协议见 scripts/calibrate_rule_seeds_2d.py。

关节组沿用 SMAL_GROUPS 索引（Q3b 提点已拓扑对齐 K9Graph 24kp）；
死关节 {20..23} 在上游硬掩码置零（ADR 死关节事件），conf 掩码天然防御。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- 关节组（几何验证结论；左右无关）

SMAL_GROUPS: dict[str, list[int]] = {
    "paws": [0, 3, 6, 9],
    "front_tops": [2, 8],
    "rear_tops": [5, 11],
    "withers": [22],
    "throat": [23],
    "head": [14, 15, 16, 17, 18, 19, 20, 21],
    "tail": [12, 13],
}

_MIN_VALID_JOINTS = 8
_WEIGHT_FLOOR = 0.05
_FALLBACK_SCALE = 0.5

_LABEL_UNKNOWN = "unknown"

# 七类规则 → gate4 映射（AK partialclass4 校准/门禁的唯一语义桥）。
# watch（注视镜头类行为）无法由骨架几何判别——如实 None，不臆造。
RULE_TO_GATE4: dict[str, str | None] = {
    "standing": "stay",
    "sitting": "stay",
    "lying": "stay",
    "walking": "track",
    "running": "track",
    "jump": "jump",
    "rise_transition": None,
    "unknown": None,
}

# 阈值初值：承 3D 物理先验量纲（体尺度归一无量纲域，与 3D 版可比）。
# AK partialclass4 172 clips 实测校准记录见 reports/rule-seeds-2d-*.md；
# 校准若修订阈值，须连同证据一并更新此处（truth 单一性在本文件）。
DEFAULT_CONFIG_2D: dict = {
    "nominal_fps": 30.0,
    "posture": {
        "standing_min_clearance": 0.35,
        "lying_max_clearance": 0.18,
        "lying_composite_max": 0.75,
        "sitting_max_hip_ratio": 0.55,
    },
    "speed": {"walk_min": 0.30, "run_min": 1.20, "smooth_window": 5},
    "transition": {"rate_min": 1.5, "window": 5},
    "jump": {"min_air_clearance": 0.25, "spike_over_standing": 0.15},
    "segment": {"min_duration_s": 0.3},
}


# ---------------------------------------------------------------- y 轴约定归一层

def normalize_y_orientation(kp: np.ndarray, y_axis: str) -> np.ndarray:
    """把任意 y 约定的 (T,24,2) 输入归一到内部 up=+y 约定。

    y_axis="down": 图像坐标惯例（y 向下增大）→ 取反。
    y_axis="up":   数学/世界惯例 → 原样通过。
    其余值 ValueError（宁可拒绝也不静默猜约定——方向语义依赖它）。
    """
    kp = np.asarray(kp, dtype=np.float64)
    if y_axis == "down":
        out = kp.copy()
        out[:, :, 1] = -out[:, :, 1]
        return out
    if y_axis == "up":
        return kp.copy()
    raise ValueError(f"未知 y 轴约定: {y_axis!r}（合法值: 'down'/'up'）")


# ---------------------------------------------------------------- NaN 安全工具

def _safe_nanmin(arr: np.ndarray, axis: int) -> np.ndarray:
    finite = np.isfinite(arr)
    all_nan = ~finite.any(axis=axis, keepdims=True)
    masked = np.where(all_nan, 0.0, np.where(finite, arr, np.inf))
    out = np.min(masked, axis=axis)
    return np.where(np.squeeze(all_nan, axis=axis), np.nan, out)


def _masked_nanmean(arr2d: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    masked = np.where(valid, arr2d, np.nan)
    n_valid = valid.sum(axis=axis)
    total = np.nansum(masked, axis=axis)
    out = np.divide(total, n_valid, out=np.full_like(total, np.nan, dtype=float),
                    where=n_valid > 0)
    return np.asarray(out)


def _interp_nan_time(arr: np.ndarray) -> np.ndarray:
    """沿时间轴线性插值填充内部 NaN；首尾常值外推；全 NaN 列保留。"""
    t_len = arr.shape[0]
    if t_len < 2:
        return arr
    flat = arr.reshape(t_len, -1)
    out = flat.copy()
    idx = np.arange(t_len, dtype=np.float64)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        good = np.isfinite(col)
        if not good.any() or good.all():
            continue
        out[:, c] = np.interp(idx, idx[good], col[good])
    return out.reshape(arr.shape)


def _smooth_along_time(arr: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or arr.shape[0] < 2:
        return arr
    pad = window // 2
    padded = np.pad(arr, ((pad, pad),) + ((0, 0),) * (arr.ndim - 1), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    smooth_shape = (arr.shape[0],) + arr.shape[1:]
    flat = padded.reshape(padded.shape[0], -1)
    out = np.empty((arr.shape[0], flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.convolve(flat[:, c], kernel, mode="valid")
    return out.reshape(smooth_shape)


# ---------------------------------------------------------------- 体尺度 / 地面（2D 版）

def _estimate_body_scale_2d(kp: np.ndarray) -> float:
    """体尺度：逐帧可见关键点包围盒对角线的中位数（替代 3D 四顶点距离）。

    平移不变、恒正；全退化时回退 _FALLBACK_SCALE。
    """
    x_min = np.nanmin(kp[:, :, 0], axis=1)
    x_max = np.nanmax(kp[:, :, 0], axis=1)
    y_min = np.nanmin(kp[:, :, 1], axis=1)
    y_max = np.nanmax(kp[:, :, 1], axis=1)
    diag = np.hypot(x_max - x_min, y_max - y_min)
    diag = diag[np.isfinite(diag) & (diag > 1e-6)]
    if diag.size == 0:
        return _FALLBACK_SCALE
    scale = float(np.median(diag))
    if not np.isfinite(scale) or scale < 1e-3:
        return _FALLBACK_SCALE
    return scale


def estimate_ground_2d(kp_internal: np.ndarray, weight: np.ndarray) -> float:
    """地面基准（内部 up=+y 约定）：逐帧最低爪 y 的跨帧中位数。

    与 3D 版 estimate_ground（median of per-frame min paw z）同构。
    """
    valid = np.isfinite(kp_internal).all(axis=2) & (weight > _WEIGHT_FLOOR)
    paw_y = np.where(valid[:, SMAL_GROUPS["paws"]],
                     kp_internal[:, SMAL_GROUPS["paws"], 1], np.nan)
    per_frame_min = _safe_nanmin(paw_y, axis=1)
    finite = per_frame_min[np.isfinite(per_frame_min)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


# ---------------------------------------------------------------- 帧级特征

def compute_frame_features_2d(
    kp_internal: np.ndarray,
    weight: np.ndarray,
    frame_idx: np.ndarray,
    config: dict,
) -> dict[str, np.ndarray]:
    """帧级特征（输入必须是已归一到 up=+y 的坐标）。

    特征族与 3D 版一一对应，全部带符号（方向语义由调用方保证约定）。
    """
    kp = np.asarray(kp_internal, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    nominal_fps = float(config.get("nominal_fps", 30.0))
    smooth_window = int(config.get("speed", {}).get("smooth_window", 5))

    valid = np.isfinite(kp).all(axis=2) & (weight > _WEIGHT_FLOOR)  # (T,24)
    kp_filled = _interp_nan_time(np.where(np.isfinite(kp), kp, np.nan))

    scale = _estimate_body_scale_2d(kp_filled)
    ground = estimate_ground_2d(kp_filled, weight)

    eps = 1e-6
    with np.errstate(all="ignore"):
        withers_y = _masked_nanmean(
            kp_filled[:, SMAL_GROUPS["withers"], 1].reshape(len(kp), -1),
            valid[:, SMAL_GROUPS["withers"]], 1)
        shoulder_y = _masked_nanmean(kp_filled[:, SMAL_GROUPS["front_tops"], 1],
                                     valid[:, SMAL_GROUPS["front_tops"]], 1)
        hip_y = _masked_nanmean(kp_filled[:, SMAL_GROUPS["rear_tops"], 1],
                                valid[:, SMAL_GROUPS["rear_tops"]], 1)
        head_y = _masked_nanmean(kp_filled[:, SMAL_GROUPS["head"], 1],
                                 valid[:, SMAL_GROUPS["head"]], 1)

        clearance = (withers_y - ground) / max(scale, eps)          # 带符号
        denom = np.maximum(np.abs(shoulder_y - ground), eps * scale)
        hip_ratio = (hip_y - ground) / denom                        # up 约定下自然为正
        head_norm = (head_y - ground) / max(scale, eps)

        paw_y_valid = np.where(valid[:, SMAL_GROUPS["paws"]],
                               kp_filled[:, SMAL_GROUPS["paws"], 1], np.nan)
        # 跳跃判据取有效爪最低点（min）——均值会被摆动腿抬高（沿 3D 版口径）
        paw_min_raw = _safe_nanmin(paw_y_valid, axis=1)
        paw_air = (np.nan_to_num(paw_min_raw, nan=ground) - ground) / max(scale, eps)

    lying_composite = (np.nan_to_num(clearance, nan=0.0)
                       + 0.5 * np.nan_to_num(head_norm, nan=0.0))

    # ---- 质心水平速度（固定躯干子集 {2,5,8,11,22}，防遮挡质心跳变——3D 版 I-2 教训）
    n = len(kp)
    torso_idx = SMAL_GROUPS["front_tops"] + SMAL_GROUPS["rear_tops"] + SMAL_GROUPS["withers"]
    if n >= 2:
        sub_valid = valid[:, torso_idx]
        sub_pts = np.where(sub_valid[:, :, None], kp_filled[:, torso_idx, :], 0.0)
        cnt = sub_valid.sum(axis=1)
        centroid_raw = sub_pts.sum(axis=1) / np.maximum(cnt, 1)[:, None]
        centroid_raw[cnt == 0] = np.nan
        centroid = _interp_nan_time(centroid_raw)
        c_smooth = _smooth_along_time(np.nan_to_num(centroid, nan=0.0), smooth_window)
        d_centroid = np.linalg.norm(np.diff(c_smooth[:, :2], axis=0), axis=1)
        dt_c = np.maximum(np.diff(np.asarray(frame_idx, dtype=np.float64)), 1.0) / max(nominal_fps, 1e-6)
        centroid_speed = np.concatenate([[0.0], d_centroid / dt_c / scale])
    else:
        centroid_speed = np.zeros(n)

    # ---- 躯干高度变化率（带符号！rise_up/lie_down 方向语义所在）
    if n >= 2:
        mean_dt = float(np.mean(np.maximum(
            np.diff(np.asarray(frame_idx, dtype=np.float64)), 1.0))) / max(nominal_fps, 1e-6)
        clearance_filled = _interp_nan_time(np.where(np.isfinite(clearance), clearance, np.nan))
        d_clearance = np.gradient(clearance_filled) / max(mean_dt, eps)
    else:
        d_clearance = np.zeros(n)

    frame_ok = valid.sum(axis=1) >= _MIN_VALID_JOINTS
    return {
        "valid": valid,
        "frame_ok": frame_ok,
        "ground": ground,
        "scale": scale,
        "clearance": np.nan_to_num(clearance, nan=0.0),
        "hip_ratio": np.clip(np.nan_to_num(hip_ratio, nan=1.0), 0.0, 2.0),
        "head_norm": np.nan_to_num(head_norm, nan=0.0),
        "lying_composite": lying_composite,
        "paw_air": np.nan_to_num(paw_air, nan=0.0),
        "centroid_speed": np.nan_to_num(centroid_speed, nan=0.0),
        "d_clearance": np.nan_to_num(d_clearance, nan=0.0),
    }


# -------------------------------------------------------------- 帧级分类

def classify_frames_2d(
    kp: np.ndarray,
    weight: np.ndarray,
    frame_idx: np.ndarray,
    config: dict,
    y_axis: str = "down",
) -> dict:
    """逐帧规则判定（入口自动做 y 约定归一）。

    判定优先级：jump > transition > gait(需离地带) > sitting > lying > standing > unknown。
    """
    kp_internal = normalize_y_orientation(kp, y_axis)
    feat = compute_frame_features_2d(kp_internal, weight, frame_idx, config)

    cfg_p = config.get("posture", {})
    cfg_s = config.get("speed", {})
    cfg_t = config.get("transition", {"rate_min": 1.5, "window": 5})
    cfg_j = config.get("jump", {"min_air_clearance": 0.25, "spike_over_standing": 0.15})

    stand_min = float(cfg_p.get("standing_min_clearance", 0.35))
    lie_max = float(cfg_p.get("lying_max_clearance", 0.18))
    lie_comp_max = float(cfg_p.get("lying_composite_max", 0.75))
    sit_max_ratio = float(cfg_p.get("sitting_max_hip_ratio", 0.55))
    walk_min = float(cfg_s.get("walk_min", 0.30))
    run_min = float(cfg_s.get("run_min", 1.20))
    rate_min = float(cfg_t.get("rate_min", 1.5))
    trans_window = int(cfg_t.get("window", 5))
    jump_air_min = float(cfg_j.get("min_air_clearance", 0.25))
    jump_spike_over = float(cfg_j.get("spike_over_standing", 0.15))

    n = len(feat["clearance"])

    # 过渡候选帧按窗口膨胀（尖峰差分只盖 1-2 帧，真实起卧行为持续数百 ms）
    trans_mask = np.zeros(n, dtype=bool)
    for f_spike in np.where(np.abs(feat["d_clearance"]) > rate_min)[0]:
        lo = max(0, f_spike - trans_window)
        hi = min(n, f_spike + trans_window + 1)
        trans_mask[lo:hi] = True

    labels = np.full(n, _LABEL_UNKNOWN, dtype=object)
    conf = np.zeros(n, dtype=np.float64)
    rule_ids: list[list[str]] = [[] for _ in range(n)]

    for t in range(n):
        if not feat["frame_ok"][t]:
            continue
        c = feat["clearance"][t]
        ratio = feat["hip_ratio"][t]
        comp = feat["lying_composite"][t]
        v = feat["centroid_speed"][t]

        # 1) 跳跃：爪最低点明显离地 且 躯干高于站立线+尖峰余量
        air_min = feat["paw_air"][t]
        if air_min > jump_air_min and c > stand_min + jump_spike_over:
            labels[t] = "jump"
            conf[t] = min(1.0, 0.5 + 0.5 * min(air_min / max(jump_air_min * 2.0, 1e-6), 1.0))
            rule_ids[t] = ["jump_airborne"]
            continue

        # 2) 起卧过渡：带符号高度变化率（升=rise_up，降=lie_down）
        if trans_mask[t]:
            dc = feat["d_clearance"][t]
            labels[t] = "rise_transition"
            conf[t] = max(0.5, min(1.0, abs(dc) / (2.0 * rate_min)))
            rule_ids[t] = ["rise_up" if dc > 0 else "lie_down"]
            continue

        # 3) 步态：躯干离开地面带时的持续质心速度（卧姿蹭动不算步态）
        if c > lie_max:
            if v >= run_min:
                labels[t] = "running"
                conf[t] = 0.5 + 0.5 * min((v - run_min) / max(run_min - walk_min, 1e-6), 1.0)
                rule_ids[t] = ["gait_run"]
                continue
            if v >= walk_min:
                labels[t] = "walking"
                conf[t] = 0.5 + 0.5 * min((v - walk_min) / max(run_min - walk_min, 1e-6), 1.0)
                rule_ids[t] = ["gait_walk"]
                continue

        # 4) 坐姿：髋/肩高度比显著塌陷（头仍高，区别于卧）
        if ratio < sit_max_ratio:
            labels[t] = "sitting"
            conf[t] = 0.5 + 0.5 * min(max(sit_max_ratio - ratio, 0.0) / 0.30, 1.0)
            rule_ids[t] = ["sitting_posture"]
            continue

        # 5) 卧姿：躯干矮 或 复合证据低
        if c < lie_max or comp < lie_comp_max:
            labels[t] = "lying"
            margin = max(lie_max - c, lie_comp_max - comp)
            conf[t] = 0.5 + 0.5 * min(margin / 0.30, 1.0)
            rule_ids[t] = ["lying_posture"]
            continue

        # 6) 站立：躯干离地充分
        if c >= stand_min:
            labels[t] = "standing"
            conf[t] = 0.5 + 0.5 * min((c - stand_min) / 0.40, 1.0)
            rule_ids[t] = ["standing_posture"]
            continue

        labels[t] = _LABEL_UNKNOWN
        conf[t] = 0.0

    return {
        "labels": [str(x) for x in labels],
        "confidence": conf.astype(np.float32),
        "frame_rule_ids": rule_ids,
        "body_scale": feat["scale"],
        "ground_height": feat["ground"],
    }


# -------------------------------------------------------------- gate4 映射与 clip 投票

def vote_clip_gate4(frame_labels: list[str], min_coverage: float = 0.5) -> str:
    """帧级规则标签 → clip 级 gate4 投票。

    非 None 映射覆盖占比 < min_coverage 时弃权（abstain）——宁缺毋滥，
    弱标签进监督管线前必须诚实暴露不可判别区间。
    """
    if not frame_labels:
        return "abstain"
    votes: dict[str, int] = {}
    covered = 0
    for lab in frame_labels:
        gate = RULE_TO_GATE4.get(lab)
        if gate is not None:
            votes[gate] = votes.get(gate, 0) + 1
            covered += 1
    if covered / len(frame_labels) < min_coverage or not votes:
        return "abstain"
    return max(votes.items(), key=lambda kv: kv[1])[0]


# -------------------------------------------------------------- 分段合并

def merge_segments(
    labels: list[str],
    confidence: np.ndarray,
    rule_ids: list[list[str]],
    fps: float,
    min_duration_s: float,
) -> list[dict]:
    """同标签连续帧成段；短于 min_duration 的段反复并入置信度更高的邻段。"""
    n = len(labels)
    segs: list[list] = []
    s = 0
    for t in range(1, n + 1):
        if t == n or labels[t] != labels[s]:
            segs.append([s, t - 1, labels[s], float(confidence[s:t].mean()),
                         set(x for tt in range(s, t) for x in rule_ids[tt])])
            s = t

    min_frames = max(1, int(round(min_duration_s * fps)))
    while len(segs) > 1:
        durations = [(b - a + 1) for a, b, *_ in segs]
        i_short = int(np.argmin(durations))
        if durations[i_short] >= min_frames:
            break
        left = segs[i_short - 1] if i_short > 0 else None
        right = segs[i_short + 1] if i_short + 1 < len(segs) else None
        if left is None:
            target = i_short + 1 if right is not None else i_short
        elif right is None:
            target = i_short - 1
        else:
            target = i_short - 1 if left[3] >= right[3] else i_short + 1
        short, host = segs[i_short], segs[target]
        n_short = short[1] - short[0] + 1
        n_host = host[1] - host[0] + 1
        merged_conf = (host[3] * n_host + short[3] * n_short) / (n_host + n_short)
        merged = [min(host[0], short[0]), max(host[1], short[1]),
                  host[2], merged_conf, host[4] | short[4]]
        lo, hi = sorted((i_short, target))
        segs = segs[:lo] + [merged] + segs[hi + 1:]

    return [{
        "start_frame": int(a),
        "end_frame": int(b),
        "label": str(lab),
        "confidence": float(min(1.0, max(0.0, cf))),
        "rule_ids": sorted(rules),
    } for a, b, lab, cf, rules in segs]


# -------------------------------------------------------------- 主入口

def generate_seeds_2d(
    kp: np.ndarray,
    weight: np.ndarray,
    frame_idx: np.ndarray,
    config: dict,
    y_axis: str = "down",
) -> dict:
    """单 clip 2D 种子生成：y 归一 → 帧级分类 → 分段合并 → 结构化输出。"""
    res = classify_frames_2d(kp, weight, frame_idx, config, y_axis=y_axis)
    fps = float(config.get("nominal_fps", 30.0))
    min_dur = float(config.get("segment", {}).get("min_duration_s", 0.3))
    segments = merge_segments(res["labels"], res["confidence"],
                              res["frame_rule_ids"], fps=fps, min_duration_s=min_dur)
    return {
        "frame_labels": list(res["labels"]),
        "frame_confidence": res["confidence"],
        "frame_rule_ids": res["frame_rule_ids"],
        "segments": segments,
        "body_scale": res["body_scale"],
        "ground_height": res["ground_height"],
    }
