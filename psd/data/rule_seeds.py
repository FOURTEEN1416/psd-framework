"""规则引擎粗标（物理层先验种子）— W6 owner。

纯函数规则引擎：输入 W3 加载器口径的数组（kp_world (T,24,3) / kp_weight (T,24) /
frame_idx (T,)），输出种子段（类别 + 置信度 + 命中规则 ID）。

关节语义来源（2026-08-24 几何实测验证，非官方命名表，证据见 reports/rule-seeds-*.md）：
- 跨 12 只狗的平均相对位置显示 z 轴为竖直轴；站立帧四爪 z≈0。
- 四肢链 {0,1,2}/{3,4,5}/{6,7,8}/{9,10,11} 均呈 paw->mid->top 高度递增；
  前链顶端 (2,8) 高于后链顶端 (5,11)，与犬类肩高>髋高的解剖一致。
- 头簇 {14..21} 恒处最高带；16 最靠前方判为鼻，17 在其正下判为下巴；
  13（尾尖）与部分头簇关节置信度长期为 0 —— 所有统计做置信度掩码。
- 左右身份未定（y 方向跨 clip 世界系旋转被抹匀），本引擎只用功能组不依赖左右。

类别体系（物理先验 7 类 + unknown，YAML 可配）：
lying / sitting / standing / walking / running / rise_transition / jump
"""
from __future__ import annotations

import numpy as np

# 功能关节组（几何验证结论；左右无关）
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


# ---------------------------------------------------------------- 基础度量

def _estimate_body_scale(kp: np.ndarray) -> float:
    """体尺度：躯干四顶点 {2,5,8,11} 每帧最大两两距离的中位数（米）。"""
    pts = kp[:, SMAL_GROUPS["front_tops"] + SMAL_GROUPS["rear_tops"], :]  # (T,4,3)
    iu = np.triu_indices(4, k=1)
    d = np.linalg.norm(pts[:, iu[0]] - pts[:, iu[1]], axis=2)  # (T,6)
    frame_max = _safe_nanmax(np.where(np.isfinite(d), d, np.nan), axis=1)
    scale = float(np.nanmedian(frame_max)) if np.isfinite(frame_max).any() else _FALLBACK_SCALE
    if not np.isfinite(scale) or scale < 1e-3:
        scale = _FALLBACK_SCALE
    return scale


def estimate_ground(kp: np.ndarray) -> float:
    """地面高度：逐帧四爪最低 z 的中位数。全 NaN 时回退 0。"""
    paw_z = kp[:, SMAL_GROUPS["paws"], 2]  # (T,4)
    per_frame_min = _safe_nanmin(paw_z, axis=1)
    if np.isfinite(per_frame_min).any():
        return float(np.nanmedian(per_frame_min))
    return 0.0


def compute_joint_speeds(
    kp: np.ndarray,
    frame_idx: np.ndarray,
    nominal_fps: float = 30.0,
    body_scale: float | None = None,
) -> np.ndarray:
    """逐帧关节速度 (T,24)，单位：体尺度/秒。

    dt 由 frame_idx 差分 / nominal_fps 得出（容忍丢帧）；首帧复制次帧值。
    """
    kp = np.asarray(kp, dtype=np.float64)
    t_len = kp.shape[0]
    if body_scale is None:
        body_scale = _estimate_body_scale(kp)
    if t_len < 2:
        return np.zeros((t_len, 24), dtype=np.float64)
    step = np.linalg.norm(np.diff(kp, axis=0), axis=2)  # (T-1,24)
    didx = np.diff(np.asarray(frame_idx, dtype=np.float64))
    dt = np.maximum(didx, 1.0) / max(float(nominal_fps), 1e-6)
    speeds = step / dt[:, None] / body_scale
    return np.vstack([speeds[:1], speeds]).astype(np.float64)


def torso_clearance_profile(kp: np.ndarray) -> np.ndarray:
    """躯干离地廓线 (T,)：(荐部 z - 地面) / 体尺度，无量纲。"""
    kp = np.asarray(kp, dtype=np.float64)
    ground = estimate_ground(kp)
    scale = _estimate_body_scale(kp)
    withers_z = kp[:, SMAL_GROUPS["withers"], 2].reshape(-1)
    out = (withers_z - ground) / scale
    return np.asarray(out, dtype=np.float64)


# -------------------------------------------------------------- 帧级特征

def _safe_nanmin(arr: np.ndarray, axis: int) -> np.ndarray:
    """全 NaN 行安全 nanmin：返回 NaN 而不触发 RuntimeWarning。"""
    finite = np.isfinite(arr)
    all_nan = ~finite.any(axis=axis, keepdims=True)
    masked = np.where(all_nan, 0.0, np.where(finite, arr, np.inf))
    out = np.min(masked, axis=axis)
    return np.where(np.squeeze(all_nan, axis=axis), np.nan, out)


def _safe_nanmax(arr: np.ndarray, axis: int) -> np.ndarray:
    """全 NaN 行安全 nanmax。"""
    finite = np.isfinite(arr)
    all_nan = ~finite.any(axis=axis, keepdims=True)
    masked = np.where(all_nan, 0.0, np.where(finite, arr, -np.inf))
    out = np.max(masked, axis=axis)
    return np.where(np.squeeze(all_nan, axis=axis), np.nan, out)


def _masked_nanmean(arr2d: np.ndarray, valid: np.ndarray, axis: int) -> np.ndarray:
    """掩码均值；全无效行返回 NaN 且不告警。"""
    masked = np.where(valid, arr2d, np.nan)
    n_valid = valid.sum(axis=axis)
    total = np.nansum(masked, axis=axis)
    out = np.divide(total, n_valid, out=np.full_like(total, np.nan, dtype=float),
                    where=n_valid > 0)
    return np.asarray(out)


def _smooth_along_time(arr: np.ndarray, window: int) -> np.ndarray:
    """沿时间轴的居中滑动平均（边缘复制填充），用于抑制逐帧抖动噪声。

    window<=1 时原样返回。输入 (T, ...)。
    """
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


def _frame_features(
    kp: np.ndarray,
    weight: np.ndarray,
    frame_idx: np.ndarray,
    nominal_fps: float,
    smooth_window: int = 5,
) -> dict[str, np.ndarray]:
    """全部规则所需的帧级特征（无效关节掩码后计算）。

    速度在平滑轨迹上差分（居中滑动平均，标准运动学去噪），避免逐帧
    拟合抖动被差分放大为虚假速度。
    """
    kp = np.asarray(kp, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    valid = np.isfinite(kp).all(axis=2) & (weight > _WEIGHT_FLOOR)  # (T,24)

    ground = estimate_ground(kp)
    scale = _estimate_body_scale(kp)
    kp_smooth = _smooth_along_time(np.where(np.isfinite(kp), kp, 0.0), smooth_window)
    # NaN 帧平滑后仍视为无效（valid 掩码不因平滑改变）
    speeds = compute_joint_speeds(kp_smooth, frame_idx, nominal_fps, body_scale=scale)

    eps = 1e-6
    with np.errstate(all="ignore"):
        clearance = (_masked_nanmean(kp[:, SMAL_GROUPS["withers"], 2].reshape(len(kp), -1),
                                     valid[:, SMAL_GROUPS["withers"]], 1) - ground) / scale
        shoulder_z = _masked_nanmean(kp[:, SMAL_GROUPS["front_tops"], 2], valid[:, SMAL_GROUPS["front_tops"]], 1)
        hip_z = _masked_nanmean(kp[:, SMAL_GROUPS["rear_tops"], 2], valid[:, SMAL_GROUPS["rear_tops"]], 1)
        head_z = _masked_nanmean(kp[:, SMAL_GROUPS["head"], 2], valid[:, SMAL_GROUPS["head"]], 1)
        paw_z_valid = np.where(valid[:, SMAL_GROUPS["paws"]], kp[:, SMAL_GROUPS["paws"], 2], np.nan)
        # 跳跃判据 = 四爪"同时"离地 → 取有效爪的最低点（min），均值会被摆动腿抬高
        paw_min_raw = _safe_nanmin(paw_z_valid, axis=1)
        paw_air = (np.nan_to_num(paw_min_raw, nan=ground) - ground) / scale
        paw_speed = _masked_nanmean(speeds[:, SMAL_GROUPS["paws"]], valid[:, SMAL_GROUPS["paws"]], 1)

    denom = np.maximum(shoulder_z - ground, eps * scale)
    hip_ratio = (hip_z - ground) / denom
    head_norm = (head_z - ground) / scale

    # 步态信号：质心水平速度（xz 平面，体尺度/秒）——比爪速稳，无摆腿周期伪迹
    n = len(kp)
    if n >= 2:
        # 按关节轴平均（保留 xyz）：无效关节不参与质心；全无效帧记 0（frame_ok 会屏蔽）
        kp_clean = np.where(valid[:, :, None], np.nan_to_num(kp, nan=0.0), 0.0)
        centroid = kp_clean.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)[:, None]  # (T,3)
        c_smooth = _smooth_along_time(centroid, smooth_window)
        d_centroid = np.linalg.norm(np.diff(c_smooth[:, :2], axis=0), axis=1)
        didx_c = np.diff(np.asarray(frame_idx, dtype=np.float64))
        dt_c = np.maximum(didx_c, 1.0) / max(float(nominal_fps), 1e-6)
        centroid_speed = np.concatenate([[0.0], d_centroid / dt_c / scale])
    else:
        centroid_speed = np.zeros(n)

    # 躯干高度变化率（体尺度/秒）：均匀 dt 近似即可满足规则粒度
    if n >= 2:
        mean_dt = float(np.mean(np.maximum(np.diff(np.asarray(frame_idx, dtype=np.float64)), 1.0))) / max(float(nominal_fps), 1e-6)
        d_clearance = np.gradient(np.nan_to_num(clearance, nan=0.0)) / max(mean_dt, eps)
    else:
        d_clearance = np.zeros(n)

    # 卧姿复合证据：躯干矮 + 头低（坐姿头保持高位，以此区分）
    lying_composite = np.nan_to_num(clearance, nan=0.0) + 0.5 * np.nan_to_num(head_norm, nan=0.0)

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

def classify_frames(
    kp: np.ndarray,
    weight: np.ndarray,
    frame_idx: np.ndarray,
    config: dict,
) -> dict:
    """逐帧规则判定。返回 labels/confidence/rule_ids + 度量元数据。

    判定优先级：jump > transition > gait(需离地) > sitting > lying > standing > unknown。
    """
    feat = _frame_features(kp, weight, frame_idx, float(config.get("nominal_fps", 30.0)),
                           smooth_window=int(config.get("speed", {}).get("smooth_window", 5)))
    cfg_p = config.get("posture", {})
    cfg_s = config.get("speed", {})
    cfg_t = config.get("transition", {"rate_min": 0.8, "window": 5})
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

    n = len(feat["clearance"])

    # 过渡候选帧按窗口膨胀：尖峰差分只覆盖 1-2 帧，真实起卧动作持续数百毫秒，
    # 膨胀后才能形成满足最短时长的种子段。
    trans_mask = np.zeros(n, dtype=bool)
    spike_frames = np.where(np.abs(feat["d_clearance"]) > rate_min)[0]
    for f_spike in spike_frames:
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

        # 1) 跳跃：四爪同时明显离地(min-air)且躯干高于站立线+尖峰余量
        air_min = feat["paw_air"][t]
        if air_min > jump_air_min and \
           c > stand_min + float(cfg_j.get("spike_over_standing", 0.15)):
            labels[t] = "jump"
            conf[t] = min(1.0, 0.5 + 0.5 * min(air_min / max(jump_air_min * 2.0, 1e-6), 1.0))
            rule_ids[t] = ["jump_airborne"]
            continue

        # 2) 起卧过渡：躯干高度变化率超阈值（升=rise_up，降=lie_down）
        if trans_mask[t]:
            dc = feat["d_clearance"][t]
            labels[t] = "rise_transition"
            conf[t] = max(0.5, min(1.0, abs(dc) / (2.0 * rate_min)))
            rule_ids[t] = ["rise_up" if dc > 0 else "lie_down"]
            continue

        # 3) 步态：躯干离开地面带时的持续爪速（卧姿蹭动不算步态）
        if c > lie_max:
            if v >= run_min:
                labels[t] = "running"
                conf[t] = 0.5 + 0.5 * min((v - run_min) / 0.8, 1.0)
                rule_ids[t] = ["gait_run"]
                continue
            if v >= walk_min:
                labels[t] = "walking"
                conf[t] = 0.5 + 0.5 * min((v - walk_min) / max(run_min - walk_min, 1e-6), 1.0)
                rule_ids[t] = ["gait_walk"]
                continue

        # 4) 坐姿：髋/肩高度比显著塌陷（在卧姿复合判断之前——坐姿头仍高）
        if ratio < sit_max_ratio:
            labels[t] = "sitting"
            conf[t] = 0.5 + 0.5 * min(max(sit_max_ratio - ratio, 0.0) / 0.30, 1.0)
            rule_ids[t] = ["sitting_posture"]
            continue

        # 5) 卧姿：复合证据（躯干 clearance + 0.5*头部高度）低于阈值
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

        # 其余落在阈值间区 → unknown
        labels[t] = _LABEL_UNKNOWN
        conf[t] = 0.0

    return {
        "labels": [str(x) for x in labels],
        "confidence": conf.astype(np.float32),
        "rule_ids": rule_ids,
        "body_scale": feat["scale"],
        "ground_height": feat["ground"],
    }


# -------------------------------------------------------------- 分段合并

def merge_segments(
    labels: list[str],
    confidence: np.ndarray,
    rule_ids: list[list[str]],
    fps: float,
    min_duration_s: float,
) -> list[dict]:
    """同标签连续帧成段；短于 min_duration 的段反复并入置信度更高的邻段。

    返回段列表保证时间轴无缝覆盖且每段时长 ≥ min_duration_s（单段 clip 除外）。
    """
    n = len(labels)
    segs: list[list] = []  # [start,end,label,conf_sum,rules_set]
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
        # 并入相邻两段中置信度更高的一侧
        left = segs[i_short - 1] if i_short > 0 else None
        right = segs[i_short + 1] if i_short + 1 < len(segs) else None
        if left is None:
            target = i_short + 1 if right is not None else i_short
            side = "right"
        elif right is None:
            target = i_short - 1
            side = "left"
        else:
            if left[3] >= right[3]:
                target, side = i_short - 1, "left"
            else:
                target, side = i_short + 1, "right"
        short = segs[i_short]
        host = segs[target]
        n_short = short[1] - short[0] + 1
        n_host = host[1] - host[0] + 1
        merged_conf = (host[3] * n_host + short[3] * n_short) / (n_host + n_short)
        new_start = min(host[0], short[0])
        new_end = max(host[1], short[1])
        merged = [new_start, new_end, host[2], merged_conf, host[4] | short[4]]
        lo, hi = sorted((i_short, target))
        segs = segs[:lo] + [merged] + segs[hi + 1:]

    out = []
    for a, b, lab, cf, rules in segs:
        out.append({
            "start_frame": int(a),
            "end_frame": int(b),
            "label": str(lab),
            "confidence": float(min(1.0, max(0.0, cf))),
            "rule_ids": sorted(rules),
        })
    return out


# -------------------------------------------------------------- 主入口

def generate_seeds(kp: np.ndarray, weight: np.ndarray, frame_idx: np.ndarray,
                   config: dict) -> dict:
    """单 clip 种子生成：帧级分类 → 分段合并 → 结构化输出。"""
    res = classify_frames(kp, weight, frame_idx, config)
    fps = float(config.get("nominal_fps", 30.0))
    min_dur = float(config.get("segment", {}).get("min_duration_s", 0.3))
    segments = merge_segments(res["labels"], res["confidence"], res["rule_ids"],
                              fps=fps, min_duration_s=min_dur)
    return {
        "frame_labels": np.array(res["labels"], dtype=object),
        "frame_confidence": res["confidence"],
        "frame_rule_ids": res["rule_ids"],
        "segments": segments,
        "body_scale": res["body_scale"],
        "ground_height": res["ground_height"],
    }
