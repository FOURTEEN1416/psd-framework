"""AK partialclass4 172 clips — 2D 规则种子校准与效果门禁（W47）。

协议要点（先于任何实验运行的预注册声明）:
1. 数据: D:/Desktop/psd-framework/runs/public_real_dataset/partialclass4_T30.pkl
   （Q3b 全量产物, 172 clips = train 123 / val 49, gate4 有标, 只读消费）
2. y 约定: AK 提点为图像域坐标(y 向下), 统一 y_axis="down" 声明。
3. 帧率陷阱(round2 §4 已警示): T30 均匀抽样真实 fps 未知 → 本实验统一
   nominal_fps=1.0, 速度单位 = 体尺度/帧。校准与推理同单位, 内部自洽;
   与 3D 秒制阈值的数值不可直接比较(量纲不同, 如实声明)。
4. 校准/评估隔离: 阈值只在 train(123) 上拟合; val(49) 仅用于效果门禁,
   禁止回流调参。posture 类阈值无 gate4 监督对应 → 承 3D 先验不校准
   (NOT_CALIBRATED 如实登记)。
5. 效果门禁(本常量即预注册, 修改须留痕):
   - 主口径 = val 可判别子集(track/jump/stay)的 clip 级一致率
   - >=0.60 → 达标, 入监督管线(conf 加权弱标签)
   - 0.40~0.60 → 未达标, 如实降级为预训练池(保留校准证据)
   - <0.40 → 降级, 且 RULE_TO_GATE4 映射协议登记存疑
   参照系: 可判别子集均匀随机 ≈ 0.33; val 子集 majority(track) ≈ 0.50。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from psd.data.rule_seeds_2d import (
    DEFAULT_CONFIG_2D,
    RULE_TO_GATE4,
    classify_frames_2d,
    vote_clip_gate4,
)

PKL_PATH = Path("D:/Desktop/psd-framework/runs/public_real_dataset/partialclass4_T30.pkl")
OUT_JSON = Path("reports/rule-seeds-2d-calibration-w47.json")

GATE4_BY_LABEL = {"watch": 72, "track": 46, "stay": 27, "jump": 27}  # 文档参照, 非逻辑

# ---- 预注册门禁(勿改; 改动必须在报告中登记修订理由与时间) ----
PREREGISTERED_GATE = {
    "metric": "clip_level_agreement_on_discernable_subset_val",
    "discernable_classes": ["track", "jump", "stay"],
    "promote_threshold": 0.60,
    "demote_keep_evidence_band": [0.40, 0.60],
    "verdicts": {
        "promote": "supervision_pipeline",
        "band": "pretrain_pool_keep_evidence",
        "below": "pretrain_pool_mapping_suspect",
    },
}


def load_clips() -> list[dict]:
    import pickle
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    assert len(data) == 172, f"预期 172 clips, 实得 {len(data)}(冒烟残留防御)"
    for d in data:
        assert d["keypoints"].shape == (30, 24, 3), f"形状契约破坏: {d['video_id']}"
        assert d["psd_class"] in ("watch", "track", "stay", "jump"), "未知 gate4 类"
    return data


def make_config(speed_walk: float, speed_run: float, air_min: float,
                rate_min: float | None = None) -> dict:
    cfg = {
        "nominal_fps": 1.0,  # 体尺度/帧 单位(见模块 docstring 第 3 条)
        "posture": dict(DEFAULT_CONFIG_2D["posture"]),   # NOT_CALIBRATED, 承 3D 先验
        "speed": {"walk_min": speed_walk, "run_min": speed_run,
                  "smooth_window": DEFAULT_CONFIG_2D["speed"]["smooth_window"]},
        "transition": dict(DEFAULT_CONFIG_2D["transition"]),
        "jump": dict(DEFAULT_CONFIG_2D["jump"]),
        "segment": dict(DEFAULT_CONFIG_2D["segment"]),
    }
    if rate_min is not None:
        cfg["transition"]["rate_min"] = rate_min
    return cfg


def run_frames(clips: list[dict], cfg: dict) -> list[dict]:
    """对每个 clip 跑 2D 规则分类, 收集帧标签/置信度/质心速度证据。"""
    out = []
    for d in clips:
        kp = d["keypoints"][:, :, :2]
        w = d["keypoints"][:, :, 2]
        res = classify_frames_2d(kp.astype(np.float64), w.astype(np.float64),
                                 np.arange(len(kp)), cfg, y_axis="down")
        out.append({**d, "res": res})
    return out


def vote_set(runs: list[dict], min_conf: float = 0.0) -> list[str]:
    """clip 级投票(conf>=min_conf 的帧参与; 0=全帧)。"""
    preds = []
    for r in runs:
        if min_conf > 0:
            kept = [lab for lab, c in zip(r["res"]["labels"], r["res"]["confidence"])
                    if c >= min_conf]
        else:
            kept = list(r["res"]["labels"])
        preds.append(vote_clip_gate4(kept))
    return preds


def agreement_stats(runs: list[dict], subset: list[str] | None,
                    min_conf: float = 0.0) -> dict:
    """一致率统计。subset=None 用全部, 否则过滤真值类(如 watch 不可判别子集)。"""
    sel = runs if subset is None else [r for r in runs if r["psd_class"] in subset]
    preds = vote_set(sel, min_conf=min_conf)
    n = len(sel)
    correct = abstain = wrong = 0
    confusion: Counter = Counter()
    per_class: dict[str, Counter] = {}
    for r, p in zip(sel, preds):
        gt = r["psd_class"]
        if p == "abstain":
            abstain += 1
        elif p == gt:
            correct += 1
        else:
            wrong += 1
            confusion[f"{gt}->{p}"] += 1
        per_class.setdefault(gt, Counter())[p] += 1
    return {
        "n": n,
        "correct": correct,
        "wrong": wrong,
        "abstain": abstain,
        "agreement_on_all": round(correct / n, 4) if n else None,
        "agreement_excl_abstain": round(correct / max(correct + wrong, 1), 4),
        "abstain_rate": round(abstain / n, 4) if n else None,
        "confusion_top": dict(confusion.most_common(12)),
        "per_class_pred": {k: dict(v) for k, v in sorted(per_class.items())},
    }


def feature_distribution_evidence(clips: list[dict]) -> dict:
    """按真值 gate4 分组的原始运动特征分布(阈值校准的证据底座)。"""
    ev: dict[str, dict] = {}
    for cls in ("stay", "track", "jump", "watch"):
        speeds, airs = [], []
        for d in clips:
            if d["psd_class"] != cls:
                continue
            kp = d["keypoints"][:, :, :2].astype(np.float64)
            w = d["keypoints"][:, :, 2].astype(np.float64)
            from psd.data.rule_seeds_2d import normalize_y_orientation, compute_frame_features_2d
            feat = compute_frame_features_2d(normalize_y_orientation(kp, "down"),
                                             w, np.arange(len(kp)),
                                             make_config(0, 0, 0))
            ok = feat["frame_ok"]
            speeds.extend(feat["centroid_speed"][ok].tolist())
            airs.extend(feat["paw_air"][ok].tolist())
        q = lambda a, p: round(float(np.quantile(a, p)), 4) if a else None  # noqa: E731
        ev[cls] = {
            "n_frames": len(speeds),
            "centroid_speed_p10_p50_p90": [q(speeds, .1), q(speeds, .5), q(speeds, .9)],
            "paw_air_p50_p90_p99": [q(airs, .5), q(airs, .9), q(airs, .99)],
        }
    return ev


def calibrate_on_train(train_runs: list[dict]) -> tuple[dict, dict]:
    """train-only 阈值拟合(体尺度/帧 单位)。

    walk_min := track 帧速 p10 与 stay 帧速 p90 的中点(可分性下界);
    run_min  := track 帧速 p90(高速尾起点, 保守);
    jump_air_min := jump 帧 paw_air p50 与非跳 p99 之间取 p50 下探保护…
    实际取 max(jump p25, 非跳帧 p99) 保证特异性。证据全部落 JSON。
    """
    sp_by_cls: dict[str, list[float]] = {"stay": [], "track": [], "jump": [], "watch": []}
    air_jump: list[float] = []
    air_other: list[float] = []
    for r in train_runs:
        ok = r["res_frame_feat"]["frame_ok"]
        sp = r["res_frame_feat"]["centroid_speed"][ok]
        pa = r["res_frame_feat"]["paw_air"][ok]
        sp_by_cls[r["psd_class"]].extend(sp.tolist())
        (air_jump if r["psd_class"] == "jump" else air_other).extend(pa.tolist())

    def pct(a, p):
        return float(np.quantile(a, p)) if a else float("nan")

    walk_min = (pct(sp_by_cls["track"], .10) + pct(sp_by_cls["stay"], .90)) / 2.0
    run_min = pct(sp_by_cls["track"], .90)
    jump_air = max(pct(air_jump, .25), pct(air_other, .99))
    evidence = {
        "unit": "body_scale_per_frame(nominal_fps=1)",
        "speed_quantiles": {k: {"p10": round(pct(v, .1), 4), "p50": round(pct(v, .5), 4),
                                "p90": round(pct(v, .9), 4)}
                            for k, v in sp_by_cls.items()},
        "paw_air": {"jump_p25": round(pct(air_jump, .25), 4),
                    "nonjump_p99": round(pct(air_other, .99), 4)},
        "calibrated": {"walk_min": round(walk_min, 4), "run_min": round(run_min, 4),
                       "min_air_clearance": round(jump_air, 4)},
        "notes": [
            "walk_min=(track p10 + stay p90)/2 — 两行为速度分布的可分性中点",
            "run_min=track p90 — 高速尾保守起点",
            "jump_air=max(jump p25, non-jump p99) — 特异性优先",
            "posture 族阈值无 gate4 监督对应, NOT_CALIBRATED 承 3D 先验",
        ],
    }
    cfg = make_config(evidence["calibrated"]["walk_min"],
                      evidence["calibrated"]["run_min"],
                      evidence["calibrated"]["min_air_clearance"])
    return cfg, evidence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    clips = load_clips()
    train = [d for d in clips if d["split"] == "train"]
    val = [d for d in clips if d["split"] == "val"]
    print(f"[load] train={len(train)} val={len(val)}")

    result: dict = {
        "protocol_doc": __doc__.strip().splitlines()[0],
        "pkl": str(PKL_PATH),
        "gate": PREREGISTERED_GATE,
        "n_train": len(train), "n_val": len(val),
        "class_dist": {"train": dict(Counter(d["psd_class"] for d in train)),
                       "val": dict(Counter(d["psd_class"] for d in val))},
    }

    # ---- 特征分布证据(阈值校准底座, train) ----
    result["feature_evidence_train"] = feature_distribution_evidence(train)

    # ---- Step A: 默认阈值基线(3D 先验直迁, 秒制单位错配如实呈现) ----
    if not args.skip_baseline:
        base_cfg = dict(DEFAULT_CONFIG_2D)  # nominal_fps=30 → T30 抽样下速度被系统性低估
        base_runs = run_frames(train, base_cfg)
        result["baseline_default3d_train_discernable"] = agreement_stats(
            base_runs, subset=["track", "jump", "stay"])

    # ---- Step B: train 校准 ----
    from psd.data.rule_seeds_2d import compute_frame_features_2d, normalize_y_orientation
    for d in train:
        kp = d["keypoints"][:, :, :2].astype(np.float64)
        w = d["keypoints"][:, :, 2].astype(np.float64)
        d["res_frame_feat"] = compute_frame_features_2d(
            normalize_y_orientation(kp, "down"), w, np.arange(len(kp)),
            make_config(0.0, 999.0, 0.0))  # 阈值放开, 只要特征分布
    cal_cfg, cal_ev = calibrate_on_train(train)
    result["calibration"] = cal_ev
    print("[calibrate]", cal_ev["calibrated"])

    # ---- Step C: 校准后 train 自检 + val 门禁 ----
    train_cal_runs = run_frames(train, cal_cfg)
    result["calibrated_train_discernable"] = agreement_stats(
        train_cal_runs, subset=["track", "jump", "stay"])
    val_cal_runs = run_frames(val, cal_cfg)
    result["calibrated_val_discernable"] = agreement_stats(
        val_cal_runs, subset=["track", "jump", "stay"])
    result["calibrated_val_full_incl_watch"] = agreement_stats(val_cal_runs, subset=None)
    result["calibrated_val_conf08"] = agreement_stats(
        val_cal_runs, subset=["track", "jump", "stay"], min_conf=0.8)

    # ---- Step D: 预注册门禁判定 ----
    main_acc = result["calibrated_val_discernable"]["agreement_on_all"]
    g = PREREGISTERED_GATE
    if main_acc is None:
        verdict = "invalid"
    elif main_acc >= g["promote_threshold"]:
        verdict = g["verdicts"]["promote"]
    elif main_acc >= g["demote_keep_evidence_band"][0]:
        verdict = g["verdicts"]["band"]
    else:
        verdict = g["verdicts"]["below"]
    result["VERDICT"] = {
        "metric_value": main_acc,
        "threshold": g["promote_threshold"],
        "verdict": verdict,
        "rule": f">={g['promote_threshold']} promote; "
                f"{g['demote_keep_evidence_band'][0]}~{g['promote_threshold']} band; "
                f"<{g['demote_keep_evidence_band'][0]} below",
    }
    print(f"[VERDICT] discernable-val agreement={main_acc} → {verdict}")

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {OUT_JSON}")


if __name__ == "__main__":
    main()
