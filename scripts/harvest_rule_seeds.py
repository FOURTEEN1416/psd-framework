"""C1/W25 片段池规则种子草稿生成入口（两段式标签策略的第二段）。

标签策略(任务书铁律): W6 规则引擎七类规则族**自产**种子草稿,
禁止外部标签直用(df_action.xlsx 教训已固化为 DATA-CAMPAIGN-plan §0 纪律 3)。

依赖链(会师路径): 本窗片段池 --[Q3a 犬类 pose 权重提点]--> 24 点骨架序列
                 --[本脚本: 机械复用 psd/data/rule_seeds.py]--> 七类种子草稿

Q3a/Q3b 属 GPU 接力队列(relay_executor), 本脚本全程 CPU 且**不触发提点**;
提点产物未到位时 fail-fast 并如实报告, 绝不伪造标签。

预期输入目录布局(Q3b 风格, 由提点执行方产出):
  <keypoints_root>/<fragment_id>.pkl   每文件含:
    kp_world (T,24,3) float / kp_weight (T,24) float / frame_idx (T,) int

用法:
  python scripts/harvest_rule_seeds.py \
      --manifest runs/data_campaign/video/manifest.jsonl \
      --keypoints-root runs/data_campaign/video/keypoints_q3b \
      --config configs/rule_seeds_w6.yaml        # 若无则用引擎默认阈值
输出:
  runs/data_campaign/video/seed_labels_draft.jsonl  每行 {fragment_id, segments[], meta}
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from psd.data.rule_seeds import generate_seeds  # noqa: E402 — 机械复用 W6 owner

DEFAULT_CONFIG = {
    "nominal_fps": 30.0,
    "posture": {},
    "speed": {"smooth_window": 5},
    "transition": {"rate_min": 1.5, "window": 5},
    "jump": {"min_air_clearance": 0.25, "spike_over_standing": 0.15},
    "segment": {"min_duration_s": 0.3},
}


def load_keypoint_pkl(path: Path) -> dict:
    with path.open("rb") as f:
        d = pickle.load(f)
    for k, shape_hint in (("kp_world", 3), ("kp_weight", 2), ("frame_idx", 1)):
        if k not in d:
            raise KeyError(f"{path.name} 缺字段 {k}")
    # 死关节 → NaN：dog-pose GT 从未标注 idx20-23（C5/W29 盘点证据），
    # 提点端已硬掩码为零；此处转 NaN 交给 W6 引擎原生 valid-mask 语义，
    # 防止零值被体高/躯干等规则误读为真实坐标（withers=idx22 是根关节）。
    import numpy as np
    from psd.data.ak_pose_extract import DEAD_JOINTS
    kp = np.asarray(d["kp_world"], dtype=np.float32).copy()
    kp[:, list(DEAD_JOINTS), :] = np.nan
    d["kp_world"] = kp
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--keypoints-root", required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--out", default=str(
        REPO / "runs/data_campaign/video/seed_labels_draft.jsonl"))
    args = ap.parse_args()

    man_rows = [json.loads(l) for l in Path(args.manifest).read_text(encoding="utf-8").splitlines() if l.strip()]
    kp_root = Path(args.keypoints_root)
    cfg = dict(DEFAULT_CONFIG)
    if args.config and Path(args.config).exists():
        import yaml
        cfg.update(yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))

    missing = [r["fragment_id"] for r in man_rows
               if not (kp_root / f"{r['fragment_id']}.pkl").exists()]
    if len(missing) == len(man_rows):
        print(f"[rule-seeds] FAIL-FAST: 提点产物 0/{len(man_rows)} 到位于 {kp_root}")
        print("[rule-seeds] 原因: Q3a(犬类 pose 权重)/Q3b(全量提点)属 GPU 接力队列, 尚未完成。")
        print("[rule-seeds] 处置: 标签保持 label_status=rule_seed_pending; 权重到位后重跑本脚本即得七类种子草稿。")
        sys.exit(2)

    out_rows, skipped = [], 0
    for r in man_rows:
        pkl = kp_root / f"{r['fragment_id']}.pkl"
        if not pkl.exists():
            skipped += 1
            continue
        try:
            d = load_keypoint_pkl(pkl)
            res = generate_seeds(d["kp_world"], d["kp_weight"], d["frame_idx"], cfg)
            out_rows.append({
                "fragment_id": r["fragment_id"],
                "label_strategy": "w6_rule_engine_self_produced",
                "classes_family": ["lying", "sitting", "standing", "walking",
                                   "running", "rise_transition", "jump"],
                "segments": res["segments"],
                "body_scale": res["body_scale"],
                "ground_height": res["ground_height"],
                "note": "draft seeds for pseudo-label pipeline; NOT ground truth",
            })
        except Exception as e:  # noqa: BLE001 — 单片段失败不阻塞
            out_rows.append({"fragment_id": r["fragment_id"], "error": str(e)[:300]})
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_ok = sum(1 for x in out_rows if "error" not in x)
    print(f"[rule-seeds] 完成: 成功 {n_ok} / 跳过(缺提点) {skipped} -> {outp}")


if __name__ == "__main__":
    main()
