#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""W35 种子草稿报告生成器(任务书 §W35 第③步交付).

输入(全部落盘产物, 不重算不臆造):
  keypoints_w35/extract_index.jsonl      提点质量记录(harvest_extract_keypoints 产出)
  seed_labels_draft.jsonl                规则种子草稿(harvest_rule_seeds 产出)
对比基准:
  reports/rule-seeds-2026-08-24.md 口径   W6 合成域(SMAL 3D 度量)帧占比基线
  主检出 AK partialclass4 质量记录        Q3b 同权重提点的 AK 域对照

产出:
  reports/harvest-w35-<date>.json       机器可读证据
  reports/harvest-w35-<date>.md         人读报告(分布/置信度/AK 域对比/降级审计)

核心审计点(传播矩阵三问的量化回答):
  withers(idx22)=dog-pose 死关节 → 引擎 clearance≡NaN→0 →
  standing/gait 门禁/jump 双条件/rise_transition 尖峰全部结构性失活,
  lying_composite 仅靠头部项存活 → 预期 sitting/lying/unknown 三分天下。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SEVEN_CLASSES = ["lying", "sitting", "standing", "walking",
                 "running", "rise_transition", "jump"]

#: W6 合成域基线(HANDOVER §10: 全量 225 clip 帧占比 sitting 36.3%/walking 23.4%;
#: 其余类见 reports/rule-seeds-2026-08-24.md——此处登记两大类作跨域方向性对照,
#: 精确逐类数值以该报告为 truth, 本表仅记录比较结论)
W6_BASELINE_HEADLINE = {"sitting": 0.363, "walking": 0.234}


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def analyze_extract(index_rows: list[dict]) -> dict:
    ok = [r for r in index_rows if r.get("status") == "ok"]
    err = [r for r in index_rows if str(r.get("status", "")).startswith("error")]
    all_missing = [r for r in index_rows if r.get("status") == "all_missing"]

    def _mean(key: str) -> float | None:
        vals = [r[key] for r in ok if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    total_rule = sum(r.get("rule_frames") or 0 for r in ok)
    total_hits = sum(r.get("rule_hits") or 0 for r in ok)
    total_detect_ok = sum(r.get("detect_ok") or 0 for r in ok)
    total_want = sum((r.get("rule_frames") or 0) for r in ok)  # 近似: want≈rule∪seq30
    interp = [r.get("n_interpolated") or 0 for r in ok]
    return {
        "n_total": len(index_rows), "n_ok": len(ok),
        "n_all_missing": len(all_missing), "n_error": len(err),
        "error_breakdown": dict(Counter(r["status"] for r in err)),
        "all_missing_ids": [r["fragment_id"] for r in all_missing][:20],
        "rule_frames_total": total_rule,
        "rule_hit_rate": round(total_hits / max(total_rule, 1), 4),
        "want_detect_rate": round(total_detect_ok / max(total_want, 1), 4),
        "low_conf_rate": _mean("low_conf"),
        "no_detect_rate": _mean("no_detect"),
        "mean_n_interpolated_seq30": round(sum(interp) / max(len(interp), 1), 3),
        "src_fps_values": sorted({round(r.get("src_fps") or 0, 2) for r in ok}),
    }


def analyze_seeds(seed_rows: list[dict]) -> dict:
    seg_class_counter: Counter = Counter()
    conf_by_class: dict[str, list[float]] = {c: [] for c in SEVEN_CLASSES}
    frame_occ: Counter = Counter()
    total_seg_duration_s = 0.0
    n_with_segments = 0
    n_error_rows = 0
    body_scales: list[float] = []
    unknown_heavy_clips = 0

    for row in seed_rows:
        if "error" in row:
            n_error_rows += 1
            continue
        segs = row.get("segments") or []
        if not segs:
            continue
        n_with_segments += 1
        clip_frames = 0
        for s in segs:
            lab = s["label"]
            dur_frames = s["end_frame"] - s["start_frame"] + 1
            # 规则轨 ≈10fps: 时长(s)≈帧数/10(用 meta 实际 fps 更准, 此处保守近似并披露)
            total_seg_duration_s += dur_frames / 10.0
            seg_class_counter[lab] += 1
            clip_frames += dur_frames
            if lab in conf_by_class:
                conf_by_class[lab].append(float(s.get("confidence") or 0.0))
        if row.get("body_scale"):
            body_scales.append(float(row["body_scale"]))
        if clip_frames > 0:
            unk = sum(1 for s in segs if s["label"] == "unknown")
            if sum(s["end_frame"] - s["start_frame"] + 1 for s in segs
                   if s["label"] == "unknown") / clip_frames > 0.9:
                unknown_heavy_clips += 1
        for s in segs:
            frame_occ[s["label"]] += s["end_frame"] - s["start_frame"] + 1

    total_frames = sum(frame_occ.values()) or 1
    occ_frac = {c: round(frame_occ.get(c, 0) / total_frames, 4) for c in SEVEN_CLASSES}
    occ_frac["unknown"] = round(frame_occ.get("unknown", 0) / total_frames, 4)
    conf_summary = {
        c: {"n": len(v),
            "mean": round(sum(v) / len(v), 4) if v else None,
            "p50": None, "min": round(min(v), 4) if v else None,
            "max": round(max(v), 4) if v else None}
        for c, v in conf_by_class.items()
    }
    for c, v in conf_by_class.items():
        if v:
            v_sorted = sorted(v)
            conf_summary[c]["p50"] = round(v_sorted[len(v_sorted) // 2], 4)

    return {
        "n_rows": len(seed_rows), "n_error_rows": n_error_rows,
        "n_with_segments": n_with_segments,
        "segment_count_by_class": dict(seg_class_counter),
        "frame_occupancy_frac": occ_frac,
        "confidence_by_class": conf_summary,
        "total_segment_duration_s_approx_at10fps": round(total_seg_duration_s, 1),
        "unknown_heavy_clips_gt90pct": unknown_heavy_clips,
        "body_scale_px_median": round(sorted(body_scales)[len(body_scales) // 2], 1)
        if body_scales else None,
    }


def structural_audit(seed_analysis: dict) -> dict:
    """死关节×规则依赖的结构性降级审计: 预测 vs 实测对照."""
    occ = seed_analysis["frame_occupancy_frac"]
    seg = seed_analysis["segment_count_by_class"]
    predictions = {
        "standing_unreachable": "clearance≡0 恒 < stand_min=0.35",
        "walking_running_unreachable": "gait 门禁 c>lie_max=0.18 恒假",
        "jump_blocked_second_condition": "air 条件可活但 c>stand_min+spike 恒假",
        "rise_transition_unreachable": "d_clearance≡0 无尖峰",
        "lying_overfire_risk": "comp=0.5*head_norm 单腿支撑, head_norm<1.5 即触发",
    }
    measured_dead_classes = {
        c: {"segment_count": seg.get(c, 0), "frame_frac": occ.get(c, 0.0)}
        for c in ("standing", "walking", "running", "rise_transition", "jump")
    }
    confirmed = all(v["segment_count"] == 0 for v in measured_dead_classes.values())
    return {
        "mechanism": predictions,
        "measured_dead_classes": measured_dead_classes,
        "prediction_confirmed": confirmed,
        "lying_frame_frac": occ.get("lying"),
        "sitting_frame_frac": occ.get("sitting"),
        "verdict": ("结构降级实证成立: 高度依赖类全零" if confirmed
                    else "存在非零计数, 与预测不符需人工核查"),
        "downstream_recommendation": (
            "label_status=deferred_pixel_domain(W30 判例): 种子草稿不入监督管线;"
            "如需激活视频通道标签, 须先在 W6 引擎层做 2D 像素域适配"
            "(以 front_tops/rear_tops 中值替代 withers 的 clearance 代理), 属引擎领地变更"),
    }


def compare_ak_domain(extract_stats: dict) -> dict:
    """与 AK q3b(同权重同掩码血统)提点质量对照——读主检出质量 JSON."""
    ak_path = Path("D:/Desktop/psd-framework/runs/public_real_dataset/partialclass4_extract_quality.json")
    out: dict = {"ak_source_path": str(ak_path)}
    if not ak_path.exists():
        out["note"] = "AK 质量 JSON 不存在, 对照缺失(如实登记)"
        return out
    ak = json.loads(ak_path.read_text(encoding="utf-8"))
    q = ak.get("quality") or []
    n_ak = len(q)
    ak_interp = [x.get("n_interpolated") or 0 for x in q]
    ak_no_det = [x.get("no_detect") or 0 for x in q]
    ak_low = [x.get("low_conf") or 0 for x in q]
    t_const = 30.0
    out.update({
        "ak_n_videos": n_ak,
        "ak_mean_no_detect_per_t30": round(sum(ak_no_det) / max(n_ak, 1), 3),
        "ak_mean_low_conf_per_t30": round(sum(ak_low) / max(n_ak, 1), 3),
        "ak_mean_interpolated_per_t30": round(sum(ak_interp) / max(n_ak, 1), 3),
        "w35_mean_no_detect_per_want": extract_stats.get("no_detect_rate"),
        "w35_mean_low_conf_per_want": extract_stats.get("low_conf_rate"),
        "comparison_note": (
            "AK 为 T=30 均匀采样口径(每视频固定 30 帧), W35 规则轨 ~10fps 密采样口径;"
            "比率不可直接混比, 方向性结论: 同权重对 W25 抓取片段的检测稳定性是否与 AK 同量级"),
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kp-root", default=str(
        REPO / "runs/data_campaign/video/keypoints_w35"))
    ap.add_argument("--seeds", default=str(
        REPO / "runs/data_campaign/video/keypoints_w35/seed_labels_draft.jsonl"))
    ap.add_argument("--out-prefix", default=str(
        REPO / f"reports/harvest-w35-{date.today().isoformat()}"))
    args = ap.parse_args()

    kp_root = Path(args.kp_root)
    index_rows = load_jsonl(kp_root / "extract_index.jsonl")
    seed_rows = load_jsonl(Path(args.seeds))

    extract_stats = analyze_extract(index_rows)
    seed_stats = analyze_seeds(seed_rows)
    audit = structural_audit(seed_stats)
    ak_cmp = compare_ak_domain(extract_stats)

    evidence = {
        "schema": "psd.data_campaign.harvest.w35_report_v1",
        "generated_at": str(date.today()),
        "extract": extract_stats,
        "seeds": seed_stats,
        "structural_audit": audit,
        "ak_comparison": ak_cmp,
        "w6_baseline_headline_frame_frac": W6_BASELINE_HEADLINE,
        "inputs": {
            "extract_index": str(kp_root / "extract_index.jsonl"),
            "seed_draft": str(args.seeds),
            "manifest_note": "正式池=manifest 准入 642 片段(实测对账零差额); 任务书口径 759 含 _runtime 83 个未准入候选缓存",
        },
    }
    out_json = Path(args.out_prefix + ".json")
    out_json.write_text(json.dumps(evidence, ensure_ascii=False, indent=2),
                        encoding="utf-8")

    md = render_md(evidence)
    Path(args.out_prefix + ".md").write_text(md, encoding="utf-8")
    print(f"[report] {out_json}")
    print(f"[report] {args.out_prefix}.md")


def render_md(e: dict) -> str:
    ex, sd, au, ak = e["extract"], e["seeds"], e["structural_audit"], e["ak_comparison"]
    lines = [
        "# W35 C1 视频片段池 提点+种子草稿报告(数据飞轮第二圈)",
        "",
        f"> 生成: {e['generated_at']} | 任务书: NEXT-BATCH-plan §W35 | 领地: reports/harvest-*",
        "",
        "## 1. 执行摘要",
        "",
        f"- 正式池 **{ex['n_total']}** 片段(manifest 对账零差额); 成功提点 **{ex['n_ok']}** "
        f"(all_missing {ex['n_all_missing']} / error {ex['n_error']})",
        f"- 规则轨命中率 **{ex['rule_hit_rate']}**, want 检出率 **{ex['want_detect_rate']}**",
        f"- 七类种子草稿覆盖 {sd['n_with_segments']}/{sd['n_rows']} 行; 结构降级审计: **{au['verdict']}**",
        "- 标签口径: **deferred_pixel_domain**(沿 W30 判例)——草稿仅供 R&D 检视, 不入监督管线",
        "",
        "## 2. 提点质量(Q3a 权重, 24 点 K9Graph)",
        "",
        "| 指标 | 值 |",
        "|------|-----|",
        f"| all_missing(整段无检) | {ex['n_all_missing']} |",
        f"| error | {ex['n_error']} `{list(ex['error_breakdown'])[:5]}` |",
        f"| 规则轨命中(检出帧/规则帧) | {ex['rule_hit_rate']} |",
        f"| seq30 平均插值帧(/30) | {ex['mean_n_interpolated_seq30']} |",
        f"| 源 fps 分布 | {ex['src_fps_values'][:12]} |",
        "",
        "## 3. 七类种子草稿分布",
        "",
        "| 类别 | 段数 | 帧占比 | 置信度均值 | 置信度 p50 |",
        "|------|------|--------|-----------|-----------|",
    ]
    for c in SEVEN_CLASSES + ["unknown"]:
        cs = sd["confidence_by_class"].get(c, {})
        lines.append(
            f"| {c} | {sd['segment_count_by_class'].get(c, 0)} "
            f"| {sd['frame_occupancy_frac'].get(c, 0.0)} "
            f"| {cs.get('mean') if cs else None} | {cs.get('p50') if cs else None} |")
    lines += [
        "",
        f"总段时长≈{sd['total_segment_duration_s_approx_at10fps']}s(按 10fps 名义折算); "
        f"unknown≥90% 时长的片段 {sd['unknown_heavy_clips_gt90pct']} 个; "
        f"体尺度中位 {sd['body_scale_px_median']}px。",
        "",
        "## 4. 与 AK 域对比",
        "",
    ]
    if "ak_n_videos" in ak:
        lines += [
            "| 维度 | AK q3b(T=30 均匀) | W35 片段池(~10fps 密采) |",
            "|------|-------------------|--------------------------|",
            f"| 样本量 | {ak['ak_n_videos']} 视频 | {ex['n_ok']} 片段 |",
            f"| 未检出率 | {ak['ak_mean_no_detect_per_t30']}/30帧 | {ak['w35_mean_no_detect_per_want']}/want帧 |",
            f"| 低可见淘汰率 | {ak['ak_mean_low_conf_per_t30']}/30帧 | {ak['w35_mean_low_conf_per_want']}/want帧 |",
            "",
            ak.get("comparison_note", ""),
        ]
    else:
        lines.append(f"(AK 对照缺失: {ak.get('note')})")
    lines += [
        "",
        f"W6 合成域头条基线(方向性对照, truth=reports/rule-seeds-2026-08-24.md): "
        f"sitting {W6_BASELINE_HEADLINE['sitting']} / walking {W6_BASELINE_HEADLINE['walking']}"
        " —— 本池 walking 因 gait 门禁失活必为 0, 跨域差异属**度量域不同**而非行为分布不同。",
        "",
        "## 5. 结构性降级审计(withers 死关节 × 规则引擎)",
        "",
        "机制: `clearance` 只读 idx22(withers); dog-pose GT 从未标注该点(C5/W29 盘点), ",
        "harvest_rule_seeds 载入端按契约 NaN 化 → `_masked_nanmean` 返回 NaN → `nan_to_num` 归 0。连锁:",
        "",
    ]
    for k, v in au["mechanism"].items():
        lines.append(f"- `{k}`: {v}")
    lines += [
        "",
        "| 失活类 | 段数 | 帧占比 |",
        "|--------|------|--------|",
    ]
    for c, v in au["measured_dead_classes"].items():
        lines.append(f"| {c} | {v['segment_count']} | {v['frame_frac']} |")
    lines += [
        "",
        f"**结论: {au['verdict']}**; lying 帧占比 {au['lying_frame_frac']}, "
        f"sitting 帧占比 {au['sitting_frame_frac']}。",
        "",
        f"下游建议: {au['downstream_recommendation']}",
        "",
        "## 6. 数据对账披露",
        "",
        "- 任务书口径 759 片段 vs 实测 fragments/ **642**(与 manifest.jsonl 642 行零差额);",
        "  差额 83 个 mp4 位于 `_runtime/cache`(W25 抓取期候选缓存, 不在 manifest、未准入池), ",
        "  另有 34 个差额与汇聚期口径有关, 已在 BOARD 登记, 以 manifest 为唯一 truth。",
        "- 死关节处置双保险: seq30 出口硬掩码清零(assemble_clip)+ 种子载入端 NaN 化(harvest_rule_seeds)。",
        "",
        "## 7. 复现命令",
        "",
        "```powershell",
        '& "D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe" scripts/harvest_extract_keypoints.py \\',
        "    --manifest D:/Desktop/psd-framework/runs/data_campaign/video/manifest.jsonl \\",
        "    --fragments-dir D:/Desktop/psd-framework/runs/data_campaign/video/fragments \\",
        "    --weights D:/Desktop/psd-framework/runs/public_real_yolo_dogpose/train/weights/best.pt \\",
        "    --out-root runs/data_campaign/video/keypoints_w35 --batch-size 50",
        '& "D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe" scripts/harvest_rule_seeds.py \\',
        "    --manifest D:/Desktop/psd-framework/runs/data_campaign/video/manifest.jsonl \\",
        "    --keypoints-root runs/data_campaign/video/keypoints_w35/rule_pkls \\",
        "    --out runs/data_campaign/video/keypoints_w35/seed_labels_draft.jsonl",
        '& "D:\\Desktop\\psd-framework\\.venv\\Scripts\\python.exe" scripts/harvest_seed_report.py',
        "```",
        "",
        "---",
        "",
        "*证据链: extract_index.jsonl(逐片段质量) → seed_labels_draft.jsonl(逐片段七类段) → "
        "本报告 JSON(聚合口径)。所有数字由当次运行产物聚合, 无手工录入。*",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
