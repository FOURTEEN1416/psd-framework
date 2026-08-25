#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""gen_number_index.py — 论文终稿回填数字索引生成器（W32 任务书范围 1）

机械化降耗目标：让最终回填窗口退化为"机械替换"。
本脚本扫描三类输入，产出 {占位符 → 数值 → 来源文件+字段} 三列机器可读索引：

  1. dev-docs/paper-backfill-quickref.md —— 实验数字总表（单一真相，三层口径分离）
  2. docs/paper/*.md                     —— [RESULT-x] / [PENDING] / [CITATION-NEEDED] 占位符清单
  3. reports/*.json                      —— 程序化读取关键字段路径，验证 quickref 声称值

输出: reports/number-index-<日期>.json（默认当日）
      含 mismatches 节——凡 quickref 声称值与 JSON 现值不一致者显式暴露，禁止静默择一。

用法:
    python scripts/gen_number_index.py                    # 生成 reports/number-index-<今日>.json
    python scripts/gen_number_index.py --date 2026-08-25  # 指定日期
    python scripts/gen_number_index.py --check            # 只验证不写盘（CI 冒烟用）

设计约束（AGENTS.md 硬规则 3/4）:
  - 每条索引带 layer（合成/公开真实/真实K9），禁止跨层混排
  - value_json 为当次运行程序化读取的新鲜值；value_quickref 为表格声称值
  - verified=true 要求两者在容差内一致；不一致进 mismatches 并保留双值待裁决
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_DEFAULT = Path(__file__).resolve().parent.parent
QUICKREF = Path("dev-docs/paper-backfill-quickref.md")
PAPER_DIR = Path("docs/paper")
REPORTS_DIR = Path("reports")

# ---------------------------------------------------------------------------
# 确定性字段映射表：实验ID → (JSON 相对路径, 字段路径, 期望语义说明)
# 字段路径用 dot.notation；列表索引用 [n]。手工维护并注释依据，不做自动发现——
# 已知实验→已知字段的确定性映射比模糊扫描更可靠，且便于 review。
# ---------------------------------------------------------------------------
FIELD_MAP: dict[str, list[dict[str, str]]] = {
    # ── 合成层 ──
    "E1": [
        {
            "file": "reports/p05-stgcnbc-synthetic-100perclass-Y.json",
            "field": "summary/best_val_acc",
            "metric": "val_acc (22类)",
            "layer": "合成层",
            "note": "⚠️ 当前 JSON 为等预算 50ep 覆写版(early_stopping=false)；quickref 另记首跑 97.3% / 早停复核 96.4%，三种口径并存，引用须注明协议",
        }
    ],
    "E2": [
        {
            "file": "reports/p05-stgcnbc-synthetic-100perclass-Yprime.json",
            "field": "summary/best_val_acc",
            "metric": "val_acc (21类 Y')",
            "layer": "合成层",
            "note": "等预算 50ep 版为当前报告内容；早停版数字仅存 quickref 行内",
        }
    ],
    "E3": [
        {
            "file": "reports/p05-stgcnbc-synthetic-50perclass-Y.json",
            "field": "summary/best_val_acc",
            "metric": "val_acc (22类, 50样本/类)",
            "layer": "合成层",
            "note": "(消融)",
        }
    ],
    "E4": [
        {
            "file": "reports/p05-stgcnbc-synthetic-20perclass-Y.json",
            "field": "summary/best_val_acc",
            "metric": "val_acc (22类, 20样本/类)",
            "layer": "合成层",
            "note": "(消融)",
        }
    ],
    "E-C": [
        {
            "file": "reports/p02-smq-iou-eC-seeds-recheck.json",
            "field": "aggregate/mean_matched_iou",
            "metric": "mean_matched_iou (种子伪GT)",
            "layer": "公开真实层",
            "companion_fields": "aggregate/std; aggregate/boundary_f1_mean; aggregate/n_episodes",
            "note": "recheck 与首跑逐位一致；随机基线 ~0.30 为估计值非实测",
        },
        {
            "file": "reports/p02-smq-iou-eC-seeds-recheck.json",
            "field": "aggregate/boundary_f1_mean",
            "metric": "boundary_F1@16",
            "layer": "公开真实层",
            "compare_quickref": "false",
            "note": "(伴随指标；quickref 主值列为 IoU，不参与本条对比)",
        },
    ],
    "E-A": [
        {
            "file": "reports/p02-smq-iou-eA-seeds.json",
            "field": "aggregate/mean_matched_iou",
            "metric": "mean_matched_iou (mse=1.0 修复基线)",
            "layer": "公开真实层",
            "note": "(历史基线)",
        },
    ],
    # ── 公开真实层 ──
    "P0.1": [
        {
            "file": "reports/p01-knn-result.json",
            "field": "knn_top1_mean_pct",
            "metric": "kNN(k=1) top-1 (%)",
            "layer": "公开真实层",
            "companion_fields": "random_baseline_pct; ratio_vs_random; fold_top1_acc",
            "note": "dog-ID 代理 probe 口径披露随行",
        },
    ],
    "P0.3-main": [
        {
            "file": "reports/p03-jia-phasea-results.json",
            "field": "e3_main/alpha_1.0/purity_mean",
            "metric": "原型聚类 purity",
            "layer": "公开真实层",
            "companion_fields": "e3_main/alpha_1.0/random_baseline_purity; e3_main/alpha_1.0/purity_std",
            "note": "3 run_seeds 同值；随机基线 Σπ²=0.3306",
        },
    ],
    "P0.3-noise30": [
        {
            "file": "reports/p03-jia-phasea-results.json",
            "field": "noise_ablation[3]/purity_mean",
            "metric": "purity@30% 标签噪声",
            "layer": "公开真实层",
            "compare_quickref": "false",
            "note": "noise_ablation 为按 noise_rate 升序的列表，[3] 即 q=30%；quickref 该行为主行伴生（P0.3 同 ID），不参与主值对比",
        },
    ],
    "P0.4-main": [
        {
            "file": "reports/p04-tcl-results.json",
            "field": "cells/on_consensus_a1.0/rounds_agg[1]/precision_mean",
            "metric": "pool_precision (r1 操作点, cov≈35%)",
            "layer": "公开真实层",
            "companion_fields": "cells/on_consensus_a1.0/rounds_agg[1]/precision_std; cells/on_consensus_a1.0/paired_first_vs_final/delta_pp_mean",
            "note": "quickref 主数值 0.691 取 r1_selftrain 峰值轮",
        },
    ],
}

# 补充数值发现层（无 quickref 行但论文引用需要的报告数字）：
# RESULT-3 三候选 + warm-start 换轨 + AL 负结果探索性发现
EXTRA_SOURCES: list[dict[str, str]] = [
    {
        "tag": "RESULT-3-cand-A-smq-ratio",
        "file": "reports/p02-smq-iou-eC-seeds-recheck.json",
        "field": "aggregate/mean_matched_iou",
        "derive": "ratio_vs_est_random_0.30",
        "desc": "[RESULT-3] 候选A：SMQ 分割 IoU 对估计随机基线的倍率 (~1.53×)",
        "layer": "公开真实层",
    },
    {
        "tag": "RESULT-3-cand-B-pseudo-label-peak-delta",
        "file": "reports/p04-tcl-results.json",
        "field": "cells/on_consensus_a1.0/rounds_agg",
        "derive": "peak_delta_pp(r0_prototype→r1_selftrain)",
        "desc": "[RESULT-3] 候选B：伪标签迭代 r0→r1 池精度峰值增量 (+17.88pp)",
        "layer": "公开真实层",
    },
    {
        "tag": "RESULT-3-cand-B-conservative-paired",
        "file": "reports/p04-tcl-results.json",
        "field": "cells/on_consensus_a1.0/paired_first_vs_final/delta_pp_mean",
        "derive": "as-is",
        "desc": "[RESULT-3] 候选B保守口径：首末配对检验 Δpp (+10.69±3.28, p=0.030)——摘要引用建议优先此口径防 cherry-picking 指控",
        "layer": "公开真实层",
    },
    {
        "tag": "RESULT-3-cand-C-wallclock-ratio",
        "file": "reports/c1-decouple-cost-2026-08-24.json",
        "field": "aggregated",
        "derive": "wall_clock_ratio(baseline/decouple)",
        "desc": "[RESULT-3] 候选C：解耦切换墙钟比 (实测 ~7.32×, 论文措辞保守区间 ≥3×)",
        "layer": "合成层",
    },
    {
        "tag": "C7-warmstart-b20-vs-coldstart",
        "file": "reports/p05-al-efficiency-warmstart-short-2026-08-25.json",
        "field": "curves",
        "derive": "protocol_level_b20_mean_both_arms",
        "desc": "C7 换轨主张：warm-start 协议层 b=20 两臂共享初始核 acc (82.0%) vs W14 冷启动同预算 (~7.8%)",
        "layer": "合成层",
    },
    {
        "tag": "C7-warmstart-b200-ceiling-random-arm",
        "file": "reports/p05-al-efficiency-warmstart-short-2026-08-25.json",
        "field": "curves/random/200/mean",
        "derive": "as-is",
        "desc": "C7 换轨 b=200 天花板 95.7%——注意归属为 random(均匀扩展)臂，entropy 臂同点为 91.4%",
        "layer": "合成层",
    },
    {
        "tag": "C7-exploratory-negative-w14-gap",
        "file": "reports/p05-al-efficiency-short-2026-08-24.json",
        "field": "curves",
        "derive": "random_minus_entropy_pp_at_b100_b200",
        "desc": "AL 效率负结果（§5 探索性发现）：随机反超熵 +7.9pp(b=100)/+7.1pp(b=200)，3/3 seeds 同向",
        "layer": "合成层",
    },
]

TOL_REL = 0.01  # 数值一致性相对容差（四舍五入级别差异不算 mismatch）


def _walk(obj: Any, path: str) -> Any:
    """按 'a/b[0]/c' 路径取值（'/' 分隔，键名可自然含点）；不存在抛 KeyError/IndexError。"""
    cur = obj
    for part in re.split(r"/", path):
        m = re.fullmatch(r"([\w.\-]+)\[(\d+)\]", part)
        if m:
            cur = cur[m.group(1)][int(m.group(2))]
        else:
            cur = cur[part]
    return cur


def parse_quickref(repo: Path) -> list[dict[str, str]]:
    """解析 quickref 两大表格（合成层/公开真实层）的行为字典列表。"""
    text = (repo / QUICKREF).read_text(encoding="utf-8")
    rows: list[dict[str, str]] = []
    layer = "?"
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("### 合成层"):
            layer = "合成层"
        elif s.startswith("### 公开真实层"):
            layer = "公开真实层"
        if not (s.startswith("|") and not s.startswith("|--") and not s.startswith("| 实验")):
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if len(cells) < 7 or cells[0] in ("实验 ID", "---", ""):
            continue
        rows.append(
            {
                "exp_id": cells[0],
                "setup": cells[1],
                "metric": cells[2],
                "value_quickref": cells[3],
                "random_baseline": cells[4],
                "ratio": cells[5],
                "source": cells[6].strip("`"),
                "skeleton_note": cells[7] if len(cells) > 7 else "",
                "layer": layer,
            }
        )
    return rows


def scan_placeholders(repo: Path) -> list[dict[str, str]]:
    """扫描 docs/paper/*.md 中的占位符标记，记录位置与上下文片段。"""
    pats = {
        "RESULT": re.compile(r"\[RESULT-(\d+)[^\]]*\]"),
        "PENDING": re.compile(r"\[PENDING[^\]]*\]"),
        "CITATION": re.compile(r"\[CITATION-NEEDED[^\]]*\]"),
    }
    found: list[dict[str, str]] = []
    for md in sorted((repo / PAPER_DIR).glob("*.md")):
        for lineno, line in enumerate(md.read_text(encoding="utf-8").splitlines(), 1):
            for kind, pat in pats.items():
                for m in pat.finditer(line):
                    snippet = line.strip()
                    if len(snippet) > 200:
                        start = max(0, m.start() - 60)
                        snippet = "…" + line[start : m.end() + 60].strip() + "…"
                    found.append(
                        {
                            "kind": kind,
                            "token": m.group(0),
                            "file": str(PAPER_DIR / md.name),
                            "line": str(lineno),
                            "snippet": snippet,
                        }
                    )
    return found


def read_json_value(repo: Path, rel_file: str, field: str) -> tuple[Any, str]:
    """读取 JSON 字段值；返回 (value|None, status)。"""
    fp = repo / rel_file
    if not fp.exists():
        return None, "missing-file"
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return _walk(data, field), "ok"
    except (KeyError, IndexError, TypeError):
        return None, "field-not-found"
    except json.JSONDecodeError:
        return None, "json-parse-error"


def _num(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_entries(repo: Path, quickref_rows: list[dict[str, str]]) -> tuple[list[dict], list[dict]]:
    """主索引构建：quickref 行 × FIELD_MAP 交叉，程序化读 JSON 验证。"""
    entries: list[dict] = []
    mismatches: list[dict] = []

    def _emit(exp_id: str, spec: dict, qr: dict | None) -> None:
        val, status = read_json_value(repo, spec["file"], spec["field"])
        entry = {
            "experiment_id": exp_id,
            "placeholder_hint": f"[{exp_id}]",
            "layer": spec.get("layer") or (qr or {}).get("layer", "?"),
            "metric": spec["metric"],
            "value_json": val,
            "json_status": status,
            "source_file": spec["file"],
            "source_field": spec["field"],
            "companion_fields": spec.get("companion_fields", ""),
            "value_quickref": (qr or {}).get("value_quickref", ""),
            "random_baseline_quickref": (qr or {}).get("random_baseline", ""),
            "ratio_quickref": (qr or {}).get("ratio", ""),
            "verified": None,
            "note": spec.get("note", ""),
        }
        jv = _num(val)
        qv = None
        compare = spec.get("compare_quickref", "true") != "false"
        if qr and qr.get("value_quickref"):
            m = re.search(r"-?\d+(?:\.\d+)?", qr["value_quickref"].split("±")[0])
            qv = float(m.group(0)) if m else None
        # 单位自动归一：一方为 [0,1] 小数、另一方为百分数量级时，对 JSON 值换算
        # （本表无合法条目跨两个数量级仍应判不一致，启发式安全）
        if compare and jv is not None and qv is not None and jv != qv:
            if abs(jv) <= 1.001 < abs(qv):       # JSON 小数 vs quickref 百分数 → ×100
                jv *= 100.0
            elif abs(qv) <= 1.001 < abs(jv):     # JSON 百分数 vs quickref 小数 → ÷100
                jv /= 100.0
        if compare and jv is not None and qv is not None:
            tol = TOL_REL * max(abs(jv), abs(qv), 1e-9)
            entry["verified"] = abs(jv - qv) <= tol
            if not entry["verified"]:
                mismatches.append(
                    {
                        "experiment_id": exp_id,
                        "field": spec["field"],
                        "value_json": jv,
                        "value_quickref": qr["value_quickref"],
                        "hint": "quickref 声称值与 JSON 现值不一致——终稿窗口须先裁决口径再回填，禁止静默择一",
                    }
                )
        entries.append(entry)

    qr_by_id: dict[str, dict[str, str]] = {}
    for r in quickref_rows:  # first-wins：同名 ID（如 P0.3 主行+伴生行）取表格首行为主口径
        qr_by_id.setdefault(r["exp_id"], r)
    covered_base_ids: set[str] = set()
    for exp_id, specs in FIELD_MAP.items():
        base_id = exp_id.split("-main")[0].split("-noise")[0]
        qr = qr_by_id.get(base_id)
        covered_base_ids.add(base_id)
        for spec in specs:
            _emit(exp_id, spec, qr)
    # quickref 中存在但 FIELD_MAP 未覆盖的行 → 如实登记（不静默丢弃）
    for r in quickref_rows:
        if r["exp_id"] in covered_base_ids:
            continue
        entries.append(
            {
                "experiment_id": r["exp_id"],
                "placeholder_hint": f"[{r['exp_id']}]",
                "layer": r["layer"],
                "metric": r["metric"],
                "value_json": None,
                "json_status": "no-fieldmap",
                "source_file": r["source"],
                "source_field": "",
                "companion_fields": "",
                "value_quickref": r["value_quickref"],
                "random_baseline_quickref": r["random_baseline"],
                "ratio_quickref": r["ratio"],
                "verified": None,
                "note": "quickref 有行但未建字段映射（伴生行）；skeleton_note: "
                + r["skeleton_note"][:80],
            }
        )
    return entries, mismatches


def build_extras(repo: Path) -> list[dict]:
    """补充数值发现：RESULT-3 候选 / C7 换轨 / AL 负结果。"""
    out: list[dict] = []
    for spec in EXTRA_SOURCES:
        raw, status = read_json_value(repo, spec["file"], spec["field"])
        derived: Any = None
        if status == "ok":
            try:
                if spec["derive"] == "ratio_vs_est_random_0.30":
                    derived = round(float(raw) / 0.30, 3)
                elif spec["derive"].startswith("peak_delta_pp"):
                    rounds = {r["round"]: r["precision_mean"] for r in raw}
                    derived = round((rounds["r1_selftrain"] - rounds["r0_prototype"]) * 100, 2)
                elif spec["derive"] == "wall_clock_ratio(baseline/decouple)":
                    derived = round(
                        raw["baseline"]["wall_clock_sec"]["mean"]
                        / raw["decouple"]["wall_clock_sec"]["mean"],
                        2,
                    )
                elif spec["derive"] == "protocol_level_b20_mean_both_arms":
                    derived = {
                        "warm_start_b20_entropy": raw["entropy"]["20"]["mean"],
                        "warm_start_b20_random": raw["random"]["20"]["mean"],
                        "w14_coldstart_b20_entropy": 0.07777777777777778,
                        "w14_coldstart_b20_random": 0.0797979797979798,
                        "cross_run_compare_allowed": False,
                        "compare_note": "W23 meta.comparability 明示与 W14 绝对数值不可直接互比(分布不同)，此处并列仅为叙事锚点",
                    }
                elif spec["derive"] == "random_minus_entropy_pp_at_b100_b200":
                    derived = {
                        "b100_random_minus_entropy_pp": round(
                            (raw["random"]["100"]["mean"] - raw["entropy"]["100"]["mean"]) * 100, 1
                        ),
                        "b200_random_minus_entropy_pp": round(
                            (raw["random"]["200"]["mean"] - raw["entropy"]["200"]["mean"]) * 100, 1
                        ),
                    }
                else:  # as-is
                    derived = raw
            except (KeyError, TypeError, ZeroDivisionError) as exc:
                status = f"derive-failed:{exc}"
        out.append(
            {
                "tag": spec["tag"],
                "layer": spec["layer"],
                "desc": spec["desc"],
                "source_file": spec["file"],
                "source_field": spec["field"],
                "raw_status": status,
                "derived": derived,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="论文回填数字索引生成器（W32）")
    ap.add_argument("--repo-root", type=Path, default=REPO_DEFAULT)
    ap.add_argument("--date", default=_dt.date.today().isoformat())
    ap.add_argument("--check", action="store_true", help="只验证不写盘")
    args = ap.parse_args()
    repo: Path = args.repo_root

    quickref_rows = parse_quickref(repo)
    placeholders = scan_placeholders(repo)
    entries, mismatches = build_entries(repo, quickref_rows)
    extras = build_extras(repo)

    doc = {
        "meta": {
            "generator": "scripts/gen_number_index.py",
            "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "target_date": args.date,
            "purpose": "W32 任务书范围1——终稿回填机械化降耗的三列索引（占位符→数值→来源文件+字段）",
            "inputs": [str(QUICKREF), str(PAPER_DIR) + "/*.md", str(REPORTS_DIR) + "/*.json"],
            "conventions": {
                "value_json": "当次运行程序化读取的新鲜值",
                "value_quickref": "quickref 表格声称值（可能滞后于 JSON 覆写）",
                "verified": "两者相对差 ≤1% 视为一致",
                "layer_rule": "三层口径（合成/公开真实/真实K9）禁止混报——AGENTS.md 硬规则 3",
            },
        },
        "number_index": entries,
        "placeholder_index": placeholders,
        "result3_candidates_and_c7_evidence": extras,
        "mismatches": mismatches,
        "stats": {
            "quickref_rows_parsed": len(quickref_rows),
            "placeholders_found": len(placeholders),
            "index_entries": len(entries),
            "verified_true": sum(1 for e in entries if e["verified"] is True),
            "verified_false": sum(1 for e in entries if e["verified"] is False),
            "extras": len(extras),
        },
    }

    payload = json.dumps(doc, ensure_ascii=False, indent=2)

    if args.check:
        print(payload)
        print(
            f"\n[index][check-mode] rows={len(quickref_rows)} placeholders={len(placeholders)} "
            f"entries={len(entries)} verified_ok={doc['stats']['verified_true']} "
            f"mismatches={len(mismatches)} extras={len(extras)}",
            file=sys.stderr,
        )
        return 0

    out_path = repo / REPORTS_DIR / f"number-index-{args.date}.json"
    out_path.write_text(payload + "\n", encoding="utf-8")
    print(
        f"[index] rows={len(quickref_rows)} placeholders={len(placeholders)} "
        f"entries={len(entries)} verified_ok={doc['stats']['verified_true']} "
        f"mismatches={len(mismatches)} extras={len(extras)}",
        file=sys.stderr,
    )
    print(f"[index] written -> {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
