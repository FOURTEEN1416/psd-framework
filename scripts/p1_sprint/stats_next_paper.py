# -*- coding: utf-8 -*-
"""stats_next_paper — 为 Anchor Recycling 主结果补配对统计与 bootstrap CI。

读取已归档的 per-seed 汇总（只读），输出：
  reports/p1_sprint_evidence/next_paper_stats.json
  docs/next_paper/stats_table.md

统计口径：
  · 配对差 d_i = recycling retention_seed_i − frozen (b_arm_i / full_ref)
  · 均值差的 95% bootstrap CI（10,000 次，按 seed 重采样）
  · 符号检验（双边精确二项，ties 剔除）
  · Cohen's dz = mean(d) / sd(d)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "reports" / "p1_sprint_evidence"
OUT_JSON = EV / "next_paper_stats.json"
OUT_MD = REPO / "docs" / "next_paper" / "stats_table.md"

TIERS = {
    "NTU120": (
        EV / "labelfree_tiers/ntu120_labelfree_summary.json",
        EV / "adaptation/ntu120/summary_aug8_lr2e2.json",
    ),
    "PanAf500": (
        EV / "labelfree_tiers/panaf500_labelfree_summary.json",
        EV / "adaptation/panaf500/summary_aug4.json",
    ),
    "UCF101": (
        EV / "labelfree_tiers/ucf101_labelfree_summary.json",
        EV / "adaptation/ucf101/summary_same10.json",
    ),
    "NTU120-120c": (
        EV / "labelfree_tiers/ntu120_full_labelfree_summary.json",
        EV / "adaptation/ntu120_full/summary_full10.json",
    ),
    "NTU120-32ch": (
        EV / "labelfree_tiers/ntu120_32ch_labelfree_summary.json",
        EV / "adaptation/ntu120_32ch/summary_aug8_lr1e2.json",
    ),
}


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def bootstrap_ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(name: str, frozen_path: Path, recycle_path: Path) -> dict:
    fz = load(frozen_path)
    rc = load(recycle_path)
    base = np.array(fz["b_arms"], dtype=float)
    ref = float(fz["full_ref"])
    rows = sorted(rc["per_seed"], key=lambda r: int(r["seed"]))
    rec_b = np.array([r["b_arm"] for r in rows], dtype=float)
    rec_ret = np.array([r["retention_seed"] for r in rows], dtype=float)
    frozen_ret = base / ref
    raw_diff = rec_b - base
    ret_diff = rec_ret - frozen_ret

    def summarize(diff: np.ndarray) -> dict:
        lo, hi = bootstrap_ci(diff)
        pos = int((diff > 0).sum())
        neg = int((diff < 0).sum())
        ties = int((diff == 0).sum())
        n_eff = pos + neg
        p = float(stats.binomtest(pos, n_eff, 0.5, alternative="two-sided").pvalue) if n_eff else 1.0
        dz = float(diff.mean() / diff.std(ddof=1)) if diff.std(ddof=1) > 0 else float("inf")
        return {
            "mean_diff_pp": float(diff.mean() * 100),
            "ci95_pp": [lo * 100, hi * 100],
            "positive": pos,
            "negative": neg,
            "ties": ties,
            "sign_test_p": p,
            "cohens_dz": dz,
        }

    return {
        "tier": name,
        "n": len(base),
        "frozen_retention": float(frozen_ret.mean()),
        "recycled_retention": float(rec_ret.mean()),
        "raw_b_arm": summarize(raw_diff),
        "retention": summarize(ret_diff),
        "per_seed_raw_diff_pp": [round(float(x) * 100, 2) for x in raw_diff],
        "per_seed_ret_diff_pp": [round(float(x) * 100, 2) for x in ret_diff],
        "recycled_per_seed_pct": [round(float(x) * 100, 2) for x in rec_ret],
    }


BUDGET = [
    ("5%", EV / "adaptation/ntu120/summary_ref5.json", EV / "adaptation/ntu120/summary_split5.json"),
    ("10%", EV / "labelfree_tiers/ntu120_labelfree_summary.json", EV / "adaptation/ntu120/summary_same10.json"),
    ("15%", EV / "adaptation/ntu120/summary_ref15.json", EV / "adaptation/ntu120/summary_dual15.json"),
]


def analyze_budget(name: str, frozen_path: Path, recycle_path: Path) -> dict:
    fz = load(frozen_path)
    rc = load(recycle_path)
    base = np.array(fz["b_arms"], dtype=float)
    rows = sorted(rc["per_seed"], key=lambda r: int(r["seed"]))
    rec = np.array([r["b_arm"] for r in rows], dtype=float)
    diff = rec - base
    lo, hi = bootstrap_ci(diff)
    pos = int((diff > 0).sum())
    neg = int((diff < 0).sum())
    ties = int((diff == 0).sum())
    n_eff = pos + neg
    p = float(stats.binomtest(pos, n_eff, 0.5, alternative="two-sided").pvalue) if n_eff else 1.0
    return {
        "budget": name,
        "mean_diff_pp": float(diff.mean() * 100),
        "ci95_pp": [lo * 100, hi * 100],
        "positive": pos,
        "negative": neg,
        "ties": ties,
        "sign_test_p": p,
    }



def analyze_ratio(name: str, frozen_path: Path, recycle_path: Path) -> dict:
    """Headline retention = ratio of means; bootstrap seeds jointly."""
    fz = load(frozen_path)
    rc = load(recycle_path)
    base = np.array(fz["b_arms"], dtype=float)
    ref = float(fz["full_ref"])
    rows = sorted(rc["per_seed"], key=lambda r: int(r["seed"]))
    b = np.array([r["b_arm"] for r in rows], dtype=float)
    fr = np.array([r["full_ref"] for r in rows], dtype=float)
    frozen = float(base.mean() / ref)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(b), size=(10000, len(b)))
    ratios = b[idx].mean(axis=1) / fr[idx].mean(axis=1)
    diff = (ratios - frozen) * 100
    return {
        "tier": name,
        "headline_pct": float(b.mean() / fr.mean() * 100),
        "mean_diff_pp": float(diff.mean()),
        "ci95_pp": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
        "share_positive": float((ratios > frozen).mean()),
    }

def main() -> None:
    out = [analyze(k, *v) for k, v in TIERS.items()]
    ratio = [analyze_ratio(k, *v) for k, v in TIERS.items()]
    budget = [analyze_budget(k, f, r) for k, f, r in BUDGET]
    OUT_JSON.write_text(json.dumps({"tiers": out, "ratio": ratio, "budget": budget}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Per-seed paired statistics (recycling vs frozen)",
        "",
        "## Raw (b)-arm accuracy",
        "",
        "| Tier | n | Mean diff (pp) | 95% bootstrap CI | up/tie/down | Sign-test p | Cohen's dz |",
        "|:--|--:|--:|:--|:--|--:|--:|",
    ]
    for r in out:
        s = r["raw_b_arm"]
        lines.append(
            f"| {r['tier']} | {r['n']} | {s['mean_diff_pp']:+.2f} | "
            f"[{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}] | "
            f"{s['positive']}/{s['ties']}/{s['negative']} | {s['sign_test_p']:.4f} | {s['cohens_dz']:.2f} |"
        )
    lines += [
        "",
        "## Retention ratio (per-seed b-arm / full-label reference)",
        "",
        "| Tier | Frozen (%) | Recycled (%) | Mean diff (pp) | 95% bootstrap CI | up/tie/down | Sign-test p | Cohen's dz |",
        "|:--|--:|--:|--:|:--|:--|--:|--:|",
    ]
    for r in out:
        s = r["retention"]
        lines.append(
            f"| {r['tier']} | {r['frozen_retention']*100:.2f} | {r['recycled_retention']*100:.2f} | "
            f"{s['mean_diff_pp']:+.2f} | [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}] | "
            f"{s['positive']}/{s['ties']}/{s['negative']} | {s['sign_test_p']:.4f} | {s['cohens_dz']:.2f} |"
        )
    lines += [
        "",
        "## Headline retention (ratio of means, seed bootstrap)",
        "",
        "| Tier | Headline (%) | Mean gain (pp) | 95% bootstrap CI | share > frozen |",
        "|:--|--:|--:|:--|--:|",
    ]
    for r in ratio:
        lines.append(
            f"| {r['tier']} | {r['headline_pct']:.2f} | {r['mean_diff_pp']:+.2f} | "
            f"[{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}] | {r['share_positive']*100:.1f}% |"
        )
    lines += [
        "",
        "## Budget sweep (NTU120, raw (b)-arm accuracy)",
        "",
        "| Budget | Mean diff (pp) | 95% bootstrap CI | up/tie/down | Sign-test p |",
        "|:--|--:|:--|:--|--:|",
    ]
    for b in budget:
        lines.append(
            f"| {b['budget']} | {b['mean_diff_pp']:+.2f} | "
            f"[{b['ci95_pp'][0]:+.2f}, {b['ci95_pp'][1]:+.2f}] | "
            f"{b['positive']}/{b['ties']}/{b['negative']} | {b['sign_test_p']:.4f} |"
        )
    lines += ["", "Bootstrap: 10,000 resamples over seeds; sign test: exact binomial, ties excluded.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
