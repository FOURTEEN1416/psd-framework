# -*- coding: utf-8 -*-
"""R12 工件补齐: (1) Holm-Bonferroni 校真实执行 (p10/p12 族); (2) 8 类同空间对照可复现脚本。
产出: reports/r12-holm-2026-09-05.json + reports/r12-eightclass-control-2026-09-05.json"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "reports"

# ---------- (1) Holm ----------
def holm(pmap):
    """pmap: {name: p}. 返回 {name: corrected_p}（step-down, 单调化）。"""
    items = sorted(pmap.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, p * (m - i))
        running = max(running, adj)
        out[name] = round(running, 4)
    return out

p10 = json.loads((OUT / "p10-seedexpansion-2026-09-04.json").read_text(encoding="utf-8"))
p12 = json.loads((OUT / "p12-akv2-replication-2026-09-04.json").read_text(encoding="utf-8"))

fam10 = {k: v["wilcoxon_p"] for k, v in p10["paired_tests"].items()}
fam12 = {k: v["wilcoxon_p"] for k, v in p12["paired_tests"].items()}
holm10, holm12 = holm(fam10), holm(fam12)

# ---------- (2) 8-class same-space control ----------
import pickle
p07 = json.loads((OUT / "p07-endtoend-ak-full12-2026-09-04.json").read_text(encoding="utf-8"))
full_run = [r for r in p07["runs"] if r["arm"] == "warm" and r.get("spc") == -1][0]
pc = full_run["per_class"]
d1 = pickle.load(open(REPO / "runs/public_real_dataset/full12_T30.pkl", "rb"))
val1 = [x for x in d1 if x["split"] == "val"]
sup = Counter(x["psd_class"] for x in val1)
V2_CLASSES = {"track", "stay", "watch", "retrieve", "apprehend", "jump", "bite", "sit"}
num = sum(pc[c]["acc"] * sup[c] for c in V2_CLASSES if c in pc)
den = sum(sup[c] for c in V2_CLASSES if c in pc)
top1_8 = num / den
maj8 = max(sup[c] for c in V2_CLASSES) / den

result = {
    "date": datetime.now().isoformat(timespec="seconds"),
    "holm": {
        "family_p10_v1tier": {"raw": fam10, "corrected": holm10},
        "family_p12_v2tier": {"raw": fam12, "corrected": holm12},
        "method": "Holm-Bonferroni step-down within each experiment family (m=6 each), monotonicity enforced",
        "note": "exact Wilcoxon p from n=10 sign patterns: 10-0-0 -> 2/2^10=0.001953; 7-0-3 -> 2/2^7=0.015625",
    },
    "eightclass_control": {
        "method": "v1 full-supervision arm per-class accuracies (p07 JSON, deterministic single run) reweighted by v1 validation support counts restricted to the 8 classes present in v2; NOT a re-run of the model on v2 clips",
        "v1_full_8class_top1": round(top1_8, 4),
        "v1_val_support_8cls": den, "v1_val_majority_8cls": round(maj8, 4),
        "v2_full_top1": 0.375,
        "same_space_delta_pp": round((0.375 - top1_8) * 100, 2),
        "above_chance_1_8": {"v1_8cls": round(top1_8 * 100 - 12.5, 1), "v2": 25.0},
        "above_majority": {"v1_8cls": round((top1_8 - maj8) * 100, 1), "v2": round(37.5 - 34.4, 1)},
        "residual_confound_disclosed": "v1(8cls) evaluates on v1's 52 val clips, v2 on its own 96 val clips; the control removes the chance-rate confound but not the clip-set difference",
    },
}
out = OUT / "r12-holm-eightclass-2026-09-05.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=1))
