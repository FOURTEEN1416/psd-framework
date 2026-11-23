# -*- coding: utf-8 -*-
"""make_next_paper_appendix — 由实测 summary 生成附录 A–E（md 真源 + LaTeX）。

输入（全部为归档证据，只读）：
  reports/p1_sprint_evidence/labelfree_tiers/*.json    四 tier 冻结基线
  reports/p1_sprint_evidence/adaptation/**.json         阴性套件 + recycling 主结果/网格
  runs/p1_sprint/adapt_tta_ntu120/summary.json          TENT
输出：
  docs/next_paper/paper_manuscript_v1.md               追加/替换 APPENDIX 标记块
  docs/next_paper/latex/appendix.tex                   完整附录（\\appendix 后 \\input）
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "reports" / "p1_sprint_evidence"
RUNS = REPO / "runs" / "p1_sprint"
MD = REPO / "docs" / "next_paper" / "paper_manuscript_v1.md"
TEX = REPO / "docs" / "next_paper" / "latex" / "appendix.tex"

SEEDS = list(range(42, 52))


def load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def pct(x: float) -> str:
    return f"{x * 100:.2f}"


def b_arms_pct(d: dict) -> list[str]:
    return [pct(v) for v in d["b_arms"]]


def per_seed_pct(d: dict) -> dict[int, str]:
    return {int(r["seed"]): pct(r["retention_seed"]) for r in d["per_seed"]}


# ---------------- data ----------------
frozen = {
    "ntu120": load(EV / "labelfree_tiers/ntu120_labelfree_summary.json"),
    "ucf101": load(EV / "labelfree_tiers/ucf101_labelfree_summary.json"),
    "panaf500": load(EV / "labelfree_tiers/panaf500_labelfree_summary.json"),
}
neg = {
    "TENT": load(RUNS / "adapt_tta_ntu120/summary.json"),
    "LoRA-CP": load(EV / "adaptation/ntu120/summary_lora_samedomain.json"),
    "Zero-shot": load(EV / "adaptation/ntu120/summary_zeroshot.json"),
    "LoRA-xfer": load(EV / "adaptation/ntu120/summary_lora_xfer.json"),
    "SupCon": load(EV / "adaptation/ntu120/summary.json"),
}
main3 = {
    "ntu120": load(EV / "adaptation/ntu120/summary_aug8_lr2e2.json"),
    "panaf500": load(EV / "adaptation/panaf500/summary_aug4.json"),
    "ucf101": load(EV / "adaptation/ucf101/summary_same10.json"),
}
scale120 = load(EV / "adaptation/ntu120_full/summary_full10.json")
frozen120 = load(EV / "labelfree_tiers/ntu120_full_labelfree_summary.json")
scale32 = load(EV / "adaptation/ntu120_32ch/summary_aug8_lr1e2.json")
frozen32 = load(EV / "labelfree_tiers/ntu120_32ch_labelfree_summary.json")
grid = [
    ("lr 1e-3", "1e-3", "1", "100", "1.0", "8", EV / "adaptation/ntu120/summary_same10.json"),
    ("lr 1e-2", "1e-2", "1", "100", "1.0", "8", EV / "adaptation/ntu120/summary_sweep_lr1e2.json"),
    ("lr 2e-2", "2e-2", "1", "100", "1.0", "8", EV / "adaptation/ntu120/summary_sweep_lr2e2.json"),
    ("lr 4e-2", "4e-2", "1", "100", "1.0", "8", None),
    ("aug x4", "2e-2", "4", "50", "1.0", "8", EV / "adaptation/ntu120/summary_aug4_lr2e2.json"),
    ("aug x8", "2e-2", "8", "50", "1.0", "8", EV / "adaptation/ntu120/summary_aug8_lr2e2.json"),
    ("aug x12", "2e-2", "12", "50", "1.0", "8", EV / "adaptation/ntu120/summary_aug12_lr2e2.json"),
    ("aug x4, lr 1e-2", "1e-2", "4", "50", "1.0", "8", EV / "adaptation/ntu120/summary_aug4_lr1e2.json"),
    ("epochs 200", "1e-3", "1", "200", "1.0", "8", EV / "adaptation/ntu120/summary_sweep_ep200.json"),
    ("no distillation", "1e-3", "1", "100", "0.0", "8", EV / "adaptation/ntu120/summary_sweep_d0.json"),
    ("rank 16", "1e-2", "1", "100", "1.0", "16", EV / "adaptation/ntu120/summary_sweep_lr1e2_r16.json"),
]

assert main3["ntu120"]["retention"] == 0.8516
assert main3["panaf500"]["retention"] == 0.9221
assert main3["ucf101"]["retention"] == 0.7121
assert frozen["ntu120"]["retention"] == 0.7496
assert neg["TENT"]["retention"] == 0.7408
assert scale120["retention"] == 0.7719
assert frozen120["retention"] == 0.6633
assert scale32["retention"] == 0.8355
assert frozen32["retention"] == 0.6848

# ---------------- markdown ----------------
def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(":" + "-" * max(3, len(h) + 1) for h in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


b1_rows = [
    ["Frozen baseline", "no adaptation", "—", "42–51", pct(frozen["ntu120"]["retention"])],
    ["TENT", "entropy minimization over BN affine parameters", "lr 1e-3, 1 step/batch, eval-statistics", "42–51", pct(neg["TENT"]["retention"])],
    ["Same-domain LoRA continued pretraining", "low-rank continued pretext on unlabeled target clips", "30 epochs, lr 1e-2, r 8, alpha 16, lambda 1.0", "42–51", pct(neg["LoRA-CP"]["retention"])],
    ["Cross-domain zero-shot", "NTU60 source encoder, no target update", "NTU60 phaseB epoch300", "42–51", pct(neg["Zero-shot"]["retention"])],
    ["Cross-domain LoRA transfer", "NTU60 source + low-rank update on target clips", "30 epochs, lr 1e-2, r 8, alpha 16", "42–51", pct(neg["LoRA-xfer"]["retention"])],
    ["Pseudo-label SupCon feedback", "encoder retrained on semantic-layer pseudo-labels", "1 round, 20 epochs, batch 256, lr 1e-3, tau 0.1, 7,239 labels", "42–51", pct(neg["SupCon"]["retention"])],
]

b2_rows = []
for i, s in enumerate(SEEDS):
    b2_rows.append([
        str(s),
        b_arms_pct(frozen["ntu120"])[i],
        b_arms_pct(neg["TENT"])[i],
        b_arms_pct(neg["LoRA-CP"])[i],
        b_arms_pct(neg["Zero-shot"])[i],
        b_arms_pct(neg["LoRA-xfer"])[i],
        b_arms_pct(neg["SupCon"])[i],
    ])

c1_rows = []
for s in SEEDS:
    c1_rows.append([
        str(s),
        per_seed_pct(main3["ntu120"])[s],
        per_seed_pct(main3["panaf500"])[s],
        per_seed_pct(main3["ucf101"])[s],
    ])

c2_rows = []
for name, lr, v, ep, lam, rank, path in grid:
    if path is None:
        c2_rows.append([name, lr, v, ep, lam, rank, "gate", "tau = 0"])
    else:
        d = load(path)
        c2_rows.append([name, lr, v, ep, lam, rank, pct(d["retention"]), d["verdict"]])

scale120_rows = sorted(scale120["per_seed"], key=lambda r: int(r["seed"]))
c3_rows = [
    [str(s), b_arms_pct(frozen120)[i], pct(r["b_arm"]), pct(r["retention_seed"])]
    for i, (s, r) in enumerate(zip(SEEDS, scale120_rows))
]
scale32_rows = sorted(scale32["per_seed"], key=lambda r: int(r["seed"]))
c4_rows = [
    [str(s), b_arms_pct(frozen32)[i], pct(r["b_arm"]), pct(r["retention_seed"])]
    for i, (s, r) in enumerate(zip(SEEDS, scale32_rows))
]

a1_rows = [
    ["NTU60", "60", "—", "—", "—", "74.45", "90.70 [41]"],
    ["PanAf500", "9", "400", "25", "40", "64.00", "82.50"],
    ["NTU120", "60", "7,337", "4,663", "734", "58.37", "74.96"],
    ["UCF101", "101", "9,537", "463", "953", "19.87", "51.09"],
    ["NTU120-120c", "120", "12,000", "2,400", "1,200", "44.33", "66.33"],
    ["NTU120-32ch", "60", "7,337", "4,663", "734", pct(frozen32["full_ref"]), pct(frozen32["retention"])],
]

# ---- paired statistics (generated by stats_next_paper.py) ----
_SJSON = json.loads((EV / "next_paper_stats.json").read_text(encoding="utf-8"))
_ST = {r["tier"]: r for r in _SJSON["tiers"]}
_RATIO = {r["tier"]: r for r in _SJSON["ratio"]}


def _ci(tier: str, key: str, tex: bool = False) -> str:
    s = _ST[tier][key]
    pct = r"\%" if tex else "%"
    return (f"{s['mean_diff_pp']:+.2f}~pp [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}]"
            if tex else
            f"{s['mean_diff_pp']:+.2f} pp [{s['ci95_pp'][0]:+.2f}, {s['ci95_pp'][1]:+.2f}]")


stats_md = (
    "Bootstrap of the ratio of means (10,000 resamples over seeds) gives 95% "
    f"confidence intervals of {_RATIO['NTU120']['mean_diff_pp']:+.2f} pp "
    f"[{_RATIO['NTU120']['ci95_pp'][0]:+.2f}, {_RATIO['NTU120']['ci95_pp'][1]:+.2f}] on NTU120, "
    f"{_RATIO['PanAf500']['mean_diff_pp']:+.2f} pp "
    f"[{_RATIO['PanAf500']['ci95_pp'][0]:+.2f}, {_RATIO['PanAf500']['ci95_pp'][1]:+.2f}] on PanAf500, "
    f"{_RATIO['UCF101']['mean_diff_pp']:+.2f} pp "
    f"[{_RATIO['UCF101']['ci95_pp'][0]:+.2f}, {_RATIO['UCF101']['ci95_pp'][1]:+.2f}] on UCF101, and "
    f"{_RATIO['NTU120-120c']['mean_diff_pp']:+.2f} pp "
    f"[{_RATIO['NTU120-120c']['ci95_pp'][0]:+.2f}, {_RATIO['NTU120-120c']['ci95_pp'][1]:+.2f}] on the 120-class replication, and "
    f"{_RATIO['NTU120-32ch']['mean_diff_pp']:+.2f} pp "
    f"[{_RATIO['NTU120-32ch']['ci95_pp'][0]:+.2f}, {_RATIO['NTU120-32ch']['ci95_pp'][1]:+.2f}] on the 32-channel capacity control. "
    "The per-seed sign tests are 10/0/0 ($p=0.002$), 5/3/2 ($p=0.453$), 9/0/1 ($p=0.022$), and "
    "10/0/0 ($p=0.002$) on the 120-class replication, and "
    f"{_ST['NTU120-32ch']['retention']['positive']}/{_ST['NTU120-32ch']['retention']['ties']}/{_ST['NTU120-32ch']['retention']['negative']} "
    f"($p={_ST['NTU120-32ch']['retention']['sign_test_p']:.3f}$) on the capacity control; "
    "PanAf500's sign test is not significant, so its CONFIRMS verdict is descriptive rather than a "
    "significance claim. In raw (b)-arm accuracy the corresponding intervals are "
    f"{_ci('NTU120', 'raw_b_arm')}, {_ci('PanAf500', 'raw_b_arm')}, {_ci('UCF101', 'raw_b_arm')}, and "
    f"{_ci('NTU120-120c', 'raw_b_arm')} on the 120-class replication, and "
    f"{_ci('NTU120-32ch', 'raw_b_arm')} on the capacity control."
)
stats_tex = (
    "Bootstrap of the ratio of means (10,000 resamples over seeds) gives $95\\%$ "
    f"confidence intervals of {_RATIO['NTU120']['mean_diff_pp']:+.2f}~pp "
    f"[{_RATIO['NTU120']['ci95_pp'][0]:+.2f}, {_RATIO['NTU120']['ci95_pp'][1]:+.2f}] on NTU120, "
    f"{_RATIO['PanAf500']['mean_diff_pp']:+.2f}~pp "
    f"[{_RATIO['PanAf500']['ci95_pp'][0]:+.2f}, {_RATIO['PanAf500']['ci95_pp'][1]:+.2f}] on PanAf500, "
    f"{_RATIO['UCF101']['mean_diff_pp']:+.2f}~pp "
    f"[{_RATIO['UCF101']['ci95_pp'][0]:+.2f}, {_RATIO['UCF101']['ci95_pp'][1]:+.2f}] on UCF101, and "
    f"{_RATIO['NTU120-120c']['mean_diff_pp']:+.2f}~pp "
    f"[{_RATIO['NTU120-120c']['ci95_pp'][0]:+.2f}, {_RATIO['NTU120-120c']['ci95_pp'][1]:+.2f}] on the 120-class replication, and "
    f"{_RATIO['NTU120-32ch']['mean_diff_pp']:+.2f}~pp "
    f"[{_RATIO['NTU120-32ch']['ci95_pp'][0]:+.2f}, {_RATIO['NTU120-32ch']['ci95_pp'][1]:+.2f}] on the 32-channel capacity control. "
    "The per-seed sign tests are 10/0/0 ($p=0.002$), 5/3/2 ($p=0.453$), 9/0/1 ($p=0.022$), and "
    "10/0/0 ($p=0.002$) on the 120-class replication, and "
    f"{_ST['NTU120-32ch']['retention']['positive']}/{_ST['NTU120-32ch']['retention']['ties']}/{_ST['NTU120-32ch']['retention']['negative']} "
    f"($p={_ST['NTU120-32ch']['retention']['sign_test_p']:.3f}$) on the capacity control; "
    "PanAf500's sign test is not significant, so its CONFIRMS verdict is descriptive rather than a "
    "significance claim. In raw (b)-arm accuracy the corresponding intervals are "
    f"{_ci('NTU120', 'raw_b_arm', True)}, {_ci('PanAf500', 'raw_b_arm', True)}, {_ci('UCF101', 'raw_b_arm', True)}, and "
    f"{_ci('NTU120-120c', 'raw_b_arm', True)} on the 120-class replication, and "
    f"{_ci('NTU120-32ch', 'raw_b_arm', True)} on the capacity control."
)


md_block = f"""<!-- APPENDIX:BEGIN (generated by scripts/p1_sprint/make_next_paper_appendix.py) -->

## Appendix A. Tier construction and anchor draws

All tiers pass through one export pipeline. Keypoints are resampled to
$T=30$, centered, and mapped to the 25-slot feeder consumed by the AimCLR
pretext. Human tiers use the COCO-17 to NTU25 map of
`export_tier_feeder.py`: the nose becomes the head slot, shoulders, elbows,
and wrists fill the six arm slots, hips, knees, and ankles fill the six leg
slots, and the hip root, spine mid, and neck are synthesized from shoulder
and hip means; slots 16–24 are zero. PanAf500 keypoints arrive in a 24-slot
PSD24 layout produced by an AP-10K 17-keypoint animal pose model; forelimbs
map to the arm slots, hindlimbs to the leg slots, and the same torso
synthesis applies. The collected tiers therefore share one graph
topology, fixed at pretraining, and differ only in data and reference
quality. Anchor draws are stratified by class at 10% of training clips per
seed (734, 953, and 40 anchors for the three headline weak-reference tiers;
1,200 for the 120-class replication), with seeds 42–51 throughout.

{md_table(["Tier", "Classes", "Train", "Val", "Anchors", "Full-label ref (%)", "Frozen retention (%)"], a1_rows)}

NTU60 is not re-run in this paper; its row reports the published retention
of the prior study [41] and serves as the strong-reference anchor of the
gradient.

## Appendix B. Zero-label adaptation suite

Table B1 lists the five adaptation arms of §4.2 and their configurations.
All arms receive the same 10% anchor draw per seed and are re-measured by
the same three-arm protocol; no arm sees a label beyond the anchor set, and
the TENT, continued-pretraining, zero-shot, and cross-domain arms see no
labels at all.

{md_table(["Arm", "Mechanism", "Key settings", "Seeds", "Retention (%)"], b1_rows)}

Table B2 gives the per-seed (b)-arm accuracies behind Table 1. The frozen
column is the semantic-only arm of the frozen encoder; the remaining columns
are the five adaptation arms. No column crosses the 75% line except the
frozen baseline itself.

{md_table(["Seed", "Frozen", "TENT", "LoRA-CP", "Zero-shot", "LoRA-xfer", "SupCon"], b2_rows)}

Two diagnostics complete the picture. First, a naive TENT configuration that
adapts in BatchNorm train mode collapses the 10% head from 44% to 27.0%;
the collapse is a train/evaluation statistics artifact, and once the
statistics contract is fixed the same arm moves retention by only −0.88 pp.
Second, cross-domain LoRA transfer produces the highest absolute
separability of any arm (full-label reference 59.66% and 10% linear head
45.27%, both above the frozen 58.37%/44.18%), yet its retention falls to
71.98%: the semantic layer cannot harvest the improved geometry.

## Appendix C. Recycling per-seed results and hyperparameter grid

Table C1 lists the per-seed retention of the three headline recycling
configurations. Every NTU120 seed improves; PanAf500 improves on five seeds,
ties on three, and regresses on two (the quantization caveat of §4.8);
UCF101 improves on nine of ten seeds (one seed regresses by 1.94 pp).

{md_table(["Seed", "NTU120 (lr 2e-2, x8)", "PanAf500 (lr 1e-2, x4)", "UCF101 (lr 2e-2)"], c1_rows)}

Per-seed retention is the ratio of the seed's (b)-arm accuracy to its own
full-label reference and can exceed 100% when the (b) arm beats that
reference on a seed; the headline number is the ratio of means.

{stats_md}

Table C2 gives the full NTU120 sensitivity grid. The augmentation rows use
50 epochs and the no-augmentation rows 100 epochs, as in the main text;
every cell uses ten seeds. The lr 4e-2 cell terminates at the calibration
gate (Appendix D) and therefore has no retention value.

{md_table(["Configuration", "lr", "V", "Epochs", "lambda", "Rank", "Retention (%)", "Verdict"], c2_rows)}

Table C3 gives the per-seed detail of the 120-class scale replication. Every
seed improves in both raw (b)-arm accuracy and retention; per-seed retention
ranges from 74.39% to 82.61%.

{md_table(["Seed", "Frozen b-arm (%)", "Recycling b-arm (%)", "Recycling retention (%)"], c3_rows)}

Table C4 repeats the per-seed detail for the 32-channel capacity control. Every
seed improves in retention; the frozen b-arm mean is
{pct(sum(frozen32["b_arms"]) / len(frozen32["b_arms"]))}% and the recycled
mean is {pct(scale32["mean_b"])}%.

{md_table(["Seed", "Frozen b-arm (%)", "Recycling b-arm (%)", "Recycling retention (%)"], c4_rows)}

## Appendix D. Calibration-gate failure cases

The calibration gate turns a silent quality collapse into an explicit
error. Three configurations exceed the adaptation envelope and all stop in
the same place. On NTU120 at lr 4e-2, the anchor objective never leaves
chance: the loss stays at 5.10 for 100 epochs and the anchor accuracy ends
at 1.7% over 60 classes. On PanAf500 at lr 2e-2 with 4x augmentation, the
anchor objective does fit (loss 5.51 to 1.75, anchor accuracy 35.7% over 9
classes), but the adapted feature geometry still drives the selected
similarity threshold to zero. In the 32-channel capacity control, the
16-channel-optimal recipe (lr 2e-2, 8x augmentation) completes five seeds and
trips on the sixth (seed 47): the anchor objective stays at chance (loss
5.10, anchor accuracy 1.7% over 60 classes). In all cases the semantic stage
raises `ValueError: tau must be positive: 0.0` from the threshold constructor
before any pseudo-label is emitted.

The three cases are complementary. The first shows a gate trip caused by an
objective that never converged; the second shows a gate trip caused by a
converged objective whose geometry is nevertheless incompatible with the
semantic layer's calibration. The third shows that the upper edge moves with
encoder width: the recipe that is stable at 16 channels is too strong at 32,
so the capacity control uses lr 1e-2, where all ten seeds pass. A
training-loss monitor alone would have
accepted the second run. The gate is therefore not a duplicate of the
training objective but an independent check on the feature geometry that
the next stage will actually consume.

## Appendix E. Reproduction

The three-arm measurement is implemented once, in
`labelfree_tier_replication.py` (`run_arms`), and reused by every table:
arm (a) fits a linear head on the `b`% anchors; arm (b) runs the semantic
pipeline on the same anchors; arm (c) fits the full-label linear reference
on the same frozen features. The recycling driver
`t4_budget_split_tier.py` performs, per seed, the LoRA update on that seed's
anchors, re-exports the features, and re-runs the same three arms, so the
denominator always reflects the encoder actually measured.

```bash
# 1) feeder export (T=30, 25-slot map)
python scripts/p1_sprint/export_tier_feeder.py --tier ntu120

# 2) frozen label-free pretext (150 epochs) and three-arm readout
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120 --seeds 10

# 3) anchor recycling (one command per configuration; tag fixes the summary name)
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120 --arm same10 \\
  --seeds 10 --epochs 100 --lr 0.02 --aug-views 8 --tag aug8_lr2e2

# 4) label-free baselines
python scripts/p1_sprint/tta_tent_tier.py --tier ntu120 --seeds 10 --bn-mode eval
python scripts/p1_sprint/lora_continue_tier.py --tier ntu120 --epochs 30 --lr 0.01 \\
  --source-ckpt runs/p1_sprint/labelfree_ntu120/pretext/epoch150_model.pt
python scripts/p1_sprint/supcon_feedback_tier.py --tier ntu120 --rounds 1 --epochs 20 \\
  --psd-seed 42

# 5) 120-class scale replication (balanced 120 x 100/20, fresh pretext)
python scripts/p1_sprint/export_tier_feeder.py --tier ntu120_full
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120_full --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120_full --arm same10 \\
  --seeds 10 --epochs 50 --lr 0.02 --aug-views 8 --tag full10

# 6) 32-channel capacity control (same feeder via directory junction)
python scripts/run_ntu_phaseb.py --config configs/labelfree_ntu120_32ch_pretext.yaml
python scripts/p1_sprint/labelfree_tier_replication.py --tier ntu120_32ch --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py --tier ntu120_32ch --arm same10 \\
  --seeds 10 --epochs 50 --lr 0.02 --aug-views 8 --tag aug8_lr2e2
```

Checkpoints: the frozen pretexts are
`runs/p1_sprint/labelfree_<tier>/pretext/epoch150_model.pt`; the NTU60 source
used by the cross-domain arms is
`runs/ntu_phaseB/joint_pretext/epoch300_model.pt`; the released adapted
checkpoints are `adapt_lora_xfer_ntu120/lora_epoch30_model.pt` and
`adapt_supcon_ntu120/supcon_r1_epoch20_model.pt`. Every summary JSON carries
its seeds, per-seed rows, configuration, and wall-clock time; the evidence
mirror under `reports/p1_sprint_evidence/` is byte-identical to the run
outputs.

<!-- APPENDIX:END -->
"""

text = MD.read_text(encoding="utf-8")
begin = "<!-- APPENDIX:BEGIN"
end = "<!-- APPENDIX:END -->"
if begin in text:
    head = text.split(begin)[0].rstrip()
    tail = text.split(end, 1)[1] if end in text else ""
    text = head + "\n\n" + md_block + tail
else:
    text = text.rstrip() + "\n\n---\n\n" + md_block
MD.write_text(text, encoding="utf-8")
print(f"[md] appendix block written ({len(md_block)} chars)")

# ---------------- LaTeX ----------------
def tex_table(colspec: str, head: list[str], rows: list[list[str]], caption: str, label: str, size: str = r"\small") -> str:
    lines = [
        r"\begin{table}[H]", r"\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{colspec}@{{}}}}",
        r"\toprule", " & ".join(head) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular*}", r"\end{table}"]
    return "\n".join(lines)


B1 = tex_table(
    ">{\\raggedright\\arraybackslash}p{5.0cm}>{\\raggedright\\arraybackslash}p{4.0cm}rr",
    ["Arm (mechanism)", "Key settings", "Seeds", "Retention (\\%)"],
    [[f"{r[0]} --- {r[1]}", r[2], r[3], r[4]] for r in b1_rows],
    "Zero-label adaptation arms on NTU120 (10\\% budget). All arms share the same anchor draws and the same three-arm measurement.", "tab:appB1", r"\footnotesize")

B2 = tex_table(
    "r" * 7,
    ["Seed", "Frozen", "TENT", "LoRA-CP", "Zero-shot", "LoRA-xfer", "SupCon"],
    b2_rows,
    "Per-seed (b)-arm accuracy (\\%) behind Table~\\ref{tab:labelfree}.", "tab:appB2", r"\scriptsize")

C1 = tex_table(
    "r" * 4,
    ["Seed", "NTU120 (lr 2e-2, $\\times 8$)", "PanAf500 (lr 1e-2, $\\times 4$)", "UCF101 (lr 2e-2)"],
    c1_rows,
    "Per-seed retention (\\%) of the three headline recycling configurations.", "tab:appC1", r"\small")

C2 = tex_table(
    "lrrrrrrl",
    ["Configuration", "lr", "$V$", "Epochs", "$\\lambda$", "Rank", "Retention (\\%)", "Verdict"],
    c2_rows,
    "NTU120 sensitivity grid (ten seeds per cell; augmentation rows use 50 epochs, no-augmentation rows 100).", "tab:appC2", r"\footnotesize")

C3 = tex_table(
    "rrrr",
    ["Seed", "Frozen b-arm (\\%)", "Recycling b-arm (\\%)", "Recycling retention (\\%)"],
    c3_rows,
    "Per-seed detail of the 120-class scale replication (ten seeds).", "tab:appC3", r"\small")

C4 = tex_table(
    "rrrr",
    ["Seed", "Frozen b-arm (\\%)", "Recycling b-arm (\\%)", "Recycling retention (\\%)"],
    c4_rows,
    "Per-seed detail of the 32-channel capacity control (ten seeds).", "tab:appC4", r"\small")

A1 = tex_table(
    "lrrrrrr",
    ["Tier", "Classes", "Train", "Val", "Anchors", "Full ref (\\%)", "Frozen ret. (\\%)"],
    a1_rows,
    "Tier construction. NTU60 is the published point of the prior study~\\cite{prior}; the other tiers are re-run under the shared protocol.", "tab:appA1", r"\footnotesize")

tex = rf"""% Generated by scripts/p1_sprint/make_next_paper_appendix.py — do not edit by hand.
\appendix

\section{{Tier construction and anchor draws}}
\label{{app:tiers}}
\setcounter{{table}}{{0}}
\renewcommand{{\thetable}}{{A.\arabic{{table}}}}

All tiers pass through one export pipeline. Keypoints are resampled to
$T=30$, centered, and mapped to the 25-slot feeder consumed by the AimCLR
pretext. Human tiers use the COCO-17 to NTU25 map of
\texttt{{export\_tier\_feeder.py}}: the nose becomes the head slot, shoulders,
elbows, and wrists fill the six arm slots, hips, knees, and ankles fill the
six leg slots, and the hip root, spine mid, and neck are synthesized from
shoulder and hip means; slots 16--24 are zero. PanAf500 keypoints arrive in a
24-slot PSD24 layout produced by an AP-10K 17-keypoint animal pose model;
forelimbs map to the arm slots, hindlimbs to the leg slots, and the same
torso synthesis applies. The collected tiers therefore share one graph
topology, fixed at pretraining, and differ only in data and reference
quality. Anchor draws are stratified by class at 10\% of training clips per
seed (734, 953, and 40 anchors for the three headline weak-reference tiers;
1,200 for the 120-class replication), with seeds 42--51 throughout.

{A1}

NTU60 is not re-run in this paper; its row reports the published retention
of the prior study~\cite{{prior}} and serves as the strong-reference anchor
of the gradient.

\section{{Zero-label adaptation suite}}
\label{{app:negative}}
\setcounter{{table}}{{0}}
\renewcommand{{\thetable}}{{B.\arabic{{table}}}}

Table~\ref{{tab:appB1}} lists the five adaptation arms of \S4.2 and their
configurations. All arms receive the same 10\% anchor draw per seed and are
re-measured by the same three-arm protocol; no arm sees a label beyond the
anchor set, and the TENT, continued-pretraining, zero-shot, and cross-domain
arms see no labels at all.

{B1}

Table~\ref{{tab:appB2}} gives the per-seed (b)-arm accuracies behind
Table~\ref{{tab:labelfree}}. The frozen column is the semantic-only arm of
the frozen encoder; the remaining columns are the five adaptation arms. No
column crosses the 75\% line except the frozen baseline itself.

{B2}

Two diagnostics complete the picture. First, a naive TENT configuration that
adapts in BatchNorm train mode collapses the 10\% head from 44\% to 27.0\%;
the collapse is a train/evaluation statistics artifact, and once the
statistics contract is fixed the same arm moves retention by only
$-0.88$~pp. Second, cross-domain LoRA transfer produces the highest absolute
separability of any arm (full-label reference 59.66\% and 10\% linear head
45.27\%, both above the frozen 58.37\%/44.18\%), yet its retention falls to
71.98\%: the semantic layer cannot harvest the improved geometry.

\section{{Recycling per-seed results and hyperparameter grid}}
\label{{app:grid}}
\setcounter{{table}}{{0}}
\renewcommand{{\thetable}}{{C.\arabic{{table}}}}

Table~\ref{{tab:appC1}} lists the per-seed retention of the three headline
recycling configurations. Every NTU120 seed improves; PanAf500 improves on
five seeds, ties on three, and regresses on two (the quantization caveat of
\S4.8); UCF101 improves on nine of ten seeds (one seed regresses by
1.94~pp).

{C1}

Per-seed retention is the ratio of the seed's (b)-arm accuracy to its own
full-label reference and can exceed 100\% when the (b) arm beats that
reference on a seed; the headline number is the ratio of means.

{stats_tex}

Table~\ref{{tab:appC2}} gives the full NTU120 sensitivity grid. The
augmentation rows use 50 epochs and the no-augmentation rows 100 epochs, as
in the main text; every cell uses ten seeds. The lr 4e-2 cell terminates at
the calibration gate (Appendix~\ref{{app:gate}}) and therefore has no
retention value.

{C2}

Table~\ref{{tab:appC3}} gives the per-seed detail of the 120-class scale
replication. Every seed improves in both raw (b)-arm accuracy and retention;
per-seed retention ranges from 74.39\% to 82.61\%.

{C3}

Table~\ref{{tab:appC4}} repeats the per-seed detail for the 32-channel
capacity control. Every seed improves in retention; the frozen b-arm mean is
{pct(sum(frozen32["b_arms"]) / len(frozen32["b_arms"]))}\% and the recycled
mean is {pct(scale32["mean_b"])}\%.

{C4}

\section{{Calibration-gate failure cases}}
\label{{app:gate}}

The calibration gate turns a silent quality collapse into an explicit
error. Three configurations exceed the adaptation envelope and all stop in
the same place. On NTU120 at lr 4e-2, the anchor objective never leaves
chance: the loss stays at 5.10 for 100 epochs and the anchor accuracy ends
at 1.7\% over 60 classes. On PanAf500 at lr 2e-2 with $4\times$
augmentation, the anchor objective does fit (loss 5.51 to 1.75, anchor
accuracy 35.7\% over 9 classes), but the adapted feature geometry still
drives the selected similarity threshold to zero. In the 32-channel capacity
control, the 16-channel-optimal recipe (lr 2e-2, $8\times$ augmentation)
completes five seeds and trips on the sixth (seed 47): the anchor objective
stays at chance (loss 5.10, anchor accuracy 1.7\% over 60 classes). In all
cases the semantic
stage raises \texttt{{ValueError}} with the message
``tau must be positive: 0.0'' from the
threshold constructor before any pseudo-label is emitted.

The three cases are complementary. The first shows a gate trip caused by an
objective that never converged; the second shows a gate trip caused by a
converged objective whose geometry is nevertheless incompatible with the
semantic layer's calibration. The third shows that the upper edge moves with
encoder width: the recipe that is stable at 16 channels is too strong at 32,
so the capacity control uses lr 1e-2, where all ten seeds pass. A
training-loss monitor alone would have
accepted the second run. The gate is therefore not a duplicate of the
training objective but an independent check on the feature geometry that the
next stage will actually consume.

\section{{Reproduction}}
\label{{app:repro}}

The three-arm measurement is implemented once, in
\path{{labelfree_tier_replication.py}} (\texttt{{run\_arms}}), and reused
by every table: arm (a) fits a linear head on the $b$\% anchors; arm (b) runs
the semantic pipeline on the same anchors; arm (c) fits the full-label
linear reference on the same frozen features. The recycling driver
\path{{t4_budget_split_tier.py}} performs, per seed, the LoRA update on
that seed's anchors, re-exports the features, and re-runs the same three
arms, so the denominator always reflects the encoder actually measured.

{{\footnotesize
\begin{{verbatim}}
# 1) feeder export (T=30, 25-slot map)
python scripts/p1_sprint/export_tier_feeder.py \
  --tier ntu120

# 2) frozen label-free pretext and three-arm readout
python scripts/p1_sprint/labelfree_tier_replication.py \
  --tier ntu120 --seeds 10

# 3) anchor recycling (tag fixes the summary name)
python scripts/p1_sprint/t4_budget_split_tier.py \
  --tier ntu120 --arm same10 --seeds 10 --epochs 100 \
  --lr 0.02 --aug-views 8 --tag aug8_lr2e2

# 4) label-free baselines
python scripts/p1_sprint/tta_tent_tier.py \
  --tier ntu120 --seeds 10 --bn-mode eval
python scripts/p1_sprint/lora_continue_tier.py \
  --tier ntu120 --epochs 30 --lr 0.01 \
  --source-ckpt runs/p1_sprint/labelfree_ntu120/pretext/epoch150_model.pt
python scripts/p1_sprint/supcon_feedback_tier.py \
  --tier ntu120 --rounds 1 --epochs 20 --psd-seed 42

# 5) 120-class scale replication (balanced 120 x 100/20, fresh pretext)
python scripts/p1_sprint/export_tier_feeder.py \
  --tier ntu120_full
python scripts/p1_sprint/labelfree_tier_replication.py \
  --tier ntu120_full --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py \
  --tier ntu120_full --arm same10 --seeds 10 --epochs 50 \
  --lr 0.02 --aug-views 8 --tag full10

# 6) 32-channel capacity control (same feeder via directory junction)
python scripts/run_ntu_phaseb.py \
  --config configs/labelfree_ntu120_32ch_pretext.yaml
python scripts/p1_sprint/labelfree_tier_replication.py \
  --tier ntu120_32ch --seeds 10
python scripts/p1_sprint/t4_budget_split_tier.py \
  --tier ntu120_32ch --arm same10 --seeds 10 --epochs 50 \
  --lr 0.02 --aug-views 8 --tag aug8_lr2e2
\end{{verbatim}}
}}

Checkpoints: the frozen pretexts are
\path{{runs/p1_sprint/labelfree_<tier>/pretext/epoch150_model.pt}}; the
NTU60 source used by the cross-domain arms is
\path{{runs/ntu_phaseB/joint_pretext/epoch300_model.pt}}; the released
adapted checkpoints are
\path{{adapt_lora_xfer_ntu120/lora_epoch30_model.pt}} and
\path{{adapt_supcon_ntu120/supcon_r1_epoch20_model.pt}}. Every summary
JSON carries its seeds, per-seed rows, configuration, and wall-clock time;
the evidence mirror under \path{{reports/p1_sprint_evidence/}} is
byte-identical to the run outputs.
"""
TEX.write_text(tex, encoding="utf-8")
print(f"[tex] appendix.tex written ({len(tex)} chars)")
