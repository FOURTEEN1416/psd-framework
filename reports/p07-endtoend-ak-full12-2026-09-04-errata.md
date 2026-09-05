# Errata — p07 end-to-end AK full12 (2026-09-05, R16/R17 adversarial review)

This file is the errata companion to `reports/p07-endtoend-ak-full12-2026-09-04.json` (JSON left unaltered as the historical artifact).

1. **Protocol error (R16)**: the reported end-to-end arms trained the FINAL linear head on the TRUE labels of the pseudo-labeled pool clips (driver `run_p07_endtoend_ak.py` L137-139), and the iteration's precision-drop stopping consumed training-split labels. The reported budget percentages (13%) describe the seeds only; the reported head consumed 31-100% of training labels depending on arm/seed (e.g., warm-spc2 seed43: 18 seeds + 122 pool = 140/141 clips, top-1 0.3393 identical to the full-supervision arm). The `leakage_guard: "truth eval-only"` field in the JSON is contradicted by its own driver.
2. **Evaluator bug (R17)**: `evaluate()` computed macro-F1 over `range(len(class_names))` while integer labels are non-contiguous ({0,3,4,...,10}), dropping out-of-range classes; archived macro-F1 values (e.g., full-arm 0.1465) are superseded (corrected 0.165).
3. **Superseded by**: `reports/r16-endtoend-pseudo-2026-09-05.json` (corrected protocol: final head = seed truth + pool pseudo-labels; precision-drop stopping disabled). Pure-supervised arms (full-budget reference 0.3393 top-1) remain valid.
