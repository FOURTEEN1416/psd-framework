# E10 域适配三线汇总 — ntu120

| 臂 | retention | verdict | full_ref | 10% linear | 备注 |
|---|---|---|---|---|---|
| E9 frozen (published baseline) | 0.7496 | FAILS | 0.5837 | 0.4418 | frozen from-scratch tier pretext |
| T1 TTA (10pct head, eval-BN) | 0.7408 | FAILS | 0.5904 | 0.4407 | TENT test-time entropy minimization over BatchNorm affine params on the unlabele |
| T2 LoRA same-domain | 0.7499 | FAILS | 0.578 | 0.4338 |  |
| T2b zero-shot NTU60→tier | 0.6977 | FAILS | 0.558 | 0.407 |  |
| T2b LoRA cross-domain | 0.7198 | FAILS | 0.5966 | 0.4527 |  |
| T3 SupCon feedback | 0.7498 | FAILS | 0.576 | 0.4338 | PSD pseudo-label feedback loop: confidence-gated pseudo-labels from the (b) sema |
