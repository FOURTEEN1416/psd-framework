# Per-seed paired statistics (recycling vs frozen)

## Raw (b)-arm accuracy

| Tier | n | Mean diff (pp) | 95% bootstrap CI | up/tie/down | Sign-test p | Cohen's dz |
|:--|--:|--:|:--|:--|--:|--:|
| NTU120 | 10 | +5.85 | [+5.04, +6.72] | 10/0/0 | 0.0020 | 4.07 |
| PanAf500 | 10 | +4.00 | [-2.40, +10.80] | 5/3/2 | 0.4531 | 0.36 |
| UCF101 | 10 | +3.63 | [+2.03, +5.14] | 9/0/1 | 0.0215 | 1.37 |

## Retention ratio (per-seed b-arm / full-label reference)

| Tier | Frozen (%) | Recycled (%) | Mean diff (pp) | 95% bootstrap CI | up/tie/down | Sign-test p | Cohen's dz |
|:--|--:|--:|--:|:--|:--|--:|--:|
| NTU120 | 74.96 | 85.18 | +10.22 | [+8.72, +11.93] | 10/0/0 | 0.0020 | 3.70 |
| PanAf500 | 82.50 | 91.64 | +9.14 | [-0.37, +19.32] | 7/0/3 | 0.3438 | 0.55 |
| UCF101 | 51.09 | 71.81 | +20.72 | [+11.83, +28.82] | 9/0/1 | 0.0215 | 1.44 |

## Budget sweep (NTU120, raw (b)-arm accuracy)

| Budget | Mean diff (pp) | 95% bootstrap CI | up/tie/down | Sign-test p |
|:--|--:|:--|:--|--:|
| 5% | +2.13 | [+1.08, +3.34] | 10/0/0 | 0.0020 |
| 10% | +3.35 | [+2.30, +4.39] | 9/0/1 | 0.0215 |
| 15% | +4.87 | [+4.27, +5.51] | 10/0/0 | 0.0020 |

Bootstrap: 10,000 resamples over seeds; sign test: exact binomial, ties excluded.
