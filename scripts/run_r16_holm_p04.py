# -*- coding: utf-8 -*-
"""R17: 生成 reports/r16-holm-p04-<date>.json —— E4 (P0.4) 首末配对 t 检验族的
Holm-Bonferroni 校正工件（原始 p 取自 reports/p04-tcl-2026-08-24.md §4 三行）。"""
import json
from datetime import datetime
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
FAM = {'main_calib_consensus_alpha1': 0.030, 'B2off_no_consensus': 0.143, 'alpha0_control': 0.033}
items = sorted(FAM.items(), key=lambda kv: kv[1]); m = len(items)
out, run = {}, 0.0
for k, (name, p) in enumerate(items):
    adj = min(1.0, max(run, (m - k) * p)); run = adj
    out[name] = {'raw_p': p, 'holm_adjusted': round(adj, 4)}
art = {'date': datetime.now().strftime('%Y-%m-%d'),
       'protocol': 'R16/R17 statistics artifact: Holm-Bonferroni over the P0.4 (E4) first-to-last paired-t family (m=3), closing the R16 finding that E4 printed a raw p while Section 4.2 promises corrected values',
       'family': FAM, 'adjusted': out,
       'source': 'reports/p04-tcl-2026-08-24.md section 4 (paired t=5.65 p=0.030; B-2-off p=0.143; alpha=0 p=0.033)',
       'note': 'main-test corrected p=0.090: the r0-to-last pool-precision gain is direction-consistent but NOT significant after family correction; the peak r0-to-r1 +17.88pp is a separate untested caliber'}
date = datetime.now().strftime('%Y-%m-%d')
p = REPO / 'reports' / f'r16-holm-p04-{date}.json'
p.write_text(json.dumps(art, ensure_ascii=False, indent=2), encoding='utf-8')
print('wrote', p)
