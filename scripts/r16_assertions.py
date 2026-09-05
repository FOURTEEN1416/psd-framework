# -*- coding: utf-8 -*-
"""R16 收尾断言：残留清零 + 修正数字在位 + 结构完整性（refs/labels/cites/bib）。"""
import io, glob, re, sys, os

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
tex = {}
files = (['docs/paper/latex/main.tex', 'docs/paper/latex/highlights.tex']
         + sorted(glob.glob('docs/paper/latex/sections/*.tex')))
for f in files:
    tex[f] = io.open(f, encoding='utf-8').read()
all_tex = '\n'.join(tex.values())
fails = []

def gone(pat, why):
    hits = [f for f, s in tex.items() if pat in s]
    if hits:
        fails.append('RESIDUE %r (%s) in %s' % (pat, why, hits))

def present(pat, why):
    if pat not in all_tex:
        fails.append('MISSING %r (%s)' % (pat, why))

# ---- superseded protocol-error numbers / falsified citations / fixed wording ----
for pat, why in [
    ('94\\%', 'old E7 retention'), ('99.5', 'old E9 retention'), ('31.96', 'old E7 top1'),
    ('33.23', 'old E7b warm'), ('88.6', 'old EP2 retention / TCL'), ('74.11', 'old E9'),
    ('+8.1', 'old E9 gain'), ('34.4k', 'old E9 pool'), ('26.07', 'old aimclr spc2'),
    ('24.11', 'old aimclr spc4'), ('14.36', 'old mf1'), ('14.78', 'old mf1'),
    ('recovering most of the gap', 'old E9 framing'), ('near-flat', 'old fig5 narrative'),
    ('four independent tiers', 'old fig5 caption'), ('self-collected', 'BCST provenance'),
    ('a five new arms', 'grammar'), ('; A motion-word', 'grammar'),
    ('and left to future work', 'grammar'), ('decaying monotonically', 'L10 monotonicity'),
    (r'at the operating $K$', 'tab3 K=14'), ('three data calibers', 'conclusion'),
    ('released-checkpoint table', '80.9 provenance'), ('pre-registration ledger', 'false pointer'),
    ('E5: taxonomy', 'E-numbering'), ('(E5) tests', 'E-numbering'),
    ('82.7', 'TCL falsified'), ('94.43', 'BCST old'), ('77.2', 'AimCLR++ old'),
    ('12,000', 'Grimm old'), ('USD 12', 'Grimm old'), ('40,128', 'NTU official count'),
    (r'$\pm$ 4.45', 'E1 old std'), (r'$\pm$4.45', 'E1 old std'),
    (r'$\pm$0.5\,pp observed', '96.6 unsourced dispersion'),
    (r'corrected $0.047$', 'old Holm'),
    (r'corrected $0.77$', 'old Holm'), ('1.71', 'old aimclr v2 mf1'),
    (r', $p=0.030$', 'E4 raw p alone'), (r'24.6\\%', 'old scratch'),
    ('matches the official reference', 'conclusion overclaim'),
    ('none exists there', 'E9 false no-GT claim'), ('head-estimated precision', 'E9 false stopping claim'),
    # R17 regression: evaluator-bug & family-caliber & wording fixes
    ('0.043', 'R17 per-tier Holm replaces combined'), ('0.024', 'R17 per-tier Holm v2'),
    ('14.7', 'EP3 buggy-evaluator macro-F1'), ('7.83', 'EP3 buggy-evaluator macro-F1'),
    (r'9--12\%', 'head_calib loose range'), ('matched absolute budget', 'v2 seeds 14 not 16'),
    (r'6\% of the v2 training pool', 'v2 seed fraction'),
    ('cold-start protocols remain near chance', 'intro unsupported clause'),
    ('grows complete labeled coverage', '03 completeness wording'),
    ('plus a full scan of the maintained', 'novelty double-count'),
    ('requires a labeled reference', 'E7 stopping misdescription'),
    ('reported as-is. These', 'abstract redundant tail'),
]:
    gone(pat, why)

for pat, why in [
    ('0.023', 'R17 v1 mf1 per-tier Holm'), ('0.049', 'R17 v1 spc4 per-tier Holm'),
    ('0.012', 'R17 v2 per-tier Holm'), ('88.7', 'E9 linear-only retention context'),
    ('19.33', 'corrected macro-F1'), ('14 of 256', 'v2 realized seeds'),
    ('inert on this taxonomy', 'gate disclosure E7'), ('inert on the numeric NTU taxonomy', 'gate disclosure E9'),
    ('Warm-start small-budget usability', 'warm-start paragraph'),
    ('C2} is the novelty claim', 'claim-tag note'),
    (r'13\%/5.5\%', 'fig5 caption fraction'),
    ('run\\_r16\\_holm\\_p04.py', 'chain generator'),
]:
    present(pat, why)

# ---- corrected numbers / disclosures present ----
for pat, why in [
    ('90.6', 'E9 corrected retention'), ('67.5', 'E9 corrected top1'),
    ('9.8', 'E7 corrected warm'), ('15.2', 'E7 aimclr'), ('13.1', 'E7b warm v2'),
    ('17.6', 'E7b aimclr v2'), ('0.090', 'E4 Holm corrected'), ('E6: taxonomy', 'E-numbering fixed'),
    ('never on pool ground truth', 'protocol fix disclosed'),
    ('solver-path noise', 'EP3 sensitivity'), (r'Figure~\ref{fig:budget}', 'fig5 referenced'),
    ('converge', 'stopping disclosure'), ('post-hoc diagnostic', 'oracle disclosure'),
]:
    present(pat, why)

# ---- structural ----
_main = tex['docs/paper/latex/main.tex']
if _main.count('\\end{document}') != 1:
    fails.append('main.tex end{document} count != 1: %d' % _main.count('\\end{document}'))
refs = set(re.findall(r'\\ref\{([^}]+)\}', all_tex))
labels = set(re.findall(r'\\label\{([^}]+)\}', all_tex))
dangling = refs - labels
if dangling:
    fails.append('dangling refs: %s' % sorted(dangling))
cites = set()
for s in tex.values():
    cites.update(re.findall(r'\\citep?\{([^}]+)\}', s))
cites = {c.strip() for grp in cites for c in grp.split(',')}
bibtxt = io.open('docs/paper/latex/refs.bib', encoding='utf-8').read()
bib = set(re.findall(r'@\w+\{([^,]+),', bibtxt))
miss = cites - bib
if miss:
    fails.append('cites missing in bib: %s' % sorted(miss))
orphans = bib - cites
print('orphan bib entries (info, not fail):', sorted(orphans))
print('refs=%d labels=%d cites=%d bib=%d' % (len(refs), len(labels), len(cites), len(bib)))
if fails:
    print('FAILURES (%d):' % len(fails))
    for f in fails:
        print(' -', f)
    sys.exit(1)
print('ALL ASSERTIONS PASS')
