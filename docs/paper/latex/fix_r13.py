# -*- coding: utf-8 -*-
"""R13 修复批: 统计断言证伪处如实降级 + 口径标签纠错 + 披露补全 + 附录链修正。"""
from pathlib import Path

def fix(path, pairs):
    p = Path(path); t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, f"{path}: {t.count(old)}x :: {old[:70]}"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8"); print(f"OK {path} ({len(pairs)})")

fix("sections/04-experiments.tex", [
# 1) E1 std 修正 + 12.89 归档措辞
(r"pretraining reaches \textbf{20.89\% $\pm$ 4.45\%} top-1 against an 8.33\% random baseline",
 r"pretraining reaches \textbf{20.89\% $\pm$ 4.04\%} top-1 (sample std over the five folds) against an 8.33\% random baseline"),
(r"a projection-head feature caliber yields 12.89\% ($1.55\times$) and is reported alongside, since both calibers are legal views of the same encoder.",
 r"a projection-head feature caliber yields 12.89\% ($1.55\times$), recorded in the run report rather than a separate JSON, and both calibers are legal views of the same encoder."),
# 2) 0.691 操作点 + accuracy→precision
(r"Iterating from round~0 to round~1 raises pseudo-label pool accuracy from 0.5125 to \textbf{0.691 $\pm$ 0.013}.",
 r"Iterating from round~0 to round~1 raises pseudo-label pool precision from 0.5125 to \textbf{0.691 $\pm$ 0.013}; the precision-drop stopping rule fired at the next round (r2 degraded), so round~1 is the reported operating point."),
(r"a frozen-backbone classifier with a re-trained head reaches 44.90\% overall ($1.80\times$ the 25\% random baseline)",
 r"a frozen-backbone classifier with a re-trained head reaches 44.90\% overall (single seed; the three-seed rerun of this protocol has std $\pm$4.81\,pp) ($1.80\times$ the 25\% random baseline)"),
# 3) E3 purity 口径 + majority 基线 + operating K
(r"Anchor-guided prototype clustering reaches purity \textbf{0.5339} against a 0.3306 random-assignment baseline ($1.61\times$).",
 r"Anchor-guided prototype clustering reaches purity \textbf{0.5339} against a 0.3306 random-assignment baseline ($1.61\times$) and a 0.4858 all-majority baseline ($+4.8$\,pp---the binding comparison)."),
(r"and enlarging the seed ratio from 25\% to 100\% adds at most 1\,pp---prototype quality saturates early",
 r"and enlarging the seed ratio from 25\% to 100\% adds at most 1\,pp at the operating $K$ (other $K$ values can degrade purity)---prototype quality saturates early"),
# 4) E7 confirms→supports + 9-类口径
(r"On the 12-class public-real tier (197 self-extracted clips;",
 r"On the 12-class-protocol public-real tier (9 classes with samples; 197 self-extracted clips;"),
(r"an attribution that the pre-registered v2 replication tests rather than asserts, and confirms (E7b below)",
 r"an attribution that the pre-registered v2 replication tests rather than asserts, and supports on top-1 (E7b below)"),
# 5) E7b 嵌套括号
(r"(protocol PSD-AKV2-PREREG-001, frozen before the build (protocol and results landed in the same repository batch))",
 r"(protocol PSD-AKV2-PREREG-001, frozen before the build; protocol and results landed in the same repository batch)"),
# 6) fig3 正文 zoom 从句删除
(r"visualizes the resulting boundaries against the seed pseudo-ground-truth on two representative episodes, including a zoomed region where the quantized motion sequence changes regime.",
 r"visualizes the resulting boundaries against the seed pseudo-ground-truth on the lowest- and highest-IoU episodes."),
# 7) tab2 行口径: purity 空间 + accuracy→precision + 96.6 已有注
(r"Public-real & prototype purity (12-class space, 9 with samples) & purity & 0.5339 ($1.61\times$) \\",
 r"Public-real & prototype purity (InterPet4D 7-label seed space) & purity & 0.5339 ($1.61\times$ random; $+4.8$\,pp majority) \\"),
(r"Public-real & pseudo-label pool accuracy & acc. & 0.691 $\pm$ 0.013 \\",
 r"Public-real & pseudo-label pool precision (r1 operating point) & prec. & 0.691 $\pm$ 0.013 \\"),
# 8) APTv2 官方 vs 本地清点
(r"APTv2~\citep{yang2023aptv2} supplies a large pose pool (2{,}749 clips, 41{,}235 frames; consumed unlabeled in this study) for expansion experiments.",
 r"APTv2~\citep{yang2023aptv2} supplies a large pose pool (official counts: 2{,}749 clips, 41{,}235 frames; our local inventory holds 41{,}179 images; consumed unlabeled in this study) for expansion experiments."),
# 9) §4.1 增 claim 标签映射句
(r"Table~\ref{tab:main} consolidates the main results across the three tiers plus the public human-domain benchmark used for implementation-equivalence verification; each experiment below opens by stating the claim it tests.",
 r"Table~\ref{tab:main} consolidates the main results across the three tiers plus the public human-domain benchmark used for implementation-equivalence verification; each experiment below opens by stating the claim it tests. Claim tags used throughout: \textbf{C1} decoupled transition cost, \textbf{C3} pretraining value, \textbf{C4} unsupervised segmentation, \textbf{C5} anchor-guided semantic expansion, \textbf{C6} low-budget warm start, \textbf{C7} active learning (exploratory)."),
])

fix("sections/05-ablation-analysis.tex", [
# 10) tab3 C7 行种子级如实
(r"random leads by $+7.9$\,pp at budget 100 and $+7.1$\,pp at 200 (3/3 seeds); direction confirmed with strong in-domain scorers ($+4.2$--$5.0$\,pp at all budgets)",
 r"random leads in mean by $+7.9$\,pp at budget 100 (2/3 seeds) and $+7.1$\,pp at 200 (3/3 seeds); direction confirmed with strong in-domain scorers ($+4.2$--$5.0$\,pp at every budget above the shared initial set)"),
# 11) tab3 伪标签行 precision + 停止规则
(r"Pool accuracy 0.5125 $\to$ 0.6913 ($+17.9$\,pp at round 1)",
 r"Pool precision 0.5125 $\to$ 0.6913 ($+17.9$\,pp at round 1; precision-drop stopping fired after r1)"),
# 12) tab3 锚点行 majority
(r"Prototype purity drops from 0.5339 to a 0.3306 random-assignment baseline ($1.61\times$); seed-ratio 25\%$\to$100\% adds $\leq$1\,pp",
 r"Prototype purity drops from 0.5339 to a 0.3306 random-assignment baseline ($1.61\times$; all-majority baseline 0.4858); seed-ratio 25\%$\to$100\% adds $\leq$1\,pp at the operating $K$"),
# 13) §5 段种子级 + warm-start 预算限定
(r"entropy sampling never outperformed random selection: random led by 7.9\,pp at budget 100 and 7.1\,pp at 200, with all three seeds agreeing.",
 r"entropy sampling showed no detectable advantage: random led in mean by 7.9\,pp at budget 100 (2 of 3 seeds) and 7.1\,pp at 200 (3 of 3 seeds)."),
(r"with warm-started in-domain fine-tuned scorers random still led by 4.2--5.0\,pp at every budget.",
 r"with warm-started in-domain fine-tuned scorers random still led by 4.2--5.0\,pp at every budget above the shared initial set ($b\geq 50$)."),
# 14) below both arms' std 两处
(r"$-2.3$\,pp at 352 (crossover region, below both arms' std)",
 r"$-2.3$\,pp at 352 (crossover region, within one arm's std)"),
(r"$-2.3$\,pp at 352 clips (inside the theoretical crossover region, smaller than both arms' standard deviations and disclosed as such)",
 r"$-2.3$\,pp at 352 clips (inside the theoretical crossover region, within one arm's standard deviation and disclosed as such)"),
# 15) fig4 caption 修正
(r"Both arms saturate far above the 4.5\% random-guess baseline (22 classes). Under this cold-start protocol, uncertainty sampling (softmax entropy) shows no efficiency advantage over random selection: it does not outperform random at any budget and is exceeded by random for budgets $\geq 100$ (e.g., 69.9\% vs.\ 77.8\% at $b=100$; 80.9\% vs.\ 88.0\% at $b=200$).",
 r"Both arms reach 81--88\% at $b=200$, far above the 4.5\% chance line (22 classes). Under this cold-start protocol, uncertainty sampling (softmax entropy) shows no detectable efficiency advantage: differences at $b=20/50$ lie within noise (entropy's mean is nominally higher at $b=50$), and random leads in mean at $b\geq 100$ (e.g., 69.9\% vs.\ 77.8\% at $b=100$, 2/3 seeds; 80.9\% vs.\ 88.0\% at $b=200$, 3/3 seeds)."),
])

fix("sections/06-conclusion-limitations.tex", [
# 16) L3 与已完成 re-verification 对齐
(r"arXiv/Google Scholar re-verification was scheduled before submission, and we cannot exclude unpublished or non-indexed concurrent work.",
 r"an arXiv/CrossRef re-verification at submission time surfaced point-supervised skeleton segmentation as the nearest neighbor (differentiated in Section~\ref{sec:related}), and we cannot exclude unpublished or non-indexed concurrent work."),
# 17) full coverage + any task + 44.9 单 seed
(r"a lightweight semantic layer grows rule-engine seeds into full taxonomy coverage",
 r"a lightweight semantic layer grows rule-engine seeds toward full taxonomy coverage"),
(r"applies to any recognition task whose evaluation criteria evolve with operational practice.",
 r"applies to recognition tasks whose evaluation criteria evolve with operational practice---we verify the pattern on skeleton data in two domains and state it as a design claim elsewhere."),
(r"The self-extraction round-one model reaches 44.9\% overall validation accuracy on the four-class subset ($1.80\times$ the 25\% random baseline)",
 r"The self-extraction round-one model reaches 44.9\% overall validation accuracy on the four-class subset (single seed; $\pm$4.81\,pp three-seed; $1.80\times$ the 25\% random baseline)"),
])

fix("sections/02-related-work.tex", [
# 18) §2.2 降级 + Supplementary 死指针 + fourteen→eighteen
(r"our repository-scale survey found no published self-supervised animal-skeleton recognition pipeline.",
 r"our repository-scale survey found no published animal-skeleton pipeline combining self-supervised representation learning with a label-expansion loop."),
(r"(systematic repository-scale survey; full matrix in Supplementary Material)",
 r"(systematic repository-scale survey; full matrix in Appendix~\ref{app:novelty})"),
(r"ten query groups and fourteen candidate works with zero occupancy",
 r"ten query groups and eighteen candidate works with zero occupancy"),
])

fix("sections/03-method.tex", [
# 19) Algorithm 停止条件 + τ 自适应 + B 定义 + warm-start 来源
(r"\Until{proposal assignment stabilizes or iteration budget reached}",
 r"\Until{proposal assignment stabilizes, the iteration budget is reached, or estimated pseudo-label precision drops}"),
(r"Proposals above threshold $\tau$ receive pseudo-labels",
 r"Proposals above the round-adaptive threshold $\tau$ (quantile-targeted on estimated coverage) receive pseudo-labels"),
(r"a warm-start protocol initializes this stage from previously trained semantic-layer weights",
 r"a warm-start protocol initializes this stage from the semantic-layer weights of a full-budget model trained under the previous taxonomy on the same distribution"),
])

fix("main.tex", [
# 20) 摘要: full coverage / kNN probe 标注 / 9-类 / highlights 同步 ≤85
(r"a lightweight semantic layer expands rule-engine seeds into full taxonomy coverage",
 r"a lightweight semantic layer expands rule-engine seeds toward full taxonomy coverage"),
(r"pretraining alone yields kNN top-1 of \textbf{20.89\% versus an 8.33\% random baseline (2.51$\times$)}",
 r"pretraining alone yields subject-identification $k$-NN top-1 of \textbf{20.89\% versus an 8.33\% random baseline (2.51$\times$; a representation probe, not behavior accuracy)"),
(r"on a 12-class canine benchmark",
 r"on a canine benchmark (12-class protocol, 9 classes with samples)"),
(r"\item First (to our knowledge) anchor-cluster-pseudo-label transfer to temporal skeleton recognition",
 r"\item First, to our knowledge, anchor--cluster pseudo-label transfer to skeletons"),
# 21) 附录计数统一 + TIP 2024 + 未引行注
(r"ten query groups over GitHub code search plus a full scan of the maintained awesome-skeleton-AR catalog (19 candidate entries)",
 r"ten query groups over GitHub code search plus a full scan of the maintained awesome-skeleton-AR catalog (18 survey candidates; the submission-time re-verification added the point-supervised row, giving 19 table rows)"),
(r"Momentum-contrastive teacher (TIP 2025); GRA (TNNLS)",
 r"Momentum-contrastive teacher (TIP 2024); GRA (TNNLS)"),
(r"``Occupies niche?'' = does the work instantiate all three mechanism parts for temporal skeleton recognition.",
 r"``Occupies niche?'' = does the work instantiate all three mechanism parts for temporal skeleton recognition. Rows without bibliography keys are catalog-scan entries identified by name and venue in the released survey log (\texttt{dev-docs/research/})."),
# 22) 复现链: E7/E8→E7 + 合成层链 + NTU 前置 + 标签修正
(r"$\to$ \texttt{scripts/run\_p05\_al\_warmstart.py} (warm-start) $\to$ \texttt{scripts/run\_p05\_public\_real\_full12.py}",
 r"$\to$ \texttt{scripts/run\_p05\_al\_warmstart.py} (warm-start) $\to$ synthetic tier: \texttt{gen\_synth\_22class.py} $\to$ \texttt{run\_p05\_full.py} (96.6\%) $\to$ \texttt{run\_c1\_decouple.py} (C1) $\to$ \texttt{run\_p05\_al\_efficiency.py} (C7/fig4) $\to$ \texttt{run\_ablation\_pretrain.py} (tab3 gradient) $\to$ \texttt{seg\_strategy\_ablation.py} (tab3 segmentation) $\to$ \texttt{scripts/run\_p05\_public\_real\_full12.py}"),
(r"$\to$ \texttt{run\_p07\_endtoend\_ak.py} (E7/E8) $\to$",
 r"$\to$ \texttt{run\_p07\_endtoend\_ak.py} (E7) $\to$"),
(r"$\to$ \texttt{run\_p14\_ntu\_featuredump.py}, \texttt{run\_p14\_ntu\_lowres.py}, \texttt{run\_r12\_artifacts.py} (E9, statistics). NTU equivalence: \texttt{run\_ntu\_phaseb.py} + \texttt{run\_ntu\_lineareval.py} + \texttt{ntu\_ensemble\_3s.py}.",
 r"NTU equivalence first: \texttt{run\_ntu\_phaseb.py} + \texttt{run\_ntu\_lineareval.py} + \texttt{ntu\_ensemble\_3s.py}; then \texttt{run\_p14\_ntu\_featuredump.py}, \texttt{run\_p14\_ntu\_lowres.py} (E9), \texttt{run\_r12\_artifacts.py} (statistics artifacts cited in E7b/tab3)."),
])
print("R13 tex batch done")
