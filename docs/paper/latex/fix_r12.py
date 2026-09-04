# -*- coding: utf-8 -*-
"""R12 修复批 1: 正文 tex（统计校正落地/措辞诚实化/新颖性防御/数字修正）。"""
from pathlib import Path

def fix(path, pairs):
    p = Path(path); t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, f"{path}: {t.count(old)}x :: {old[:60]}"
        t = t.replace(old, new)
    p.write_text(t, encoding="utf-8"); print(f"OK {path} ({len(pairs)})")

fix("sections/04-experiments.tex", [
# 1) §4.2 统计协议句: fold→seed + Holm 真实执行 + dispersion 声明
(r"aggregate comparisons are accompanied by paired fold-level tests, multiple comparisons within a table are corrected (Holm--Bonferroni), effect sizes are reported alongside raw values with a random-baseline reference",
 r"aggregate comparisons are accompanied by paired seed-level tests; within each experiment family the reported $p$ values are Holm--Bonferroni corrected (corrected values printed alongside the test), effect sizes are reported alongside raw values with a random-baseline reference"),
(r"$k$-NN probes and fine-tuning report 5-fold mean$\pm$std with the fold-level spread disclosed;",
 r"$k$-NN probes report 5-fold mean$\pm$std; end-to-end arms report seed-level dispersion, which is a spread estimate and not a confidence interval;"),
# 2) E7 Holm 校正 p
(r"winning all ten same-seed pairs (Wilcoxon $p=0.002$)",
 r"winning all ten same-seed pairs (Wilcoxon $p=0.002$; Holm-corrected $p=0.012$)"),
# 3) E7b committed-before 时序诚实化 + 25.96 算式明说 + 校正 p
(r"protocol PSD-AKV2-PREREG-001, committed before any v2 build",
 r"protocol PSD-AKV2-PREREG-001, frozen before the build (protocol and results landed in the same repository batch)"),
(r"we therefore add a same-space control---re-scoring the v1 full-supervision arm on the eight classes v2 retains yields 25.96\%---so the same-space rise is $+11.54$\,pp",
 r"we therefore add a same-space control---reweighting the v1 full-supervision arm's per-class accuracies over the eight classes v2 retains, on v1's validation clips, yields 25.96\% (released artifact \texttt{r12-holm-eightclass}); the control removes the chance-rate confound but not the clip-set difference---so the same-space rise is $+11.54$\,pp"),
(r"(top-1 $+12.9$\,pp and macro-F1 $+6.1$\,pp, all ten seed pairs, $p=0.002$)",
 r"(top-1 $+12.9$\,pp and macro-F1 $+6.1$\,pp, all ten seed pairs, Holm-corrected $p=0.012$)"),
# 4) E9 时序 + 40,091 差异 + 过滤器接受率
(r"we pre-registered (protocol PSD-NTU-PREREG-001, committed before the experiment) a budget-shrinkage test on NTU RGB+D 60 cross-subject~\citep{shahroudy2016ntu}. Holding the frozen epoch-300 joint pretext backbone fixed, a 10\% class-stratified training subset yields",
 r"we pre-registered (protocol PSD-NTU-PREREG-001, frozen before the experiment) a budget-shrinkage test on NTU RGB+D 60 cross-subject~\citep{shahroudy2016ntu}. Holding the frozen epoch-300 joint pretext backbone fixed, a 10\% class-stratified subset of the 40{,}091 exported training clips (the official split lists 40{,}128; the difference is the provider's frame-50 export) yields"),
(r"the confidence-filtered pool adds $\approx$34.4k pseudo-labeled clips per seed---the near-restored training-set size \emph{is} the mechanism---and pool precision is not separately evaluated against ground truth on NTU",
 r"the confidence-filtered pool adds $\approx$34.4k pseudo-labeled clips per seed---the near-restored training-set size \emph{is} the mechanism---the gate accepts $\approx$95\% of the unlabeled pool at this scale, so this arm cannot separate filter quality from pool-size restoration (Section~\ref{sec:conclusion}, L11), and pool precision is not separately evaluated against ground truth on NTU"),
# 5) E5 等价措辞
(r"and accuracy is statistically equivalent between arms ($-0.91$\,pp full / $+2.27$\,pp small, both inside the $<2.3$\,pp noise band)",
 r"and accuracy differences fall inside the pre-registered noise band ($-0.91$\,pp full / $+2.27$\,pp small, both $<2.3$\,pp)"),
# 6) 1.62→1.61 (两处)
(r"purity \textbf{0.5339} against a 0.3306 random-assignment baseline ($1.62\times$)",
 r"purity \textbf{0.5339} against a 0.3306 random-assignment baseline ($1.61\times$)"),
(r"prototype purity (12-class space, 9 with samples) & purity & 0.5339 ($1.62\times$)",
 r"prototype purity (12-class space, 9 with samples) & purity & 0.5339 ($1.61\times$)"),
# 7) tab2 96.6 单发披露
(r"Synthetic & 22-class, full budget (matched 50-ep) & top-1 & 96.6\%",
 r"Synthetic & 22-class, full budget (matched 50-ep) & top-1 & 96.6\% (single run; $\pm$0.5\,pp observed)"),
])

fix("sections/01-introduction.tex", [
# 8) 贡献 bullet 2 首次性 hedge + 附录指向
(r"\item \textbf{The first transfer of image-domain anchor--cluster--pseudo-labeling to temporal-skeleton recognition}, supported by a systematic repository-scale survey with zero occupancy (Supplementary Material); arXiv/Scholar re-verification is scheduled before submission.",
 r"\item \textbf{A transfer of image-domain anchor--cluster--pseudo-labeling to temporal-skeleton recognition that, to our knowledge, is the first of its kind}, supported by a repository-scale survey (Appendix~\ref{app:novelty}) complemented by an arXiv/CrossRef re-verification at submission time that identified point-supervised skeleton segmentation as the closest neighbor~\citep{pointsup2026}, differentiated in Section~\ref{sec:related}; we cannot exclude unpublished concurrent work."),
# 9) ≈0.30 → 0.32
(r"versus an $\approx$0.30 equal-segment-count random-cut null",
 r"versus an $0.32$ equal-segment-count random-cut null"),
# 10) bullet 4 三档措辞 + cold-start 落点
(r"\item \textbf{A complete low-resource pipeline evaluated under a three-caliber protocol} (synthetic / public-real / real-K9); warm-started semantic-layer initialization makes a 20-clip budget usable (\textbf{82.0\% top-1 on the synthetic-offset tier}; cold-start protocols remain near chance at this budget under their respective distributions)",
 r"\item \textbf{A complete low-resource pipeline evaluated under two exercised tiers of a three-tier pre-registered protocol} (synthetic / public-real; the real-K9 tier pre-registered but not exercised); warm-started semantic-layer initialization makes a 20-clip budget usable (\textbf{82.0\% top-1 on the synthetic-offset tier}, versus $\approx$7.8\% for the cold-start control at the same budget)"),
])

fix("sections/02-related-work.tex", [
# 11) §2.1 gap 绝对句软化 + 补引
(r"None of them exploits unlabeled video streams at scale, and none is designed so that changing evaluation criteria can be absorbed without re-annotating the corpus.",
 r"Few exploit unlabeled streams---domain-adaptive pretraining improves primate behavior recognition~\citep{primateda2025}, and self-supervised video pretraining extends to multi-animal settings~\citep{mabe22,beast2025}---and none is designed so that changing evaluation criteria can be absorbed without re-annotating the corpus."),
# 12) §2.2 骨架 foundation/MAE 线补入 + 区分
(r"our repository-scale survey found no published self-supervised animal-skeleton recognition pipeline",
 r"our repository-scale survey found no published self-supervised animal-skeleton recognition pipeline. In the human skeleton domain, masked autoencoders~\citep{skeletonmae2022} and foundation-model efforts~\citep{skeletonfm2025} pretrain representations for broad downstream transfer; our decoupling claim is orthogonal---the frozen component is the physics encoder, the revisable component is the taxonomy-facing semantic layer, and the measured quantity is taxonomy-transition cost rather than representation quality"),
# 13) §2.3 最近邻 point-supervised 区分
(r"MAC-Learning introduces anchor-based contrastive learning, where ``anchors'' are contrastive sample pairs selected to structure the embedding space~\citep{mac2022learning}; it does not maintain seed-annotated anchors that supervise cluster assignment, and it includes no iterative pseudo-label loop.",
 r"MAC-Learning introduces anchor-based contrastive learning, where ``anchors'' are contrastive sample pairs selected to structure the embedding space~\citep{mac2022learning}; it does not maintain seed-annotated anchors that supervise cluster assignment, and it includes no iterative pseudo-label loop. Point-supervised skeleton action segmentation is the closest mechanism neighbor: it generates prototype-similarity pseudo-labels with clustering from one labeled frame per segment~\citep{pointsup2026}; our combination differs in rule-engine seed anchors, confidence-filtered iteration, and the taxonomy-transition cost framing---we claim firstness for that combination, not for prototype pseudo-labels per se."),
# 14) 首次性证据句诚实化 (L56)
(r"Our claim of firstness rests on a systematic GitHub-scale survey covering ten query groups and fourteen candidate works with zero occupancy (full survey matrix in Supplementary Material; arXiv/Google Scholar re-verification is scheduled before submission)",
 r"Our claim of firstness rests on a repository-scale survey covering ten query groups and fourteen candidate works with zero occupancy (full matrix in Appendix~\ref{app:novelty}), complemented by an arXiv/CrossRef re-verification at submission time that surfaced point-supervised skeleton segmentation~\citep{pointsup2026} as the nearest neighbor---differentiated above"),
])

fix("sections/05-ablation-analysis.tex", [
# 15) tab3 行 7 校正 p
(r"2 clips/class: 31.96 vs.\ 26.07, all ten seed pairs, Wilcoxon $p=0.002$; 4: 31.43 vs.\ 24.11, $p=0.016$; full: 33.93 vs.\ 30.36); macro-F1 is at parity (14.36 vs.\ 14.78, $p=0.77$)",
 r"2 clips/class: 31.96 vs.\ 26.07, all ten seed pairs, Wilcoxon $p=0.002$ (Holm-corrected $0.012$); 4: 31.43 vs.\ 24.11, corrected $0.047$; full: 33.93 vs.\ 30.36); macro-F1 is at parity (14.36 vs.\ 14.78, corrected $0.77$)"),
# 16) §5 段 校正 p
(r"$+5.9$\,pp at 2 clips/class (all ten same-seed pairs positive, Wilcoxon $p=0.002$), $+7.3$\,pp at 4 ($p=0.016$)",
 r"$+5.9$\,pp at 2 clips/class (all ten same-seed pairs positive, Wilcoxon $p=0.002$; Holm-corrected $p=0.012$), $+7.3$\,pp at 4 (corrected $p=0.047$)"),
# 17) 1.62→1.61
(r"Prototype purity drops from 0.5339 to a 0.3306 random-assignment baseline ($1.62\times$)",
 r"Prototype purity drops from 0.5339 to a 0.3306 random-assignment baseline ($1.61\times$)"),
])

fix("sections/06-conclusion-limitations.tex", [
# 18) 结论等价措辞
(r"accuracy statistically equivalent at $-0.91$\,pp full / $+2.27$\,pp small, both $<2.3$\,pp",
 r"accuracy differences inside the pre-registered $<2.3$\,pp noise band ($-0.91$\,pp full / $+2.27$\,pp small)"),
# 19) Ten→Eleven + L11
(r"Ten limitations bound our claims.",
 r"Eleven limitations bound our claims."),
(r"We present the curve as a whole and make no claim about pretraining value outside the measured tiers.",
 r"We present the curve as a whole and make no claim about pretraining value outside the measured tiers." + "\n\n" +
 r"\paragraph{L11: E9's retention cannot separate filtering from pool size.}" + "\n" +
 r"At NTU scale the confidence gate accepts $\approx$95\% of the unlabeled pool, so the 99.5\% retention characterizes the pipeline's budget behavior under pseudo-label restoration and not the contribution of filtering per se; the 10\% labeled subset is a single stratified draw, so the reported spread covers self-training seeds but not subset resampling; and pool precision is not evaluated against ground truth (none exists on NTU)."),
])
print("R12 batch 1 done")
