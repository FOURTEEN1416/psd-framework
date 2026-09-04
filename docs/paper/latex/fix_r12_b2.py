# -*- coding: utf-8 -*-
"""R12 修复批 2: bib 新增/修正 + main.tex highlights/DA/附录 + highlights.tex + README。"""
from pathlib import Path

def fix(path, pairs, append=None):
    p = Path(path); t = p.read_text(encoding="utf-8")
    for old, new in pairs:
        assert t.count(old) == 1, f"{path}: {t.count(old)}x :: {old[:60]}"
        t = t.replace(old, new)
    if append:
        t = t.rstrip("\n") + "\n" + append
    p.write_text(t, encoding="utf-8"); print(f"OK {path}")

# ---------- refs.bib: 6 新增（arXiv API 实证作者/题名）+ 3 题录修正 ----------
new_entries = """
@article{pointsup2026,
  author = {Wang, Hongsong and Shen, Yiqin and Yan, Pengbo and Gui, Jie and Zhang, Yang},
  title = {Point-Supervised Skeleton-Based Human Action Segmentation},
  journal = {arXiv preprint arXiv:2603.06201},
  year = {2026}
}

@article{primateda2025,
  author = {Mueller, Felix B. and Lueddecke, Timo and Vogg, Richard and Ecker, Alexander S.},
  title = {Domain-Adaptive Pretraining Improves Primate Behavior Recognition},
  journal = {arXiv preprint arXiv:2509.12193},
  year = {2025}
}

@article{beast2025,
  author = {Wang, Yanchen and Yu, Han and Blau, Ari and Zhang, Yizi and {The International Brain Laboratory} and Paninski, Liam and Hurwitz, Cole and White, Matthew E.},
  title = {Animal Behavioral Analysis and Neural Encoding with Transformer-Based Self-Supervised Pretraining},
  journal = {arXiv preprint arXiv:2507.09513},
  year = {2025}
}

@inproceedings{mabe22,
  author = {Sun, Jennifer J. and Marks, Markus and Ulmer, Andrew and Chakraborty, Dipam and Geuther, Brian and Hayes, Edward and Jia, Heng and Kumar, Vivek and others},
  title = {{MABe22}: A Multi-Species Multi-Task Benchmark for Learned Representations of Behavior},
  booktitle = {NeurIPS Datasets and Benchmarks},
  year = {2022}
}

@article{skeletonfm2025,
  author = {Wang, Hongsong and Weng, Wanjiang and Wang, Junbo and Zhao, Fang and Xie, Guo-Sen and Geng, Xin and Wang, Liang},
  title = {Foundation Model for Skeleton-Based Human Action Understanding},
  journal = {arXiv preprint arXiv:2508.12586},
  year = {2025}
}

@article{skeletonmae2022,
  author = {Wu, Wenhan and Hua, Yilei and Zheng, Ce and Wu, Shiqian and Chen, Chen and Lu, Aidong},
  title = {{SkeletonMAE}: Spatial-Temporal Masked Autoencoders for Self-supervised Skeleton Action Recognition},
  journal = {arXiv preprint arXiv:2209.02399},
  year = {2022}
}
"""
fix("refs.bib", [
# ASBAR 题录按 eLife 官方
("  title = {{ASBAR}: Unified Skeleton-Based Animal Action Recognition},",
 "  title = {{ASBAR}: An Animal Skeleton-Based Action Recognition Framework. Recognizing Great Ape Behaviors in the Wild Using Pose Estimation},"),
# BCST-GCN 题录按 Crossref
("  title = {{BCST-GCN}: Bidirectional Cross-Attention Graph Convolution for Pig Behavior Recognition},",
 "  title = {{BCST-GCN}: A Skeleton-Based Spatiotemporal Graph Convolutional Network with Bidirectional Cross-Attention for Pig Behavior Recognition},"),
# Grimm 题录按 Crossref 登记
("  title = {Can Science Build a Better Working Dog?},",
 "  title = {Building a Better Working Dog},"),
], append=new_entries)

# ---------- main.tex: highlights 两条加 synthetic 限定 + DA 措辞 + 附录启用 ----------
fix("main.tex", [
(r"\item Taxonomy transitions absorbed at $\geq$3x lower wall-clock cost, matched accuracy",
 r"\item Taxonomy transitions cost $\geq$3x less wall clock on a synthetic benchmark"),
(r"\item Warm start makes a 20-clip budget usable: 82.0\% top-1 under shift",
 r"\item Warm start makes a 20-clip budget usable: 82.0\% top-1 (synthetic shift)"),
(r"All reported numbers are reproducible from archived configurations and one-command chains in the public repository (github.com/FOURTEEN1416/psd-framework, tag \texttt{review-snapshot}, the immutable review snapshot)%",
 r"All reported numbers are reproducible from archived configurations, released scripts, and the reproduction appendix in the public repository (github.com/FOURTEEN1416/psd-framework, tag \texttt{review-snapshot}, the immutable review snapshot); external datasets and pretrained backbones must be obtained under their own licenses, and derived skeletons regenerate from provider data via the released scripts%"),
(r"""%% ---------------- 附录（计入 20-35 页窗口! 超页时移 Supplementary Material）----------------
%% \appendix
%% \section{One-command reproduction chain}\label{app:repro}   % truth: reports/p01-aimclr-2026-08-23.md §7 等
%% \section{Novelty survey matrix}\label{app:novelty}          % truth: dev-docs/research/NOVELTY_CHECK_YAOQING_JIA.md""",
 r"""%% ---------------- 附录（R12 启用: 首次性证据矩阵 + 复现链, 消除"Supplementary Material 不存在"指控）----------------
\appendix
\section{Novelty survey matrix}\label{app:novelty}

Table~\ref{tab:novelty} records the repository-scale survey behind the firstness claim (Section~\ref{sec:related}): ten query groups over GitHub code search plus a full scan of the maintained awesome-skeleton-AR catalog (19 candidate entries), each candidate checked against the three-part mechanism (self-supervised skeleton encoder $\to$ rule-engine seed anchors $\to$ clustered, confidence-filtered iterated pseudo-labels). The submission-time arXiv/CrossRef re-verification added the point-supervised row.

\begin{table}[t]
\centering
\caption{Novelty survey matrix. ``Occupies niche?'' = does the work instantiate all three mechanism parts for temporal skeleton recognition.}
\label{tab:novelty}
\small
\begin{tabularx}{\linewidth}{p{3.1cm}p{2.2cm}X p{1.2cm}}
\hline
Candidate & Domain / paradigm & Difference from our combination & Niche? \\
\hline
MAC-Learning (TPAMI 2022) & human skel., anchor contrastive & anchors are contrastive sample pairs; no seed-annotated anchors, no cluster pseudo-label loop & no \\
Momentum-contrastive teacher (TIP 2025); GRA (TNNLS) & human skel., semi-sup. & consistency/alignment regularization; no anchors, no taxonomy framing & no \\
PSP-Learning / X-Invariant / Decouple-and-Squeeze / joint-bone fusion (2022) & human skel., semi-sup. & NTU consistency paradigms; none couples seeds+clustering+iteration & no \\
PAINet (ICCV 2023); HAA4D; ISBFSAR; FICAMA; SMAM; UMEG-Net; SkelHCC & human skel., few/zero-shot & episode/metric learning; no pseudo-label expansion & no \\
Skeleton-to-image (arXiv 2603.05963) & human skel., VFM features & representation only; no label-expansion loop & no \\
TP-CanineNet (Animals 2025) & canine \textbf{RGB} video, pseudo-labels & video domain, no skeleton, no anchor clustering & no \\
SMQ (ICCV 2025); hierarchical ST-VQ & unsup.\ skeleton segmentation & quantizes motion; no taxonomy semantics & no \\
Point-supervised skeleton segmentation (arXiv 2603.06201) & human skel., prototype pseudo-labels & one labeled frame per segment; no rule-engine anchors, no confidence-filtered iteration, no transition-cost framing & no \\
\hline
\end{tabularx}
\end{table}

\section{Reproduction chain}\label{app:repro}

Data acquisition (provider licenses required): InterPet4D (Hugging Face \texttt{ohicarip/interpet4d}, smal\_npy), Animal Kingdom action-recognition lists + videos (CVPR 2022 release), APTv2 pose pool (official repository), NTU RGB+D 60 frame-50 export (research-use license). Then, in order: \texttt{scripts/export\_interpet4d.py} (p01 views) $\to$ physics pretext \texttt{scripts/train\_aimclr.py} $\to$ \texttt{scripts/eval\_aimclr.py} (E1) $\to$ SMQ \texttt{scripts/train\_smq\_segmentation.py} + \texttt{eval\_smq\_segmentation.py} + \texttt{diag\_p02\_motion\_words.py} (E2) $\to$ \texttt{scripts/run\_p03\_phasea.py} (E3) $\to$ \texttt{scripts/run\_p04\_tcl.py} (E4) $\to$ \texttt{scripts/run\_p05\_al\_warmstart.py} (warm-start) $\to$ \texttt{scripts/run\_p05\_public\_real\_full12.py} \texttt{--stage manifest|extract} + \texttt{run\_p05\_public\_real\_full12\_train.py} (v1 tier; dataset paths configurable via the constants block) $\to$ \texttt{run\_p07\_endtoend\_ak.py} (E7/E8) $\to$ \texttt{run\_p08\_aimclr\_arm.py}, \texttt{run\_p10\_seedexpansion.py} (ablation) $\to$ \texttt{run\_p11\_ak\_v2\_build.py}, \texttt{run\_p12\_ak\_v2\_replicate.py}, \texttt{run\_p13\_v2\_finetune.py} (E7b) $\to$ \texttt{run\_p14\_ntu\_featuredump.py}, \texttt{run\_p14\_ntu\_lowres.py}, \texttt{run\_r12\_artifacts.py} (E9, statistics). NTU equivalence: \texttt{run\_ntu\_phaseb.py} + \texttt{run\_ntu\_lineareval.py} + \texttt{ntu\_ensemble\_3s.py}. Every report JSON cited in the text is committed under \texttt{reports/}."""),
])

# ---------- highlights.tex 同步 ----------
fix("highlights.tex", [
(r"\item Taxonomy transitions absorbed at $\geq$3x lower wall-clock cost, matched accuracy",
 r"\item Taxonomy transitions cost $\geq$3x less wall clock on a synthetic benchmark"),
(r"\item First anchor-cluster-pseudo-label transfer to temporal skeleton recognition",
 r"\item First, to our knowledge, anchor--cluster pseudo-label transfer to skeletons"),
(r"\item Warm start makes a 20-clip budget usable: 82.0\% top-1 under shift",
 r"\item Warm start makes a 20-clip budget usable: 82.0\% top-1 (synthetic shift)"),
])
print("R12 batch 2 done")
