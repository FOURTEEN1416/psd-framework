# Pre-registered protocol: PanAf500 seed expansion to n=30 (Amendment 2 of PSD-PANAF-PREREG-001)

> **Status**: **FROZEN 2026-09-07**（用户令"不要随便省去实验"；冻结先于运行）
> **Origin**: 恶意审稿 A10-adjacent + 正文自注——PanAf gap "+1.2pp, 5/10 wins" 未做显著性检验（臂分布与线性臂打平）。n=10 时 seed 方差 ±9.1pp 主导，检验功效不足。扩至 n=30 提升功效，**本修正案冻结于看到任何 30-seed 结果之前**。

## 1. Amendment content (frozen)

- 自训练种子从 42–51（n=10）扩至 **42–71（n=30）**；arms/budget/subset/评估与原协议逐项不变（10% stratified subset 仍为 seed-42 单次抽取；spread 语义不变：覆盖自训练种子，不覆盖子集重抽）。
- E9 系列惯例（3→10）此刻顺势延至 30：**全部四 tier 是否同步扩至 30？** 否——本修正案仅针对 PanAf（唯一 gap 方向性存疑的 tier）。其余三 tier 的结论已有显著性检验支撑（NTU60/NTU120 显著、UCF101 如实 n.s.），无扩容动机。此非对称扩容的理由如实入文（防"只给弱臂加种子"指控：PanAf 是唯一未检验的臂；扩容后若结论翻转将如实降格）。

## 2. Decision rule (frozen)

- n=30 Wilcoxon（含打平，zero_method 默认 wilcox）双侧 p：
  - p < 0.05 且 Holm-8 校正后仍 < 0.05 → PanAf gap 升格为"statistically supported (exploratory family)"，正文从 "directional only" 改写；
  - 校正后 ≥ 0.05 → 维持 "directional only" 并**补功效说明**（"n=30 仍不显著，该层的 gap 读数受限于长尾九类方差"）；
  - 方向翻转（中位数 < 0）→ 如实降格为 "no pipeline advantage on this tier"，摘要/结论相应回改。
- 族归属：PanAf 依旧不进 eight-test family（ties 结构不变）；本检验作为该 tier 的独立探索性检验报告，标注 exploratory。

## 3. Evidence

Driver: `scripts/run_p23_panaf_retention.py --seeds 30`（已参数化，协议零代码变更）；evidence: `reports/p23-panaf-retention-30seed-<date>.json`。

## 修订历史

| 版本 | 日期 | 说明 |
|---|---|---|
| A2 | 2026-09-07 | FROZEN：n=10→30 扩容（seeds 42–71），判据三向；仅 PanAf，理由冻结。 |
