# Pre-registered protocol: filtering-vs-pool-size decomposition on NTU60 E9 harness (EXP-B)

> **Protocol ID**: PSD-POOLDECOMP-PREREG-001 | **Status**: **FROZEN v1.0 2026-09-07**（用户令"先补实验"启动；冻结先于任何运行）
> **Origin**: R23 对抗审稿 I1-adjacent + L11 自认缺口——"this arm cannot separate filter quality from pool-size restoration"。本协议用 2×2 因子设计把两个因子正交分解。
> **驱动非本协议变更**: 全部复用 E9 冻结特征（`runs/ntu_lowres/features_joint_ep300.npz`）与 r16 修正协议，唯一新变量是 pool 构成方式。

## 1. Harness (frozen)

与 E9 完全同源：epoch-300 joint pretext 的冻结 256-d 特征；同一 10% 分层子集（seed 42）；同一评估（arm 级 top-1 on val）；自训练种子 42–51（n=10）。

## 2. Arms (frozen — 2×2 因子)

| Arm | 过滤 | Pool 规模 | Pool 构成 |
|---|---|---|---|
| F1-Pfull | **无过滤（全收）** | 全量 | 10% 标签 + **全部** ≈35.9k 未标注 clips（随机/置信无关伪标签） |
| F1-Psel | **无过滤** | 选择性 | 10% 标签 + 随机抽取 ≈35.4k 中 98% 等量的**随机**子集（规模匹配 F0-Psel，置信无关） |
| F0-Pfull | **置信过滤（r16 原门）** | 全量 | 10% 标签 + 全部 ≈35.9k clips 走 r16 原置信门（≈98% 接受） |
| F0-Psel | **置信过滤** | 选择性 | = **E9 原臂**（r16 协议复刻，≈35.4k 池） |

- **伪标签来源统一**：全部 arms 的 pool 伪标签由同一轮 anchor-guided 聚类产生（与 r16 相同管线），唯一操作差异是入池前是否过置信门/是否随机下采样。F1（不过滤）臂的 pool 标签 = 同一聚类输出**不过门直接采用**——即"如果接受率 100% 会怎样"。
- F1-Psel 的随机子集 seed 固定 42（一次抽取，与子集抽取惯例一致，disclosed）。

## 3. Endpoint and decision rule (frozen)

- **主终点**：四臂 val top-1（n=10 seeds）的 2×2 ANOVA 式读数——过滤主效应 = F0 均值 − F1 均值；规模主效应 = Pfull 均值 − Psel 均值；交互项同报。
- **判读（写入协议防事后）**：若 |规模主效应| ≥ 3×|过滤主效应| → "pool 规模是主导因子"成立（支持 L11 的机制读法）；若过滤主效应 ≥ 规模主效应 → L11 的"cannot separate"降级为"can now separate, and filtering matters"。
- **无筛选上报**：无论方向，四臂全报。Wilcoxon 各臂 vs F0-Psel（E9 原臂）只作方向参考，不做显著性主张（n=10 探索性诊断，非确认性检验——如实标注）。

## 4. Disclosures (frozen)

- 本实验是 post-hoc 机制诊断（注册于全部 E9 系结果可见之后），**不是**预注册确认性检验；论文引用时必须带此限定。
- F1 臂不过门即用伪标签，预期精度显著受损（pool 伪标签含 ~2% 噪声 + 低置信样本），该臂结果是机制读数的上界探针。
- 冻结特征/子集/评估与 E9 逐位同源，除 pool 构成外零自由度。

## 5. Evidence

Driver: `scripts/run_r23_pool_decomp.py`；evidence: `reports/r23-pool-decomp-<date>.json`。

## 修订历史

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-09-07 | FROZEN：用户令"先补实验"启动；2×2 因子设计 + 判读规则冻结。 |
