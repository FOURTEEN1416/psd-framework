# Pre-registered protocol: published-method retention anchor (AimCLR++ checkpoint, EXP-D)

> **Protocol ID**: PSD-EXTBASE-PREREG-001 | **Status**: **FROZEN v1.0 2026-09-07**（冻结先于运行）
> **Origin**: 恶意审稿 S2 + L13——全文零发表方法对比。本协议给出一个**发表方法检查点**在同一 10% 预算协议下的保留率锚点，使论文的管线数字首次有一个可直接引用的公开参照。**这是锚点不是 SOTA 比较**（骨干/数据口径差异如实披露）。

## 1. Published method & checkpoint (frozen)

- **AimCLR++**（aimclrpp2024, Pattern Recognition 2024）——官方仓库 `Levigty/AimCLR-v2` 发布的 NTU60 xsub joint 预训练检查点（与本文 physics layer 同源方法族的 journal 扩展；正文 §2 已引其 80.9% 官方线性评估数）。
- 下载路径与文件 SHA-256 记录入 evidence JSON（checkpoint provenance 强制）。

## 2. Protocol (frozen)

- 用该检查点的 joint encoder 对 NTU60 xsub 的同一 frame-50 导出抽特征（与 `run_p14_ntu_featuredump.py` 同一评估前处理：T=20 clips、同一 40,091/16,487 切分），冻结后走 **E9 的 (a)/(c) 臂**：(c) full-budget linear-probe 参考 + (a) 10% 线性臂。
- **不含 (b) 臂**（不在 AimCLR++ 特征上跑我们的语义管线——那是另一个问题；本协议只回答"发表方法的冻结特征在本协议下保留率是多少"）。
- 判读：无 pass/fail——纯描述性锚点，报 (c) 参考、(a) 10% 值与 retention=(a)/(c)。与 E9 的 (a) 66.05%、(c) 74.45% 并排呈现。

## 3. Disclosures (frozen)

1. AimCLR++ 预训练于 NTU60 xsub **train split 的全部 40,091 clips**（与我们的 pretext 同数据——公平）；其输入为 joint 骨架序列，frame-50 导出与官方 preprocessing 的差异（若导出格式转换引入任何重采样）在 evidence JSON 中披露。
2. 该锚点**不是**方法比较：我们的 pretext 300ep 自监督目标与 AimCLR++ 不同（extreme-augmentation vs ++ 变体），retention 差异不能归因优劣，只能并排引用。
3. 若官方仓库无 NTU60 xsub joint 检查点或下载失败，本协议 FAIL-TO-RUN 如实记录，不入文（不留空承诺）。

## 4. Evidence

Driver: `scripts/run_r23_extbase_anchor.py`（待写；特征提取复用官方 linear_eval 权重加载路径）；evidence: `reports/r23-extbase-anchor-<date>.json`。

## 修订历史

| 版本 | 日期 | 说明 |
|---|---|---|
| v1.0 | 2026-09-07 | FROZEN：AimCLR++ checkpoint → E9 (a)/(c) 臂描述性锚点。 |

## Amendment 1 (2026-09-07, dated, frozen before any download completed)

Investigation of `Levigty/AimCLR-v2` found **no released AimCLR++ checkpoint** (the journal repo has no model-release section; its README points only to dataset mirrors). Per the FAIL-TO-RUN clause, the anchor target is switched to the **official released AimCLR (AAAI 2022) NTU60 xsub joint checkpoint** from the conference repo's `released_model` folder (Google Drive, file id `14rayAgGWAHFL-JdeCJQOwP4cJ-fHglJS`, recorded at download time with SHA-256). Rationale: the anchor's purpose is "a published method's public checkpoint under our harness" — the conference AimCLR is the direct ancestor of our physics layer and equally qualifies; the AimCLR++ inaccessibility is disclosed verbatim in the evidence JSON. The paper's §2/§4 wording must say "official released AimCLR checkpoint" (not AimCLR++).
