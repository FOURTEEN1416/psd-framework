# P0.3 Phase B 开工任务书

> 窗口: W13（新建）或并入现有空闲窗口
> 依据: stage-plan "Phase B 22 类映射待路径 a 合成数据" + ADR-0002 裁决②
> 前置条件: ✅ 合成数据已就位 (`data/synthetic/syn_22class_100per_class_seed42.pkl`, 2200 样本)
>            ✅ P0.3 Phase A 已达标 (purity 0.534)
>            ✅ assets-map.md §1 登记 22 类权威清单

## 任务目标（一句话）

将 P0.3 Phase A 产出的 22 类行为原型聚类结果，映射到 ADR-0002 裁决②锁定的 22 类体系（Y），产出带 22 类语义标签的伪标签池，供 P0.5 微调实验消费（替代纯合成标签）。

## 输入

1. P0.3 Phase A 产物：原型聚类结果 + 种子伪 GT 消费接口（`psd/data/rule_seeds.py` 只读 import）
2. 合成数据集：`data/synthetic/syn_22class_100per_class_seed42.pkl`（2200 样本，22 类×100）
3. 22 类权威清单：`docs/assets-map.md` §1（禁止另抄一份）

## 关键设计约束

- Phase B **不改** Phase A 代码（只读 import）
- 映射逻辑写入新模块 `psd/training/jia_phaseB_mapper.py`
- 伪标签池格式与 P0.4 消费接口对齐（参照 P0.4 移交池 `data/processed/p04/pseudo_pool_main_consensus_a1.0_seed42.jsonl`）
- 22 类 → 22 类保持一一映射（不合并，E6 双贴合是另一条消融线）

## 交付物

1. `psd/training/jia_phaseB_mapper.py` — 映射器实现
2. `psd/training/tests/test_jia_phaseB_mapper.py` — TDD（至少 5 绿）
3. `reports/p03-jia-phaseb-<日期>.md` — 报告（含映射准确率、与 Phase A purity 关系、与合成标签对照）
4. `data/processed/p03_phaseB/` — 伪标签池产物（gitignore）
5. stage-plan P0.3 行状态列回写 ✅

## 完成标准

- [ ] 伪标签池覆盖率 ≥ Phase A coverage(α=1)
- [ ] 伪标签池精度（vs 合成真标签）≥ 0.50（Phase A purity 0.534 的合理下界）
- [ ] 端到端一条命令复现
- [ ] 报告归档 + Conventional Commits 中文提交

## 风险提示

- 若映射准确率显著低于 Phase A purity（差距 >10pp），需诊断原型-类别对齐质量而非简单扩大数据
- 与本仓其他窗口的领地互斥：禁触 `*smq*` / `*p05*` / `docs/paper/**` / `dev-docs/decisions/**`
