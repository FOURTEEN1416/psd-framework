# W12 复核 Checklist

> Owner: `dev-docs/w12-review-checklist.md`
> 触发时机: W12 提交完成、GPU 空闲后执行
> 原则: 无新鲜验证，无完成声明

---

## 1. 实验数字新鲜度核验（逐项复跑）

- [ ] **1a. P0.5 Y (22类, 100样本/类)**: 复跑 `scripts/run_p05_full.py --config configs/p05_stgcn_bc_full.yaml` → 比对 `reports/p05-stgcnbc-synthetic-100perclass-Y.json` 中 `summary.best_val_acc` 是否 ≥0.97
- [ ] **1b. P0.5 E6 Y' (21类, 100样本/类)**: 复跑 `scripts/run_p05_full.py --config configs/p05_e6_taxonomy.yaml` → 比对 `reports/p05-stgcnbc-synthetic-100perclass-Yprime.json` 中 `summary.best_val_acc` 是否 ≥0.95
- [ ] **1c. P0.5 消融链**: 复跑 20/50 样本版本，确认 77.3%/95.0% 趋势成立
- [ ] **1d. 随机基线自检**: 每个实验独立计算随机基线（22类=4.5%, 21类=4.76%），确认报告数字一致

## 2. 日志-JSON 一致性校验

- [ ] **2a.** `runs/p05_stgcn_bc_full/history.json` 每 epoch val_acc ≤ 对应 report JSON 的 best_val_acc
- [ ] **2b.** `runs/p05_stgcn_bc_Yprime/history.json` 同上
- [ ] **2c.** best.pt 时间戳 ≥ history.json 最后条目时间戳
- [ ] **2d.** checkpoint 可正常加载（`torch.load + model.load_state_dict` 不报错）

## 3. 测试回归守卫

- [ ] **3a.** 全量 pytest `psd/` 目录 → 必须 172 passed（W11 基线）
- [ ] **3b.** W12 新增测试（如有）→ 全绿
- [ ] **3c.** 无未提交文件在 W12 白名单内（`git status --short` 筛查）

## 4. 三层口径合规扫描

- [ ] **4a.** 合成层数字（E1-E4）与公开真实层数字（P0.1-P0.4）在报告文档中分表呈现，无混排
- [ ] **4b.** 每个数字标注层级来源（`metric_layer` 字段或报告正文声明）
- [ ] **4c.** 无跨层对比表格（如"合成层 97% vs 公开真实层 20%"同表并列）

## 5. E6 双贴合映射正确性抽验

- [ ] **5a.** Y→Y' 映射：原始 idx=2(stand) 和 idx=8(track) 的所有样本在 Y' 中均为 idx=2(locomotion)
- [ ] **5b.** Y' 类别数 = 21（22-1），验证 `len(Y_PRIME_LABEL_NAMES) == 21`
- [ ] **5c.** 抽样 3 个 Y' 样本，打印其原始 label_name 验证合并正确性

## 6. 一条命令复现验证

- [ ] **6a.** 完整复现序列可独立运行（从 config 到 report JSON），无需人工干预
- [ ] **6b.** 复现输出与已归档 JSON 数字一致（±0.001 容差）

## 7. GPU 进程清理确认

- [ ] **7a.** 执行复核前确认无残留 python 训练进程（`nvidia-smi` memory.used 回落至 <2GB）
- [ ] **7b.** 若发现 W12 进程未完全退出，先等待或温和终止再复核

---

## 复核通过标准

全部勾选项打 ✓ → P0.5 合成层验收通过，可进入论文回填阶段。

任何 ✗ → 记录问题清单，逐项修复后重跑相关项，不整体重跑。

---

*编制: 歆歆 2026-08-24 · 依据: AGENTS.md 无新鲜验证无完成声明 / ADR-0004 走法 A*
