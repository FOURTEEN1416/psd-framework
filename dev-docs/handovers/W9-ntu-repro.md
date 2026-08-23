# W9 交接文档 — NTU 复现验证（实现正确性防线 · Phase A 可立即开工）

> **你是 W9 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.4 → `dev-docs/decisions/0002-user-rulings-ntu-synthetic-e6.md` 裁决 1 → `reports/p01-aimclr-2026-08-23.md` §2 口径披露。
> **任务性质**：证明本仓的 AimCLR 适配实现与官方等价（重实现类论文的审稿防线，风险登记册 R4）。**两相拆分：Phase A 纯 CPU 立即做；Phase B 需 GPU 等 P0.2 救援释放后再跑。**

---

## 1. 任务目标（一句话）

在 NTU60 上用本仓适配管线复现 AimCLR 原文参照成绩（79.18%，协议细节待从原文核实），偏差在合理范围内即证明实现正确，随后该结论写入论文 experiment-skeleton §4.4。

## 2. Phase A — 数据获取与预处理（纯 CPU，立即执行）

1. **读官方指引**（只读）：`external/AimCLR` 的 README / 数据准备文档，确认其期望的数据格式、预处理脚本与下载来源。
2. **数据获取**（按官方渠道优先级）：
   - 官方指定渠道若需申请表单（NTU 原始数据有授权协议）→ 停下上报用户，用机构邮箱申请；
   - 若 AimCLR 官方提供预处理版下载链接 → 直接使用并记录出处；
   - ⚠️ AGENTS.md 禁 WebSearch：找链接只用 GitHub 工具链（AimCLR 仓 README/issues 里找官方指向）+ 用户提供的渠道。
3. **磁盘预算**：骨架版约 6GB 量级 + 预处理产物，开工前 `Get-PSDrive` 确认余量 >20GB。
4. **预处理**：跑官方预处理流程产出 AimCLR 输入格式；落盘路径登记进报告（数据本体 gitignore）。
5. **协议核实**：从原文（arXiv:2104.10213 待核，以 external/AimCLR README 引用为准）确认 79.18% 的评估协议（xsub/xview、kNN or linear probe、backbone 特征口径），写进报告——related-work.md 已挂"协议待核"标记，你的核实结果回填给它。

## 3. Phase B — 复现训练与对照（GPU，等 P0.2 释放后）

- 排程协调：开跑前在 HANDOVER §8 或会话记录声明占卡区间，避免与 P0.2/W8 冲突。
- 用本仓适配链（`psd/training/p01_processor.py` 同源的适配思路 + 官方超参配置）在 NTU60 xsub 上训练。
- **验收判据（预注册，防事后挪门柱）**：复现值 ≥ 参照值 −2 个百分点（即 ≥77.18%）视为实现等价通过；低于此线 → 报告差异分析，不得静默调参凑数（一次系统调参机会，结果无论好坏如实归档）。
- 双口径纪律：公开基准层结果单独汇报，禁止与 InterPet4D 结果混表。

## 4. 边界

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `scripts/*ntu*`、`configs/*ntu*`、`psd/data/*ntu*` 新文件、`reports/ntu-*`、NTU 数据目录（gitignore）、本报告 |
| ❌ 禁触 | `external/**` 内部实现（只读）、一切 `*smq*`（W4 重启领地）、`*jia*/*p03*`（W8）、`docs/paper/**`（只读引用）、`.venv` 只读 import |
| GPU | Phase B 开跑前确认 P0.2 已释放（看 HANDOVER §8 状态或询问用户） |

## 5. 完成标准

- [ ] Phase A：数据就绪 + 协议核实结论 + 出处记录
- [ ] Phase B：复现数值 JSON/md 归档 + 与参照对照表 + 判据结论
- [ ] 回填：`experiment-skeleton.md` §4.4 行状态 + `related-work.md` 协议标记解除
- [ ] 一条命令复现序列 + 中文 Conventional Commit

## 6. 升级路径

- 官方渠道全部需要申请且周期不可控 → 上报用户裁决（等待申请 vs 换公开镜像 vs 降级为"官方 checkpoint 直接 kNN 对照"方案）
- 磁盘不足 → 上报，不自作主张清理任何现有目录

---

*交接编制: 歆歆（规划会话）2026-08-24 · 依据: ADR 0002 v1.1 裁决 1*
