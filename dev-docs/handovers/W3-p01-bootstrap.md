# W3 交接文档 — P0.1 AimCLR 预训练开工

> **你是 W3 窗口**。读完本文档即开工，无需等待其他窗口。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` → `dev-docs/stage-plan.md` §1（P0.1 验收定义）→ `research/RESEARCH_DATA_BLOCKADE_SOLUTION.md`（原方案细节）。

---

## 1. 任务目标（一句话）

打通 P0.1 全链：**venv 建立 → PyTorch cu128 → 克隆 AimCLR 到 external/ → InterPet4D smal_npy 加载器 → 自监督预训练跑通 → kNN top-1 显著超随机基线**。

验收定义（stage-plan 已定）：`scripts/eval_aimclr.py --knn` + JSON 报告归档 `reports/`。

## 2. 执行链（顺序执行）

1. **环境**：Python 3.12 venv → 安装 PyTorch **2.11+cu128**（RTX 5060 = sm_120 架构，cu126 及以下会 CUDA 不可用——K9 已踩坑）；验证 `torch.cuda.get_device_capability()` 返回 `(12, 0)`
2. **克隆**：AimCLR 官方实现 → `external/`（gitignore 已排除内容，不入库）；**禁改其内部实现**
3. **加载器**：`psd/data/interpet4d.py`——读 smal_npy（骨架维度以 W2 盘点实测为准；若 W2 未完成，自行抽查一个 .npy 实测），适配 AimCLR 的 NTU 输入格式约定
4. **训练入口**：`scripts/train_aimclr.py` + `configs/p01_aimclr.yaml`
5. **评估**：`scripts/eval_aimclr.py --knn` → top-1 准确率
6. **归档**：训练日志关键段 + kNN 结果 JSON → `reports/p01-aimclr-<日期>.md`

## 3. 本阶段坑位提示（前人经验，直接绕开）

| 坑 | 应对 |
|----|------|
| sm_120 需要 cu128 | 见执行链第 1 步 |
| mamba_ssm 编译地狱 | **P0.1 用不到 Mamba，别装、别编译**——那是 P0.5 的事 |
| kNN 基线盲套 4.5% | ❌ 4.5% 是 K9 合成 22 类口径。InterPet4D 类别数以实际数据为准，随机基线 = 100/N%；「显著超基线」= 至少 2-3 倍于随机水平并在报告中给出对照表 |
| AimCLR 输入假设 NTU (T,25,3) | InterPet4D 关节布局不同——适配代码写在 `psd/data/`，通过配置映射关节索引，不改 external/ |
| 三层指标口径 | P0.1 数据 = InterPet4D = **公开真实层**，报告标题必须标注口径，禁止与 K9 合成层数字混报 |

## 4. 边界(并行窗口互斥,严格执行)

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `.venv/`（已 ignore）、`requirements.txt`、`external/*`（内容已 ignore）、`psd/data/*`、`psd/models/*`（仅当适配必需）、`psd/training/*`（仅当必需）、`scripts/*`、`configs/*`、`reports/p01-*.md`、`dev-docs/stage-plan.md` 仅限「P0.1 行状态列」改为 ✅（前置 #2 行归 W2 管，不要动） |
| ❌ 禁触 | `docs/DATA_LOCATIONS.md`、`dev-docs/project-brief.md`、`dev-docs/research/**`、`PAPER_POSITIONING.md`、`dev-docs/handovers/**` |

> 编辑 stage-plan 前先重新 Read 最新内容再 Edit，只做该行精确替换。

## 5. 完成标准与 Git

- [ ] GPU 能力实测 `(12, 0)` 记录在案
- [ ] 预训练 loss 曲线收敛证据（日志片段）
- [ ] kNN top-1 数值 + 随机基线对照（JSON 归档 reports/）
- [ ] 一条命令可复现：README 或报告内写明完整命令序列
- [ ] 提交（可分多个）：`feat: P0.1 ...`，白名单外文件一律不 add
- [ ] 遇 `index.lock` 冲突等待重试；禁 push

## 6. 风险预案与升级

- AimCLR 在狗骨架上不收敛 → stage-plan §2 预案允许换 SimCLR/BYOL，但**属方向变更：先在回复中向用户报告证据并等确认再换**
- 同一报错连续修复失败 3 次 → 停止局部修补，按 systematic-debugging 重开诊断并向用户报告
- W2 若已完成盘点，优先采用 `docs/DATA_LOCATIONS.md` 实测维度值
