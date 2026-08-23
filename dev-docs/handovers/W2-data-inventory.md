# W2 交接文档 — 数据深盘点（数量级复核）

> **你是 W2 窗口**。读完本文档即开工，无需等待其他窗口。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` §5（数据资产）→ `docs/DATA_LOCATIONS.md`（回填目标）。

---

## 1. 任务目标（一句话）

复核三大数据集的**声明数量**（来自 K9 truth、本仓未验证），把实数回填 `docs/DATA_LOCATIONS.md`，解除「未逐一复核」警告。

## 2. 复核清单

| # | 数据集 | 声明值 | 复核方法 |
|---|--------|--------|---------|
| 1 | InterPet4D `smal_npy/` | 226 序列 | 先探明结构（.npy 直存还是序列子目录），再计数；抽查 2-3 个文件可被 `numpy.load` 正常读取且形状一致 |
| 2 | Animal Kingdom 犬科 | 338 视频 / 239 帧级标注 | 在 `action_recognition/` 标注 JSON 中过滤 canidae 类目计数；若标注结构复杂，先读其 README |
| 3 | APTv2 全量 | 242K 文件 | `Get-ChildItem -Recurse -File \| Measure-Object` 计数（量大，注意超时，可只统计顶层分布+总数） |

**附带产出**：smal_npy 单文件骨架维度（T×J×C 具体值）——P0.1 加载器直接需要，一并写入报告。

## 3. 铁律

1. **K9 仓数据绝对只读**：`D:\Desktop\k9-training-system\data\*` 禁止搬移/改名/删除/修改，只允许读取统计
2. 统计命令原始输出归档证据，不允许"目测估计"
3. 实数与声明值不符时：如实记录差异，不猜测谁对，交用户裁决

## 4. 边界(并行窗口互斥,严格执行)

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `docs/DATA_LOCATIONS.md`（数量列 + 回填记录表 + 删除头部「⚠️ 未复核」警告行）、`dev-docs/project-brief.md` §4 过期警告行、`dev-docs/stage-plan.md` 启动前置 #2 勾选为 ✅、`reports/data-inventory-2026-08-23.md`（新建，证据归档） |
| ❌ 禁触 | `psd/`、`scripts/`、`configs/`、`external/`、`dev-docs/research/`、`PAPER_POSITIONING.md`、stage-plan 其他任何行 |

> **stage-plan 编辑纪律**：用精确文本替换只改前置 #2 那一行的勾选状态，禁止重写全文或改动其他章节（W3 窗口稍后会改 P0.1 行状态，避免踩踏）。
> 编辑 project-brief/stage-plan 前先重新 Read 最新内容再 Edit。

## 5. 完成标准与 Git

- [ ] 三项数量全部有实数 + 统计命令输出存于 `reports/data-inventory-2026-08-23.md`
- [ ] smal_npy 骨架维度实测值写入报告
- [ ] DATA_LOCATIONS.md 数量列更新 + 回填记录表新增一行
- [ ] 两处过期漂移已修（brief §4 / stage-plan 前置#2）
- [ ] 提交：`git add docs/DATA_LOCATIONS.md reports/data-inventory-2026-08-23.md dev-docs/project-brief.md dev-docs/stage-plan.md && git commit -m "docs: 数据深盘点完成——<核心差异摘要>"`
- [ ] 遇 `index.lock` 冲突等待重试；禁 push

## 6. 卡住升级

标注 JSON 结构无法解析出犬科计数 → 如实记录结构发现，该项标「需 K9 侧口径确认」，不要阻塞其他两项；整体受阻 → 向用户报告。
