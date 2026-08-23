# W4 交接文档 — P0.2 SMQ 时序分割

> **你是 W4 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md`（v1.2，§8 已指向本任务）→ `dev-docs/stage-plan.md` §1 → **W3 成果复用清单（§3）**。

---

## 1. 任务目标（一句话）

SMQ 无监督时序分割跑通：**克隆 SMQ 官方实现到 external/ → 适配 InterPet4D 输入 → episode 边界分割 → IoU 评估 + 可视化**。

验收定义（stage-plan 已定）：`scripts/eval_smq_segmentation.py --iou` + 报告归档 `reports/`。

## 2. 前置检查与执行链

1. **环境**：复用 W3 已建好的 `.venv`（torch 2.11+cu128 已锁定可用，勿重装）；若 SMQ 有额外依赖，增量安装进同一 venv 并更新 `requirements.txt`
2. **克隆**：SMQ（ICCV 2025）官方实现 → `external/`；**禁改其内部实现**，适配代码一律写在 `psd/`
3. **输入适配**：stage-plan 注明 SMQ 输入 (T,24,3) 与 InterPet4D smal_npy 直接兼容——以 W3 的加载器实测维度为准（见 §3）
4. **分割管线**：`psd/training/` + `scripts/train_smq_segmentation.py` + `configs/p02_smq.yaml`
5. **评估**：`scripts/eval_smq_segmentation.py --iou`（episode 边界 IoU）+ 分割可视化图
6. **归档**：`reports/p02-smq-<日期>.md`（含 IoU 数值 JSON + 可视化样本）

## 3. W3 成果复用清单（先读再用，勿重复造轮子）

| 资产 | 路径 | 用途 |
|------|------|------|
| InterPet4D 加载器 | `psd/data/`（TDD 9 passed） | 直接复用读 smal_npy |
| npz→NTU 视图导出器 | `psd/data/` | SMQ 输入格式适配参考 |
| 表征坍缩教训 | `reports/p01-aimclr-2026-08-23.md`（E1-E7 实验链） | ⚠️ 官方 weights_init(N(0,0.02)) 曾致 InfoNCE 不可逃逸坍缩——SMQ 若含类似初始化，警惕同类问题 |
| kNN 达标口径 | 随机基线=100/N%（InterPet4D 实际类别数），勿盲套 4.5% | IoU 无随机基线概念，但报告需标注「公开真实层」口径 |

## 4. 边界（白名单互斥）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `.venv/`（增量装包）、`requirements.txt`、`external/*`（SMQ 克隆）、`psd/data/*`（仅当适配必需）、`psd/models/*`、`psd/training/*`、`scripts/*`、`configs/*`、`reports/p02-*`、`dev-docs/stage-plan.md` 仅限「P0.2 行状态列」改 ✅ |
| ❌ 禁触 | `docs/DATA_LOCATIONS.md`、`dev-docs/project-brief.md`、`dev-docs/research/**`、`dev-docs/handovers/**`、`PAPER_POSITIONING.md`、stage-plan 其他任何行、`external/AimCLR`（W3 产物） |

> stage-plan 编辑前先重新 Read 最新内容再做该行精确替换。

## 5. 完成标准与 Git

- [ ] SMQ 克隆 + 增量依赖记录
- [ ] 分割输出 episode 边界 + IoU 数值（JSON 归档 reports/）
- [ ] ≥2 个可视化样本图
- [ ] 一条命令可复现序列写入报告
- [ ] 提交：`feat: P0.2 ...` 白名单外文件一律不 add；遇 `index.lock` 重试；禁 push

## 6. 风险预案与升级

- SMQ 分割质量差 → stage-plan §2 预案：降级滑动窗口+相似度分割；**属方向变更，先向用户报告证据再切换**
- 同一报错连续修复失败 3 次 → 停止局部修补，systematic-debugging 重开诊断并上报
