# W8 交接文档 — P0.3 姚青 JIA 迁移 Phase A（锚点学习 + 原型聚类）

> **你是 W8 窗口**。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.4 → `reports/rule-seeds-2026-08-24.md`（你的输入）→ `docs/paper/method.md` §3.3（方法口径）→ `docs/paper/experiment-skeleton.md` E3（验收指标）。
> **关键设计决策**：Phase A 与 P0.2 SMQ 的结果**完全解耦**——锚点学习直接消费 W6 规则种子段，不等分割救援。

---

## 1. 任务目标（一句话）

在规则种子段 + P0.1 冻结骨干特征上跑通**锚点学习 → 原型聚类 → 伪标签置信分配**闭环，产出聚类纯度曲线与敏感性扫描（论文实验 E3），为伪标签迭代建立可量化基线。

## 2. 为什么能开工（依赖裁决记录）

| 传统依赖 | 本 Phase 的替代 | 依据 |
|---------|----------------|------|
| SMQ 分割提案（P0.2 输出） | 规则种子段直接作为"提案代理" | W6 已交付全量 225 clip 种子（4,945 段），带置信度与规则 ID |
| 人工标注锚点 | 规则引擎粗标种子（高召回低精度先验） | method.md §3.3.2 设计决策已冻结 |
| VFM 骨干 | P0.1 冻结 AimCLR 骨干（物理层等价物） | `runs/p01_aimclr_pretext/` checkpoints |

> 接口契约：本 Phase 一切下游消费以「段列表 → Φ 特征 → 聚类」为接口；未来 SMQ 提案可用同接口替换种子段——**不要把种子来源硬编码进聚类代码**。

## 3. 执行链

### Step 1：种子消费适配（先读后用）
- 消费过滤（W6 报告 §8 移交建议，强制执行）：置信度 ≥0.8；最短持续 ≥0.5s。
- 口径标注：公开真实层-物理先验伪标签（禁止与合成层混报）。
- 类别体系：**7+unknown 物理先验类**（sitting/walking/standing/running/jump/rise_transition/lying）。22 类映射属 Phase B——需路径 a 合成数据到位并与用户对齐后设计，本阶段不做。

### Step 2：特征抽取
- 用 P0.1 骨干（frozen Φ）对每个种子段抽 embedding；复用 `psd/data/interpet4d.py` 加载器与 NTU 视图导出逻辑（只读 import）。

### Step 3：锚点学习 + 原型聚类
- 按 method.md Algorithm 1 实现初始化版：原型从种子锚点初始化 → 最近原型分配 + 置信 κ → （本 Phase 只到分配与纯度评估，**迭代伪标签自训练留给 P0.4 TCL 窗口衔接**）。
- 设计决策遵守 method.md §3.3.2：prototype-margin 置信、frequency-aware margin 处理类别不平衡（sitting 36% vs lying 1.6% 的长尾现实）、Φ 全程冻结。

### Step 4：评估（对齐论文 E3）
- 主指标：聚类纯度 / NMI / 覆盖率；按种子真伪标签对照。
- 敏感性扫描：种子比例 {25%,50%,75%,100%} × K ∈ {5,7,10,14} × τ 扫描。
- 统计纪律：≥3 seeds 报 mean±std（experiment-skeleton §统计协议）。
- **种子噪声注入消融**（风险登记册 R8 缓解项）：向种子注入 {10%,20%,30%} 标签噪声，验证纯度不崩——这是审稿防线，必做。

### Step 5：归档
- `reports/p03-jia-phasea-<日期>.md`：纯度曲线 JSON + 敏感性表 + 一条命令复现序列。
- TDD：核心函数（过滤/特征池化/聚类/纯度计算）先测后码。

## 4. 边界（白名单互斥）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | `psd/models/jia*/**`、`psd/training/*jia* 或 *p03*` 新文件、`psd/data/*p03*或*jia*` 新文件、`scripts/*p03*`、`configs/p03*`、`reports/p03-*`、stage-plan 仅「P0.3 行状态列」 |
| ❌ 禁触 | 一切 `*smq*` 文件与 `psd/training/smq_runner.py`/`segment_iou.py`（W7 领地，只读 import 允许）；`psd/data/rule_seeds.py` 及其测试（W6 产物只读）；`external/**`；`docs/paper/**`（W5）；`.venv` 只读 import；dev-docs 其余文件 |
| 共享资源 | P0.1 checkpoint 只读；`data/seeds/**` 只读 |

## 5. 完成标准

- [ ] 纯度/NMI 显著优于随机分配基线（附基线数值）
- [ ] 敏感性扫描表 + 噪声注入消融结果
- [ ] TDD 测试绿 + 报告归档 + 中文 Conventional Commit
- [ ] 向下移交声明：给 P0.4 的伪标签池格式定义写入报告附录

## 6. 升级路径

- 种子质量疑似不足（如纯度 <1.5× 随机基线）→ 先查过滤阈值与特征抽取，再上报用户；禁自行放宽口径
- 发现需要 SMQ 提案才能继续的设计点 → 记录到报告「阻塞项」，切 Phase A 内可并行的子任务，不空转等待 W7

---

*交接编制: 歆歆（规划会话）2026-08-24 · 依据: HANDOVER v1.4 §8 / stage-plan v1.2 / W6 报告移交清单*
