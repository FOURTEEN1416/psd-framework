# PSD-Framework 项目交接文档（完整版）

> **版本**: v1.0
> **日期**: 2026-08-26 23:30
> **编写者**: 歆歆（协调者）
> **定位**: 本文件为项目级交接文档，持此文件可接手本项目全部工作而无上下文丢失。

---

## 快速导航（5 分钟版）

| 我想知道… | 看哪里 |
|-----------|--------|
| 项目是什么、为什么做 | §1 项目身份 |
| 现在跑到哪了 | §2 当前进度全景 |
| 环境怎么搭 | §3 环境与依赖 |
| 数据在哪 | §4 数据资产 |
| 代码结构 | §5 代码地图 |
| 实验数字 | §6 关键实验结果 |
| 做过哪些重大决策 | §7 决策记录 |
| 下一步干什么 | §8 待办任务与阻塞项 |
| 怎么接手操作 | §9 接手操作手册 |
| 有哪些坑 | §10 已知坑与避雷 |

---

## 1. 项目身份

**一句话**：物理-语义解耦的低资源动物行为识别框架——学术研究仓库，目标是发论文。

**论文标题（暂定）**: *A Physics-Semantics Decoupled Framework for Low-Resource Animal Behavior Recognition under Evolving Evaluation Criteria*

**投稿目标**: Pattern Recognition（IF≈8）/ IJCV

**核心创新候选**: 图像域小样本伪标签方法（姚青 JIA：VFM+锚点+聚类+伪标签）迁移到时序骨架域——"首次性"待核验，这是当前最高优先级的阻塞项。

**双仓关系**（用户 2026-08-23 拍板）：

| 层 | 内容 | 归属 |
|----|------|------|
| ① 基座愿景 / ② 论文 / ③ P0 数据解阻 | 研究线（发论文实现野心） | **本仓库** |
| ④ K9 产品系统 / ⑤ Phase 4 升级 | 产品升级线 | `k9-training-system`（只读，禁止 import） |

**铁律**：
1. GitHub-First 调研（零容忍，禁止 WebSearch）
2. 真理单一性（22 类标签只在 K9 仓权威定义）
3. 三层指标口径：合成 / 公开真实 / 真实 K9 分别汇报，禁止混报
4. 跨仓只允许文档指针，代码复用走 `docs/assets-map.md` 显式移植

---

## 2. 当前进度全景（2026-08-26 23:30）

### 2.1 子阶段完成度

| 子阶段 | 状态 | 关键证据 | 备注 |
|--------|------|---------|------|
| P0.1 AimCLR 预训练 | ✅ 完成 | kNN 20.89% vs 随机 8.33%（2.51×） | `reports/p01-aimclr-2026-08-23.md` |
| P0.2 SMQ 时序分割 | ✅ 完成 | 端到端 IoU 0.4577 vs 随机 ~0.300（1.53×） | `reports/p02-2026-08-24.md` |
| P0.3 姚青 JIA 迁移 | ✅ 完成 | Phase A 纯度 0.534（1.615×）；Phase B heldout 0.658≥0.50 | `reports/p03-jia-phasea/b-fix-2026-08-24.md` |
| P0.4 TCL 半监督增强 | ✅ 完成 | 池精度峰值 0.6913（+10.69pp vs r0） | `reports/p04-tcl-2026-08-24.md` |
| P0.5 骨干微调（合成层） | ✅ 完成 | best_val_acc=97.27% @ ep35（22 类 / 2200 样本） | `reports/w12-p05-stgcnbc-full-2026-08-24.md` |
| P0.5 主动学习效率 | ✅ 完成（负结果） | 冷启动熵未优于随机；best.pt 真实池打分饱和退化 | `reports/p05-al-efficiency-2026-08-24.md` |
| P0.5 warm-start 协议 | ✅ 完成（负结果） | 强域内打分器下随机反超熵 4.2~5.0pp | `reports/p05-warmstart-2026-08-25.md` |
| P0.5 公开真实层扩展池 | ✅ 完成（含 round3） | round1 44.90% 基线；aptv2only 48.98%（+4.08pp 正向信号） | `reports/p05-public-real-round3-2026-08-26.json` |
| C1 解耦切换成本 | ⚠️ small 档完成 | 墙钟比 7.32×、精度 +2.27pp | full 档待 GPU |
| P0.6 论文初稿 | ⏳ 写作侧清空待回填 | LaTeX 脚手架已建（W41） | 数字回填依赖 P0.2-P0.5 |

### 2.2 窗口完成情况（W1-W47）

| 窗口范围 | 主要产出 | 状态 |
|----------|---------|------|
| W1-W2 | 创新性核验 + 数据盘点 | ✅ 已收编 |
| W3-W4 | P0.1/P0.2 外部仓库克隆 | ✅ 已收编 |
| W5-W7 | P0.2 SMQ 救援冲刺 | ✅ 已收编 |
| W8-W10 | P0.3 PhaseA + P0.4 TCL + P0.5 前置 | ✅ 已收编 |
| W11-W12 | P0.5 前置工程 + 合成层完整训练 | ✅ 已收编 |
| W13-W14 | P0.3 PhaseB + P0.5 AL 管线 | ✅ 已收编 |
| W18-W19 | GPU 接力 + C1 解耦 small 档 | ✅ 已收编 |
| W20-W23 | P0.5 公开真实层 partialclass + warm-start | ✅ 已收编 |
| W24-W29 | 多窗口 worktree 机制 + C5 dogpose | ✅ 已收编 |
| W30-W35 | 统一真实扩展池 v1（9844 条） | ✅ 已收编 |
| W36-W37 | NTU 数据渠道 | ✅ 已收编 |
| W38-W39 | 梯度档消融实验 | ✅ 已收编 |
| W40 | 共存模式 + 报告燃料 | ⏳ 待触发（等 bone 完成） |
| W41 | LaTeX 脚手架 + 结论章 | ✅ 已收编 |
| W42 | K9 数据获取材料包 + AL 选样 | ✅ 已收编 |
| W43 | 多域适应 round3（per-source BN） | ✅ 已代收（md 待补） |
| W44 | 融合数（依赖 W40） | ⏳ 待触发 |
| W45 | 预训练价值梯度曲线入论文 | ✅ 已收编 |
| W46 | round2 负结果进 Limitations | ✅ 已收编 |
| W47 | 2D 规则种子适配（负结果） | ✅ 已收编 |

### 2.3 当前运行态

| 组件 | 状态 |
|------|------|
| GPU | 100%（7702/8151 MiB）：W33 三流链 E2 bone 重训运行中（08-27 10:20 实测 epoch 119/300，轮速正常）；看门狗 w33-chain-watchdog 每 2h 巡检正常（git log 可见 supervisor 提交至 08-27 08:37） |
| CPU | W39 梯度档已完成；W33 NTU 两进程运行中 |
| 待触发 | 链自动续跑 E4 bone LE→E3 motion→E5 motion LE→E6 融合 → W44 回写（W40 共存已于 08-26 01:50 完成 round2 收编；round3 已由 W43 完成） |
| 已闭合 | 论文证据矩阵 v1.0 / LICENSE 三件套 / W42 双线 / W42-AL 选样 / W43 round3（aptv2only +4.08pp 正向信号）/ W47 2D 规则种子负结果归档 |

---

## 3. 环境与依赖

### 3.1 硬件

| 项 | 值 |
|----|-----|
| GPU | NVIDIA RTX 5060 Laptop 8GB（driver 573.24，sm_120，capability 12,0） |
| CPU | 主力 CPU（多核，具体型号见系统信息） |
| 存储 | D 盘为主（`D:\Desktop\psd-framework`） |

### 3.2 Python 环境

| 项 | 值 |
|----|-----|
| Python | 3.12.4（`.venv`） |
| PyTorch | 2.11.0+cu128（从 K9 venv 本地复制安装，官方源直连仅 ~130KB/s 不可行） |
| 关键依赖 | pyskl, ultralytics（YOLO26-pose）, scikit-learn, numpy, scipy |
| mamba_ssm | ❌ 未安装（P0.5 事项，按交接指引刻意不装） |
| 安装源 | `requirements.txt`（含 torch 本地 wheel 路径注释） |

### 3.3 初始化步骤

```bash
# 1. 克隆仓库
git clone <repo-url> D:\Desktop\psd-framework

# 2. 创建虚拟环境
cd D:\Desktop\psd-framework
python -m venv .venv
.venv\Scripts\activate

# 3. 安装依赖（torch 需从 K9 venv 复制 wheel）
pip install -r requirements.txt

# 4. 验证
python -c "import torch; print(torch.cuda.is_available())"  # 应为 True
pytest tests/ -q  # 应全绿
```

---

## 4. 数据资产

**核心原则**：数据文件不搬移，本仓通过路径配置访问。

### 4.1 数据集一览

| 数据集 | 路径 | 规模 | 用途 |
|--------|------|------|------|
| InterPet4D (smal_npy) | `D:\Desktop\k9-training-system\data\interpet4d\smal_npy` | 226 个有效片段（1 个全帧 NaN 已剔除） | P0.1 AimCLR 预训练 |
| Animal Kingdom | `D:\Desktop\k9-training-system\data\animal_kingdom` | 329 犬科视频（231 train/98 test），34,772 帧行 | P0.3 weak label + P0.5 真实层 |
| APTv2 | `D:\Desktop\k9-training-system\data\APTv2\APTv2` | 83,304 文件（84,611 标注 + 390 未标注） | 无标签池扩展 |
| 合成骨架集 | `data/synthetic/syn_22class_*_seed42.pkl` | 2200 样本（100/类），W12 扩量版 | P0.5 合成层训练 |
| 统一真实扩展池 v1 | `runs/data_campaign/unified_pool_v1/` | 9844 条（四源汇聚） | 公开真实层实验 |
| W42 选样清单 | `runs/data_campaign/al_selection_*` | Top-100/200 | Day-1 人工核验 |

### 4.2 数据路径配置

所有路径在 `docs/DATA_LOCATIONS.md` 中正式登记。实验配置文件（`configs/*.yaml`）通过绝对路径引用。

### 4.3 数据质量已知问题

| 问题 | 影响 | 处置 |
|------|------|------|
| APTv2 num_keypoints 字段脏数据 | 少量标注格式不一致 | 已在挖掘脚本中过滤 |
| AK 329≠K9 声明 338 | K9 规划期估计偏差 | K9 truth 已修正（commit `faaab28`） |
| APTv2 83K≠K9 声明 242K | 同上 | 已修正 |
| w42v 隔离池 208 片段 | 未并入统一池 | 待组装器补丁（小任务） |

---

## 5. 代码地图

### 5.1 目录结构

```
psd-framework/
├── AGENTS.md                    # 硬规则宪法（必读）
├── configs/                     # 实验配置（YAML）
│   ├── p05_stgcn_bc.yaml       # P0.5 合成层训练配置
│   ├── p05_public_real_*.yaml  # 公开真实层配置
│   └── ntu60_*.yaml             # NTU PhaseB 配置
├── data/                        # 数据（gitignore 大文件）
│   └── synthetic/               # 合成骨架集
├── dev-docs/                    # 项目文档
│   ├── board/BOARD.md           # 跨窗看板（实时更新）
│   ├── decisions/               # 决策记录（ADR）
│   ├── handovers/               # 窗口交接文档（25+ 份）
│   ├── stage-plan.md            # 阶段计划
│   └── project-brief.md         # 项目简介
├── docs/                        # 技术文档
│   ├── assets-map.md            # K9→PSD 资产映射（核心）
│   ├── DATA_LOCATIONS.md        # 数据路径登记
│   └── paper/                   # 论文相关
├── external/                    # 第三方仓库（gitignore）
│   ├── AimCLR/
│   └── SMQ/
├── psd/                         # 核心代码包
│   ├── data/                    # 数据加载
│   ├── models/                  # 模型定义
│   └── training/                # 训练管线
├── reports/                     # 实验报告（核心产出）
├── runs/                        # 运行产物（checkpoints 等）
├── scripts/                     # 入口脚本
└── tests/                       # 测试（499+ 绿）
```

### 5.2 核心模块

| 模块 | 路径 | 说明 |
|------|------|------|
| 数据加载 | `psd/data/` | 合成/真实数据加载器 |
| 模型 | `psd/models/` | AimCLR / SMQ / ST-GCN+BC / 骨干 |
| 训练 | `psd/training/` | 训练管线 |
| 入口脚本 | `scripts/` | 实验运行脚本 |
| 配置 | `configs/` | YAML 实验配置 |

### 5.3 ST-GCN+BC 移植状态（⚠️ 接手时勘误：本节原文误标"尚未移植"）

**已于 W11（2026-08-24，commit `580460f`）完成移植进仓**，且 P0.5 合成层训练（best_val_acc=97.27%）即在其上完成。下表保留为映射参照：

| K9 源文件 | PSD 落点（已存在） |
|-----------|------------------|
| `stgcn_bc/k9_graph.py` | `psd/models/stgcn_k9_graph.py` ✓ |
| `stgcn_bc/stgcn.py` | `psd/models/stgcn_backbone.py` ✓ |
| `stgcn_bc/bc_head.py` | `psd/models/stgcn_bc_head.py` ✓ |
| `stgcn_bc/loss.py` | `psd/training/stgcn_loss.py` ✓ |
| `stgcn_bc/model.py` | `psd/models/stgcn_bc.py` ✓ |
| `stgcn_bc/dataset.py` | 合成函数已随仓落地（synth 生成器）✓ |
| `stgcn_bc/trainer.py` | `psd/training/train_stgcn_bc.py` ✓ |

完整清单可 `ls psd/models/stgcn_*.py psd/training/*stgcn*` 核验；出处：`dev-docs/stage-plan.md` v1.3 修订记录。

---

## 6. 关键实验结果

### 6.1 合成层（22 类 / 2200 样本）

| 实验 | 指标 | 数值 | 报告 |
|------|------|------|------|
| P0.5 完整训练 | best_val_acc | **97.27%** @ ep35 | `reports/w12-p05-stgcnbc-full-2026-08-24.md` |
| 消融 20 样/类 | val_acc | 77.27% | 同上 |
| 消融 50 样/类 | val_acc | 95.00% | 同上 |
| E6 粗粒度 Y' | val_acc | 95.91%（−1.36pp vs Y） | 同上 |
| K9 参照基线 | val_acc | 46.97% @ ep21 | K9 phase-3 |

### 6.2 公开真实层

| 实验 | 指标 | 数值 | 报告 |
|------|------|------|------|
| round1 基线（复跑） | best_val_acc | **44.90%** | `reports/p05-public-real-round3-2026-08-26.json` |
| round2 直接混合 | best_val_acc | 40.82%（−4.08pp） | 同上 |
| round3 aptv2only | best_val_acc | **48.98%**（+4.08pp） | 同上（首个正向信号） |
| round3 w35only | best_val_acc | 44.90%（持平） | 同上 |
| round3 full (per-source BN) | best_val_acc | 40.82%（仍稀释） | 同上 |

**关键发现**：飞轮不是死的——aptv2 单源增强 +4.08pp 超基线，混合策略是瓶颈。

### 6.3 其他关键数字

| 实验 | 数值 | 说明 |
|------|------|------|
| C1 解耦切换成本（small） | 墙钟比 **7.32×**，精度 +2.27pp | 解耦 vs 全重训 |
| P0.5 warm-start 协议 | 正收益基线 7.8%→82%，但强域内打分器下负结果 | 协议层有效，打分器层无效 |
| P0.5 AL 效率 | 冷启动熵未优于随机 | 负结果，如实登记 |
| 梯度档消融 | spc5 +21.21pp / spc10 +9.85pp / spc20 −2.27pp | 收益随标注资源单调衰减 |

---

## 7. 决策记录（重大 ADR）

| 编号 | 决策 | 日期 | 影响 |
|------|------|------|------|
| ADR-0001 | 双仓拆分 | 2026-08-23 | 研究线/产品线分离，K9 仓只读 |
| ADR-0002 | NTU 数据渠道 + 合成方案（路径 a = 移植 K9 生成器） | 2026-08-24 | P0.5 合成层数据来源锁定 |
| ADR-0002 v1.1 | E6 双贴合设计（K9 报表粒度差业务动机） | 2026-08-24 | 论文 E6 实验设计冻结 |
| ADR-0003 | SMQ 主口径 = 种子伪 GT | 2026-08-24 | P0.2 评估口径锁定 |
| ADR-0004 | 论文回填策略 = 走法 A（等 P0.5 完成后统一回填） | 2026-08-26 | P0.6 写作流程 |
| 用户裁决 | LICENSE = MIT 代码 + CC BY-NC 4.0 资产 | 2026-08-26 | 三件套已落地 |
| 用户裁决 | B-full worktree 隔离 | 2026-08-24 | 多窗口并行机制 |
| 用户裁决 | A2 warm-start 协议 | 2026-08-24 | P0.5 warm-start 实验 |

---

## 8. 待办任务与阻塞项

### 8.1 阻塞链（按依赖顺序）

```
bone 三流链重训（E2 进行中，ETA 见 runs/w33_chain + BOARD）
    ↓ （链脚本自动续跑 E4/E3/E5/E6）
E6 3s 融合数 JSON 落盘（reports/ntu-phaseB-3s-ensemble.json 主检出出现）
    ↓
W44 回写 experiment-skeleton R4 行（过预注册线 ≥77.18% 三流口径 / 不过则单流口径+标题降级提请用户终裁）
    ↓
证据矩阵 100%
    ↓
P0.6 论文数字回填（走法 A 统一回填，ADR-0004）
    ↓
投稿终审
```

### 8.2 可立即启动（不依赖 GPU）

| 任务 | 说明 | 优先级 |
|------|------|--------|
| **ST-GCN+BC 移植** | 7 个文件 CPU 操作，assets-map §3.1 有完整映射表 | **高** |
| **w42v 池合并补丁** | build_unified_pool.py 加第六源支持（208 片段） | 中 |
| **W43 round3 md 报告** | JSON 已有，缺文字报告 | 低 |
| **记忆库补写** | 5 条积压（服务恢复后） | 低 |

### 8.3 等待外部输入

| 任务 | 等什么 | 谁负责 |
|------|--------|--------|
| 真实 K9 层 ≥85% 验收 | 产品侧数据到位 | samtwl 邮件已发，等回复 |
| 论文标题终裁 | 用户拍板 | 用户 |
| 三项投稿终审 | 打包时 | 用户 |
| 骨架特征随包边界（LICENSE A） | 用户裁决 | 用户 |
| NTU 许可文本（LICENSE B） | 用户裁决 | 用户 |
| 再分发承诺边界（LICENSE C） | 用户裁决 | 用户 |

---

## 9. 接手操作手册

### 9.1 第一天必做

1. **读文档**：`AGENTS.md` → `dev-docs/HANDOVER.md` → 本文件 → `dev-docs/stage-plan.md`
2. **搭环境**：§3.2 初始化步骤
3. **跑测试**：`pytest psd -q`（应 499 绿；⚠️ v1.1 勘误：测试布局在 `psd/*/tests/` 包内，仓库根无 `tests/` 目录）
4. **看板**：`dev-docs/board/BOARD.md`（最新状态）
5. **查报告**：`reports/` 目录，按时间排序读最新的

### 9.2 工作窗口协议

- 新窗口一律用 `pwsh scripts/new_window_worktree.ps1 -Name <窗口名>` 隔离
- 提交用 Conventional Commits 中文描述
- 白名单精确 `git add`，禁用 `git add .`
- 完成后 `pwsh scripts/window_checkin.ps1 -Name <窗口名>` 自助收编
- 跨窗信息写入 `dev-docs/board/BOARD.md`

### 9.3 关键脚本速查

| 脚本 | 用途 |
|------|------|
| `scripts/new_window_worktree.ps1` | 创建 worktree |
| `scripts/window_checkin.ps1` | 自助收编（三道门禁） |
| `scripts/window_board.ps1` | 看板操作 |
| `scripts/run_p05_full.py` | P0.5 合成层训练 |
| `scripts/run_p05_public_real_round2.py` | 公开真实层实验 |
| `scripts/run_ntu_phaseb.py` | NTU PhaseB 训练 |

---

## 10. 已知坑与避雷

### 10.1 技术坑

| 坑 | 表现 | 规避 |
|----|------|------|
| 进程蒸发 | GPU 训练进程随机消失（今日 4 起） | 已部署 w33-chain-watchdog 计划任务（20 分钟巡检） |
| GPU 内存溢出 | 8GB RTX 5060 三流链吃满 7.7GB | bone 段完成后内存释放；大 batch 实验需排队 |
| torch 安装慢 | 官方源直连仅 ~130KB/s | 从 K9 venv 复制 wheel（见 `requirements.txt`） |
| LaTeX 编译 | latexmk 缺 Perl / LaTeX3 全展开禁自定义宏 | 用经典三连替代（pdflatex+bibtex+pdflatex） |
| CJK 编码 | pdflatex 不支持 Unicode CJK | 非注释文本禁 CJK |

### 10.2 流程坑

| 坑 | 教训 | 规避 |
|----|------|------|
| W41 自助收编丢文件 | checkin 前未确认 wt 分支有新提交，-Remove 卸窗致交付物物理丢失 | checkin 前必 `git log wt/<名> -1` 核对 |
| 窗口编号撞车 | W42=K9 材料包 与 W42-AL=标签入环选样撞号 | 后续编号从 W47 顺延 |
| 主检出工作泄漏 | W43 的 TDD + 编排扩展写在主检出而非 worktree | 新窗口一律 worktree 开工 |
| 三代数据混报 | 三代计数器曾混用 | 已修：new_pools/old_pools/total 严格分离 |

### 10.3 科学坑（负结果，如实登记）

| 负结果 | 影响 | 论文处置 |
|--------|------|---------|
| AL 冷启动熵未优于随机 | P0.5 AL 效率实验 | 如实入 Limitations |
| warm-start 强域内打分器下负结果 | P0.5 warm-start 协议 | 协议层正收益 7.8%→82% 入文，打分器层入 Limitations |
| round2 扩展池混合 −4.08pp | 公开真实层飞轮 | 但 aptv2only +4.08pp 正向信号修正结论 |
| 2D 规则种子 gate4 不可分 | W47 负结果 | 2D 弱标签降级预训练池，引擎入库供未来使用 |
| 预训练收益随标注衰减 | 梯度档消融 | spc10-20 交叉区如实，线性插值 ~15-20 |

---

## 附录 A：Git 提交速查

```bash
# 最近提交（查看项目活跃度）
git log --oneline -20

# 查看某个窗口的提交
git log --oneline --author="W12" 

# 查看所有实验报告
ls reports/*.md | sort -t'-' -k2 -rn
```

## 附录 B：联系信息

| 角色 | 说明 |
|------|------|
| 项目所有者 | 用户（拍板决策） |
| 协调者 | 歆歆（sliver-vibe-coding） |
| 上游产品仓 | `D:\Desktop\k9-training-system` |
| 数据来源 | samtwl（邮件已发，等回复） |

## 附录 C：修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-26 | 初版，覆盖项目全貌 + 当前进度 + 环境 + 数据 + 代码 + 实验 + 决策 + 待办 + 接手手册 + 已知坑 |
| v1.1 | 2026-08-27 | 接手勘误（新协调者）：①§5.3 ST-GCN+BC 实已 W11 移植进仓（原文误标待移植）②§9.1 测试命令改 `pytest psd -q` ③§2.3 运行态与 §8.1 阻塞链按 08-27 上午实况更新 |

---

> **交接确认**：本文档为 2026-08-26 23:30 快照。接手后请优先更新 `dev-docs/board/BOARD.md`（实时看板）和 `dev-docs/stage-plan.md`（阶段计划），它们比本文件更实时。
