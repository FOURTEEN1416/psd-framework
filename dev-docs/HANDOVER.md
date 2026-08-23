# PSD-Framework — 项目交接文档

> **版本**: v1.0
> **日期**: 2026-08-23
> **项目根目录**: `D:\Desktop\psd-framework`
> **交接会话**: 歆歆（sliver-vibe-coding）于 2026-08-23 完成建仓拆分 + truth 链初始化 + 数据盘点
> **上游决策**: `D:\Desktop\k9-training-system\dev-docs\decisions\0013-research-repo-split.md`（ADR 0013）

---

## ⚡ 新窗口快速启动（按顺序读）

1. 本文（5 分钟了解全貌）
2. `AGENTS.md` —— 硬规则宪法（GitHub-First / 三层指标口径 / 跨仓边界）
3. `dev-docs/project-brief.md` v1.0 —— 立项口径
4. `dev-docs/stage-plan.md` v1.0 —— P0.1-P0.6 任务清单（全部 ⏳ 未开工）
5. 验证状态：`git log --oneline` 应见首提交 `a512daf`

---

## 1. 项目身份（30 秒版）

学术研究仓库：**物理-语义解耦的低资源动物行为识别框架**。
终极野心：适应动态业务逻辑的视觉基座系统。当前目标：发论文。

- **Title**: *A Physics-Semantics Decoupled Framework for Low-Resource Animal Behavior Recognition under Evolving Evaluation Criteria*
- **期刊**: Pattern Recognition（IF≈8）/ IJCV
- **核心创新候选**: 图像域小样本伪标签方法（姚青 JIA：VFM+锚点+聚类+伪标签）迁移到时序骨架域——**"首次性"待核验，这是当前最高优先级的阻塞项**

## 2. 双仓关系（用户口径，2026-08-23 拍板）

| 层 | 内容 | 归属 |
|----|------|------|
| ① 基座愿景 / ② 论文 / ③ P0 数据解阻 | 研究线（发论文实现野心） | **本仓库** |
| ④ K9 产品系统 / ⑤ Phase 4 升级（LLM 解释器 + Mamba 产品化） | 产品升级线 | `k9-training-system` |

- Mamba 双角色：产品替换在 K9 仓；85.61% 合成基线可移植到本仓作论文骨干候选
- **跨仓只允许文档指针，禁止 import**；代码复用走 `docs/assets-map.md` 显式移植后以本仓测试为准

## 3. 当前状态（诚实版）

| 事项 | 状态 |
|------|------|
| 建仓 + truth 链 + 宪法 + 资产地图 | ✅ 完成（commit `a512daf`） |
| Python 环境 / venv | ❌ 未建立（P0.1 前置） |
| P0.1-P0.6 全部子阶段 | ⏳ 未开工（无任何代码） |
| external/ 第三方仓库克隆（AimCLR/SMQ/TCL） | ⏳ 未克隆 |
| 姚青 JIA 创新性核验调研 | ⏳ 未做（阻塞 P0.3 与论文框架设计） |
| 远程仓库（GitHub） | ⏳ 未创建 |

## 4. 环境基线（继承 K9 运行时经验，本仓环境未建——未验证项）

| 项 | 经验值（来源 K9 runtime） |
|----|--------------------------|
| Python | 3.12 |
| PyTorch | 2.11+cu128（RTX 5060 为 sm_120 架构，必须 cu128 构建） |
| GPU | RTX 5060（本地） |
| mamba_ssm | WSL 内编译成功经验：CUDA 12.8 + gcc-12 + sm_120；纯 PyTorch 回退实现也在 K9 仓（videomamba_skeleton.py） |
| Git 身份 | 本仓已配置 local：`K9 Training Dev <k9-training@local>`（与 K9 仓一致） |

## 5. 数据资产（2026-08-23 已盘点，路径实锤 ✓）

详见 `docs/DATA_LOCATIONS.md`。核心三条：

| 数据集 | 路径（均验证存在） | 关键结构 |
|--------|------------------|---------|
| InterPet4D | `D:\Desktop\k9-training-system\data\interpet4d` | **smal_npy/** ✓（P0.1 直接可用）、pet_npy、smpl_npy |
| Animal Kingdom | `D:\Desktop\k9-training-system\data\animal_kingdom` | action_recognition/、pose_estimation/、video_grounding/ |
| APTv2 | `D:\Desktop\k9-training-system\data\APTv2`（另有 aptv2_annotations/canidae/yolo/yolo_pose 四个伴生目录） | APTv2/ 子目录 |

数据文件**不搬移**，本仓通过路径配置访问。数量级声明（226 序列/338 视频/242K 文件）来自 K9 truth 文档，本仓尚未逐一复核。

## 6. 可复用资产指针

见 `docs/assets-map.md`。最常用两项：

- ST-GCN+BC 训练栈（合成 46.97%）：K9 仓 `backend/ml/behavior/stgcn_bc/`
- Mamba 基线（合成 85.61%）：K9 仓 `backend/ml/behavior/`（具体文件名待确认）

## 7. 关键数字速查表（写论文/做实验常查）

| 数字 | 含义 | 来源 |
|------|------|------|
| 79.18% | AimCLR 在 NTU60 的参照成绩（AAAI 2022 原文） | 调研文档 |
| 82.7% | TCL 用 10% 标注达到的成绩（全监督参照 88.6%，CVPR 2021） | 调研文档 |
| 4.5% | 22 类随机猜测基线 | K9 实验 |
| 46.97% | ST-GCN+BC 合成数据 best_val_acc（epoch 21） | K9 phase-3 |
| 85.61% | Mamba 合成数据基线 | K9 phase-4 |
| 100-200 片段 | 主动学习人工标注预算 | stage-plan |
| ≥85% | P0.5 微调验收线（22 类） | stage-plan |
| 3-4 周 | P0 总周期 | stage-plan |

## 8. 下一步任务

**三窗口并行已全部收官（2026-08-23）**：W1 创新核验 ✅ / W2 数据盘点 ✅ / W3 P0.1 达标 ✅。收尾回写已完成（PAPER_POSITIONING / brief / stage-plan 前置状态）。

**当前主线 → P0.2 SMQ 时序分割**（见 `stage-plan.md` §1；开工前先克隆 SMQ 到 external/）。

> 历史路由表存档：W1=`handovers/W1-novelty-check.md`，W2=`handovers/W2-data-inventory.md`，W3=`handovers/W3-p01-bootstrap.md`（均已完成，交接文档保留作过程证据）。

## 9. 硬规则速查（完整版见 AGENTS.md）

1. 技术调研**禁止 WebSearch**，强制 GitHub 工具链（搜索 → awesome 目录 → 逐仓验证活跃度）
2. 三层指标口径：合成 / 公开真实 / 真实 K9 分别汇报，禁止混报
3. 无新鲜验证，无完成声明；报告归档 `reports/`
4. external/ 内第三方代码不改内部实现，适配写在 `psd/`
5. Conventional Commits 中文描述

## 10. 会话历史

| 日期 | 会话 | 产出 |
|------|------|------|
| 2026-08-23 | 歆歆（拆分建仓） | 双仓拆分执行：truth 链 7 文件 + 调研复制 5 份 + 数据盘点 + Git 检查点（本仓 `a512daf` / K9 仓 `d295634`+`4690661`+`6455146`）；K9 侧同步宪法 v1.31 / stage-plan v1.12 / brief v2.4 / HANDOVER v1.5 |
| 2026-08-23 | 歆歆（接管+W1） | 接管基线审计通过；三窗口并行改造：handovers/ 三份交接文档 + §8 路由更新；本窗口执行 W1 创新性核验 |
| 2026-08-23 | W2 窗口 | 数据深盘点：smal_npy 226✓ / AK 犬科实测 329≠声明338 / APTv2 实测 83,304≠声明242K → 溯源为规划期估计，K9 truth 已修正（`faaab28`）；DATA_LOCATIONS 回填 + brief §4 更新 |
| 2026-08-23 | W3 窗口 | P0.1 全链：InterPet4D 加载器（TDD 9 passed）+ torch 2.11+cu128 锁定 + 修复表征坍缩（E1-E7 实验链）+ kNN 20.89% vs 随机 8.33%（2.51×达标）；AimCLR 已克隆 external/ |
| 2026-08-23 | 歆歆（W1+收尾） | niais 目录补扫确认结论不变；统一收尾回写：PAPER_POSITIONING / brief §1§7§8 / stage-plan 前置#1 与头部状态 / HANDOVER v1.2。注：发现一处未提交越权编辑（PAPER_POSITIONING，内容正确已保留收录） |

## 11. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-23 | 建仓交接：项目身份/双仓边界/环境基线/数据盘点/任务路由 |
| v1.1 | 2026-08-23 | 任务路由改三窗口并行：handovers/ 白名单互斥边界 + 收尾回写约定 |
| v1.2 | 2026-08-23 | 三窗口收官：§8 路由改为完成态 + P0.2 指向；会话历史补 W2/W3/收尾记录 |
