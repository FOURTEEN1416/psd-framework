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
| Python 环境 / venv | ✅ 已建立：`.venv` Python 3.12.4 + torch 2.11.0+cu128（GPU capability (12,0) 实测通过；torch 系经 K9 venv 本地复制安装，见 `requirements.txt` 注释） |
| P0 子阶段 | **P0.1 ✅ 达标**（kNN 20.89% vs 随机 8.33%，2.51×，`reports/p01-aimclr-2026-08-23.md`）；P0.2-P0.6 ⏳ |
| external/ 第三方仓库克隆 | AimCLR ✅（P0.1 使用）；SMQ / TCL ⏳ P0.2/P0.4 开工前 |
| 姚青 JIA 创新性核验调研 | ✅ 初步通过（W1 零占坑，见 `dev-docs/research/NOVELTY_CHECK_YAOQING_JIA.md`；投稿前 Scholar 终审保留） |
| 远程仓库（GitHub） | ⏳ 未创建 |

## 4. 环境基线（✅ 本仓已建成并实测，2026-08-23）

| 项 | 实测值 |
|----|--------|
| Python | 3.12.4（`.venv`） |
| PyTorch | **2.11.0+cu128 实测**（RTX 5060 sm_120，capability (12,0)；获取方式：K9 venv 同版本二进制本地复制——官方源直连仅 ~130KB/s 不可行，见 `requirements.txt`） |
| GPU | RTX 5060 Laptop 8GB（driver 573.24） |
| mamba_ssm | ⏳ 未安装（P0.5 事项，按交接指引刻意不装）；WSL 编译经验 + 纯 PyTorch 回退在 K9 仓 |
| Git 身份 | 本仓 local：`K9 Training Dev <k9-training@local>` |

## 5. 数据资产（2026-08-23 已盘点，路径实锤 ✓）

详见 `docs/DATA_LOCATIONS.md`。核心三条：

| 数据集 | 路径（均验证存在） | 关键结构 |
|--------|------------------|---------|
| InterPet4D | `D:\Desktop\k9-training-system\data\interpet4d` | **smal_npy/** ✓（P0.1 直接可用）、pet_npy、smpl_npy |
| Animal Kingdom | `D:\Desktop\k9-training-system\data\animal_kingdom` | action_recognition/、pose_estimation/、video_grounding/ |
| APTv2 | `D:\Desktop\k9-training-system\data\APTv2`（另有 aptv2_annotations/canidae/yolo/yolo_pose 四个伴生目录） | APTv2/ 子目录 |

数据文件**不搬移**，本仓通过路径配置访问。✅ 数量级已由 W2 实测复核（2026-08-23）：AK 犬科 329 视频 / 34,772 帧级标注行、APTv2 83,304 文件、InterPet4D smal_npy 226 ✓——原 K9 声明「338/242K」确证为规划期未验证估计，详见 `reports/data-inventory-2026-08-23.md`。P0.1 补充实测：smal fit 覆盖 dog01–12 共 12 类，1 个 clip 全帧 NaN 已剔除（225 有效）。

## 6. 可复用资产指针

见 `docs/assets-map.md`。最常用两项：

- ST-GCN+BC 训练栈（合成 46.97%）：K9 仓 `backend/ml/behavior/stgcn_bc/`
- Mamba 基线（合成 85.61%）：K9 仓 `backend/ml/behavior/`（具体文件名待确认）

## 7. 关键数字速查表（写论文/做实验常查）

| 数字 | 含义 | 来源 |
|------|------|------|
| 79.18% | AimCLR 在 NTU60 的参照成绩（AAAI 2022 原文） | 调研文档 |
| **20.89% (2.51×)** | **P0.1 AimCLR 在 InterPet4D 的 kNN top-1（公开真实层，dog ID 12 类代理 probe，随机基线 8.33%）** | `reports/p01-knn-result.json` |
| 77.2% | AimCLR++（PR 2024）NTU xsub，v1 为 74.3%——骨干升级候选 | W1 附带发现 |
| 82.7% | TCL 用 10% 标注达到的成绩（全监督参照 88.6%，CVPR 2021） | 调研文档 |
| 4.5% | 22 类随机猜测基线 | K9 实验 |
| 46.97% | ST-GCN+BC 合成数据 best_val_acc（epoch 21） | K9 phase-3 |
| 85.61% | Mamba 合成数据基线 | K9 phase-4 |
| 100-200 片段 | 主动学习人工标注预算 | stage-plan |
| ≥85% | P0.5 微调验收线（22 类） | stage-plan |
| 3-4 周 | P0 总周期 | stage-plan |

## 8. 下一步任务

**当前一窗执行 + 两线待排（2026-08-24 复核换防）**：

| 窗口 | 交接文档 | 任务 | 领地（互斥） |
|------|---------|------|-------------|
| **W4 重启** | `handovers/W7-p02-smq-rescue.md`（唯一任务书） | P0.2 SMQ 分割**救援**（诊断优先，双口径评估，切换需用户裁决） | 原 W4 白名单全部（`*smq*` 文件、psd/models+training、scripts/configs 的 p02 文件、reports/p02-*）；含 W4 未提交产物接管清单 |
| **W8** | `handovers/W8-p03-jia-phaseA.md` | ✅ **Phase A 完成**（commit `5d790c7`：纯度 0.534=随机 1.615×、噪声消融 30% 仅降 3.1pp、TDD 98 绿、P0.4 池格式已移交）；Phase B 22 类映射等路径 a 合成数据 | 已收官 |
| ~~W7~~（原 W4 重启） | `handovers/W7-p02-smq-rescue.md` | ✅ **救援+冲刺达标**：根因=官方超参致编码器码塌缩，mse_loss_weight→1.0 修复；种子伪 GT 口径 IoU 0.409 → 冲刺 E-B vm6 **0.476** / 过夜 E-C K=8 **0.4577±0.0488**（均 ≥ 预注册 0.45）。⚠️ `reports/p02-smq-iou-eC-seeds.json` 尚未提交，E-C 定稿收编归 W4 owner 收尾 | 原 W4 白名单；窗口锁 `reports/p02-window-lock.md` |
| **W9** | `handovers/W9-ntu-repro.md` | Phase A ✅ **全部完成**（协议核实 + 数据已就位并校验通过，见 `reports/ntu-phasea-2026-08-24.md` v1.1；用户经 GKD 加速器走通渠道 A）；Phase B 复现训练**GPU 已解禁可排程**（P0.2 冲刺收官释放显卡；错峰默认=夜间长训练，与 W11 白天冒烟互斥让行） | `*ntu*` 文件、reports/ntu-* |
| ~~W10~~ | `handovers/W10-p04-tcl.md` | ✅ **完成且经复核确认**（2026-08-24 复核会话：commit `4f04ad4` 边界合规、pytest 116 绿新鲜复跑、全量实验重跑逐格复现、移交池哈希一致；B+ 收口不改代码——iter3 池 191 条为交付物，r1 早停池为可选替代见报告 §5①/§8） | 已收官 |
| **W11（新）** | `handovers/W11-p05-prep.md` | P0.5 前置工程：assets-map 补链（⚠️ 阻断发现：该文件从未落盘但被三处 truth 引用）+ 合成数据生成器移植重建 22 类骨架集（ADR-0002 裁决②触发点已到）+ ST-GCN+BC 栈进仓 + 合成层冒烟；微调达标实验留后续窗口 | `docs/assets-map.md`、`psd/data/*synth*`、`psd/models/**`、`psd/training/*stgcn*`、scripts/configs 的 p05 文件、reports/w11-*、DATA_LOCATIONS 合成层小节 |
| ~~W5~~ | `handovers/W5-p06-paper-draft.md` | ✅ 写作侧收官，剩余为 P0 数据依赖 | `docs/paper/**` 保持只读待回填 |

> ⏳ ~~用户待决一项：NTU 数据获取渠道~~ → **已决并完成（渠道 A，2026-08-24）**：数据就位校验通过。
> 三项用户裁决已落档 `dev-docs/decisions/0002-user-rulings-ntu-synthetic-e6.md`：① NTU 验证纳入（两相推进中）；② 路径 a 合成数据选移植重建（P0.5 前执行）；③ E6 双贴合场景方向确认。

> 并行纪律：各窗口只提交白名单内文件；stage-plan 编辑前重读最新内容做行级替换；`.venv` 归实验窗口主用、他人只读 import；遇 git `index.lock` 等待重试。

**已收官存档**：W1 创新核验 ✅ / W2 数据盘点 ✅ / W3→P0.1 达标 ✅ / W4→P0.2 未达标转 W7 救援（交接文档保留：`W4-p02-smq.md`）/ **W6→规则种子完成 ✅（commit `aaa1e1c`，报告 `rule-seeds-2026-08-24.md`，P0.3 锚点备料就绪）** / **W5→论文写作侧清空 ✅（`be15708`..`6da4396`，评审记录 `docs/paper/review-log.md`）**。

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
| 2026-08-24 | W6 窗口 | 规则引擎粗标完成：路径 b 前提证伪（用户裁定方案 3）+ SMAL 关节语义几何实测 + 物理先验 7 类规则族（TDD 11 passed，commit `aaa1e1c`）；全量 225 clip 种子落盘，帧占比 sitting 36.3%/walking 23.4%；移交 P0.3 消费规则（置信 ≥0.8、时长 ≥0.5s） |
| 2026-08-24 | W4 窗口（进行中→转救援） | P0.2 两轮评估均失败：pred=单段[0,640]、IoU 0.20 < 随机 0.43、boundary F1=0；产物全部未提交 → 规划会话裁决转 W7 救援接管 |
| 2026-08-24 | 歆歆（W5 论文线） | P0.6 四件套 + 增量二（introduction/conclusion-limitations/figure-specs）+ 两轮对抗评审 + 风险登记册 R1-R10 + 投稿就绪门；commits `be15708`..`6da4396`；写作侧清空待 P0 数据回填 |
| 2026-08-24 | 歆歆（规划会话） | 全窗口复核（W4 失败态/W6 完成/W5 收官）→ 重规划：跨窗口协同洞察（种子伪 GT 救评估协议、P0.3 与 SMQ 解耦并行）→ 编制 W7/W8 交接文档 + HANDOVER v1.4 + stage-plan v1.2 |
| 2026-08-24 | 歆歆（验收复核+W11 规划） | W10-P0.4 全量复核通过（116 测试复跑 / 全量实验逐格复现 / 移交池哈希一致；B+ 收口零代码改动）；新发现：`docs/assets-map.md` 从未落盘（truth 断链）、E-C 结果 JSON 未收编；编制 W11 前置工程交接文档 + HANDOVER v1.5 + stage-plan 同步 |

## 11. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-23 | 建仓交接：项目身份/双仓边界/环境基线/数据盘点/任务路由 |
| v1.1 | 2026-08-23 | 任务路由改三窗口并行：handovers/ 白名单互斥边界 + 收尾回写约定 |
| v1.2 | 2026-08-23 | 三窗口收官：§8 路由改为完成态 + P0.2 指向；会话历史补 W2/W3/收尾记录 |
| v1.3 | 2026-08-23 | W3 补完收尾：§3 状态表/§4 环境基线同步实测事实（venv✅ P0.1✅ AimCLR✅ 核验✅）、§5 数据复核口径回填、§7 速查表增补 P0.1 kNN 与 AimCLR++ 候选 |
| v1.4 | 2026-08-24 | 规划会话换防：§8 路由改 W7（P0.2 救援）/W8（P0.3 Phase A）两线并行，W5/W6 移交收官存档；§10 补 W6 完成、W4 转救援、W5 论文线、规划会话四条记录 |
| v1.5 | 2026-08-24 | 复核会话换防：§8 路由更新——W10 收官（经复核确认）/ W7 冲刺达标（E-C 待定稿收编）/ W9-B GPU 解禁 / **新增 W11 P0.5 前置工程**；§10 补验收复核记录；阻断发现登记（assets-map 缺失） |
