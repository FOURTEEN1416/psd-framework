# PSD-Framework — Physics-Semantics Decoupled Framework

物理-语义解耦的低资源动物行为识别框架（学术研究主仓库）

> **论文目标**：Pattern Recognition（IF≈8）/ IJCV
> **Title**: *A Physics-Semantics Decoupled Framework for Low-Resource Animal Behavior Recognition under Evolving Evaluation Criteria*
> **状态**：🔄 立项完成（2026-08-23），P0 数据解阻未开工

## 仓库分工（2026-08-23 拆分决策）

| 仓库 | 角色 |
|------|------|
| 本仓库 `psd-framework` | **研究线**：基座方法论 + 论文 + P0 数据解阻管线（终极野心载体） |
| `k9-training-system` | **产品线**：工作犬评估系统（Phase 1-3 已关闭，作为方法验证平台；Phase 4 产品升级继续） |

## 技术路线

```
规则种子 → AimCLR 自监督预训练（AAAI 2022）
        → SMQ 无监督时序分割（ICCV 2025）
        → 姚青 JIA 方法迁移 🔥（VFM+锚点+聚类+伪标签，核心创新候选，首次性待核验）
        → TCL 半监督增强（CVPR 2021）
        → 主动学习（100-200 片段标注）
        → 骨干微调（ST-GCN+BC / Mamba）→ 22 类 ≥85%
```

## 目录结构

```
psd/            # 核心 Python 包（data / models / training）
scripts/        # 入口脚本（训练 / 评估）
configs/        # 实验配置
external/       # 第三方方法官方仓库克隆（gitignore）
data/           # 数据集链接与索引（gitignore）
runs/           # 训练输出（gitignore）
dev-docs/       # 内部 truth（brief / stage-plan / decisions / research）
docs/           # 资产地图 / 数据位置登记
reports/        # 评估报告归档
```

## Truth 入口

- 项目 brief：`dev-docs/project-brief.md`
- 阶段计划：`dev-docs/stage-plan.md`
- 决策记录：`dev-docs/decisions/`
- 可复用资产：`docs/assets-map.md`

## 许可

**双轨许可（2026-08-26 用户裁决 A）**：
- **代码**（psd/ scripts/ configs/）：[MIT](LICENSE)
- **非代码资产**（论文文本/图表/报告/文档）：[CC BY-NC 4.0](LICENSE-ASSETS.md)
- 第三方许可审计清单见 [LICENSE](LICENSE#third-party-notice第三方许可审计清单)
- 训练权重研究教育用途；数据集均不随包分发（见论文 Data Availability 声明）
- 开源时机：论文投稿时随代码公开（复现主张）

## Reproduction

论文全部报告数字可由本仓库复现：外部数据集按各自许可获取（InterPet4D / Animal Kingdom / APTv2 / NTU60 frame-50），派生骨架由发布脚本从原始数据再生成（不随包分发）。完整命令链见论文附录 B（Reproduction chain）与 reports/ 下各实验报告；数据路径常量集中在 scripts/run_p05_public_real_full12.py 顶部常量块（META/TRAIN_CSV/VIDEO_DIR 等），按自身数据位置修改后即可运行下游链。预训练权重不入库（许可与体积），训练命令见附录 B。
