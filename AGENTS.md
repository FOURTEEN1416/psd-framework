# AGENTS.md — PSD-Framework Agent Constitution

> 状态: v1.0（2026-08-23 建仓）
> 方法论继承自 k9-training-system AGENTS.md（sliver-vibe-coding 框架），此处只保留本仓库必需条款

## 1. 项目身份

学术研究仓库：物理-语义解耦低资源动物行为识别框架（论文 + 基座方法论）。
产品功能一律不做（归 k9-training-system）。Truth root: `dev-docs/`。

## 2. 硬规则

1. **GitHub-First 调研（零容忍）**：技术调研完全禁止 WebSearch；必须使用 GitHub 工具链（搜索 → awesome 目录发现 → 逐仓验证活跃度/Issues），违反触发漂移处理
2. **Truth 单一性**：每个概念只有一个 owner 文档，禁止重复 truth
3. **三层指标口径**：合成 / 公开真实 / 真实 K9 分别汇报，禁止混报
4. **无新鲜验证，无完成声明**：所有实验结论必须有当次运行证据 + 报告归档 `reports/`
5. **外部方法仓库**进 `external/`（gitignore）：不修改其内部实现；适配/封装代码写在本仓库 `psd/` 包内
6. **跨仓边界**：只允许文档指针指向 k9-training-system，禁止跨仓 import；代码复用走 `docs/assets-map.md` 显式移植
7. Conventional Commits 中文描述；代码注释语言跟随用户最新消息

## 3. Owner Map

| 模块 | 路径 |
|------|------|
| 数据加载 | `psd/data/` |
| 模型（AimCLR / SMQ / TCL / 姚青迁移 / 骨干） | `psd/models/` |
| 训练与评估管线 | `psd/training/` |
| 入口脚本 | `scripts/` |
| 实验配置 | `configs/` |
| 阶段计划 | `dev-docs/stage-plan.md` |
| 决策记录 | `dev-docs/decisions/` |
| 评估报告 | `reports/` |

## 4. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-23 | 建仓初始化，继承上游硬规则（GitHub-First / truth 单一性 / 三层口径） |
