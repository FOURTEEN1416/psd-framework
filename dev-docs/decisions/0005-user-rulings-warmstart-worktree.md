# ADR 0005 — 用户双裁决：A2 warm-start 协议 + B-full worktree 物理隔离

> 状态: 已裁决（用户，2026-08-24 歆歆协调会话；2026-08-25 W24 落档）
> 关联: `dev-docs/handovers/W23-p05-al-warmstart.md`、`dev-docs/handovers/W24-collab-worktree.md`、`dev-docs/HANDOVER.md` §10、AGENTS.md §4 并行纪律
> 决策人: 用户（裁决辅助与建册由歆歆协调会话执行）
> 编号说明: 任务书 D4 原拟编号 0003 与既有 `0003-p02-sprint-and-metric-ruling.md` 冲突（0001-0004 均被占用），按时间顺延为 **0005**；既有裁决零改动

## 裁决原文登记（2026-08-24 协调会话）

用户在歆歆复核+裁决会话上对两项决策菜单做出选择（HANDOVER §10 同日记录）:

1. **A2 — warm-start 协议采纳**: P0.5-AL 效率实验从「冷启动」切换为「warm-start + 加噪偏移」协议——每个预算点从 `runs/p05_stgcn_bc_full/best.pt` 初始化微调；熵打分器须为上一累计预算的域内微调模型（禁止原始 best.pt 直接跨域打分）。
   - 否决项: 冷启动延续（W14 已归档负结果: b≥100 随机反超熵 7~8pp、真实池打分 softmax 饱和退化）
   - 授权载体: `handovers/W23-p05-al-warmstart.md`（「授权以本册为准」）

2. **B-full — 每窗口 git worktree 物理隔离采纳**: 根治 W14 并行窗口共用主检出导致 4 次覆盖已提交文件的事故根因。新窗口一律在自己的 worktree 开工，主检出只保留协调合并用途，跨窗产物经显式 merge 收编。
   - 否决项: B-lite（仅纪律条款、无物理隔离）
   - 授权载体: `handovers/W24-collab-worktree.md`（「授权以本册为准」）

## 生效范围

| 裁决 | 生效范围 | 机制交付 |
|------|---------|---------|
| A2 | W23 及后续全部 P0.5-AL 实验（E5 正证据候选的唯一现行协议） | W23 窗口按其任务书五步门执行 |
| B-full | 所有新开窗口（W18/W21/W22 等进行中窗口不强制迁移，按原方式收尾后自行切换） | W24: 建窗脚本 + AGENTS.md §4 + 冒烟验证 |

## 影响与后续

- B-full 机制已于 2026-08-25 由 W24 冒烟验证通过（288 测试全绿 + Junction 数据链 + tiny 训练），运维要点与卸窗陷阱见 `reports/w24-worktree-governance-2026-08-25.md`
- A2 实验结论无论正负均须如实归档（三层口径铁律不变）；若产生正证据，论文回填由后续窗口承接
