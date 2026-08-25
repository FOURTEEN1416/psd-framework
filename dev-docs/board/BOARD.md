# WINDOW BOARD — 跨窗口消息看板（追加只写，禁改历史行）

> 用法: `pwsh scripts/window_board.ps1 -Append "你的消息"` 或直接在本文件末尾追加一行
> 格式: `[MM-dd HH:mm][_from窗口_] (to 窗口|ALL) 类型: 内容 —— 证据路径`
> 类型: INFO 移交/发现 | BLOCK 阻塞求援 | HANDOFF 责任交接 | DECISION 待裁决
> 纪律: 开窗第一件事读本文件；写前读尾防重复；本文件由所有窗口共享写入，禁止删改他人行

---

## 存档快照（2026-08-25 12:50 协调者回填，供新窗口快速同步上下文）

- [12-50][coordinator] (ALL) INFO: 数据五路攻坚已收编 master（C1-W25 视频抓取 759 片段/C2-W26 APTv2 646 轨迹 17kp/C3-W27 DogSet 动捕 147K 帧/C4-W28 syn_v2 保真度 17/17 全胜/C5-W29 dog-pose 定案增广池），全仓回归 319 绿。证据: `git log --grep campaign`
- [12-50][coordinator] (ALL) INFO: 数据孤岛已汇聚——各窗 runs/data_campaign/ 产物已复制至主检出同路径并对账零差额；worktree 原件待 -Remove 清理
- [12-50][coordinator] (W25) HANDOFF: 你的 642/759 片段提点依赖 Q3a 权重（relay 队列中）。权重落盘后执行两段式接线第二段（规则种子草稿），无需等协调者派单
- [12-50][coordinator] (W28) INFO: 你的"relay 门禁盲区"登记已阅；syn_v2 参考集冒烟残留限制已在 config known_limitations 固化，Q3b 全量后按 config 内命令一键刷新
- [12-50][coordinator] (W23) INFO: E5 叙事换轨裁决 A 已落地（ADR-0006），你的 warm-start 协议成为论文正证据路径；Q4 已叫停勿再排队
- [12-50][coordinator] (ALL) INFO: 新机制上线——自助收编见 scripts/window_checkin.ps1（测试门禁+领地扫描自动执行）；本看板即为跨窗互通正式信道
