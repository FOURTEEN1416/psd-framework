# W7 交接文档 — P0.2 SMQ 分割救援（接管 W4 全部领地）

> **你是 P0.2 救援窗口**（用户指令「重启 W4」即由你执行本任务书——窗口叫 W4 还是 W7 不重要，**本文档是唯一任务书**）。读完本文档即开工。
> 必读顺序：本文档 → `AGENTS.md` → `dev-docs/HANDOVER.md` v1.4 → `dev-docs/handovers/W4-p02-smq.md`（原任务定义）→ `reports/p01-aimclr-2026-08-23.md` §4（坍缩教训）。
> **性质**：这不是新功能开发，是**失败诊断救援**——先分诊，后动手，禁止盲目换方向。

---

## 1. 现状盘点（2026-08-24 复核结论，全部有据）

### 1.1 失败证据（两轮评估同结果）

| 项 | 实测值 | 来源 |
|----|--------|------|
| 预测段 | **单段 [0,640]**（覆盖全 episode） | `reports/p02-smq-iou.json` / `p02-smq-iou-v3-kmeans.json` |
| mean_matched_iou | **0.20**（=128/640，即 GT 每段与全集的平凡重叠） | 同上 ×4 episodes |
| boundary F1 / P / R | **全 0.0** | 同上 |
| 随机基线 IoU | **0.40–0.43（模型大幅跑输随机）** | 同上 |
| checkpoint | `runs/p02_smq/models/epoch-30.model` | JSON 内记录 |

### 1.2 未提交产物（接管清单，第一步必须先打检查点）

```
M   requirements.txt              （W4 增量依赖）
??  configs/p02_smq.yaml
??  psd/data/smq_input.py
??  psd/data/tests/test_smq_input.py
??  psd/training/segment_iou.py
??  psd/training/smq_runner.py
??  psd/training/tests/
??  scripts/train_smq_segmentation.py
??  scripts/eval_smq_segmentation.py
??  reports/p02-smq-iou*.json + p02-vis-*.png     （评估产物）
??  external/SMQ/                                  （已克隆；确认 .gitignore 覆盖则不入库）
```

**开工第 0 步**：审阅上述文件 → 打 WIP 检查点提交（`feat(wip): P0.2 SMQ 救援前现场保全——W4 未提交产物入库`），防止丢失。此后每完成一个诊断实验都提交一次。

## 2. 任务目标（一句话）

查明"模型输出单段"的根因并让 SMQ 分割达到**显著优于随机基线**的可验收状态；若确诊 SMQ 方法本身不适配，整理证据上报用户裁决后再切换 stage-plan §2 预案（滑动窗口+相似度分割）。

**验收标准不变**（stage-plan）：episode 边界 IoU 评估 + 可视化 + 报告归档 `reports/p02-<日期>.md` + 一条命令复现。

## 3. 诊断路线图（按序执行，每步留证据）

### Step A：分诊——模型坏 vs 协议盲（最优先）

当前 GT 是**人为拼接协议**：5 个 clip 各 128 帧拼接成 T=640，GT 边界=拼接点。注意：不同狗做相似行为时，拼接点在运动特征上可能本来就不显著——**"输出单段"可能是模型的正确感知而非故障**。

两个必做实验：
1. **运动词用量体检**：统计 checkpoint 在评估集上的运动词直方图/熵。若词表坍缩到极少数词 → 是模型侧坍缩（回看 P0.1 E1-E7 教训：官方 `weights_init(N(0,0.02))` 前科）；若词分布健康但边界分为零 → 查阈值/后处理。
2. **强边界 sanity check**：用**规则种子伪 GT** 替换拼接 GT 重评一次（见 §4）。若在行为真边界上同样输出单段 → 模型/管线故障实锤；若显著变好 → 拼接协议不可检，换评估基准而非改模型。

### Step B：候选根因清单（逐项排查，按先验概率排序）

| # | 假设 | 检查方法 |
|---|------|---------|
| 1 | 后处理把碎段合并成一段（min-length/合并逻辑过激） | 读 smq_runner 输出原始边界 vs 最终段 |
| 2 | 运动词坍缩（同 P0.1 初始化前科） | Step A-1 直方图 |
| 3 | 边界阈值定死/标定不当 | 阈值扫描曲线 |
| 4 | 输入归一化/通道序错位（NTU 视图适配错误） | 单 clip 过管线可视化中间量 |
| 5 | checkpoint 欠训练（epoch-30 是否早停） | loss 曲线 + 加训对照 |
| 6 | 评估脚本切片 bug | 用手工构造已知边界的假数据单测 eval |

### Step C：修复或升级

- 三振出局铁律生效：同一假设连续修 3 次无效 → 停手，重开系统性诊断并上报用户。
- 确诊 SMQ 不适配（如方法前提与本数据根本冲突）→ 写证据报告（含 Step A/B 全部数据）→ **等用户裁决**再切预案，不许自行切换。

## 4. 新评估基准（跨窗口协同，本窗口红利）

W6 已交付全量规则种子（`reports/rule-seeds-2026-08-24.md`，commit `aaa1e1c`）：

- **伪 GT 行为边界** = 高置信种子段的边界序列。消费规则：置信度 ≥0.8、最短持续 ≥0.5s（W6 报告 §8 建议）、口径标注「公开真实层-物理先验伪标签」。
- 在该基准上重算 IoU/boundary-F1 作为第二口径，与拼接协议并列汇报（两层口径并存披露，不择优单报——对齐论文 experiment-skeleton E1 的双口径纪律）。
- 种子数据位置：`data/seeds/**`（gitignore 生成物），读取用 `psd/data/rule_seeds.py` 公共接口，**只读 import，禁改其实现**。

## 5. 边界（继承 W4 白名单，互斥对象更新）

| 类型 | 路径 |
|------|------|
| ✅ 可写 | W4 原白名单全部（`.venv` 主用、`requirements.txt`、`external/SMQ`、`psd/*` 的 smq 相关文件、`scripts/*p02*`、`configs/p02*`、`reports/p02-*`、stage-plan 仅 P0.2 行状态列） |
| ❌ 禁触 | `psd/data/rule_seeds.py` 及其测试（W6 产物，只读 import）；`docs/paper/**`（W5）；`reports/rule-seeds-*`；`psd/models/jia*`、`scripts/*p03*`、`configs/p03*`、`reports/p03-*`（W8 新领地）；其余同 W4 禁触清单 |

> 与 W8 并行纪律：两者共享 `.venv` 只读原则不变；git 提交只动各自白名单；遇 `index.lock` 等待重试。

## 6. 完成标准

- [ ] 根因结论写入报告（含 Step A 两个分诊实验的证据）
- [ ] 双口径评估结果表（拼接协议 vs 种子伪 GT）+ ≥2 可视化
- [ ] 显著优于随机基线，或附用户裁决记录的降级/切换方案
- [ ] 一条命令复现序列 + 报告归档 + Conventional Commits 中文提交

---

*交接编制: 歆歆（规划会话）2026-08-24 · 依据: HANDOVER v1.4 §8 / stage-plan v1.2*
