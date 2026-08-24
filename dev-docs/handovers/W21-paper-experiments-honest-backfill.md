# W21 任务书 — 论文实验章诚实修正与矩阵同步（E5/E6/tab3/Limitations）

> 窗口: W21（全新独立窗口，纯文档工作零 GPU）
> 日期: 2026-08-25 | 编制: 协调者歆歆
> 领地: `docs/paper/experiment-skeleton.md`、`docs/paper/outline.md`、`docs/paper/conclusion-limitations.md`、`docs/paper/review-log.md`（W17 已收官，领地移交本窗）
> ⚠️ 核心纪律: AGENTS.md 硬规则 3（三层口径禁混排）+ 无新鲜验证无完成声明 + 诚实原则优先于论文好看

## 1. 任务目标（一句话）

清除 experiment-skeleton 中 E5/E6 的过期占位污染，把今晚新产生的真实证据（C1 成本实验、AL 负结果、已完成消融）按三层口径诚实回填，同步刷新 Claims-Evidence 矩阵与 Limitations。

## 2. 必读输入（按序）

1. `dev-docs/HANDOVER.md` 快速启动节 + `AGENTS.md`
2. `reports/w14-p05-al-efficiency-2026-08-24.md`（AL 负结果全文——E5 的事实基础）
3. `reports/c1-decouple-cost-2026-08-24.md`（C1 证据全文——E6 的事实基础）
4. `reports/p03-jia-phasea-2026-08-24.md` §噪声消融 + `reports/p04-tcl-2026-08-24.md`（tab3 已完成消融的数据源）
5. `docs/paper/experiment-skeleton.md` 当前版（找出全部过期占位）
6. `docs/paper/outline.md` §2 Claims-Evidence 矩阵 + §8 风险登记册

## 3. 执行清单

### Step 1 — E5 诚实重写（当前内容违规：P0.4 数字冒充 AL 结果）
- 删除错误的 `0.691 ± 0.013` 占位
- 如实写入 W14 结果：合成层短预算档，熵采样未显示优势（b=100 随机 +7.9pp、b=200 +7.1pp，3/3 seeds 同向）；真实池打分 softmax 全饱和诊断（margin 均值 100.9）；full-budget 待 GPU 队列
- C7 claim 状态改为 ⏳ 并标注 `[PENDING-用户措辞裁决]`——候选方向两条供用户选：①降级为"探索性发现"写入分析节；②移出贡献列表。**本窗不得擅自定稿措辞**
- fig4 引用保留（负结果曲线照画，见 W22 任务书）

### Step 2 — E6 回填 C1 真证据
- 写入：解耦臂墙钟 7.32×（保守区间 ≥3×，CPU 路径污染风险已披露）、精度 +2.27pp、三 seed 全向一致、标注单元数两臂打平（如实）
- 口径标注：**合成层 small 档**；full 档 GPU 排队中，趋势矛盾以 full 为准（预注册条款）
- 补 Y′ 合理性论证段素材：ADR 0002 v1.1 预注册的 K9 报表粒度差业务动机已可成文（outline §8 R10 有底稿）

### Step 3 — tab3 消融表映射（大部分数据已存在，逐行核对来源后填入）
| 消融行 | 数据源 | 状态 |
|--------|--------|------|
| 种子噪声注入 {10,20,30}% | P0.3 报告（30% 仅降 3.1pp） | ✅ 可填 |
| −主动学习（熵→随机对照） | W14 JSON 曲线 | ✅ 可填（负结果如实） |
| 锚点/伪标签迭代开关 | P0.3 Phase A + P0.4 迭代曲线 | ✅ 可填（引用具体报告节号） |
| −自监督预训练 | ⏳ 无直接实验，标 PENDING 不编造 | ⏳ |
| −无监督分割 | ⏳ 同上 | ⏳ |
- 每格必须标注层级与来源报告路径；没有实验的格子写 PENDING，禁止用相邻数字充数

### Step 4 — Claims-Evidence 矩阵与风险册同步（outline.md）
- C1: ⏳→🟡（合成层实证支持，full 待确认）；C7: 保持 ⏳ + 负结果注记
- R2: 🔴→🟡；新增风险行（如适用）：AL 负结果对"低资源管线完整性叙事"的影响评估

### Step 5 — Limitations 刷新（conclusion-limitations.md）
新增三条今晚产生的诚实发现：
1. 冷启动弱打分器场景下不确定性采样无优势（附 softmax 饱和跨域诊断——best.pt margin 均值 100.9 vs 合成域 10.8）
2. AK 公开真实层的结构性约束（标签覆盖 4/12 类 + PE 骨架帧数不足），自提取管线为缓解方案（结果待 Q3 接力）
3. 相应更新 rebuttal 预案

### Step 6 — 提交
- Conventional Commits 中文；一次逻辑一提交（E5/E6 修正在先，tab3 其次，矩阵+Limitations 最后）
- 完成后在 review-log.md 登记本轮修订

## 4. 领地边界

**可写**: 上列四文件
**禁触**: `docs/paper/figures/**`（W22 领地）、`docs/paper/introduction.md`/`method.md`/`related-work.md`（等 Q3 数字落地后统一终稿窗口处理）、一切代码/reports/dev-docs（只读）

## 5. 完成标准

- [ ] 全文检索无 `0.691 ± 0.013` 冒充 E5/E6 的残留
- [ ] 每个 E5/E6/tab3 数字可溯源到具体 reports 文件+字段
- [ ] C7 措辞裁决项显式标记待用户
- [ ] 三层口径零混排（逐表自查）

## 修订历史
| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-25 | 建册：E5/E6 过期占位清理 + tab3 映射 + 矩阵/Limitations 同步 |
