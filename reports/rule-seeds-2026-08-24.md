# W6 报告 — 规则引擎粗标（物理层先验种子生成器）

> 日期: 2026-08-24 | 窗口: W6 | 状态: ✅ 完成（P0.3 锚点备料）
> 交接依据: `dev-docs/handovers/W6-rule-seeds.md` | 配置: `configs/rule_seeds.yaml`
> 复现命令: `python scripts/make_rule_seeds.py --config configs/rule_seeds.yaml`

---

## 1. 开工前置裁决记录（路径 b 前提证伪）

交接文档路径 b 预设「InterPet4D 原生标注类别」。**四重独立证据一致证伪**：

| # | 证据 |
|---|------|
| 1 | 官方 README schema：smal_npy 仅 9 个几何键，无标注键 |
| 2 | 本机实测 226 npz 键名逐一核对吻合，`.cache` 仅 HF 下载缓存 |
| 3 | W3 加载器 docstring：「无行为标签 → dog ID 代理标签」 |
| 4 | `dev-docs/research/RESEARCH_DATA_BLOCKADE_SOLUTION.md` §1.1：「无行为标签（v1 数据集限制）」 |

触发交接文档 §6 升级条款后上报用户；用户委托决策，**裁定方案 3**：物理先验类别体系先行打通管线（YAML 可配置），合成数据到位后第二迭代切换路径 a。

## 2. SMAL 关节语义 — 几何实测验证（非官方命名）

InterPet4D 不提供关节名称表。跨 12 只狗平均相对位置 + 单帧站立姿态双重验证：

| 功能组 | 关节索引 | 几何证据 |
|--------|---------|---------|
| 前肢链 ×2 | {0,1,2}, {6,7,8}（paw→mid→top） | 站立帧 paw z≈0，肩 top z≈0.20m |
| 后肢链 ×2 | {3,4,5}, {9,10,11} | 髋 top z≈0.15m < 肩高（犬类解剖一致）|
| 尾 | 12=尾根, 13=尾尖(置信度常态为 0) | 尾根位于臀部高位，尾尖后上方延伸 |
| 头簇 | {14..21}，鼻=16(最靠前)，下巴=17(其正下) | 头簇恒居最高带 z≈0.46–0.51m |
| 背参考(荐部) | 22 | 双肩之间躯干最高点 |
| 颈参考 | 23 | 头下方颈胸交界 |

左右身份未定（各 clip 世界系水平旋转不同被抹匀）；全部规则只依赖功能组，不依赖左右。
**z 轴竖直性核验**：225 个有效 clip 的荐部-四爪高度差全部落在 0.257–0.604m（中位 0.348m），无需逐 clip 重力重对齐。

## 3. 类别体系与规则族（物理先验 7 类 + unknown）

| 类别 | 判定规则 | 关键阈值（YAML 可调）|
|------|---------|---------------------|
| lying 卧姿 | clearance<0.18 或复合证据(clearance+0.5·head)<0.75 | `posture.*` |
| sitting 坐姿 | 髋/肩离地比 <0.55（先于卧姿判定——坐姿头仍高） | `sitting_max_hip_ratio` |
| standing 站立 | clearance ≥0.35 且低速 | `standing_min_clearance` |
| walking 走 | 质心水平速度 ∈ [0.30, 1.20) 体长/s 且躯干离地 | `speed.walk_min` |
| running 跑 | 质心速度 ≥1.20 | `speed.run_min` |
| rise_transition 过渡 | \|d(clearance)/dt\|>1.5/s，尖峰 ±5 帧膨胀；升=rise_up/降=lie_down | `transition.*` |
| jump 跳跃 | 四爪**最低点**离地 >0.25（全腾空语义）且躯干超站立线+0.15 | `jump.min_air_clearance` |

优先级：jump > transition > gait > sitting > lying > standing > unknown。
体尺度归一：躯干四顶点 {2,5,8,11} 每帧最大两两距离的时序中位数；轨迹平滑窗 5 帧。

## 4. 质检统计（全量 225 clip / 137,710 帧 + 40 clip 抽样）

抽样明细: `data/seeds/qc_sample.csv`（seed=42，满足交接 ≥30 要求）；汇总: `data/seeds/seed_summary.json`

### 4.1 数据集帧占比（公开真实层口径）

| 类别 | 占比 | 备注 |
|------|------|------|
| sitting | 36.3% | 人犬互动场景主态 |
| walking | 23.4% | |
| standing | 21.7% | |
| running | 6.2% | 室内真跑少数，合理 |
| rise_transition | 5.7% | 偶发事件 |
| jump | 4.9% | 含真实跳跃与残余误判，P0.3 置信度过滤可压 |
| lying | 1.6% | **新鲜证据**：该池犬只少卧姿；旧调研「以卧为主」结论源自 kpm 不同数据池，不适用本池 |
| unknown | 0.1% | |

### 4.2 规则命中与质量指标

| 规则 ID | 帧覆盖* | | 置信度 | p10 / p50 / p90 |
|---|---|---|---|---|
| gait_walk | 61.1% | | clip 平均置信度 | 0.731 / 0.798 / 0.877 |
| sitting_posture | 49.6% | | 高置信 clip (≥0.7) | **96.4%** |
| standing_posture | 44.6% | | 种子段总数 | 4,945 |
| gait_run | 14.1% | | 段数/clip p50 / p90 / max | 21 / 34 / 70 |
| jump_airborne | 12.2%† | | 有效 clip | 225/226（剔除 1 个全 NaN，同 P0.1 口径）|
| rise_up / lie_down | 7.5% / 7.4% | | | |
| lying_posture | 2.5% | | | |

\* 规则命中按段帧数计且段合并携带并集规则，故总和 >100%，仅作相对参考。
† jump 段规则命中含被吸收邻段的并集扩散，实际 jump 标签帧占 4.9%。

### 4.3 一致率指标声明

InterPet4D 无原生行为标签 → **一致率不可算**（交接 §3 第 4 步为条件式）。替代质量证据：
① 上表置信度分布；② 合成骨架回归测试 11 项（见 §5）；③ `qc_sample.csv` 可供人工抽检复核。

## 5. 缺陷记录与修复（全程 TDD，11/11 passed）

| # | 缺陷 | 根因 | 修复 + 回归测试 |
|---|------|------|----------------|
| 1 | 全量首跑 jump 占 55.4% | 四爪离地用均值，摆动腿抬高均值越过 0.08 阈值 | 改 min 语义（全腾空）+ 阈值 0.25；`test_partial_paw_lift_never_trigger_jump` |
| 2 | running 曾达 26.5% | 步态信号用爪速（摆腿周期伪迹）+ 阈值未标定 | 换质心水平速度 + 按 225 clip 实测分位标定（walk 0.30/run 1.20）|
| 3 | transition 曾达 14.1% | rate_min=0.8 过敏感（实测 p90=0.92） | 提至 1.5 |
| 4 | 规则 ID 存储截断（`standing_postur`） | NPZ 结构化数组 rules 字段 U64 < 最长联合串 66 字符 | U128；复测截断段数=0 |

## 6. 边界合规声明

- 写入文件：`psd/data/rule_seeds.py`、`psd/data/tests/test_rule_seeds.py`、`configs/rule_seeds.yaml`、`scripts/make_rule_seeds.py`、本报告 —— 全部在白名单内
- 未触碰：`*smq*`(W4)、`psd/models|training`、`docs/paper/**`(W5)、`external/**`、`DATA_LOCATIONS.md`、`research/**`、`project-brief.md`
- `.venv` 只读 import（numpy/pandas/pyyaml 均已存在，零新装包）；`data/seeds/**` 为生成物已 gitignore；禁 push 遵守

## 7. 三层指标口径

本产出属**公开真实层**（InterPet4D v1）伪标签种子，非合成层、非真实 K9 层；供 P0.3 时序锚点使用时须保持该口径标注。

## 8. 未验证项与移交 P0.3 建议

- [未验证] jump 4.9% 中真实跳跃与误判的比例（无真值可对照；建议 P0.3 以置信度 ≥0.8 过滤后再消费）
- [未验证] 段粒度较碎（p50=21 段/clip），下游锚点抽取建议叠加最短持续时长过滤（≥0.5s）
- [未验证] 类别映射到 stage-plan 22 类的具体方案（属 P0.3 设计域，本窗口不做）
- 路径 a（K9 合成数据 22 类）磁盘路径仍未登记；`synthetic/`（gt.txt+单 mp4）与 `generated/`（空）初查未见骨架集，需用户提供口径后启用第二迭代
