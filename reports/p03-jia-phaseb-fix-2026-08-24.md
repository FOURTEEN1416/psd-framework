# W13 窗口 — P0.3 Phase B 修复报告（语义桥 + 22 类伪标签池交付）

> 日期: 2026-08-24 · 窗口: W13（窗口 C）· 任务书链: `W13-p03-phaseb.md` → `W13-C1-phaseb-fix.md`
> 用户裁决记录: 方案①（消费 W12 checkpoint）→ 发现产物被并发覆写不可用 → **改裁决方案 B（本窗口自训桥）**
> 状态: ✅ C1 三验收门全过，伪标签池已交付 `data/processed/p03_phaseB/`
> **指标口径: heldout 精度属【合成层】留出协议；池作用于【公开真实层】（InterPet4D 段）；真实 K9 层不在本窗口范围（AGENTS.md 规则#3）**

---

## 1. 执行摘要

Phase B 首跑质量门红灯后，经 C1 任务书与两次用户裁决，最终以「自训 ST-GCN+BC 语义桥 + 均值中心化对齐 + 余弦最近语义原型映射」达成全部验收门：

| 验收门（C1 §四） | 结果 |
|---|---|
| heldout_accuracy ≥ 0.50 | **0.6582** ✅ |
| 真实池标签分布 ≥ 6 类 | **15 类** ✅（首跑为 1 类全坍缩） |
| pytest 全绿（原 18 + 新增 ≥3） | **202 passed**（新增 12）✅ |
| 方案 A 数字入报告 | ✅ §2.2 |
| 一条命令复现 | ✅ §6 |

**交付物**: `data/processed/p03_phaseB/pseudo_pool_phaseB_22class_seed42.jsonl`（1430 行，覆盖全部过滤段，P0.4 接口逐字段对齐）+ 对齐嵌入 npz + 本报告 + 双结果 JSON。

---

## 2. 最终数字

### 2.1 方案对比（同一留出协议 seed=42，合成层口径）

| 配置 | heldout 精度 | 真实池类数 | 门 |
|------|------------|----------|----|
| A: 冻结 Φ + z-score 白化对齐 | 0.5691 | 19 | ✅ |
| **B（主攻）: 自训桥 penultimate + mean-only 中心化** | **0.6582** | **15** | ✅ |

### 2.2 归因阶梯（关键实验证据，全部当次运行）

从首跑红灯到过门，四层因素按贡献排序：

| # | 因素 | 实验对照 | 贡献 |
|---|------|---------|------|
| 1 | **合成数据源失配**（主导）：仓内 pkl 与当前生成器输出不一致——标签一致但关键点漂移（diff std=0.146，远超噪声量级 0.05）。桥按生成器分布训练，pkl 输入=域外 | 同一桥同一协议，仅换数据源：0.3800 → 0.6582 | **+27.8pp** |
| 2 | 特征判别力（C1 根因①确认）：冻结 AimCLR Φ 上 22 合成类原型近共线（两两余弦 0.9994），换自训桥特征空间 | 数据源修复后 A vs B：0.5691 → 0.6582 | +8.9pp |
| 3 | 均值中心化对齐（C1 根因②缓解）：μ_syn/μ_real 各自拟合消跨域整体偏移，单类坍缩解除（1 类 → 15/19 类） | 首跑全 sit → 现分布 top8 分散 | 质变 |
| 4 | 读出方式：线性头 96% vs 余弦原型 ~66%——原型映射有天花板但已满足 gate；LR 参照 57%（stale-pkl 域上测得） | §附证据脚本 | 备案 |

> ⚠️ 诚实披露：合成集是模板匹配任务（train/val 同模板），heldout 0.66 的绝对值系统性偏高，**不可外推真实层精度**；其作用是验证「映射管线在真值可得处工作正常」。

### 2.3 池画像（公开真实层，1430 段 = Phase A 全部过滤段）

- 覆盖率 1.0000（评估侧）≥ Phase A coverage(α=1, τ=0)=1.0 ✅
- 分布 top5: alert_down 568 / obstacle 426 / bark 180 / stand 163 / alert_sit 21
- 每行携带 Phase A 元数据（proto_idx / kappa_margin），label_source=`p03_phaseB_stgcnbc_centered_seed42`
- 桥模型: `runs/p03_phaseb_bridge/best.pt`，sha256 锁定 `394abba079…`，val_acc 96.36%@ep21（落盘复核 96.14%，元数据一致）

---

## 3. 事件记录（工程过程，供复盘）

### 3.1 并行窗口干扰（5 次，均已恢复）
未跟踪文件被删 ×1；已提交代码被 git 级回退 ×3（测试/mapper/config/脚本）；训练入口脚本被删 ×1。对策：小步提交、HEAD 秒恢复。**建议给并行窗口捎话：勿对本仓做仓级 restore/clean。**

### 3.2 W12 checkpoint 并发覆写事故
13:18–13:20 实测 `runs/p05_stgcn_bc_full/best.pt`/`last.pt` 正被并行训练写入（mtime 与检查时刻差 15s）；加载所得权重与其内嵌 val_acc=96.36% 元数据矛盾（实测 9–13%）。方案①因此不可用，触发用户改裁决 B。
**反证闭环**: 本窗口单写者落盘的桥 ckpt 元数据与权重一致（96.36% vs 96.14%，差 0.2pp<2pp 门限）——问题确系并发写而非保存机制。

### 3.3 pkl 与生成器不一致发现（⚠️ 待跨窗处置，本窗口未动共享产物）
`data/synthetic/syn_22class_100per_class_seed42.pkl`（11:40 生成，登记于 _manifest）与当前代码 `make_synthetic_dataset(spc=100,T=30,seed=42)` 输出**标签一致、关键点不同**（diff std=0.146）。影响所有以 pkl 为输入、以生成器分布为参照的消费方（含 W12 报告中引用该 pkl 的表述）。建议由协调者裁定：重生成 pkl 并更新 manifest + DATA_LOCATIONS，或登记「pkl=历史实现快照」口径。**本窗口已通过现生成同源规避，不依赖 pkl。**

### 3.4 W12 切分口径澄清（虚惊记录）
曾怀疑 `train_stgcn_bc.py::train()` 尾部截断切分导致 val 只覆盖 5 类；查实 W12 实际走 `run_p05_full.py` 的 `rng.permutation` 随机切分，无此问题。旧入口的尾部切分缺陷仍存在但未被主运行使用——备案供后续修复。

---

## 4. 口径与边界

- **合成层**: heldout 协议 = 合成分层对半切（ref 建原型+拟合 μ / probe 验证），seed=42；对齐统计仅在 ref 半区拟合，防泄漏
- **公开真实层**: 池标签无真值可验，质量以「合成层代理精度 + 分布健康度（15 类非坍缩）」佐证；三层严禁混报
- 生产池的原型用全量合成构建（gate 用 ref 半区），μ_real 用全部 1430 段拟合
- 池分布偏斜（alert_down 占 40%）如实呈现，不做再平衡——是否分层消费归 P0.5 决策

## 5. 提交清单

```
psd/training/stgcnbc_feature_extractor.py        [新建] penultimate 抽取 + 居中 + 对齐
psd/training/tests/test_stgcnbc_feature_extractor.py [新建 TDD 12 绿]
scripts/train_p03_phaseb_bridge.py               [新建] 自训桥 + sha256 锁定 + 复核
scripts/run_p03_phaseb.py                        [修改] --encoder 开关 + 同源数据 + 双路对齐
configs/p03_jia_phaseb.yaml                      [修改] bridge 段指向本窗桥
reports/p03-jia-phaseb-results{,_aimclr}.json    [新建] 两方案当次运行证据
data/* runs/*                                    [gitignore 不入库]
```

提交序列: `6958c2b`(mapper TDD) → `6a23b58`(自训桥入口) → `dfbbaa1`(数据源修复过 gate) + 本次报告提交。

## 6. 复现命令

```bash
# 自训桥（一次性基建，~5min GPU）
.venv/Scripts/python.exe scripts/train_p03_phaseb_bridge.py
# 方案 B 主实验（默认）
.venv/Scripts/python.exe scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml --encoder stgcnbc
# 方案 A 消融
.venv/Scripts/python.exe scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml --encoder aimclr
# 测试
.venv/Scripts/python.exe -m pytest psd -q   # 202 passed
```

## 7. 遗留与移交 P0.5

1. 池文件即插即用：schema 与 P0.4 池逐字段一致，`label_source` 可区分来源
2. 池分布偏斜的分层采样策略 → P0.5 自定
3. pkl 重生成裁决（§3.3）→ 待用户/协调者
4. 读出方式天花板（余弦原型 vs 线性头）如需进一步压榨精度 → 论文窗口消融素材

---

*报告编制: W13 窗口（歆歆）2026-08-24 · 依据: AGENTS.md v1.0 / W13-C1 任务书 / 用户裁决记录*
