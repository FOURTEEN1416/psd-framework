# W14 报告 — P0.5 主动学习效率实验（短预算管线版）

> 日期: 2026-08-24 | 窗口: W14 | 状态: ✅ 管线跑通 + 曲线归档；**科学结论为负结果（熵采样未显示优势），C7 支撑状态维持 ⏳ 待 P0.5+**
> 数据: `reports/p05-al-efficiency-short-2026-08-24.json`（git `241f9db` 时点运行，总耗时 2468.5s CPU）
> 口径: **合成层**（AGENTS.md 硬规则 3）；真实池打分为公开真实层排序清单，非精度数字

---

## 1. 执行摘要

| 项 | 结果 |
|----|------|
| 管线 | ✅ TDD 17 绿（全仓回归 189 绿）；冒烟/短预算/full 三档入口 + 断点续跑 |
| 主曲线 | ✅ 2 策略 × {20,50,100,200} × 3 seeds = 6 轨迹 24 次冷启动训练，mean±std 归档 |
| 熵 vs 随机 | ❌ **熵未占优**：b=100 随机 +7.9pp、b=200 随机 +7.1pp（差距超种子噪声带） |
| 真实池打分 | ⚠️ best.pt 在 P0.4 池上 softmax 完全饱和（margin 均值 100.9），**熵信号退化**（max<1e-12），Top-K 清单不可行动 |
| full-budget | ✅ `configs/p05_al_full.yaml` 已备好（epochs 120 / GPU auto / AMP），一键复跑命令见 §7 |

## 2. 协议（预注册于实施计划，未事后修改）

- **配对增量式 AL**：同 seed 两臂共享随机初始核 b=20；增量 {+30,+50,+100} 由各臂策略以上一累计预算训得的模型打分选择；每预算点**冷启动重训**（固定初始化种子）消除累积混淆。
- **不确定性方法选型**：softmax 熵（单次确定性前向；MC-dropout 需多次前向且 GPU 被占、STGCNBC dropout=0；边际忽略 22 类尾部概率质量）。
- **数据隔离**：池 seed=20261（22×10=220 ≥ 预算上限 200）、验证集 seed=20262（22×15=330 固定 GT）。均避开 W12 seed42 训练数据，防记忆泄漏。
- **曲线指标**：best_val_acc（22 类 top-1），误差棒 = 3 seeds 的 std(ddof=1)。

## 3. 结果

### 3.1 主曲线（合成层，best_val_acc mean±std, n_seeds=3）

| 预算 | entropy | random | Δ(随机−熵) |
|------|---------|--------|-----------|
| 20   | 0.0778 ± 0.0334 | 0.0798 ± 0.0304 | +0.2pp |
| 50   | 0.0939 ± 0.0109 | 0.0909 ± 0.0455 | −0.3pp |
| 100  | 0.6990 ± 0.0895 | 0.7778 ± 0.0456 | **+7.9pp** |
| 200  | 0.8091 ± 0.0289 | 0.8798 ± 0.0185 | **+7.1pp** |

per-seed 明细与选择清单见归档 JSON（`curves_per_seed` / `runs/p05_al_efficiency/state_short/state_*.json`）。

### 3.2 真实池打分（P0.4 移交池 191 条 × runs/p05_stgcn_bc_full/best.pt）

- scored=191 / skipped=0；topk 上限如实截断至 191（预算上限 200 > 池容量，登记不虚报）。
- **负结果**：logit top1−top2 边际 mean=100.9 / min=32.6 / max=206.0 → softmax 全饱和，池最大熵 ≈ −7.7e-13（数值零）→ `entropy_degenerate=true`，Top-K 清单**不可作为标注优先级依据**。
- 佐证：合成验证集上同一模型边际均值仅 ≈10.8——边际尺度本身随域漂移了约 10×，合成域拟合的温度系数无法迁移（故未采用温度校准补丁，避免无据修补）。

## 4. 四步分析（observe → interpret → implicate → next）

**observe**：① 低预算区（20/50）两臂都停在随机水平（≈5–10%），曲线在 50→100 之间陡升；② b≥100 后随机稳定反超熵 7–8pp，超出噪声带；③ best.pt 对任何输入都给出近似 one-hot 输出。

**interpret**：
- 低平台：20 个样本训 22 类冷启动模型必然欠拟合——这是协议的预期形态，不是缺陷。
- 熵反超的机理（按可信度排序）：(a) **弱打分器问题**——b=20/50 的打分模型本身准确率仅 5–10%，其熵排序近似噪声，"按噪声选难例"劣于均匀覆盖；(b) 合成数据的"高熵样本"≈高噪声draw，先学难例损害模板特征形成，而随机采样天然保证类别/难度覆盖；(c) 种子方差大（b=100 处 ±0.09）提示 3 seeds 尚不足以分辨 <5pp 差异，但 7–8pp 的反向差距方向一致（3/3 seeds 随机更高）。
- 饱和机理：W12 全量训练 train_cls_loss≈7e-4 → logit 边际巨大 → 对域外输入同样过度自信。这与 P0.4 报告"B-1 校准必要性实证"互相印证。

**implicate**：
1. 本实验**不支持**"不确定性采样效率优于随机"——E5/C7 的支撑证据不能由本协议产生，论文侧维持 PENDING，禁止回填乐观数字。
2. 熵采样优势的经典前提是**较强打分器 + 域内校准**；冷启动+跨域代理两个前提在本仓当前资产下都不满足。
3. best.pt 直接做真实池打分器不可行——AL 闭环必须先注入少量域内标注再校准/微调打分器，这反过来强化了"100–200 片段预算"叙事的必要性（先标→校准→再选）。

**next**：
1. full-budget 复跑（§7 命令）验证结论随训练预算的稳定性；
2. 若要给 E5 产生正证据，需改协议为「warm-start 打分器 + 域内少量标注校准」，该改动属预注册变更，须用户裁决后另开窗口；
3. 真实 K9 数据到位后，C7 终验仍以真实层为准。

## 5. 双向论证（对抗自查）

**正方（熵采样应当更好）**：文献基线成熟（Settles 2009）；本协议配对设计公平；b=50 处熵的 std 显著更小（0.0109 vs 0.0455），稳定性或有优势迹象。
**反方质疑（成立）**：① 打分器与被改进模型同源同弱，误差自我强化（本实验正是此设定）；② 冷启动协议下首轮增量占预算 40%（20/50），两臂行为趋同，区分度被压缩到后半程；③ 3 seeds 对 ±8pp 级差异勉强够，对更细结论不足；④ 合成层结论外推到真实 K9 层无效（三层口径铁律）。
**裁决**：反方在当前协议下占优。正方论点中"稳定性"一签值得 full-budget 复跑时跟踪。

## 6. 过程事故登记（并行窗口干扰）

执行期间发生 4 次工作树文件被并行进程删除/回退（已提交内容均从 HEAD 恢复，零丢失）：
active_learning.py ×2、计划文档 ×1、configs/p05_al_short.yaml ×1；另误判 .venv 启动器的父子进程为重复任务而错杀一次健康扫描（自摆乌龙，非外部因素）。教训已固化：产物落盘即提交、重启前必 `git checkout HEAD --` 自查白名单文件。

## 7. 复现指引

```powershell
# 短预算版（本次结果，CPU ~41min）
.venv\Scripts\python.exe scripts\run_p05_al_efficiency.py --config configs\p05_al_short.yaml --fresh

# 冒烟（~1min 端到端）
.venv\Scripts\python.exe scripts\run_p05_al_efficiency.py --smoke

# full-budget 一键复跑（GPU 空闲后；复跑前把 config 内 RUNDATE 改当日）
.venv\Scripts\python.exe scripts\run_p05_al_efficiency.py --config configs\p05_al_full.yaml --fresh
```

测试：`.venv\Scripts\python.exe -m pytest psd/training/tests/test_active_learning.py -q`（17 passed @ 当次运行）

## 8. 移交边界

- 本窗未触碰：`docs/paper/**`、`dev-docs/decisions/**`、`*ntu*` 文件（遵守禁触令）。
- 并行窗口的 docs/paper 修改与 AK 转换脚本未纳入本窗提交范围。


---

## Errata (2026-09-05, R16)

§3.1 states "3/3 seeds 同向" for b=100; per-seed values in `reports/p05-al-efficiency-short-2026-08-24.json` show random > entropy at only **2/3** seeds at b=100 (seed 43: entropy 0.7909 > random 0.7818); 3/3 holds at b=200. The paper text already uses 2/3 (R13). Corrected here for archive consistency.
