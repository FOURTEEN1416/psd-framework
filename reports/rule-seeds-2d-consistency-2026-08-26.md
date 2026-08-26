# 2D 规则种子效果门禁报告 — W47 补救窗

**日期**: 2026-08-26  
**owner**: W47 (wt/W47 @ `95448ad`RED + `965a1d2`GREEN)  
**上游任务**: W30 移交项 (NEXT-BATCH-plan.md §W30) + W40 报告燃料第 2 条  
**对应主仓入口**: `psd/data/rule_seeds_2d.py` + `scripts/calibrate_rule_seeds_2d.py`  
**证据锚点**: `reports/rule-seeds-2d-calibration-w47.json`, `reports/rule-seeds-2d-diagnosis-w47.json`

---

## 0. 结论速览（三行版）

1. **门禁判定: pretrain_pool_mapping_suspect（降级）**——val 可判别子集 `{track,jump,stay}` clip 级一致率 **23.53%**（< 预注册阈值 40%），且低于均匀随机 33%，落入最低档。
2. **根因层一（工程）**: v1 特征族仍依赖 `withers(idx22)` 死关节——masked mean NaN→nan_to_num(0)→`clearance` 全帧恒 0 → `gait` 分支永不可达 → 全部归 `stay`（混淆矩阵实锤）。
3. **根因层二（科学）**: 即便用活关节 `front_tops` 替代 `withers`（`shoulder` 替代），`train` 四类分布 `clearance p50 ∈ [0.08,0.14]` 几乎重合 → **gate4 行为类在 2D 姿态几何特征空间原理性不可分**——规则族路线对 AK partialclass4 门禁不达标，降级合理。

---

## 1. TDD 交付留痕

| 提交 | SHA | 内容 |
|------|-----|------|
| RED | `95448ad` | 测试先行——31 个测试用例写入，ImportError 正确形态失败 |
| GREEN | `965a1d2` | 实现最小可用模块——31 绿 + 全仓 499 绿 |
| chore | `0f85a37` | 校准脚本 + 预注册门禁常量（先于实验落盘，符合 W40 先例）|

双提交留痕达成（W23 复核流程建议）。

---

## 2. 实验协议（预注册）

### 2.1 数据
- 主源: `D:/Desktop/psd-framework/runs/public_real_dataset/partialclass4_T30.pkl`（Q3b 全量，172 clips，K9Graph 24kp）
- 拆分: train 123 / val 49；gate4 分布 `watch:72 / track:46 / stay:27 / jump:27`
- **只读消费**，禁回流调参

### 2.2 引擎契约
- 入口归一层: `normalize_y_orientation(kp, y_axis="down")` —— 图像域 y-down 声明，内部 up=+y 归一
- 体尺度: bounding-box 对角线（per-frame median，平移不变）
- 地面: per-frame min(paw_y) 跨帧中位数（与 3D 版同构）
- 速度单位: **体尺度/帧**（`nominal_fps=1.0`），因 round2 §4 预警"T30 抽样帧率未知"——同单位自洽

### 2.3 阈值策略
- **行为类阈值**（`walk_min`/`run_min`/`jump_air`）: train-only 分位数拟合（见 `calibrated` JSON 字段）
- **姿态族阈值**（`standing_min_clearance`/`lying_max_clearance` 等）: **NOT_CALIBRATED**，承 3D 先验（无 gate4 监督对应）
- `RULE_TO_GATE4` 映射显式写死，`watch` 如实 `None`

### 2.4 门禁常量（预注册，修改须登记）
```json
PREREGISTERED_GATE = {
  "metric": "clip_level_agreement_on_discernable_subset_val",
  "discernable_classes": ["track", "jump", "stay"],
  "promote_threshold": 0.60,
  "demote_keep_evidence_band": [0.40, 0.60],
}
```
**参照系**: 可判别子集均匀随机 ≈ 33%；val 子集 majority(track) ≈ 50%。

---

## 3. 主结果

### 3.1 校准后 val 可判别子集（主门禁口径）

| 指标 | 值 |
|------|-----|
| n_clip | 34 |
| correct | 8 |
| wrong | 26 |
| abstain | 0 |
| **agreement_on_all** | **0.2353** |
| **preregistered verdict** | **< 0.40 → pretrain_pool_mapping_suspect** |

混淆矩阵（绝对数）:
| gt→pred | stay |
|---------|------|
| track (17 val) | 17 |
| jump (9 val) | 9 |
| stay (8 val) | 8 |

**100% 无差别判 stay**——engine 未学到任何行为差异信号。

### 3.2 train 自检（非独立，供过拟合参考）
- n=66, agreement=0.2879, confusion: track→stay 29 / jump→stay 18 / stay→stay 19
- train/val 一致率差距仅 5pp（<1 std of n=34 binomial）→ 不是过拟合，是系统性失效

### 3.3 默认阈值基线（3D 先验直迁）
- train 可判别 agreement=0.2879，混淆矩阵结构相同 → 3D 阈值在 2D 域无迁移性

---

## 4. 根因诊断（证据链）

### 4.1 层一：v1 特征族 withers 死依赖残留

| 特征 | p10 | p50 | p90 | p99 | >lie_max(0.18) 占比 |
|------|-----|-----|-----|-----|-------------------|
| clearance (withers) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% |

**确凿**: `withers(idx22)` 是死关节（ADR 死关节事件定义 idx20-23），masked mean → NaN → `nan_to_num(..., nan=0.0)` → `clearance` 全帧 0 → `gait` 分支要求 `clearance > 0.18` 永假。

`lying_composite` 仅靠 `head_norm`（live joints 14-19 有效）提供微弱变化，但仍不足以驱动 gait 分类。

**教训**: W30/W35 已发现"W35 规则引擎 clearance 硬依赖 withers"但本次 v1 设计沿用——是**重复犯 W30 已披露的结构弱点**。

### 4.2 层二：shoulder 替代后区分度不足

修复层一后（用 `front_tops{2,8}` 替代 `withers` 算 `shoulder_clearance`）：

| gate4 | p10 | p50 | p90 | frac >0.18 |
|-------|-----|-----|-----|------------|
| jump  | -0.0798 | 0.0831 | 0.2556 | 0.257 |
| stay  | 0.0203 | 0.1268 | 0.2185 | 0.226 |
| track | -0.0122 | 0.1051 | 0.2157 | 0.185 |
| watch | 0.0442 | 0.1362 | 0.2207 | 0.288 |

**p50 挤在 0.08~0.14 区间，四分位距大量重叠**——四类在单一维度 `shoulder_clearance` 上无可分性。加其他维度（centroid_speed、paw_air）亦同构（见 calibration JSON）。

### 4.3 结构性归因

AK partialclass4 的 gate4 是**注意力/注视行为类**（watching/track/jump/stay）而非**姿态类**。骨架几何特征（高度、速度、离地）天然刻画的是姿态，而非"狗在看什么/是否追踪"。这是类别体系的不匹配，而非特征工程不足。

---

## 5. 双向论证

### 正方（负结果仍有学术价值）

1. **引擎工程质量过关**: 31 测试绿（含方向语义修复回归、y-flip 不变性、死关节防御）；纯 CPU；API 兼容 3D 版
2. **负结果定位明确**: 两条根因（死关节残留 + 特征不可分）都已给出实证，不会像 W40 round2 那样停留在"机制有效但端点不赢"的模糊地带
3. **与 W40 round2 形成互证链**: round2 发现"几何适应让冻结骨干特征重塑但无监督无法导向标签"；本窗发现"姿态规则引擎在行为类上本身无判别力"——两条独立证据都指向"无监督→标签"断链需要显式监督信号
4. **W35 判例升级**: W35 的 `deferred_pixel_domain` 定性判断被本轮量化证实为正确

### 反方（降级合理性）

1. **预注册门禁自动执行**: 23.53% < 40% 阈值触发 `pretrain_pool_mapping_suspect`——无人为干预，自动裁定
2. **比随机还差**: 23.53% < 均匀随机 33%——弱标签信号比乱猜更误导
3. **watch 类完全不可判别**: 规则族对 watch 输出永远 None/abstain → 全量口径下 72/172 真值（42%）必然错误
4. **与 W30 判例一致**: 本次结果延续 W30 的 `deferred_pixel_domain` 立场，不是新问题而是**同一问题在新域上的实证验证**

---

## 6. 判定与处置

**判定**: `pretrain_pool_mapping_suspect`（预注册自动执行，无需用户裁决）

**处置**:
- ✅ 2D 规则引擎代码入库（TDD 31 绿、y-flip 不变性、方向语义修复）——引擎本体保留供未来 K9 真实域（z 轴活）或姿态型数据集使用
- ✅ 映射协议 `RULE_TO_GATE4` 标记为 **mapping_suspect**，禁止在 supervision pipeline 直接消费
- ⏸️ 2D 弱标签降级为**预训练池**候选，等待未来校准证据或新特征族
- 📋 本窗产出不入监督管线、不入 tab2 公开真实列

---

## 7. 下一步建议（交协调者/下游窗口裁量）

1. **若要救此路线**: 需帧级姿态人工小标注集（预算 ~100 帧，覆盖四类行为场景）做 posture 阈值真校准；或放弃姿态七类，直接面向 gate4 设计行为规则族（如 head-tail 相对运动、注视方向 proxy）
2. **引擎维护**: `rule_seeds_2d.py` 继续保留在库（API 稳定 + 测试全集绿），但文档注释加注 "NOT validated on AK partialclass4"
3. **判例更新**: W30 `deferred_pixel_domain` 升级为 "quantitatively confirmed on AK partialclass4"——证据入 `dev-docs/decisions/`

---

## 8. 三层口径声明

本报告所有数字属 **公开真实层**（AK partialclass4）。禁止与合成层（simu）、真实 K9 层（k9）混报。

---

*报告落盘: `reports/rule-seeds-2d-consistency-2026-08-26.md`*  
*实验证据: `reports/rule-seeds-2d-calibration-w47.json`*  
*诊断证据: `reports/rule-seeds-2d-diagnosis-w47.json`*