# P0.3 JIA Phase B C1 修复报告（2026-08-24）

## 1. 修复概述

**任务**：将 heldout accuracy 从 36% 提升至 ≥50% gate，实施 Phase B C1 方案：ST-GCN+BC penultimate 语义桥 + 中心化对齐。

**根因（已确认）**：特征空间域偏移，而非映射逻辑。余弦最近原型代码正确，输入特征的域分布差异导致同域精度仅 25.18%，真实段全坍缩至 single class "sit"。

**方案 B（主攻）**：使用 `runs/p05_stgcn_bc_full/best.pt`（val_acc 96.4%）的 ST-GCN+BC penultimate 特征作为新语义桥 Φ'，并在合成/真实两侧分别拟合均值后 L2 归一，消除跨域整体偏移。

**方案 A（对照消融）**：仅使用冻结 AimCLR Φ 特征 + z-score 归一化，用于量化纯对齐能救多少。

---

## 2. 实验对比数据

| 指标 | 方案 B：ST-GCN+BC | 方案 A：AimCLR |
|------|------------------|----------------|
| heldout_accuracy | **0.7382** (≥0.50 gate ✓) | 0.5691 (≥0.50 gate ✓) |
| distinct_classes | **16** (≥6 gate ✓) | 19 (≥6 gate ✓) |
| eval_side_coverage | 1.0 (≥1.0 gate ✓) | 1.0 (≥1.0 gate ✓) |
| 使用 checkpoint | `runs/p05_stgcn_bc_full/best.pt` | 冻结 AimCLR Φ (`runs/p01_aimclr_pretext/epoch120_model.pt`) |
| 特征来源 | ST-GCN+BC penultimate (BCHead 前一层) | 冻结 Φ (backbone 输出) |
| 对齐方式 | μ_syn/μ_real 独立拟合 → 减去 → L2 归一 | μ/σ 统一 z-score → L2 归一 |
| 备注 | 笔记本本次 W13-C1 任务核心改动：换 checkpoint + 保持已有对齐逻辑不变 | 消融量化：纯对齐下 heldout 仍≥0.50，但特征判别力弱于 ST-GCN+BC |

**核心归因**：ST-GCN+BC penultimate 特征的判别力（96.4% val_acc, 22 类线性可分性）优于冻结 AimCLR Φ，换用后跨域整体偏移被显著削弱， heldout accuracy 从 0.5691 提升至 0.7382（+16.9% 绝对值）。

---

## 3. 接受门状态

| 门项 | 方案 B | 方案 A |
|------|--------|--------|
| heldout_accuracy ≥ 0.50 | **PASS** (0.7382) | PASS (0.5691) |
| 真实池分布 ≥ 6 类 | **PASS** (16 类) | PASS (19 类) |
| 覆盖率 ≥ 1.0 | **PASS** (1.0) | PASS (1.0) |
| all_gates_pass | **PASS** | PASS |

---

## 4. 一键复现命令

```bash
# 方案 B（主推，默认）
python scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml

# 方案 A（消融对照）
python scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml --encoder aimclr
```

---

## 5. 提交信息

`fix(p03): use p05_stgcn_bc_full checkpoint + mean-centering alignment for Phase B C1; heldout 0.7382 > 0.50 gate`