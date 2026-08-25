# W31 tab3「−自监督预训练」消融（冒烟）

> 日期: 2026-08-25 | 模式: CPU 冒烟（tiny 档，仅验证管线，非科学结论） | 指标口径: **合成层**（val_acc, 22 类）
> warm 臂初始权重来源: 公开真实层（P0.1 AimCLR @ InterPet4D, epoch120_model.pt）——两层口径分别披露

## 设计

- 数据: 合成 2×22=44 样本（W12 口径, T=30, seed=42, 切分 8:2）
- 两臂: scratch（随机初始化）vs warm（加载 P0.1 encoder_q 权重）；同 seed 下头初始化逐位相等、切分与洗牌一致，唯一差异 = encoder 初始权重
- 训练: STGCNBCTrainer 复用, 等预算（早停关闭）

## 结果（best_val_acc, mean±std over seeds）

| 臂 | mean | std | n_seeds | per_seed |
|----|------|-----|---------|----------|
| scratch | 0.0000 | 0.0000 | 1 | [0.0] |
| warm | 0.0000 | 0.0000 | 1 | [0.0] |

## 判读

- 方向判定: Δ = mean(warm) − mean(scratch)
- ⚠️ 冒烟档数字无科学意义——只证明管线端到端可跑通；full 档待 relay ALL_DONE 后点火。

## 复现命令

```bash
# 冒烟（CPU）
python scripts/run_ablation_pretrain.py --smoke
# full（GPU, 排 relay 之后）
python scripts/run_ablation_pretrain.py --config configs/ablation_pretrain.yaml
```
