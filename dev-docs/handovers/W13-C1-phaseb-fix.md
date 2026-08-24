# W13-C1 Phase B 映射修复任务书

> 触发: 2026-08-24 Phase B 首跑质量门红灯（`reports/p03-jia-phaseb-results.json`）
> Owner: W13 窗口（窗口 C）
> 前置: 已提交 `6958c2b`（TDD 18 绿），本任务在其上修复映射质量

---

## 一、已确认根因（协调者诊断，勿重复排查）

| # | 症状 | 证据 | 定性 |
|---|------|------|------|
| 1 | 同域精度仅 25.18% | `mapping_quality_synthetic_layer.heldout_accuracy=0.2518`（合成 ref→probe，同编码器同域） | **冻结 AimCLR 骨干特征判别力不足**（旁证：P0.1 kNN 仅 20.89%） |
| 2 | 真实段全坍缩到 sit | `pool_label_distribution_22class={"sit":1430}`；桥接诊断 6 个原型全→sit | **合成↔真实嵌入域偏移**（domain shift） |
| 3 | 预处理无罪 | 两侧均为 `build_segment_view(target_t=64)` + 同一 encoder | 排除管线 bug |

结论：**病根在特征空间，不在映射逻辑**。余弦最近原型代码正确，喂进去的特征不行。

## 二、修复方案（B 主 + A 对照消融）

### 方案 B（主攻）：换 P0.5 ST-GCN+BC penultimate 特征作 Φ'

- 权重: `runs/p05_stgcn_bc_full/best.pt`（合成层 val_acc 96.4%，22 类线性可分性已验证）
- 取分类头前一层输出作为嵌入（D 维以实际模型为准）
- 合成侧与真实侧用同一 wrapper 提取，口径一致
- **必须叠加均值中心化对齐**: μ_syn、μ_real 分别计算，各自减去后再 L2 归一 + 余弦匹配（消跨域整体偏移）

### 方案 A（对照消融）：现有 AimCLR 特征 + 中心化白化

- 只加「各自减均值 → 除 std → L2 归一」再匹配
- 目的不是达标，是量化"纯对齐能救多少"，写进报告作归因

### 禁止事项

- 禁止改 `psd/training/jia_phaseB_mapper.py` 的映射逻辑（它是对的）
- 禁止动 `runs/p05_stgcn_bc_full/best.pt`（只读加载）
- 禁止碰其他窗口领地：`*smq*` / `docs/paper/**` / `dev-docs/decisions/**`
- 新增代码进 `psd/training/stgcnbc_feature_extractor.py`（新文件）

## 三、执行步骤

1. **TDD 先行**: 写 `psd/training/tests/test_stgcnbc_feature_extractor.py`
   - 用例1: checkpoint 加载成功且输出维度正确
   - 用例2: 同输入两次提取结果一致（确定性）
   - 用例3: 中心化对齐函数——同分布输入输出不变，偏移分布被拉回
   - CPU 可跑（构造随机张量 mock，不依赖真权重跑 CI）
2. 实现 `stgcnbc_feature_extractor.py`: 加载 best.pt → forward 到 penultimate → numpy 输出
3. 改 `scripts/run_p03_phaseb.py`: 加 `--encoder {aimclr,stgcnbc}` 开关，默认 stgcnbc；两条路径都过中心化对齐
4. 跑方案 B 主实验 + 方案 A 消融各一次
5. 报告 `reports/p03-jia-phaseb-fix-2026-08-24.md`: 两方案数字对比 + 归因分析
6. Conventional Commits 中文提交（fix(p03): ...）

## 四、验收门（硬性）

- [ ] `heldout_accuracy ≥ 0.50`（方案 B）
- [ ] 真实池标签分布 ≥ 6 个不同类别（不再单类坍缩）
- [ ] 全部 pytest 保持绿（原 18 + 新增 ≥3）
- [ ] 方案 A 数字入报告（无论是否达标，作归因证据）
- [ ] 一条命令复现: `python scripts/run_p03_phaseb.py --config configs/p03_jia_phaseb.yaml`

## 五、失败预案

若方案 B 跨域仍坍缩（池分布仍 <3 类）：
- 升级 CORAL 二阶统计对齐（协方差对齐），代码量 +30 行
- 仍失败则上报用户裁决，不得擅自降验收门
