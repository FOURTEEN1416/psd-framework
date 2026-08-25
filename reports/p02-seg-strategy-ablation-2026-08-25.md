# P0.2 分割策略三臂消融（tab3 −无监督分割行 · W34 设计落地）

> **实验**: p02-seg-strategy-ablation | **窗口**: W38 (wt/W38) | **执行**: 2026-08-26 00:21 纯 CPU
> **层级口径**: 公开真实层（InterPet4D smal_npy，种子伪 GT 协议，需披露非人工 GT）
> **设计来源**: `docs/paper/experiment-skeleton.md` tab3「−无监督分割」行 W34 入册最小版
> （c1 报告 §9 排查定案：P0.2 既有对照系等段数随机切分 null 而非滑窗方法臂，本实验补齐）
> **数据**: `reports/p02-seg-strategy-ablation-2026-08-25.json`（同目录，逐 episode 全量证据）

---

## 1. 三臂定义（同 episode 同 GT 同匹配数学，唯一变量 = 边界放置策略）

| 臂 | 定义 | 来源 |
|----|------|------|
| **A. SMQ** | 运动词量化自适应边界（E-C checkpoint 只读推理，drop/无 vocab merge 与 P0.2 定稿路径逐步一致） | P0.2 既有 |
| **B. uniform（新方法臂）** | 等段数均匀切分：段数 = max(len(pred_smq), 2)，array_split 语义无缝覆盖 | 本实验新增 |
| **B′. grid（附属旁证）** | 固定网格平铺 stride=patch_size=16（W34 设计"或"字第二选项；不参与判据主判定） | 本实验新增 |
| **C. null** | 等段数随机切分蒙特卡洛期望（n_sims=200） | 直接引用 `psd/training/segment_iou.py::random_baseline_mean_iou` |

等段数控度对称性：B 与 C 段数同为 max(len(pred_smq), 2)，两臂差异只剩边界放置策略本身。

## 2. Seeds 规范对齐（逐项写入 JSON `seeds_alignment`）

| 项 | 值 | 与 P0.2 对齐方式 |
|----|-----|----------------|
| eval clip 抽选种子 | 42 | 同 `select_eval_clips(seed=eval_seed)` |
| episode 分组 | 4 episodes × 5 clips（dog 互斥校验） | 同 `group_into_episodes` |
| 种子伪 GT 消费规则 | conf ≥ 0.8 且持续 ≥ 0.5 s（30 fps） | 同 `build_seed_gt_episode`（W6 §8 规则） |
| MC null 种子 | eval_seed + episode_id | 同 eval 脚本逐 episode 公式 |
| uniform/grid 臂 | 确定性（无 RNG） | — |
| 匹配数学 | Hungarian mean IoU + boundary F1@tol=16 | 复用 `segment_iou.py`（零重复实现） |
| 设备策略 | 进程级 `CUDA_VISIBLE_DEVICES=""` 于 torch 导入前置空 | 纯 CPU，零占卡 |

**复现门（协议对齐实证）**：SMQ 臂与 null 臂均与既有定稿报告 `reports/p02-smq-iou-eC-seeds.json` **逐位一致**
（SMQ: 0.4196 / 0.423 / 0.4483 / 0.54；null: 0.2908 / 0.3145 / 0.3359 / 0.3499）——推理确定性 +
同 episode 同协议同 ckpt 三重实证，本实验三臂可比性成立。

## 3. 结果

### 3.1 主表（mean matched IoU，4 episodes）

| 臂 | ep1 | ep2 | ep3 | ep4 | **aggregate** |
|----|------|------|------|------|---------------|
| A. SMQ 自适应边界 | 0.420 | 0.423 | 0.448 | **0.540** | **0.4577 ± 0.0488** |
| B. 等段数均匀切分 | 0.361 | 0.376 | **0.452** | 0.405 | 0.3986 ± 0.0347 |
| B′. 固定网格平铺（旁证） | 0.494 | 0.418 | 0.458 | 0.443 | 0.4532 ± 0.0274 |
| C. 随机切分 null | 0.291 | 0.315 | 0.336 | 0.350 | 0.3228 ± 0.0224 |

Boundary F1@tol=16（确定性三臂）：SMQ 0.3425 / uniform 0.3963 / grid 0.2229。

### 3.2 判据判定（W34 预注册：均匀窗显著劣于 SMQ 且 ≥ 随机 null 才构成边界增益消融）

| 门 | 判据 | 实测 | 结果 |
|----|------|------|------|
| 方向门 | uniform < SMQ 的 episode 数 ≥ ⌈2n/3⌉=3 | 3/4（ep3 反超） | ✅ |
| 幅度门 | 均值差超出臂间噪声量级 | Δ=0.0591 > max(std)=0.0488 | ✅ |
| 下界门 | uniform ≥ null | 0.3986 ≥ 0.3228 | ✅ |

**boundary_gain_ablation_established = true**（预注册判据通过）。

## 4. 双向论证

**正方（判据成立的三条理由）**：
1. 三臂谱系清晰分层：结构化分割（0.40–0.46）全面高于随机 null（0.32），分割策略本身对结果有实质影响——这正是 tab3 该行要论证的"无监督分割不可约简为任意切分"。
2. 等段数控制下 SMQ 对均匀窗的增益（+0.059）超出两臂 seed 间噪声量级且方向 3/4 一致，符合预注册双门。
3. 复现门逐位通过，协议对齐无解释空间；对照系直接引用 P0.2 蒙特卡洛实现，无第二套数学。

**反方质疑（必须随数字一并披露）**：
1. **网格臂平价**：固定网格平铺（不参与判据的旁证臂）聚合 0.4532，与 SMQ 差仅 −0.0045，在 std 内打平；ep1 上网格甚至高于 SMQ。即：对最强的平凡基线，SMQ 自适应边界的边际收益在本协议下是边缘性的。
2. **指标依赖**：boundary F1 上均匀窗反而最高（0.3963 vs SMQ 0.3425）——规则网格天然密集过采样边界容差带；IoU 与 F1 两指标方向不一致，单指标引用有断章取义风险。
3. **样本量**：n=4 episodes，方向门 3/4 的置信度有限；ep3 单点反超说明逐 episode 层面并不稳。
4. **GT 口径**：种子伪 GT 是物理先验伪标签而非人工标注（公开真实层披露义务），绝对数值不可外推到真 GT 场景。

## 5. 结论与 tab3 回填素材

**一句话结论**：按 W34 预注册判据，均匀滑窗第三臂显著劣于 SMQ 且不低于随机 null，**tab3 −无监督分割行的边界增益消融成立**；但须并列披露"固定网格旁证臂与 SMQ 统计等效"，主张措辞应落在"结构化分割优于任意切分、SMQ 在等段数控制下最优"而非"自适应边界大幅领先一切平凡基线"。

**tab3 行建议文案**（交协调者/W36 择用，非本窗领地不入表）：

> − 无监督分割 | 分割策略消融（seeds 伪 GT，4 episodes）：SMQ 0.458±0.049 > 等段数均匀滑窗 0.399±0.035（预注册判据：方向 3/4、Δ=0.059>噪声）> 随机切分 null 0.323±0.022 → 结构化分割不可约简为任意切分 ✅；⚠️ 固定网格旁证臂 0.453±0.027 与 SMQ 统计等效（Δ=0.005<std），自适应边界对最强平凡基线的边际收益为边缘性，论文措辞以"优于任意切分 + 等段数下最优"为准 | C4 | 报告 `reports/p02-seg-strategy-ablation-2026-08-25.md`

## 6. 复现命令

```powershell
& "D:\Desktop\psd-framework\.venv\Scripts\python.exe" `
    scripts/seg_strategy_ablation.py `
    --config configs/seg_ablation_p02.yaml `
    --out reports/p02-seg-strategy-ablation-2026-08-25.json
# 单元测试（31 绿）:
& "D:\Desktop\psd-framework\.venv\Scripts\python.exe" -m pytest psd/training/tests/test_seg_strategy_ablation.py -q
```

## 7. 领地与禁触合规声明

- 新增：`scripts/seg_strategy_ablation.py`、`configs/seg_ablation_p02.yaml`、`psd/training/tests/test_seg_strategy_ablation.py`（TDD RED→GREEN 双循环留痕）、本报告 + 配套 JSON。
- 只读复用未改动：`scripts/eval_smq_segmentation.py`、`psd/training/segment_iou.py`、`psd/data/smq_input.py`、`psd/training/smq_runner.py`、`external/SMQ`（gitignore 资产自主检出复制）、`runs/p02_smq_eC/models/epoch-30.model`（同上）、全部既有 `reports/p02-*`。
- E-C checkpoint 经本窗 runs/ 本地副本加载（上游资产只读）；全程 CPU，GPU 零占用（W33 线性评估不受影响）。
