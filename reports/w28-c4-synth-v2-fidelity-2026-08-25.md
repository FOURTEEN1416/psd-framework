# W28/C4 — 合成管线保真度 v2 (syn_v2) 交付报告

> 窗口: W28 (worktree `wt/W28`, 目录 `D:\Desktop\psd-framework-W28`)
> 日期: 2026-08-25
> 任务书: `dev-docs/handovers/DATA-CAMPAIGN-plan.md` §2-C4 + §0 收敛契约
> 领地: `psd/data/synth_stgcn_v2.py`(新增) / tests / `configs/syn_v2_*` / 本报告 —— 零越界
> 三层口径: **本实验全部属合成层自证**（生成分布 vs 公开真实层参考分布的距离），无任何行为识别精度数字，严禁与训练指标混排

---

## 1. 结论速览

| 保真度指标 (vs 公开真实层参考) | v1 方法论基线* | syn_v2 | 改善 |
|---|---|---|---|
| 角度边缘 KS 距离 (17 关节均值) | 0.6125 | **0.3009** | −50.9% |
| 速度谱直方图 L1 差 (17 关节均值) | 1.8878 | **1.2988** | −31.2% |
| 逐关节胜负计数 | — | — | **17/17 全胜 × 双指标** |

\* 对照基线 = v1 方法论(姿态模板+sin 波+iid 高斯噪声)在 17 关节归一化域的忠实移植（`make_v1style_baseline_17j`，独立实现）。**不与旧 `psd/data/synth_stgcn.py` 直接对比的原因**：v1 原生为 24 关节 K9Graph 体尺度坐标，与参考(coco17 归一化坐标)存在拓扑+坐标系双重失配；对照设计控制拓扑、隔离"分布拟合"单一变量。

证据 JSON: `reports/w28-c4-synth-v2-fidelity-2026-08-25.json`（当次运行, git_sha `bb094ee`, seed=42 可复现）。

## 2. 数据基础事实链（开工实勘发现，重要）

任务书假设参考源为"Q3b 真实提点产物"。实测发现：

1. **GPU relay 从未启动 Q3b**：主仓 `runs/relay_exec/state.json` 的 `steps=[]`，transcript 显示执行器自 2026-08-25 09:05 起持续巡检等待 GPU（NTU 任务占满 ~7.6GB 显存），Q3a/Q3b/Q3c 均未执行。
2. **现有 `partialclass4_T30.pkl` 为冒烟残留**：仅 **1 clip × 30 帧**（6.5KB），而 manifest 登记 172 视频/172 样本计划；`extract_quality.json` 缺失（stage_extract 未完整跑过）。
3. **拓扑失配证据**：pkl 内 keypoints 为 `(30,17,3)` coco17 归一化坐标(xyn+conf)，但管线代码 `run_p05_public_real_pipeline.py` 的 `EXPECTED_KPTS=24` 且 fail-fast 要求 dog-pose 权重——说明冒烟期实际用了 COCO 权重默认值提点。
4. **relay 门禁盲区上报**：`Invoke-Step -VerifyPaths` 只做 `Test-Path` 存在性检查，不校验样本量/拓扑/质量字段。若 Q3b 未来以残缺产物落盘仍会被判 OK 并提交。建议协调窗在 relay 脚本加内容校验（本窗领地不含 relay 文件，未改动）。

### 处置决策（已在开工时向用户呈报并获推荐路径）
不等 GPU（C4 属 CPU 任务，禁触 GPU 队列）。以现有 n=1 参考打通**方法论全链**：参考源完全参数化（CLI `--reference-pkl`），全量产物落盘后替换路径重跑同一命令即刷新全部数字。报告如实标注 n=1 统计功效局限。

## 3. 方法论

### 3.1 分布统计定义
- **逐关节角**：每关节由固定三元组 `(parent, self, child)` 定义三点内角，值域 [0,π]，无方向角周期性问题。拓扑表 `COCO17_PARENT`（nose←肩锚定，眼←鼻，耳←眼，肘←肩，腕←肘，髋↔肩闭环，膝←髋，踝←膝）；叶关节(耳/腕/踝)用祖父补齐三元组，度量肢体折叠程度。17 关节全覆盖。
- **帧间速度谱**：逐关节帧间位移幅值 `||Δp||`（xy 分量）。

### 3.2 syn_v2 参数化拟合（闭式解，无迭代优化）
逐关节逐坐标 AR(1) 平滑模型 `p_t = mu + φ(p_{t−1}−mu) + s·ε`，由实测三统计量(mu, σ_pos², Var(v))闭式求解：

```
φ   = 1 − Var(v) / (2σ_pos²)      (clamp [0, 0.99])
s²  = σ_pos²(1 − φ²)
```

推导：稳态下 Var(v) = s²·2/(1+φ) 与 σ_pos² = s²/(1−φ²) 两约束两自由度联立即得。**性质：生成序列的位置边缘方差与帧间速度方差同时精确匹配实测**——这正是保真度指标的两个靶点。conf 通道按实测逐关节 bootstrap 池采样。

本次实测参考参数：`σ_pos_mean=0.0066`，`φ_mean=0.755`（强时序相关——真实提点的帧间平滑性，恰是 v1 iid 噪声模型缺失的属性，解释了 v2 全面胜出的机理）。

### 3.3 已知裁剪（诚实声明）
**角度边缘未显式建模**（列为 v3 迭代项）：角度由骨架几何从位置耦合导出，AR(1) 模型通过位置拟合间接带动角度分布匹配（实验证实角度 KS 也大幅下降），但不保证任意参考下的收敛。备选方案（逐关节切向增益校正）经双向论证被否：n_ref=30 帧下角度 std 估计本身极噪，显式校准易过拟合噪声且翻倍实现面；当前阶段管线稳健优先。

## 4. TDD 留痕（双提交纪律）

| 阶段 | 提交 | 内容 |
|---|---|---|
| RED-1 → GREEN-1 | `b6447ff` → `58ae8e3` | 统计层 12 测试（手算已知值：直角 π/2、平角 π、KS D=1/3 等），新鲜验证失败后最小实现 |
| RED-2 → GREEN-2 | `5a6b188` → `7377597` | 拟合闭式解恢复真值(T=4000 已知 AR(1))、生成契约、seed 可复现、科学优势断言、**旧模块字节级冻结保护**(md5 `0d67fcaf...`) |
| RED-3 → GREEN-3 | `bb094ee` | 实验入口证据 JSON 契约 + 非 coco17 参考 fail-fast 门禁 |

最终 `pytest psd -q`: **全仓 317 passed**（含本窗新增 23），零破坏。旧行为保护测试常驻回归。

## 5. 实验结果明细

- 生成规模: gate4 类 × 8 样本/类 = 32 clips × 30 帧（synv2 与 v1style 各一份, 同 seed）
- 参考统计: `σ_pos_mean=0.0066`, `φ_mean=0.755`
- 最差关节（v2 口径 KS）: 眼部 1/2 与耳 3/6 区（0.41~0.45）——低可见度关节（mean vis 0.03~0.29）提点抖动大，符合预期
- 最好关节: 髋/腕区（0.07~0.23）
- 完整逐关节数值见证据 JSON `ks_per_joint` / `vel_hist_per_joint`

## 6. 双向论证

**正方（v2 有效）**：双指标 17/17 全胜非偶然——闭式解数学上保证速度方差匹配；φ≈0.76 捕获的真实时序平滑性直接压缩了速度谱差异；角度 KS 经位置拟合间接受益（−50.9%）。

**反方（结论强度受限）**：① n_ref=1 clip×30 帧，KS 绝对值无统计显著性，单视频可能不代表 AK 犬科全域分布；② 角度改善可能部分来自"参考角度分布本身窄"（静态 watch 类主导），换高动态类参考幅度可能缩水；③ conf bootstrap 只复刻边缘分布不复刻时空相关。→ 应对：所有数字限定为"同管线相对比较"；全量 Q3b 重跑前不得写入论文正文（可入方法学附录）。

## 7. 移交与后续

1. **Q3b 全量产物到位后的重跑命令**（一行刷新全部数字）：
   ```
   & "D:\Desktop\psd-framework\.venv\Scripts\python.exe" -m psd.data.synth_stgcn_v2 `
     --reference-pkl <新 pkl 路径> --output-json reports/w28-c4-synth-v2-fidelity-vnext.json `
     --samples-per-class 8 --seed 42 --bins 20
   ```
   ⚠️ 若全量产物为 dog-pose 24 关节拓扑，需先适配拓扑映射（模块已 fail-fast 保护，不会静默出错）。
2. **relay 门禁盲区**建议协调窗处置（VerifyPaths 加内容校验）。
3. v3 迭代候选：角度切向增益显式校准 / 多模态位置模型(GMM)/ 类条件参数化。
4. 会师路径就位：`make_synthetic_dataset_v2` 输出契约与 v1 兼容(keypoints/label/label_name/boundary/frame_dir)，下游 STGCNBCDataset 可消费；接入 unified real-expansion pool 时按 DATA-CAMPAIGN §3 执行。

## 8. 白名单自检

| 文件 | 性质 |
|---|---|
| `psd/data/synth_stgcn_v2.py` | 新增（未触碰 synth_stgcn.py 任何字节，md5 冻结测试兜底）|
| `psd/data/tests/test_synth_stgcn_v2.py` | 新增测试 |
| `configs/syn_v2_fidelity.yaml` | 新增配置（含重跑指引与局限登记）|
| `reports/w28-c4-synth-v2-fidelity-2026-08-25.md/.json` | 报告+当次运行证据 |
