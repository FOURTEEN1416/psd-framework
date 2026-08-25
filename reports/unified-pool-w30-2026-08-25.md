# W30 — 统一真实扩展池组装报告

> 窗口: W30（wt/W30 worktree，B-full 协议）｜日期: 2026-08-25
> 任务书: `dev-docs/handovers/NEXT-BATCH-plan.md` §W30（唯一任务书）
> 会师蓝图: `dev-docs/handovers/DATA-CAMPAIGN-plan.md` §0 收敛契约 / §3 会师路径
> 组装器: `scripts/build_unified_pool.py`（CPU-only，源只读）｜配置: `configs/unified_pool.yaml`
> 测试: `psd/data/tests/test_unified_pool.py` **22 passed**（TDD RED→GREEN 双提交留痕）

---

## 0. 执行摘要（三行版）

1. 统一池 v1 落盘 **9030 条**：APTv2 微序列 503（拓扑已映射 K9Graph 24kp）+ dog-pose 静态 8476 + DogSet 动捕 51；AK partialclass4 因**冒烟残留未过内容断言被自动隔离**（诚实登记，待 Q3b 全量落盘后重跑即收编）。
2. APTv2 17kp→K9Graph 24kp 映射定案：17 源点全消费（16 直接 + `neck→withers` 全表唯一近似），**7 槽位诚实置 NaN**（尾尖/双耳×4/下巴/喉部）；协调者遗留调研尾巴已答——**APTv2 原生覆盖死关节事件缺失 4 点中的双眼，throat 无对应**。
3. 15 帧时序策略三选一定案 = **仅预训练池**（原生 T=15 不插值不拼接，`usage_scope=pretrain_geometric`）；规则种子打标降级为 `label_status=deferred_pixel_domain`（rule_seeds 需度量 3D z 轴，APTv2 为 2D 像素，阈值不可跨域迁移——证据见 §5）。

## 1. 池产物与来源构成

| 项 | 值 |
|----|----|
| 池文件 | `runs/data_campaign/unified/real_expansion_pool_v1.pkl`（43.6MB，合并式 dict + entries） |
| 溯源 manifest | 同目录 `real_expansion_pool_v1_manifest.json`（聚合统计/映射表/honesty/config echo） |
| schema | `psd.data_campaign.unified.real_expansion_v1` |
| 条目契约 | `{sample_id, source_channel, split, topology_name, V, T, keypoints(T,V,3)f32, coords_semantic, fps_or_sampling, usage_scope, label_status, static, provenance}` |

### 来源构成（当次运行实测）

| source_channel | 条数 | 拓扑 | usage_scope | label_status | split 分布 |
|------|------|------|------|------|------|
| aptv2_c2_w26 | 503 | K9Graph(映射自 aptv2_quadruped_17kp) | pretrain_geometric | deferred_pixel_domain | train 356 / val 68 / test 79 |
| dogpose_c5_w29 | 8476 | K9Graph(原生) | augment_static | none_static_gt | train 6773 / val 1703 |
| mocap_c3_w27 | 51 | mann_dogset_21j(原生不映射) | kinematic_prior | not_applicable_prior | unsplit 51 |
| ak_public_q3b | **0（隔离）** | —— | —— | —— | 待 Q3b |

split 对账零差额：train 7129 = 356+6773；val 1771 = 68+1703；test 79；unsplit 51。

## 2. 决策① APTv2 17kp→K9Graph 24kp 拓扑映射

**官方顺序权威出处**: `train_annotations.json` categories.keypoints（W30 当次逐名核对，与 W26 报告 §5 一致）：
`left_eye, right_eye, nose, neck, root_of_tail, left_shoulder, left_elbow, left_front_paw, right_shoulder, right_elbow, right_front_paw, left_hip, left_knee, left_back_paw, right_hip, right_knee, right_back_paw`

### 映射表（17 源点 → 24 目标槽位）

| APTv2 idx | 名 | → K9 idx | K9 名 | 状态 | 语义理由 |
|---|---|---|---|---|---|
| 0 | left_eye | 20 | left_eye | exact | 左眼同名同义；**死关节事件中 dog-pose GT 缺失的 4 点之一，APTv2 原生覆盖** |
| 1 | right_eye | 21 | right_eye | exact | 右眼同名同义（同上） |
| 2 | nose | 16 | nose | exact | 鼻尖同名同义 |
| 3 | neck | 22 | withers | **approx** | 全表唯一近似点：APTv2 无独立鬐甲定义，neck 标注于颈背基部，与鬐甲（肩胛间最高点、K9Graph 根关节）位置相近但可能偏上 |
| 4 | root_of_tail | 12 | tail_start | exact | 尾根同名同义 |
| 5 | left_shoulder | 2 | front_left_elbow | positional | 前左肢近端链位对齐（肩胛点接躯干；K9Graph 肢体三点链命名粗化） |
| 6 | left_elbow | 1 | front_left_knee | positional | 前左肢中段链位对齐（APTv2 称肘 / K9Graph 简化称腕部膝位） |
| 7 | left_front_paw | 0 | front_left_paw | exact | 末端着地点同名同义 |
| 8-10 | right_shoulder/elbow/front_paw | 8/7/6 | front_right_* | positional/exact/… | 前右肢与前左肢完全对称 |
| 11 | left_hip | 5 | rear_left_elbow | positional | 后左肢近端链位对齐（髋点接骨盆侧） |
| 12 | left_knee | 4 | rear_left_knee | exact | 后左膝关节同名同义 |
| 13 | left_back_paw | 3 | rear_left_paw | exact | 后左末端同名同义 |
| 14-16 | right_hip/knee/back_paw | 11/10/9 | rear_right_* | positional/exact | 后右肢与后左肢完全对称 |

### 无法对应的点（诚实置 NaN + vis=0，下游按缺失消费）

| K9 idx | 名 | 缺失理由 |
|---|---|---|
| 13 | tail_end | APTv2 仅尾根单点，无尾尖 |
| 14/15 | left/right_ear_base | APTv2 拓扑无耳部点 |
| 17 | chin | 无下巴点 |
| 18/19 | left/right_ear_tip | 无耳部点 |
| 23 | throat | 无喉部点——呼应死关节事件（dog-pose GT 该点亦零标注） |

有效监督口径：**名义 24 点、实际可监督 ≤17 点**（其中 withers 为近似）。与 dog-pose 的"20/24"口径**不可混报**：两者缺失模式不同（dog-pose 缺双眼+withers+throat；APTv2-mapped 缺耳×4+下巴+尾尖+ throat，双眼反而可用）。

**验证**: 已知答案 spot-check（测试）+ 真实数据 seed42 抽样 5 条对回源 pkl **max|Δxy|=0.0 逐位一致** + 全池 503 条不可映射槽位 NaN 不变量扫描零违规。

## 3. 决策② 15 帧微序列时序策略三选一 —— 定案：仅预训练池

| 方案 | 正方（能得到什么） | 反方（代价/风险） | 裁决 |
|------|------|------|------|
| 滑窗拼接升采样 | 凑出 T=30 直进主管线；样本量形式翻倍 | 插值=运动学假动态，速度谱毒化 C4 保真度基线（W26 §7-b/C5 §3-a 双重否决先例）；跨段拼接产生段间跳变（W26 §7-c 明确否决）；违背"宁缺毋滥" | ❌ 拒绝 |
| 直接短序列训练支持 | 零失真；监督样本 +503 | 需改 psd/training 与 dataset 层（越本窗领地且动 W12 交付物）；T=15@unknown_fps 无法承载 ≥30 帧行为语义；与 T=30 合成层混训引入时序长度混杂变量，伤害 warm-start 对照设计；fps 未知使速度类规则不可标定 | ⚠️ 否决 |
| **仅预训练池** | AimCLR 式自监督前文本就为短片段设计（P0.1 kNN 先例证明表征可分）；真实几何监督零失真入池；与主管线 T=30 解耦，不污染实验对照；manifest 显式用途域防误用 | 对监督微调无直接贡献；价值依赖预训练路线有效性（E5 PENDING，但 ADR-0006 已把 warm-start 定为论文正证据路径，几何预训练是其自然燃料） | ✅ **采纳** |

落地：APTv2 503 条保留原生 T=15，`usage_scope="pretrain_geometric"` 固化于每条目与 manifest。

## 4. 决策③ 组装器四源汇聚与 AK 内容断言

- **AK partialclass4 隔离声明**：开工盘点证实 `partialclass4_T30.pkl` 仍是冒烟残留（pkl list len=1 vs manifest samples=172；V=17≠24），与记忆库 2026-08-25 W28-P1 警报一致。组装器实施**内容级断言**（样本数对账 + 形状契约 + 标签∈gate4），断言失败→该源整体隔离并写入 `honesty.ak_source`，绝不静默混入。`options.require_ak_full=true` 时改为硬失败。Q3b 全量产物落盘后**重跑本脚本即自动收编**。
- **拓扑异构性如实保留**：DogSet 21 关节不强行映射（超本窗范围），以 `topology_name` 字段隔离；honesty 块明示"下游必须按 topology_name 过滤，禁止跨拓扑混批"。

## 5. 规则种子打标降级声明（label_status=deferred_pixel_domain）

任务书原文括注"APTv2 轨迹(规则种子打标)"。实现前证据核查发现：`psd/data/rule_seeds.py::_frame_features` 硬依赖**度量空间 3D 坐标**（`kp[...,2]` 作 z 高度轴参与 clearance/ground/体尺度归一）+ SMAL 24 关节索引组；而 APTv2 是 **2D 图像像素 (x,y,vis)**。直接套用会把可见性标志当高度算出伪物理量，且各姿态阈值（standing_min_clearance 等）在像素域不可迁移、无校准数据。据此降级为 `label_status="deferred_pixel_domain"`：本池承载几何与预训练用途，像素域规则适配（y 翻转代理 + 体尺度归一 + 阈值再校准）归后续窗口专项处理。此为有证据的范围收缩，非功能缺失。

## 6. 与 22 类体系的覆盖关系（诚实口径）

| 层 | 覆盖 |
|----|----|
| 合成层 | 不在本池范围（syn_v2 另册） |
| 公开真实层·监督 | 当前仅 AK gate4 四类（stay/track/watch/jump ⊂ 22 类体系）——**因 Q3b 未跑，本池暂为零**；其余三源均无行为标签 |
| 公开真实层·几何/预训练 | APTv2 503 微序列（犬科 dog/fox/wolf）+ dogpose 8476 静态 GT + DogSet 51 动捕序列 |
| 真实 K9 层 | 本池不含任何真实 K9 数据 |

## 7. 三层口径声明

本报告所有数字属**公开真实层**（微序列子池/静态子池/动播先验池三分，禁止互相混报，禁止与合成层混报，禁止计入真实 K9 层）。APTv2-mapped 的"≤17/24 有效点"与 dog-pose 的"20/24 有效关节"是两套不同的缺失模式，引用时必须分开陈述。

## 8. TDD 与新鲜验证证据链

1. RED 先行：22 测试先写后实现，首次运行留痕 `ModuleNotFoundError: No module named 'scripts.build_unified_pool'`（双提交留痕）
2. 过程中测试抓到两个真实缺陷：dogpose 条目缺 T 键（补形状计算）、mocap pkl 无 sequence_id 键（id 必须取自 manifest 行——夹具按真实 schema 重演后修复）
3. GREEN：22 passed
4. 产物新鲜验证 17/17 PASS：schema 全扫 9030/9030、sample_id 全局唯一、四源计数与 split 逐项对账、AK quarantine 记录在案、真实样本映射 seed42 抽检逐位一致、NaN 不变量全池扫描、mocap/dogpose 零 NaN
5. 复现：
```
& D:\Desktop\psd-framework\.venv\Scripts\python.exe -m pytest psd/data/tests/test_unified_pool.py -q   # cwd=仓根
& D:\Desktop\psd-framework\.venv\Scripts\python.exe scripts/build_unified_pool.py --config configs/unified_pool.yaml
```

## 9. 移交与后续

| 事项 | 建议 |
|----|----|
| Q3b 全量落盘后 | 重跑 build_unified_pool.py（require_ak_full 可开 true）即收编 AK 有标分支 |
| APTv2 规则种子 | 像素域适配专项（y 翻转+体尺度归一+阈值校准），需 rule_seeds owner 窗口协作 |
| DATA_LOCATIONS 登记 | 本窗领地不含 docs/，请协调者在收编时补登 unified 池条目（指向本报告） |
| 预训练消费方 | AimCLR/warm-start 线消费 pretrain_geometric 子池时按 topology_name=K9Graph 过滤，7 个 NaN 槽位走 valid-mask |
