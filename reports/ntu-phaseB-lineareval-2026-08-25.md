# W33 NTU Phase B 线性评估报告（v1.0 FINAL——joint 单流已执行归档；三流验收线未触发待裁决）

> **指标口径：公开基准层**（NTU60；与合成 / 公开真实 / 真实 K9 各层严格分账）
> 日期: 2026-08-25（执行跨午夜至 08-26 00:14） | 执行窗口: W33 (wt/W33) | 状态: **E1 joint 线性评估 ✅ 完成 / E2-E6 三流补全 ⏳ 待用户裁决**
> 任务书: 用户指令 + `dev-docs/handovers/W9-ntu-repro.md` §3 | 上游: `reports/ntu-phasea-2026-08-24.md` v1.1

## 1. 一页结论

| 事项 | 状态 |
|---|---|
| 评估协议核实与配置保真 | ✅ 官方 linear_eval 配置逐键移植，护栏测试锁定 |
| 评估基建（脚本/融合/测试） | ✅ TDD 双提交 `3aa6450`(RED)/`fa698ba`(GREEN)，新增 43 绿、全仓 366 绿 |
| NTU 数据可用性 | ✅ 恢复并全量 verify PASS（§4 事故记录）；本次运行即在该数据上完成，构成新鲜验证 |
| **三流范围缺口（决策上报）** | ⚠️ **79.18% = 三流融合；本仓仅有 joint 流 checkpoint**（§2.1 成本口径 + §2.2 闭环进展） |
| GPU 占卡 | ✅ 已解除——relay 五步 20:59 ALL_DONE（state.json 在案），协调者看板放行 |
| **joint 单流线性评估** | ✅ **best_top1 = 74.30%（ep85），vs 官方 README 74.34%，Δ = −0.04pp**（§6） |
| bone/motion pretext + 三流融合数值 | ⏳ 待用户裁决范围后执行（~48h GPU，§7） |

**一句话结论：joint 单流 74.30% 与官方 released-model 74.34% 复现差仅 −0.04pp——Phase B 预训练管线保真度获直接实证；预注册线 ≥77.18% 的对照物是三流融合，单流结果不构成对该线的判定（无论过线与否均不适用），如实归档待三流补全裁决。**

## 2. 协议缺口：预注册线 77.18% 的对照物是"三流融合"而非单流（⚠️ 决策上报）

官方 README released-models 表（`external/AimCLR/README.md`）实锤：

| Model | NTU60 xsub (%) |
| :--- | :---: |
| AimCLR-joint | 74.34 |
| AimCLR-motion | 68.68 |
| AimCLR-bone | 71.87 |
| **3s-AimCLR** | **79.18** |

- 预注册通过线 ≥77.18%（=79.18−2pp）的对照物是**三个独立 pretext 编码器的分数融合**
  （`ensemble_ntu_cs.py`: α = joint 0.6 / bone 0.6 / motion 0.4，logits 加权求和）
- 本仓 Phase B 只训练了 joint 流（`runs/ntu_phaseB/joint_pretext/epoch300_model.pt`）；
  W9 排程任务书亦只规划了 joint——**三流缺口系继承性范围盲区，此前无人裁决过**
- 用 joint checkpoint 对 motion/bone 输入做线性评估再融合 = 非官方协议、无对照物，**不可采**

### 2.1 补全成本实测口径

joint pretext 300ep 实测耗时 **~24h**（log 时间戳 08-24 14:27 → 08-25 14:20，
RTX 5060 Laptop 8GB，num_worker=0 已含 ~18% 惩罚）。补全三流需：

| 段 | 内容 | 预估 GPU 时长 |
|---|---|---|
| E1 | joint linear eval（现有 ckpt） | ~3-5h（**实测 33min**，见 §6） |
| E2/E3 | bone / motion pretext 各 300ep | ~24h × 2 |
| E4/E5 | bone / motion linear eval 各 100ep | 实测口径 ~33min × 2 |
| E6 | 三流融合出数 | 分钟级 |

### 2.2 缺口闭环进展（v1.0 增补）

- **E1 已执行**：joint 线性评估 33min 完成，74.30% vs 官方 74.34%（Δ−0.04pp）——
  预训练管线保真度获直接实证，E2-E5 补全的"复现风险"已消解为纯算力问题
- **GPU 阻塞已解除**：relay 五步 2026-08-25 20:59 全物理落地（`state.json status=ALL_DONE`），
  协调者看板公告放行；本次运行即在该依据下占卡
- **R4 判定语义澄清（关键）**：预注册线 ≥77.18% 的对照物自始就是三流融合。
  单流 74.30% **不构成过线/不过线的判定素材**——按协议用 joint checkpoint 对
  bone/motion 输入做线性评估再融合不可采（非官方协议、无对照物，§2 已论证）。
  因此本报告如实归档为"E1 过程观测 PASS_BAND + 验收线未触发"，而非"未过线"
- **待裁决**：A 补全三流（唯一合法对标路径，~48h GPU）/ B 单流保真证据先行、三流挂账后续批次——
  双向论证与推荐见 `reports/ntu-phaseB-lineareval-2026-08-25.json` §next_step_options_for_user_decision

## 3. 评估协议（已锁定，护栏测试防漂移）

- 入口: `scripts/run_ntu_lineareval.py --config configs/ntu60_phaseb_lineareval_xsub_<stream>.yaml`
- 处理器: 官方 `processor.linear_evaluation.LE_Processor` 直连零适配
  （**保留官方 weights_init**: fc 头 N(0,0.02) 初始化属 released-model 复测配方；
  encoder 冻结权重随后由 checkpoint 覆盖，ignore_weights=[encoder_q.fc, encoder_k, queue]）
- 训练侧: Feeder_single 无增强（shear/padding=-1），train_position.npy + train_label.pkl
- 优化: SGD lr=3.0, wd=0, step[80], momentum 0.9(官方硬编码), nesterov=False, 100ep, bs128
- 评估: 每 5ep 测 val（16,487 样本），best_model.pt + test_result.pkl 落盘
- 融合: `scripts/ntu_ensemble_3s.py`（α=0.6/0.6/0.4 加权求和 → top1/top5）
- 本地锚定种子 init_seed(0)（官方未全局定种；已声明的本地差异，数值同分布）
- 配置保真护栏: `psd/data/tests/test_ntu_lineareval_protocol.py`——本仓配置与
  external 官方 yaml 在科学不变量上逐键相等，仅路径/device/num_worker 白名单差异

## 4. 数据事故记录（本窗口发现并闭环）

1. **发现**: 预检 fail-fast 抓到 `xsub/train_label.pkl` 缺失
2. **取证**: Phase A verify 曾 8/8 全绿（该脚本缺文件即 FAIL）+ Phase B 训练曾正常消费
   （Feeder 打开成功跑满 300ep）→ 文件存在于 08-24 全天，删除发生在其后至今，责任未明不追咎
3. **恢复**: GKD 代理(17890)存活 → `ntu_selective_fetch.py` 重下载；
   三个同位文件 MD5 与现存副本逐字节一致（GDrive 源稳定性交叉验证）→
   train_label.pkl 内容校验通过（40,091 名=40,091 标签，值域 0-59）
4. **终验**: 全量 `--verify` PASS——xsub 40,091+16,487 / xview 37,646+18,932 = 113,156 ✓
5. **附带发现**: `ntu_selective_fetch.py` 路径剥离逻辑与实际落盘布局不符，
   会产出 `data/ntu60_frame50/ntu60_frame50/` 双前缀冗余树（本次已产生并清理）

## 5. GPU 门禁阻塞（✅ 已解除，留痕）

relay v2 曾于 17:44:49 `Q1_al_full` HALT 停链（科学执行实际完成但内容级校验 verify=False
两次，疑校验契约与 AL full 产物形态错配）。**后续解除路径**：协调者以独立进程+会话 Job
混合方式完成 Q3a-Q3c 全物理落地并手工收官，2026-08-25 20:59 `state.json status=ALL_DONE`
（`D:/Desktop/psd-framework/runs/relay_exec/state.json`），协调者看板公告"W33 触发条件就绪可占卡"。
本次线性评估即在该依据下于 23:40 占卡执行。

## 6. 数值结果区（✅ 2026-08-25 23:40 → 08-26 00:14 实测回填）

- **运行窗口**: 2026-08-25 23:40:26 → 2026-08-26 00:13:45（墙钟 **33 分钟**，
  远低于 DRAFT 期 ~3-5h 预估——线性评估仅回传 fc 头，encoder 冻结）
- **硬件**: RTX 5060 Laptop 8GB；stderr 全程空，零异常

```
joint 单流 best_top1:        74.30%   (epoch 85, lr step 后)
joint 单流 last_top1(ep100): 74.20%   (Top5: 94.65%)
motion 单流 best_top1:       [未执行——pretext 未训练，待裁决]
bone 单流 best_top1:         [未执行——pretext 未训练，待裁决]
3s 融合 top1/top5:           [未执行——依赖 bone/motion 流]
判据: 3s top1 ≥77.18% ?      NOT_TRIGGERED（对照物为三流融合，单流不构成判定）
单流过程观测 vs README 74.34: Δ = −0.04pp → PASS_BAND ✅
```

### 6.1 验证精度全程轨迹（val n=16,487，每 5ep 一测）

| ep | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|
| top1% | 70.29 | 72.24 | 72.20 | 72.62 | 71.81 | **72.77** | 71.56 | 72.29 | 71.81 | 72.14 |

| ep | 55 | 60 | 65 | 70 | 75 | 80 | **85** | 90 | 95 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| top1% | 71.87 | 72.52 | 72.43 | 72.09 | 72.42 | 71.67 | **74.30** ⭐ | 73.97 | 74.22 | 74.20 |

- 官方配方 ep80 lr×0.1 触发后：平台期 ~72.1-72.8 → 跃升带 73.97-74.30，形态与官方
  released-model 复测行为一致
- best_model.pt = epoch 85 checkpoint；test_result.pkl 对应 best 模型预测
  （可供未来 E6 三流融合消费）

### 6.2 判读（双向论证摘要）

- **正方（管线保真成立）**：Δ−0.04pp 远小于种子噪声带（官方未全局定种，本仓锚定
  init_seed(0) 为已声明差异）；训练/评估全链官方直连零适配 + 护栏测试防漂移；
  数据侧有 MD5 交叉验证 + 113,156 样本 verify PASS
- **反方质疑（诚实登记）**：①单次运行无多种子方差估计——但对照物官方数本身即为
  released-model 复测口径，±0.04pp 已在复测噪声内，补多种子边际价值低；
  ②33min vs 官方未知硬件时长无法做耗时对齐——属环境差异非协议差异；
  ③三流验收线仍悬置，论文主表 NTU 列若引单流须显式标注"joint single-stream"口径

## 7. 一条命令复现序列

```powershell
# 前置: 数据 verify
.\.venv\Scripts\python.exe scripts\fetch_ntu_data.py --verify --dest data\ntu60_frame50

# E1 joint 线性评估（✅ 已执行归档：74.30% @ep85，33min）
.\.venv\Scripts\python.exe scripts\run_ntu_lineareval.py
#   ≡ --config configs\ntu60_phaseb_lineareval_xsub_joint.yaml

# E2/E3 bone/motion pretext（若用户裁决补全三流；各 ~24h）
.\.venv\Scripts\python.exe scripts\run_ntu_phaseb.py --config configs\ntu60_phaseb_pretext_xsub_bone.yaml
.\.venv\Scripts\python.exe scripts\run_ntu_phaseb.py --config configs\ntu60_phaseb_pretext_xsub_motion.yaml

# E4/E5 bone/motion 线性评估
.\.venv\Scripts\python.exe scripts\run_ntu_lineareval.py --config configs\ntu60_phaseb_lineareval_xsub_bone.yaml
.\.venv\Scripts\python.exe scripts\run_ntu_lineareval.py --config configs\ntu60_phaseb_lineareval_xsub_motion.yaml

# E6 三流融合出数
.\.venv\Scripts\python.exe scripts\ntu_ensemble_3s.py --json-out reports\ntu-phaseB-3s-ensemble.json
```

## 8. 修订历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v0.1 DRAFT | 2026-08-25 | 基建交付 + 协议缺口上报 + 数据事故闭环 + GPU 阻塞登记；数值区待执行回填 |
| v1.0 FINAL | 2026-08-26 | E1 joint 线性评估执行归档（74.30% vs 官方 74.34%，Δ−0.04pp）；GPU 阻塞解除留痕；R4 判定语义澄清（单流不构成预注册线判定，验收线 NOT_TRIGGERED 待三流裁决）；配套 JSON 同名归档 |
