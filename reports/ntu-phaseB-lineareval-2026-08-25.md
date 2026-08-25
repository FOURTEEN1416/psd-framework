# W33 NTU Phase B 线性评估报告（DRAFT——评估执行被阻塞中，数字区未产生）

> **指标口径：公开基准层**（NTU60；与合成 / 公开真实 / 真实 K9 各层严格分账）
> 日期: 2026-08-25 | 执行窗口: W33 (wt/W33) | 状态: **基建就绪 ✅ / 执行 🚧 被两事件阻塞**
> 任务书: 用户指令 + `dev-docs/handovers/W9-ntu-repro.md` §3 | 上游: `reports/ntu-phasea-2026-08-24.md` v1.1

## 1. 一页结论

| 事项 | 状态 |
|---|---|
| 评估协议核实与配置保真 | ✅ 官方 linear_eval 配置逐键移植，护栏测试锁定 |
| 评估基建（脚本/融合/测试） | ✅ TDD 双提交 `3aa6450`(RED)/`fa698ba`(GREEN)，新增 43 绿、全仓 366 绿 |
| NTU 数据可用性 | ✅ 恢复并全量 verify PASS（§4 事故记录） |
| **三流范围缺口（决策上报）** | ⚠️ **79.18% = 三流融合；本仓仅有 joint 流 checkpoint**（§2） |
| GPU 占卡 | 🚧 relay Q1 HALT 停链致门禁条件无法自然满足（§5） |
| joint 单流线性评估数值 | ⏳ 待占卡执行（~3-5h） |
| bone/motion pretext + 三流融合数值 | ⏳ 待用户裁决范围后执行 |

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
| E1 | joint linear eval（现有 ckpt） | ~3-5h |
| E2/E3 | bone / motion pretext 各 300ep | ~24h × 2 |
| E4/E5 | bone / motion linear eval 各 100ep | ~3-5h × 2 |
| E6 | 三流融合出数 | 分钟级 |

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

## 5. GPU 门禁阻塞（BLOCK 已登看板）

relay v2 于 17:44:49 `Q1_al_full` HALT 停链：科学执行实际完成（6 轨迹 + curve +
`reports/p05-al-efficiency-full-2026-08-25.json`），但内容级校验 verify=False 两次，
疑校验契约与 AL full 产物形态错配（W18 领地）。后果：
- Q2-Q3c 未执行；state.json 无 status=ALL_DONE → 本窗任务书占卡前提无法自然满足
- GPU 当前实际空闲（仅桌面基线 ~2GB）
- 待协调者/W18 判读 Q1 + 定续跑方案 + 定 W33 占卡时点

## 6. 数值结果区（⏳ 占位——执行后回填，禁止提前声明）

```
joint 单流 best_top1:        [待执行]
motion 单流 best_top1:       [待执行]
bone 单流 best_top1:         [待执行]
3s 融合 top1/top5:           [待执行]
判据: 3s top1 ≥77.18% ?      [待执行]
单流过程观测（非验收判据）: joint vs README 74.34 参考带 [待执行]
```

## 7. 一条命令复现序列

```powershell
# 前置: 数据 verify
.\.venv\Scripts\python.exe scripts\fetch_ntu_data.py --verify --dest data\ntu60_frame50

# E1 joint 线性评估（GPU 空闲时）
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
