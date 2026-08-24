# W14-W17 抢救清单（协调者复核产物，2026-08-24 晚）

> Owner: `dev-docs/rescue-plan-2026-08-24.md`（单一 truth）
> 触发: 四窗口复核发现 W15 未完成、W16 映射被证伪、调试脚本散落
> 关键事实: `df_action.xlsx` 已定位 `D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\annotation\df_action.xlsx`（12285 B, 140 动作全表）

---

## 0. W16 映射证伪记录（最高优先级事实）

W16 的序数映射假设「AK 标签值 1-22 按索引对齐 PSD 类」**已被 df_action.xlsx 证伪**：

| W16 错误映射 | AK index 真实动作 |
|-------------|------------------|
| "1" → sit | **Attacking（攻击）** |
| "2" → down | **Attending（注意）** |
| "5" → stand | **Being Carried In Mouth（被叼）** |
| "13" → bite | Carrying In Mouth（叼东西） |

**权威语义映射表（基于 df_action.xlsx action 列，唯一合法依据）**：

| PSD 类 | AK 动作 (index) | 强度 |
|--------|----------------|------|
| sit | Sitting (108) | 强 |
| down | Lying Down (70), Lying on its side (73) | 强 |
| stand | Standing (116) | 强 |
| stay | Keeping still (68) | 中 |
| bark | Barking (3) | 强 |
| bite | Biting (8) | 强 |
| watch | Attending (2), Sensing (102) | 中 |
| apprehend | Attacking (1) | 中 |
| retrieve | Carrying In Mouth (13) | 中 |
| scale | Climbing (16)（obstacle 与 scale 同源，二选一或合并披露） | 中 |
| jump | Jumping (15) | 强 |
| track | Exploring (45) / Chasing (14) | 弱 |
| heel / sit_up / alert_sit / alert_down / escort / recall / guard / release / search_blind | 无等价物（K9 特有） | ❌ 零覆盖 |

**裁决**：公开真实层实验必须改为**部分类协议**——仅在 12±2 个可覆盖类上评估，论文 tab2 加脚注披露子集协议（参照 P0.1 dog-ID 代理探针的先例）。22 类全量训练在 AK 上不可能诚实完成。

---

## 1. 窗口状态与抢救任务

### ✅ W14（C7 主动学习）— 已验收通过，无抢救项
- 负结果如实登记；full-budget 配置待 GPU。
- **遗留排队项**: NTU 训练结束后复跑 `configs/p05_al_full.yaml`。

### ✅ W17（文献+图表）— 已验收通过，无抢救项
- 11 条题录补齐 + fig1/fig2 矢量图落盘。

### 🟡 W15（C1 解耦成本）— 抢救：补 TDD → 提交 → 排队执行
- 现状: `scripts/run_c1_decouple.py` 未提交、无测试、零执行证据。
- 抢救路径见 §2 提示词 A。

### 🔴 W16（公开真实层）— 抢救：废弃错误映射，重建部分类协议
- 现状: 序数映射已证伪；`configs/p05-public-real-layer.yaml` 与 `reports/p05-public-real-layer-mapping.json` 均基于错误假设；5 个调试脚本污染仓库根目录。
- 抢救路径见 §2 提示词 B。

### ⚪ 全局清理
- 删除根目录调试脚本: `add_yprime.py`, `check_e6.py`, `debug_phaseb.py`, `test_ak*.py`(×3), `test_build*.py`(×2), `param_names.txt`
- 归位保留价值脚本至 `scripts/dev-tools/`（若确需保留）
- `.playwright-mcp/` 已入 .gitignore，无需处理

---

## 2. 抢救提示词（复制到对应窗口）

### 提示词 A｜W15 续命

```
接手 psd-framework W15 窗口续作。现状：scripts/run_c1_decouple.py 已有草稿但未提交、无 TDD、未执行。按以下顺序完成闭环：
① 先 git checkout HEAD -- 自查并清理工作树（并行窗口曾有干扰事故）；
② 为 run_c1_decouple.py 补 TDD（至少覆盖：Y→Y' 映射正确性含 stand/track→locomotion 合并、backbone 冻结断言、成本度量字段完整性），测试落 psd/training/tests/；
③ 脚本与配置一并提交（Conventional Commits 中文）；
④ 执行前检查 GPU：nvidia-smi 显存 <2GB 才跑 full；否则先用 --n-per-class 20 的 CPU/小规模档跑通管线出冒烟证据；
⑤ 结果 JSON 归档 reports/c1-decouple-cost-<日期>.json + markdown 报告（含标注单元数、墙钟时间双维度 vs 非解耦基线；结论不利则如实写并附标题降级预案）；
⑥ 回写 dev-docs/HANDOVER.md §8 W15 行状态。禁触 docs/paper/**、dev-docs/decisions/**、*ntu* 文件。
```

### 提示词 B｜W16 重生

```
接手 psd-framework W16 窗口重生。上一轮的序数映射已被 df_action.xlsx 证伪（例：AK index 1 是 Attacking 不是 sit），全部作废。按以下顺序重做：
① 读 D:\Desktop\k9-training-system\data\animal_kingdom\action_recognition\annotation\df_action.xlsx（140 动作真值表）与 dev-docs/rescue-plan-2026-08-24.md §0 的权威语义映射表；
② 清理根目录调试垃圾：删除 test_ak.py test_ak2.py test_ak3.py test_build.py test_build2.py param_names.txt add_yprime.py check_e6.py debug_phaseb.py；
③ 重写映射模块：仅采用 rescue-plan §0 表中强度≥中的 12±2 个类，写成显式 dict 常量（AK_index→PSD_class），每个映射附一行语义理由；零覆盖的 10 个 K9 特有类显式登记为 excluded_classes 并说明原因；
④ 重写 configs/p05-public-real-layer.yaml 为部分类协议（num_classes=12±2，注明子集清单）；
⑤ TDD：映射正确性（Barking(3)→bark 等 spot check）+ excluded 断言 + 多标签取主标签规则；
⑥ 从 train.csv/val.csv 过滤犬科视频（329 个，物种判定依据 DATA_LOCATIONS.md 登记口径），统计各 PSD 类样本量分布写入报告——若某类 <10 视频须在报告中标记不可训练；
⑦ GPU 排队纪律不变：NTU 占卡期间只做数据管线+TDD+统计，训练等空闲。产出 reports/p05-public-real-partialclass-<日期>.md。
禁触 dev-docs/HANDOVER.md 主文档（协调者统一回写）、docs/paper/**。
```

### 提示词 C｜GPU 排程看护（可选，若你想让 full-budget 自动接力）

```
接手 psd-framework W18 窗口（GPU 排程看护）。当前 runs/p05_stgcn_bc_Yprime 与 NTU Phase B 长训共用过 GPU；NTU 进程 PID 35208 存活中。任务：每 30 分钟巡检一次 nvidia-smi，当显存占用回落 <500MiB 且无 python 训练进程时，按顺序自动接力两个排队任务：
① AL full-budget：.venv\Scripts\python.exe scripts\run_p05_al_efficiency.py --config configs\p05_al_full.yaml --fresh（先把 config 内 RUNDATE 改当日）
② C1 full 档：python scripts/run_c1_decouple.py --n-per-class 100 --checkpoint runs/p05_stgcn_bc_full/best.pt（前提：W15 已提交可运行版本）
每完成一项立即提交证据 JSON 并在 commit message 注明接力触发时间。任一任务失败按其报告的熔断条款处理并上报用户。禁止同时启动两项。
```

---

## 3. 验收标准（协调者复核用）

| 窗口 | 通过条件 |
|------|---------|
| W15 | 测试绿 + 脚本入库 + 冒烟或 full 结果 JSON + 报告含双向论证 |
| W16 | 新映射模块全部基于 df_action.xlsx 真值 + excluded_classes 显式登记 + 犬科样本量分布统计落盘 |
| W18 | 两项接力各有当次运行证据，串行无并发 |

## 4. 修订历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-08-24 | 协调者复核后建册：W16 映射证伪记录 + 权威语义表 + 三份抢救提示词 |
