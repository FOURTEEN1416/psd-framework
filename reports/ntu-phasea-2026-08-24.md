# W9 NTU 复现 Phase A 报告 — 协议核实完成 · 数据获取待渠道裁决

> **指标口径：公开基准层**（NTU60；与合成 / 公开真实 / 真实 K9 各层严格分账，禁止混表）
> 日期: 2026-08-24 | 执行窗口: W9 | 状态: **协议核实 ✅ / 数据获取 🚧 受阻上报（选项已备好）**

## 1. 一页结论

| Phase A 任务书条目 | 结果 |
|---|---|
| ① 读官方指引（只读） | ✅ 数据格式/预处理链/依赖缺口全部摸清（§3） |
| ② 数据获取 | 🚧 两条官方渠道本机均无法全自动直下——GDrive 不可达、百度云需账号交互；**升级上报用户裁决（§4）** |
| ③ 磁盘预算 | ✅ D 盘余量 76GB > 20GB 达标；预处理版实际需求 ≈2GB（原任务书"约 6GB"系原始骨架包口径，见 §3.3 修正） |
| ④ 预处理 | ⏳ 随数据到位后执行（官方包已是 frame50 成品，预计仅需校验+登记，无需再跑预处理） |
| ⑤ 协议核实 | ✅ 完成并发现参照数字口径修正：**79.18% 出自官方 released-model 复测，论文正文为 78.9%**（§5） |

## 2. 论文出处实锤（交接文档勘误）

- 交接文档给的 `arXiv:2104.10213` **错误**（实为一篇 NLP 综述）。AimCLR 原文无 arXiv 预印本。
- 正确出处：Guo, T., Liu, H., Chen, Z., Liu, M., Wang, T., & Ding, R. (2022). *Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-Supervised Action Recognition*. **Proceedings of AAAI-22, 36(1), 762–770. DOI: 10.1609/aaai.v36i1.19957**
- 官方 PDF：ojs.aaai.org/index.php/AAAI/article/view/19957/19716（本次已全文解析核对）
- 定位路径（合规）：GitHub 工具链 → firework8/Awesome-Skeleton-based-Action-Recognition → AAAI OJS 页

## 3. 官方数据管线调研结论（external/AimCLR 只读）

### 3.1 AimCLR 期望的输入格式（config/ntu60/*.yaml + feeder/ntu_feeder.py）

```
{data_root}/ntu60_frame50/{xsub,xview}/{train,val}_position.npy   # (N,3,50,25,2) float32 mmap
{data_root}/ntu60_frame50/{xsub,xview}/{train,val}_label.pkl      # pickle((sample_names), labels∈[0,60))
```

- motion/bone 流**不需要单独文件**：由官方 processor 从 position 现算（作者在 issue #2 明确答复）；feeder 的三视角来自对同一 position 的三种增强
- 预处理链 = 原始 `.skeleton` → `tools/ntu_gendata.py`（300 帧截断 + xsub/xview 划分）→ `feeder/preprocess_ntu.py`（降采样至 50 帧）

### 3.2 发现的依赖缺口（若走原始数据自处理路线）

`feeder/preprocess_ntu.py` 依赖 `NTUDatasets.NTUMotionProcessor` 模块——**该模块不在 external/AimCLR 克隆内**（源自 CrosSCLR 血统）。即：即使拿到原始骨架数据，官方第二段预处理脚本也无法直接跑。→ 进一步支持采用**官方预处理成品直下**路线。

### 3.3 体量修正（对任务书"约 6GB"口径的更正）

- position 单流全量 ≈ 56.6k 样本 × 29.3KiB ≈ **1.7GiB**，加 label 可忽略 → 预处理版 **≈2GB 量级**
- "约 6GB"对应的是 NTU RGB+D 原始 `.skeletons` 压缩包口径
- 磁盘结论不变：D 盘 76GB 充裕

## 4. 数据获取现状（🚧 唯一阻塞项，已到升级点）

### 4.1 官方渠道清单与实测（2026-08-24 本机）

| 渠道 | 来源 | 实测 | 结论 |
|---|---|---|---|
| Google Drive `action_dataset`（含 ntu60_frame50 成品） | 官方 README | drive.google.com / drive.usercontent.google.com / docs.google.com 全部 12s 超时；注册表历史代理端口 17890 已死；常见代理端口 7890/7897/10809/1080 均无监听 | ❌ 本机不可达，除非用户开代理 |
| 百度网盘镜像（提取码 0211） | 作者 2022-02-11 于 issue #2 专为国内传输添加 | pan.baidu.com HTTP 200 可达 | ⚠️ 可达但需账号登录+客户端转存下载，无法无人值守 |
| NTU RGB+D 原始数据官方申请 | 官方 README 指向 shahroudy/NTURGB-D | 未启动 | 需表单申请（机构邮箱），周期不可控 |
| 降级方案（官方 checkpoint 直接 kNN 对照） | README released_model | 同在 Google Drive | 同样被网络阻塞 |

### 4.2 已备好的执行件（裁决后一条命令即可续跑）

- `scripts/fetch_ntu_data.py`：`--channel gdrive [--proxy]` 自动下载 / `--channel baidu` 打印转存指引 / `--verify` 结构契约校验（形状/dtype/label 值域/样本计数）
- 落盘默认目录 `data/ntu60_frame50`（已被 .gitignore `/data/*` 覆盖，数据本体不入库）

### 4.3 请用户裁决（三选一）

| 选项 | 动作 | 预计周期 | 备注 |
|---|---|---|---|
| **A（推荐）** | 用户开启任意可用代理后告知端口，W9 以 `--channel gdrive --proxy http://127.0.0.1:<port>` 全自动完成 | 分钟级 | 最快且全程可审计 |
| B | 用户用百度网盘 App 转存并下载 `ntu60_frame50` 子树至 `data/ntu60_frame50`，W9 校验接管 | 小时级（视带宽） | 免代理；链接+提取码见脚本指引输出 |
| C | 机构邮箱向 NTU 官方申请原始数据授权 | 天~周级 | 仅当 A/B 均不可行；Phase A 其余成果不受影响 |

## 5. 协议核实结论（✅ 回填依据）

### 5.1 原文协议定义（PDF 全文解析摘录）

- **Linear Evaluation Protocol**（79.18%/78.9% 所属协议）："we train a linear classifier (a fully-connected layer followed by a softmax layer) supervised with encoder fixed"
- **KNN Evaluation Protocol** 另有其名：KNN 分类器直接作用于训练好的 encoder 特征（P0.1 在 InterPet4D 上用的就是这一族，但数字不与 linear eval 混用）
- **划分定义**："In xsub, half of the subjects are used as training sets, and the rest are used as test sets. In xview, the samples of camera 2 and [camera 3 are for training and those of camera 1 for test]"（NTU60 共 56,578 序列/60 类，原文口径）
- **3s 含义**：joint + motion + bone 三独立模型推理融合（score ensemble），非单模型多头

### 5.2 数字口径修正（重要）

| 口径来源 | NTU60 xsub | xview | 说明 |
|---|---|---|---|
| 论文正文 Table 4（linear eval, 300ep） | **78.9%** | 83.8% | 原文成绩 |
| 论文 Table 3 epoch 曲线 | 100ep 76.5 / 150ep 77.4 / 200ep 78.3 / 300ep 78.9 | — | 训练时长敏感性 |
| 官方仓 README released models 表 | **79.18%** | 84.02% | 作者复测 released checkpoint："performance is better than that reported in the paper" |

→ **此前所有文档引用的"原文参照成绩 79.18%"实为官方 checkpoint 复测口径**，论文正文是 78.9%。两者相差 0.28pp，不影响防线性质，但论文写作与 Phase B 判据必须写明口径。

### 5.3 对预注册验收判据的影响（Phase B 用，防挪门柱）

- 任务书预注册线：≥ 79.18 − 2pp = **77.18%**（以更高的 README 口径为参照，天然更严格）→ **建议维持不变**
- 若改以论文正文 78.9% 为参照则 ≥76.9%；两线均低于时按任务书出差异分析报告
- Phase B 对照实验建议双报：本仓复现值 vs 78.9%（同协议自训口径）与 vs 79.18%（官方权重口径）

## 6. 交付物清单（本窗口白名单内）

```
scripts/fetch_ntu_data.py            # 渠道化获取器 + 结构校验（gdown 懒加载，不进 requirements.txt）
reports/ntu-phasea-2026-08-24.md     # 本报告
docs/paper/related-work.md           # 定点回填：AimCLR 协议标记解除（授权依据：任务书 §2⑤ 与完成标准）
docs/paper/experiment-skeleton.md    # 定点回填：§4.4 行状态与口径修正（授权依据：任务书 §5）
```

## 7. 未验证项 / 移交 Phase B

- [ ] 数据到位后：`--verify` 结构校验 + 样本计数入档 + `DATA_LOCATIONS.md` 登记申请（该文件不在 W9 白名单，届时按边界提交或请协调窗口代登）
- [ ] Phase B 训练前：占卡区间声明（HANDOVER §8）、P0.2 是否释放确认
- [ ] related-work 中 AimCLR++ 77.2%、SMQ 75.3% 两数字仍待原文复核（非 W9 范围，已在该文件 checklist 保留）

## 8. 一条命令复现序列

```powershell
# 渠道裁决后二选一：
.\.venv\Scripts\python.exe scripts\fetch_ntu_data.py --channel gdrive --proxy http://127.0.0.1:<PORT>
.\.venv\Scripts\python.exe scripts\fetch_ntu_data.py --channel baidu   # 按打印指引手动转存

# 到位后校验（结构契约 + 计数）：
.\.venv\Scripts\python.exe scripts\fetch_ntu_data.py --verify --dest data\ntu60_frame50
```

## 9. 修订历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v1.0 | 2026-08-24 | Phase A：协议核实闭环 + 数据获取受阻上报（选项 A/B/C 已备） |
| v1.1 | 2026-08-24 | **数据获取完成 ✅（用户选渠道 A）**：用户启动桌面 GKD 加速器（内核=clash，端口 17890 实测谷歌全通）；修复 fetch 脚本 gdown v6 API 兼容；新增 `scripts/ntu_selective_fetch.py`（枚举+逐文件断点续传，抗不稳代理）；8/8 文件到位后 `--verify` 全绿——xsub 40,091+16,487=56,578、xview 37,646+18,932=56,578，形状 (N,3,50,25,2) float32，样本数与论文口径完全吻合。Phase B 待 GPU 排程 |
