# P0.2 E-A 唤醒评估任务书（定时任务专用）

> 触发方式：定时任务于 2026-08-24 03:35 唤醒，代理读本文件并严格执行。
> 身份：W7 救援持锁窗口的授权代表（锁：`reports/p02-window-lock.md`）。
> **最高禁令：本任务禁止启动任何新的训练。**

## 步骤

### 1. 等待训练完成（幂等护栏）

每 3 分钟轮询一次，最多 40 分钟。完成判定 = 同时满足：

- 不存在命令行含 `train_smq_segmentation.py` 的 python 进程
- `runs/p02_smq_eA/models/epoch-30.model` 存在

若进程已消失但 epoch-30 缺失 = 训练崩溃：将 `runs/p02_smq_eA/train_log.txt` 与 `console_err.log` 各末尾 50 行另存 `reports/p02-eA-crash-evidence.md` 并 git 提交，输出失败摘要后**停止**。

### 2. 幂等检查

若 `reports/p02-smq-iou-eA-concat.json` 已存在 → 跳过步骤 3-5 直接判读。

### 3. 保护旧可视化证据

若 `reports/p02-vis-episode1.png` 存在：重命名为 `p02-vis-v3baseline-episode1.png`（episode2 同理）。

### 4. 双口径评估（统一用 `.\.venv\Scripts\python.exe`）

```powershell
.\.venv\Scripts\python.exe scripts\eval_smq_segmentation.py --config configs\p02_smq_eA.yaml --iou --vis --ckpt runs\p02_smq_eA\models\epoch-30.model --gt-protocol concat --out reports\p02-smq-iou-eA-concat.json
.\.venv\Scripts\python.exe scripts\eval_smq_segmentation.py --config configs\p02_smq_eA.yaml --iou --vis --ckpt runs\p02_smq_eA\models\epoch-30.model --gt-protocol seeds --out reports\p02-smq-iou-eA-seeds.json
```

### 5. 码本复检

```powershell
.\.venv\Scripts\python.exe scripts\diag_p02_motion_words.py --config configs\p02_smq_eA.yaml --ckpt runs\p02_smq_eA\models\epoch-30.model --out reports\p02-diag-motionwords-eA.json
```

### 6. 判读（只读分析，不启动新训练）

对照基线（来源 `reports/p02-smq-iou-v3-kmeans.json` / `reports/p02-diag-motionwords.json`）：

| 指标 | v3 失败态 |
|------|----------|
| mean_matched_iou | 0.20 |
| 随机基线 IoU | 0.40–0.43 |
| boundary F1 | 0.0 |
| latent patch cos | 1.0（坍缩） |

- 若新结果 IoU 显著优于随机基线 且 latent cos 明显 <1、码本使用健康 → 写 `reports/p02-eA-findings.md` 简报（双口径对比表 + 全部数字）
- 否则如实记录「仍塌缩」及证据，**不启动修复实验**（留给用户裁决下一步）

指标口径一律标注「公开真实层」（InterPet4D smal_npy）。

### 7. 提交

仅 add `reports/p02-*` 白名单文件。Conventional Commits 中文描述：
`feat(wip): P0.2 E-A 双口径评估+码本复检——<一句话结论>`

### 8. 共享记忆

`memory_memory_add`（agent_id=opencode，tags 含 psd-framework,W7,E-A）记录一条结果要点。

### 9. 输出紧凑摘要

```
Status: success | failed
原因: <一行>
产出: <文件路径列表>
```

## 硬规则

AGENTS.md 全文生效：禁 WebSearch；三层指标禁止混报；无新鲜验证不得声明完成。
