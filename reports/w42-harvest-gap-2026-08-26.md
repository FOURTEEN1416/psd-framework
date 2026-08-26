# W42 — C1 缺口词族补采报告

> 窗口: wt/W42 | 日期: 2026-08-26 | 全程 CPU/网络，零占卡
> 触发: K9-ACQUISITION-PACK.md §3.2a 实证缺口（警犬 训练 31 / dog obedience trial 2 / 搜救犬族空白）
> 配置: configs/harvest_video_w42.yaml（新增）| 管线复用 scripts/harvest_video_pipeline.py 零改动

## 1. 结果总览

| 指标 | 值 |
|------|-----|
| 定稿片段 | **208 条**（runs/data_campaign/video_gap_w42/manifest.jsonl） |
| fragment_id 前缀 | `w42v-`（管线硬编码 w25v-，manifest 定稿步统一改名，见 §4） |
| 唯一新源视频 | 27 个（youtube 15 / bilibili 12 按片段计 youtube 122 / bilibili 86） |
| 契约校验 | 7 字段缺失 0 行；>30s 违规 0 行 |
| 主池重复剔除 | 47 条（origin_url 与主检出池 88 源比对） |

## 2. 阶段明细

| 阶段 | 数值 | 备注 |
|------|------|------|
| search | 177 候选（6 词族 × 2 平台） | B 站 412 风控 2 次，cookie 自动刷新均恢复 |
| download | 42 成功 / 2 失败 / 4 跳过 | 失败=B站付费课程（优雅跳过）；跳过=时长越界 |
| split | 463 原始片段 | scene 0.30 / ≤30s / ≥1.5s |
| filter | 255 过筛（55.1%） | YOLO11n CPU，4 帧出现率 ≥0.5 |
| manifest+去重 | **208 定稿** | 对主池 88 源 origin_url 去重 |

## 3. 缺口闭合情况

| 词族 | 补采前（主池） | 本轮新增 | 判定 |
|------|--------------|---------|------|
| 搜救犬 训练 | 0 | 44 | ✅ 空白族建立 |
| search and rescue dog training | 0 | 46 | ✅ 同上(en) |
| k9 agility training | 0 | 76 | ✅ 新族最大增量 |
| 警犬 训练 科目 | 31(近似族) | 27 | ✅ 加深 |
| 工作犬 服从 训练 | — | 13 | 一般 |
| 犬 越障 训练 | 0 | 2 | ⚠️ 弱命中，B站该词候选质量差，后续换词（如「军犬 训练」「犬 障碍 训练」） |

## 4. 与主池合并须知（交协调者/用户裁决）

1. **前缀映射**：管线 stable_id 硬编码 `w25v-{hash}`；本窗在 manifest 定稿步将前缀改为 `w42v-{同 hash}`（哈希本体不变，可追溯）。合并入主池时直接追加行即可，无 id 冲突风险；
2. **去重已前置**：47 条与主池同源片段已剔除，合并不会造成同一源视频双重计数（主池侧无需改动）;
3. 合并动作本身属统一池刷新范畴（W30 build_unified_pool 五源→六源），建议随下一轮池刷新由 owner 执行，本窗不越界。

## 5. 运行环境备注

- YouTube 代理 127.0.0.1:17890 当日 16:37 起恢复（晨间曾不可达致频道名补拉失败，PACK §3.2c 重跑协议仍有效但已部分被本轮覆盖）；
- 全程未触碰 GPU；B 站风控由既有 guest-cookie 机制自动消化。

## 6. 复现

```powershell
$py = "D:\Desktop\psd-framework\.venv\Scripts\python.exe"
& $py scripts/harvest_video_pipeline.py --config configs/harvest_video_w42.yaml --stage selftest
& $py scripts/harvest_video_pipeline.py --config configs/harvest_video_w42.yaml --stage all    # 或分阶段
# manifest 定稿步(去重+改名)为本窗一次性后处理, 逻辑见本报告 §1/§4 与会话记录
```
