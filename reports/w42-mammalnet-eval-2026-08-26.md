# W42 — MammalNet dog 子集评估报告

> 窗口: wt/W42 | 日期: 2026-08-26 | 纯 CPU/网络任务
> 触发: K9-ACQUISITION-PACK.md §5 行动 5（MammalNet dog 子集评估下载）
> 结论速览: **标注包已解析、犬子集 1,435 条实锤；巨型视频包双 403 且磁盘不可容纳；发现"文件名即 YouTube ID"的定向抓取替代路径**——按现有纪律该源降级为"预训练池广度候选"，优先级排在基地拍摄之后。

## 1. 下载实测

| 包 | 实测 | 判定 |
|----|------|------|
| annotation.tar | ✅ 200 OK，1,512,456 B（1.44 MB），Range GET 206 正常 | 已下载解包 |
| trimmed_videos.tar.gz | ❌ HEAD 与 Range GET 均 **403 Forbidden**（官方 README 直链当前失效） | 无法获取 |
| full_video.tar.gz | ⚠️ 可访问但 **365 GB**；D 盘仅剩 49.4 GB | 出局 |

仓库 Issues 无 403 相关报告（检索零命中）；recognition/README.md 数据准备节为 "ongoing"，无备用源。

## 2. 标注结构（当次解析实证）

- `composition/{train,val,test}.csv` 行格式：`trimmed_videos/<YouTubeID>.mp4 <col1> <col2>`
- **列语义（值域全表验证）**：col1 = 行为 id ∈ [0,11]（12 类）；col2 = 动物类目 id ∈ [0,172]
- `genus_to_id.txt`：173 属；**canis = 8**
- `behavior_to_id.txt`：12 类生物习性谱

### 解析事故披露（防后人踩坑）

首次过滤把"含 canis 的行"误判为 4,438 条——根因是 **fights_against_other_animals(行为 id=8) 与 canis(属 id=8) 数字撞车**，两个命名空间混用所致。以"col1 全表 ≤11 且 col2 全表最大 172"完成列语义定案后重算，犬子集实数如下。教训：跨命名空间 id 必须先验证值域再过滤。

## 3. 犬子集规模与行为分布

| split | 条数 |
|-------|------|
| train | 1,050 |
| val | 115 |
| test | 270 |
| **合计（每视频一 clip）** | **1,435** |

| 行为 | 条数 | | 行为 | 条数 |
|------|------|-|------|------|
| sleeps | 398 | | mates_with_other_animals | 97 |
| fights_against_other_animals | 208 | | gives_birth_to_a_baby | 56 |
| hunts_other_animals | 197 | | pees | 45 |
| nurses_or_breastfeeds_its_baby | 147 | | poops | 30 |
| eats_food | 135 | | grooms/cleans | 9 |
| drinks_water | 105 | | vomits | 8 |

完整清单落盘：`runs/data_campaign/mammalnet/dog_clips_1435.tsv`（yt_id/split/behavior_id/behavior_name）

## 4. 对本管线的适配判定

1. **行为体系不匹配（主判读）**：12 类为生物习性谱，与本仓七类物理先验（lying/sitting/standing/walking/running/rise_transition/jump）近乎零重叠；且为视频级单标签，非骨架序列。外部标签直用禁令下这些标签本身不可入监督管线。
2. **视频本体仍有价值**：1,435 段真实犬类视频可作为无标签预训练池广度扩展原料——标签策略沿 C1 判例走规则引擎自产。
3. **获取路径转向（关键发现）**：文件名即 YouTube ID，无需依赖失效的 S3 巨型包；代理恢复时可用 yt-dlp 定向抓取（工具链与 C1 相同）。已验证本日代理恢复可用。

## 5. 建议处置

- 该源定位降级：**预训练池广度候选**（域多样性），优先级排在①基地拍摄（直接命中七类）②C1 缺口词族补采（已完成）之后；
- 抓取时机：随下一次需要 YouTube 通道的窗口顺路执行，不单独立项；
- 许可注记：YouTube 来源研究用途、引用制（Chen et al., CVPR 2023）、不再分发，沿用 C1 license_note 模板并追加论文引用要求；
- DATA_LOCATIONS 登记待其冲突标记修复后由 owner 一并处理（W42 DEFECT 已看板登记）。

## 6. 复现命令

```powershell
curl.exe -sS -o runs\data_campaign\mammalnet\annotation.tar https://mammalnet.s3.amazonaws.com/annotation.tar
tar -xf runs\data_campaign\mammalnet\annotation.tar -C runs\data_campaign\mammalnet
# 列语义验证 + 犬子集统计脚本见会话记录；核心断言: max(col1)=11, max(col2)=172
```
