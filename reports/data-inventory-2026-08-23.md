# 数据深盘点报告（W2 数量级复核）

> 执行窗口: W2（2026-08-23）
> 任务来源: `dev-docs/handovers/W2-data-inventory.md`
> 复核对象: K9 仓三大数据集**声明数量** vs 本地实测（K9 仓数据全程只读）
> 运行环境: Windows + PowerShell 7 + Python 3.12.4（numpy 2.5.1 / pandas 2.3.3 / openpyxl，系统 Python）

---

## 结论速览

| # | 数据集 | 声明值 | 实测值 | 判定 |
|---|--------|--------|--------|------|
| 1 | InterPet4D `smal_npy/` | 226 序列 | **226 个 .npz 文件** | ✅ 吻合 |
| 2 | Animal Kingdom 犬科 | 338 视频 / 239 帧级标注 | **329 视频**（train 231 / test 98）；帧级行 **34,772** | ❌ 不符 → 需 K9 口径确认 |
| 3 | APTv2 全量 | 242K 文件 | **83,304 文件** | ❌ 严重不符 → 需 K9 口径确认 |

附带产出：smal_npy 骨架维度实测（见 §2.3）——P0.1 加载器直接可用。

---

## 1. 复核① InterPet4D smal_npy

### 1.1 结构探明

```
路径: D:\Desktop\k9-training-system\data\interpet4d\smal_npy
顶层条目: 226（全部为文件，0 个子目录）
扩展名分布: {'.npz': 226}   ← 注意：实际为 .npz 压缩包格式，非 .npy 裸数组
```

### 1.2 抽查加载（首/中/尾各 1 个，`np.load(allow_pickle=True)` 全部成功）

| 文件 | 大小 | T（帧数） |
|------|------|-----------|
| `interpet_dog01_p01_take01_ego_001.npz` | 483 KB | 326 |
| `interpet_dog05_p12_take08_ego_001.npz` | 827 KB | 556 |
| `interpet_dog12_p11_take01_ego_003.npz` | 753 KB | 509 |

每个 .npz 内含 9 个同名键的数组（三份样本键集合完全一致）：

```
pose_rotmat: (T, 35, 3, 3) float32
betas:       (T, 30)       float32
betas_limbs: (T, 7)        float32
R_world:     (T, 3, 3)     float32
t_world:     (T, 3)        float32
s_world:     (T,)          float64
kp_world:    (T, 24, 3)    float32
kp_weight:   (T, 24)       float32
frame_idx:   (T,)          int32
```

### 1.3 骨架维度结论（P0.1 加载器输入口径）

- **SMAL 姿态表示**：`pose_rotmat` → **T×35×9**（35 关节 × 3×3 旋转矩阵展平）
- **世界系关键点表示**：`kp_world` → **T×24×3**（24 关键点）
- T 为逐序列可变长度（抽查 326 / 556 / 509），加载器需按序列裁剪滑窗
- ⚠️ 格式为 `.npz`（压缩多数组容器），P0.1 读取代码应使用 `numpy.load(...)` 后按键取数组，不能按单数组 `.npy` 直接 mmap

---

## 2. 复核② Animal Kingdom 犬科

### 2.1 结构探明（与交接文档预想不同）

```
D:\Desktop\k9-training-system\data\animal_kingdom\
├── action_recognition\
│   ├── annotation\  train.csv (217 MB) / val.csv (54 MB) / df_action.xlsx   ← 实为【空格分隔】文本（非逗号 CSV、非 JSON）
│   ├── dataset\     video\ + video.tar.gz (15.5 GB)
│   └── AR_metadata.xlsx  （sheets: AR / CARe / Action / Animal / video_url）
├── pose_estimation\  仅 dataset.tar.gz (2.6 GB)，标注未解压（本地不可复核）
└── video_grounding\  本地为空目录
```

表头（charcode 验证为单空格 0x20 分隔）：
`original_vido_id video_id frame_id path labels type`
- 每行 = 一帧标注；`labels` 字段内用逗号分隔多标签；AR_metadata.xlsx 的 AR 表共 **30,100 视频**

### 2.2 计数过程与证据

犬科精确物种集（10 种，已排除假阳性 *African Wild Boar*=猪、*Dog Faced Water Snake*=蛇、*Malayan Flying Fox*=蝙蝠）：

```
{African Painted Dog, Coyote, Desert Fox, Dholes, Dingo Dog,
 Dog, Fox, Jackal, Wild Dog, Wolf}
```

**口径A（精确物种交集，解析 `list_animal` 列）→ 329 视频**

```
split 分布: train 231 / test 98
分物种视频数: Wolf 142, Fox 79, Wild Dog 35, Dog 31, Coyote 18,
             Desert Fox 12, Jackal 8, Dingo Dog 2, African Painted Dog 1, Dholes 1
CSV 帧行数（空格分隔逐行过滤 original_vido_id ∈ 犬科视频集）:
  train.csv: 总帧行 3,500,840 | 犬科帧行 24,865 | 涉及视频 231
  val.csv:   总帧行   911,360 | 犬科帧行  9,907 | 涉及视频  98
  合计犬科帧行: 34,772
```

**其他口径变体（穷举验证声明值是否可复现）**

| 口径 | 视频数 |
|------|--------|
| A + Dog Faced Water Snake | 337 |
| A + Malayan Flying Fox | 331 |
| A + 蛇 + 蝙蝠（= 朴素子串正则口径B） | 339 |
| CARe 表（animal 列 ∈ 犬科10种） | 0 行 |

### 2.3 判定

- 声明「338 视频」无法以任何合理物种清单变体复现（最接近 337 / 339）；「239 帧级标注」与实测帧行 34,772 相差两个数量级，最接近的数字是 train split 视频数 231。
- 按 W2 铁律#3：如实记录差异、不猜测谁对，**该项标「需 K9 侧口径确认」**，不阻塞其他两项。
- 对 P0 的实质影响：无论按 329 还是 338，犬科子集数量级一致（数百视频 / 数万帧），不改变 P0.1 弱监督预训练可行性判断。

---

## 3. 复核③ APTv2 全量

### 3.1 结构与计数证据（单遍递归枚举，1.8 s）

```
路径: D:\Desktop\k9-training-system\data\APTv2\APTv2\{annotations, data}
├── annotations\   59 文件（含 fewshot/ leaveoneout_train70/ leaveoneout_train70onlyeasy/ tracking/ 子目录 + 7 个 COCO 大 JSON）
├── data\easy      49,957 文件
├── data\hard      33,288 文件
└── 总计: 83,304 文件
扩展名: .json 41,733 / .jpg 41,569 / .bak 1 / .txt 1
伴生目录（另计）: aptv2_annotations 59 / aptv2_canidae 409 / aptv2_yolo 581 / aptv2_yolo_pose 582
```

COCO 标注内容统计（7 个顶层 JSON 全量解析）：

```
images: 每个 split 均 41,179 条（同一图像全集）
annotations: train 58,029 / val 11,315 / test 15,267 → 去重合计 84,611
             （若把 easy/hard 子集重复计入则 111,193）
```

### 3.2 判定

- 实测文件数 **83,304 ≈ 声明 242K 的 34%**，差异巨大。
- 已排查的可能口径：本地文件总数（83,304）、伴生目录合并（≈84,935）、COCO images 条目（288,253 含 split 重复）、annotations 条目（84,611~111,193）——均 ≠ 242K。
- 可能解释（不下结论，交用户裁决）：① 242K 为官方全量规模而本地是部分下载；② K9 统计时点后数据曾被清理；③ 当年统计口径不同。
- 对 P0 的实质影响：83K 文件仍是无标签池的有效规模，但「242K」不应再作为论文数字引用，直至 K9 口径澄清。

---

## 4. 方法与命令留痕

- 结构探明 / 计数：PowerShell `Get-ChildItem` / `[IO.Directory]::EnumerateFiles` 单遍枚举
- npz 抽查：Python `numpy.load`（首/中/尾 3 样本，键+shape+dtype 全打印，见 §1.2）
- AK 过滤：pandas 读 xlsx + `ast.literal_eval` 解析 `list_animal`；CSV 用 `sep=r'\s+'` 读取（分隔符经 charcode 验证）
- APTv2 JSON：`json.load` 逐个统计 `images`/`annotations` 长度
- K9 truth 口径溯源：`data-unblocking.md` L64-66 与本仓 stage-plan.md L86 仅写结论未写口径（已确认）
- 分析脚本存档：`%TEMP%\opencode\w2_ak_count.py` / `w2_care_count.py` / `w2_ak_variants.py`

## 5. 遗留事项（交收尾会话 / 用户）

1. 「338/239」「242K」需 K9 侧口径确认后再决定是否回改 K9 truth 或修订声明
2. `project-brief.md` §8「数据集磁盘路径盘点」待确认项建议由收尾会话勾销（超出 W2 白名单，未动）
