# C5/W29 — dog-pose 静态池诚实可行性评估报告

> 窗口: W29（wt/W29 worktree，B-full 协议）｜日期: 2026-08-25
> 任务书: `dev-docs/handovers/DATA-CAMPAIGN-plan.md` §2-C5
> 数据: `D:\Desktop\datasets\dog-pose`（ultralytics 打包，kpt_shape=[24,3]，单类 dog）
> 证据: `runs/data_campaign/dogpose/inventory-evidence-2026-08-25.json`（当次运行全量校验）

## 0. 结论（三选一）

> **(b) 仅作预训练/增广池。**
> 一句话理由：数据本体完整干净（8476 图配对/标注/包围盒零缺陷），但**全库不存在任何序列分组元数据**，且 GT 有效关节仅 20/24（双眼/withers/throat 全库零标注，含 K9Graph 根关节）——做时序构造是假动态毒化（拒绝 a），弃之不用则是浪费零投影资产（排除 c），故按"静态子池"入统一真实扩展池。

已按格式 B 入库（详见 §4）：`runs/data_campaign/dogpose/sequences/*.pkl`，8476 条 T=1 静态骨架条目，`synthetic_dynamic=false` 固化于 manifest。

## 1. 样本量与结构盘点

| 项 | 实测（2026-08-25 当次运行） |
|----|------|
| 图片总量 | **8476** = train 6773 + val 1703 ✓（与 yaml/p05 声明吻合） |
| 图↔标注配对 | 双向零缺失（train 6773↔6773 / val 1703↔1703） |
| 标注解析 | 8476/8476 均 77 字段（cls + bbox4 + 24×[x,y,v]），class 恒为 0 |
| 可见性取值 | 仅 {2.0, 0.0} 两值；总槽位 203,424，可见 116,574（**57.3%**） |
| 单图可见点 | mean 13.75 / median 14 / min 4 / **max 20（无一图达 24）** |
| 分布 | ≥12 点占 73.6%；≥18 点占 12.9%；<6 点仅 0.02% |
| 退化样本 | 退化 bbox 0 个；可见点坐标越界 0 张 |
| 品种覆盖 | 112 个 Stanford Dogs synset（top: n02088094×116；bottom: 各×1） |
| 图片尺寸抽样(n=300) | 宽 143–1268px，高 103–928px，读取零失败 |

### ⚠️ 逐关节可见率的关键发现

| 关节（K9Graph 名） | 可见率 |
|----|----|
| left_eye / right_eye / **withers** / throat | **0.0%（全部 8476 图）** |
| rear_*_elbow / tail_start / tail_end / rear_*_knee | 39–49% |
| nose | 99.3%（唯一近全覆盖） |

即：**GT 有效关节 = 20/24**。p05 报告（§4.1）"24 点原生=K9Graph 零投影"的结论在**拓扑命名层**成立（yaml 逐名一致，本报告不推翻），但在**本打包 GT 标注层**须追加此限定——后续若消费其微调出的 YOLO 权重，该 4 关节将无可学习信号，输出恒为零/垃圾。此发现建议协调层转知 W20/p05 管线消费方（本窗不改其报告）。

## 2. 序列分组元数据探查（C5 核心问题）

| 探查轴 | 结果 |
|----|------|
| 全库递归非图片/非标注文件 | **仅 dog-pose.yaml 一个**（无 JSON/CSV/pkl 元数据） |
| 文件名模式 | `synset_照片ID.jpg`（Stanford Dogs 独立 Flickr 照片 ID） |
| 帧序/视频线索名（frame/f0001 等） | train/val 均 0 命中 |
| 同 ID 跨 split 泄漏 | train∩val ID 重叠 = 0 |

**判定：无序列分组元数据。** 每张图是独立照片，不存在"同视频相邻帧"的任何可依据字段。任务书预判（"大概率无"）被证实。

## 3. 三选一双问论证

| 方案 | 正方（能得到什么） | 反方（代价/风险） | 裁决 |
|----|----|----|----|
| (a) 时序升采样构造过渡序列 | 形式上凑出格式 B 的 (T,V,C)；量大管饱 | 不同犬只/视角/尺度间插值是**运动学假动态**：帧间速度谱与真实犬类运动分布完全不符（C4 保真度 KS 口径下必然爆炸）；缺根关节 withers 的骨架插值更无意义；混入训练会污染 warm-start 主线的动态先验；即便标注"合成动态"防住口径混报，也防不住分布毒化——违背"宁缺毋滥" | ❌ 拒绝 |
| (b) 仅作预训练/增广池 | ① GT 关键点现成，入库**零模型推理成本**（纯 CPU）；② 拓扑命名层与 K9Graph 零投影，下游无需映射表；③ 8476 张规模可观且质检零缺陷；④ 明确用途三件套：提点器微调源（p05 已在用）、规则引擎静态关节角先验标定、ST-GCN 预训练单帧增广 | 不提供时序动态，对行为识别主粮（T≥30 序列）只是间接贡献；4 关节系统性缺失限制可标定的角度族；Stanford 展示照偏站/坐姿，动作多样性低（后肢可见率 39–55% 佐证侧视遮挡普遍）；域上与 AK 野生/家庭场景有差 | ✅ **采纳** |
| (c) 不可用 | —— | 被 §1 证据直接反驳：配对/解析/包围盒零缺陷、规模大、许可可用 | ❌ 排除 |

## 4. 入库产物（收敛契约格式 B）

```
runs/data_campaign/dogpose/
├── inventory-evidence-2026-08-25.json     # §1/§2 全量盘点证据
├── manifest-2026-08-25.json               # schema/诚实声明/许可/sha256/复现命令
└── sequences/
    ├── dogpose_train.pkl                  # 6773 条目
    └── dogpose_val.pkl                    # 1703 条目
```

- 条目契约字段：`{keypoints:(1,24,3) float32 像素坐标+vis01, topology_name:"K9Graph", V:24, fps_or_sampling:null(静态), source, split}`；注记字段 `{sample_id, n_visible, coords_semantic:"image_pixel_xy", static:true}`
- 打包说明：每 split 一个合并式 .pkl（仓库既有合成集同为合并式惯例），目录满足契约 `sequences/*.pkl`
- 诚实固化：manifest 内 `has_sequence_grouping_metadata:false`、`synthetic_dynamic:false`、有效关节 20/24
- 复现：
  ```
  & D:\Desktop\psd-framework\.venv\Scripts\python.exe scripts/assess_dogpose_ingest.py --root "D:\Desktop\datasets\dog-pose" --out runs/data_campaign/dogpose
  & D:\Desktop\psd-framework\.venv\Scripts\python.exe scripts/assess_dogpose_verify.py --root "D:\Desktop\datasets\dog-pose" --pool runs/data_campaign/dogpose
  ```
- 当次复核证据：随机抽样对账 max|Δxy|=0.0000px、vis 逐位一致、sha256 与 manifest 一致、契约字段齐全（verify 脚本退出码 0）

## 5. 与 AK 域分布对比（报告要素）

| 维度 | Animal Kingdom 犬科（公开真实层既有） | dog-pose（本池） |
|----|----|----|
| 形态 | 329 视频 / 34,772 帧，**真时序** | 8476 张独立照片，**零时序** |
| 标签 | 动作标签（经 ak_mapping 映射 12 类） | 无行为标签，仅姿态 GT |
| 场景 | 野生/家庭实拍 | Stanford Flickr 展示照（多为站/坐摆拍） |
| 骨架来源 | Q3a 管线模型提点（预测） | 人工 GT（无需推理） |

结论：两池互补不互替——AK 出"真时序+行为弱标签"，dog-pose 出"高置信静态姿态先验"。dog-pose 不得计入任何时序样本口径。

## 6. W26 合流建议（静态姿态资产统一处置方案 · C5 侧草案）

1. 统一口径：APTv2 canidae 子集若经 W26 盘点同为静态为主，则与本池共用 `format_b.static_v1` schema 与"静态子池"标签，汇入 unified real-expansion pool 的静态分支；
2. 分工边界：dog-pose 管"犬类专用 24 点 GT 先验"，APTv2 管"物种覆盖广度"，互不重复建设；
3. 本草案待 W26 报告收编后在协调层合流出正式方案（W26 产出当前不可读：主检出无 `runs/data_campaign/`，worktree 列表无 wt/W26）。

## 7. 三层口径声明

本报告所有数字属**公开真实层·静态子池**。禁止与合成层（syn/syn_v2）混报；禁止计入真实 K9 层；禁止作为时序样本参与任何 ST-GCN 时序指标。

## 8. 遗留风险与移交

| 风险 | 处置 |
|----|----|
| 许可复合性：ultralytics 打包 AGPL-3.0 + 图像源 Stanford Dogs（学术研究惯例） | 已登记 `docs/DATA_LOCATIONS.md`；论文引用须同时致谢 Stanford Dogs 与 ultralytics dog-pose 打包 |
| p05 "零投影"表述需加 GT 层限定（4 关节零标注） | 本报告 §1 已披露；建议协调层转知 W20 消费方 |
| 下游误用时序 | manifest honesty 字段 + 本报告 §7 双重声明；加载方可 assert `static==True` |
