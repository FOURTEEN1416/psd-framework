# C3/W27 动捕犬数据获取与转换 — 运行证据报告

> 日期: 2026-08-25 | 窗口: W27 (wt/W27) | 任务书: dev-docs/handovers/DATA-CAMPAIGN-plan.md §2-C3
> 结论 owner: dev-docs/research/MOCAP_DATASETS.md（本文件只存当次运行证据）

## 1. 当次运行证据

### 1.1 自测（合成 BVH 已知答案）
```
& D:\Desktop\psd-framework\.venv\Scripts\python.exe scripts\bvh_dogset_to_sequence.py --self-test
[self-test] PASS — 解析/FK/映射/质检全链路 OK, kp(2, 21, 3),
qc={'v_max_cm_s': 0.0, 'bone_len_cv': 0.0, 'bbox_diag_cm': 56.8}
```
地标位断言：hips(0,100,0) / head(40,100,0) / fl_paw(35,100,±5) / tail_b(-10,110,0) / hl_paw(0,85,±5) 全部通过（零旋转型位=OFFSET 沿链累加，验证父链遍历与按名映射）。

### 1.2 全量转换（51 条真实犬类 BVH）
命令：
```
python scripts/bvh_dogset_to_sequence.py --src external/dogset-mann-siggraph2018/raw
  --out-dir runs/data_campaign/mocap/sequences --manifest runs/data_campaign/mocap/manifest.jsonl
```
输出尾部：`=== 完成 51 条 | 总帧数 147541 (2459.0s@60fps) ===`；pkl 文件计数 51 ✓

### 1.3 产物抽验（D1_001_KAN01_001.pkl）
```
格式B必填字段: keypoints=ndarray(1531,21,3)float32 / topology_name=str / V=int /
fps_or_sampling=float(60.0) / source=str / split=str
NaN: 0 ; joint_order_canonical[:6]=['hips','spine_a','spine_b','neck','head','tail_a']
manifest 行数: 51 ；总T=147541 帧
```

## 2. 质检统计（当次实测）

| 项 | 值 |
|----|-----|
| 序列数 | 51（拓扑单一、60fps 统一） |
| 总帧数 | 147,541（2459.0s ≈ 41 分钟） |
| 单条时长 | min 2.58s / max 223.32s |
| 骨长 CV | 全库 = 0.0000（FK 刚体性通过） |
| qc_flag=suspect_glitch | **45/51**（单关节帧间峰值 >1500cm/s） |
| 热帧形态 | 孤立段(≤2帧) 1420 vs 持续段(≥4帧) 32 → 判定为源动捕标记毛刺而非真实快运动 |
| 典型案例 | D1_013 head t=541→542 单帧位移 ~93cm 后复位（标记交换特征） |

## 3. FK 正确性的判定依据（速度尖峰为何不判 FK 错误）

1. 骨长帧间 CV 恒 0 —— 层级变换数值稳定；
2. 尖峰为**单关节孤立单帧**跳变（如 D1_008 hr_paw y+23cm/z−32cm），同帧其余关节速度小一个量级——欧拉序错误会表现为全关节持续性系统偏差；
3. 根平移速度谱平滑（中位 85 / p99 122 / max 128 cm/s）——根通道无跳变；
4. 合成自测的地标位精确命中解析解。

## 4. 边界声明

- 原始 zip 与转换 pkl 均为本地派生数据（gitignore/runs），不入 git；许可为 Edinburgh 研究/教育专用，禁再分发。
- 动捕犬定位 = 真实运动学先验（MOCAP_DATASETS.md §0），禁止作行为分类主粮引用。
