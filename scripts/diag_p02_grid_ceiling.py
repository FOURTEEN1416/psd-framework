"""P0.2 冲刺预注册分析：预测网格分辨率对种子伪GT的理论IoU天花板。

假设：VQ 分割的段边界只能落在 patch_size 的整数倍网格上（run 长度必为 P 的倍数）。
即使码字分配完美，单段预测对 GT 段的 IoU 也有结构性上限。
本脚本暴力枚举所有网格对齐候选段，计算每个 GT 段的最优可达 IoU，
再按 episode 平均，量化 32→16 网格的收益上限。纯 CPU 只读分析。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, r"D:\Desktop\psd-framework")
from psd.data.smq_input import build_seed_gt_episode  # noqa: E402

FEATURES = Path(r"D:\Desktop\psd-framework\data\processed\p02\features_all")
SEEDS = Path(r"D:\Desktop\psd-framework\data\seeds\rule_seeds")

EPISODES = [
    ["interpet_dog08_p17_take01_ego_001", "interpet_dog06_p02_take01_ego_001",
     "interpet_dog03_p02_take01_ego_001", "interpet_dog09_p19_take01_ego_001",
     "interpet_dog10_p07_take01_ego_001"],
    ["interpet_dog07_p07_take01_ego_001", "interpet_dog12_p11_take01_ego_001",
     "interpet_dog04_p02_take01_ego_001", "interpet_dog05_p02_take02_ego_001",
     "interpet_dog01_p01_take01_ego_001"],
    ["interpet_dog02_p04_take01_ego_001", "interpet_dog11_p22_take01_ego_002",
     "interpet_dog08_p17_take01_ego_002", "interpet_dog06_p02_take01_ego_002",
     "interpet_dog03_p02_take01_ego_002"],
    ["interpet_dog09_p19_take03_ego_001", "interpet_dog10_p07_take01_ego_002",
     "interpet_dog07_p07_take01_ego_002", "interpet_dog12_p11_take01_ego_002",
     "interpet_dog04_p02_take01_ego_002"],
]


def iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union else 0.0


def grid_ceiling(gt: tuple[int, int], total: int, p: int) -> float:
    """所有 [k*P, (k+m)*P) 候选段对该 GT 的最大 IoU（m>=1）。"""
    best = 0.0
    starts = range(0, total // p + 1)
    for si in starts:
        s = si * p
        if s >= gt[1]:
            break
        for e in range(min(s + p, total), total + 1, p):
            v = iou((s, e), gt)
            if v > best:
                best = v
            if s >= gt[1] or e > gt[1] + p * 4:
                break
    return best


for name, clips in [("ep1", EPISODES[0]), ("ep2", EPISODES[1]),
                    ("ep3", EPISODES[2]), ("ep4", EPISODES[3])]:
    segs = build_seed_gt_episode(SEEDS, FEATURES, clips)
    t_total = max(s["end"] for s in segs)
    lens = [s["end"] - s["start"] for s in segs]
    ceil32 = np.mean([grid_ceiling((s["start"], s["end"]), t_total, 32) for s in segs])
    ceil16 = np.mean([grid_ceiling((s["start"], s["end"]), t_total, 16) for s in segs])
    short = sum(1 for L in lens if L < 32)
    print(f"{name}: {len(segs)}段 最短{min(lens)}帧 中位{int(np.median(lens))}帧 "
          f"| <32帧短段: {short}/{len(lens)} "
          f"| 天花板 P=32: {ceil32:.3f} → P=16: {ceil16:.3f} (+{(ceil16-ceil32)*100:.1f}pp)")
