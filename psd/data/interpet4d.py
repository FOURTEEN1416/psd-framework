"""InterPet4D smal_npy 加载器 — P0.1 数据 owner。

职责：把 InterPet4D 的 SMAL 狗骨架 (.npz, kp_world=(T,24,3)) 转成
AimCLR 官方 feeder 可消费的 NTU 兼容视图 (C=3, T, V=25, M=1)。
适配逻辑全部在本模块，禁止修改 external/AimCLR 内部实现。

实测口径（2026-08-23，本机 K9 盘点路径抽查）：
- smal_npy/*.npz 共 226 个（227 clips 中 1 个缺 fit）
- kp_world: (T, 24, 3) float32 世界坐标；kp_weight: (T, 24) 置信度
- 无行为标签 → kNN probe 使用文件名内嵌 dog ID（13 类）作代理标签，
  随机基线 = 100/13 ≈ 7.69%
"""
import re
from pathlib import Path

import numpy as np

_CLIP_RE = re.compile(r"interpet_(dog\d+)_p\d+_take\d+_ego_\d+")

# NTU 兼容视图常量
NTU_NUM_JOINTS = 25  # AimCLR ntu-rgb+d graph 固定 V=25
SMAL_NUM_JOINTS = 24  # InterPet4D kp_world 关节数


def parse_clip_id(filename: str) -> str:
    """从 clip 文件名提取 dog ID（如 'dog01'）。

    命名规范（官方 README）：interpet_dog{DD}_p{PP}_take{TT}_ego_{NNN}
    """
    m = _CLIP_RE.search(str(filename))
    if not m:
        raise ValueError(f"无法从文件名解析 dog ID: {filename!r}")
    return m.group(1)


def load_clip(path: str | Path) -> dict:
    """加载单个 smal_npy .npz clip。

    返回 dict：kp_world (T,24,3) float32 / kp_weight (T,24) / frame_idx (T,) int32
    """
    path = Path(path)
    with np.load(path) as npz:
        return {
            "kp_world": np.ascontiguousarray(npz["kp_world"], dtype=np.float32),
            "kp_weight": np.ascontiguousarray(npz["kp_weight"], dtype=np.float32),
            "frame_idx": np.asarray(npz["frame_idx"]),
        }


def resample_to_fixed_t(arr: np.ndarray, target_t: int = 64) -> np.ndarray:
    """线性插值均匀重采样到固定帧数（batch 对齐必需），保留时间端点。

    支持 (T, ...) 任意后维形状（骨架坐标、置信度权重等共用同一插值索引）。
    """
    t_orig = arr.shape[0]
    if t_orig == target_t:
        return arr.astype(np.float32, copy=True)
    if t_orig == 1:
        return np.repeat(arr, target_t, axis=0).astype(np.float32)
    pos = np.linspace(0.0, t_orig - 1.0, target_t)
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    frac_shape = (target_t,) + (1,) * (arr.ndim - 1)
    frac = (pos - lo).astype(np.float32).reshape(frac_shape)
    out = (1.0 - frac) * arr[lo] + frac * arr[hi]
    return out.astype(np.float32)


def _normalize_sequence(kp: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """序列级归一化：有效关节质心去中心 + 中位数半径尺度归一。

    不猜测 SMAL 关节语义（官方未提供名称表），用质心代替单一中心关节。
    """
    valid = weight >= 0  # 归一化统计不区分置信度；置零在后处理统一做
    coords = kp[valid]  # (N_valid, 3)
    centroid = coords.mean(axis=0, keepdims=True)  # (1,3)
    centered = kp - centroid
    radii = np.linalg.norm(centered[valid], axis=-1)  # (N_valid,)
    scale = float(np.median(radii))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return (centered / scale).astype(np.float32)


def to_ntu_view(
    kp: np.ndarray,
    weight: np.ndarray,
    conf_threshold: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """(T,24,3) + 置信度 → AimCLR NTU 兼容视图 (3, T, 25, 1) float32。

    - 恒等映射 SMAL 0-23 → NTU 槽位 0-23；槽位 24 恒零（死关节，graph 孤立节点）
    - kp_weight < conf_threshold 的关节数值置零（NTU 惯例）
    - normalize=True 时做序列级质心去中心 + 尺度归一（先归一后置零）
    """
    kp = kp.astype(np.float32, copy=True)
    weight = np.asarray(weight, dtype=np.float32)
    assert kp.ndim == 3 and kp.shape[1] == SMAL_NUM_JOINTS and kp.shape[2] == 3
    assert weight.shape == kp.shape[:2]

    if normalize:
        kp = _normalize_sequence(kp, weight)

    low_conf = weight < conf_threshold
    kp[low_conf] = 0.0

    t = kp.shape[0]
    view = np.zeros((3, t, NTU_NUM_JOINTS, 1), dtype=np.float32)
    view[:, :, :SMAL_NUM_JOINTS, 0] = np.transpose(kp, (2, 0, 1))
    return view


def is_valid_clip(kp: np.ndarray) -> bool:
    """clip 骨架有效性：所有数值必须有限（拒绝 SMAL 拟合失败的全/半 NaN clip）。"""
    return bool(np.isfinite(kp).all())


def build_label_index(sample_names: list[str]) -> tuple[list[str], list[int], int]:
    """由 clip 文件名列表构建 kNN probe 代理标签（dog ID 分类）。

    返回 (sample_names, labels, num_class)。labels 与输入顺序一一对应。
    """
    dog_ids = [parse_clip_id(n) for n in sample_names]
    classes = sorted(set(dog_ids))
    cls_to_label = {c: i for i, c in enumerate(classes)}
    labels = [cls_to_label[c] for c in dog_ids]
    return list(sample_names), labels, len(classes)
