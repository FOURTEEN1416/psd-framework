# -*- coding: utf-8 -*-
"""AK→PSD 部分类协议权威映射（W20，基于 df_action.xlsx 真值从零重建）.

背景
----
W16 的序数映射假设「AK 标签值 1-22 按索引对齐 PSD 类」已被 df_action.xlsx
证伪并清除（rescue-plan §0，commit 8fabca2）。本模块是 W20 从零重建的
唯一合法映射：仅收录可诚实映射的 12 个 PSD 类，构成公开真实层部分类协议
（披露范式沿用 P0.1 dog-ID 代理探针先例，reports/p01-aimclr-2026-08-23.md §2）。

真值依据
--------
K9 仓 ``data/animal_kingdom/action_recognition/annotation/df_action.xlsx``
（140 动作全表，列: S/N / action_category / action / index / segment / count）。
权威编号 = **index 列**。

⚠️ 与 rescue-plan §0 表的两处差异（已逐行核验 xlsx，报告将披露）:
  - Lying on its side: §0 记 (73)，xlsx index 列真实值 = **74**
    （73 是 S/N 列值；index 73 = Licking 舔毛，绝不可映射为 down）
  - Jumping:           §0 记 (15)，xlsx index 列真实值 = **67**
    （15 是 S/N 列值；index 15 = Chirping 鸣叫，绝不可映射为 jump）

单一真相
--------
PSD 22 类清单权威来源 = ``docs/assets-map.md`` §1；本模块通过
``psd.data.synth_stgcn.ALL_BEHAVIORS_22`` 引用校验，不另抄一份。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from psd.data.synth_stgcn import ALL_BEHAVIORS_22

# ---------------------------------------------------------------------------
# 权威映射表：AK action index -> PSD 类名
# 每项附一行语义理由；index 均经 df_action.xlsx pandas 读表逐项核验。
# ---------------------------------------------------------------------------
AK_INDEX_TO_PSD: Dict[int, str] = {
    # sit
    108: "sit",        # Sitting 坐姿直译（count=69）
    # down
    70: "down",        # Lying Down 趴卧直译（count=42）
    74: "down",        # Lying on its side 侧卧同属卧态（⚠️ 非 73=Licking；count=2）
    # stand
    116: "stand",      # Standing 站立直译（count=126）
    # stay
    68: "stay",        # Keeping still 保持静止 ≈ 停留指令的静态语义（count=6340）
    # bark
    3: "bark",         # Barking 吠叫直译（count=73）
    # bite
    8: "bite",         # Biting 咬直译（count=434）
    # watch
    2: "watch",        # Attending 注视警觉 ≈ 警戒观察（count=2368）
    102: "watch",      # Sensing 探察感知 ≈ 警戒的前置感知行为（count=6941）
    # apprehend
    1: "apprehend",    # Attacking 扑咬攻击 ≈ 扑咬制伏动作核心（W16 曾误当 sit；count=511）
    # retrieve
    13: "retrieve",    # Carrying In Mouth 口衔物品 ≈ 衔取的核心执行段（count=463）
    # scale
    16: "scale",       # Climbing 攀爬 ≈ 攀登障碍（obstacle 与其同源，见 excluded；count=233）
    # jump
    67: "jump",        # Jumping 跳跃直译（⚠️ 非 15=Chirping；count=695）
    # track
    45: "track",       # Exploring 嗅探探索 ≈ 追踪行为的弱对应（count=486）
    14: "track",       # Chasing 追逐移动 ≈ 追踪的弱对应（count=93）
}

# ---------------------------------------------------------------------------
# 可映射 PSD 子类（12 类）：顺序 = assets-map §1 的 22 类相对顺序（升序子序列），
# 保证与合成层标签口径可对齐；训练时按此顺序重编 0..11。
# ---------------------------------------------------------------------------
MAPPED_PSD_CLASSES: List[str] = [
    "sit",        # assets idx 0
    "down",       # assets idx 1
    "stand",      # assets idx 2
    "stay",       # assets idx 5
    "bark",       # assets idx 6
    "bite",       # assets idx 7
    "track",      # assets idx 8
    "apprehend",  # assets idx 11
    "watch",      # assets idx 15
    "retrieve",   # assets idx 18
    "jump",       # assets idx 19
    "scale",      # assets idx 20
]

# ---------------------------------------------------------------------------
# 零覆盖排除类（10 个 K9 特有科目动作，AK 野生动物行为体系中无诚实等价物）：
# heel/sit_up/alert_sit/alert_down/escort/recall/guard/release/search_blind
# 为 K9 训练科目特有；obstacle 与 scale 同源（Climbing），按 rescue-plan §0
# 裁决二选一保留 scale 并在此显式登记 obstacle 排除。任何把这些类硬塞给
# AK 数据的标签均属捏造——审稿人对照 AK 论文一查即穿。
# ---------------------------------------------------------------------------
EXCLUDED_CLASSES: List[str] = [
    "heel",         # 随行：K9 服从科目特有位置约束
    "sit_up",       # 坐立：K9 特有训练动作
    "alert_sit",    # 示警坐：护卫科目特有
    "alert_down",   # 示警卧：护卫科目特有
    "escort",       # 押解：护卫科目特有
    "recall",       # 返回：K9 指令特有（AK 无召回语义）
    "guard",        # 守卫：K9 护卫科目特有
    "release",      # 放口：K9 护卫科目特有
    "search_blind", # 搜索盲区：追踪科目特有
    "obstacle",     # 障碍穿越：与 scale(Climbing) 同源合并披露
]

# ---------------------------------------------------------------------------
# 映射强度三档（来自 rescue-plan §0 语义强度判定）:
#   strong = 直译对应; medium = 语义近似需脚注说明; weak = 弱对应须显著披露
# ---------------------------------------------------------------------------
MAPPING_STRENGTH: Dict[str, str] = {
    "sit": "strong",
    "down": "strong",
    "stand": "strong",
    "bark": "strong",
    "bite": "strong",
    "jump": "strong",
    "stay": "medium",
    "watch": "medium",
    "apprehend": "medium",
    "retrieve": "medium",
    "scale": "medium",
    "track": "weak",
}

# 部分类协议训练编号：PSD 类名 -> 0..11（与 MAPPED_PSD_CLASSES 顺序一致）
PSD_PARTIAL_CLASS_TO_IDX: Dict[str, int] = {
    name: i for i, name in enumerate(MAPPED_PSD_CLASSES)
}

__all__ = [
    "AK_INDEX_TO_PSD",
    "MAPPED_PSD_CLASSES",
    "EXCLUDED_CLASSES",
    "MAPPING_STRENGTH",
    "PSD_PARTIAL_CLASS_TO_IDX",
    "map_ak_index",
    "resolve_primary_label",
]


def map_ak_index(ak_index: int) -> Optional[str]:
    """查 AK 动作 index 的 PSD 映射；未收录返回 None（绝不猜测）。"""
    return AK_INDEX_TO_PSD.get(int(ak_index))


def resolve_primary_label(labels_field: str) -> Optional[int]:
    """多标签视频取主标签规则 = train.csv 标签列表第一项.

    Args:
        labels_field: train.csv ``labels`` 列原始字符串，如 ``"2,40"``。

    Returns:
        第一项的 AK 动作 index；空串/解析失败返回 None。
    """
    if labels_field is None:
        return None
    text = str(labels_field).strip()
    if not text:
        return None
    first = text.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def _validate() -> None:
    """导入期自检：值域 ⊆ 权威 22 类、12+10=22 无交无漏、强度全覆盖。"""
    mapped = set(MAPPED_PSD_CLASSES)
    excluded = set(EXCLUDED_CLASSES)
    assert set(AK_INDEX_TO_PSD.values()) <= set(ALL_BEHAVIORS_22), \
        "映射值域泄漏出权威 22 类清单"
    assert len(mapped) == 12 and len(excluded) == 10, "部分类协议必须为 12+10"
    assert mapped | excluded == set(ALL_BEHAVIORS_22), "12+10 未覆盖 22 全集"
    assert not (mapped & excluded), "mapped/excluded 存在交集"
    assert set(MAPPING_STRENGTH.keys()) == mapped, "强度表与可映射类不一致"
    assert list(PSD_PARTIAL_CLASS_TO_IDX.values()) == list(range(12)), \
        "部分类编号必须连续 0..11"


_validate()
