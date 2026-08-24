"""ST-GCN+BC 常量定义 — 24 关键点 + 22 类行为.

Owner: W11 窗口（PSD-Framework）
来源: K9 仓 `backend/ml/behavior/constants.py`（只读参考，本文件独立重实现）
"""
from __future__ import annotations

# ===== 24 关键点索引 =====
FRONT_LEFT_PAW = 0
FRONT_LEFT_KNEE = 1
FRONT_LEFT_ELBOW = 2
REAR_LEFT_PAW = 3
REAR_LEFT_KNEE = 4
REAR_LEFT_ELBOW = 5
FRONT_RIGHT_PAW = 6
FRONT_RIGHT_KNEE = 7
FRONT_RIGHT_ELBOW = 8
REAR_RIGHT_PAW = 9
REAR_RIGHT_KNEE = 10
REAR_RIGHT_ELBOW = 11
TAIL_START = 12
TAIL_END = 13
LEFT_EAR_BASE = 14
RIGHT_EAR_BASE = 15
NOSE = 16
CHIN = 17
LEFT_EAR_TIP = 18
RIGHT_EAR_TIP = 19
LEFT_EYE = 20
RIGHT_EYE = 21
WITHERS = 22  # 鬐甲（肩峰），骨架根节点
THROAT = 23

NUM_KEYPOINTS = 24

# 关键点分组
FRONT_PAWS = (FRONT_LEFT_PAW, FRONT_RIGHT_PAW)
FRONT_KNEES = (FRONT_LEFT_KNEE, FRONT_RIGHT_KNEE)
FRONT_ELBOWS = (FRONT_LEFT_ELBOW, FRONT_RIGHT_ELBOW)
REAR_PAWS = (REAR_LEFT_PAW, REAR_RIGHT_PAW)
REAR_KNEES = (REAR_LEFT_KNEE, REAR_RIGHT_KNEE)
REAR_ELBOWS = (REAR_LEFT_ELBOW, REAR_RIGHT_ELBOW)
ALL_PAWS = FRONT_PAWS + REAR_PAWS
ALL_KNEES = FRONT_KNEES + REAR_KNEES
HEAD_POINTS = (NOSE, CHIN, LEFT_EAR_BASE, RIGHT_EAR_BASE,
               LEFT_EAR_TIP, RIGHT_EAR_TIP, LEFT_EYE, RIGHT_EYE)

# ===== 22 类行为类别（P0 基础 8 + P1 训练 8 + P2 高级 6）=====
P0_BEHAVIORS = (
    "sit", "down", "stand", "heel",
    "sit_up", "stay", "bark", "bite",
)
NUM_P0_BEHAVIORS = 8

P1_BEHAVIORS = (
    "track", "alert_sit", "alert_down", "apprehend",
    "escort", "obstacle", "recall", "watch",
)
NUM_P1_BEHAVIORS = 8

P2_BEHAVIORS = (
    "guard", "release", "retrieve", "jump", "scale", "search_blind",
)
NUM_P2_BEHAVIORS = 6

ALL_BEHAVIORS_22 = P0_BEHAVIORS + P1_BEHAVIORS + P2_BEHAVIORS
NUM_BEHAVIORS_22 = NUM_P0_BEHAVIORS + NUM_P1_BEHAVIORS + NUM_P2_BEHAVIORS  # 22

BEHAVIOR_NAMES_CN = {
    "sit": "坐", "down": "卧", "stand": "立", "heel": "随行",
    "sit_up": "坐立", "stay": "停留", "bark": "叫", "bite": "咬",
    "track": "追踪", "alert_sit": "示警坐", "alert_down": "示警卧",
    "apprehend": "扑咬", "escort": "押解", "obstacle": "障碍穿越",
    "recall": "返回", "watch": "警戒",
    "guard": "守卫", "release": "放口", "retrieve": "衔取",
    "jump": "跳跃", "scale": "攀登", "search_blind": "搜索盲区",
}

BEHAVIOR_SUBJECTS = {
    "sit": "服从", "down": "服从", "stand": "服从", "heel": "服从",
    "sit_up": "服从", "stay": "服从", "bark": "服从/警戒", "bite": "服从",
    "track": "追踪", "alert_sit": "搜毒/搜爆", "alert_down": "血迹搜索",
    "apprehend": "搜捕", "escort": "搜捕", "obstacle": "服从/巡逻",
    "recall": "服从", "watch": "巡逻",
    "guard": "IGP-C 护卫", "release": "IGP-C 护卫",
    "retrieve": "IGP-B 服从", "jump": "IGP-B 服从",
    "scale": "IGP-B 服从", "search_blind": "IGP-A 追踪",
}

__all__ = [
    "NUM_KEYPOINTS", "WITHERS", "THROAT", "NOSE", "CHIN",
    "LEFT_EAR_BASE", "RIGHT_EAR_BASE", "LEFT_EAR_TIP", "RIGHT_EAR_TIP",
    "LEFT_EYE", "RIGHT_EYE", "TAIL_START", "TAIL_END",
    "FRONT_LEFT_PAW", "FRONT_LEFT_KNEE", "FRONT_LEFT_ELBOW",
    "REAR_LEFT_PAW", "REAR_LEFT_KNEE", "REAR_LEFT_ELBOW",
    "FRONT_RIGHT_PAW", "FRONT_RIGHT_KNEE", "FRONT_RIGHT_ELBOW",
    "REAR_RIGHT_PAW", "REAR_RIGHT_KNEE", "REAR_RIGHT_ELBOW",
    "FRONT_PAWS", "FRONT_KNEES", "FRONT_ELBOWS",
    "REAR_PAWS", "REAR_KNEES", "REAR_ELBOWS", "ALL_PAWS", "ALL_KNEES",
    "HEAD_POINTS",
    "P0_BEHAVIORS", "P1_BEHAVIORS", "P2_BEHAVIORS",
    "ALL_BEHAVIORS_22", "NUM_BEHAVIORS_22",
    "BEHAVIOR_NAMES_CN", "BEHAVIOR_SUBJECTS",
]
