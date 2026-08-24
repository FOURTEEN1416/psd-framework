"""ST-GCN+BC 常量定义测试.

TDD 前置：确认 22 类标签、关键点数、分层分组正确。
Owner: W11 窗口
"""
import pytest


def test_num_keypoints_is_24():
    from psd.models.stgcn_bc_constants import NUM_KEYPOINTS
    assert NUM_KEYPOINTS == 24


def test_all_behaviors_22_has_22_items():
    from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22
    assert len(ALL_BEHAVIORS_22) == 22


def test_p0_p1_p2_split():
    from psd.models.stgcn_bc_constants import P0_BEHAVIORS, P1_BEHAVIORS, P2_BEHAVIORS
    assert len(P0_BEHAVIORS) == 8
    assert len(P1_BEHAVIORS) == 8
    assert len(P2_BEHAVIORS) == 6


def test_all_behaviors_is_concatenation():
    from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22, P0_BEHAVIORS, P1_BEHAVIORS, P2_BEHAVIORS
    assert ALL_BEHAVIORS_22 == P0_BEHAVIORS + P1_BEHAVIORS + P2_BEHAVIORS


def test_known_behavior_names():
    from psd.models.stgcn_bc_constants import ALL_BEHAVIORS_22
    assert "sit" in ALL_BEHAVIORS_22
    assert "jump" in ALL_BEHAVIORS_22
    assert "search_blind" in ALL_BEHAVIORS_22
    assert "bite" in ALL_BEHAVIORS_22


def test_chinese_names_coverage():
    from psd.models.stgcn_bc_constants import BEHAVIOR_NAMES_CN, ALL_BEHAVIORS_22
    for name in ALL_BEHAVIORS_22:
        assert name in BEHAVIOR_NAMES_CN, f"Missing CN name for {name}"


def test_witners_index():
    from psd.models.stgcn_bc_constants import WITHERS, NUM_KEYPOINTS
    assert WITHERS == 22
    assert WITHERS < NUM_KEYPOINTS
