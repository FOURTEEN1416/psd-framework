"""ST-GCN+BC 标签映射测试.

TDD 前置：确认 22 类 idx↔name 双向一致。
Owner: W11 窗口
"""
import pytest


def test_num_behaviors_is_22():
    from psd.models.stgcn_bc_labels import NUM_BEHAVIORS
    assert NUM_BEHAVIORS == 22


def test_behavior_to_idx_all_22():
    from psd.models.stgcn_bc_labels import BEHAVIOR_TO_IDX, NUM_BEHAVIORS
    assert len(BEHAVIOR_TO_IDX) == NUM_BEHAVIORS
    assert all(isinstance(v, int) for v in BEHAVIOR_TO_IDX.values())
    assert set(BEHAVIOR_TO_IDX.values()) == set(range(NUM_BEHAVIORS))


def test_idx_to_behavior_inverse():
    from psd.models.stgcn_bc_labels import BEHAVIOR_TO_IDX, IDX_TO_BEHAVIOR, NUM_BEHAVIORS
    assert len(IDX_TO_BEHAVIOR) == NUM_BEHAVIORS
    for idx, name in IDX_TO_BEHAVIOR.items():
        assert BEHAVIOR_TO_IDX[name] == idx
    for name, idx in BEHAVIOR_TO_IDX.items():
        assert IDX_TO_BEHAVIOR[idx] == name


def test_p0_p1_p2_idx_groups():
    from psd.models.stgcn_bc_labels import P0_IDX, P1_IDX, P2_IDX
    assert P0_IDX == list(range(0, 8))
    assert P1_IDX == list(range(8, 16))
    assert P2_IDX == list(range(16, 22))


def test_get_behavior_idx_raises_on_unknown():
    from psd.models.stgcn_bc_labels import get_behavior_idx
    with pytest.raises(KeyError):
        get_behavior_idx("nonexistent_behavior")


def test_get_behavior_name_raises_on_out_of_range():
    from psd.models.stgcn_bc_labels import get_behavior_name
    with pytest.raises(KeyError):
        get_behavior_name(99)


def test_get_all_labels_returns_22():
    from psd.models.stgcn_bc_labels import get_all_labels
    labels = get_all_labels()
    assert len(labels) == 22
    for item in labels:
        assert len(item) == 5  # (idx, name, cn, layer, stage)


def test_layer_labels():
    from psd.models.stgcn_bc_labels import LAYER_LABELS
    assert len(LAYER_LABELS) == 22
    assert LAYER_LABELS[:8] == ["P0"] * 8
    assert LAYER_LABELS[8:16] == ["P1"] * 8
    assert LAYER_LABELS[16:] == ["P2"] * 6
