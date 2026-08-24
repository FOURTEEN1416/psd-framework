# -*- coding: utf-8 -*-
"""W20 AK→PSD 部分类协议映射测试.

真值依据: K9 data/animal_kingdom/action_recognition/annotation/df_action.xlsx (140 动作表)
  - 权威列 = ``index`` 列（rescue-plan §0 表中 2 处括号数字系 S/N 列笔误:
    Lying on its side 真实 index=74 而非 73(=Licking); Jumping 真实 index=67
    而非 15(=Chirping)。本测试以 xlsx 真值为准并在报告中披露差异。）
协议: 仅 12±2 个可诚实映射的 PSD 类（部分类协议，参照 P0.1 dog-ID 代理探针披露范式）
"""
import pytest

from psd.data.ak_mapping import (
    AK_INDEX_TO_PSD,
    EXCLUDED_CLASSES,
    MAPPED_PSD_CLASSES,
    MAPPING_STRENGTH,
    PSD_PARTIAL_CLASS_TO_IDX,
    map_ak_index,
    resolve_primary_label,
)
from psd.data.synth_stgcn import ALL_BEHAVIORS_22


class TestSpotChecks:
    """df_action.xlsx 真值 spot check（每项均经 pandas 读表核验）。"""

    def test_barking_maps_to_bark(self):
        assert AK_INDEX_TO_PSD[3] == "bark"

    def test_sitting_maps_to_sit(self):
        assert AK_INDEX_TO_PSD[108] == "sit"

    def test_attacking_maps_to_apprehend_not_sit(self):
        # W16 序数毒映射曾把 index 1 当 sit；xlsx 真值 index 1 = Attacking
        assert AK_INDEX_TO_PSD[1] == "apprehend"

    def test_jumping_real_index_is_67(self):
        # df_action.xlsx: Jumping 的 index 列 = 67（S/N 列才是 15）
        assert AK_INDEX_TO_PSD[67] == "jump"

    def test_lying_side_real_index_is_74(self):
        # df_action.xlsx: Lying on its side 的 index 列 = 74（S/N 列才是 73）
        assert AK_INDEX_TO_PSD[74] == "down"

    def test_all_fourteen_source_indices_present(self):
        expected_sources = {108, 70, 74, 116, 68, 3, 8, 2, 102, 1, 13, 16, 67, 45, 14}
        assert set(AK_INDEX_TO_PSD.keys()) == expected_sources


class TestExcludedClasses:
    """10 个 K9 特有类零覆盖断言。"""

    def test_excluded_list_complete(self):
        assert sorted(EXCLUDED_CLASSES) == sorted(
            [
                "heel", "sit_up", "alert_sit", "alert_down", "escort",
                "recall", "guard", "release", "search_blind", "obstacle",
            ]
        )

    def test_excluded_never_in_mapping_values(self):
        values = set(AK_INDEX_TO_PSD.values())
        for cls in EXCLUDED_CLASSES:
            assert cls not in values, f"excluded class {cls} 泄漏进映射值域"

    def test_partition_covers_all_22(self):
        # 12 可映射 + 10 排除 = 22 全集，无交集无遗漏
        mapped = set(MAPPED_PSD_CLASSES)
        assert len(mapped) == 12
        assert mapped | set(EXCLUDED_CLASSES) == set(ALL_BEHAVIORS_22)
        assert mapped.isdisjoint(EXCLUDED_CLASSES)


class TestAntiOrdinalRegression:
    """反向证伪回归：W16 序数假设「1-22 按索引对齐」永久性禁止复活。"""

    def test_old_ordinal_1_is_not_sit(self):
        assert AK_INDEX_TO_PSD.get(1) != "sit"  # 1 = Attacking

    def test_old_ordinal_2_is_not_down(self):
        assert AK_INDEX_TO_PSD.get(2) != "down"  # 2 = Attending

    def test_old_ordinal_5_is_not_stand(self):
        # 5 = Being Carried In Mouth（未被本协议映射，必须返回 None 而非 stand）
        assert map_ak_index(5) is None
        assert AK_INDEX_TO_PSD.get(5) != "stand"

    def test_sn_trap_73_is_not_any_class(self):
        # 73 = Licking（rescue-plan §0 笔误来源）；绝不允许映射进任何 PSD 类
        assert map_ak_index(73) is None

    def test_sn_trap_15_is_not_jump(self):
        # 15 = Chirping（鸣叫）；照抄会把鸟鸣训练成跳跃
        assert map_ak_index(15) is None


class TestMappingHelpers:
    """多标签取首规则与查询接口。"""

    def test_resolve_primary_takes_first_item(self):
        assert resolve_primary_label("2,40") == 2
        assert resolve_primary_label("102,1,39") == 102
        assert resolve_primary_label("68") == 68

    def test_map_ak_index_hit_and_miss(self):
        assert map_ak_index(108) == "sit"
        assert map_ak_index(999) is None
        assert map_ak_index(-1) is None


class TestStrengthAndIndexing:
    """强度三档与部分类编号。"""

    def test_strength_covers_all_mapped_classes(self):
        assert set(MAPPING_STRENGTH.keys()) == set(MAPPED_PSD_CLASSES)

    def test_strength_values_valid(self):
        assert set(MAPPING_STRENGTH.values()) <= {"strong", "medium", "weak"}

    def test_partial_index_bijective_ordered_by_assets_map(self):
        # 编号顺序 = assets-map §1 的 22 类相对顺序（升序子序列），保证与合成层口径可对齐
        order_in_22 = [b for b in ALL_BEHAVIORS_22 if b in PSD_PARTIAL_CLASS_TO_IDX]
        assert order_in_22 == MAPPED_PSD_CLASSES
        assert sorted(PSD_PARTIAL_CLASS_TO_IDX.values()) == list(range(len(MAPPED_PSD_CLASSES)))

    @pytest.mark.parametrize(
        "psd,strength",
        [
            ("sit", "strong"), ("down", "strong"), ("stand", "strong"),
            ("bark", "strong"), ("bite", "strong"), ("jump", "strong"),
            ("stay", "medium"), ("watch", "medium"), ("apprehend", "medium"),
            ("retrieve", "medium"), ("scale", "medium"), ("track", "weak"),
        ],
    )
    def test_strength_table_matches_rescue_plan(self, psd, strength):
        assert MAPPING_STRENGTH[psd] == strength
