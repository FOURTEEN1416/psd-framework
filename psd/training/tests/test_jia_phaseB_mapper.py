"""P0.3 Phase B 映射器测试（W13 任务书，TDD 先行）。

覆盖：合成语义原型构建、22 类最近原型映射、Phase A 原型桥接诊断、
伪标签池行构造（与 P0.4 消费接口字段对齐）、覆盖率/精度指标。

设计约束（任务书）：
- Phase A 代码只读 import，本模块不修改其行为；
- 22 类清单以 assets-map.md §1 为唯一 truth（经 psd.data.synth_stgcn.ALL_BEHAVIORS_22 引用，
  该副本由 assets-map §3.2 登记移植，禁止本测试另抄字面清单）；
- 纯 numpy 可测：编码器注入式，无 torch 依赖。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.training.jia_phaseB_mapper import (
    POOL_SCHEMA_FIELDS,
    build_pool_rows,
    build_semantic_prototypes,
    bridge_map_prototypes,
    coverage_ratio,
    map_embeddings_to_22,
    nearest_semantic_label,
    split_synthetic_ref_probe,
)
from psd.data.synth_stgcn import ALL_BEHAVIORS_22, NUM_CLASSES


# ------------------------------------------------------------ 测试工具

def _blob(center: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return center + rng.normal(0, 0.02, (n, len(center)))


def _make_synthetic_embeddings(n_per_class: int = 4, dim: int = 8):
    """构造 22 类完全分离的嵌入簇 + 对齐标签（模拟 Φ 编码后的合成嵌入）。"""
    rng = np.random.default_rng(7)
    centers = rng.normal(0, 5.0, (NUM_CLASSES, dim))
    embs, labels = [], []
    for ci in range(NUM_CLASSES):
        embs.append(_blob(centers[ci], n_per_class, seed=100 + ci))
        labels.extend([ALL_BEHAVIORS_22[ci]] * n_per_class)
    return np.vstack(embs).astype(np.float32), np.array(labels)


# ------------------------------------------------------------ 语义原型

class TestBuildSemanticPrototypes:
    def test_shapes_and_unit_norm(self):
        emb, labels = _make_synthetic_embeddings()
        protos, proto_labels = build_semantic_prototypes(emb, labels)
        assert protos.shape == (NUM_CLASSES, emb.shape[1])
        assert list(proto_labels) == sorted(ALL_BEHAVIORS_22)
        np.testing.assert_allclose(np.linalg.norm(protos, axis=1), 1.0, rtol=1e-6)

    def test_deterministic(self):
        emb, labels = _make_synthetic_embeddings()
        p1, l1 = build_semantic_prototypes(emb, labels)
        p2, l2 = build_semantic_prototypes(emb, labels)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(l1, l2)

    def test_rejects_non_finite(self):
        emb, labels = _make_synthetic_embeddings()
        emb = emb.copy()
        emb[0, 0] = np.nan
        with pytest.raises(ValueError):
            build_semantic_prototypes(emb, labels)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            build_semantic_prototypes(np.zeros((0, 8), dtype=np.float32), np.array([]))


# ------------------------------------------------------------ 22 类映射

class TestMapEmbeddingsTo22:
    def test_separated_clusters_recover_true_labels(self):
        emb, labels = _make_synthetic_embeddings(n_per_class=6)
        # 分层留出：ref 建原型（覆盖全部 22 类），probe 验证映射
        ref_idx, probe_idx = split_synthetic_ref_probe(labels, seed=42)
        protos, proto_labels = build_semantic_prototypes(emb[ref_idx], labels[ref_idx])
        pred, sim = map_embeddings_to_22(emb[probe_idx], protos, proto_labels)
        assert (pred == labels[probe_idx]).all()
        assert sim.shape == (len(pred),)
        assert ((sim > 0.9) & (sim <= 1.0)).all()

    def test_prediction_domain_is_22class_list(self):
        emb, labels = _make_synthetic_embeddings()
        protos, proto_labels = build_semantic_prototypes(emb, labels)
        pred, _ = map_embeddings_to_22(emb, protos, proto_labels)
        assert set(pred.tolist()).issubset(set(ALL_BEHAVIORS_22))

    def test_nearest_semantic_label_single_point(self):
        emb, labels = _make_synthetic_embeddings()
        protos, proto_labels = build_semantic_prototypes(emb, labels)
        idx, name, sim = nearest_semantic_label(emb[3], protos, proto_labels)
        assert name == labels[3]
        assert 0 <= idx < NUM_CLASSES
        assert -1.0001 <= sim <= 1.0001


# ------------------------------------------------------------ 桥接诊断

class TestBridgeMapPrototypes:
    def test_perfect_bridge_maps_proto_to_own_cluster_label_equivalent(self):
        # Phase A 原型取自两个合成类簇的均值 → 桥接应各自命中最近 22 类
        emb, labels = _make_synthetic_embeddings(n_per_class=6)
        protos22, names22 = build_semantic_prototypes(emb, labels)
        pa_protos = np.vstack([
            emb[labels == "sit"].mean(axis=0),
            emb[labels == "down"].mean(axis=0),
        ])
        pa_labels = np.array(["sitting", "lying"])  # 物理先验名仅作标识
        mapping, sim_mat = bridge_map_prototypes(pa_protos, pa_labels, protos22, names22)
        assert mapping["sitting"] == "sit"
        assert mapping["lying"] == "down"
        assert sim_mat.shape == (2, NUM_CLASSES)

    def test_sim_matrix_bounded(self):
        rng = np.random.default_rng(0)
        pa = rng.normal(0, 1, (3, 8)).astype(np.float32)
        s22 = rng.normal(0, 1, (5, 8)).astype(np.float32)
        names = ALL_BEHAVIORS_22[:5]
        _, sim_mat = bridge_map_prototypes(pa, np.array(["a", "b", "c"]), s22, names)
        assert (np.abs(sim_mat) <= 1.0 + 1e-6).all()


# ------------------------------------------------------------ 留出切分

class TestSplitSyntheticRefProbe:
    def test_stratified_disjoint_and_complete(self):
        emb, labels = _make_synthetic_embeddings(n_per_class=10)
        ref_idx, probe_idx = split_synthetic_ref_probe(labels, seed=42)
        assert len(ref_idx) + len(probe_idx) == len(labels)
        assert not (set(ref_idx) & set(probe_idx))
        ref_counts = {c: int((labels[ref_idx] == c).sum()) for c in set(labels)}
        probe_counts = {c: int((labels[probe_idx] == c).sum()) for c in set(labels)}
        assert all(v == 5 for v in ref_counts.values())
        assert all(v == 5 for v in probe_counts.values())

    def test_deterministic_given_seed(self):
        _, labels = _make_synthetic_embeddings(n_per_class=4)
        r1, p1 = split_synthetic_ref_probe(labels, seed=42)
        r2, p2 = split_synthetic_ref_probe(labels, seed=42)
        assert list(r1) == list(r2) and list(p1) == list(p2)


# ------------------------------------------------------------ 池行构造

_SEGMENTS = [
    {"clip_id": "interpet_dog01_p01_take01_ego_001", "start_frame": 10, "end_frame": 40,
     "label": "standing", "confidence": 0.92, "rule_ids": ["standing_posture"]},
    {"clip_id": "interpet_dog01_p01_take01_ego_001", "start_frame": 41, "end_frame": 80,
     "label": "sitting", "confidence": 0.95, "rule_ids": ["sitting_posture"]},
]


class TestBuildPoolRows:
    def _prepare(self):
        emb, labels = _make_synthetic_embeddings()
        protos, names = build_semantic_prototypes(emb, labels)
        seg_emb = np.asarray([[1.0] + [0.0] * 7, [0.0] * 7 + [1.0]], dtype=np.float32)
        proto_idx = np.array([0, 1])
        kappa = np.array([0.31, 0.12])
        rows = build_pool_rows(
            segments=_SEGMENTS, seg_emb=seg_emb, proto_idx=proto_idx, kappa=kappa,
            semantic_protos=protos, semantic_names=names,
            embedding_ref="segment_embeddings_test.npz",
            label_source="p03_phaseB_seed42",
        )
        return rows

    def test_row_schema_matches_p04_interface(self):
        rows = self._prepare()
        expected = {
            "clip_id", "start_frame", "end_frame", "pseudo_label", "proto_idx",
            "kappa_margin", "tau_pass", "embedding_ref", "label_source", "metric_layer",
        }
        for row in rows:
            assert expected == set(row.keys())

    def test_labels_in_22class_domain(self):
        rows = self._prepare()
        assert all(r["pseudo_label"] in ALL_BEHAVIORS_22 for r in rows)

    def test_tau_pass_true_at_zero_threshold(self):
        rows = self._prepare()
        assert all(r["tau_pass"] is True for r in rows)

    def test_metric_layer_is_public_real(self):
        rows = self._prepare()
        assert all(r["metric_layer"] == "public_real_physics_prior" for r in rows)

    def test_length_matches_segments(self):
        rows = self._prepare()
        assert len(rows) == len(_SEGMENTS)


# ------------------------------------------------------------ 指标

class TestMetrics:
    def test_coverage_ratio_full(self):
        tau_pass = np.array([True, True, True])
        assert coverage_ratio(tau_pass) == pytest.approx(1.0)

    def test_accuracy_computation(self):
        pred = np.array(["sit", "down", "stand"])
        truth = np.array(["sit", "sit", "stand"])
        acc = float((pred == truth).mean())
        assert acc == pytest.approx(2 / 3)
