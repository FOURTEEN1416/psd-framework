# -*- coding: utf-8 -*-
"""W40 数据飞轮 round2 — TDD 测试（任务书: dev-docs/handovers/NEXT-BATCH-plan.md §W40）.

预注册配置: configs/public_real_round2.yaml（commit 996e9b6 先于本文件）

覆盖五轴:
  1. APTv2 调和变换: 序列级 bbox 归一化 / NaN 补零 / 目标域死掩码 / vis01 保真 / 无 NaN 输出
  2. w35 直通保真: 数组逐位不变
  3. DogSet 运动学先验门禁: 无量纲速度比 / 阈值带 / 物理离群硬排除 / >20% 退化保护
  4. AdaBN 机制: BN 统计量更新而权重与 head 冻结 / 常量输入累积均值精确性 / momentum=None
  5. 适应集组装契约: 仅 pretrain_geometric 槽位 / 确定性排序 / 变长 T 分桶 / 协议一致性回显
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from scripts.public_real_round2_lib import (  # noqa: E402
    APTV2_NAN_SLOTS,
    TARGET_DEAD_JOINTS,
    adabn_adapt,
    apply_kinematic_gate,
    build_adaptation_set,
    harmonize_aptv2_keypoints,
    kinematic_gate_thresholds,
    kinematic_ratio,
    make_train_config,
    prepare_w35_keypoints,
)

RNG_SEED_APTV2 = 1234
RNG_SEED_W35 = 5678


def _rng(seed: int) -> np.random.Generator:
    """每夹具独立固定种子——测试结果与执行顺序无关。"""
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# 夹具: 池条目 schema 重演（沿 W30 测试的"按真实 schema 重演"模式）
# ---------------------------------------------------------------------------

def _mk_entry(sample_id: str, source: str, usage: str, kp: np.ndarray, topo: str = "K9Graph") -> dict:
    t, v, _ = kp.shape
    return {
        "sample_id": sample_id,
        "source_channel": source,
        "split": "train",
        "topology_name": topo,
        "V": v,
        "T": t,
        "keypoints": kp.astype(np.float32),
        "coords_semantic": "fixture",
        "fps_or_sampling": 30.0,
        "usage_scope": usage,
        "label_status": "deferred_pixel_domain",
        "static": False,
        "provenance": {},
    }


def _mk_aptv2(T: int = 15, shift: float = 500.0, scale: float = 800.0) -> np.ndarray:
    """原始像素域 APTv2 形态: xy ∈ [shift, shift+scale], NaN@7槽位, ch3∈{0,1}."""
    g = _rng(RNG_SEED_APTV2 + T)
    xy = g.random((T, 24, 2)) * scale + shift
    vis = (g.random((T, 24, 1)) > 0.2).astype(np.float32)
    kp = np.concatenate([xy, vis], axis=2).astype(np.float32)
    kp[:, list(APTV2_NAN_SLOTS), :] = np.nan
    # 少量不可见点置 NaN（模拟源标注缺失）
    kp[0, 16, :] = np.nan
    return kp


def _mk_w35(T: int = 30) -> np.ndarray:
    """归一化死掩码形态: xy∈[0,1], 死关节全零, ch3∈[0,1] 连续 conf."""
    g = _rng(RNG_SEED_W35 + T)
    xy = g.random((T, 24, 2))
    conf = g.random((T, 24, 1)) * 0.9
    kp = np.concatenate([xy, conf], axis=2).astype(np.float32)
    kp[:, list(TARGET_DEAD_JOINTS), :] = 0.0
    return kp


# ---------------------------------------------------------------------------
# 1. APTv2 调和变换
# ---------------------------------------------------------------------------

class TestHarmonizeAptv2:
    def test_output_finite_and_shape(self):
        out = harmonize_aptv2_keypoints(_mk_aptv2())
        assert out.shape == (15, 24, 3)
        assert np.isfinite(out).all(), "调和后不允许残留 NaN/Inf"

    def test_nan_and_dead_slots_zeroed(self):
        out = harmonize_aptv2_keypoints(_mk_aptv2())
        for j in APTV2_NAN_SLOTS | TARGET_DEAD_JOINTS:
            assert np.all(out[:, j, :] == 0.0), f"关节 {j} 应为补零/死掩码"

    def test_bbox_normalization_range(self):
        out = harmonize_aptv2_keypoints(_mk_aptv2(shift=500.0, scale=800.0))
        live = [j for j in range(24) if j not in (APTV2_NAN_SLOTS | TARGET_DEAD_JOINTS)]
        xy = out[:, live, :2][out[:, live, 2] > 0] if False else np.concatenate(
            [out[:, j, :2][out[:, j, 2] > 0] for j in live], axis=0)
        # 序列级联合包围盒 → 可见点应铺满 [0,1]
        assert xy.min() == pytest.approx(0.0, abs=1e-5)
        assert xy.max() == pytest.approx(1.0, abs=1e-5)

    def test_vis01_channel_preserved(self):
        kp = _mk_aptv2()
        out = harmonize_aptv2_keypoints(kp)
        live = [j for j in range(24) if j not in (APTV2_NAN_SLOTS | TARGET_DEAD_JOINTS)]
        for j in live[:5]:
            vis_src = kp[:, j, 2]
            vis_src = np.nan_to_num(vis_src, nan=0.0)
            assert np.allclose(out[:, j, 2], vis_src), "ch3 必须保持 vis01 原值"

    def test_deterministic(self):
        kp = _mk_aptv2()
        a = harmonize_aptv2_keypoints(kp)
        b = harmonize_aptv2_keypoints(kp)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 2. w35 直通保真
# ---------------------------------------------------------------------------

class TestW35Passthrough:
    def test_bitwise_identical(self):
        kp = _mk_w35()
        out = prepare_w35_keypoints(kp)
        assert np.array_equal(np.asarray(out), kp)
        assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. DogSet 运动学先验门禁
# ---------------------------------------------------------------------------

class TestKinematicPrior:
    def test_ratio_zero_for_static(self):
        kp = np.full((10, 21, 3), 5.0, dtype=np.float32)  # 静止姿态（度量域）
        assert kinematic_ratio(kp, fps=60.0, dims=3) == pytest.approx(0.0, abs=1e-8)

    def test_ratio_positive_for_motion(self):
        kp = _rng(99).random((100, 21, 3)).astype(np.float32) * 50.0
        r = kinematic_ratio(kp, fps=60.0, dims=3)
        assert r > 0.0

    def test_dims_guard_for_conf_channel(self):
        """clips 的 ch3 是 conf 不是坐标——dims=2 时速度只算 xy。"""
        kp = _mk_w35()
        r2 = kinematic_ratio(kp, fps=30.0, dims=2)
        assert np.isfinite(r2)

    def test_thresholds_percentile_band(self):
        refs = [0.1 * i for i in range(100)]
        th = kinematic_gate_thresholds(refs)
        assert th["lo"] == pytest.approx(float(np.percentile(refs, 0.5)))
        assert th["hi"] == pytest.approx(float(np.percentile(refs, 99.5)))

    def test_hard_exclusion_only_extreme_outliers(self):
        refs = [0.01] * 98 + [0.02, 0.02]           # 干净参考分布
        th = kinematic_gate_thresholds(refs)
        # 足量正常样本 + 单个毛刺 → 排除率 1/21 ≈ 4.8% < 20%，不触发退化保护
        samples = {f"n{i:02d}": 0.01 for i in range(20)}
        samples["glitch"] = th["hi"] * 10.0
        kept, rep = apply_kinematic_gate(samples, th)
        assert kept == [f"n{i:02d}" for i in range(20)]
        assert rep["excluded"] == ["glitch"]
        assert rep["report_only"] is False

    def test_degenerate_guard_over_20pct(self):
        refs = [0.01] * 100
        th = kinematic_gate_thresholds(refs)
        samples = {f"s{i}": 0.05 for i in range(5)}  # 全部越界 → 排除率 100% > 20%
        kept, rep = apply_kinematic_gate(samples, th)
        assert len(kept) == 5                        # 退化为纯报告模式
        assert rep["report_only"] is True


# ---------------------------------------------------------------------------
# 4. AdaBN 机制
# ---------------------------------------------------------------------------

class TestAdabnAdapt:
    def _model(self):
        from psd.models.stgcn_bc import STGCNBC
        torch.manual_seed(7)
        return STGCNBC(in_channels=3, num_classes=4)

    def _bn_modules(self, model):
        from torch.nn.modules.batchnorm import _BatchNorm
        return [m for m in model.backbone.modules() if isinstance(m, _BatchNorm)]

    def test_bn_stats_move_weights_frozen_head_touched_not(self):
        model = self._model()
        w_before = {n: p.detach().clone() for n, p in model.named_parameters()}
        arrays = [(_rng(31).random((30, 24, 3)).astype(np.float32) * 0.5 + 0.25)
                  for _ in range(8)]
        summary = adabn_adapt(model, arrays, batch_size=4, seed=42, device="cpu")
        assert summary["n_forward_samples"] == 8
        # 权重全部不动
        for n, p in model.named_parameters():
            assert torch.equal(w_before[n], p.detach()), f"参数 {n} 被 AdaBN 污染"
        # BN 统计量确实移动
        moved = sum(summary["per_bn_moved"])
        assert moved > 0, "至少一个 BN 统计量应发生变化"
        # 适应结束后 BN 回到 eval
        assert all(not bn.training for bn in self._bn_modules(model))

    def test_cumulative_mean_exact_on_constant_input(self):
        model = self._model()
        const = 0.37
        arrays = [np.full((30, 24, 3), const, dtype=np.float32) for _ in range(6)]
        adabn_adapt(model, arrays, batch_size=4, seed=42, device="cpu")
        dbn = model.backbone.data_bn
        assert torch.allclose(dbn.running_mean, torch.full_like(dbn.running_mean, const),
                              atol=1e-4), "momentum=None 单遍累积均值应对常量输入精确"

    def test_variable_length_buckets(self):
        model = self._model()
        arrays = [_mk_w35(30) for _ in range(4)] + [_mk_aptv2_harmonized(15) for _ in range(3)]
        summary = adabn_adapt(model, arrays, batch_size=4, seed=42, device="cpu")
        assert summary["buckets"] == {30: 4, 15: 3}


def _mk_aptv2_harmonized(T: int) -> np.ndarray:
    return harmonize_aptv2_keypoints(_mk_aptv2(T=T))


# ---------------------------------------------------------------------------
# 5. 适应集组装契约 + 协议一致性
# ---------------------------------------------------------------------------

class TestBuildAdaptationSet:
    def test_slot_selection_and_order(self):
        entries = [
            _mk_entry("w03", "video_c1_w35", "pretrain_geometric", _mk_w35()),
            _mk_entry("a01", "aptv2_c2_w26", "pretrain_geometric", _mk_aptv2()),
            _mk_entry("d01", "dogpose_c5_w29", "augment_static",
                      np.full((1, 24, 3), 0.5, dtype=np.float32)),
            _mk_entry("m01", "mocap_c3_w27", "kinematic_prior",
                      _rng(77).random((50, 21, 3)).astype(np.float32), topo="mann_dogset_21j"),
            _mk_entry("a00", "aptv2_c2_w26", "pretrain_geometric", _mk_aptv2()),
            _mk_entry("k01", "ak_public_q3b", "supervised_partialclass4", _mk_w35()),
        ]
        arrays, metas, report = build_adaptation_set(
            entries, gate_ref=None, fps_assumptions={"aptv2_c2_w26": 15.0})
        ids = [m["sample_id"] for m in metas]
        assert ids == sorted(ids), "必须确定性排序（sample_id 升序）"
        assert set(ids) == {"a00", "a01", "w03"}, "仅 pretrain_geometric 槽位入选"
        assert report["counts"]["aptv2_c2_w26"] == 2
        assert report["counts"]["video_c1_w35"] == 1

    def test_gate_reference_integration(self):
        entries = [
            _mk_entry("w01", "video_c1_w35", "pretrain_geometric", _mk_w35()),
            _mk_entry("a01", "aptv2_c2_w26", "pretrain_geometric", _mk_aptv2()),
        ]
        gate_ref = {"lo": 0.0, "hi": 1e9}   # 宽松带 → 全保留
        arrays, metas, report = build_adaptation_set(
            entries, gate_ref=gate_ref, fps_assumptions={"aptv2_c2_w26": 15.0})
        assert len(arrays) == 2
        assert report["gate"]["report_only"] is False
        assert report["gate"]["excluded"] == []

    def test_meta_records_transform_provenance(self):
        entries = [_mk_entry("a01", "aptv2_c2_w26", "pretrain_geometric", _mk_aptv2())]
        _, metas, _ = build_adaptation_set(entries, gate_ref=None, fps_assumptions={"aptv2_c2_w26": 15.0})
        assert metas[0]["transform"] == "bbox_norm+nanslot_fill+deadmask_harmonize"
        entries2 = [_mk_entry("w01", "video_c1_w35", "pretrain_geometric", _mk_w35())]
        _, metas2, _ = build_adaptation_set(entries2, gate_ref=None, fps_assumptions={})
        assert metas2[0]["transform"] == "passthrough"


class TestProtocolIdentity:
    def test_phase_b_echo_matches_round1(self):
        cfg_path = REPO_ROOT / "configs" / "public_real_round2.yaml"
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        echo = cfg["round1_baseline"]["protocol_echo"]
        tc = make_train_config(echo)
        assert tc.epochs == 60
        assert tc.batch_size == 16
        assert tc.patience == 15
        assert tc.seed == 42
        assert tc.use_amp is True

    def test_archived_baseline_values_in_config(self):
        cfg_path = REPO_ROOT / "configs" / "public_real_round2.yaml"
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        base = cfg["round1_baseline"]
        assert base["archived_best_val_acc"] == pytest.approx(44.90 / 100, abs=1e-4)
        assert base["protocol_echo"]["split"] == {"train": 123, "val": 49}
