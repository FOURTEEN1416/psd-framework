"""ST-GCN+BC 特征抽取器测试（W13-C1 任务书，TDD 先行）。

覆盖（任务书 §三.1 三用例 + 输入居中工具）：
1. checkpoint 加载成功且输出维度正确（trainer 格式 roundtrip，不依赖真权重）
2. 同输入两次提取结果一致（确定性）
3. 中心化对齐函数：同分布输入输出归零、偏移分布被拉回
4. 真实段输入居中（质心居中 + 非有限值兜底）

全部 CPU 可跑：小尺寸随机权重模型，不加载 17MB 真实 checkpoint。
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from psd.training.stgcnbc_feature_extractor import (
    STGCNBCFeatureExtractor,
    apply_feature_alignment,
    center_keypoints,
    fit_feature_alignment,
)


def _tiny_arch():
    return {"base_channels": 8, "num_stages": 2, "tcn_type": "unit_tcn"}


def _tiny_model():
    from psd.models.stgcn_bc import build_stgcn_bc

    torch.manual_seed(11)
    return build_stgcn_bc(in_channels=3, num_classes=22, **_tiny_arch())


def _save_trainer_format(model, path):
    """按 STGCNBCTrainer._save_checkpoint 的格式落盘（roundtrip 用）。"""
    torch.save({"epoch": 1, "model_state_dict": model.state_dict(),
                "val_acc": 0.5, "best_val_acc": 0.5}, path)


# ------------------------------------------------------------ 用例 1

class TestCheckpointLoadAndDims:
    def test_trainer_format_roundtrip_output_dims(self, tmp_path):
        model = _tiny_model()
        ckpt = tmp_path / "best.pt"
        _save_trainer_format(model, ckpt)

        ext = STGCNBCFeatureExtractor.from_checkpoint(str(ckpt), device="cpu", **_tiny_arch())
        kp = np.random.default_rng(0).normal(0, 0.1, (4, 30, 24, 3)).astype(np.float32)
        feats = ext.extract(kp)

        expected_dim = model.backbone.out_channels
        assert feats.shape == (4, expected_dim)
        assert np.isfinite(feats).all()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            STGCNBCFeatureExtractor.from_checkpoint(str(tmp_path / "nope.pt"))


# ------------------------------------------------------------ 用例 2

class TestDeterminism:
    def test_same_input_identical_features(self, tmp_path):
        model = _tiny_model()
        ckpt = tmp_path / "best.pt"
        _save_trainer_format(model, ckpt)
        ext = STGCNBCFeatureExtractor.from_checkpoint(str(ckpt), device="cpu", **_tiny_arch())

        kp = np.random.default_rng(1).normal(0, 0.1, (3, 30, 24, 3)).astype(np.float32)
        f1 = ext.extract(kp)
        f2 = ext.extract(kp)
        np.testing.assert_array_equal(f1, f2)


# ------------------------------------------------------------ 用例 3

class TestFeatureAlignment:
    def test_same_distribution_zeroed_and_unit_variance(self):
        rng = np.random.default_rng(5)
        ref = rng.normal(0, 1, (300, 16))
        stats = fit_feature_alignment(ref)
        # 居中在归一之前：减均值后列均值≈0
        centered = ref - stats["mean"]
        np.testing.assert_allclose(centered.mean(axis=0), 0.0, atol=1e-8)
        # 对齐输出逐行 L2 归一
        aligned = apply_feature_alignment(ref, stats, use_std=True)
        np.testing.assert_allclose(np.linalg.norm(aligned, axis=1), 1.0, rtol=1e-9)

    def test_shifted_distribution_pulled_back(self):
        """各域用自己的 μ/σ（μ_syn、μ_real 分别计算）→ 偏移被拉回、两域可比。"""
        rng = np.random.default_rng(6)
        ref = rng.normal(0, 1, (300, 16))
        shifted = ref + 7.0
        a_ref = apply_feature_alignment(ref, fit_feature_alignment(ref), use_std=True)
        a_shift = apply_feature_alignment(shifted, fit_feature_alignment(shifted), use_std=True)
        np.testing.assert_allclose(a_shift, a_ref, atol=1e-6)

    def test_cross_stats_does_not_cancel_offset(self):
        """反证：用参照侧统计量对齐偏移数据，偏移保留（防止误实现成全局中心化）。"""
        rng = np.random.default_rng(8)
        ref = rng.normal(0, 1, (200, 8))
        shifted = ref + 7.0
        stats_ref = fit_feature_alignment(ref)
        out = apply_feature_alignment(shifted, stats_ref, use_std=False)
        assert not np.allclose(out, apply_feature_alignment(ref, stats_ref, use_std=False))

    def test_mean_only_mode_matches_plan_b_spec(self):
        rng = np.random.default_rng(7)
        ref = rng.normal(2.0, 3.0, (100, 8))
        stats = fit_feature_alignment(ref)
        aligned = apply_feature_alignment(ref, stats, use_std=False)
        expected = ref - stats["mean"]                      # 只减均值
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
        np.testing.assert_allclose(aligned, expected, atol=1e-12)

    def test_rejects_non_finite(self):
        bad = np.zeros((5, 4))
        bad[0, 0] = np.nan
        with pytest.raises(ValueError):
            fit_feature_alignment(bad)


# ------------------------------------------------------------ 用例 4：输入居中

class TestCenterKeypoints:
    def test_centers_finite_centroid_to_origin(self):
        rng = np.random.default_rng(3)
        kp = rng.normal(0, 0.1, (30, 24, 3)) + np.array([5.0, -3.0, 1.2])
        out = center_keypoints(kp)
        assert out.shape == kp.shape
        np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), 0.0, atol=1e-6)

    def test_non_finite_filled_at_centroid(self):
        kp = np.zeros((10, 24, 3), dtype=np.float64)
        kp += np.array([1.0, 2.0, 3.0])
        kp[5, 7] = np.nan
        out = center_keypoints(kp)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out[5, 7], 0.0, atol=1e-6)

    def test_all_nan_frame_does_not_crash(self):
        kp = np.zeros((10, 24, 3), dtype=np.float64)
        kp[:3] = np.nan
        out = center_keypoints(kp)
        assert np.isfinite(out).all()

    def test_batch_extract_applies_centering(self, tmp_path):
        """整段批量入口：世界坐标偏移不改变特征（平移不变性由居中保证）。"""
        model = _tiny_model()
        ckpt = tmp_path / "best.pt"
        _save_trainer_format(model, ckpt)
        ext = STGCNBCFeatureExtractor.from_checkpoint(str(ckpt), device="cpu", **_tiny_arch())

        rng = np.random.default_rng(9)
        kp = rng.normal(0, 0.05, (2, 30, 24, 3)).astype(np.float32)
        shifted = kp + np.array([10.0, -5.0, 2.0], dtype=np.float32)
        f0 = ext.extract(kp)
        f1 = ext.extract(shifted)
        np.testing.assert_allclose(f0, f1, atol=1e-5)
