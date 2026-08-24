"""P0.5 主动学习效率实验单元测试 — W14 窗口.

覆盖: 熵/随机采样器、增量模拟运行器、真实池打分。
口径: 合成层（主曲线）/ 公开真实层（池打分仅排序清单）。
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from psd.training.active_learning import (
    ALSimulationRunner,
    EntropySelector,
    RandomSelector,
    entropy_scores,
)


# ---------------------------------------------------------------------------
# Task 1: 熵打分与采样器
# ---------------------------------------------------------------------------

class TestEntropyScores:
    def test_uniform_max_onehot_zero(self):
        """均匀分布熵 = log(C)（最大），one-hot 熵 = 0（最小）。"""
        c = 22
        uniform = np.full((5, c), 1.0 / c, dtype=np.float64)
        onehot = np.zeros((3, c), dtype=np.float64)
        onehot[:, 0] = 1.0

        s_uniform = entropy_scores(uniform)
        s_onehot = entropy_scores(onehot)

        assert s_uniform.shape == (5,)
        assert np.allclose(s_uniform, math.log(c), atol=1e-6)
        assert s_onehot.shape == (3,)
        assert np.allclose(s_onehot, 0.0, atol=1e-6)

    def test_monotonic_with_confidence(self):
        """置信度越高熵越低：p=[0.9,0.1] < p=[0.6,0.4]。"""
        probs = np.array([[0.9, 0.1], [0.6, 0.4]])
        s = entropy_scores(probs)
        assert s[0] < s[1]


class TestRandomSelector:
    def test_deterministic_given_rng(self):
        """同 rng 状态两次选择结果一致（可复现）。"""
        rng_a = np.random.default_rng(42)
        rng_b = np.random.default_rng(42)
        sel_a = RandomSelector().select(pool_size=50, k=10, rng=rng_a)
        sel_b = RandomSelector().select(pool_size=50, k=10, rng=rng_b)
        assert sel_a == sel_b

    def test_excludes_labeled_and_within_budget(self):
        """选择不含已标注 id，数量正确，范围合法。"""
        labeled = {0, 1, 2}
        # RandomSelector.select 接受 exclude 集合
        sel = RandomSelector().select(pool_size=20, k=7, rng=np.random.default_rng(0), exclude=labeled)
        assert len(sel) == 7
        assert len(set(sel)) == 7  # 无重复
        assert not (set(sel) & labeled)
        assert all(0 <= i < 20 for i in sel)


class TestEntropySelector:
    def test_picks_highest_entropy_first(self):
        """选中的是熵最大的样本（排除已标注后）。"""
        scores = np.array([0.1, 2.5, 0.3, 9.9, 1.0])
        labeled = [2]
        sel = EntropySelector().select(scores=scores, exclude=set(labeled), k=2)
        assert len(sel) == 2
        # 排除 idx2 后熵 top-2 是 idx3(9.9), idx1(2.5)，且按分降序返回
        assert sel[0] == 3
        assert sel[1] == 1

    def test_k_capped_by_available(self):
        """k 超过剩余可用量时截断到可用量。"""
        scores = np.array([0.1, 2.5, 0.3])
        sel = EntropySelector().select(scores=scores, exclude={0}, k=10)
        assert sorted(sel) == [1, 2]

# ---------------------------------------------------------------------------
# Task 3: 增量式 AL 模拟运行器
# ---------------------------------------------------------------------------

def _tiny_pool_and_val():
    """tiny 合成池/验证集（隔离 seed，避免 W12 seed42 记忆）。"""
    from psd.data.synth_stgcn import make_synthetic_dataset
    pool = make_synthetic_dataset(samples_per_class=2, T=10, noise_std=0.05, seed=20261)   # 44
    val = make_synthetic_dataset(samples_per_class=1, T=10, noise_std=0.05, seed=20262)    # 22
    return pool, val


class TestALSimulationRunner:
    def _make_runner(self, budgets=(4, 8)):
        from psd.models.stgcn_bc import build_stgcn_bc
        from psd.training.train_stgcn_bc import TrainConfig
        pool, val = _tiny_pool_and_val()
        model_fn = lambda: build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2)
        cfg = TrainConfig(
            epochs=2, batch_size=16, use_amp=False, device="cpu",
            warmup_epochs=0, early_stopping=False, save_interval=1000,
            output_dir="runs/_tmp_al_test",
        )
        return ALSimulationRunner(
            build_model=model_fn, pool_samples=pool, val_samples=val,
            train_config=cfg, budgets=budgets,
        )

    def test_incremental_rounds_nested_selections(self):
        """轨迹产出各预算点指标；labeled 集合嵌套增长。"""
        r = self._make_runner()
        result = r.run_trajectory(strategy="entropy", seed=42)
        assert set(result.keys()) == {4, 8}
        assert all(isinstance(v, float) and math.isfinite(v) for v in result.values())
        sel4 = set(r.selected_at(4))
        sel8 = set(r.selected_at(8))
        assert sel4 <= sel8 and len(sel8) == 8

    def test_paired_initial_core_identical_across_strategies(self):
        """同 seed 两臂初始核逐 id 相等（配对设计核心保证）。"""
        r1 = self._make_runner()
        r2 = self._make_runner()
        core_a = r1._initial_core(seed=42)
        core_b = r2._initial_core(seed=42)
        assert core_a == core_b

    def test_small_batch_no_empty_loader(self):
        """n_train < batch_size 时训练不产生空 loader（drop_last 规避）。"""
        r = self._make_runner(budgets=(3,))
        acc = r._train_stage(sample_ids=list(range(3)), init_seed=7)
        assert isinstance(acc, float) and math.isfinite(acc)

    def test_unknown_strategy_raises(self):
        r = self._make_runner()
        with pytest.raises(ValueError):
            r.run_trajectory(strategy="mc_dropout", seed=1)

# ---------------------------------------------------------------------------
# Task 4: P0.4 真实池熵打分（best.pt 迁移代理，公开真实层口径）
# ---------------------------------------------------------------------------

def _fake_kp(n_frames: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.5, 0.1, size=(n_frames, 24, 3)).astype(np.float32)


class TestClipSegmentToStgcnInput:
    def test_shape_T30(self):
        from psd.training.active_learning import clip_segment_to_stgcn_input
        kp = _fake_kp(1041)
        seg = clip_segment_to_stgcn_input(kp, start=175, end=196, T=30)
        assert seg.shape == (30, 24, 3)
        assert np.isfinite(seg).all()

    def test_endpoints_preserved_after_resize(self):
        """重采样保端点：输出首末帧 = 切片段首末帧的逐帧归一化结果。"""
        from psd.training.active_learning import clip_segment_to_stgcn_input, _framewise_normalize
        kp = _fake_kp(100, seed=3)
        seg = clip_segment_to_stgcn_input(kp, start=10, end=40, T=15)
        expect_first = _framewise_normalize(kp[10:40])[None][0][0]  # 归一后首帧
        expect_last = _framewise_normalize(kp[10:40])[-1]
        assert np.allclose(seg[0], expect_first, atol=1e-5)
        assert np.allclose(seg[-1], expect_last, atol=1e-5)

    def test_nan_segment_raises(self):
        from psd.training.active_learning import clip_segment_to_stgcn_input
        kp = _fake_kp(50)
        kp[20:25] = np.nan
        with pytest.raises(ValueError):
            clip_segment_to_stgcn_input(kp, start=18, end=30, T=10)


class TestScoreRealPool:
    def _tiny_model(self):
        from psd.models.stgcn_bc import build_stgcn_bc
        return build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2)

    def test_output_schema_and_ranking(self):
        from psd.training.active_learning import score_real_pool
        pool_entries = [
            {"clip_id": f"interpet_dog01_p01_take02_ego_00{i}", "start_frame": 0, "end_frame": 30,
             "pseudo_label": "standing", "kappa_margin": 0.3}
            for i in range(1, 4)
        ]
        loader = lambda clip_id: {"kp_world": _fake_kp(60, seed=len(clip_id))}
        result = score_real_pool(pool_entries, loader, self._tiny_model(), budgets=[2, 5], device="cpu")
        assert result["n_scored"] == 3
        assert len(result["ranked_ids"]) == 3
        assert set(result["topk"].keys()) == {"2", "5"}
        assert len(result["topk"]["2"]) == 2
        # topk 应为 ranked 前缀
        assert result["topk"]["2"] == result["ranked_ids"][:2]
        for k in ("mean", "std", "min", "max", "q25", "q50", "q75"):
            assert k in result["entropy_stats"]

    def test_nan_entry_skipped_counted(self):
        from psd.training.active_learning import score_real_pool
        pool_entries = [
            {"clip_id": "clip_ok", "start_frame": 0, "end_frame": 20},
            {"clip_id": "clip_bad", "start_frame": 0, "end_frame": 20},
        ]
        def loader(clip_id):
            kp = _fake_kp(40)
            if clip_id == "clip_bad":
                kp[5:10] = np.nan
            return {"kp_world": kp}
        result = score_real_pool(pool_entries, loader, self._tiny_model(), budgets=[1], device="cpu")
        assert result["n_scored"] == 1
        assert result["n_skipped"] == 1

    def test_production_loader_resolves_npz_path(self, tmp_path):
        """生产 loader 按 {clip_id}.npz 约定解析路径。"""
        from psd.training.active_learning import make_clip_loader
        kp = _fake_kp(35)
        np.savez(tmp_path / "interpet_dog09_p19_take03_ego_001.npz",
                 kp_world=kp, kp_weight=np.ones((35, 24), dtype=np.float32),
                 frame_idx=np.arange(35, dtype=np.int32))
        loader = make_clip_loader(str(tmp_path))
        d = loader("interpet_dog09_p19_take03_ego_001")
        assert d["kp_world"].shape == (35, 24, 3)

class TestSaturationDiagnostics:
    def test_score_real_pool_reports_margins_and_degeneracy(self):
        """输出含 logit 边际统计与熵退化标志（负结果显式登记）。"""
        from psd.models.stgcn_bc import build_stgcn_bc
        from psd.training.active_learning import score_real_pool
        entries = [{"clip_id": f"c{i}", "start_frame": 0, "end_frame": 20} for i in range(2)]
        loader = lambda cid: {"kp_world": _fake_kp(40, seed=len(cid))}
        result = score_real_pool(entries, loader,
                                 build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2),
                                 budgets=[1], device="cpu")
        ms = result["logit_margin_stats"]
        for k in ("mean", "min", "max"):
            assert isinstance(ms[k], float)
        assert isinstance(result["entropy_degenerate"], bool)
        # 一致性: 退化标志 ⇔ 池最大熵 < 1e-3
        assert result["entropy_degenerate"] == (result["entropy_stats"]["max"] < 1e-3)


# ---------------------------------------------------------------------------
# W23 窗口: warm-start 协议（预注册 docs/superpowers/plans/2026-08-25-w23-warmstart-al.md）
# 三断言: ① warm 初始化在每个预算点生效 ② 偏移数据管线确定且实质偏移 ③ 打分器域内微调非原始 ckpt
# ---------------------------------------------------------------------------

def _tiny_warm_runner(init_ckpt=None, budgets=(4, 8)):
    """warm-start 用 tiny 运行器工厂（模型架构与 W14 tiny 测试一致）。"""
    from psd.models.stgcn_bc import build_stgcn_bc
    from psd.training.active_learning import ALSimulationRunner
    from psd.training.train_stgcn_bc import TrainConfig
    pool, val = _tiny_pool_and_val()
    model_fn = lambda: build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2)
    cfg = TrainConfig(
        epochs=2, batch_size=16, use_amp=False, device="cpu",
        warmup_epochs=0, early_stopping=False, save_interval=1000,
        output_dir="runs/_tmp_al_test",
    )
    return ALSimulationRunner(
        build_model=model_fn, pool_samples=pool, val_samples=val,
        train_config=cfg, budgets=budgets, init_from_ckpt=init_ckpt,
    )


class TestWarmStartProtocolW23:
    def test_warm_init_loaded_at_every_stage(self, monkeypatch):
        """每个预算点的模型在 trainer.fit() 前权重 == init_from_ckpt（逐张量相等）。"""
        import torch
        import psd.training.active_learning as al_mod
        from psd.models.stgcn_bc import build_stgcn_bc

        torch.manual_seed(777)
        warm_sd = {k: v.detach().clone() for k, v in
                   build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2).state_dict().items()}

        captured = []

        from psd.training.train_stgcn_bc import STGCNBCTrainer as RealTrainer

        class SpyTrainer(RealTrainer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured.append({k: v.detach().clone() for k, v in self.model.state_dict().items()})

        monkeypatch.setattr(al_mod, "STGCNBCTrainer", SpyTrainer)

        r = _tiny_warm_runner(init_ckpt=warm_sd)
        r.run_trajectory(strategy="random", seed=42)

        assert len(captured) == len(r.budgets), "每个预算点都应经过一次 trainer 构建"
        for i, snap in enumerate(captured):
            assert set(snap.keys()) == set(warm_sd.keys())
            for k in warm_sd:
                assert torch.equal(snap[k], warm_sd[k]), \
                    f"阶段 {i} 张量 {k} 未从 init_from_ckpt 起步（warm 初始化失效）"

    def test_cold_path_unaffected_without_init(self, monkeypatch):
        """不传 init_from_ckpt 时保持冷启动行为（回归保护）。"""
        import torch
        import psd.training.active_learning as al_mod
        from psd.training.train_stgcn_bc import STGCNBCTrainer as RealTrainer

        captured = []

        class SpyTrainer(RealTrainer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                captured.append({k: v.detach().clone() for k, v in self.model.state_dict().items()})

        monkeypatch.setattr(al_mod, "STGCNBCTrainer", SpyTrainer)
        # 复刻 _fit_stage 的置种顺序: manual_seed(seed*1000+budgets[0]) 后再 build_model
        from psd.models.stgcn_bc import build_stgcn_bc
        torch.manual_seed(42 * 1000 + 4)  # seed=42, 首预算点 b=4
        m = build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2)
        expected_first = {k: v.detach().clone() for k, v in m.state_dict().items()}
        r = _tiny_warm_runner(init_ckpt=None)
        r.run_trajectory(strategy="random", seed=42)
        # 冷启动: 第一阶段权重应等于该种子序列下的随机初始化
        for k in expected_first:
            assert torch.equal(captured[0][k], expected_first[k])

    def test_scorer_is_in_domain_finetuned_not_raw_ckpt(self, monkeypatch):
        """熵打分所用模型权重必须 ≠ 原始 init_from_ckpt（禁止原始 best.pt 直接跨域打分）。"""
        import torch
        import numpy as np
        import psd.training.active_learning as al_mod
        from psd.models.stgcn_bc import build_stgcn_bc

        torch.manual_seed(888)
        warm_sd = {k: v.detach().clone() for k, v in
                   build_stgcn_bc(in_channels=3, num_classes=22, base_channels=8, num_stages=2).state_dict().items()}

        scorer_states = []
        real_predict_probs = al_mod.predict_probs

        def spy_predict_probs(model, samples, **kw):
            scorer_states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return real_predict_probs(model, samples, **kw)

        monkeypatch.setattr(al_mod, "predict_probs", spy_predict_probs)

        r = _tiny_warm_runner(init_ckpt=warm_sd)
        r.run_trajectory(strategy="entropy", seed=42)

        assert len(scorer_states) >= 1, "增量阶段至少发生一次池打分"
        for i, snap in enumerate(scorer_states):
            max_diff = max(float((snap[k] - warm_sd[k]).abs().max()) for k in warm_sd)
            assert max_diff > 0.0, \
                f"第 {i} 次打分的模型权重与原始 ckpt 完全相同——违反 scorer 域内性规则"

    def test_offset_dataset_pipeline_deterministic_and_shifted(self):
        """偏移数据管线: 同种子确定性复现；noise 0.10 与 0.05 数据实质不同；类别均衡。"""
        from collections import Counter
        from psd.data.synth_stgcn import make_synthetic_dataset, ALL_BEHAVIORS_22

        pool_a = make_synthetic_dataset(samples_per_class=2, T=10, noise_std=0.10, seed=20263)
        pool_b = make_synthetic_dataset(samples_per_class=2, T=10, noise_std=0.10, seed=20263)
        val_set = make_synthetic_dataset(samples_per_class=1, T=10, noise_std=0.10, seed=20264)
        pool_ref05 = make_synthetic_dataset(samples_per_class=2, T=10, noise_std=0.05, seed=20263)

        # 形状与有限性
        assert pool_a[0]["keypoints"].shape == (10, 24, 3)
        assert all(np.isfinite(s["keypoints"]).all() for s in pool_a + val_set)

        # 确定性: 同种子两次生成逐元素相等
        for sa, sb in zip(pool_a, pool_b):
            assert np.array_equal(sa["keypoints"], sb["keypoints"])

        # 实质偏移: 同种子不同噪声档数据不同
        diff_any = any(not np.array_equal(sa["keypoints"], sr["keypoints"])
                       for sa, sr in zip(pool_a, pool_ref05))
        assert diff_any, "noise_std=0.10 与 0.05 生成结果相同——偏移未生效"

        # 类别均衡: 22 类各 spc 条
        cnt = Counter(s["label"] for s in pool_a)
        assert sorted(cnt.keys()) == list(range(len(ALL_BEHAVIORS_22)))
        assert all(v == 2 for v in cnt.values())

        # 池/验证实例隔离: 不同 seed 数据不同
        assert not np.array_equal(pool_a[0]["keypoints"], val_set[0]["keypoints"])
