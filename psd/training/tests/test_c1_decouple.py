"""C1 解耦切换成本实验测试（W19 窗口，TDD 先行）.

覆盖任务书 Step 1 必测点:
  1. _Y_TO_YP_MAP 正确性（22 类全覆盖，stand/track→同一 locomotion idx，其余一一对应）
  2. backbone 冻结断言（重训后 backbone 参数不变/无梯度）
  3. 成本记录字段完整性（标注单元数、墙钟秒数、epoch 数、best_val_acc）

测试策略: 全封闭——用 tiny 架构(base_channels=8, num_stages=2)在 tmp 目录自造
checkpoint，不依赖 runs/p05_stgcn_bc_full/best.pt 等外部产物。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.synth_stgcn import make_synthetic_dataset
from psd.models.stgcn_bc_constants import (
    ALL_BEHAVIORS_22,
    Y_PRIME_LABEL_NAMES,
)
from psd.models.stgcn_bc import build_stgcn_bc

# 被测模块: scripts/run_c1_decouple.py 以可导入函数形式提供逻辑
from scripts.run_c1_decouple import (
    Y_NUM_CLASSES,
    Y_PRIME_NUM_CLASSES,
    build_y_to_yp_map,
    map_samples_to_yprime,
    load_y_backbone,
    freeze_backbone,
    run_arm,
    aggregate_runs,
)

TINY_KW = dict(base_channels=8, num_stages=2)


# ---------------------------------------------------------------------------
# 1. Y→Y′ 映射正确性
# ---------------------------------------------------------------------------

class TestYToYPMap:
    def test_covers_all_22_classes(self):
        m = build_y_to_yp_map()
        assert set(m.keys()) == set(ALL_BEHAVIORS_22)
        assert len(m) == Y_NUM_CLASSES == 22

    def test_stand_track_merge_into_same_locomotion_idx(self):
        m = build_y_to_yp_map()
        loco_idx = list(Y_PRIME_LABEL_NAMES).index("locomotion")
        assert loco_idx == 2
        assert m["stand"] == loco_idx
        assert m["track"] == loco_idx

    def test_identity_for_unmerged_classes(self):
        """未合并的类必须一一对应到 Y′ 中同名类的下标."""
        m = build_y_to_yp_map()
        for name in ALL_BEHAVIORS_22:
            if name in ("stand", "track"):
                continue
            assert m[name] == list(Y_PRIME_LABEL_NAMES).index(name), name

    def test_all_targets_in_range_21(self):
        m = build_y_to_yp_map()
        assert len(Y_PRIME_LABEL_NAMES) == Y_PRIME_NUM_CLASSES == 21
        assert all(0 <= v < 21 for v in m.values())


class TestMapSamplesToYPrime:
    def _samples(self):
        return make_synthetic_dataset(samples_per_class=2, T=30, seed=42)

    def test_relabels_and_updates_label_name(self):
        samples = self._samples()
        mapped = map_samples_to_yprime(copy.deepcopy(samples))
        ref = build_y_to_yp_map()
        for orig, new in zip(samples, mapped):
            expect_label = ref[orig["label_name"]]
            assert new["label"] == expect_label
            assert new["label_name"] == Y_PRIME_LABEL_NAMES[expect_label]

    def test_preserves_keypoints_and_boundary(self):
        samples = self._samples()
        mapped = map_samples_to_yprime(samples)
        for orig, new in zip(samples, mapped):
            assert np.array_equal(np.asarray(orig["keypoints"]),
                                  np.asarray(new["keypoints"]))
            assert np.array_equal(np.asarray(orig["boundary"]),
                                  np.asarray(new["boundary"]))

    def test_labels_within_21_classes(self):
        mapped = map_samples_to_yprime(self._samples())
        labels = {s["label"] for s in mapped}
        assert min(labels) >= 0 and max(labels) <= 20

    def test_unknown_label_raises(self):
        bad = [{"label": 0, "label_name": "不存在的类",
                "keypoints": np.zeros((30, 24, 3), dtype=np.float32),
                "boundary": np.zeros(30, dtype=np.float32)}]
        with pytest.raises(ValueError):
            map_samples_to_yprime(bad)


# ---------------------------------------------------------------------------
# 2. backbone 加载与冻结断言
# ---------------------------------------------------------------------------

def _make_tiny_checkpoint(tmp_path: Path) -> Path:
    """构造一个 Y(22类) tiny checkpoint（模拟 runs/p05_stgcn_bc_full/best.pt 结构）."""
    model = build_stgcn_bc(num_classes=Y_NUM_CLASSES, **TINY_KW)
    ckpt = {
        "epoch": 38,
        "model_state_dict": model.state_dict(),
        "val_acc": 0.9659,
        "best_val_acc": 0.9659,
    }
    path = tmp_path / "y_tiny_best.pt"
    torch.save(ckpt, path)
    return path


class TestBackboneLoadAndFreeze:
    def test_load_y_backbone_strips_head(self, tmp_path):
        ckpt_path = _make_tiny_checkpoint(tmp_path)
        model = build_stgcn_bc(num_classes=Y_PRIME_NUM_CLASSES, **TINY_KW)
        info = load_y_backbone(model, ckpt_path)
        # head 键全部缺失（随机初始化），backbone 全部加载成功，无多余键
        assert len(info.missing) > 0
        assert all(k.startswith("head.") for k in info.missing)
        assert info.unexpected == []
        assert info.loaded_backbone_tensors > 0

    def test_loaded_backbone_weights_match_checkpoint(self, tmp_path):
        ckpt_path = _make_tiny_checkpoint(tmp_path)
        src_sd = torch.load(ckpt_path, map_location="cpu",
                            weights_only=False)["model_state_dict"]
        model = build_stgcn_bc(num_classes=Y_PRIME_NUM_CLASSES, **TINY_KW)
        load_y_backbone(model, ckpt_path)
        model_sd = model.state_dict()
        for k, v in src_sd.items():
            if k.startswith("backbone."):
                assert torch.equal(model_sd[k], v), k

    def test_head_remains_random_after_load(self, tmp_path):
        ckpt_path = _make_tiny_checkpoint(tmp_path)
        src_sd = torch.load(ckpt_path, map_location="cpu",
                            weights_only=False)["model_state_dict"]
        torch.manual_seed(1234)
        model = build_stgcn_bc(num_classes=Y_PRIME_NUM_CLASSES, **TINY_KW)
        load_y_backbone(model, ckpt_path)
        model_sd = model.state_dict()
        for k, v in src_sd.items():
            if k.startswith("head."):
                # 21 类 vs 22 类形状必然不同；即便同形也不应等于源值
                if model_sd[k].shape == v.shape:
                    assert not torch.equal(model_sd[k], v), k


class TestFreezeBackbone:
    def test_requires_grad_flags(self):
        model = build_stgcn_bc(num_classes=Y_PRIME_NUM_CLASSES, **TINY_KW)
        freeze_backbone(model)
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                assert not p.requires_grad, name
            else:
                assert p.requires_grad, name

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_training_step_leaves_backbone_intact(self):
        """一步优化后: backbone 参数数值不变、无梯度累积；head 参数被更新."""
        torch.manual_seed(7)
        model = build_stgcn_bc(num_classes=Y_PRIME_NUM_CLASSES, **TINY_KW)
        freeze_backbone(model)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-2
        )
        model.train()  # 模拟 trainer.fit() 每轮的 train 模式切换（触发 BN 冻结补丁）
        x = torch.randn(2, 30, 24, 3)
        cls_logits, bnd_logits = model(x)
        loss_dict = model.compute_loss(
            cls_logits, bnd_logits,
            torch.tensor([0, 1]),
            torch.rand(2, 30),
        )
        loss_dict["total"].backward()
        opt.step()

        after = model.state_dict()
        for k in before:
            if k.startswith("backbone."):
                assert torch.equal(before[k], after[k]), f"backbone 被改动: {k}"
        head_cls_w = "head.fc_cls.weight"
        assert not torch.equal(before[head_cls_w], after[head_cls_w])
        # backbone 参数不应有梯度
        for name, p in model.named_parameters():
            if name.startswith("backbone."):
                assert p.grad is None, f"backbone 出现梯度: {name}"


# ---------------------------------------------------------------------------
# 3. 成本记录字段完整性 + 两臂运行
# ---------------------------------------------------------------------------

REQUIRED_COST_FIELDS = [
    "arm", "seed", "n_per_class", "taxonomy",
    "labeled_units_train",   # 标注单元数口径 = 训练样本量
    "val_size", "total_labeled_units",
    "epochs_configured", "epochs_run", "best_epoch",
    "wall_clock_sec", "best_val_acc", "final_val_acc",
    "device", "gpu_state",
    "frozen_params", "trainable_params",
]


@pytest.fixture()
def tiny_decouple_record(tmp_path):
    """模块级 fixture：tiny 解耦臂完整跑一轮（函数作用域，避免类作用域弃用告警）."""
    outdir = tmp_path / "c1_smoke"
    return run_arm(
        arm="decouple", seed=42, n_per_class=2, epochs=1, patience=1,
        device="cpu", T=30, batch_size=8, warmup_epochs=0,
        base_channels=8, num_stages=2, output_dir=str(outdir / "dec"),
        checkpoint_path=None,  # 无 ckpt 时退化为随机 backbone（管线仍需完整跑通）
    )


class TestRunArmCostRecord:
    def test_required_fields_present(self, tiny_decouple_record):
        for field in REQUIRED_COST_FIELDS:
            assert field in tiny_decouple_record, field

    def test_field_types_and_values(self, tiny_decouple_record):
        r = tiny_decouple_record
        assert r["arm"] == "decouple"
        assert r["seed"] == 42
        assert r["n_per_class"] == 2
        assert r["taxonomy"] == "Y_prime"
        # 标注单元数 = 22 类 × 2 样本 × 0.8 ≈ 35（8:2 切分）
        total = 22 * 2
        assert r["labeled_units_train"] == total - int(total * 0.2)
        assert r["val_size"] == int(total * 0.2)
        assert r["total_labeled_units"] == total
        assert r["epochs_run"] == 1
        assert r["wall_clock_sec"] > 0
        assert 0.0 <= r["best_val_acc"] <= 1.0
        assert r["device"].startswith("cpu")

    def test_baseline_arm_runs_without_checkpoint(self, tmp_path):
        rec = run_arm(
            arm="baseline", seed=42, n_per_class=2, epochs=1, patience=1,
            device="cpu", T=30, batch_size=8, warmup_epochs=0,
            base_channels=8, num_stages=2,
            output_dir=str(tmp_path / "base"), checkpoint_path=None,
        )
        assert rec["arm"] == "baseline"
        assert rec["frozen_params"] == 0
        assert rec["trainable_params"] > 0

    def test_invalid_arm_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            run_arm(arm="quantum", seed=42, n_per_class=2, epochs=1,
                    device="cpu", output_dir=str(tmp_path / "x"))


class TestAggregateRuns:
    def test_mean_std_grouping_by_arm(self):
        runs = [
            {"arm": "decouple", "seed": 42, "wall_clock_sec": 10.0,
             "best_val_acc": 0.80, "epochs_run": 5, "labeled_units_train": 100},
            {"arm": "decouple", "seed": 43, "wall_clock_sec": 14.0,
             "best_val_acc": 0.90, "epochs_run": 7, "labeled_units_train": 100},
            {"arm": "baseline", "seed": 42, "wall_clock_sec": 30.0,
             "best_val_acc": 0.70, "epochs_run": 9, "labeled_units_train": 100},
            {"arm": "baseline", "seed": 43, "wall_clock_sec": 40.0,
             "best_val_acc": 0.70, "epochs_run": 11, "labeled_units_train": 100},
        ]
        agg = aggregate_runs(runs)
        # 双臂各自聚合；两臂齐全时附 _comparison
        assert {"decouple", "baseline"}.issubset(agg.keys())
        assert "_comparison" in agg
        d, b = agg["decouple"], agg["baseline"]
        assert d["wall_clock_sec"]["mean"] == pytest.approx(12.0)
        assert d["wall_clock_sec"]["std"] == pytest.approx(2.0)
        assert d["best_val_acc"]["mean"] == pytest.approx(0.85)
        assert b["epochs_run"]["mean"] == pytest.approx(10.0)
        assert d["labeled_units_train"]["mean"] == pytest.approx(100.0)

    def test_speedup_ratio_computed(self):
        runs = [
            {"arm": "decouple", "seed": 42, "wall_clock_sec": 10.0,
             "best_val_acc": 0.9, "epochs_run": 5, "labeled_units_train": 50},
            {"arm": "baseline", "seed": 42, "wall_clock_sec": 40.0,
             "best_val_acc": 0.9, "epochs_run": 5, "labeled_units_train": 50},
        ]
        agg = aggregate_runs(runs)
        assert agg["_comparison"]["wall_clock_ratio_baseline_over_decouple"] \
            == pytest.approx(4.0)
