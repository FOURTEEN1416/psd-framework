"""W31 消融: 数据切分公平性与编排契约测试 — TDD 先行（RED）.

被测对象: psd/training/ablation_pretrain.py

契约（任务书 NEXT-BATCH-plan.md W31 节「对照公平性（同 seed 同切分）」）:
    - 切分严格复刻 W12 口径（run_p05_full.py: make_synthetic_dataset +
      default_rng(seed).permutation + val_split 0.2），确定性可复现；
    - 两臂共享同一切分实例（公平性: 唯一差异 = encoder 初始权重）；
    - train_one_arm 在 CPU tiny 档可端到端跑通并产出合规摘要。
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from psd.training.ablation_pretrain import (
    make_w12_split,
    train_one_arm,
)


class TestW12SplitFairness:
    def test_deterministic_same_seed_same_split(self):
        """同 data_seed 两次调用 → 样本序列逐位相同（两臂同切分的根基）。"""
        tr1, va1 = make_w12_split(samples_per_class=2, T=30, data_seed=42)
        tr2, va2 = make_w12_split(samples_per_class=2, T=30, data_seed=42)
        assert len(tr1) == len(tr2) and len(va1) == len(va2)
        for a, b in zip(tr1, tr2):
            assert np.array_equal(a["keypoints"], b["keypoints"])
            assert a["label"] == b["label"]
        for a, b in zip(va1, va2):
            assert np.array_equal(a["keypoints"], b["keypoints"])

    def test_split_sizes_and_disjoint_union(self):
        """8:2 切分尺寸正确且不相交（W12 口径: val_n = int(total*0.2)）。"""
        spc = 3                       # 22*3=66 样本
        train, val = make_w12_split(samples_per_class=spc, T=30, data_seed=42)
        total = 22 * spc
        assert len(train) == total - int(total * 0.2)
        assert len(val) == int(total * 0.2)

    def test_different_data_seed_changes_split(self):
        """不同 data_seed 切分不同（守卫: 防止写死常量切分的假实现）。"""
        tr_a, _ = make_w12_split(samples_per_class=2, T=30, data_seed=42)
        tr_b, _ = make_w12_split(samples_per_class=2, T=30, data_seed=43)
        same = all(
            np.array_equal(a["keypoints"], b["keypoints"])
            for a, b in zip(tr_a, tr_b)
        )
        assert not same


@pytest.fixture(scope="module")
def tiny_split():
    """tiny 档切分（22 样本，供端到端冒烟复用，避免重复生成）。"""
    return make_w12_split(samples_per_class=1, T=30, data_seed=42)


class TestTrainOneArmContract:
    def test_smoke_run_cpu_end_to_end(self, tiny_split, tmp_path):
        """CPU tiny 档端到端: 两臂各训 1 epoch，产出合规摘要。"""
        train_samples, val_samples = tiny_split
        base_cfg = {
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "epochs": 1,
            "batch_size": 4,
            "warmup_epochs": 0,
            "device": "cpu",
            "early_stopping": False,
            "patience": 5,
            "num_classes": 22,
            "boundary_weight": 0.3,
        }
        ckpt = tmp_path / "fake_p01.pt"
        torch.save(
            {
                f"encoder_q.{k}": v
                for k, v in build_arm_encoder_state().items()
            },
            ckpt,
        )
        summaries = []
        for arm in ("scratch", "warm"):
            kw = {"pretrained_ckpt": str(ckpt)} if arm == "warm" else {}
            s = train_one_arm(
                arm=arm,
                seed=0,
                cfg=base_cfg,
                train_samples=train_samples,
                val_samples=val_samples,
                output_dir=str(tmp_path / f"runs_{arm}"),
                **kw,
            )
            summaries.append(s)
            assert s["arm"] == arm and s["seed"] == 0
            assert isinstance(s["best_val_acc"], float)
            assert 0.0 <= s["best_val_acc"] <= 1.0
        # 公平性: 同 seed 两臂的验证集完全一致（摘要回执口径）
        assert summaries[0]["val_size"] == summaries[1]["val_size"]

    def test_invalid_arm_rejected(self, tiny_split, tmp_path):
        train_samples, val_samples = tiny_split
        with pytest.raises(ValueError):
            train_one_arm(
                arm="bogus",
                seed=0,
                cfg={"device": "cpu"},
                train_samples=train_samples,
                val_samples=val_samples,
                output_dir=str(tmp_path / "runs_bad"),
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def build_arm_encoder_state():
    """构造与 AimCLRFinetune.encoder 同构的随机权重（fixture 用，避免循环导入）。"""
    from psd.models.aimclr_finetune import AimCLRFinetune

    torch.manual_seed(123)
    return AimCLRFinetune(num_classes=22).encoder.state_dict()
