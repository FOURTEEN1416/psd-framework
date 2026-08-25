"""W31 消融: 两臂初始化契约测试 — TDD 先行.

被测对象: psd/models/aimclr_finetune.py（本测试先于实现编写，RED→GREEN）

契约（任务书 NEXT-BATCH-plan.md W31 节）:
    两臂 {随机初始化, 加载 P0.1 AimCLR 预训练 backbone} 的初始化必须可辨异:
    - warm 臂 encoder 权重 == ckpt 张量（逐张量位相等, 承袭 W23 warm-start 断言惯例）
    - scratch 臂 encoder 权重 != ckpt（随机初始化未消费预训练）
    - 同 seed 下两臂分类/边界头逐位相等（公平性: 唯一差异 = encoder 初始权重）
"""
from __future__ import annotations

import pytest
import torch

from psd.models.aimclr_finetune import (
    AimCLRFinetune,
    build_arm_model,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _fake_pretrained_ckpt(tmp_path):
    """构造 P0.1 格式的假预训练 ckpt（裸 state_dict, encoder_q. 前缀）.

    架构参数与 P0.1 configs/p01_aimclr.yaml 一致，仅权重为随机初始化产物。
    """
    torch.manual_seed(123)
    model = AimCLRFinetune(num_classes=22)
    sd = model.encoder.state_dict()
    prefixed = {f"encoder_q.{k}": v for k, v in sd.items()}
    ckpt_path = tmp_path / "epoch120_model.pt"
    torch.save(prefixed, ckpt_path)
    return ckpt_path


@pytest.fixture(scope="module")
def fake_ckpt(tmp_path_factory):
    return _fake_pretrained_ckpt(tmp_path_factory.mktemp("w31_ckpt"))


# ---------------------------------------------------------------------------
# 两臂初始化断言
# ---------------------------------------------------------------------------

class TestArmInitContract:
    def test_warm_encoder_equals_checkpoint_bitwise(self, fake_ckpt):
        """warm 臂: encoder 每个张量与 ckpt 逐位相等（strict 加载成功即结构同构）。"""
        saved = torch.load(fake_ckpt, map_location="cpu", weights_only=False)
        stripped = {
            k[len("encoder_q."):]: v for k, v in saved.items()
            if k.startswith("encoder_q.")
        }
        assert len(stripped) > 0, "fixture 自身损坏: 无 encoder_q 键"

        model = build_arm_model("warm", seed=7, pretrained_ckpt=fake_ckpt)
        actual = model.encoder.state_dict()
        assert set(actual.keys()) == set(stripped.keys()), "encoder 键集合与 ckpt 不一致"
        for k, v in stripped.items():
            assert torch.equal(actual[k], v), f"warm 臂张量 {k} 未从 ckpt 起步"

    def test_scratch_encoder_differs_from_checkpoint(self, fake_ckpt):
        """scratch 臂: 不消费 ckpt，encoder 权重与预训练可辨异。"""
        saved = torch.load(fake_ckpt, map_location="cpu", weights_only=False)
        stripped = {
            k[len("encoder_q."):]: v for k, v in saved.items()
            if k.startswith("encoder_q.")
        }
        scratch = build_arm_model("scratch", seed=7)
        diff = [
            k for k, v in stripped.items()
            if not torch.allclose(scratch.encoder.state_dict()[k], v)
        ]
        assert len(diff) > 0, "scratch 臂意外与 ckpt 全等（随机性失效）"

    def test_same_seed_heads_identical_across_arms(self, fake_ckpt):
        """同 seed 公平性: 两臂 cls/bnd 头逐位相等——唯一差异变量是 encoder 初始权重。"""
        warm = build_arm_model("warm", seed=7, pretrained_ckpt=fake_ckpt)
        scratch = build_arm_model("scratch", seed=7)
        for head_name in ("cls_head", "bnd_head"):
            w_params = dict(getattr(warm, head_name).named_parameters())
            s_params = dict(getattr(scratch, head_name).named_parameters())
            assert set(w_params) == set(s_params), f"{head_name} 参数名不一致"
            for k in w_params:
                assert torch.equal(w_params[k], s_params[k]), \
                    f"同 seed 下 {head_name}.{k} 两臂不等——头初始化未受控"

    def test_invalid_arm_raises(self, fake_ckpt):
        with pytest.raises(ValueError):
            build_arm_model("no_such_arm", seed=0)

    def test_missing_ckpt_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            build_arm_model("warm", seed=0,
                            pretrained_ckpt=tmp_path / "not_exists.pt")


# ---------------------------------------------------------------------------
# 前向契约（STGCNBCTrainer 兼容接口）
# ---------------------------------------------------------------------------

class TestForwardContract:
    def test_forward_shapes_match_trainer_contract(self):
        """forward((B,T,24,3)) → (cls(B,22), bnd(B,T')); T'=8 由 stride[5,8] 推得。"""
        model = AimCLRFinetune(num_classes=22).eval()
        x = torch.randn(2, 30, 24, 3)
        with torch.no_grad():
            cls_logits, bnd_logits = model(x)
        assert cls_logits.shape == (2, 22)
        assert bnd_logits.shape == (2, 8)

    def test_compute_loss_finite(self):
        """compute_loss 契约: 与 STGCNBC 相同的三键字典且有限。"""
        model = AimCLRFinetune(num_classes=22)
        x = torch.randn(2, 30, 24, 3)
        labels = torch.tensor([0, 21])
        boundaries = torch.rand(2, 30)
        cls_logits, bnd_logits = model(x)
        loss = model.compute_loss(cls_logits, bnd_logits, labels, boundaries)
        assert {"total", "cls", "boundary"} <= set(loss.keys())
        for name in ("total", "cls", "boundary"):
            assert torch.isfinite(loss[name]), f"loss[{name}] 非有限值"

    def test_joint24_input_padded_to_ntu25(self):
        """槽 24 零填充契约: 输入 24 关节，encoder 内部按 NTU 25 点消费。"""
        model = AimCLRFinetune(num_classes=22).eval()
        x = torch.randn(1, 30, 24, 3)
        # 不抛形状错误即证明内部补齐; 同时验证输出确定性（eval 模式两次前向一致）
        with torch.no_grad():
            c1, b1 = model(x)
            c2, b2 = model(x)
        assert torch.equal(c1, c2) and torch.equal(b1, b2)
