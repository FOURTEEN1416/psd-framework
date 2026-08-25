"""Q3c 公开真实层微调脚本测试——数据装配/初始化/冻结三防线（CPU tiny）."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from psd.models.stgcn_bc import STGCNBC  # noqa: E402
from run_c1_decouple import freeze_backbone, load_y_backbone  # noqa: E402
from run_p05_public_real_finetune import load_dataset, per_class_val_acc  # noqa: E402


def _fake_pkl(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    data = []
    for split, n in (("train", 6), ("val", 4)):
        for i in range(n):
            data.append({
                "keypoints": rng.normal(0, 0.1, (30, 24, 3)).astype(np.float32),
                "label": i % 4,
                "boundary": np.zeros(30, dtype=np.float32),
                "video_id": f"v_{split}_{i}", "split": split,
                "psd_class": ["stay", "track", "watch", "jump"][i % 4],
            })
    p = tmp_path / "tiny.pkl"
    with open(p, "wb") as f:
        pickle.dump(data, f)
    return p


def test_load_dataset_split_and_dist(tmp_path):
    train, val, dist = load_dataset(_fake_pkl(tmp_path))
    assert len(train) == 6 and len(val) == 4
    assert set(dist) == {"stay", "track", "watch", "jump"}


def test_load_dataset_rejects_missing_val(tmp_path):
    p = tmp_path / "bad.pkl"
    with open(p, "wb") as f:
        pickle.dump([{"split": "train", "psd_class": "stay"}], f)
    with pytest.raises(ValueError):
        load_dataset(p)


def test_init_from_y_ckpt_head_fresh_and_frozen(tmp_path):
    """22 类 ckpt → backbone 载入 + 4 类 head 全新；冻结后一步优化 backbone 不动."""
    torch.manual_seed(0)
    src = STGCNBC(in_channels=3, num_classes=22)
    ck = tmp_path / "y.pt"
    torch.save({"model_state_dict": src.state_dict(), "epoch": 1}, ck)

    model = STGCNBC(in_channels=3, num_classes=4)
    info = load_y_backbone(model, ck)   # 完整模型：backbone.* 载入，head.* 被剥离保持随机
    assert len(info.unexpected) == 0
    assert len(info.missing) > 0        # 4 类 head 键缺失 = 全新待训
    assert all(k.startswith("head.") for k in info.missing)
    # head 必须是全新随机（与 src 不同）
    head_differs = any(
        not torch.equal(p_src, p_new)
        for p_src, p_new in zip(src.head.parameters(), model.head.parameters())
    )
    assert head_differs

    freeze_backbone(model)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    before = {n: p.detach().clone() for n, p in model.backbone.named_parameters()}
    x = torch.randn(2, 30, 24, 3)
    cls_logits, boundary = model(x)
    loss = cls_logits.sum() + boundary.sum()
    opt.zero_grad(); loss.backward(); opt.step()
    for n, p in model.backbone.named_parameters():
        assert torch.equal(before[n], p), f"backbone 参数被改写: {n}"


def test_per_class_val_acc_cpu():
    torch.manual_seed(1)
    model = STGCNBC(in_channels=3, num_classes=4)
    val = [{"keypoints": np.zeros((30, 24, 3), dtype=np.float32),
            "label": 2, "psd_class": "watch"}]
    acc = per_class_val_acc(model, val, "cpu")
    assert set(acc) == {"watch"} and 0.0 <= acc["watch"] <= 1.0
