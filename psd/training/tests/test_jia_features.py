"""P0.3 段特征池化测试（W8 交接 Step 2）。

Φ 冻结骨干的注入式封装：本模块测试用纯 numpy 桩编码器，
不依赖 torch / external —— GPU 前向由 scripts 层装配真实编码器。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.training.jia_features import build_segment_view, extract_segment_embeddings


def _fake_clip(n_frames=100, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "kp_world": rng.normal(0, 0.1, (n_frames, 24, 3)).astype(np.float32),
        "kp_weight": rng.uniform(0.6, 1.0, (n_frames, 24)).astype(np.float32),
        "frame_idx": np.arange(n_frames, dtype=np.int32),
    }


def _mk_seg(i, clip="c0", start=10, end=41):
    return {"clip_id": clip, "start_frame": start, "end_frame": end,
            "label": "sitting", "confidence": 0.9, "rule_ids": []}


class TestBuildSegmentView:
    def test_shape_ntu_layout(self):
        clip = _fake_clip()
        kp_seg = clip["kp_world"][10:42]
        w_seg = clip["kp_weight"][10:42]
        v = build_segment_view(kp_seg, w_seg, target_t=64)
        assert v.shape == (3, 64, 25, 1)
        assert v.dtype == np.float32
        assert np.isfinite(v).all()
        assert (v[:, :, 24, :] == 0).all()  # NTU 死关节槽恒零

    def test_single_frame_segment(self):
        clip = _fake_clip()
        v = build_segment_view(clip["kp_world"][5:6], clip["kp_weight"][5:6], target_t=16)
        assert v.shape == (3, 16, 25, 1)


class TestExtractSegmentEmbeddings:
    def _stub_encoder(self, calls):
        def enc(x):
            calls.append(x.shape[0])
            return x.mean(axis=3).squeeze(-1).mean(axis=2) * 100.0  # (B, 3, V=25 -> mean, C? )
        return enc

    def test_order_preserved_and_batched(self):
        clips = {"c0": _fake_clip(100, 0), "c1": _fake_clip(80, 1)}
        segs = [_mk_seg(0, "c0", 10, 41), _mk_seg(1, "c1", 0, 31), _mk_seg(2, "c0", 50, 65),
                _mk_seg(3, "c0", 66, 97), _mk_seg(4, "c1", 40, 79)]
        calls: list[int] = []
        emb = extract_segment_embeddings(segs, clips.__getitem__, self._stub_encoder(calls),
                                         batch_size=2)
        assert emb.shape[0] == 5
        assert sum(calls) == 5 and max(calls) <= 2  # 分批且全覆盖

    def test_deterministic(self):
        clips = {"c0": _fake_clip(60, 7)}
        segs = [_mk_seg(0, "c0", 0, 31), _mk_seg(1, "c0", 20, 51)]
        e1 = extract_segment_embeddings(segs, clips.__getitem__, lambda x: x.sum(axis=(2, 3)))
        e2 = extract_segment_embeddings(segs, clips.__getitem__, lambda x: x.sum(axis=(2, 3)))
        assert np.allclose(e1, e2)

    def test_different_segments_give_different_features(self):
        clips = {"c0": _fake_clip(120, 3)}
        segs = [_mk_seg(0, "c0", 0, 31), _mk_seg(1, "c0", 80, 111)]
        emb = extract_segment_embeddings(segs, clips.__getitem__, lambda x: x.sum(axis=(2, 3)))
        assert not np.allclose(emb[0], emb[1])

    def test_nonfinite_embedding_raises(self):
        clips = {"c0": _fake_clip(40, 0)}
        segs = [_mk_seg(0, "c0", 0, 31)]

        def bad_enc(x):
            out = x.sum(axis=(2, 3))
            out[0, 0] = np.nan
            return out

        with pytest.raises(ValueError, match="有限"):
            extract_segment_embeddings(segs, clips.__getitem__, bad_enc)

    def test_unknown_clip_raises(self):
        segs = [_mk_seg(0, "ghost", 0, 9)]
        with pytest.raises(KeyError):
            extract_segment_embeddings(segs, {"c0": _fake_clip()}.get, lambda x: x.sum(axis=(2, 3)))
