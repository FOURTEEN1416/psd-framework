"""P0.2 契约测试：SMQ 输入适配层（RED→GREEN）。

覆盖：
- clip → SMQ 视图 (3,T,24,1) 的形状/有限性/置信度置零
- 导出器跳过无效 clip
- episode 拼接与 GT 段元数据
- eval clip 确定性抽选（同 seed 同结果、不重复、跨狗）
"""
import numpy as np
import pytest

from psd.data.smq_input import (
    build_episode,
    clip_to_smq_view,
    export_smq_features,
    select_eval_clips,
)


def _fake_clip(t=200, nan=False):
    rng = np.random.default_rng(0)
    kp = rng.normal(size=(t, 24, 3)).astype(np.float32)
    if nan:
        kp[:] = np.nan
    weight = rng.uniform(0.0, 1.0, size=(t, 24)).astype(np.float32)
    return {"kp_world": kp, "kp_weight": weight, "frame_idx": np.arange(t)}


class TestClipToSmqView:
    def test_shape_dtype_finite(self):
        clip = _fake_clip(200)
        view = clip_to_smq_view(clip["kp_world"], clip["kp_weight"], target_t=128)
        assert view.shape == (3, 128, 24, 1)
        assert view.dtype == np.float32
        assert np.isfinite(view).all()

    def test_conf_zeroing(self):
        clip = _fake_clip(64)
        weight = clip["kp_weight"]
        low = weight < 0.5
        assert low.any(), "fixture 需含低置信关节"
        view = clip_to_smq_view(
            clip["kp_world"], weight, target_t=64, conf_threshold=0.5, normalize=False
        )
        # 重排为 (T,24,...) 后用二维掩码取低置信关节（setitem 广播语义的镜像校验）
        per_joint = np.transpose(view, (1, 2, 0, 3))  # (T,24,3,1)
        assert (per_joint[low] == 0).all()

    def test_target_t_one_frame(self):
        clip = _fake_clip(1)
        view = clip_to_smq_view(clip["kp_world"], clip["kp_weight"], target_t=128)
        assert view.shape == (3, 128, 24, 1)

    def test_native_length_preserved(self):
        # E-D 协议：target_t=None 保留原生帧长
        clip = _fake_clip(173)
        view = clip_to_smq_view(clip["kp_world"], clip["kp_weight"], target_t=None)
        assert view.shape == (3, 173, 24, 1)
        assert np.isfinite(view).all()


class TestExportSmqFeatures:
    def test_skips_invalid(self, tmp_path):
        good = _fake_clip(100)
        bad = _fake_clip(100, nan=True)
        np.savez(tmp_path / "interpet_dog01_p01_take01_ego_001.npz", **good)
        np.savez(tmp_path / "interpet_dog02_p01_take01_ego_001.npz", **bad)
        out = tmp_path / "features"
        report = export_smq_features(tmp_path, out, target_t=64)
        names = [p.stem for p in sorted(out.glob("*.npy"))]
        assert len(names) == 1
        assert names[0] == "interpet_dog01_p01_take01_ego_001"
        assert report["skipped"] == ["interpet_dog02_p01_take01_ego_001"]


class TestBuildEpisode:
    def test_concat_and_segments(self, tmp_path):
        feats = tmp_path / "features"
        feats.mkdir()
        t = 8
        for name in ("clipA", "clipB", "clipC"):
            np.save(feats / f"{name}.npy", np.zeros((3, t, 24, 1), dtype=np.float32))
        ep = build_episode(feats, ["clipA", "clipB", "clipC"])
        assert ep["data"].shape == (3, 24, 24, 1)
        segs = [(s["name"], s["start"], s["end"]) for s in ep["segments"]]
        assert segs == [("clipA", 0, 8), ("clipB", 8, 16), ("clipC", 16, 24)]

    def test_variable_t_concat(self, tmp_path):
        # E-D 协议变更：允许不等长拼接（边界取累计和），不再拒绝
        feats = tmp_path / "features"
        feats.mkdir()
        np.save(feats / "a.npy", np.zeros((3, 8, 24, 1), dtype=np.float32))
        np.save(feats / "b.npy", np.zeros((3, 16, 24, 1), dtype=np.float32))
        ep = build_episode(feats, ["a", "b"])
        assert ep["data"].shape == (3, 24, 24, 1)
        assert [(s["name"], s["start"], s["end"]) for s in ep["segments"]] == [
            ("a", 0, 8), ("b", 8, 24)]


class TestSelectEvalClips:
    def test_deterministic_and_unique(self):
        names = [f"interpet_dog{i:02d}_p01_take01_ego_{j:03d}"
                 for i in range(1, 13) for j in range(3)]
        a = select_eval_clips(names, total=20, seed=42)
        b = select_eval_clips(names, total=20, seed=42)
        assert a == b
        assert len(a) == len(set(a)) == 20

    def test_rotate_by_dog_chunks_are_dog_disjoint(self):
        from psd.data.smq_input import rotate_by_dog

        names = [f"interpet_dog{i:02d}_p01_take01_ego_{j:03d}"
                 for i in range(1, 13) for j in range(5)]
        rot = rotate_by_dog(names, seed=7)
        assert sorted(rot) == sorted(names)
        assert rot == rotate_by_dog(names, seed=7)
        for i in range(0, len(rot), 4):
            chunk = rot[i:i + 4]
            dogs = {n.split("_")[1] for n in chunk}
            assert len(dogs) == len(chunk)

    def test_episode_groups_cross_dog(self):
        from psd.data.smq_input import group_into_episodes

        names = [f"interpet_dog{i:02d}_p01_take01_ego_{j:03d}"
                 for i in range(1, 13) for j in range(3)]
        picked = select_eval_clips(names, total=20, seed=42)
        groups = group_into_episodes(picked, clips_per_episode=5)
        assert len(groups) == 4
        for g in groups:
            dogs = {n.split("_")[1] for n in g}
            assert len(dogs) == len(g), "同一 episode 内 dog 应互不相同"
