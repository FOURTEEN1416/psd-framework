"""W33 — NTU 三流融合（3s ensemble）数学与产物收集测试。

协议依据: external/AimCLR/ensemble_ntu_cs.py（官方移植源）
  r = joint*0.6 + bone*0.6 + motion*0.4（logits 加权求和）
  top1 = argmax(r)；top5 = label ∈ argsort(r)[-5:]
本仓移植差异（在报告与本文件同步声明）:
  1. 三流分数按 val_label.pkl 的 sample_name 顺序对齐（官方隐式依赖三个
     pickle 的插入顺序一致），键缺失/不一致时 fail-fast；
  2. 输出结构化 dict 供 JSON 归档，替代官方纯 print。
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.ntu_ensemble import (  # noqa: E402
    DEFAULT_ALPHA,
    collect_stream_result,
    fuse_scores,
    load_score_pkl,
    load_labels,
    run_ensemble,
    topk_accuracy,
)


def _write_score_pkl(path: Path, mapping: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(mapping, f)


def _write_label_pkl(path: Path, names: list, labels: list) -> None:
    with open(path, "wb") as f:
        pickle.dump((names, labels), f)


class TestLoadScorePkl:
    def test_roundtrip_preserves_order_and_arrays(self, tmp_path):
        p = tmp_path / "test_result.pkl"
        src = {"S001": np.array([0.1, 0.9]), "S002": np.array([0.8, 0.2])}
        _write_score_pkl(p, src)

        loaded = load_score_pkl(p)

        assert list(loaded.keys()) == ["S001", "S002"]
        np.testing.assert_allclose(loaded["S001"], [0.1, 0.9])

    def test_missing_file_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_score_pkl(tmp_path / "nope.pkl")


class TestFuseScores:
    def test_weighted_sum_matches_official_alpha(self):
        joint = {"a": np.array([1.0, 0.0, 0.0])}
        bone = {"a": np.array([0.0, 1.0, 0.0])}
        motion = {"a": np.array([0.0, 0.0, 1.0])}

        names, fused = fuse_scores(
            {"joint": joint, "bone": bone, "motion": motion}
        )

        assert names == ["a"]
        # 0.6*[1,0,0] + 0.6*[0,1,0] + 0.4*[0,0,1]
        np.testing.assert_allclose(fused[0], [0.6, 0.6, 0.4])

    def test_fusion_flips_argmax_against_single_stream(self):
        # joint 单看 argmax=类0；bone 高分注入后融合 argmax 应翻到类1
        joint = {"a": np.array([2.0, 1.0, 0.0])}
        bone = {"a": np.array([0.0, 9.0, 0.0])}
        motion = {"a": np.array([0.0, 0.0, 0.0])}

        _, fused = fuse_scores({"joint": joint, "bone": bone, "motion": motion})

        assert int(np.argmax(fused[0])) == 1

    def test_key_set_mismatch_fail_fast(self):
        joint = {"a": np.array([1.0, 0.0])}
        bone = {"b": np.array([0.0, 1.0])}  # 键不一致
        motion = {"a": np.array([1.0, 0.0])}

        with pytest.raises(ValueError, match="bone"):
            fuse_scores({"joint": joint, "bone": bone, "motion": motion})

    def test_row_order_follows_joint_insertion_when_keys_align(self):
        mk = lambda v: np.array([v, 0.0])  # noqa: E731
        joint = {"s2": mk(1), "s1": mk(2)}
        bone = {"s1": mk(5), "s2": mk(6)}  # 插入顺序不同但键集合一致
        motion = {"s1": mk(0), "s2": mk(0)}

        names, _ = fuse_scores({"joint": joint, "bone": bone, "motion": motion})

        assert names == ["s2", "s1"]


class TestTopkAccuracy:
    def test_top1_counts_argmax_hits(self):
        scores = np.array(
            [[0.1, 0.9, 0.0], [0.8, 0.1, 0.1], [0.2, 0.2, 0.6]]
        )
        labels = [1, 0, 2]

        assert topk_accuracy(scores, labels, k=1) == pytest.approx(1.0)

    def test_top5_boundary_membership(self):
        rng = np.random.RandomState(0)
        scores = rng.rand(20, 10)
        labels = [int(np.argmin(scores[i])) for i in range(20)]  # 全部垫底

        acc1 = topk_accuracy(scores, labels, k=1)
        acc5 = topk_accuracy(scores, labels, k=5)

        assert acc1 == pytest.approx(0.0)
        assert acc5 == pytest.approx(0.0)

    def test_matches_official_argsort_last5_semantics(self):
        # 官方语义: rank=argsort(r); hit = l in rank[-5:]
        row = np.array([0.05, 0.25, 0.45, 0.65, 0.85, 0.15, 0.35, 0.55, 0.75, 0.95])
        scores = row[None, :]
        assert topk_accuracy(scores, [4], k=5) == pytest.approx(1.0)   # 0.85 是第2高→进前5
        assert topk_accuracy(scores, [0], k=5) == pytest.approx(0.0)   # 0.05 最低→不在前5


class TestRunEnsemble:
    def test_end_to_end_matches_manual_computation(self, tmp_path):
        names = ["n1", "n2", "n3", "n4"]
        labels = [0, 1, 2, 0]
        streams = {
            "joint": {"n1": [9, 0, 0], "n2": [0, 9, 0], "n3": [0, 0, 9], "n4": [5, 1, 1]},
            "bone": {"n1": [0, 8, 0], "n2": [0, 8, 0], "n3": [0, 0, 8], "n4": [1, 4, 1]},
            "motion": {"n1": [0, 0, 7], "n2": [0, 7, 0], "n3": [0, 0, 7], "n4": [1, 1, 6]},
        }
        paths = {}
        for s, m in streams.items():
            p = tmp_path / f"{s}.pkl"
            _write_score_pkl(p, {k: np.array(v, dtype=np.float64) for k, v in m.items()})
            paths[s] = p
        label_pkl = tmp_path / "val_label.pkl"
        _write_label_pkl(label_pkl, names, labels)

        result = run_ensemble(paths, label_pkl)

        assert result["n"] == 4
        # 手工复算 n4: 0.6*[5,1,1]+0.6*[1,4,1]+0.4*[1,1,6] = [4.0,3.4,3.6] → argmax=类0
        # （夹具刻意避开融合平分: 官方 argsort[-k:] 语义在平分时命中索引不确定）
        assert result["top1"] == pytest.approx(1.0)
        assert result["top5"] == pytest.approx(1.0)
        assert result["alpha"] == DEFAULT_ALPHA

    def test_missing_sample_in_one_stream_raises(self, tmp_path):
        _write_score_pkl(tmp_path / "joint.pkl", {"n1": np.zeros(3)})
        _write_score_pkl(tmp_path / "bone.pkl", {"n2": np.zeros(3)})  # 缺 n1
        _write_score_pkl(tmp_path / "motion.pkl", {"n1": np.zeros(3)})
        label_pkl = tmp_path / "val_label.pkl"
        _write_label_pkl(label_pkl, ["n1"], [0])

        with pytest.raises(ValueError, match="bone"):
            run_ensemble(
                {
                    "joint": tmp_path / "joint.pkl",
                    "bone": tmp_path / "bone.pkl",
                    "motion": tmp_path / "motion.pkl",
                },
                label_pkl,
            )


class TestCollectStreamResult:
    def test_parses_best_top1_from_log(self, tmp_path):
        work = tmp_path / "lineareval_joint"
        work.mkdir()
        (work / "log.txt").write_text(
            "\tTop1: 71.20%\n\tBest Top1: 72.30%\n"
            "\tTop1: 74.01%\n\tBest Top1: 74.01%\n",
            encoding="utf-8",
        )

        info = collect_stream_result(work)

        assert info["best_top1"] == pytest.approx(74.01)
        assert info["last_top1"] == pytest.approx(74.01)

    def test_missing_log_returns_none_fields(self, tmp_path):
        info = collect_stream_result(tmp_path)

        assert info["best_top1"] is None
