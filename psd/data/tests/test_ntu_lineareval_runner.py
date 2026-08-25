"""W33 — 线性评估入口脚本与融合 CLI 的胶水层测试。

覆盖: scripts/run_ntu_lineareval.py::preflight（fail-fast 路径校验）
     scripts/ntu_ensemble_3s.py::assemble_output（JSON 归档组装）
"""
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))


def _load_module(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_fake_repo(tmp_path, *, with_weights=True, with_data=True) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    data = repo / "data" / "ntu60_frame50" / "xsub"
    data.mkdir(parents=True)
    if with_data:
        for f in (
            "train_position.npy",
            "train_label.pkl",
            "val_position.npy",
            "val_label.pkl",
        ):
            (data / f).write_bytes(b"x")
    if with_weights:
        ck = repo / "runs" / "ntu_phaseB" / "joint_pretext"
        ck.mkdir(parents=True)
        (ck / "epoch300_model.pt").write_bytes(b"x")

    cfg = {
        "weights": "runs/ntu_phaseB/joint_pretext/epoch300_model.pt",
        "train_feeder_args": {
            "data_path": "data/ntu60_frame50/xsub/train_position.npy",
            "label_path": "data/ntu60_frame50/xsub/train_label.pkl",
        },
        "test_feeder_args": {
            "data_path": "data/ntu60_frame50/xsub/val_position.npy",
            "label_path": "data/ntu60_frame50/xsub/val_label.pkl",
        },
    }
    cfg_path = repo / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return repo, cfg_path


class TestPreflight:
    def test_valid_fake_repo_has_no_problems(self, tmp_path):
        mod = _load_module("scripts/run_ntu_lineareval.py", "run_ntu_lineareval")
        repo, cfg_path = _make_fake_repo(tmp_path)

        # external 缺失也应给出问题项（fake repo 无 AimCLR）
        problems = mod.preflight(cfg_path, repo)

        assert any("AimCLR" in p for p in problems)
        assert not any("checkpoint" in p for p in problems)
        assert not any("feeder_args" in p for p in problems)

    def test_missing_weights_is_flagged(self, tmp_path):
        mod = _load_module("scripts/run_ntu_lineareval.py", "run_ntu_lineareval")
        repo, cfg_path = _make_fake_repo(tmp_path, with_weights=False)

        problems = mod.preflight(cfg_path, repo)

        assert any("epoch300_model.pt" in p for p in problems)

    def test_missing_data_files_are_flagged(self, tmp_path):
        mod = _load_module("scripts/run_ntu_lineareval.py", "run_ntu_lineareval")
        repo, cfg_path = _make_fake_repo(tmp_path, with_data=False)

        problems = mod.preflight(cfg_path, repo)

        assert sum("不存在" in p for p in problems) >= 4

    def test_missing_config_returns_single_problem(self, tmp_path):
        mod = _load_module("scripts/run_ntu_lineareval.py", "run_ntu_lineareval")

        problems = mod.preflight(tmp_path / "nope.yaml", tmp_path)

        assert problems == [f"配置不存在: {tmp_path / 'nope.yaml'}"]


class TestAssembleOutput:
    def test_merges_ensemble_and_per_stream_info(self, tmp_path):
        mod = _load_module("scripts/ntu_ensemble_3s.py", "ntu_ensemble_cli")

        result = {"top1": 0.79, "top5": 0.93, "n": 100,
                  "alpha": {"joint": 0.6, "bone": 0.6, "motion": 0.4},
                  "stream_paths": {"joint": "j.pkl", "bone": "b.pkl", "motion": "m.pkl"}}

        out = mod.assemble_output(result, tmp_path)

        assert out["top1"] == 0.79 and out["n"] == 100
        for s in ("joint", "bone", "motion"):
            assert s in out["per_stream"]
            assert "best_top1" in out["per_stream"][s]
        assert "generated_at" in out and "protocol" in out

    def test_collect_reads_real_log_fixture(self, tmp_path):
        mod = _load_module("scripts/ntu_ensemble_3s.py", "ntu_ensemble_cli")
        work = tmp_path / "lineareval_bone"
        work.mkdir()
        (work / "log.txt").write_text("\tTop1: 70.00%\n\tBest Top1: 71.87%\n", encoding="utf-8")

        out = mod.assemble_output(
            {"top1": 0.8, "top5": 0.9, "n": 10,
             "alpha": {}, "stream_paths": {}}, tmp_path
        )

        assert out["per_stream"]["bone"]["best_top1"] == 71.87
