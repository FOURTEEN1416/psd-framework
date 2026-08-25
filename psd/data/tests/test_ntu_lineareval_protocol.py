"""W33 — NTU 线性评估/预训练配置协议保真护栏测试。

目的（R4 重实现正确性防线）: 本仓改编配置与 external/AimCLR 官方配置在
"科学不变量"上逐键相等，仅允许声明的本机适配差异（路径/device/num_worker/
work_dir/print_log）。官方超参被静默改动时此测试必须红——防挪门柱、防漂移。

参照源:
  - linear_eval: external/AimCLR/config/ntu60/linear_eval/linear_eval_aimclr_xsub_<stream>.yaml
  - pretext:     external/AimCLR/config/ntu60/pretext/pretext_aimclr_xsub_<stream>.yaml
本仓目标:
  - configs/ntu60_phaseb_lineareval_xsub_<stream>.yaml
  - configs/ntu60_phaseb_pretext_xsub_<stream>.yaml   (motion/bone 为 W33 备用件)
"""
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.ntu_aimclr_env import resolve_aimclr_root  # noqa: E402

# external/ 不随 worktree 走（gitignore），经只读解析器回退主检出
AIMCLR_CONFIG = resolve_aimclr_root(REPO_ROOT) / "config"

LINEAR_STREAMS = ["joint", "motion", "bone"]
# 新文件护栏只覆盖 motion/bone 备用 pretext；joint 沿用 W9 文件名，
# 由下方 TestJointLegacyConfig 专属护栏保护
PRETEXT_STREAMS = ["motion", "bone"]

# 官方有、本仓允许不同的键（本机适配白名单）
LOCAL_DIFF_KEYS = {"work_dir", "weights", "device", "num_worker", "print_log"}
# feeder_args 内允许不同的键
LOCAL_DIFF_FEEDER_KEYS = {"data_path", "label_path"}


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _strip_local_diffs(cfg: dict) -> dict:
    """剔除本机适配键，返回科学不变量视图。"""
    out = {}
    for k, v in cfg.items():
        if k in LOCAL_DIFF_KEYS:
            continue
        if k.endswith("_feeder_args") and isinstance(v, dict):
            out[k] = {
                kk: vv for kk, vv in v.items() if kk not in LOCAL_DIFF_FEEDER_KEYS
            }
        else:
            out[k] = v
    return out


@pytest.mark.parametrize("stream", LINEAR_STREAMS)
class TestLinearEvalProtocolFidelity:
    def test_invariants_equal_official(self, stream):
        official = _load(
            AIMCLR_CONFIG
            / "ntu60/linear_eval"
            / f"linear_eval_aimclr_xsub_{stream}.yaml"
        )
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_lineareval_xsub_{stream}.yaml"
        )

        assert _strip_local_diffs(ours) == _strip_local_diffs(official), (
            "线性评估配置偏离官方协议——R4 护栏触发，禁止静默改参"
        )

    def test_weights_point_to_own_stream_pretext_epoch300(self, stream):
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_lineareval_xsub_{stream}.yaml"
        )

        assert ours["weights"] == str(
            Path("runs") / "ntu_phaseB" / f"{stream}_pretext" / "epoch300_model.pt"
        ).replace("\\", "/")

    def test_data_paths_under_xsub_frame50(self, stream):
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_lineareval_xsub_{stream}.yaml"
        )

        train_dp = ours["train_feeder_args"]["data_path"].replace("\\", "/")
        test_dp = ours["test_feeder_args"]["data_path"].replace("\\", "/")
        assert train_dp == "data/ntu60_frame50/xsub/train_position.npy"
        assert test_dp == "data/ntu60_frame50/xsub/val_position.npy"

    def test_work_dir_isolated_per_stream(self, stream):
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_lineareval_xsub_{stream}.yaml"
        )

        assert f"lineareval_{stream}" in ours["work_dir"]


@pytest.mark.parametrize("stream", PRETEXT_STREAMS)
class TestPretextProtocolFidelity:
    def test_invariants_equal_official(self, stream):
        official = _load(
            AIMCLR_CONFIG
            / "ntu60/pretext"
            / f"pretext_aimclr_xsub_{stream}.yaml"
        )
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_pretext_xsub_{stream}.yaml"
        )

        assert _strip_local_diffs(ours) == _strip_local_diffs(official), (
            "pretext 配置偏离官方协议——R4 护栏触发"
        )

    def test_stream_matches(self, stream):
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_pretext_xsub_{stream}.yaml"
        )

        assert ours["stream"] == stream

    def test_num_epoch_is_official_300(self, stream):
        ours = _load(
            REPO_ROOT / "configs" / f"ntu60_phaseb_pretext_xsub_{stream}.yaml"
        )

        assert ours["num_epoch"] == 300


class TestJointLegacyConfig:
    def test_joint_pretext_existing_config_still_faithful(self):
        """既有 W9 joint 预训练配置持续受护栏保护（防后续窗口误改）。"""
        official = _load(
            AIMCLR_CONFIG
            / "ntu60/pretext"
            / "pretext_aimclr_xsub_joint.yaml"
        )
        ours = _load(REPO_ROOT / "configs" / "ntu60_phaseb_xsub_joint.yaml")

        assert _strip_local_diffs(ours) == _strip_local_diffs(official)


class TestLocalDiffAllowlistClosed:
    def test_no_unknown_extra_keys_in_linear_configs(self):
        for stream in LINEAR_STREAMS:
            official = _load(
                AIMCLR_CONFIG
                / "ntu60/linear_eval"
                / f"linear_eval_aimclr_xsub_{stream}.yaml"
            )
            ours = _load(
                REPO_ROOT
                / "configs"
                / f"ntu60_phaseb_lineareval_xsub_{stream}.yaml"
            )
            extra = set(ours.keys()) - set(official.keys()) - LOCAL_DIFF_KEYS
            assert not extra, f"{stream}: 出现未声明的新增配置键 {extra}"


class TestAimclrRootResolver:
    def test_prefers_local_external(self, tmp_path):
        from psd.data.ntu_aimclr_env import resolve_aimclr_root

        repo = tmp_path / "psd-framework-W33"
        local = repo / "external" / "AimCLR"
        local.mkdir(parents=True)

        assert resolve_aimclr_root(repo) == local

    def test_falls_back_to_main_checkout_sibling(self, tmp_path):
        from psd.data.ntu_aimclr_env import resolve_aimclr_root

        main = tmp_path / "psd-framework"
        (main / "external" / "AimCLR").mkdir(parents=True)
        repo = tmp_path / "psd-framework-W33"
        repo.mkdir()

        assert resolve_aimclr_root(repo) == main / "external" / "AimCLR"

    def test_exact_main_name_wins_over_other_worktrees(self, tmp_path):
        from psd.data.ntu_aimclr_env import resolve_aimclr_root

        other = tmp_path / "psd-framework-W20"
        (other / "external" / "AimCLR").mkdir(parents=True)
        main = tmp_path / "psd-framework"
        (main / "external" / "AimCLR").mkdir(parents=True)
        repo = tmp_path / "psd-framework-W33"
        repo.mkdir()

        assert resolve_aimclr_root(repo) == main / "external" / "AimCLR"

    def test_missing_everywhere_raises_listing_candidates(self, tmp_path):
        from psd.data.ntu_aimclr_env import resolve_aimclr_root

        repo = tmp_path / "psd-framework-W33"
        repo.mkdir()

        with pytest.raises(FileNotFoundError, match="external"):
            resolve_aimclr_root(repo)
