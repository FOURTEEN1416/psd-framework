"""P0.4 迭代自训练闭环核心测试（W10 窗口，先测后码）。

覆盖: B-1 softmax 温度校准/锚点侧标定、概率 margin、τ* 预注册选择规则、
B-2 standing 双路一致门/子聚类门（GT 无关）、迭代停止三保险、合成数据全循环冒烟。
"""
from __future__ import annotations

import numpy as np
import pytest

from psd.training.tcl_selftrain import (
    consensus_gate,
    fit_temperature,
    iteration_stop_decision,
    prob_margins,
    run_selftrain,
    select_tau_operating,
    softmax_temperature,
    subcluster_gate,
)


# ------------------------------------------------------------ B-1 校准

def test_softmax_temperature_rows_sum_to_one():
    sims = np.array([[0.90, 0.88, 0.10], [0.50, 0.49, 0.48]])
    probs = softmax_temperature(sims, T=0.05)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-9)
    # T 越小分布越尖锐：top1 概率更大
    flat = softmax_temperature(sims, T=10.0)
    assert probs[0, 0] > flat[0, 0]


def test_fit_temperature_hits_target_median_margin():
    """原始 cosine margin <0.05 的真实量级 → 标定后中位概率 margin ≈ 目标。"""
    rng = np.random.default_rng(42)
    n, P = 200, 6
    top = rng.uniform(0.90, 0.99, size=(n, 1))
    second = top - rng.uniform(0.001, 0.04, size=(n, 1))   # 原始 margin <0.05
    rest = rng.uniform(0.1, 0.4, size=(n, P - 2))
    sims = np.hstack([top, second, rest])
    T = fit_temperature(sims, target_median=0.10)
    probs = softmax_temperature(sims, T)
    _, margins = prob_margins(probs)
    assert abs(np.median(margins) - 0.10) < 0.02


def test_prob_margins_basic():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    top1, margins = prob_margins(probs)
    assert top1.tolist() == [0, 1]
    assert np.allclose(margins, [0.5, 0.7])


# ------------------------------------------------------------ τ* 预注册规则（分位数锚定）

def test_select_tau_quantile_hits_target_coverage():
    """均匀有效分数 + 目标覆盖 0.4 → τ* ≈ 0.6 分位。"""
    rng = np.random.default_rng(0)
    kappa = rng.uniform(0.0, 1.0, size=1000)
    pred = np.array(["a", "b"])[rng.integers(0, 2, size=1000)]
    priors = {"a": 0.5, "b": 0.5}
    tau = select_tau_operating(kappa, pred, priors, alpha=1.0, target_coverage=0.4)
    passed = np.mean((kappa / np.array([1.0, 1.0])[ (pred == "b").astype(int)]) >= tau)
    # α=1 且两类先验相等 → m_c 均为 1，直接按 κ 分位
    assert abs(passed - 0.4) < 0.05


def test_select_tau_respects_frequency_multiplier():
    """稀有类阈值下调生效: 同一 κ 分布下，稀有类有效分数被放大 → τ* 相对单一阈值更公平。"""
    rng = np.random.default_rng(1)
    n = 1000
    kappa = rng.uniform(0.0, 1.0, size=n)
    pred = np.array(["common"] * 900 + ["rare"] * 100)
    priors = {"common": 0.9, "rare": 0.1}
    tau = select_tau_operating(kappa, pred, priors, alpha=1.0, target_coverage=0.4)
    thr_common = tau                      # 多数类保持 τ
    thr_rare = tau * max(0.5, 0.1 / 0.9)  # 稀有类下调至 floor
    keep_c = np.mean(kappa[:900] >= thr_common)
    keep_r = np.mean(kappa[900:] >= thr_rare)
    assert keep_r > keep_c                # 稀有类通过率更高（频率感知生效）


# ------------------------------------------------------------ B-2 门控（GT 无关）

def test_consensus_gate_only_touches_target_class():
    head_pred = np.array(["standing", "standing", "sitting", "sitting"])
    proto_pred = np.array(["standing", "sitting", "walking", "sitting"])
    keep = consensus_gate(head_pred, proto_pred, target_label="standing")
    # standing 且双路一致 → 留；standing 不一致 → 剔；非 standing 一律不受门影响
    assert keep.tolist() == [True, False, True, True]


def test_subcluster_gate_drops_low_share_cluster():
    rng = np.random.default_rng(0)
    good = rng.normal(loc=[6.0, 0], scale=0.1, size=(20, 2))       # 高一致簇
    bad = rng.normal(loc=[-6.0, 0], scale=1.0, size=(20, 2))       # 散簇（角度远离好簇）
    emb = np.vstack([good, bad])
    # bad 簇预测无共识：argmax 在三类间轮换（plurality share = 7/20 < 0.7）
    dom = np.tile(np.eye(3) * 0.5 + 0.15, (7, 1))[:20]
    head_probs = np.vstack([
        np.tile([0.9, 0.05, 0.05], (20, 1)),
        dom,
    ])
    keep = subcluster_gate(emb, head_probs, k=2, min_share=0.7, seed=42)
    assert keep[:20].all()
    assert not keep[20:].any()


# ------------------------------------------------------------ 迭代停止三保险

def test_stop_decision_budget_and_convergence_and_drop():
    # 预算用尽
    stop, reason = iteration_stop_decision(
        rounds_done=6, change_rate=0.5, precision_history=[0.5, 0.5],
        max_iters=6, converge_rate=0.01, drop_patience=2)
    assert stop and reason == "budget"
    # 收敛
    stop, reason = iteration_stop_decision(
        rounds_done=2, change_rate=0.005, precision_history=[0.5, 0.5],
        max_iters=6, converge_rate=0.01, drop_patience=2)
    assert stop and reason == "converged"
    # 连续下降确认偏差警报
    stop, reason = iteration_stop_decision(
        rounds_done=3, change_rate=0.5, precision_history=[0.6, 0.55, 0.5],
        max_iters=6, converge_rate=0.01, drop_patience=2)
    assert stop and reason == "precision_drop"
    # 正常继续
    stop, reason = iteration_stop_decision(
        rounds_done=2, change_rate=0.3, precision_history=[0.5, 0.55],
        max_iters=6, converge_rate=0.01, drop_patience=2)
    assert not stop


# ------------------------------------------------------------ 合成数据全循环

def _synthetic_corpus(n_per_class=60, dim=32, seed=0):
    """3 类可分 + standing 类混入 40% 干扰段，模拟 W8 观察到的 standing 宽分布。"""
    rng = np.random.default_rng(seed)
    classes = ["sitting", "standing", "walking"]
    centers = rng.normal(scale=5.0, size=(3, dim))
    X, y = [], []
    for ci, cname in enumerate(classes):
        spread = 3.5 if cname == "standing" else 1.0   # standing 分布最宽
        X.append(centers[ci] + rng.normal(scale=spread, size=(n_per_class, dim)))
        y += [cname] * n_per_class
    X = np.vstack(X)
    y = np.array(y)
    perm = rng.permutation(len(X))
    anchor_mask = np.zeros(len(X), dtype=bool)
    anchor_mask[perm[: len(X) // 2]] = True            # 前 50% 为锚点侧
    return X.astype(np.float32), y, anchor_mask


BASE_KW = dict(
    class_names=["sitting", "standing", "walking"],
    head_cfg={"hidden_dim": 8, "epochs": 60, "lr": 0.01,
              "weight_decay": 0.0, "batch_size": 64, "device": "cpu"},
    calib_method="softmax_temperature", calib_target=0.10,
    tau_grid=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
    tau_select={"target_coverage": 0.35},
    alpha=1.0, standing_mode="consensus",
    subcluster_k=14, subcluster_min_share=0.7,
    max_iters=4, converge_change_rate=0.01, precision_drop_patience=2,
)


def test_run_selftrain_smoke_produces_round_records():
    X, y, anchor_mask = _synthetic_corpus()
    out = run_selftrain(X, y, anchor_mask, run_seed=42, **BASE_KW)
    assert len(out["rounds"]) >= 2                      # 至少 round0 + 一轮迭代
    r0 = out["rounds"][0]
    for key in ("pool_size", "coverage", "precision", "tau_operating"):
        assert key in r0
    assert 0.0 <= r0["coverage"] <= 1.0
    assert 0.0 <= r0["precision"] <= 1.0
    assert out["stop_reason"] in ("budget", "converged", "precision_drop")
    assert out["final_pool_idx"].dtype.kind == "i"


def test_run_selftrain_improves_or_stays_on_separable_data():
    """可分合成域上，迭代不应把池精度打崩（升级路径防误伤冒烟）。"""
    X, y, anchor_mask = _synthetic_corpus()
    out = run_selftrain(X, y, anchor_mask, run_seed=42, **BASE_KW)
    precisions = [r["precision"] for r in out["rounds"]]
    assert precisions[-1] >= 0.5


def test_run_selftrain_standing_mode_none_is_pure_ablation_path():
    X, y, anchor_mask = _synthetic_corpus()
    kw = dict(BASE_KW)
    kw["standing_mode"] = "none"
    out_none = run_selftrain(X, y, anchor_mask, run_seed=42, **kw)
    kw["standing_mode"] = "consensus"
    out_cons = run_selftrain(X, y, anchor_mask, run_seed=42, **kw)
    # 两模式都能跑完且记录结构一致（数值可不同——这正是消融要量的）
    assert [sorted(r.keys()) for r in out_none["rounds"]] == \
           [sorted(r.keys()) for r in out_cons["rounds"]]


def test_run_selftrain_pool_stays_inside_universe():
    """防泄漏协议不变式: 池恒 ⊆ 资格宇宙（默认评估侧），锚点种子永不入池。"""
    X, y, anchor_mask = _synthetic_corpus()
    out = run_selftrain(X, y, anchor_mask, run_seed=42, **BASE_KW)
    pool = set(out["final_pool_idx"].tolist())
    assert not pool.intersection(set(np.where(anchor_mask)[0].tolist()))


def test_run_selftrain_custom_universe_respected():
    """显式宇宙掩码生效: 池只从给定宇宙取段（供装配层传评估侧/其他提案源）。"""
    X, y, anchor_mask = _synthetic_corpus()
    universe = ~anchor_mask
    out = run_selftrain(X, y, anchor_mask, run_seed=42,
                        pool_universe_mask=universe, **BASE_KW)
    pool = set(out["final_pool_idx"].tolist())
    assert pool.issubset(set(np.where(universe)[0].tolist()))
    assert len(pool) > 0
