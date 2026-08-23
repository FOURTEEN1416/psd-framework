"""P0.3 评估指标 — W8 窗口 owner（Phase A Step 4，experiment-skeleton E3 口径）。

指标族：聚类纯度 / NMI / 覆盖率(τ) / 随机分配基线 / 多数类基线 /
种子噪声注入（风险登记册 R8 缓解项，审稿防线）。

统计纪律（experiment-skeleton §统计协议）：所有主数字由调用方以
≥3 seeds 聚合为 mean±std；本模块只提供单次计算原语。
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np


# ---------------------------------------------------------------- 一致性映射

def _pair_arrays(pred, true) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred)
    t = np.asarray(true)
    if len(p) != len(t):
        raise ValueError(f"pred/true 长度不一致: {len(p)} vs {len(t)}")
    return p, t


def label_priors(labels) -> dict[str, float]:
    """经验类别先验 π_c = n_c / N。"""
    arr = np.asarray(labels)
    cnt = Counter(arr.tolist())
    n = len(arr)
    return {k: v / n for k, v in cnt.items()}


# ---------------------------------------------------------------- 主指标

def purity(pred, true) -> float:
    """聚类纯度：Σ_簇 max_类 |簇∩类| / N（pred 为簇/预测标签均可）。"""
    p, t = _pair_arrays(pred, true)
    n = len(p)
    groups: defaultdict = defaultdict(Counter)
    for pi, ti in zip(p.tolist(), t.tolist()):
        groups[pi][ti] += 1
    correct = sum(c.most_common(1)[0][1] for c in groups.values())
    return float(correct / n)


def nmi(u, v) -> float:
    """归一化互信息（算术平均规范：2·I/(H_u+H_v)），纯 numpy 实现。

    约定：双方均为常数 → 1.0；单方常数另一方非平凡 → 0.0。
    """
    a = np.asarray([str(x) for x in np.asarray(u).tolist()])
    b = np.asarray([str(x) for x in np.asarray(v).tolist()])
    n = len(a)
    joint: Counter = Counter(zip(a.tolist(), b.tolist()))
    ca: Counter = Counter(a.tolist())
    cb: Counter = Counter(b.tolist())

    mi = 0.0
    for (ai, bj), nij in joint.items():
        pij = nij / n
        mi += pij * np.log(pij / ((ca[ai] / n) * (cb[bj] / n)))
    def _entropy(c: Counter) -> float:
        h = 0.0
        for nj in c.values():
            pj = nj / n
            h -= pj * np.log(pj)
        return h
    ha, hb = _entropy(ca), _entropy(cb)
    denom = ha + hb
    if denom == 0.0:
        return 1.0
    return float(max(0.0, 2.0 * mi / denom))


def coverage(margins: np.ndarray, tau_grid) -> list[float]:
    """覆盖率曲线：coverage(τ) = P(κ ≥ τ)。"""
    m = np.asarray(margins, dtype=np.float64)
    return [float((m >= t).mean()) for t in tau_grid]


def purity_at_threshold(pred, true, margins: np.ndarray, tau: float) -> tuple[float, float]:
    """τ 过滤后的 (覆盖子集纯度, 覆盖率)。覆盖为空时纯度记 NaN。"""
    p, t = _pair_arrays(pred, true)
    m = np.asarray(margins, dtype=np.float64)
    mask = m >= tau
    cov = float(mask.mean())
    if not mask.any():
        return float("nan"), cov
    return purity(p[mask], t[mask]), cov


# ---------------------------------------------------------------- 基线

def random_assignment_purity(label_priors_map: dict[str, float]) -> float:
    """随机标签分配的期望纯度 = Σ π_c²（解析基线，E3 验收对照）。"""
    return float(sum(p * p for p in label_priors_map.values()))


def majority_class_baseline(labels) -> float:
    """多数类预测器准确率（强于随机的参照下界）。"""
    priors = label_priors(labels)
    return max(priors.values()) if priors else 0.0


# ---------------------------------------------------------------- 噪声注入（R8）

def inject_label_noise(labels, rate: float, seed: int = 42) -> np.ndarray:
    """向锚点标签注入均匀换类噪声：恰好 round(rate·N) 条换成异类。

    - 换类目标从「其余已出现类别」均匀抽取（保持类别体系不变）；
    - 确定性：同 rate 同 seed 结果一致；
    - 用途：验证伪标签迭代不放大种子错误（experiment-skeleton 消融行）。
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"rate 必须在 [0,1]: {rate}")
    arr = np.asarray(labels)
    classes = sorted(set(arr.tolist()))
    if len(classes) < 2:
        raise ValueError("噪声注入需要 ≥2 个类别")
    rng = np.random.default_rng(seed)
    out = arr.copy()
    n = len(arr)
    n_flip = int(round(rate * n))
    if n_flip == 0:
        return out
    flip_idx = rng.choice(n, size=n_flip, replace=False)
    for i in flip_idx:
        others = [c for c in classes if c != arr[i]]
        out[i] = rng.choice(others)
    return out
