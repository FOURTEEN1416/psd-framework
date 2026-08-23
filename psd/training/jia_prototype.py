"""P0.3 锚点原型聚类 — W8 窗口 owner（Phase A Step 3，method.md Algorithm 1 初始化版）。

实现范围（Phase A 只到分配与置信，迭代自训练留给 P0.4）：
- 原型初始化：class_mean（Algorithm 1 忠实版：每类锚点嵌入均值）或
  kmeans（敏感性扫描用：K 簇 k-means++，簇标签 = 成员锚点多数表决）；
- 最近原型分配 + prototype-margin 置信 κ（method.md §3.3.2 冻结决策）；
- frequency-aware margin：稀有类阈值下调（τ_c = τ·max(0.5,(π_c/π_max)^α)），
  处理 sitting~37% vs lying~1.7% 的长尾现实——阈值层面处理而非重采样。

Φ 全程冻结；本模块纯 numpy，无训练回路。
"""
from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


class PrototypeClusterer:
    """锚点初始化的原型聚类器（余弦空间）。"""

    def __init__(self, mode: str = "class_mean", k: int | None = None, seed: int = 42,
                 max_iter: int = 100):
        if mode not in ("class_mean", "kmeans"):
            raise ValueError(f"未知 mode: {mode}")
        if mode == "kmeans" and (k is None or k < 1):
            raise ValueError("mode=kmeans 需要正整数 k")
        self.mode = mode
        self.k = k
        self.seed = seed
        self.max_iter = max_iter
        self.prototypes: np.ndarray | None = None      # (P, D) 已 L2 归一
        self.prototype_labels: np.ndarray | None = None  # (P,) str

    # ---------------------------------------------------------- 拟合

    def fit(self, emb: np.ndarray, labels: np.ndarray | None) -> "PrototypeClusterer":
        emb = np.asarray(emb, dtype=np.float64)
        feats = _l2_normalize(emb)
        if self.mode == "class_mean":
            if labels is None:
                raise ValueError("mode=class_mean 需要 labels")
            labels_arr = np.asarray(labels)
            classes = sorted(set(labels_arr.tolist()))
            protos = []
            for c in classes:
                mean_vec = feats[labels_arr == c].mean(axis=0)
                protos.append(mean_vec)
            self.prototypes = _l2_normalize(np.vstack(protos))
            self.prototype_labels = np.array(classes)
        else:
            if labels is None:
                raise ValueError("kmeans 簇需成员锚点标签做多数表决映射")
            assign_idx = self._kmeans(feats, int(self.k))
            proto_rows, proto_labs = [], []
            for j in range(int(self.k)):
                member = assign_idx == j
                if not member.any():  # 理论不可达（空簇已在 _kmeans 内修复）
                    continue
                proto_rows.append(feats[member].mean(axis=0))
                labs_j, counts_j = np.unique(np.asarray(labels)[member], return_counts=True)
                # 平票取字典序最小（确定性）
                proto_labs.append(labs_j[np.argmax(counts_j)])
            self.prototypes = _l2_normalize(np.vstack(proto_rows))
            self.prototype_labels = np.array(proto_labs)
        return self

    def _kmeans(self, feats: np.ndarray, k: int) -> np.ndarray:
        """确定性 k-means（k-means++ 远点采样 + Lloyd），空簇回迁最远点。"""
        rng = np.random.default_rng(self.seed)
        n = len(feats)
        centers = [int(rng.integers(n))]
        dist_sq = ((feats - feats[centers[0]]) ** 2).sum(axis=1)
        while len(centers) < min(k, n):
            prob = dist_sq / max(dist_sq.sum(), 1e-12)
            nxt = int(rng.choice(n, p=prob))
            centers.append(nxt)
            dist_sq = np.minimum(dist_sq, ((feats - feats[nxt]) ** 2).sum(axis=1))
        C = feats[np.asarray(centers)].copy()

        assign = np.zeros(n, dtype=np.int64)
        for _ in range(self.max_iter):
            sims = feats @ C.T
            new_assign = sims.argmax(axis=1)
            # Lloyd 更新
            for j in range(len(C)):
                member = new_assign == j
                if member.any():
                    C[j] = feats[member].mean(axis=0)
            # 空簇修复：把离各自中心最远的点迁入空簇
            empty = [j for j in range(len(C)) if not (new_assign == j).any()]
            if empty:
                per_point = ((feats - C[new_assign]) ** 2).sum(axis=1)
                for j in empty:
                    far = int(per_point.argmax())
                    new_assign[far] = j
                    per_point[far] = -1.0
                for j in range(len(C)):
                    member = new_assign == j
                    if member.any():
                        C[j] = feats[member].mean(axis=0)
            if (new_assign == assign).all():
                break
            assign = new_assign
        # 重编号为稳定序（按首现顺序），保证同数据同 seed 输出一致
        _, first = np.unique(assign, return_index=True)
        order = {old: new for new, old in enumerate(np.argsort(first))}
        return np.vectorize(order.__getitem__)(assign).astype(np.int64)

    # ---------------------------------------------------------- 分配

    def _cosine_sim(self, emb: np.ndarray) -> np.ndarray:
        assert self.prototypes is not None, "先 fit 再 assign"
        feats = _l2_normalize(np.asarray(emb, dtype=np.float64))
        return feats @ self.prototypes.T  # (N, P)

    def assign(self, emb: np.ndarray):
        """最近原型分配。返回 (proto_idx, pred_labels, kappa_margin)。

        κ = top1−top2 相似度差（prototype-margin，§3.3.2）；单原型时
        退化为 top1 相似度本身（无竞争对象可比较）。
        """
        sims = self._cosine_sim(emb)
        r = np.arange(len(sims))
        if sims.shape[1] == 1:
            top1 = np.zeros(len(sims), dtype=np.int64)
            margin = sims[:, 0].copy()
        else:
            part = np.argpartition(-sims, 1, axis=1)[:, :2]
            s12 = sims[r[:, None], part]
            order = s12.argsort(axis=1)[:, ::-1]
            top1 = part[r[:, None], order[:, :1]].reshape(-1)
            second = part[r[:, None], order[:, 1:]].reshape(-1)
            margin = sims[r, top1] - sims[r, second]
        margin = np.clip(margin, 0.0, 2.0)
        pred = self.prototype_labels[top1]
        return top1, np.asarray(pred), margin.astype(np.float64)


def frequency_aware_thresholds(
    tau: float,
    label_priors: dict[str, float],
    alpha: float = 1.0,
    floor_ratio: float = 0.5,
) -> dict[str, float]:
    """类别相关置信阈值：稀有类下调、多数类保持 τ。

    τ_c = τ · max(floor_ratio, (π_c / π_max)^α)；α=0 关闭（全类统一 τ）。
    floor_ratio 防止极端长尾把阈值压到失去过滤意义。
    """
    if tau <= 0:
        raise ValueError(f"tau 必须为正: {tau}")
    if alpha < 0:
        raise ValueError(f"alpha 必须 >=0: {alpha}")
    if not label_priors:
        return {}
    pi_max = max(label_priors.values())
    out: dict[str, float] = {}
    for lab, pi in label_priors.items():
        ratio = (pi / pi_max) ** alpha if pi_max > 0 else 1.0
        out[lab] = float(tau * max(floor_ratio, ratio))
    return out
