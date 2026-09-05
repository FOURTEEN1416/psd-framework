"""P0.4 伪标签迭代自训练闭环 — W10 窗口 owner（method.md Algorithm 1 第 4-6 步）。

每轮: Ω 重分配全量段 → 新 κ（B-1 校准后概率 margin）→ τ_c 过滤入池
→ 池并入训练集 → 重训 Ω；原型按 Algorithm 1 第 6 步从种子∪池 class_mean 重估，
承担 B-2 standing 双路一致门的第二意见。迭代预算/收敛判据/精度下降熔断
写死进 config（防确认偏差跨轮累积）。

GT 泄漏防线（本模块最高纪律）:
- truth_all 仅用于评估指标计算，绝不进入任何训练/门控/阈值标定路径;
- 温度 T 只在锚点侧标定; 类别先验只取自当前已标注集合（种子∪池）。
- R16 修正: precision_stop=False 时，oracle 池精度只作诊断记录，
  不进入停止控制路径（端到端臂无规则共识参照，熔断禁用）;
  precision_stop=True（默认）保留 P0.4 语义——其 truth_all 为规则引擎
  共识伪 GT（非人工标注），熔断消费的是该弱参照精度。

口径: public_real_physics_prior（metric_layer 锁字段由装配层写入导出物）。
"""
from __future__ import annotations

import numpy as np

from psd.training.jia_metrics import label_priors
from psd.training.jia_prototype import PrototypeClusterer, frequency_aware_thresholds

STANDING_LABEL = "standing"   # B-2 治理目标类（W8: 错误伪标签绝对量最大，逐类精度 0.162）


# ---------------------------------------------------------------- B-1 校准原语

def softmax_temperature(sims: np.ndarray, T: float) -> np.ndarray:
    """(N,P) 余弦相似度 -> softmax(sim/T) 概率（行和 1）。"""
    z = np.asarray(sims, dtype=np.float64) / max(float(T), 1e-9)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def prob_margins(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(top1 索引, top1-top2 概率 margin)；单列时退化为该列概率本身。"""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[1] == 1:
        return np.zeros(len(probs), dtype=np.int64), probs[:, 0].copy()
    part = np.argpartition(-probs, 1, axis=1)[:, :2]
    r = np.arange(len(probs))
    s12 = probs[r[:, None], part]
    order = s12.argsort(axis=1)[:, ::-1]
    top1 = part[r[:, None], order[:, :1]].reshape(-1)
    second = part[r[:, None], order[:, 1:]].reshape(-1)
    return top1.astype(np.int64), (probs[r, top1] - probs[r, second]).astype(np.float64)


def fit_temperature(anchor_sims: np.ndarray, target_median: float = 0.10,
                    lo: float = 1e-3, hi: float = 20.0, iters: int = 64) -> float:
    """锚点侧二分标定温度 T：校准后 top1-top2 概率 margin 中位数 ≈ 目标。

    med(T) 关于 T 单调递减（T 越大分布越平）。仅消费锚点侧相似度——GT 无关。
    若最平分布仍达不到目标，返回 hi（能给出的最平校准，诚实退化）。
    """
    sims = np.asarray(anchor_sims, dtype=np.float64)

    def median_margin(T: float) -> float:
        _, m = prob_margins(softmax_temperature(sims, T))
        return float(np.median(m))

    if median_margin(hi) > target_median:
        return hi
    for _ in range(int(iters)):
        mid = (lo + hi) / 2.0
        if median_margin(mid) > target_median:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------- τ* 预注册选择

def select_tau_operating(kappa_eligible: np.ndarray, pred_eligible: np.ndarray,
                         priors: dict[str, float], alpha: float,
                         target_coverage: float) -> float:
    """预注册规则（分位数锚定）: 对可入池段的有效分数 κ/m_c 取 (1-target) 分位数为 τ*。

    构造性保证第 0 轮覆盖率 ≈ target（非退化）；后续轮冻结复用同一 τ*，
    不随轮重选（防漂移择优）。只消费预测侧统计与锚点先验，GT 无关。
    """
    kappa_eligible = np.asarray(kappa_eligible, dtype=np.float64)
    if len(kappa_eligible) == 0:
        raise ValueError("可入池宇宙为空——无法标定 τ*")
    mult = frequency_aware_thresholds(1.0, priors, alpha=alpha)
    eff = kappa_eligible / np.array(
        [mult.get(str(l), 1.0) for l in np.asarray(pred_eligible)])
    return float(np.quantile(eff, 1.0 - float(target_coverage)))


# ---------------------------------------------------------------- B-2 门控（GT 无关）

def consensus_gate(head_pred: np.ndarray, proto_pred: np.ndarray,
                   target_label: str) -> np.ndarray:
    """双路一致门: 预测为 target_label 的段须原型路 argmax 与头路一致才可入池；
    非 target 段一律放行（治理面只覆盖 standing，避免误伤他类）。"""
    hp = np.asarray(head_pred)
    pp = np.asarray(proto_pred)
    return ~((hp == target_label) & (pp != hp))


def subcluster_gate(emb_cand: np.ndarray, head_probs_cand: np.ndarray,
                    k: int, min_share: float, seed: int = 42) -> np.ndarray:
    """standing 候选子聚类门: 候选内 k-means 子聚类，簇内头预测主占比 < min_share
    整簇剔除（W8 K 扫描示 K=14 受益；纯无监督，GT 无关）。"""
    n = len(emb_cand)
    if n == 0:
        return np.zeros(0, dtype=bool)
    k_eff = int(min(k, n))                      # 候选数不足 K 时退化为单例簇（全保留）
    cl = PrototypeClusterer(mode="kmeans", k=k_eff, seed=int(seed)).fit(
        emb_cand, np.zeros(n, dtype=np.int64))  # kmeans 模式的 labels 仅作多数表决占位
    cluster_ids, _, _ = cl.assign(emb_cand)
    hard_pred = np.asarray(head_probs_cand).argmax(axis=1)
    keep = np.zeros(n, dtype=bool)
    for j in np.unique(cluster_ids):
        member = cluster_ids == j
        votes = np.bincount(hard_pred[member], minlength=head_probs_cand.shape[1])
        share = votes.max() / max(member.sum(), 1)
        keep[member] = share >= min_share
    return keep


# ---------------------------------------------------------------- 迭代停止三保险

def iteration_stop_decision(rounds_done: int, change_rate: float | None,
                            precision_history: list[float], max_iters: int,
                            converge_rate: float, drop_patience: int
                            ) -> tuple[bool, str | None]:
    """停止裁决顺序: 预算用尽 > 分配稳定 > 连续精度下降（确认偏差熔断）。"""
    if rounds_done >= int(max_iters):
        return True, "budget"
    if change_rate is not None and change_rate < float(converge_rate):
        return True, "converged"
    h = precision_history[-(int(drop_patience) + 1):]
    if len(h) == int(drop_patience) + 1 and all(
            h[i + 1] < h[i] for i in range(int(drop_patience))):
        return True, "precision_drop"
    return False, None


# ---------------------------------------------------------------- 循环主体

def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _proto_path(cl: PrototypeClusterer, emb_all: np.ndarray, calib_on: bool,
                calib_target: float, anchor_idx: np.ndarray):
    """原型路全量分配: 返回 (pred_str, margin, anchor_sims)。"""
    feats_n = _normalize_rows(emb_all)
    sims = feats_n @ cl.prototypes.T
    # 原始 cosine margin（未校准口径，B-1 off 行沿用 W8 语义）
    r = np.arange(len(sims))
    if sims.shape[1] == 1:
        raw_margin = sims[:, 0].copy()
    else:
        part = np.argpartition(-sims, 1, axis=1)[:, :2]
        s12 = sims[r[:, None], part]
        order = s12.argsort(axis=1)[:, ::-1]
        t1 = part[r[:, None], order[:, :1]].reshape(-1)
        t2 = part[r[:, None], order[:, 1:]].reshape(-1)
        raw_margin = np.clip(sims[r, t1] - sims[r, t2], 0.0, 2.0)
    pred = cl.prototype_labels[sims.argmax(axis=1)]
    if calib_on:
        T = fit_temperature(sims[anchor_idx], target_median=calib_target)
        probs = softmax_temperature(sims, T)
        pred_idx, margin = prob_margins(probs)
        pred = cl.prototype_labels[pred_idx]
    else:
        margin = raw_margin
    return np.asarray(pred), np.asarray(margin, dtype=np.float64)


def run_selftrain(
    emb_all: np.ndarray,
    truth_all: np.ndarray,                 # ⚠️ 仅评估用，绝不进入训练/门控路径
    anchor_mask: np.ndarray,
    *,
    run_seed: int,
    class_names: list[str],
    head_cfg: dict,
    calib_method: str,
    calib_target: float,
    tau_grid: list[float],
    tau_select: dict,
    alpha: float,
    standing_mode: str,
    subcluster_k: int,
    subcluster_min_share: float,
    pool_universe_mask: np.ndarray | None = None,
    # ↑ 池资格宇宙: 默认 ~anchor_mask（与 W8 力学镜像: 种子=锚点侧，
    #   未标注提案=评估侧）——池恒 ⊆ 宇宙，防泄漏协议不变式
    max_iters: int = 6,
    converge_change_rate: float = 0.01,
    precision_drop_patience: int = 2,
    precision_stop: bool = True,
    # ↑ R16 协议修正: False 时 oracle 池精度仅作诊断记录，绝不进入停止控制路径
    #   （端到端臂无规则共识参照，熔断禁用，停止=预算/收敛）
    head_calib: bool = False,
    # ↑ R16 诊断: 每轮对头路概率做锚点侧温度再校准（GT 无关），
    #   使 κ 与 τ*（原型路校准尺度）可比——检验门控失效是否为尺度错配所致
) -> dict:
    """Algorithm 1 第 4-6 步迭代闭环。返回逐轮记录与最终池索引。

    池语义: REPLACE（每轮重过滤，非单调累积——标准自训练做法，
    阈值随先验演化时旧低质条目自然退出）。
    """
    emb_all = np.asarray(emb_all, dtype=np.float32)
    truth_all = np.asarray(truth_all)
    anchor_mask = np.asarray(anchor_mask, dtype=bool)
    N, D = emb_all.shape
    calib_on = calib_method == "softmax_temperature"
    if not calib_on and calib_method != "off":
        raise ValueError(f"未知 calib_method: {calib_method}")
    if standing_mode not in ("none", "consensus", "subcluster"):
        raise ValueError(f"未知 standing_mode: {standing_mode}")

    anchor_idx = np.where(anchor_mask)[0]
    if pool_universe_mask is None:
        eligible = ~anchor_mask                   # 可入池宇宙 = 非锚点段（评估侧提案）
    else:
        eligible = np.asarray(pool_universe_mask, dtype=bool) & ~anchor_mask

    cls_of = {c: i for i, c in enumerate(class_names)}
    y_anchor_id = np.array([cls_of[str(v)] for v in truth_all[anchor_idx]])

    labeled_mask = anchor_mask.copy()             # 已标注集合（种子∪当轮池）
    pool_mask = np.zeros(N, dtype=bool)
    rounds: list[dict] = []
    prev_pred_full: np.ndarray | None = None
    stop_reason: str | None = None
    precision_history: list[float] = []

    def _record(pred_full, kappa, pool_mask_r, round_tag: str, extra=None):
        idx_pool = np.where(pool_mask_r)[0]
        prec = float(np.mean(pred_full[idx_pool] == truth_all[idx_pool])) \
            if len(idx_pool) else float("nan")
        rec = {
            "round": round_tag,
            "pool_size": int(len(idx_pool)),
            "coverage": round(float(len(idx_pool)) / max(eligible.sum(), 1), 4),
            "precision": (round(prec, 4) if np.isfinite(prec) else None),
        }
        by_cls = {}
        for c in class_names:
            sel = idx_pool[pred_full[idx_pool] == c]
            if len(sel):
                by_cls[c] = round(float(np.mean(pred_full[sel] == truth_all[sel])), 4)
        rec["precision_by_pred_class"] = by_cls
        if extra:
            rec.update(extra)
        rounds.append(rec)
        if rec["precision"] is not None and precision_stop:
            precision_history.append(rec["precision"])
        return rec

    # ---------------- 第 0 轮: W8 分配基线（原型路，无头）
    cl0 = PrototypeClusterer(mode="class_mean", seed=run_seed).fit(
        emb_all[anchor_mask], truth_all[anchor_mask])
    pred0, kappa0 = _proto_path(cl0, emb_all, calib_on, calib_target, anchor_idx)

    priors0 = label_priors(truth_all[anchor_idx])          # GT 无关（锚点即训练标签）
    grid0 = _tau_curve_over_grid(tau_grid, pred0, kappa0, priors0, alpha, eligible)
    tau_star = select_tau_operating(
        kappa0[eligible], pred0[eligible], priors0, alpha,
        target_coverage=float(tau_select["target_coverage"]))
    thr0 = frequency_aware_thresholds(tau_star, priors0, alpha=alpha)
    pool0 = eligible & (kappa0 >= np.array([thr0.get(str(l), tau_star) for l in pred0]))
    _record(pred0, kappa0, pool0, "r0_prototype",
            {"tau_operating": round(float(tau_star), 4),
             "tau_grid_curve": grid0})
    pool_mask = pool0
    prev_pred_full = pred0
    train_label_str = np.full(N, "", dtype=object)
    train_label_str[anchor_mask] = [str(v) for v in truth_all[anchor_mask]]

    # ---------------- 迭代 r ≥ 1: Ω 重分配 → τ_c 入池 → 并入训练集 → 重训 Ω
    from psd.training.tcl_head import TorchHead

    rounds_done = 0
    while rounds_done < int(max_iters):
        lab_idx = np.where(labeled_mask)[0]
        y_lab_id = np.array([cls_of[train_label_str[i]] for i in lab_idx])

        if calib_on:
            # Ω 训练（冻结 Φ 特征上）
            head = TorchHead(dim_in=D, n_classes=len(class_names),
                             hidden_dim=int(head_cfg.get("hidden_dim", 64)),
                             seed=run_seed, epochs=int(head_cfg.get("epochs", 150)),
                             lr=float(head_cfg.get("lr", 1e-3)),
                             weight_decay=float(head_cfg.get("weight_decay", 1e-4)),
                             batch_size=int(head_cfg.get("batch_size", 128)),
                             device=str(head_cfg.get("device", "cpu"))).fit(
                emb_all[lab_idx], y_lab_id)
            h_probs = head.predict_proba(emb_all)
            if head_calib:
                # R16 诊断: 头路概率锚点侧温度再校准（只用锚点行的 margin 分布，GT 无关）
                logp = np.log(np.clip(h_probs[anchor_idx], 1e-12, 1.0))
                T = fit_temperature(logp, target_median=calib_target)
                h_probs = softmax_temperature(np.log(np.clip(h_probs, 1e-12, 1.0)), T)
            h_top, h_margin = prob_margins(h_probs)
            h_pred = np.array([class_names[i] for i in h_top])
        else:
            # B-1-off 行: 原型路自迭代（无头），κ 沿用 W8 原始余弦 margin 口径
            h_probs, h_pred = None, None

        # 原型重估（Algorithm 1 第 6 步）+ 第二意见分配
        cl_r = PrototypeClusterer(mode="class_mean", seed=run_seed).fit(
            emb_all[labeled_mask],
            np.array([train_label_str[i] for i in lab_idx], dtype=object))
        p_pred, p_margin = _proto_path(cl_r, emb_all, calib_on, calib_target, anchor_idx)

        # 门控（GT 无关）
        keep = np.ones(N, dtype=bool)
        if standing_mode == "consensus" and calib_on:
            keep &= consensus_gate(h_pred, p_pred, STANDING_LABEL)
        elif standing_mode == "subcluster" and calib_on:
            cand = np.where(h_pred == STANDING_LABEL)[0]
            if len(cand):
                g = subcluster_gate(emb_all[cand], h_probs[cand],
                                    k=subcluster_k, min_share=subcluster_min_share,
                                    seed=run_seed)
                gate_full = np.ones(N, dtype=bool)
                gate_full[cand] = g
                keep &= gate_full
        elif standing_mode != "none":
            pass  # 原型路自迭代行无双路可共识——治理门自然失效，如实记录于报告

        # κ 口径: 校准开 → 头路概率 margin（天然 [0,1]）; 关 → 原型路原始余弦 margin
        kappa_r = h_margin if calib_on else p_margin
        assigner_pred = h_pred if calib_on else p_pred

        priors_r = label_priors(np.asarray([train_label_str[i] for i in lab_idx]))
        thr_r = frequency_aware_thresholds(tau_star, priors_r, alpha=alpha)
        pool_new = (eligible & keep &
                    (kappa_r >= np.array([thr_r.get(str(l), tau_star)
                                          for l in assigner_pred])))

        change_rate = float(np.mean(assigner_pred != prev_pred_full))
        _record(assigner_pred, kappa_r, pool_new, f"r{rounds_done + 1}_selftrain",
                {"change_rate": round(change_rate, 4)})
        rounds_done += 1
        prev_pred_full = assigner_pred
        pool_mask = pool_new
        labeled_mask = anchor_mask | pool_new
        train_label_str[pool_new] = assigner_pred[pool_new]

        stop_reason = iteration_stop_decision(
            rounds_done, change_rate, precision_history,
            max_iters, converge_change_rate, precision_drop_patience)[1]
        if stop_reason:
            break
    if stop_reason is None:
        stop_reason = "budget"

    return {
        "rounds": rounds,
        "tau_operating": rounds[0]["tau_operating"],
        "stop_reason": stop_reason,
        "final_pool_idx": np.where(pool_mask)[0].astype(np.int64),
        "final_pred_full": prev_pred_full,          # 导出池用（装配层消费）
        "final_kappa_full": kappa_r,                # 同上（最终轮 κ 口径）
    }


def _tau_curve_over_grid(tau_grid, pred, kappa, priors, alpha, eligible):
    """τ 全网格 (coverage) 曲线——仅预测侧统计，GT 无关。"""
    out = []
    for tau in sorted(float(t) for t in tau_grid):
        mult = frequency_aware_thresholds(max(tau, 1e-12), priors, alpha=alpha) \
            if tau > 0 else {k: 0.0 for k in priors}
        thr = np.array([mult.get(str(l), tau) for l in pred])
        mask = eligible & (kappa >= thr)
        out.append({"tau": round(tau, 4),
                    "coverage": round(float(mask.sum()) / max(eligible.sum(), 1), 4)})
    return out
