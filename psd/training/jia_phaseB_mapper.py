"""P0.3 Phase B 映射器 — W13 窗口 owner（原型聚类结果 → ADR-0002 裁决② 22 类体系）。

职责（W13 任务书）：
把 P0.3 Phase A 的原型聚类产物（只读复用，见 psd.training.jia_prototype）与
ADR-0002 裁决②路径 a 锁定的 22 类体系（assets-map.md §1 单一 truth，经
psd.data.synth_stgcn.ALL_BEHAVIORS_22 引用）对接，产出带 22 类语义标签的
伪标签池供 P0.5 消费。

映射机制（设计备案，报告 §2 详述）：
- 语义桥：合成样本经同一冻结 Φ 编码后按类取均值 → 22 个语义原型；
  任一段嵌入按余弦最近语义原型获得 22 类伪标签。
- 为什么不是「Phase A 原型→单一 22 类标签」的硬桥接：Phase A 原型 ≤7 个，
  单标签发射在 22 类均匀真值上的精度上界 ≈7/22≈32%，构造性低于完成标准
  0.50——故标签由 22 路最近语义原型直接产生；Phase A 的 proto_idx/kappa_margin
  作为置信元数据随池携带（与 P0.4 消费接口字段对齐），桥接映射矩阵作为
  「原型-类别对齐质量」诊断量输出（任务书风险提示的对应物）。

口径标注：伪标签池作用于公开真实层段（metric_layer=public_real_physics_prior）；
映射精度在合成层以留出协议度量（唯一存在 22 类真值的层），三层口径分开汇报。

纯 numpy、编码器注入式（无 torch 依赖），保证 CPU 可 TDD。
"""
from __future__ import annotations

import numpy as np

# P0.4 消费接口字段（data/processed/p04/pseudo_pool_*.jsonl 口径）
POOL_SCHEMA_FIELDS = frozenset({
    "clip_id", "start_frame", "end_frame", "pseudo_label", "proto_idx",
    "kappa_margin", "tau_pass", "embedding_ref", "label_source", "metric_layer",
})

_METRIC_LAYER_PUBLIC_REAL = "public_real_physics_prior"


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def _assert_finite(arr: np.ndarray, what: str) -> None:
    if not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
        raise ValueError(f"{what} 含非有限值——拒绝进入映射（NaN 防线）")


# ---------------------------------------------------------------- 语义原型

def build_semantic_prototypes(
    syn_emb: np.ndarray, syn_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """合成嵌入 + 合成真标签 → 22 类均值原型（L2 归一，字典序稳定输出）。

    返回 (protos (C,D), proto_labels (C,) str)。空输入或非有限值抛 ValueError。
    """
    emb = np.asarray(syn_emb, dtype=np.float64)
    labels = np.asarray(syn_labels)
    if emb.size == 0 or len(labels) == 0:
        raise ValueError("合成嵌入为空——无法构建语义原型")
    _assert_finite(emb, "syn_emb")
    if len(emb) != len(labels):
        raise ValueError(f"嵌入数 {len(emb)} 与标签数 {len(labels)} 不一致")
    feats = _l2_normalize(emb)
    classes = sorted(set(labels.tolist()))
    protos = np.vstack([feats[labels == c].mean(axis=0) for c in classes])
    return _l2_normalize(protos), np.array(classes)


def nearest_semantic_label(
    emb_vec: np.ndarray, semantic_protos: np.ndarray, semantic_names: np.ndarray,
) -> tuple[int, str, float]:
    """单嵌入 → 最近语义原型 (idx, 22类名, 余弦相似度)。"""
    sims = _l2_normalize(np.asarray(emb_vec).reshape(1, -1)) @ _l2_normalize(
        semantic_protos).T  # (1,C)
    idx = int(np.argmax(sims[0]))
    return idx, str(semantic_names[idx]), float(sims[0, idx])


def map_embeddings_to_22(
    emb: np.ndarray, semantic_protos: np.ndarray, semantic_names: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """批量映射：嵌入矩阵 → (22 类预测标签, top1 余弦相似度)。"""
    _assert_finite(emb, "emb")
    sims = _l2_normalize(emb) @ _l2_normalize(semantic_protos).T  # (N,C)
    top1 = sims.argmax(axis=1)
    pred = np.asarray([str(semantic_names[i]) for i in top1])
    return pred, sims[np.arange(len(sims)), top1]


# ---------------------------------------------------------------- 桥接诊断

def bridge_map_prototypes(
    pa_prototypes: np.ndarray, pa_proto_labels: np.ndarray,
    semantic_protos: np.ndarray, semantic_names: np.ndarray,
) -> tuple[dict[str, str], np.ndarray]:
    """诊断用桥接：每个 Phase A 原型 → 余弦最近的 22 类。

    返回 ({phaseA原型名: 22类名}, 相似度矩阵 (P,C))。仅作对齐质量诊断，
    不作为池标签来源（理由见模块 docstring 设计备案）。
    """
    sims = _l2_normalize(pa_prototypes) @ _l2_normalize(semantic_protos).T  # (P,C)
    mapping = {
        str(pa_proto_labels[i]): str(semantic_names[int(sims[i].argmax())])
        for i in range(len(pa_proto_labels))
    }
    return mapping, sims


# ---------------------------------------------------------------- 留出切分

def split_synthetic_ref_probe(
    labels: np.ndarray, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """合成集按类分层对半切（ref 建原型 / probe 验证映射），防 train=test 循环。

    返回 (ref_idx, probe_idx)，两类不相交并集为全量；每类奇数时 ref 多 1。
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    ref_parts: list[np.ndarray] = []
    probe_parts: list[np.ndarray] = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        perm = rng.permutation(len(idx))
        n_ref = (len(idx) + 1) // 2
        ref_parts.append(idx[perm[:n_ref]])
        probe_parts.append(idx[perm[n_ref:]])
    ref = np.sort(np.concatenate(ref_parts))
    probe = np.sort(np.concatenate(probe_parts))
    return ref, probe


# ---------------------------------------------------------------- 池行构造

def build_pool_rows(
    segments: list[dict],
    seg_emb: np.ndarray,
    proto_idx: np.ndarray,
    kappa: np.ndarray,
    semantic_protos: np.ndarray,
    semantic_names: np.ndarray,
    embedding_ref: str,
    label_source: str,
) -> list[dict]:
    """段列表 + 嵌入 + Phase A 分配元数据 → P0.4 接口对齐的池行列表。

    tau_pass 恒 True（τ=0 全覆盖口径，对应完成标准「覆盖率 ≥ Phase A coverage(α=1)=1.0」）。
    行 schema 严格等于 POOL_SCHEMA_FIELDS（与 P0.4 jsonl 消费接口逐字段对齐）。
    """
    assert len(segments) == len(seg_emb) == len(proto_idx) == len(kappa), (
        f"长度不一致: segments={len(segments)} emb={len(seg_emb)} "
        f"proto={len(proto_idx)} kappa={len(kappa)}")
    pred, sim = map_embeddings_to_22(seg_emb, semantic_protos, semantic_names)
    rows: list[dict] = []
    for i, s in enumerate(segments):
        rows.append({
            "clip_id": s["clip_id"],
            "start_frame": int(s["start_frame"]),
            "end_frame": int(s["end_frame"]),
            "pseudo_label": str(pred[i]),
            "proto_idx": int(proto_idx[i]),
            "kappa_margin": float(kappa[i]),
            "tau_pass": True,
            "embedding_ref": embedding_ref,
            "label_source": label_source,
            "metric_layer": _METRIC_LAYER_PUBLIC_REAL,
        })
    return rows


# ---------------------------------------------------------------- 指标

def coverage_ratio(tau_pass: np.ndarray) -> float:
    """覆盖率 = tau_pass 均值（空输入返回 0.0）。"""
    arr = np.asarray(tau_pass, dtype=bool)
    return float(arr.mean()) if arr.size else 0.0
