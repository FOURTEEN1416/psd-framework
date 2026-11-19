# -*- coding: utf-8 -*-
"""E10-T4 — 预算再分配: 把标注预算的一部分花在编码器适配（LoRA-CE 微调）上。

背景: E10 三线证明零额外标注的编码器适配全部无效（T1/T2/T2b/T3 全 FAILS），
k-ladder 指出真正能驱动编码器的是标签。本实验检验最后一个未试杠杆——
**把同一批 10% 预算锚点的一部分（或全部）同时用于编码器微调**：

  ref5   : 冻结编码器 + 5% 语义层（5% 预算的冻结参照，run_arms 统一尺）
  same10 : 每个种子的 10% 锚点**既**微调编码器**又**跑语义层（总预算仍 10%，
           检验"同一批标签再榨一层"是否增益）
  split5 : 每个种子的 5% 锚点微调编码器 + 同一 5% 锚点跑语义层（总预算 5%，
           与 ref5 同预算对照——检验"预算切给物理层"是否胜过全给语义层）

逐种子耦合设计（预算诚实）: 每个种子用**该种子自己的锚集**微调编码器
（CE + 冻结源蒸馏锚，LoRA 低秩 + 新线性头，BN running stats 冻结保持口径一致），
重导特征后用**同一锚集**跑 (b) 语义臂，并算该种子全量线性 ref → 保留率。
微调只在锚点上监督，绝不触碰池/无标签监督（那是 T3 的路线，已证伪）。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

from adapt_common import (REPO, adapt_out_dir, cosine_distill, count_params,
                          extract_backbone, inject_lora, load_frozen_model,
                          lora_parameters)


def build_anchor(labels_str: np.ndarray, seed: int, budget_frac: float) -> np.ndarray:
    """与 run_arms 逐式一致的分层锚点构造（同 seed 同锚集）。"""
    anchor = np.zeros(len(labels_str), dtype=bool)
    rng = np.random.default_rng(seed)
    for c in sorted(set(labels_str.tolist())):
        ci = np.where(labels_str == c)[0]
        k = max(1, int(round(len(ci) * budget_frac)))
        anchor[rng.choice(ci, size=k, replace=False)] = True
    return anchor


def finetune_lora_ce(model, teacher, X_t, anchor_idx: np.ndarray, y_anchor: np.ndarray,
                     n_cls: int, device: str, epochs: int, batch: int, lr: float,
                     distill: float, aug_views: int = 1, feeder_args: dict | None = None):
    """LoRA + 新线性头在锚点上 CE 微调；BN running stats 冻结（eval 口径一致）。

    aug_views>1: 锚点经 Feeder_single 增广（shear 0.5 / temporal pad 6，与 pretext
    同族），每 epoch 采样 n*aug_views 个增广视图——小锚集（如 PanAf 40）数据饥饿的主干预。
    """
    head = torch.nn.Linear(256, n_cls).to(device)
    torch.nn.init.xavier_uniform_(head.weight)
    params = lora_parameters(model.encoder_q) + list(head.parameters())
    for p in model.encoder_q.parameters():
        p.requires_grad = id(p) in {id(x) for x in lora_parameters(model.encoder_q)}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    model.eval()  # BN 用 pretext running stats，与后续特征导出口径一致
    x_all = X_t[anchor_idx].float().to(device)
    y_all = torch.from_numpy(y_anchor.astype(np.int64)).to(device)
    n = len(x_all)
    eff_batch = min(batch, n)  # 小 tier 锚点可能少于 batch（如 PanAf 40 个）——否则零迭代
    rng = np.random.default_rng(42)
    losses, accs = [], []
    if aug_views > 1:
        from feeder.ntu_feeder import Feeder_single  # noqa: E402  (external/AimCLR)
        feeder = Feeder_single(feeder_args["data_path"], feeder_args["label_path"],
                               shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True)
        steps = max(1, (n * aug_views) // batch)
        for ep in range(1, epochs + 1):
            ep_l = []
            for _ in range(steps):
                pick = rng.choice(n, size=batch, replace=True)
                xs = torch.stack([
                    torch.from_numpy(np.asarray(feeder[int(anchor_idx[p])][0])).float()
                    for p in pick]).to(device)
                ys = torch.from_numpy(y_anchor[pick].astype(np.int64)).to(device)
                f_s = extract_backbone(model.encoder_q, xs)
                with torch.no_grad():
                    f_t = extract_backbone(teacher, xs)
                loss = F.cross_entropy(head(f_s), ys)
                if distill > 0:
                    loss = loss + distill * cosine_distill(f_s, f_t)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_l.append(float(loss.item()))
        losses.append(round(float(np.mean(ep_l)), 4) if ep_l else None)
        with torch.no_grad():
            acc = float((head(extract_backbone(model.encoder_q, x_all)).argmax(1)
                         == y_all).float().mean())
        accs.append(round(acc, 4))
        if ep % max(1, epochs // 5) == 0 or ep == epochs:
            print(f"  [ft] epoch {ep}/{epochs} loss={losses[-1]} anchor_acc={acc:.3f}",
                  flush=True)
    else:
        for ep in range(1, epochs + 1):
            perm = rng.permutation(n)
            ep_l = []
            for i in range(0, n - n % eff_batch, eff_batch):
                idx = perm[i:i + eff_batch]
                x = x_all[idx]
                y = y_all[idx]
                f_s = extract_backbone(model.encoder_q, x)
                with torch.no_grad():
                    f_t = extract_backbone(teacher, x)
                logits = head(f_s)
                loss = F.cross_entropy(logits, y)
                if distill > 0:
                    loss = loss + distill * cosine_distill(f_s, f_t)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                opt.step()
                ep_l.append(float(loss.item()))
            with torch.no_grad():
                acc = float((head(extract_backbone(model.encoder_q, x_all)).argmax(1)
                             == y_all).float().mean())
            losses.append(round(float(np.mean(ep_l)), 4) if ep_l else None)
            accs.append(round(acc, 4))
            if ep % max(1, epochs // 5) == 0 or ep == epochs:
                print(f"  [ft] epoch {ep}/{epochs} loss={losses[-1]} anchor_acc={acc:.3f}",
                      flush=True)
    return model, head, losses, accs


def run_semantic_arm(tr_feat, va_feat, tr_lab, va_lab, anchor, seed, budget_frac):
    """与 run_arms 的 (b) 臂逐式一致的单种子语义评测。"""
    from run_p5b_generic_retention import KW
    from psd.training.tcl_selftrain import run_selftrain
    from labelfree_tier_replication import ScaledLR
    labels_str = np.array([str(l) for l in tr_lab])
    full_labels_str = np.array([str(l) for l in np.concatenate([tr_lab, va_lab])])
    class_names = [str(c) for c in sorted(set(tr_lab.tolist()))]
    universe = np.ones(len(tr_lab), dtype=bool)
    universe[anchor] = False
    r = run_selftrain(tr_feat, labels_str, anchor, run_seed=seed,
                      class_names=class_names,
                      head_cfg={"hidden_dim": 64, "epochs": 100, "lr": 1e-3,
                                "weight_decay": 1e-4, "batch_size": 128, "device": "cpu"},
                      pool_universe_mask=universe, **KW)
    tm2 = anchor.copy()
    tm2[r["final_pool_idx"]] = True
    y2 = np.array([None] * len(labels_str), dtype=object)
    y2[anchor] = labels_str[anchor]
    y2[r["final_pool_idx"]] = r["final_pred_full"][r["final_pool_idx"]]
    va_lab_str = full_labels_str[len(tr_lab):]
    acc_b = float(np.mean(ScaledLR()
                          .fit(tr_feat[tm2], y2[tm2]).predict(va_feat) == va_lab_str))
    return acc_b, len(r["final_pool_idx"]), r["stop_reason"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", default="ntu120")
    ap.add_argument("--arm", required=True,
                    choices=["ref5", "same10", "split5", "ref15", "dual15"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--distill", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--aug-views", type=int, default=1,
                    help="锚点增广视图倍数（>1 时经 Feeder_single shear0.5/pad6 采样）")
    ap.add_argument("--tag", default=None,
                    help="输出文件名标签（默认=arm 名；扫参时必须显式给，防覆盖）")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed-list", type=int, nargs="+", default=None,
                    help="显式种子列表（断点续跑单种子用；配合 summary_<tag>.json 已有 per_seed 跳过）")
    args = ap.parse_args()
    tier = args.tier
    tag = args.tag or args.arm
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.smoke:
        args.seeds, args.epochs = 1, 5
    seeds = list(args.seed_list) if args.seed_list else list(range(42, 42 + args.seeds))
    budget = {"ref5": 0.05, "same10": 0.10, "split5": 0.05,
              "ref15": 0.15, "dual15": 0.15}[args.arm]
    # 断点续跑: 读取已有 summary 的 per_seed 行，跳过已完成种子（前台分片执行用）
    out_path = adapt_out_dir("t4", tier) / f"summary_{tag}.json"
    done_rows: dict = {}
    if out_path.exists():
        try:
            for r in json.loads(out_path.read_text(encoding="utf-8")).get("per_seed", []):
                if r.get("ft_anchor_acc") is not None:
                    done_rows[int(r["seed"])] = r
        except (json.JSONDecodeError, KeyError, TypeError):
            done_rows = {}
    seeds = [s for s in seeds if s not in done_rows]
    if done_rows:
        print(f"[resume] {len(done_rows)} seeds already done: {sorted(done_rows)}", flush=True)
    t0 = time.time()

    from labelfree_tier_replication import dump_features
    from labelfree_tier_replication import ScaledLR

    if args.arm in ("ref5", "ref15"):
        model, ckpt = load_frozen_model(tier, device)
        from labelfree_tier_replication import run_arms
        arms = run_arms(model, tier, tuple(seeds), device, budget_frac=budget)
        result = {
            "doneMarker": f"P1_ADAPT_T4_{args.arm.upper()}_{tier.upper()}_DONE",
            "experiment": "e10_t4_budget_split", "arm": args.arm, "tier": tier,
            "budget_frac": budget, "date": datetime.now().isoformat(timespec="seconds"),
            "full_ref": round(arms["full_ref"], 4),
            "linear_10pct_note": "arm (a) inside run_arms used budget_frac too",
            "b_arms": [round(a, 4) for a in arms["b_arms"]],
            "retention": round(arms["retention"], 4), "verdict": arms["verdict"],
            "seeds": seeds,
            "wall_clock_sec": round(time.time() - t0, 1),
        }
        out = adapt_out_dir("t4", tier) / f"summary_{tag}.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[done] {out}")
        print(json.dumps({k: result[k] for k in ("retention", "verdict")},
                         ensure_ascii=False, indent=1))
        return

    # same10 / split5: 逐种子耦合（微调 → 导特征 → ref → 同锚语义臂）
    # 标签一次取自 feeder pkl（pkl=(names, labels)，labels 与 dump_features 同序，
    # 首种子断言校验）
    import pickle
    _pkl = pickle.load(open(REPO / "runs" / "p1_sprint" / f"labelfree_{tier}"
                            / "feeder" / "train_label.pkl", "rb"))
    pkl_labels = np.asarray(_pkl[1]).reshape(-1)
    X = np.array(np.load(REPO / "runs" / "p1_sprint" / f"labelfree_{tier}"
                         / "feeder" / "train_position.npy", mmap_mode="r"))
    X_t = torch.from_numpy(X)
    rows = []
    done_rows_list = list(done_rows.values())
    teacher = None
    labels_checked = False
    for seed in seeds:
        print(f"[{args.arm}] seed {seed}", flush=True)
        model, ckpt = load_frozen_model(tier, device)
        if not labels_checked:
            store0 = dump_features(model, tier, device)
            assert np.array_equal(np.asarray(store0["train_label"]), pkl_labels), \
                "pkl 标签序与 dump_features 不一致——锚集构造将失真"
            labels_checked = True
        tr_lab0 = pkl_labels
        labels_str0 = np.array([str(l) for l in tr_lab0])
        anchor = build_anchor(labels_str0, seed, budget)
        anchor_idx = np.where(anchor)[0]
        y_anchor = tr_lab0[anchor]
        if teacher is None:
            teacher = copy.deepcopy(model.encoder_q)
            for p in teacher.parameters():
                p.requires_grad_(False)
            teacher.eval()
        inject_lora(model.encoder_q, r=args.rank)
        model = model.to(device)
        fd = REPO / "runs" / "p1_sprint" / f"labelfree_{tier}" / "feeder"
        model, head, losses, accs = finetune_lora_ce(
            model, teacher, X_t, anchor_idx, y_anchor, len(set(tr_lab0.tolist())),
            device, args.epochs, args.batch, args.lr, args.distill,
            aug_views=args.aug_views,
            feeder_args={"data_path": str(fd / "train_position.npy"),
                         "label_path": str(fd / "train_label.pkl")})
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        store = dump_features(model, tier, device)
        tr_feat, tr_lab = store["train_feat"], store["train_label"]
        va_feat, va_lab = store["val_feat"], store["val_label"]
        ref_j = float(np.mean(ScaledLR().fit(tr_feat, tr_lab).predict(va_feat) == va_lab))
        # 锚集按标签串域重建（与 run_arms 完全同式，保证同种子同锚）
        labels_str = np.array([str(l) for l in tr_lab])
        anchor2 = build_anchor(labels_str, seed, budget)
        acc_b, pool, stop = run_semantic_arm(tr_feat, va_feat, tr_lab, va_lab,
                                             anchor2, seed, budget)
        rows.append({"seed": seed, "full_ref": round(ref_j, 4), "b_arm": round(acc_b, 4),
                     "retention_seed": round(acc_b / ref_j, 4) if ref_j > 0 else 0.0,
                     "pool": pool, "stop": stop,
                     "ft_anchor_acc": accs[-1], "ft_loss_tail": losses[-3:]})
        print(f"[{args.arm}] seed{seed}: ref={ref_j:.4f} b={acc_b:.4f} "
              f"ret={rows[-1]['retention_seed']:.4f} pool={pool}", flush=True)
        # 每种子落盘部分结果（前台分片执行时防丢）
        all_rows = sorted(done_rows_list + rows, key=lambda r: r["seed"])
        _mb = round(float(np.mean([r["b_arm"] for r in all_rows])), 4)
        _mr = round(float(np.mean([r["full_ref"] for r in all_rows])), 4)
        _rt = round(_mb / _mr, 4) if _mr > 0 else 0.0
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "experiment": "e10_t4_budget_split", "arm": args.arm, "tier": tier,
            "budget_frac": budget, "per_seed": all_rows, "mean_b": _mb,
            "mean_full_ref": _mr, "retention": _rt, "partial": True,
            "seeds": sorted(r["seed"] for r in all_rows),
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    all_rows = sorted(done_rows_list + rows, key=lambda r: r["seed"])
    mean_b = round(float(np.mean([r["b_arm"] for r in all_rows])), 4)
    mean_ref = round(float(np.mean([r["full_ref"] for r in all_rows])), 4)
    ret = round(mean_b / mean_ref, 4) if mean_ref > 0 else 0.0
    verdict = "CONFIRMS" if ret >= 0.90 else ("PARTIAL" if ret >= 0.85 else "FAILS")
    result = {
        "doneMarker": f"P1_ADAPT_T4_{args.arm.upper()}_{tier.upper()}_DONE",
        "experiment": "e10_t4_budget_split", "arm": args.arm, "tier": tier,
        "mode": "smoke" if args.smoke else "full",
        "budget_frac": budget,
        "protocol": "per-seed coupled: LoRA-CE fine-tune on THIS seed's anchors "
                    "(+cosine distill anchor, BN running stats frozen), then re-dump "
                    "features, full-label linear ref, and the SAME-anchor (b) semantic arm",
        "date": datetime.now().isoformat(timespec="seconds"),
        "train": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                  "distill_lambda": args.distill, "rank": args.rank,
                  "aug_views": args.aug_views},
        "per_seed": all_rows, "mean_b": mean_b, "mean_full_ref": mean_ref,
        "retention": ret, "verdict": verdict, "seeds": sorted(r["seed"] for r in all_rows),
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = adapt_out_dir("t4", tier) / f"summary_{tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps({"mean_b": mean_b, "mean_full_ref": mean_ref,
                      "retention": ret, "verdict": verdict}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
