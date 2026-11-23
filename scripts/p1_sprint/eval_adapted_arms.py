# -*- coding: utf-8 -*-
"""eval_adapted_arms — 加载已适配的 encoder ckpt，用统一尺重测 E9 三臂。

用于: 适配训练已落盘（如 adapt_lora_<tier>/lora_epoch30_model.pt）后，
单独补跑/重跑三臂评测（与已发布数字同一 run_arms 实现）。
自动识别 ckpt 是否含 LoRA 键（含则先注入同参 LoRA 再 strict 加载）。
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import torch

from adapt_common import REPO, adapt_out_dir, count_params, inject_lora, load_frozen_model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tag", default="eval", help="输出文件名标签")
    ap.add_argument("--source-ckpt", default=None, help="仅用于汇总里标注来源")
    args = ap.parse_args()
    tier = args.tier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = tuple(range(42, 42 + args.seeds))
    t0 = time.time()

    ckpt = Path(args.ckpt)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    has_lora = any("lora_A" in k for k in sd.keys())
    if has_lora:
        from labelfree_tier_replication import TIERS, build_model, channels_for
        model = build_model(TIERS[tier], "cpu", channels_for(tier))
        inject_lora(model.encoder_q, r=args.rank, alpha=args.alpha)
        inject_lora(model.encoder_k, r=args.rank, alpha=args.alpha)
        model.load_state_dict(sd, strict=True)
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
    else:
        model, _ = load_frozen_model(tier, device, ckpt_path=ckpt)
    tr, tt = count_params(model)
    print(f"[eval] ckpt={ckpt.name} lora={has_lora} trainable={tr:,}/{tt:,}", flush=True)

    from labelfree_tier_replication import run_arms
    arms = run_arms(model, tier, seeds, device)

    result = {
        "doneMarker": f"P1_ADAPT_EVAL_{tier.upper()}_DONE",
        "experiment": "e10_eval_adapted_arms",
        "tier": tier,
        "date": datetime.now().isoformat(timespec="seconds"),
        "ckpt": str(ckpt.relative_to(REPO)) if ckpt.is_relative_to(REPO) else str(ckpt),
        "source_ckpt": args.source_ckpt,
        "lora_loaded": bool(has_lora),
        "full_ref": round(arms["full_ref"], 4),
        "linear_10pct": round(arms["linear_10pct"], 4),
        "b_arms": [round(a, 4) for a in arms["b_arms"]],
        "retention": round(arms["retention"], 4),
        "verdict": arms["verdict"],
        "seeds": list(seeds),
        "wall_clock_sec": round(time.time() - t0, 1),
    }
    out = adapt_out_dir("eval", tier) / f"summary_{args.tag}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {out}")
    print(json.dumps({k: result[k] for k in ("full_ref", "linear_10pct",
                                             "retention", "verdict")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
