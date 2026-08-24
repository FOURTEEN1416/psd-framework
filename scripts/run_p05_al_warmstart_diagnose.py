"""P0.5-AL warm-start 协议诊断 — W23 窗口（任务书 Step 1）.

目的: best.pt 在「加噪偏移」合成域上的零微调基线体检，为 warm-start 协议选噪声档。
执行序地位: 本脚本先于一切正式扫描代码运行（任务书 §1 Step 1 是 Step 2 的门）。

协议（W23 任务书预写，未事后修改）:
    - 偏移数据: 池 seed=20263 / 验证 seed=20264（均避开 W14 的 20261/20262 与 W12 seed42）
      spc 与 W14 相同: 池 10/类(220)、验证 15/类(330)；noise_std ∈ {0.10, 0.15, 0.20}
    - best.pt(runs/p05_stgcn_bc_full/best.pt) 零微调直接评估:
      记录各档 val_acc 与 logit top1−top2 边际（val/pool 两侧都记——熵打分作用于池，
      池侧饱和同样致命，判据从严双查）
    - 选档判据: 基线 val_acc ∈ [40%, 70%] 且未饱和（边际均值 < 10，val/pool 双侧满足）
    - 熔断判据（预写）: 所有档位基线 >85%（无适应空间）或全部饱和 → 实验不可行，
      停手留证上报用户重选方案，不得自行换招

用法:
    .venv\\Scripts\\python.exe scripts\\run_p05_al_warmstart_diagnose.py

输出:
    reports/p05-al-efficiency-warmstart-diagnosis.json + 控制台表格
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.synth_stgcn import make_synthetic_dataset  # noqa: E402
from psd.models.stgcn_bc import build_stgcn_bc  # noqa: E402
from psd.training.active_learning import predict_logits  # noqa: E402

CKPT = REPO_ROOT / "runs/p05_stgcn_bc_full/best.pt"
OUT_JSON = REPO_ROOT / "reports/p05-al-efficiency-warmstart-diagnosis.json"

NOISE_LEVELS = [0.10, 0.15, 0.20]
POOL_SEED = 20263
VAL_SEED = 20264
POOL_SPC = 10   # 与 W14 相同
VAL_SPC = 15    # 与 W14 相同
T = 30

MODEL_CFG = dict(in_channels=3, num_classes=22, base_channels=64, num_stages=10)

# 判据常量（任务书预写）
ACC_LO, ACC_HI = 0.40, 0.70
MARGIN_MAX = 10.0
FUSE_ACC_HI = 0.85


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT), text=True,
        ).strip()
    except Exception:
        return "unknown"


def margin_stats(logits: np.ndarray) -> dict:
    """logit top1−top2 边际统计."""
    top2 = np.sort(logits, axis=-1)[:, ::-1]
    m = top2[:, 0] - top2[:, 1]
    return {
        "mean": float(m.mean()),
        "min": float(m.min()),
        "max": float(m.max()),
    }


def zero_shot_eval(model, samples, device: str) -> dict:
    """零微调评估: 返回 {val_acc, logit_margin_stats}."""
    logits = predict_logits(model, [{"keypoints": s["keypoints"]} for s in samples],
                            device=device)
    labels = np.array([s["label"] for s in samples])
    acc = float((logits.argmax(axis=-1) == labels).mean())
    return {"n": len(samples), "val_acc": acc, "logit_margin_stats": margin_stats(logits)}


def main() -> None:
    t0 = time.time()
    assert CKPT.exists(), f"checkpoint 不存在: {CKPT}"
    device = "cpu"  # GPU 被 NTU 长训占用（W18 巡检中），显式 CPU 不抢卡

    model = build_stgcn_bc(**MODEL_CFG)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[ckpt] loaded {CKPT.name} (epoch={ckpt.get('epoch')}, "
          f"best_val_acc={ckpt.get('best_val_acc')}) -> device={device}")

    rows = []
    for noise in NOISE_LEVELS:
        pool = make_synthetic_dataset(samples_per_class=POOL_SPC, T=T,
                                      noise_std=noise, seed=POOL_SEED)
        val = make_synthetic_dataset(samples_per_class=VAL_SPC, T=T,
                                     noise_std=noise, seed=VAL_SEED)
        r_val = zero_shot_eval(model, val, device)
        r_pool = zero_shot_eval(model, pool, device)
        row = {
            "noise_std": noise,
            "val": r_val,
            "pool": r_pool,
            "acc_in_band": bool(ACC_LO <= r_val["val_acc"] <= ACC_HI),
            "unsaturated_val": bool(r_val["logit_margin_stats"]["mean"] < MARGIN_MAX),
            "unsaturated_pool": bool(r_pool["logit_margin_stats"]["mean"] < MARGIN_MAX),
        }
        row["eligible"] = bool(row["acc_in_band"] and row["unsaturated_val"]
                               and row["unsaturated_pool"])
        rows.append(row)
        print(f"[diag] noise={noise:.2f} | val_acc={r_val['val_acc']:.4f} "
              f"margin(val)={r_val['logit_margin_stats']['mean']:.2f} "
              f"margin(pool)={r_pool['logit_margin_stats']['mean']:.2f} "
              f"| band={row['acc_in_band']} unsat={row['unsaturated_val']}/{row['unsaturated_pool']} "
              f"-> eligible={row['eligible']}")

    # ---- 选档 / 熔断判定（任务书预写判据） --------------------------------
    eligible = [r["noise_std"] for r in rows if r["eligible"]]
    all_above_85 = all(r["val"]["val_acc"] > FUSE_ACC_HI for r in rows)
    all_saturated = all(not (r["unsaturated_val"] or r["unsaturated_pool"]) for r in rows)

    if all_above_85 or all_saturated:
        verdict = {
            "status": "FUSE",
            "reason": ("所有档位基线 >85%（无适应空间）" if all_above_85
                       else "所有档位 logit 边际饱和（熵信号退化）"),
            "selected_noise_std": None,
        }
    elif len(eligible) == 1:
        verdict = {"status": "SELECT", "selected_noise_std": eligible[0],
                   "reason": "唯一满足选档判据的档位"}
    elif len(eligible) == 0:
        verdict = {"status": "NO_CANDIDATE", "selected_noise_std": None,
                   "reason": "无档位同时满足 40~70% 带宽与未饱和判据；"
                             "取最接近带宽数值的档位上报用户裁决，不自行定档"}
        # 上报辅助: 距 [0.40, 0.70] 带中心最近的档位
        center = (ACC_LO + ACC_HI) / 2
        nearest = min(rows, key=lambda r: abs(r["val"]["val_acc"] - center))
        verdict["nearest_to_band_center"] = {
            "noise_std": nearest["noise_std"],
            "val_acc": nearest["val"]["val_acc"],
        }
    else:
        # 多档合格: 取最靠近带宽中心者（适应空间与学习信号折中），理由入预注册
        center = (ACC_LO + ACC_HI) / 2
        pick = min(eligible, key=lambda n: abs(
            next(r for r in rows if r["noise_std"] == n)["val"]["val_acc"] - center))
        verdict = {"status": "SELECT_MULTI",
                   "selected_noise_std": pick,
                   "candidates": eligible,
                   "reason": f"多档合格，按距带宽中心({center:.0%})最近选 {pick}"}

    report = {
        "meta": {
            "date": datetime.now().isoformat(timespec="seconds"),
            "window": "W23",
            "step": "Step1 diagnosis (task book v1.0)",
            "git_sha": git_sha(),
            "device": device,
            "ckpt": str(CKPT.relative_to(REPO_ROOT)),
            "ckpt_sha256_note": "哈希未记录则说明文件在诊断后未被复算——本字段由下方 sha256 提供",
            "model_cfg": MODEL_CFG,
            "data_gen": {
                "pool": {"samples_per_class": POOL_SPC, "T": T, "seed": POOL_SEED},
                "val": {"samples_per_class": VAL_SPC, "T": T, "seed": VAL_SEED},
                "noise_levels": NOISE_LEVELS,
            },
            "criteria": {
                "select": f"val_acc ∈ [{ACC_LO}, {ACC_HI}] 且 margin_mean(val,pool) < {MARGIN_MAX}",
                "fuse": f"全档 val_acc > {FUSE_ACC_HI} 或全档饱和",
            },
            "total_time_sec": round(time.time() - t0, 1),
        },
        "rows": rows,
        "verdict": verdict,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n=== W23 Step1 诊断结论 ===")
    print(f"status={verdict['status']} selected_noise={verdict.get('selected_noise_std')}")
    print(f"reason: {verdict['reason']}")
    print(f"[output] JSON -> {OUT_JSON}")

    if verdict["status"] == "FUSE":
        print("!! 熔断触发：实验不可行，停手留证上报用户重选方案（不得自行换招）")
        sys.exit(2)


if __name__ == "__main__":
    main()
