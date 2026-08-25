"""W33 — NTU60 xsub 三流融合 CLI（3s-AimCLR ensemble）。

移植自 external/AimCLR/ensemble_ntu_cs.py（数学内核在 psd/data/ntu_ensemble.py，
按 sample_name 键对齐 + fail-fast 加固）。官方 alpha: joint=0.6, bone=0.6, motion=0.4。

用法（三流 linear_eval 全部完成后执行）:
    python scripts/ntu_ensemble_3s.py \
        --joint runs/ntu_phaseB/lineareval_joint/test_result.pkl \
        --bone runs/ntu_phaseB/lineareval_bone/test_result.pkl \
        --motion runs/ntu_phaseB/lineareval_motion/test_result.pkl \
        --json-out reports/ntu-phaseB-3s-ensemble.json
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psd.data.ntu_ensemble import (  # noqa: E402
    DEFAULT_ALPHA,
    collect_stream_result,
    run_ensemble,
)

PROTOCOL_NOTE = (
    "3s-AimCLR linear-eval score ensemble (alpha joint0.6/bone0.6/motion0.4; "
    "port of external/AimCLR/ensemble_ntu_cs.py with key-aligned hardening)"
)


def assemble_output(result: dict, streams_dir: str | Path) -> dict:
    """合并融合成绩与各单流收集信息，形成可归档 JSON 结构。"""
    streams_dir = Path(streams_dir)
    per_stream = {}
    for stream in ("joint", "bone", "motion"):
        info = collect_stream_result(streams_dir / f"lineareval_{stream}")
        per_stream[stream] = {
            "best_top1": info["best_top1"],
            "last_top1": info["last_top1"],
            "work_dir": info["work_dir"],
        }
    return {
        "protocol": PROTOCOL_NOTE,
        "top1": result["top1"],
        "top5": result["top5"],
        "n": result["n"],
        "alpha": result["alpha"],
        "stream_paths": result.get("stream_paths", {}),
        "per_stream": per_stream,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--joint", default="runs/ntu_phaseB/lineareval_joint/test_result.pkl")
    ap.add_argument("--bone", default="runs/ntu_phaseB/lineareval_bone/test_result.pkl")
    ap.add_argument("--motion", default="runs/ntu_phaseB/lineareval_motion/test_result.pkl")
    ap.add_argument("--label-pkl", default="data/ntu60_frame50/xsub/val_label.pkl")
    ap.add_argument("--alpha-joint", type=float, default=DEFAULT_ALPHA["joint"])
    ap.add_argument("--alpha-bone", type=float, default=DEFAULT_ALPHA["bone"])
    ap.add_argument("--alpha-motion", type=float, default=DEFAULT_ALPHA["motion"])
    ap.add_argument("--streams-dir", default="runs/ntu_phaseB",
                    help="各流 work_dir 所在目录（收集 log.txt 内 best_top1 用）")
    ap.add_argument("--json-out", default=None, help="归档 JSON 路径（可选）")
    args = ap.parse_args()

    alpha = {"joint": args.alpha_joint, "bone": args.alpha_bone, "motion": args.alpha_motion}
    result = run_ensemble(
        {"joint": args.joint, "bone": args.bone, "motion": args.motion},
        args.label_pkl,
        alpha=alpha,
    )
    out = assemble_output(result, args.streams_dir)

    print("=" * 56)
    print("NTU60 xsub 3s-AimCLR 三流融合 (linear eval)")
    print(f"  alpha: {out['alpha']}")
    for s in ("joint", "bone", "motion"):
        b = out["per_stream"][s]["best_top1"]
        print(f"  {s:>6}: best_top1={b if b is not None else 'N/A'}")
    print(f"  Top1: {out['top1'] * 100:.2f}%   Top5: {out['top5'] * 100:.2f}%   (n={out['n']})")
    print("=" * 56)

    if args.json_out:
        dest = REPO_ROOT / args.json_out if not Path(args.json_out).is_absolute() else Path(args.json_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ntu-ensemble] JSON 已归档: {dest}")


if __name__ == "__main__":
    main()
