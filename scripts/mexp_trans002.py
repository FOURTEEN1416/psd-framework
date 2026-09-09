# -*- coding: utf-8 -*-
"""TRANS-002 MEXP: C'/M 配对扩容到 10 seeds (42-51, NaN 补位池 52-55)。

设计（docs/paper/ntu-transition-002-preregistration.md dated addendum 2026-09-09，先于任何新训练冻结）：
  - C'_s: lr 0.00625 全量微调 80ep（与冻结三 seed 完全同配方），净评估 top-1
  - M_s:  同一 checkpoint penultimate 特征 + LR 头（复用 run_r23_trans002 的 dump+头）
  - 产出: reports/r23-trans002-mexp-2026-09-09.json（逐对数值+Wilcoxon+sign test）
可恢复架构与 fixfull 相同: state 进度文件 + FATAL 甄别 + 单实例锁 + 软退出 exit3。
"""
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
LOG = REPO / "reports" / "trans002_mexp.log"
DONE = REPO / "reports" / "trans002_mexp_DONE.flag"
FATAL = REPO / "reports" / "trans002_mexp_FATAL.txt"
STATE = REPO / "reports" / "trans002_mexp_state.json"
LOCK = REPO / "reports" / "trans002_mexp_LOCK"
PYTHON = sys.executable
SEED_POOL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
NAN_POOL = [52, 53, 54, 55]


def log(msg):
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fatal(msg):
    log(f"FATAL: {msg}")
    FATAL.write_text(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n", encoding="utf-8")
    sys.exit(1)


def _lock_held_by_live_process():
    if not LOCK.exists():
        return False
    try:
        pid = int(LOCK.read_text().strip())
    except ValueError:
        return False
    import ctypes
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
    if h:
        ctypes.windll.kernel32.CloseHandle(h)
        return True
    return False


def load_state():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1, ensure_ascii=False), encoding="utf-8")


def write_cfg(seed, work_dir, epochs=80, lr=0.00625):
    cfg = f"""# TRANS-002 MEXP arm C' seed {seed} (frozen recipe identical to fixfull)
work_dir: {work_dir.as_posix()}

train_feeder: feeder.ntu_feeder.Feeder_single
train_feeder_args:
  data_path: data/ntu60_frame50/xsub/train_position.npy
  label_path: data/ntu60_frame50/xsub/train_label_yp49.pkl
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True

test_feeder: feeder.ntu_feeder.Feeder_single
test_feeder_args:
  data_path: data/ntu60_frame50/xsub/val_position.npy
  label_path: data/ntu60_frame50/xsub/val_label_yp49.pkl
  shear_amplitude: -1
  temperal_padding_ratio: -1
  mmap: True

model: net.st_gcn.Model
model_args:
  in_channels: 3
  hidden_channels: 16
  hidden_dim: 256
  num_class: 49
  dropout: 0.5
  graph_args:
    layout: 'ntu-rgb+d'
    strategy: 'spatial'
  edge_importance_weighting: True

nesterov: True
weight_decay: 0.0001
base_lr: {lr}
optimizer: SGD
device: [0]
batch_size: 16
test_batch_size: 64
num_epoch: {epochs}
num_worker: 0
save_interval: 5
eval_interval: 20
print_log: False
"""
    work_dir.mkdir(parents=True, exist_ok=True)
    p = work_dir / f"cfg_mexp_s{seed}.yaml"
    p.write_text(cfg, encoding="utf-8")
    return p


def _seed_dirs(seed):
    """冻结三 seed 在 c_fullfix_s*, 扩容 seed 在 mexp_s* —— 双前缀扫描。"""
    return list((REPO / "runs" / "trans002").glob(f"c_fullfix_s{seed}_*")) + \
           list((REPO / "runs" / "trans002").glob(f"mexp_s{seed}_*"))


def best_ckpt_for_seed(seed):
    import re
    cands = []
    for d in _seed_dirs(seed):
        for p in d.glob("epoch*_acc*_model.pt"):
            m = re.match(r"epoch(\d+)_acc([\d.]+)_model\.pt", p.name)
            if m:
                cands.append((float(m.group(2)), p))
    if cands:
        return max(cands, key=lambda t: (t[0], t[1].stat().st_mtime))[1]
    bs = [d / "best_model.pt" for d in _seed_dirs(seed) if (d / "best_model.pt").exists()]
    return sorted(bs, key=lambda p: p.stat().st_mtime)[-1] if bs else None


def resume_ckpt_for_seed(seed):
    import re
    pts = []
    for d in _seed_dirs(seed):
        for p in d.glob("epoch*_model.pt"):
            m = re.match(r"epoch(\d+)_model\.pt", p.name)
            if m:
                pts.append((int(m.group(1)), p, d))
                continue
            m = re.match(r"epoch(\d+)_acc[\d.]+_model\.pt", p.name)
            if m:
                pts.append((int(m.group(1)), p, d))
    if not pts:
        return None
    ep, ck, d = max(pts, key=lambda t: t[0])
    return ep, ck, d


def run_one_seed(seed, frozen_cfg_path=None):
    """跑单 seed C'（断点续训）。返回 (wall_total, acc, ckpt)。"""
    import re
    rs = resume_ckpt_for_seed(seed)
    if frozen_cfg_path is not None:
        cfg = Path(frozen_cfg_path)
        work_dir = cfg.parent
    elif rs is not None:
        ep, _ck, work_dir = rs
        cfg = write_cfg(seed, work_dir)
        log(f"  seed {seed} RESUME from epoch {ep} ({work_dir.name})")
    else:
        ts = datetime.now().strftime("%H%M%S")
        work_dir = REPO / "runs" / "trans002" / f"mexp_s{seed}_{ts}"
        cfg = write_cfg(seed, work_dir)
    cmd = [PYTHON, "-u", str(REPO / "scripts" / "run_r23_trans002_train_one.py"),
           "--config", str(cfg.resolve()), "--seed", str(seed),
           "--resume-dir", str(work_dir)]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, timeout=12 * 3600)
    wall = round(time.time() - t0, 1)
    if r.returncode != 0:
        raise RuntimeError(f"seed {seed} failed: {r.stdout[-300:]!r} {r.stderr[-300:]!r}")
    acc = float(r.stdout.decode().strip().splitlines()[-1])
    ck = best_ckpt_for_seed(seed)
    if ck is None:
        raise RuntimeError(f"seed {seed}: no checkpoint after training")
    return wall, acc, ck


def m_arm_from_ckpt(seed, ckpt, td):
    tr_npz = REPO / "runs" / "trans002" / f"mexp_train_feat_s{seed}.npz"
    va_npz = REPO / "runs" / "trans002" / f"mexp_val_feat_s{seed}.npz"
    t0 = time.time()
    td.dump_penultimate(ckpt, "train", "full", tr_npz)
    td.dump_penultimate(ckpt, "val", "full", va_npz)
    clf, sc = td.train_lr_head(tr_npz)
    acc = td.eval_lr(clf, sc, va_npz)
    return {"seed": seed, "val_acc": round(acc, 4), "wall_clock_sec": round(time.time() - t0, 1)}


def main():
    import random
    time.sleep(random.uniform(0.3, 1.5))
    for _ in range(5):
        if not LOCK.exists():
            try:
                fd = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                break
            except FileExistsError:
                time.sleep(1.0)
        elif _lock_held_by_live_process():
            log("another MEXP instance alive (lock); exiting")
            sys.exit(0)
        else:
            LOCK.unlink()
    else:
        log("lock acquire failed; exiting")
        sys.exit(0)

    if DONE.exists():
        DONE.unlink()
    if FATAL.exists():
        FATAL.unlink()

    st = load_state()
    st.setdefault("attempt", 0)
    st["attempt"] += 1
    st.setdefault("pairs", [])
    st.setdefault("nan_failed", [])
    save_state(st)
    log(f"MEXP pipeline started (attempt {st['attempt']}, pairs done: {len(st['pairs'])})")

    # 冻结的 42-44 检查点直接复用（cfg 已存在）
    frozen_cfg = {42: "runs/trans002/c_fullfix_s42_205407/cfg_c_fullfix_s42.yaml",
                  43: "runs/trans002/c_fullfix_s43_061239/cfg_c_fullfix_s43.yaml",
                  44: "runs/trans002/c_fullfix_s44_092543/cfg_c_fullfix_s44.yaml"}
    frozen_m = {42: 0.8283, 43: 0.8267, 44: 0.8225}

    pool = list(SEED_POOL)
    failed_now = []
    while len([p for p in st["pairs"] if p.get("c_acc") is not None]) < 10 and pool:
        seed = pool.pop(0)
        if seed in [p["seed"] for p in st["pairs"]] or seed in st["nan_failed"] or seed in failed_now:
            continue
        try:
            if seed in frozen_cfg:
                # 复用冻结三 seed: C' 净评估数已定, checkpoint 用 best_ckpt 找
                ck = best_ckpt_for_seed(seed) or None
                if seed == 42:
                    ck = REPO / "runs" / "trans002" / "c_fullfix_s42_205407" / "epoch080_acc82.80_model.pt"
                elif seed == 43:
                    ck = REPO / "runs" / "trans002" / "c_fullfix_s43_061239" / "epoch080_acc81.62_model.pt"
                elif seed == 44:
                    ck = REPO / "runs" / "trans002" / "c_fullfix_s44_092543" / "epoch080_acc81.54_model.pt"
                c_acc = {42: 0.8154, 43: 0.7840, 44: 0.7953}[seed]
                m_row = {"seed": seed, "val_acc": frozen_m[seed], "reused": True}
                st["pairs"].append({"seed": seed, "c_acc": c_acc, "m_acc": m_row["val_acc"],
                                    "c_reused": True, "m_reused": True, "c_ckpt": str(ck.name)})
                save_state(st)
                log(f"  seed {seed} (frozen reuse): C'={c_acc} M={m_row['val_acc']}")
                continue
            log(f"training C' mexp seed {seed} (fresh/resume, lr 0.00625)...")
            wall, acc, ck = run_one_seed(seed)
            import importlib
            td = importlib.import_module("run_r23_trans002")
            m_row = m_arm_from_ckpt(seed, ck, td)
            pair = {"seed": seed, "c_acc": round(acc, 4), "m_acc": m_row["val_acc"],
                    "c_wall_sec": wall, "m_wall_sec": m_row["wall_clock_sec"],
                    "c_ckpt": ck.name}
            st["pairs"].append(pair)
            save_state(st)
            log(f"  seed {seed}: C'={pair['c_acc']} M={pair['m_acc']} (C' wall {wall}s, M wall {m_row['wall_clock_sec']}s)")
        except Exception as e:
            msg = str(e)
            log(f"  seed {seed} FAILED: {msg[:200]}")
            if "nan" in msg.lower():
                st["nan_failed"].append(seed)
                save_state(st)
                if NAN_POOL:
                    pool.append(NAN_POOL.pop(0))
            else:
                failed_now.append(seed)
            continue

    pairs = [p for p in st["pairs"] if p.get("c_acc") is not None]
    if len(pairs) < 10:
        log(f"WAITING-FOR-RESUME: {len(pairs)}/10 pairs; crashed seeds resume via watchdog/session restart")
        sys.exit(3)

    # ---- 统计（预注册方案: Wilcoxon signed-rank + sign test）----
    from scipy import stats as sps
    diffs = [round(p["m_acc"] - p["c_acc"], 4) for p in pairs]
    w_stat, w_p = sps.wilcoxon(diffs)
    n_pos = sum(1 for d in diffs if d > 0)
    s_p = sps.binomtest(n_pos, len(diffs), 0.5).pvalue
    import statistics as stm
    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "protocol": "PSD-NTU-TRANS-002-MEXP (dated addendum 2026-09-09, frozen before any new run)",
        "n_pairs": len(pairs),
        "pairs": pairs,
        "diffs_m_minus_c": diffs,
        "mean_C": round(stm.mean([p["c_acc"] for p in pairs]), 4),
        "mean_M": round(stm.mean([p["m_acc"] for p in pairs]), 4),
        "wilcoxon": {"stat": round(float(w_stat), 2), "p_two_sided": round(float(w_p), 5)},
        "sign_test": {"n_positive": n_pos, "p_two_sided": round(float(s_p), 5)},
        "frozen_verdict_untouched": "PARTIAL (seeds 42-44, 15.8x/-4.49pp) per original decision rule",
    }
    out = REPO / "reports" / "r23-trans002-mexp-2026-09-09.json"
    out.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"written: {out}")
    log(f"MEXP result: mean M-C = {stm.mean(diffs):+.4f}, Wilcoxon p={w_p:.5f}, sign p={s_p:.5f}, M wins {n_pos}/{len(diffs)}")

    DONE.write_text(datetime.now().isoformat(timespec="seconds"))
    log("DONE")
    LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        log("FATAL(unwrapped): " + traceback.format_exc()[-1500:])
        FATAL.write_text(traceback.format_exc()[-2000:], encoding="utf-8")
        sys.exit(1)
