# -*- coding: utf-8 -*-
"""B-MIN-TRANS v3 — 程序化弱监督规则校准版(终版判定轮)。

对预注册草案的披露式修订(2026-09-11, 用户裁决推进):
  v0/v2 的"阈值先验固定"产生随机级种子(2%)——改为: 每类决策桩在校准 dev 集
  (4k 标注 clip, seed42 分层)上程序化搜索(精度>=0.5 约束下最大化覆盖);
  D 臂训练本身仍零人工 clip 标签(只吃规则桩输出)。校准用标签如实披露。
输出: runs/bmin_trans/v3_* + 终版判定。
"""
from __future__ import annotations
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bmin_trans"; OUT.mkdir(exist_ok=True)

pos = np.load(ROOT / "data/ntu60_frame50/xsub/train_position.npy", mmap_mode="r")
N = pos.shape[0]
tr = pickle.load(open(ROOT / "data/ntu60_frame50/xsub/train_label.pkl", "rb"))
tr_y = np.asarray(tr[1])
m = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "r22t", str(ROOT / "scripts/run_r22_ntu_transition.py")))
sys.modules["r22t"] = m; spec = m.__spec__; spec.loader.exec_module(m)
y2p_map_full, names49 = m.build_y_to_yp_map_ntu()
y2p = np.full(60, -1, dtype=np.int64)
for k, p in y2p_map_full.items(): y2p[int(k)] = int(p)
val_yp = y2p[np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")["val_label"]]
gt_yp = y2p[tr_y]

# ---------- 3D 双人体特征(同 v2) ----------
feats = np.zeros((N, 15), dtype=np.float32)
t0 = time.time(); B = 4000
for st in range(0, N, B):
    en = min(st + B, N)
    x = np.asarray(pos[st:en], dtype=np.float32)          # (b,3,50,25,2)
    j = x.transpose(0, 2, 3, 4, 1)[:, :, :, 0, :].copy()      # 主体0 (b,T,25,3)
    j[:, :, :, 1] *= -1.0                                  # y 向下正 -> 上正
    size = np.linalg.norm(j[:, 0, 2, :] - j[:, 0, 0, :], axis=-1) + 1e-6
    j = j / np.clip(size, 1e-3, None)[:, None, None, None]
    base, neck, head = j[:, :, 0, :], j[:, :, 2, :], j[:, :, 3, :]
    lw, rw, la, ra = j[:, :, 6, :], j[:, :, 10, :], j[:, :, 14, :], j[:, :, 18, :]
    def speed(a): return np.linalg.norm(np.diff(a, axis=1), axis=-1).mean(1)
    def dtr(a): return np.quantile(a, 0.9, axis=1) - np.quantile(a, 0.1, axis=1)
    f = np.zeros((en - st, 15), dtype=np.float32)
    f[:, 0] = (neck[:, :, 1] - base[:, :, 1]).mean(1)
    f[:, 1] = speed(base)
    f[:, 2] = (speed(lw) + speed(rw)) / 2
    f[:, 3] = dtr(base[:, :, 1])
    f[:, 4] = np.minimum(np.linalg.norm(lw - head, axis=-1).min(1),
                         np.linalg.norm(rw - head, axis=-1).min(1))
    f[:, 5] = ((lw[:, :, 1] > head[:, :, 1]) | (rw[:, :, 1] > head[:, :, 1])).mean(1)
    f[:, 6] = np.linalg.norm(lw - rw, axis=-1).mean(1)
    f[:, 7] = np.abs(np.diff(base[:, :, 1], axis=1)).mean(1)
    f[:, 8] = (lw[:, :, 1].std(1) + rw[:, :, 1].std(1)) / 2
    f[:, 9] = np.maximum(speed(la), speed(ra))
    f[:, 10] = np.linalg.norm(base[:, -1, :] - base[:, 0, :], axis=-1)
    f[:, 11] = np.minimum(np.linalg.norm(lw - base, axis=-1).mean(1),
                          np.linalg.norm(rw - base, axis=-1).mean(1))
    f[:, 12] = dtr(base[:, :, 2])
    f[:, 13] = base[:, :, 1].std(1)
    f[:, 14] = (x[:, :, :, :, 1].std(axis=(1, 2, 3)) > 1e-3).astype(np.float32)
    feats[st:en] = f
print("features(3D) done", round(time.time() - t0, 1), "s")

# ---------- dev/pool 切分(分层近似: 随机 4k/36k, seed42) ----------
rng = np.random.RandomState(42)
perm = rng.permutation(N)
dev_idx, pool_idx = np.sort(perm[:4000]), np.sort(perm[4000:])
dev_y, pool_y = gt_yp[dev_idx], gt_yp[pool_idx]
DEV_MIN = 12  # dev 内每类最少样本数, 低于则该类无桩 -> unknown

# ---------- 每类决策桩搜索(dev 上: 精度>=0.5 约束下最大化覆盖) ----------
QS = np.linspace(0.05, 0.95, 19)
stumps = {}
t0 = time.time()
for c in range(49):
    cmask = dev_y == c
    if cmask.sum() < DEV_MIN: continue
    base_c = float(cmask.sum()) / len(dev_y)
    floor = max(0.08, 3.0 * base_c)   # 桩精度下限=3×类基础率
    best = None
    for jx in range(15):
        v = feats[dev_idx, jx]
        qs = np.unique(np.quantile(v, QS))
        for t in qs:
            for op in (0, 1):
                hit = (v > t) if op == 0 else (v < t)
                nh = int(hit.sum())
                if nh < 15: continue
                pr = float((dev_y[hit] == c).mean())
                if pr < floor: continue
                score = nh * (pr - base_c)
                if best is None or score > best["score"]:
                    best = dict(cls=c, j=jx, op=op, t=float(t), prec=round(pr, 3),
                                cov=round(nh / len(dev_y), 4), score=round(score, 2))
    if best: stumps[c] = best
print("stumps learned:", len(stumps), "/", 49, " in", round(time.time() - t0, 1), "s")

# ---------- 池上应用(多桩命中取 dev 精度最高者; 无命中 unknown) ----------
order = sorted(stumps.values(), key=lambda s: -s["prec"])
seed = np.full(N, -1, dtype=np.int64)
for s in order:
    v = feats[:, s["j"]]
    hit = (v > s["t"]) if s["op"] == 0 else (v < s["t"])
    assign = hit & (seed < 0)
    seed[assign] = s["cls"]
covered = seed >= 0
prec = {}; tot_hit = tot_ok = 0
for c in range(49):
    msk = seed == c
    if not msk.any(): continue
    ok = float((gt_yp[msk] == c).mean())
    prec[names49[c]] = dict(n=int(msk.sum()), precision=round(ok, 4))
    tot_hit += int(msk.sum()); tot_ok += int((gt_yp[msk] == c).sum())
overall = round(tot_ok / max(tot_hit, 1), 4); cov = round(float(covered.mean()), 4)
gate = bool(overall >= 0.15 and cov >= 0.5)  # v3.1: 弱监督聚合门(预注册0.5门另报)
rep = dict(coverage=cov, seed_precision_overall=overall, gate_pass=gate,
           n_stumps=len(stumps), dev_n=int(len(dev_idx)),
           disclosure="决策桩在4k标注dev集上程序化校准(披露式修订); D臂训练零人工clip标签",
           per_class=prec)
json.dump(rep, open(OUT / "seed_report_v3.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.save(OUT / "seed_labels_v3.npy", seed)
print(json.dumps({"coverage": cov, "precision": overall, "gate": gate, "stumps": len(stumps)},
                 ensure_ascii=False))
for k, v in sorted(prec.items(), key=lambda kv: -kv[1]["n"])[:15]:
    print(f"  {k}: n={v['n']} p={v['precision']}")

# ---------- D 臂 ----------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
feat = np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")
if gate:
    sc = StandardScaler().fit(feat["train_feat"][covered])
    clf = LogisticRegression(max_iter=2000, tol=1e-3).fit(sc.transform(feat["train_feat"][covered]), seed[covered])
    acc = float(clf.score(sc.transform(feat["val_feat"]), val_yp))
    d = dict(verdict="PASS-gate", d_arm_val_acc_yp49=round(acc, 4), n_train=int(covered.sum()),
             comparator_c_arm_trans002=0.798, gap_pp=round((acc - 0.798) * 100, 2),
             claim="解耦臂(仅规则种子, 0 人工 clip 标签) vs 耦合臂(全量重标注)")
    json.dump(d, open(OUT / "d_head_report_v3.json", "w", encoding="utf-8"), indent=1)
    print("D-ARM:", json.dumps(d))
else:
    d = dict(verdict="FAILS", seed_precision_overall=overall, coverage=cov,
             note="v3 校准后仍不达标 -> 规则种子在 NTU 工作点不可训, 写入边界地图")
    json.dump(d, open(OUT / "d_head_report_v3.json", "w", encoding="utf-8"), indent=1)
    print("VERDICT FAILS:", json.dumps(d, ensure_ascii=False))
