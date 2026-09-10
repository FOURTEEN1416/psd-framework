# -*- coding: utf-8 -*-
"""B-MIN-TRANS D 臂 v2 — 3D 双人体特征版（pack=(C3[x,y,z],T50,V25,M2) 已破译）。
纪律同 v0: 规则只来自公开类名语义+几何先验, 阈值先验固定, 精度报告即结果。"""
from __future__ import annotations
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bmin_trans"; OUT.mkdir(exist_ok=True)

pos = np.load(ROOT / "data/ntu60_frame50/xsub/train_position.npy", mmap_mode="r")
N = pos.shape[0]
tr = pickle.load(open(ROOT / "data/ntu60_frame50/xsub/train_label.pkl", "rb"))
yp = pickle.load(open(ROOT / "data/ntu60_frame50/xsub/train_label_yp49.pkl", "rb"))
tr_y = np.asarray(tr[1])
m = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
    "r22t", str(ROOT / "scripts/run_r22_ntu_transition.py")))
sys.modules["r22t"] = m; spec = m.__spec__; spec.loader.exec_module(m)
y2p_map_full, names49 = m.build_y_to_yp_map_ntu()
y_idx_to_p_idx = np.full(60, -1, dtype=np.int64)
for k, p in y2p_map_full.items(): y_idx_to_p_idx[int(k)] = int(p)
val_yp = y_idx_to_p_idx[np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")["val_label"]]
train_gt_yp = y_idx_to_p_idx[tr_y]

# ---------- 3D 双人体特征 ----------
feats = np.zeros((N, 15), dtype=np.float32)
t0 = time.time(); B = 2000
for st in range(0, N, B):
    en = min(st + B, N)
    x = np.asarray(pos[st:en], dtype=np.float32)          # (b,3,50,25,2)
    j = x.transpose(0, 2, 3, 4, 1)                        # (b,50,25,2,3)
    b0 = j[:, :, :, 0, :]                                 # 主体 (b,T,25,3)
    j0 = b0.copy(); j0[:, :, :, 1] *= -1.0                # y 向下正 -> 上正
    size = np.linalg.norm(j0[:, 0, 2, :] - j0[:, 0, 0, :], axis=-1) + 1e-6
    size = np.clip(size, 1e-3, None)[:, None]
    j0 = j0 / size[:, :, None, None]
    base, neck, head = j0[:, :, 0, :], j0[:, :, 2, :], j0[:, :, 3, :]
    lw, rw, la, ra = j0[:, :, 6, :], j0[:, :, 10, :], j0[:, :, 14, :], j0[:, :, 18, :]
    def speed(a):
        return np.linalg.norm(np.diff(a, axis=1), axis=-1).mean(1)
    def dtravel(a):
        return np.quantile(a, 0.9, axis=1) - np.quantile(a, 0.1, axis=1)
    f = np.zeros((en - st, 15), dtype=np.float32)
    f[:, 0] = (neck[:, :, 1] - base[:, :, 1]).mean(1)               # posture(直立≈1)
    f[:, 1] = speed(base)                                           # locomotion
    f[:, 2] = (speed(lw) + speed(rw)) / 2                           # wrist speed 3D
    f[:, 3] = dtravel(base[:, :, 1])                                # base vertical travel
    f[:, 4] = np.minimum(np.linalg.norm(lw - head, axis=-1).min(1),
                         np.linalg.norm(rw - head, axis=-1).min(1)) # hand-head 3D min
    f[:, 5] = ((lw[:, :, 1] > head[:, :, 1]) | (rw[:, :, 1] > head[:, :, 1])).mean(1)
    f[:, 6] = np.linalg.norm(lw - rw, axis=-1).mean(1)              # inter-wrist 3D
    f[:, 7] = np.abs(np.diff(base[:, :, 1], axis=1)).mean(1)        # base jitter
    f[:, 8] = (lw[:, :, 1].std(1) + rw[:, :, 1].std(1)) / 2         # wrist vertical osc
    f[:, 9] = np.maximum(speed(la), speed(ra))                      # ankle speed
    f[:, 10] = np.linalg.norm(base[:, -1, :] - base[:, 0, :], axis=-1)
    f[:, 11] = np.minimum(np.linalg.norm(lw - base, axis=-1).mean(1),
                          np.linalg.norm(rw - base, axis=-1).mean(1))  # hand-torso
    f[:, 12] = dtravel(base[:, :, 2])                               # depth travel
    f[:, 13] = base[:, :, 1].std(1)                                 # base vertical osc(hop)
    f[:, 14] = (x[:, :, :, :, 1].std(axis=(1, 2, 3)) > 1e-3).astype(np.float32)  # 第二人体在位
    feats[st:en] = f
print("features(3D) done", round(time.time() - t0, 1), "s")

# ---------- 规则表(阈值体系同 v0, 表序=优先级) ----------
RULES = [
    (36, "falling",        lambda r: r[0] < 0.5 and r[13] > 0.25, 0.6),
    (9,  "bidirectional walk", lambda r: False, 0.0),
    (8,  "strike other person", lambda r: False, 0.0),
    (43, "pat on back",    lambda r: False, 0.0),
    (44, "point finger",   lambda r: False, 0.0),
    (45, "hugging",        lambda r: False, 0.0),
    (46, "giving something", lambda r: False, 0.0),
    (47, "touch pocket",   lambda r: False, 0.0),
    (48, "handshaking",    lambda r: False, 0.0),
    (22, "jump up",        lambda r: r[3] > 0.9 and r[13] > 0.18, 0.6),
    (21, "hopping",        lambda r: r[13] > 0.12 and r[9] > 0.8, 0.5),
    (2,  "sit-stand transition", lambda r: r[3] > 0.55, 0.55),
    (19, "kicking something", lambda r: r[9] > 1.2, 0.55),
    (12, "throw",          lambda r: r[2] > 1.5 and r[5] > 0.25, 0.5),
    (17, "cheer up",       lambda r: r[5] > 0.45 and r[8] > 0.25, 0.6),
    (18, "hand waving",    lambda r: r[5] > 0.3 and r[8] > 0.3, 0.5),
    (31, "salute",         lambda r: r[4] < 0.55 and r[5] > 0.2 and r[1] < 0.25, 0.45),
    (13, "clapping",       lambda r: r[6] < 0.28 and r[2] > 0.35, 0.6),
    (32, "palms together", lambda r: r[6] < 0.18 and r[2] < 0.3, 0.55),
    (33, "cross hands",    lambda r: r[6] < 0.35 and r[2] < 0.25, 0.45),
    (29, "rub two hands together", lambda r: r[6] < 0.3 and 0.25 < r[2] < 0.7 and r[1] < 0.2, 0.45),
    (23, "phone call",     lambda r: 0.12 < r[4] < 0.5 and r[1] < 0.2, 0.5),
    (34, "sneeze/cough",   lambda r: r[4] < 0.3 and r[0] < 1.1, 0.45),
    (0,  "consume",        lambda r: 0.15 < r[4] < 0.6 and r[2] > 0.3, 0.5),
    (1,  "groom-brush",    lambda r: r[4] < 0.35 and r[8] > 0.15 and r[0] > 1.0, 0.5),
    (37, "touch head",     lambda r: r[4] < 0.3, 0.4),
    (40, "touch neck",     lambda r: 0.25 < r[11] < 0.6 and r[1] < 0.2, 0.35),
    (38, "touch chest",    lambda r: r[11] < 0.7 and r[1] < 0.15, 0.35),
    (41, "nausea or vomiting", lambda r: r[11] < 0.8 and r[0] < 0.9, 0.35),
    (3,  "jacket on/off",  lambda r: r[11] < 0.8 and r[2] < 0.5 and r[1] < 0.2, 0.4),
    (4,  "shoe on/off",    lambda r: r[11] > 1.4 and r[1] < 0.25, 0.5),
    (5,  "glasses on/off", lambda r: r[4] < 0.2 and r[1] < 0.2, 0.35),
    (6,  "hat on/off",     lambda r: r[5] > 0.12 and r[2] < 0.4, 0.35),
    (20, "reach into pocket", lambda r: 0.7 < r[11] < 1.3 and r[0] < 1.0, 0.4),
    (10, "drop",           lambda r: r[3] > 0.4 and r[11] > 1.0, 0.4),
    (11, "pick up",        lambda r: r[3] > 0.4 and r[11] > 1.0, 0.4),
    (25, "type on a keyboard", lambda r: r[1] < 0.15 and 0.3 < r[2] < 0.9 and r[6] < 0.7, 0.3),
    (24, "play with phone/tablet", lambda r: r[1] < 0.15 and r[6] < 0.6 and r[0] < 1.1, 0.3),
    (14, "reading",        lambda r: r[1] < 0.1 and r[2] < 0.4 and r[0] > 1.0, 0.3),
    (42, "use a fan",      lambda r: r[8] > 0.2 and r[1] < 0.2, 0.3),
    (7,  "head gesture",   lambda r: r[4] > 0.8 and r[2] < 0.25, 0.35),
    (26, "point to something", lambda r: r[2] < 0.6 and r[5] > 0.1, 0.3),
    (27, "taking a selfie", lambda r: r[5] > 0.2 and r[6] < 0.7, 0.3),
    (28, "check time",     lambda r: r[11] < 0.9 and r[0] > 1.05, 0.25),
    (15, "writing",        lambda r: r[1] < 0.12 and r[2] < 0.6, 0.25),
    (16, "tear up paper",  lambda r: r[6] > 0.5 and r[2] > 0.4, 0.25),
]
seed = np.full(N, -1, dtype=np.int64)
for pi, name, pred, c in RULES:
    mask = seed < 0
    if not mask.any(): break
    hit = np.fromiter((pred(feats[i]) for i in np.nonzero(mask)[0]), dtype=bool, count=int(mask.sum()))
    idx = np.nonzero(mask)[0][hit]
    seed[idx] = pi
covered = seed >= 0
prec = {}; tot_hit = tot_ok = 0
for pi, name, _, _ in RULES:
    msk = seed == pi
    if not msk.any(): continue
    ok = float((train_gt_yp[msk] == pi).mean())
    prec[name] = dict(n=int(msk.sum()), precision=round(ok, 4))
    tot_hit += int(msk.sum()); tot_ok += int((train_gt_yp[msk] == pi).sum())
overall = round(tot_ok / max(tot_hit, 1), 4); cov = round(float(covered.mean()), 4)
gate = bool(overall >= 0.5 and covered.mean() >= 0.3)
report = dict(coverage=cov, seed_precision_overall=overall, gate_pass=gate, per_class=prec,
              note="v2=3D双人体特征; 阈值先验固定未调参; precision=事后诊断")
json.dump(report, open(OUT / "seed_report_v2.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.save(OUT / "seed_labels_v2.npy", seed)
print(json.dumps({"coverage": cov, "precision": overall, "gate": gate}, ensure_ascii=False))
for k, v in sorted(prec.items(), key=lambda kv: -kv[1]["n"])[:12]:
    print(f"  {k}: n={v['n']} p={v['precision']}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
feat = np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")
if gate:
    sc = StandardScaler().fit(feat["train_feat"][covered])
    clf = LogisticRegression(max_iter=2000, tol=1e-3).fit(sc.transform(feat["train_feat"][covered]), seed[covered])
    acc = float(clf.score(sc.transform(feat["val_feat"]), val_yp))
    d = dict(d_arm_val_acc_yp49=round(acc, 4), n_train=int(covered.sum()),
             comparator_c_arm=0.798, gap_pp=round((acc - 0.798) * 100, 2))
    json.dump(d, open(OUT / "d_head_report.json", "w", encoding="utf-8"), indent=1)
    print("D-ARM:", json.dumps(d))
else:
    print("GATE FAIL -> FAILS 记录(v2 校准后仍不达标, 此为真结果候选)")
