# -*- coding: utf-8 -*-
"""B-MIN-TRANS v4 — 30维特征 + 单桩/双特征合取规则(统一矩形搜索)。
可解释规则 + 4k dev 校准(披露) + D 臂零人工 clip 标签; NTU25 标准关节序。"""
from __future__ import annotations
import importlib.util, itertools, json, pickle, sys, time
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
pack = np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")
val_yp = y2p[pack["val_label"]]
gt_yp = y2p[tr_y]

NF = 30
feats = np.zeros((N, NF), dtype=np.float32)
t0 = time.time(); B = 4000
for st in range(0, N, B):
    en = min(st + B, N)
    x = np.asarray(pos[st:en], dtype=np.float32)          # (b,3,50,25,2)
    j = x.transpose(0, 2, 3, 4, 1)[:, :, :, 0, :].copy()  # (b,T,25,3)
    j[:, :, :, 1] *= -1.0                                  # y 上正
    size = np.linalg.norm(j[:, 0, 2, :] - j[:, 0, 0, :], axis=-1) + 1e-6
    j = j / np.clip(size, 1e-3, None)[:, None, None, None]
    base, neck, head = j[:, :, 0, :], j[:, :, 2, :], j[:, :, 3, :]
    lsh, rsh = j[:, :, 4, :], j[:, :, 8, :]
    lw, rw, lh, rh = j[:, :, 6, :], j[:, :, 10, :], j[:, :, 7, :], j[:, :, 11, :]
    la, ra = j[:, :, 14, :], j[:, :, 18, :]
    def spd(a): return np.linalg.norm(np.diff(a, axis=1), axis=-1).mean(1)
    def dtr(a): return np.quantile(a, 0.9, axis=1) - np.quantile(a, 0.1, axis=1)
    def zc(a):
        c = a - a.mean(1, keepdims=True)
        return (np.diff(np.sign(c), axis=1) != 0).sum(1)
    f = np.zeros((en - st, NF), dtype=np.float32)
    f[:, 0] = (neck[:, :, 1] - base[:, :, 1]).mean(1)
    f[:, 1] = spd(base); f[:, 2] = (spd(lw) + spd(rw)) / 2
    f[:, 3] = dtr(base[:, :, 1])
    f[:, 4] = np.minimum(np.linalg.norm(lw - head, axis=-1).min(1),
                         np.linalg.norm(rw - head, axis=-1).min(1))
    f[:, 5] = ((lw[:, :, 1] > head[:, :, 1]) | (rw[:, :, 1] > head[:, :, 1])).mean(1)
    f[:, 6] = np.linalg.norm(lw - rw, axis=-1).mean(1)
    f[:, 7] = np.abs(np.diff(base[:, :, 1], axis=1)).mean(1)
    f[:, 8] = (lw[:, :, 1].std(1) + rw[:, :, 1].std(1)) / 2
    f[:, 9] = np.maximum(spd(la), spd(ra))
    f[:, 10] = np.linalg.norm(base[:, -1, :] - base[:, 0, :], axis=-1)
    f[:, 11] = np.minimum(np.linalg.norm(lw - base, axis=-1).mean(1),
                          np.linalg.norm(rw - base, axis=-1).mean(1))
    f[:, 12] = dtr(base[:, :, 2]); f[:, 13] = base[:, :, 1].std(1)
    f[:, 14] = (x[:, :, :, :, 1].std(axis=(1, 2, 3)) > 1e-3).astype(np.float32)
    f[:, 15] = (spd(lh) + spd(rh)) / 2
    f[:, 16] = (lh[:, :, 1].std(1) + rh[:, :, 1].std(1)) / 2
    f[:, 17] = (np.linalg.norm(lw - lsh, axis=-1).mean(1) +
                np.linalg.norm(rw - rsh, axis=-1).mean(1)) / 2
    f[:, 18] = (lh[:, :, 1] + rh[:, :, 1]).mean(1) / 2 - neck[:, :, 1].mean(1)
    f[:, 19] = spd(head)
    f[:, 20] = zc(lw[:, :, 1] + rw[:, :, 1])
    f[:, 21] = (lw[:, :, 0].std(1) + rw[:, :, 0].std(1)) / 2
    f[:, 22] = np.linalg.norm(la - ra, axis=-1).mean(1)
    f[:, 23] = (la[:, :, 0].std(1) + ra[:, :, 0].std(1)) / 2
    f[:, 24] = (la[:, :, 1] + ra[:, :, 1]).mean(1) / 2 - base[:, :, 1].mean(1)
    f[:, 25] = (np.linalg.norm(j[:, :, 5, :] - base, axis=-1).mean(1) +
                np.linalg.norm(j[:, :, 9, :] - base, axis=-1).mean(1)) / 2
    f[:, 26] = np.linalg.norm(lw - rw, axis=-1).std(1)
    f[:, 27] = np.linalg.norm(lh - rh, axis=-1).mean(1)
    f[:, 28] = np.abs(np.diff(lw[:, :, 1] + rw[:, :, 1], axis=1)).mean(1)
    f[:, 29] = (neck[:, :, 0].std(1) + neck[:, :, 2].std(1)) / 2
    feats[st:en] = f
print("features(30D) done", round(time.time() - t0, 1), "s")

rng = np.random.RandomState(42)
perm = rng.permutation(N)
dev_idx, pool_idx = np.sort(perm[:4000]), np.sort(perm[4000:])
dev_y, gt_pool_y = gt_yp[dev_idx], gt_yp[pool_idx]
DEV_MIN = 12
QS = np.linspace(0.05, 0.95, 19)

def rect_scan(v1, s1, v2, s2, cmask):
    best = None
    a, b = v1 * s1, v2 * s2
    e1 = np.unique(np.quantile(a, QS)); e2 = np.unique(np.quantile(b, QS))
    k1 = np.searchsorted(e1, a, side="right"); k2 = np.searchsorted(e2, b, side="right")
    key = k1 * (len(e2) + 1) + k2
    kt, kinv = np.unique(key, return_inverse=True)
    G = (len(e1) + 1) * (len(e2) + 1)
    T = np.zeros(G); np.add.at(T, key, 1.0)
    C = np.zeros(G); np.add.at(C, key, cmask.astype(float))
    T = T.reshape(len(e1) + 1, len(e2) + 1); C = C.reshape(len(e1) + 1, len(e2) + 1)
    Tc = T.cumsum(0).cumsum(1); Cc = C.cumsum(0).cumsum(1)
    out = []
    for i in range(Tc.shape[0]):
        for jj in range(Tc.shape[1]):
            nh = int(Tc[i, jj])
            if nh < 15: continue
            pr = float(Cc[i, jj]) / nh
            out.append((pr, nh, (float(e1[i - 1]) if i > 0 else -9e9,
                                 float(e2[jj - 1]) if jj > 0 else -9e9)))
    return out

all_rules = []
t0 = time.time()
for c in range(49):
    cm = dev_y == c
    if cm.sum() < DEV_MIN: continue
    base_c = float(cm.sum()) / len(dev_y)
    floor_c = max(0.08, 3.0 * base_c)
    per_feat = []
    for jx in range(NF):
        for sgn in (1.0, -1.0):
            v = feats[dev_idx, jx] * sgn
            o = np.argsort(v, kind="stable")
            sm = cm[o]
            Tc = np.arange(1, len(v) + 1, dtype=float)
            Cc = sm.cumsum()
            sc = np.where(Tc >= 15, Tc * (Cc / Tc - base_c), -1.0)
            k = int(np.argmax(sc))
            pr = Cc[k] / Tc[k]
            if pr >= floor_c and sc[k] > 0:
                per_feat.append((float(sc[k]), dict(
                    kind="1d", cls=c, prec=round(pr, 3), nh=int(k + 1),
                    pred=(jx, sgn, float(v[k])))))
    per_feat.sort(key=lambda x: -x[0])
    for _, r in per_feat[:6]:
        all_rules.append((r["nh"] * (r["prec"] - base_c), r))
    top_feats = list(dict.fromkeys(r["pred"][0] for _, r in per_feat[:6]))[:6]
    for a, b in itertools.combinations(top_feats, 2):
        for pr, nh, (t1, t2) in rect_scan(feats[dev_idx, a], 1.0, feats[dev_idx, b], 1.0, cm):
            if pr >= floor_c:
                all_rules.append((nh * (pr - base_c), dict(
                    kind="2d", cls=c, prec=round(pr, 3), nh=nh,
                    pred=(a, 1.0, t1, b, 1.0, t2))))
print("rules generated:", len(all_rules), "in", round(time.time() - t0, 1), "s")

def apply_rule(pred, F):
    if len(pred) == 3:
        jx, sgn, t = pred
        return F[:, jx] * sgn <= t + 1e-9
    a, s1, t1, b, s2, t2 = pred
    return (F[:, a] * s1 <= t1 + 1e-9) & (F[:, b] * s2 <= t2 + 1e-9)

order = sorted(all_rules, key=lambda x: (-x[1]["prec"], -x[1]["nh"]))
seed = np.full(N, -1, dtype=np.int64)
for sc, r in order:
    mask = seed < 0
    if not mask.any(): break
    hit = apply_rule(r["pred"], feats)
    seed[hit & mask] = r["cls"]
covered = seed >= 0
prec = {}; tot_hit = tot_ok = 0
for c in range(49):
    msk = seed == c
    if not msk.any(): continue
    ok = float((gt_yp[msk] == c).mean())
    prec[names49[c]] = dict(n=int(msk.sum()), precision=round(ok, 4))
    tot_hit += int(msk.sum()); tot_ok += int((gt_yp[msk] == c).sum())
overall = round(tot_ok / max(tot_hit, 1), 4); cov = round(float(covered.mean()), 4)
gate = bool(overall >= 0.15 and cov >= 0.5)
rep = dict(coverage=cov, seed_precision_overall=overall, gate_pass=gate,
           n_rules=len(all_rules), disclosure="单桩+双特征合取, 4k dev 校准(披露); D 臂零人工 clip 标签",
           per_class=prec)
json.dump(rep, open(OUT / "seed_report_v4.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.save(OUT / "seed_labels_v4.npy", seed)
print(json.dumps({"coverage": cov, "precision": overall, "gate": gate, "rules": len(all_rules)},
                 ensure_ascii=False))
for k, v in sorted(prec.items(), key=lambda kv: -kv[1]["n"])[:15]:
    print(f"  {k}: n={v['n']} p={v['precision']}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
feat = np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")
if gate:
    sc = StandardScaler().fit(feat["train_feat"][covered])
    clf = LogisticRegression(max_iter=2000, tol=1e-3).fit(sc.transform(feat["train_feat"][covered]), seed[covered])
    acc = float(clf.score(sc.transform(feat["val_feat"]), val_yp))
    d = dict(verdict="FAILS" if acc < 0.798 else "PASS",
             d_arm_val_acc_yp49=round(acc, 4), n_train=int(covered.sum()),
             comparator_c_arm_trans002=0.798, gap_pp=round((acc - 0.798) * 100, 2),
             note="D 臂=仅规则种子(0 人工 clip 标签); C 臂=全量重标注 79.8%")
    json.dump(d, open(OUT / "d_head_report_v4.json", "w", encoding="utf-8"), indent=1)
    print("D-ARM v4:", json.dumps(d, ensure_ascii=False))
else:
    print("GATE FAIL -> 覆盖/精度不足")
