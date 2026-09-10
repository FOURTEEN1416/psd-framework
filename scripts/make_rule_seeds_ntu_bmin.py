# -*- coding: utf-8 -*-
"""B-MIN-TRANS D 臂 — NTU60 60→49 迁移的规则种子生成 + 精度校准 + 解耦头训练。

设计真源: dev-docs/research/pathB-min-asym-transfer-prereg-draft-2026-09-11.md
纪律: 规则只来自公开类名语义 + 几何先验(2D 投影, 竖轴/尺寸无标签自定);
      阈值先验固定、禁用标签调参; 精度报告本身即实验结果(FAILS 亦为结果)。
输入: data/ntu60_frame50/xsub/{train_position.npy, train_label.pkl, train_label_yp49.pkl}
      runs/ntu_lowres/features_joint_ep300.npz (E9 冻结特征)
      runs/bmin_trans_yprime_map.json (49 类名 + Y→Y' 映射)
输出: runs/bmin_trans/{seed_labels.npy, seed_report.json, d_head_report.json}
"""
from __future__ import annotations
import io, json, pickle, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "bmin_trans"
OUT.mkdir(exist_ok=True)
BS = chr(92)

# ---------- 载入 ----------
pos = np.load(ROOT / "data/ntu60_frame50/xsub/train_position.npy", mmap_mode="r")
N = pos.shape[0]
tr = pickle.load(open(ROOT / "data/ntu60_frame50/xsub/train_label.pkl", "rb"))
yp = pickle.load(open(ROOT / "data/ntu60_frame50/xsub/train_label_yp49.pkl", "rb"))
y2p = json.load(open(ROOT / "runs/bmin_trans_yprime_map.json", encoding="utf-8"))
names49 = y2p["names49"]; y2p_map = y2p["y2p"]
tr_ids, tr_y = tr[0], np.asarray(tr[1])
_, yp_y = yp[0], np.asarray(yp[1])
# Y 索引 -> 类名 -> Y' 索引
# train_label.pkl 结构与 yp49 相同: (ids, labels); 类名表来自映射脚本侧的 Y 名单 ——
# 这里从 y2p 键(类名)与 tr_y 无关, 需 Y 名单: 从 run_r22 模块取
import importlib.util
spec = importlib.util.spec_from_file_location("r22t", str(ROOT / "scripts/run_r22_ntu_transition.py"))
m = importlib.util.module_from_spec(spec); sys.modules["r22t"] = m; spec.loader.exec_module(m)
y2p_map_full, names49_chk = m.build_y_to_yp_map_ntu()
assert names49_chk == names49
y_idx_to_p_idx = np.full(60, -1, dtype=np.int64)
for k, p in y2p_map_full.items():
    y_idx_to_p_idx[int(k)] = int(p)
val_yp = y_idx_to_p_idx[np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")["val_label"]]
train_gt_yp = y_idx_to_p_idx[tr_y]

# ---------- 特征(无标签) ----------
feats = np.zeros((N, 14), dtype=np.float32)
t0 = time.time()
CH, T, J, C = pos.shape[1], pos.shape[2], pos.shape[3], pos.shape[4]
B = 2000
for st in range(0, N, B):
    en = min(st + B, N)
    x = np.asarray(pos[st:en, 0], dtype=np.float32)  # (b,T,25,2) joint xy
    # 竖轴实测(标签无关诊断): 轴1为图像y向下为正(P(head>base)=0.13), 上=负方向
    up_axis, up_sign = 1, -1.0
    xy = x.copy()
    xy[:, :, :, up_axis] *= up_sign
    size = np.linalg.norm(xy[:, 0, 2, :] - xy[:, 0, 0, :], axis=-1) + 1e-6  # neck-base
    size = np.clip(size, 1e-3, None)[:, None]
    j = xy / size[:, :, None, None]
    base, neck, head = j[:, :, 0, :], j[:, :, 2, :], j[:, :, 3, :]
    lw, rw = j[:, :, 6, :], j[:, :, 10, :]
    lsh, rsh = j[:, :, 4, :], j[:, :, 7, :]
    lk, rk = j[:, :, 13, :], j[:, :, 17, :]
    la, ra = j[:, :, 14, :], j[:, :, 18, :]
    def speed(a):  # (b,T,2)->(b,)
        d = np.linalg.norm(np.diff(a, axis=1), axis=-1)
        return d.mean(axis=1)
    def dtravel(a):
        return (np.quantile(a, 0.9, axis=1) - np.quantile(a, 0.1, axis=1))
    f = np.zeros((en - st, 14), dtype=np.float32)
    f[:, 0] = (neck[:, :, up_axis].mean(1) - base[:, :, up_axis].mean(1)).mean(1) if False else (neck[:, :, up_axis].mean(1) - base[:, :, up_axis].mean(1))
    f[:, 1] = speed(base)                      # locomotion
    f[:, 2] = (speed(lw) + speed(rw)) / 2      # wrist speed
    f[:, 3] = dtravel(base[:, :, up_axis])     # base vertical travel (jump/sit-stand)
    f[:, 4] = np.minimum(
        np.linalg.norm(lw - head, axis=-1).min(axis=1),
        np.linalg.norm(rw - head, axis=-1).min(axis=1))  # hand-head min
    f[:, 5] = ((lw[:, :, up_axis] > head[:, :, up_axis]) | (rw[:, :, up_axis] > head[:, :, up_axis])).mean(1)
    f[:, 6] = np.linalg.norm(lw - rw, axis=-1).mean(1)  # inter-wrist
    f[:, 7] = np.abs(np.diff(base[:, :, up_axis], axis=1)).mean(1)  # base jitter
    f[:, 8] = (lw[:, :, up_axis].std(1) + rw[:, :, up_axis].std(1)) / 2  # wrist vertical osc
    f[:, 9] = np.maximum(speed(la), speed(ra))  # ankle speed (kick)
    f[:, 10] = np.linalg.norm(base[:, -1, :] - base[:, 0, :], axis=-1)  # net displacement
    f[:, 11] = np.minimum(
        np.linalg.norm(lw - base, axis=-1).mean(axis=1),
        np.linalg.norm(rw - base, axis=-1).mean(axis=1))  # hand-torso
    f[:, 12] = np.maximum(
        np.linalg.norm(np.diff(lw, axis=0), axis=-1).mean(0) if False else 0, 0)  # reserved
    f[:, 13] = (base[:, :, up_axis].std(1))  # base vertical osc (hop)
    feats[st:en] = f
    if st % 10000 == 0: print("feat", st, round(time.time() - t0, 1), "s", flush=True)
print("features done", round(time.time() - t0, 1), "s")

# ---------- 规则表(Y' 索引 -> 谓词, 先具体后泛化; 阈值=几何先验固定) ----------
F = {i: lambda r, i=i: r[i] for i in range(14)}
def near(a, b, t): return abs(a - b) < t
RULES = [
    (36, "falling",        lambda r: r[0] < 0.5 and r[13] > 0.25, 0.6),
    (9,  "bidirectional walk", lambda r: False, 0.0),          # 双人类, 单骨架不可判 -> unknown
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

# ---------- 种子生成(首匹配, 优先级=表序) ----------
seed = np.full(N, -1, dtype=np.int64)
conf = np.zeros(N, dtype=np.float32)
rule_hit = np.full(N, -1, dtype=np.int64)
for pi, name, pred, c in RULES:
    mask = seed < 0
    if not mask.any(): break
    hit = np.fromiter((pred(feats[i]) for i in np.nonzero(mask)[0]), dtype=bool, count=int(mask.sum()))
    idx = np.nonzero(mask)[0][hit]
    seed[idx] = pi; conf[idx] = c; rule_hit[idx] = pi

covered = seed >= 0
prec = {}
tot_hit = tot_ok = 0
for pi, name, _, _ in RULES:
    m = seed == pi
    if not m.any(): continue
    ok = (train_gt_yp[m] == pi).mean()
    prec[name] = dict(n=int(m.sum()), precision=round(float(ok), 4))
    tot_hit += int(m.sum()); tot_ok += int((train_gt_yp[m] == pi).sum())
overall = round(tot_ok / max(tot_hit, 1), 4)
cov = round(float(covered.mean()), 4)
gate = overall >= 0.5 and covered.mean() >= 0.3
report = dict(coverage=cov, seed_precision_overall=overall, gate_pass=bool(gate),
              per_class=prec, n=N,
              note="阈值=几何先验先验固定, 未用标签调参; precision 对 train Y' GT 的事后诊断")
json.dump(report, open(OUT / "seed_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.save(OUT / "seed_labels.npy", seed)
print(json.dumps({k: report[k] for k in ("coverage", "seed_precision_overall", "gate_pass")}, ensure_ascii=False))
for k, v in prec.items(): print(f"  {k}: n={v['n']} p={v['precision']}")

# ---------- D 臂(仅过门时训练; 不过门 = FAILS 记录) ----------
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
feat = np.load(ROOT / "runs/ntu_lowres/features_joint_ep300.npz")
if gate:
    sc = StandardScaler().fit(feat["train_feat"][covered])
    Xh = sc.transform(feat["train_feat"][covered]); yh = seed[covered]
    clf = LogisticRegression(max_iter=2000, tol=1e-3).fit(Xh, yh)
    acc = float(clf.score(sc.transform(feat["val_feat"]), val_yp))
    d_rep = dict(d_arm_val_acc_yp49=round(acc, 4), n_train_seeded=int(covered.sum()),
                 comparator_c_arm_trans002=0.798, gap_pp=round((acc - 0.798) * 100, 2),
                 gate="seeds-only, 0 human clip labels; rules from public class semantics + geometry priors")
    json.dump(d_rep, open(OUT / "d_head_report.json", "w", encoding="utf-8"), indent=1)
    print("D-ARM:", json.dumps(d_rep))
else:
    print("GATE FAIL -> B-MIN-TRANS D 臂判 FAILS (种子精度/覆盖率不足), 已记 seed_report.json")
