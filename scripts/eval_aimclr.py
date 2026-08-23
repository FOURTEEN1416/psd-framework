"""P0.1 kNN 评估 — AimCLR 预训练表征质量探针。

对预训练 checkpoint 提取全部样本特征，5-fold 交叉验证 kNN top-1，
并与随机基线（100/N 类）对照。结果 JSON 归档 reports/。

口径声明：InterPet4D v1 无行为标注，probe 标签 = 文件名 dog ID（12 类），
衡量表征的个体区分能力（公开真实层口径），非行为识别精度。

用法：
    python scripts/eval_aimclr.py --knn \
        --weights runs/p01_aimclr_pretext/epoch120_model.pt \
        --data data/processed/p01/train_data.npy \
        --labels data/processed/p01/train_label.pkl \
        --out reports/p01-knn-result.json
"""
import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
AIMCLR_ROOT = REPO_ROOT / "external" / "AimCLR"


def knn_top1(features: np.ndarray, labels: np.ndarray, k: int = 1, n_folds: int = 5, seed: int = 0):
    """5-fold CV 的 kNN top-1。每折以 4/5 为参考库、1/5 为查询。

    返回 (per_fold_acc, mean_acc)。纯 torch 实现，CPU 可跑。
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    folds = np.array_split(idx, n_folds)
    accs = []
    for f in range(n_folds):
        query_idx = folds[f]
        bank_idx = np.concatenate([folds[j] for j in range(n_folds) if j != f])
        bank = torch.from_numpy(features[bank_idx]).float()
        query = torch.from_numpy(features[query_idx]).float()
        bank_lab = labels[bank_idx]
        # 余弦相似度检索
        bank_n = torch.nn.functional.normalize(bank, dim=1)
        query_n = torch.nn.functional.normalize(query, dim=1)
        sim = query_n @ bank_n.T  # (Q, B)
        topk = sim.topk(k, dim=1).indices.numpy()  # (Q, k)
        votes = bank_lab[topk]  # (Q, k)
        pred = np.apply_along_axis(lambda v: np.bincount(v, minlength=labels.max() + 1).argmax(), 1, votes)
        accs.append(float((pred == labels[query_idx]).mean()))
    return accs, float(np.mean(accs))


def selfcheck():
    """合成数据自检：两个线性可分簇上 kNN 必须接近 100%。"""
    rng = np.random.default_rng(7)
    a = rng.normal(0, 0.05, (50, 8)) + np.array([2.0] * 8)
    b = rng.normal(0, 0.05, (50, 8)) - np.array([2.0] * 8)
    feats = np.vstack([a, b]).astype(np.float32)
    labs = np.array([0] * 50 + [1] * 50)
    _, mean_acc = knn_top1(feats, labs, k=1, n_folds=5)
    assert mean_acc > 0.95, f"selfcheck 失败: {mean_acc}"
    print(f"[selfcheck] 合成两簇 kNN top-1 = {mean_acc:.3f} (>0.95) OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--knn", action="store_true", help="运行 kNN 评估（验收门）")
    ap.add_argument("--weights", default="runs/p01_aimclr_pretext/epoch120_model.pt")
    ap.add_argument("--data", default="data/processed/p01/train_data.npy")
    ap.add_argument("--labels", default="data/processed/p01/train_label.pkl")
    ap.add_argument("--out", default="reports/p01-knn-result.json")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--feature", choices=["proj", "backbone"], default="backbone",
                    help="probe 特征: backbone=fc 前池化特征(256d, 惯例), proj=projection 输出(128d)")
    args = ap.parse_args()

    if not args.knn:
        ap.error("当前版本仅支持 --knn")

    selfcheck()

    sys.path.insert(0, str(AIMCLR_ROOT))
    sys.path.insert(0, str(AIMCLR_ROOT / "torchlight"))
    from torchlight import import_class  # noqa: E402

    model_cls = import_class("net.aimclr.AimCLR")
    model = model_cls(
        base_encoder="net.st_gcn.Model", pretrain=True, feature_dim=128,
        queue_size=1024, momentum=0.999, Temperature=0.07, mlp=True,
        in_channels=3, hidden_channels=16, hidden_dim=256, num_class=12,
        dropout=0.5, graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
        edge_importance_weighting=True,
    )
    sd = torch.load(REPO_ROOT / args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(sd, strict=True)
    model.cuda().eval()

    def extract_backbone(x):
        """st_gcn.Model 的 fc 前全局池化特征 (N, hidden_dim=256)。"""
        enc = model.encoder_q
        n, c, t, v, m = x.size()
        h = x.permute(0, 4, 3, 1, 2).contiguous().view(n * m, v * c, t)
        h = enc.data_bn(h)
        h = h.view(n, m, v, c, t).permute(0, 1, 3, 4, 2).contiguous().view(n * m, c, t, v)
        for gcn, imp in zip(enc.st_gcn_networks, enc.edge_importance):
            h, _ = gcn(h, enc.A * imp)
        h = torch.nn.functional.avg_pool2d(h, h.size()[2:])
        h = h.view(n, m, -1).mean(dim=1)
        return h

    data = np.load(REPO_ROOT / args.data, mmap_mode="r")
    with open(REPO_ROOT / args.labels, "rb") as f:
        sample_names, labels = pickle.load(f)
    labels_arr = np.asarray(labels)

    feats = []
    with torch.no_grad():
        for i in range(0, len(data), 64):
            batch = torch.from_numpy(np.array(data[i : i + 64])).float().cuda()
            if args.feature == "backbone":
                feats.append(extract_backbone(batch).cpu().numpy())
            else:
                feats.append(model.encoder_q(batch).cpu().numpy())
    feats = np.vstack(feats)
    print(f"[eval] 特征({args.feature}): {feats.shape}, 样本: {len(labels_arr)}, 类别: {labels_arr.max()+1}")

    accs, mean_acc = knn_top1(feats, labels_arr, k=args.k, n_folds=args.folds)

    num_class = int(labels_arr.max() + 1)
    random_baseline = 100.0 / num_class
    ratio = mean_acc * 100.0 / random_baseline
    passed = ratio >= 2.0  # 交接文档: 至少 2-3 倍于随机水平

    result = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "weights": args.weights,
        "feature": args.feature,
        "n_samples": int(len(labels_arr)),
        "num_class": num_class,
        "label_semantics": "dog ID (InterPet4D v1 无行为标注, 代理 probe)",
        "metric_layer": "公开真实层 (InterPet4D smal_npy)",
        "knn_k": args.k,
        "cv_folds": args.folds,
        "fold_top1_acc": accs,
        "knn_top1_mean_pct": round(mean_acc * 100.0, 2),
        "random_baseline_pct": round(random_baseline, 2),
        "ratio_vs_random": round(ratio, 2),
        "acceptance_2x_random": bool(passed),
    }
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    verdict = "通过" if passed else "未达标"
    print(f"[eval] kNN top-1 {mean_acc*100:.2f}% vs 随机 {random_baseline:.2f}% "
          f"({ratio:.2f}x) → 验收({'>=' '2x'}): {verdict}")


if __name__ == "__main__":
    main()
