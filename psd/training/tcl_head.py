"""P0.4 Ω 轻量分类头 — W10 窗口 owner（method.md §3.3.3 最小可行实现）。

职责: 在冻结 Φ 特征（256d，W8 缓存）上训练线性层或 2 层 MLP 分类头，
供伪标签迭代闭环做全量段重分配。TCL 风格时序对比巩固为 stretch goal
（W10 交接 §3 Step 1），不阻塞本模块。

设计要点:
- Φ 全程冻结——头只吃预计算 embedding，无任何骨干前向路径；
- 类别子集训练支持: 池迭代中某类可能暂时无入池样本，未训练类列恒 0 概率，
  argmax 永不命中缺席类（防幽灵标签）;
- 同种子确定性: 权重初始化与批内 shuffle 均由 seed 驱动（统计协议 ≥3 seeds）;
- 主线 device=cpu（特征已缓存零 GPU），规避与 W4 定时任务撞卡。
"""
from __future__ import annotations

import numpy as np


class TorchHead:
    """冻结特征上的小分类头。标签约定: 整数类别 id ∈ [0, n_classes)。"""

    def __init__(self, dim_in: int, n_classes: int, hidden_dim: int = 64,
                 seed: int = 42, epochs: int = 150, lr: float = 1e-3,
                 weight_decay: float = 1e-4, batch_size: int = 128,
                 device: str = "cpu"):
        self.dim_in = int(dim_in)
        self.n_classes = int(n_classes)
        self.hidden_dim = int(hidden_dim)
        self.seed = int(seed)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.device = device
        self._model = None          # torch.nn.Module，fit 时构建
        self._local_to_global: list[int] = []

    # ---------------------------------------------------------- 训练

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TorchHead":
        import torch
        from torch import nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if len(X) != len(y):
            raise ValueError(f"X/y 长度不一致: {len(X)} vs {len(y)}")
        if len(X) == 0:
            raise ValueError("训练集为空——拒绝拟合（空池防线）")

        observed = sorted(set(y.tolist()))
        if min(observed) < 0 or max(observed) >= self.n_classes:
            raise ValueError(f"标签越界 [0,{self.n_classes}): {observed}")
        local_of = {g: i for i, g in enumerate(observed)}
        self._local_to_global = observed

        torch.manual_seed(self.seed)
        layers: list[nn.Module] = []
        if self.hidden_dim > 0:
            layers += [nn.Linear(self.dim_in, self.hidden_dim), nn.ReLU(),
                       nn.Linear(self.hidden_dim, len(observed))]
        else:
            layers += [nn.Linear(self.dim_in, len(observed))]
        model = nn.Sequential(*layers).to(self.device)

        opt = torch.optim.Adam(model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()
        xt = torch.from_numpy(X).to(self.device)
        yt = torch.tensor([local_of[v] for v in y.tolist()],
                          dtype=torch.long, device=self.device)

        rng = np.random.default_rng(self.seed)
        n = len(X)
        model.train()
        for _ in range(self.epochs):
            perm = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s: s + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        model.eval()
        self._model = model
        return self

    # ---------------------------------------------------------- 推断

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(N,dim_in) -> (N,n_classes)；未参与训练的类列恒 0，行和为 1。"""
        import torch

        if self._model is None:
            raise RuntimeError("先 fit 再 predict_proba")
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            logits = self._model(torch.from_numpy(X).to(self.device))
            local = torch.softmax(logits, dim=1).cpu().numpy()
        full = np.zeros((len(X), self.n_classes), dtype=np.float64)
        full[:, self._local_to_global] = local
        return full

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
