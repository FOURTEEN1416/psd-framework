"""SMQ 训练/推理适配器 — P0.2 训练管线 owner（薄封装）。

职责：把 external/SMQ 官方实现接入本仓配置体系。
- 运行期 sys.path 引导 external/SMQ，官方代码零改动（AGENTS.md 硬规则）
- 训练：BatchGenerator 读 features/*.npy (C,T,V,M) → Trainer.train（无监督重建+VQ，
  不需要 GT；num_actions 即 codebook 大小 K，显式传入以绕开 GT 目录依赖）
- 推理：SMQModel.forward 后取 model.indices (N,T)——逐帧 motion word 序列，
  分割评估的原料

⚠️ P0.1 教训核对：SMQ 全链使用 PyTorch 默认初始化 + kaiming/KMeans codebook 初始化
   + EMA + dead-code replacement，无 AimCLR 式 N(0,0.02) 覆盖初始化（已逐文件核验）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SMQ_EXTERNAL = REPO_ROOT / "external" / "SMQ"


def _bootstrap() -> None:
    if not SMQ_EXTERNAL.is_dir():
        raise FileNotFoundError(
            f"未找到 external/SMQ：{SMQ_EXTERNAL}。先克隆官方仓库到该路径。"
        )
    if str(SMQ_EXTERNAL) not in sys.path:
        sys.path.insert(0, str(SMQ_EXTERNAL))


class SMQSegmenter:
    """SMQ 无监督时序分割封装（train / infer）。"""

    def __init__(
        self,
        *,
        in_channels: int,
        filters: int,
        num_layers: int,
        latent_dim: int,
        num_actions: int,
        num_joints: int,
        num_person: int = 1,
        patch_size: int,
        decay: float = 0.5,
        kmeans: bool = False,
        kmeans_metric: str = "euclidean",
        sampling_quantile: float = 0.5,
        replacement_strategy: str = "representative",
    ) -> None:
        _bootstrap()
        from model import Trainer  # external/SMQ/model.py

        self.trainer = Trainer(
            in_channels=in_channels,
            filters=filters,
            num_layers=num_layers,
            latent_dim=latent_dim,
            num_actions=num_actions,
            num_joints=num_joints,
            num_person=num_person,
            patch_size=patch_size,
            kmeans=kmeans,
            kmeans_metric=kmeans_metric,
            sampling_quantile=sampling_quantile,
            replacement_strategy=replacement_strategy,
            decay=decay,
        )
        self.patch_size = patch_size
        self.num_actions = num_actions

    def fit(
        self,
        features_path: str | Path,
        save_dir: str | Path,
        *,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        commit_weight: float = 1.0,
        mse_loss_weight: float = 0.001,
        joint_distance_recons: bool = True,
        sample_rate: int = 1,
    ) -> None:
        """无监督训练（官方逻辑原样透传）。逐 epoch loss 由调用方捕获 stdout 存档。"""
        from batch_gen import BatchGenerator  # external/SMQ/batch_gen.py

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_gen = BatchGenerator(
            features_path=str(features_path),
            sample_rate=sample_rate,
            num_features=self.trainer.model.encoder.stage1.conv_1x1.in_channels,
            num_joints=self.trainer.model.num_joints,
            num_person=self.trainer.model.num_person,
        )
        batch_gen.read_data()
        self.trainer.train(
            save_dir=Path(save_dir),
            batch_gen=batch_gen,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            commit_weight=commit_weight,
            mse_loss_weight=mse_loss_weight,
            device=device,
            joint_distance_recons=joint_distance_recons,
        )

    def infer_indices(self, data: np.ndarray, ckpt_path: str | Path) -> np.ndarray:
        """单条序列前向 → motion word 索引 (T,) int64（截回真实长度）。

        data: (C, T, V, M)。T 非 patch_size 整数倍时零填充 + mask 置零
        （官方 quantizer 的 valid_patch_mask 会跳过全零 patch），返回前裁掉。
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trainer.model.eval()
        self.trainer.model.to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        self.trainer.model.load_state_dict(state)

        assert data.ndim == 4, f"期望 (C,T,V,M)，实际 {data.shape}"
        t_true = data.shape[1]
        pad = (-t_true) % self.patch_size

        arr = np.zeros((data.shape[0], t_true + pad, data.shape[2], data.shape[3]),
                       dtype=np.float32)
        arr[:, :t_true] = data.astype(np.float32)

        x = torch.from_numpy(arr)[None].to(device)          # (1,C,T,V,M)
        mask = torch.zeros_like(x)
        mask[:, :, :t_true] = 1.0
        with torch.no_grad():
            self.trainer.model(x, mask)
        return self.trainer.model.indices[0][:t_true].detach().cpu().numpy()
